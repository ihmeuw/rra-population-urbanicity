import re
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import rasterio
import yaml
from click.testing import CliRunner
from rasterio.windows import Window
from rra_tools.shell_tools import mkdir

from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.classifiers.base import ClassScheme
from rra_population_urbanicity.classifiers.degurba import scheme
from rra_population_urbanicity.classify import runner
from rra_population_urbanicity.cli import purun
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.grid import GridDefinition
from rra_population_urbanicity.specification import ClassificationSpecification
from tests.conftest import TINY, ghsl_spec


def run(root: Path, region: str | None = None, **parameters: Any) -> tuple[str, Path]:
    spec = ClassificationSpecification.model_validate(ghsl_spec(**parameters))
    version = runner.run(spec, UrbanicityData(root), region)
    return version, UrbanicityData(root).classification_root(version)


def manifest(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = yaml.safe_load((root / "manifest.yaml").read_text())
    return result


def test_a_run_writes_a_new_version(ghsl_world: Path) -> None:
    version, root = run(ghsl_world)
    assert re.fullmatch(r"\d{4}_\d{2}_\d{2}\.001", version)

    saved = ClassificationSpecification.from_yaml(root / "specification.yaml")
    assert saved.version == version
    assert saved.grid == TINY
    assert ClassScheme.from_yaml(root / "scheme.yaml") == scheme.DEGURBA_L2

    with rasterio.open(root / "classes" / "2020q1.tif") as src:
        codes = src.read(1)
        assert src.nodata == scheme.NODATA
        assert src.colormap(1)[scheme.URBAN_CENTRE] == (255, 0, 0, 255)
    assert (codes[2:7, 2:7] == scheme.URBAN_CENTRE).all()
    assert (codes[:, 9] == scheme.WATER).all()

    clusters = pd.read_parquet(root / "clusters" / "2020q1.parquet")
    town = clusters.set_index("cluster_type").loc["urban_centre"]
    assert town[["population", "cells"]].tolist() == [50_000, 25]
    assert town[["row_start", "row_stop", "col_start", "col_stop"]].tolist() == [
        2,
        7,
        2,
        7,
    ]

    resolved = manifest(root)["time_points"]
    # The threshold comes from 2020 and serves 2025 too: the town's built-up share.
    assert resolved["2020q1"]["built_threshold"] == pytest.approx(0.5)
    assert resolved["2025q1"]["built_threshold"] == pytest.approx(0.5)
    assert resolved["2025q1"]["built_per_person"] == pytest.approx(25 * 0.6 / 50_000)
    assert resolved["2020q1"]["cells_by_class"][scheme.URBAN_CENTRE] == 25
    assert manifest(root)["region"] is None


def test_each_time_point_can_compute_its_own_threshold(ghsl_world: Path) -> None:
    _, root = run(ghsl_world, built_threshold_epoch="each_time_point")
    resolved = manifest(root)["time_points"]
    assert resolved["2025q1"]["built_threshold"] == pytest.approx(0.6)


def test_a_fixed_threshold_is_used_as_given(ghsl_world: Path) -> None:
    _, root = run(ghsl_world, built_threshold=0.25)
    assert manifest(root)["time_points"]["2020q1"]["built_threshold"] == 0.25


def test_a_run_without_a_built_up_layer(ghsl_world: Path) -> None:
    spec = ghsl_spec(
        uc_built_criterion=False, duc_built_criterion=False, density_on_land="none"
    )
    spec["built"] = None
    version = runner.run(
        ClassificationSpecification.model_validate(spec),
        UrbanicityData(ghsl_world),
        None,
    )
    result = manifest(UrbanicityData(ghsl_world).classification_root(version))
    assert result["built_up_layer"] == "none: read as zero"
    assert result["time_points"]["2020q1"]["built_threshold"] is None
    assert result["time_points"]["2020q1"]["built_per_person"] is None


def test_a_run_can_be_limited_to_a_region(
    ghsl_world: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(puc.REGIONS, "east", (0.001, -0.06, 0.06, 0.06))
    _, root = run(ghsl_world, region="east")
    region = manifest(root)["region"]
    with rasterio.open(root / "classes" / "2020q1.tif") as src:
        assert src.shape == (region["window"][2], region["window"][3])
        assert src.transform.c == TINY.x_min + region["window"][1] * TINY.cell_size
    assert region["name"] == "east"
    assert region["window"][1] > 0


def test_versions_count_up_within_a_day(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = runner.reserve_version(tmp_path)
    second = runner.reserve_version(tmp_path)
    assert first.endswith(".001")
    assert second.endswith(".002")

    # A run that claims .003 between our look and our claim pushes us to .004.
    def racing_mkdir(path: Path, **kwargs: Any) -> None:
        if path.name.endswith(".003"):
            mkdir(path)
        mkdir(path, **kwargs)

    monkeypatch.setattr(runner, "mkdir", racing_mkdir)
    assert runner.reserve_version(tmp_path).endswith(".004")


def test_the_command_runs_a_specification(ghsl_world: Path) -> None:
    spec_path = ghsl_world / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(ghsl_spec()))
    result = CliRunner().invoke(
        purun, ["classify", "run", str(spec_path), "-o", str(ghsl_world)]
    )
    assert result.exit_code == 0, result.output
    path = UrbanicityData(ghsl_world).class_raster_path(result.output.strip(), "2025q1")
    with rasterio.open(path) as src:
        assert np.all(src.read(1)[2:7, 2:7] == scheme.URBAN_CENTRE)


def test_cells_outside_the_projection_with_data_are_classified(
    ghsl_world: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def nowhere(self: GridDefinition, window: Window) -> npt.NDArray[np.bool_]:
        return np.zeros((int(window.height), int(window.width)), dtype=bool)

    monkeypatch.setattr(GridDefinition, "domain", nowhere)
    _, root = run(ghsl_world)
    with rasterio.open(root / "classes" / "2020q1.tif") as src:
        codes = src.read(1)
    assert (codes[2:7, 2:7] == scheme.URBAN_CENTRE).all()  # people
    assert codes[0, 0] == scheme.VERY_LOW_DENSITY_RURAL  # land
    assert (codes[:, 9] == scheme.NODATA).all()  # the river: no data of any kind
