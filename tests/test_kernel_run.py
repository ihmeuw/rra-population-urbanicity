from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner
from rasterio.windows import Window
from rra_tools import jobmon

from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.classifiers.kernel_density.scheme import RURAL, URBAN
from rra_population_urbanicity.classify import kernel, runner
from rra_population_urbanicity.cli import putask
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.grid import GRIDS
from rra_population_urbanicity.inputs import blocks
from rra_population_urbanicity.inputs.providers import ghsl, populace
from rra_population_urbanicity.reference.ghsl import GhslRaster
from rra_population_urbanicity.specification import (
    ClassificationSpecification,
    PopulaceSource,
)
from tests.conftest import TINY, ghsl_spec, write_model_raster, write_raster
from tests.test_aggregate import TINY_100M
from tests.test_tabulate import frame_units, write_units

UNIT_SETS = ("units_a", "units_b")
TIME_POINTS = ["2020q1", "2023q1"]


def frame_people(time_point: str) -> npt.NDArray[np.float64]:
    """625 people per km² west of column 5 and 62.5 east of it; block (2, 1) empty.

    In 2023 the top-left pixel has 100 people.
    """
    people = np.full((blocks.HEIGHT, blocks.WIDTH), 0.1)
    people[:, :5] = 1.0
    people[4:, 8:] = 0.0
    if time_point == "2023q1":
        people[0, 0] = 100.0
    return people


def frame_land(data: UrbanicityData, window: Window) -> npt.NDArray[np.float32]:
    """All land on the frame, and none beyond it."""
    land = np.zeros((int(window.height), int(window.width)), dtype=np.float32)
    rows = slice(max(-int(window.row_off), 0), blocks.HEIGHT - int(window.row_off))
    cols = slice(max(-int(window.col_off), 0), blocks.WIDTH - int(window.col_off))
    land[rows, cols] = 1.0
    return land


def kernel_spec() -> ClassificationSpecification:
    # A disc of one pixel's radius: the pixel and its four neighbours.
    return ClassificationSpecification.model_validate(
        {
            "population": {
                "source": "populace",
                "model_version": "2026_10_02.001",
                "resolution": "40",
            },
            "grid": TINY.name,
            "land": {"source": "ghs_land", "release": "r2022a"},
            "time_points": TIME_POINTS,
            "method": {
                "name": "kernel_density",
                "parameter_set": "kernel_default",
                "parameters": {"radius_m": 40.0},
            },
        }
    )


@pytest.fixture
def kernel_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_frame: None
) -> Path:
    monkeypatch.setitem(GRIDS, TINY.name, TINY)
    model_root = tmp_path / "population-model"
    monkeypatch.setattr(puc, "POPULATION_MODEL_ROOT", model_root)
    for time_point in TIME_POINTS:
        write_model_raster(
            model_root / f"{time_point}.tif",
            frame_people(time_point),
            blocks.MODEL_TRANSFORM,
        )
    monkeypatch.setattr(
        populace, "vrt_path", lambda root, source, tp: root / f"{tp}.tif"
    )
    monkeypatch.setattr(kernel, "read_land", frame_land)
    monkeypatch.setattr(kernel, "UNIT_SETS", UNIT_SETS)
    root = tmp_path / "output"
    data = UrbanicityData(root)
    for shapes in UNIT_SETS:
        monkeypatch.setattr("tests.test_tabulate.SHAPES", shapes)
        write_units(data, frame_units())

    def run_parallel(**kwargs: Any) -> str:
        for key in kwargs["node_args"]["block-key"]:
            kernel.run_block(data, kwargs["task_args"]["version"], key)
        return "done"

    monkeypatch.setattr(jobmon, "run_parallel", run_parallel)
    return root


def read_tabulation(
    root: Path, version: str, shapes: str, time_point: str
) -> dict[Any, float]:
    path = (
        UrbanicityData(root).classification_root(version)
        / "tabulations"
        / shapes
        / f"{time_point}.parquet"
    )
    table = pd.read_parquet(path)
    return {
        (int(row.location_id), int(row.class_code)): round(float(row.population), 6)
        for row in table.itertuples()
    }


def test_the_kernel_classifies_and_tabulates_every_block(kernel_world: Path) -> None:
    data = UrbanicityData(kernel_world)
    version = runner.run(kernel_spec(), data, None)

    # Columns 3 and 4 are urban even beside the sparse east; column 5 is not.
    expected = {
        (11, URBAN): 21.0,
        (12, URBAN): 4.0,
        (12, RURAL): 1.0,
        (21, URBAN): 10.0,
        (21, RURAL): 1.9,
    }
    for shapes in UNIT_SETS:
        assert read_tabulation(kernel_world, version, shapes, "2020q1") == expected
    later = read_tabulation(kernel_world, version, "units_a", "2023q1")
    assert later[(11, URBAN)] == 120.0

    root = data.classification_root(version)
    assert not (root / ".blocks").exists()
    assert yaml.safe_load((root / "scheme.yaml").read_text())["name"] == "binary"
    manifest = yaml.safe_load((root / "manifest.yaml").read_text())
    check = manifest["tabulations"]["units_b"]["2020q1"]
    assert check["people"] == pytest.approx(37.9)
    assert check["people_in_no_unit"] == 0.0
    assert check["urban_share"] == pytest.approx(35 / 37.9)
    saved = ClassificationSpecification.from_yaml(root / "specification.yaml")
    assert saved.version == version


def test_a_region_runs_its_blocks_in_this_process(
    kernel_world: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Only blocks (0, 0) and (1, 0): the west of the frame.
    monkeypatch.setitem(puc.REGIONS, "test", (-0.001, -0.0001, 0.001, 0.0007))
    monkeypatch.setattr(jobmon, "run_parallel", None)
    data = UrbanicityData(kernel_world)
    version = runner.reserve_version(data.classifications_root)
    kernel.run(kernel_spec(), data, version, "test")
    table = read_tabulation(kernel_world, version, "units_a", "2020q1")
    assert table[(11, URBAN)] == 12.0
    manifest = yaml.safe_load(
        (data.classification_root(version) / "manifest.yaml").read_text()
    )
    assert manifest["region"]["blocks"] == ["B-0000X-0000Y", "B-0001X-0000Y"]


def test_the_reduce_checks_that_everyone_is_kept(kernel_world: Path) -> None:
    data = UrbanicityData(kernel_world)
    version = runner.reserve_version(data.classifications_root)
    root = data.classification_root(version)
    kernel_spec().model_copy(update={"version": version}).to_yaml(
        root / "specification.yaml"
    )
    keys = [block.key for block in blocks.all_blocks()]
    for key in keys:
        kernel.run_block(data, version, key)
    totals_path = kernel.block_path(root, keys[0]).with_suffix(".yaml")
    totals = yaml.safe_load(totals_path.read_text())
    totals_path.write_text(yaml.safe_dump({**totals, "2020q1": totals["2020q1"] + 1}))
    with pytest.raises(ValueError, match="tabulation at 2020q1 holds 38 people"):
        kernel.reduce(root, keys)


def test_a_block_task_needs_a_kernel_run(
    kernel_world: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = UrbanicityData(kernel_world)
    version = runner.reserve_version(data.classifications_root)
    root = data.classification_root(version)
    ClassificationSpecification.model_validate(ghsl_spec()).to_yaml(
        root / "specification.yaml"
    )
    result = CliRunner().invoke(
        putask,
        [
            "classify",
            "kernel-block",
            "--version",
            version,
            "--block-key",
            "B-0000X-0000Y",
            "--output-dir",
            str(kernel_world),
        ],
    )
    assert isinstance(result.exception, TypeError)
    assert "is not a kernel density run" in str(result.exception)


def test_a_blocks_latitude_and_margin() -> None:
    block = blocks.Block(91, 11)
    assert kernel.block_latitude(block) == pytest.approx(24.0, abs=0.1)
    window = kernel.margin_window(block, np.ones((45, 55), dtype=np.float32))
    assert (window.col_off, window.row_off) == (91 * 8192 - 27, 11 * 8192 - 22)
    assert (window.width, window.height) == (8192 + 54, 8192 + 44)


@pytest.mark.usefixtures("small_frame")
def test_land_is_read_at_each_pixel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ghsl, "MOLL_100M", TINY_100M)
    land = np.full(TINY_100M.shape, 10_000, dtype=np.uint16)
    land[:, 61:] = 2_500  # a quarter land east of x = 100 m
    raster = GhslRaster(product="LAND", release="R2022A", epoch=2018, resolution=100)
    data = UrbanicityData(tmp_path)
    out_dir = data.reference_root("ghsl", raster.product_dir, raster.release_dir)
    write_raster(out_dir / f"{raster.stem}.tif", land, 65535, TINY_100M)

    # The frame's pixels, with a margin of one pixel all round.
    shares = kernel.read_land(data, Window(-1, -1, blocks.WIDTH + 2, blocks.HEIGHT + 2))
    assert shares.shape == (blocks.HEIGHT + 2, blocks.WIDTH + 2)
    # Pixel centres from x = -140 m to 300 m: east of 100 m is a quarter land.
    np.testing.assert_allclose(shares[:, :7], 1.0)
    np.testing.assert_allclose(shares[:, 7:], 0.25)


def test_the_vrt_holds_a_time_points_tiles() -> None:
    source = PopulaceSource(
        source="populace", model_version="2026_10_02.001", resolution="40"
    )
    path = populace.vrt_path(Path("/model"), source, "2023q1")
    assert path == Path(
        "/model/modeling/40m/models/2026_10_02.001/compiled_predictions"
        "/population/2023q1/index.vrt"
    )
