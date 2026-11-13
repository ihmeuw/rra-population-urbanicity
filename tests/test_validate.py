from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from click.testing import CliRunner

from rra_population_urbanicity.classify import runner
from rra_population_urbanicity.cli import purun
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.reference.ghsl import GhslRaster
from rra_population_urbanicity.specification import ClassificationSpecification
from rra_population_urbanicity.validate import ghs_smod_reproduction as validation
from tests.conftest import ghsl_spec, write_ghsl


@pytest.fixture
def classified(ghsl_world: Path) -> str:
    """A classification of the tiny world, and a GHS-SMOD V2.0 that differs from it.

    In 2020, GHS-SMOD has class 13 where we have 11 at (0, 0), on land, and
    class 11 where we have water at (0, 9), in the river; it has no data at
    (11, 0). In 2025 the two agree.
    """
    data = UrbanicityData(ghsl_world)
    spec = ClassificationSpecification.model_validate(ghsl_spec())
    version = runner.run(spec, data, None)
    for time_point, epoch in [("2020q1", 2020), ("2025q1", 2025)]:
        with rasterio.open(data.class_raster_path(version, time_point)) as src:
            smod = src.read(1).astype(np.int16)
        if epoch == 2020:
            smod[0, 0] = 13
            smod[0, 9] = 11
            smod[11, 0] = -200
        raster = GhslRaster(product="SMOD", release="R2023A", epoch=epoch, version=2)
        write_ghsl(ghsl_world, raster, smod, -200)
    return version


def test_a_classification_is_compared_with_ghs_smod(
    ghsl_world: Path, classified: str
) -> None:
    data = UrbanicityData(ghsl_world)
    summary = validation.validate(data, classified, "r2023a_v2")["time_points"]
    # 143 cells compared, 2 disagree; without the river and the missing land
    # cell, 130 cells and 1 disagreement.
    assert summary["2020q1"]["all"]["cells"] == 143
    assert summary["2020q1"]["all"]["agreement"] == pytest.approx(141 / 143)
    assert summary["2020q1"]["populated_or_land"]["agreement"] == pytest.approx(
        129 / 130
    )
    assert summary["2025q1"]["all"]["agreement"] == 1.0
    assert summary["2020q1"]["urban_centres"] == {"reference": 1, "ours": 1}

    classes = summary["2020q1"]["all"]["classes"]
    assert classes[13] == {
        "reference_cells": 1,
        "our_cells": 0,
        "agreeing_cells": 0,
        "recall": 0.0,
        "precision": None,
        "overlap": 0.0,
        "reference_population": 0.0,
        "our_population": 0.0,
    }
    assert classes[30]["overlap"] == 1.0
    assert classes[30]["our_population"] == 50_000
    assert classes[22]["recall"] is None  # no class 22 in either

    out_dir = (
        data.classification_root(classified)
        / "validation"
        / "ghs_smod_reproduction"
        / "r2023a_v2"
    )
    with rasterio.open(out_dir / "disagreement_2020q1.tif") as src:
        diff = src.read(1)
    assert diff[0, 0] == 1311
    assert diff[0, 9] == 1110
    assert np.count_nonzero(diff) == 2
    confusion = pd.read_csv(out_dir / "confusion_2020q1.csv")
    row = confusion.query("cells_in == 'all' and reference == 30 and ours == 30")
    assert row["cells"].item() == 25
    assert (out_dir / "summary.yaml").exists()


def test_the_command_prints_agreement(ghsl_world: Path, classified: str) -> None:
    result = CliRunner().invoke(
        purun, ["validate", "ghs-smod", classified, "-o", str(ghsl_world)]
    )
    assert result.exit_code == 0, result.output
    assert result.output.startswith(
        "2020q1 | all 98.6014% | populated_or_land 99.2308%"
    )


def test_the_v1_reference_path(tmp_path: Path) -> None:
    path = validation.smod_path(UrbanicityData(tmp_path), "r2023a_v1", "2025q1")
    assert path.parent.name == "r2023a_v1"
    assert path.name == "GHS_SMOD_E2025_GLOBE_R2023A_54009_1000_V1_0.tif"
