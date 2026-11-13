from pathlib import Path

import numpy as np
import pytest

from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.inputs.providers import ghsl
from rra_population_urbanicity.specification import (
    GhsBuiltSSource,
    GhsLandSource,
    GhsPopSource,
    PopulaceSource,
)
from tests.conftest import TINY


def test_ghsl_layers_are_read_in_people_and_shares(ghsl_world: Path) -> None:
    data = UrbanicityData(ghsl_world)
    window = TINY.full_window
    population = ghsl.load(
        data, GhsPopSource(source="ghs_pop", release="r2023a"), TINY, "2020q1", window
    )
    land = ghsl.load(
        data, GhsLandSource(source="ghs_land", release="r2022a"), TINY, "2020q1", window
    )
    built = ghsl.load(
        data,
        GhsBuiltSSource(source="ghs_built_s", release="r2023a"),
        TINY,
        "2025q1",
        window,
    )
    assert population.sum() == 50_000  # the missing cell reads as no people
    assert population[11, 11] == 0
    assert land[0, 0] == 1.0
    assert land[0, 9] == 0.0
    assert land[0, 11] == 0.0  # a missing cell reads as water
    assert built[2, 2] == pytest.approx(0.6)
    assert built.dtype == np.float64


@pytest.mark.parametrize("time_point", ["2021q1", "2020q2"])
def test_ghsl_time_points_are_epochs(time_point: str) -> None:
    with pytest.raises(ValueError, match="five-yearly epochs"):
        ghsl.epoch(time_point)


def test_other_sources_have_no_ghsl_raster() -> None:
    source = PopulaceSource(
        source="populace", model_version="2026_10_02.001", resolution="40"
    )
    with pytest.raises(NotImplementedError, match="no provider for populace"):
        ghsl.ghsl_raster(source, "2020q1")
