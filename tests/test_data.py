from pathlib import Path

import pytest

from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.data import GridSource, Layer, UrbanicityData
from rra_population_urbanicity.specification import (
    GhsLandSource,
    GhsPopSource,
    MicrosoftDensitySource,
    MicrosoftWaterSource,
    PopulaceSource,
)


@pytest.mark.parametrize(
    ("layer", "source", "time_point", "expected"),
    [
        (
            "population",
            PopulaceSource(
                source="populace", model_version="2026_10_02.001", resolution="40"
            ),
            "2023q1",
            "population/populace/40m/2026_10_02.001/2023q1.tif",
        ),
        (
            "population",
            GhsPopSource(source="ghs_pop", release="r2023a"),
            "2020q1",
            "population/ghs_pop/r2023a/2020q1.tif",
        ),
        (
            "land",
            GhsLandSource(source="ghs_land", release="r2022a"),
            puc.STATIC,
            "land/ghs_land/r2022a/static.tif",
        ),
        (
            "land",
            MicrosoftWaterSource(source="microsoft_water", building_version="v8"),
            puc.STATIC,
            "land/microsoft_water/v8/static.tif",
        ),
        (
            "built",
            MicrosoftDensitySource(
                source="microsoft_density", building_version="v8", resolution="40"
            ),
            "2023q1",
            "built/microsoft_density/v8/40m/2023q1.tif",
        ),
    ],
)
def test_grid_paths(
    tmp_path: Path, layer: Layer, source: GridSource, time_point: str, expected: str
) -> None:
    path = UrbanicityData(tmp_path).grid_path("moll_1km", layer, source, time_point)
    assert path == tmp_path / "grids" / "moll_1km" / expected


def test_reference_root(tmp_path: Path) -> None:
    path = UrbanicityData(tmp_path).reference_root("ghsl", "ghs_smod", "r2023a_v2")
    assert path == tmp_path / "reference" / "ghsl" / "ghs_smod" / "r2023a_v2"


def test_the_default_root_is_the_output_root() -> None:
    assert UrbanicityData().root == puc.URBANICITY_ROOT


def test_classification_paths(tmp_path: Path) -> None:
    data = UrbanicityData(tmp_path)
    root = tmp_path / "classifications" / "2026_11_13.001"
    assert data.classification_root("2026_11_13.001") == root
    assert data.class_raster_path("2026_11_13.001", "2020q1") == (
        root / "classes" / "2020q1.tif"
    )
