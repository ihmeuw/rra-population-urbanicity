from pathlib import Path

import numpy as np
import pytest
import rasterio
from affine import Affine

from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.inputs import blocks
from rra_population_urbanicity.inputs.blocks import MODEL_CRS, Block
from rra_population_urbanicity.inputs.providers import ghsl, microsoft, populace
from rra_population_urbanicity.specification import (
    GhsBuiltSSource,
    GhsLandSource,
    GhsPopSource,
    MicrosoftDensitySource,
    MicrosoftWaterSource,
    PopulaceSource,
)
from tests.conftest import TINY, write_model_raster


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


POPULACE = PopulaceSource(
    source="populace", model_version="2026_10_02.001", resolution="40"
)
DENSITY = MicrosoftDensitySource(
    source="microsoft_density", building_version="v8", resolution="40"
)


def group_transform(group_x: int, group_y: int) -> Affine:
    size = blocks.GROUP_BLOCKS * blocks.BLOCK_PIXELS
    return blocks.MODEL_TRANSFORM @ Affine.translation(group_x * size, group_y * size)


@pytest.mark.usefixtures("small_frame")
def test_a_populace_block_is_read_from_its_group_tile(tmp_path: Path) -> None:
    # Group G-0001X-0000Y holds blocks 2 and 3 of each row, cut to the
    # raster's last 2 columns and 7 rows.
    values = np.arange(14, dtype=np.float32).reshape(7, 2)
    values[5, 0] = np.nan
    path = populace.compiled_path(tmp_path, POPULACE, "2023q1", "G-0001X-0000Y")
    write_model_raster(path, values, group_transform(1, 0))

    people = populace.load_block(POPULACE, Block(2, 1), "2023q1", root=tmp_path)
    expected = values[4:7].copy()
    expected[1, 0] = 0.0  # no prediction: no people
    np.testing.assert_array_equal(people, expected)
    assert people.dtype == np.float32


@pytest.mark.usefixtures("small_frame")
@pytest.mark.parametrize(
    ("shift", "crs", "shape", "message"),
    [
        (1, MODEL_CRS, (7, 8), "where expected"),
        (0, "ESRI:54009", (7, 8), "where expected"),
        (0, MODEL_CRS, (6, 8), r"holds \(2, 4\) pixels"),
    ],
)
def test_a_block_must_be_where_the_model_grid_puts_it(
    tmp_path: Path,
    shift: int,
    crs: str,
    shape: tuple[int, int],
    message: str,
) -> None:
    transform = group_transform(0, 0) @ Affine.translation(shift, 0)
    path = populace.compiled_path(tmp_path, POPULACE, "2023q1", "G-0000X-0000Y")
    write_model_raster(path, np.ones(shape), transform, crs)
    with pytest.raises(ValueError, match=message):
        populace.load_block(POPULACE, Block(1, 1), "2023q1", root=tmp_path)


@pytest.mark.usefixtures("small_frame")
@pytest.mark.parametrize(
    ("time_point", "quarter_read"), [("2020q1", "2020q2"), ("2023q1", "2023q1")]
)
def test_building_density_is_read_from_the_block_file(
    tmp_path: Path, time_point: str, quarter_read: str
) -> None:
    block = Block(2, 1)
    values = np.full(block.shape, 0.25)
    values[0, 0] = np.nan  # outside the tiles' coverage: no buildings
    path = (
        tmp_path
        / "tiles"
        / "40m"
        / "microsoft_v8"
        / quarter_read
        / "B-0002X-0001Y_density.tif"
    )
    write_model_raster(path, values, block.transform)

    density = microsoft.load_density_block(DENSITY, block, time_point, root=tmp_path)
    assert density.shape == (3, 2)
    assert density[0, 0] == 0.0
    assert density[1:].tolist() == [[0.25, 0.25], [0.25, 0.25]]


WATER = MicrosoftWaterSource(source="microsoft_water", building_version="v8")
# A Web Mercator tile at zoom 11 spans 360 / 2048 degrees at the equator.
TILE_DEGREES = 360 / 2048


def write_water_tile(root: Path, tile_x: int, tile_y: int, water: np.ndarray) -> None:
    west = -180 + tile_x * TILE_DEGREES
    # The two tile rows either side of the equator, counted from the south.
    north = {1024: TILE_DEGREES, 1023: 0.0}[tile_y]
    path = microsoft.water_tile_path(root, tile_x, tile_y)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=512,
        width=512,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=Affine(TILE_DEGREES / 512, 0, west, 0, -TILE_DEGREES / 512, north),
    ) as dst:
        dst.write(water.astype(np.uint8), 1)


@pytest.mark.usefixtures("small_frame")
def test_each_pixel_takes_the_water_mask_at_its_centre(tmp_path: Path) -> None:
    # The block's pixel centres are 100, 60 and 20 m west and 20 m east of
    # the prime meridian, and 60 and 20 m north and south of the equator.
    northwest = np.zeros((512, 512))
    northwest[:, 511] = 1  # the last 38 m before the meridian are water
    northwest[511, :] = 1  # and so are the last 36 m before the equator
    write_water_tile(tmp_path, 1023, 1024, northwest)
    write_water_tile(tmp_path, 1024, 1024, np.ones((512, 512)))
    write_water_tile(tmp_path, 1023, 1023, np.zeros((512, 512)))
    # The tile south-east of the origin is an empty file, which reads as no
    # tile at all, as over the open ocean.
    microsoft.water_tile_path(tmp_path, 1024, 1023).touch()

    land = microsoft.load_land_block(WATER, Block(0, 0), root=tmp_path)
    assert land.tolist() == [
        [True, True, False, False],
        [False, False, False, False],
        [True, True, True, False],
        [True, True, True, False],
    ]


def test_there_is_no_land_where_there_are_no_mask_tiles(tmp_path: Path) -> None:
    # The first block of the model grid reaches from 71° to 90° north, past
    # the last Web Mercator tiles at 85.05°.
    assert not microsoft.load_land_block(WATER, Block(0, 0), root=tmp_path).any()
