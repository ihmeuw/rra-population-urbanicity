import math

import numpy as np
import pytest
from affine import Affine

from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.grid import GRIDS, GridDefinition, pixel_to_cell

# The Populace model grid: 40 m World Cylindrical Equal Area, in 8192-pixel blocks.
MODEL_CRS = puc.CRSES["equal_area"]
MODEL_X_MIN = -20_037_508.34
MODEL_Y_MAX = 6_363_885.33
MODEL_COLS = 1_001_876
PIXEL = 40.0
BLOCK = 8192

# The equal-area 1 km grid: 25 x 25 model pixels per cell.
CEA_1KM = GridDefinition(
    name="cea_1km",
    crs=MODEL_CRS,
    cell_size=1000.0,
    x_min=MODEL_X_MIN,
    y_max=MODEL_Y_MAX,
    n_rows=12_728,
    n_cols=40_076,
)
MOLL_1KM = GRIDS["moll_1km"]


def model_window(row: int, col: int) -> Affine:
    """The transform of a window of the model grid whose top-left pixel is (row, col)."""
    return Affine(
        PIXEL, 0.0, MODEL_X_MIN + col * PIXEL, 0.0, -PIXEL, MODEL_Y_MAX - row * PIXEL
    )


@pytest.mark.parametrize(("block_x", "block_y"), [(0, 0), (61, 19), (122, 38)])
def test_reduces_to_integer_division_on_the_equal_area_grid(
    block_x: int, block_y: int
) -> None:
    row, col = block_y * BLOCK, block_x * BLOCK
    shape = (60, 70)
    rows, cols = pixel_to_cell(model_window(row, col), shape, MODEL_CRS, CEA_1KM)

    pixel_rows, pixel_cols = np.indices(shape)
    np.testing.assert_array_equal(rows, (row + pixel_rows) // 25)
    np.testing.assert_array_equal(cols, (col + pixel_cols) // 25)


def test_mollweide_cells_at_the_equator() -> None:
    # These model rows straddle the equator, which is the edge between
    # Mollweide rows 8999 and 9000. On the equator, Mollweide x is 2√2/π times
    # equal-area x.
    row, col = 159_095, 500_000
    shape = (4, 100)
    rows, cols = pixel_to_cell(model_window(row, col), shape, MODEL_CRS, MOLL_1KM)

    centre_y = MODEL_Y_MAX - (row + np.arange(shape[0]) + 0.5) * PIXEL
    expected_rows = np.where(centre_y > 0, 8999, 9000)
    centre_x = MODEL_X_MIN + (col + np.arange(shape[1]) + 0.5) * PIXEL
    position = (2 * math.sqrt(2) / math.pi * centre_x - MOLL_1KM.x_min) / 1000
    # No pixel centre is within 1 cm of a cell edge, where rounding could matter.
    assert np.min(np.abs(position - np.round(position))) > 1e-5
    np.testing.assert_array_equal(rows, np.repeat(expected_rows[:, None], shape[1], 1))
    np.testing.assert_array_equal(cols, np.tile(np.floor(position), (shape[0], 1)))


def test_pixels_past_the_antimeridian_wrap_to_the_west_edge() -> None:
    # The model grid is about 23 m wider than the world, so the centres of its
    # last column are past 180° and land in the first Mollweide column.
    _, cols = pixel_to_cell(
        model_window(159_096, MODEL_COLS - 2), (1, 2), MODEL_CRS, MOLL_1KM
    )
    np.testing.assert_array_equal(cols, [[MOLL_1KM.n_cols - 1, 0]])


def test_rotated_rasters_are_rejected() -> None:
    rotated = model_window(0, 0) @ Affine.rotation(10)
    with pytest.raises(ValueError, match="north-up"):
        pixel_to_cell(rotated, (2, 2), MODEL_CRS, CEA_1KM)


def test_points_outside_the_grid_are_rejected() -> None:
    x = np.array([0.0, 20_000_000.0, np.nan])
    y = np.zeros(3)
    with pytest.raises(ValueError, match="2 points fall outside the moll_1km grid"):
        MOLL_1KM.cell_index(x, y)
