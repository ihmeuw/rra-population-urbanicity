"""The class of each 40 m pixel: the class of the 1 km cell containing its centre.

The same transform as aggregation and the population table, so a pixel's
class here is the class its people were counted under. Pixels beyond the
grid, or outside the window a classification covers, have the scheme's
nodata code.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import rasterio
from affine import Affine
from rasterio.windows import Window

from rra_population_urbanicity.grid import GridDefinition, pixel_centres_in
from rra_population_urbanicity.inputs.aggregate import rows_on_grid
from rra_population_urbanicity.inputs.blocks import MODEL_CRS, Block

type CodeArray = npt.NDArray[np.uint8]


def pixel_classes(
    block: Block, grid: GridDefinition, class_path: Path, nodata: int
) -> CodeArray:
    """The class of each pixel of a block, from a class raster on `grid`."""
    codes = np.full(block.shape, nodata, dtype=np.uint8)
    rows = rows_on_grid(block, grid)
    height, width = rows.stop - rows.start, block.shape[1]
    transform = block.transform @ Affine.translation(0, rows.start)
    codes[rows] = window_classes(transform, (height, width), grid, class_path, nodata)
    return codes


def window_classes(
    transform: Affine,
    shape: tuple[int, int],
    grid: GridDefinition,
    class_path: Path,
    nodata: int,
) -> CodeArray:
    """The class of each pixel of a window of the model grid on the grid."""
    cell_row, cell_col = grid.cell_index(
        *pixel_centres_in(transform, shape, MODEL_CRS, grid.crs)
    )
    codes = np.full(shape, nodata, dtype=np.uint8)
    with rasterio.open(class_path) as src:
        row_off = round((grid.y_max - src.transform.f) / grid.cell_size)
        col_off = round((src.transform.c - grid.x_min) / grid.cell_size)
        local_row, local_col = cell_row - row_off, cell_col - col_off
        inside = (
            (local_row >= 0)
            & (local_row < src.height)
            & (local_col >= 0)
            & (local_col < src.width)
        )
        if inside.any():
            first, last = int(local_row[inside].min()), int(local_row[inside].max())
            stored = src.read(1, window=Window(0, first, src.width, last - first + 1))
            codes[inside] = stored[local_row[inside] - first, local_col[inside]]
    return codes
