"""The classification grids and the one transform from pixels to grid cells.

Aggregation, tabulation and the pixel view all assign each pixel to the grid
cell that contains its centre, through `pixel_to_cell`. The grid a method
classifies is therefore exactly what the pixel-level products see.
"""

import numpy as np
import numpy.typing as npt
import pyproj
from affine import Affine
from pydantic import BaseModel, ConfigDict

from rra_population_urbanicity import constants as puc

type FloatArray = npt.NDArray[np.float64]
type IntArray = npt.NDArray[np.int32]


class GridDefinition(BaseModel):
    """A north-up grid of square cells."""

    model_config = ConfigDict(frozen=True)

    name: str
    crs: str
    cell_size: float
    x_min: float
    y_max: float
    n_rows: int
    n_cols: int

    @property
    def transform(self) -> Affine:
        return Affine(self.cell_size, 0.0, self.x_min, 0.0, -self.cell_size, self.y_max)

    @property
    def shape(self) -> tuple[int, int]:
        return self.n_rows, self.n_cols

    def cell_index(self, x: FloatArray, y: FloatArray) -> tuple[IntArray, IntArray]:
        """Return the row and column of the cell containing each point.

        Points are in the grid's CRS. A point on a cell edge belongs to the cell
        to its right or below.
        """
        col = np.floor((x - self.x_min) / self.cell_size)
        row = np.floor((self.y_max - y) / self.cell_size)
        inside = (row >= 0) & (row < self.n_rows) & (col >= 0) & (col < self.n_cols)
        if not np.all(inside):
            msg = (
                f"{np.count_nonzero(~inside)} points fall outside the {self.name} grid."
            )
            raise ValueError(msg)
        return row.astype(np.int32), col.astype(np.int32)


GRIDS = {
    # The 1 km World Mollweide grid of GHS-POP, GHS-BUILT-S, GHS-LAND and
    # GHS-SMOD, so our cells coincide with GHSL's.
    "moll_1km": GridDefinition(
        name="moll_1km",
        crs=puc.CRSES["mollweide"],
        cell_size=1000.0,
        x_min=-18_041_000.0,
        y_max=9_000_000.0,
        n_rows=18_000,
        n_cols=36_082,
    ),
}


def pixel_centres(
    transform: Affine, shape: tuple[int, int]
) -> tuple[FloatArray, FloatArray]:
    """Return the x and y coordinates of the centre of every pixel of a raster."""
    if transform.b or transform.d:
        msg = "Only north-up rasters are supported."
        raise ValueError(msg)
    n_rows, n_cols = shape
    x = transform.c + (np.arange(n_cols) + 0.5) * transform.a
    y = transform.f + (np.arange(n_rows) + 0.5) * transform.e
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def pixel_to_cell(
    transform: Affine, shape: tuple[int, int], crs: str, grid: GridDefinition
) -> tuple[IntArray, IntArray]:
    """Assign every pixel of a raster to the grid cell containing its centre.

    Parameters
    ----------
    transform
        The affine transform of the raster.
    shape
        The raster's (rows, columns).
    crs
        The raster's coordinate reference system.
    grid
        The grid to assign pixels to.

    Returns
    -------
    tuple[IntArray, IntArray]
        The grid row and column of each pixel, each with the raster's shape.
    """
    x, y = pixel_centres(transform, shape)
    transformer = pyproj.Transformer.from_crs(crs, grid.crs, always_xy=True)
    x, y = transformer.transform(x, y)
    return grid.cell_index(x, y)
