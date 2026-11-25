"""Reading an input layer on a classification grid, whatever its source.

GHSL's layers are read from their reference downloads, which are already on
`moll_1km`. Every other source is read from the grid the aggregation stage
wrote for it. Either way, population is in people per cell, and land and
built-up are shares of the cell.
"""

import numpy as np
import rasterio
from rasterio.windows import Window

from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.data import Layer, UrbanicityData
from rra_population_urbanicity.grid import GridDefinition
from rra_population_urbanicity.inputs.providers import ghsl
from rra_population_urbanicity.reference.ghsl import check_on_grid
from rra_population_urbanicity.specification import (
    GhsBuiltSSource,
    GhsLandSource,
    GhsPopSource,
    GhsWupBuiltSSource,
    GhsWupPopSource,
)

GHSL_SOURCES = (
    GhsPopSource,
    GhsWupPopSource,
    GhsLandSource,
    GhsBuiltSSource,
    GhsWupBuiltSSource,
)


def load(
    data: UrbanicityData,
    layer: Layer,
    source: ghsl.Source,
    grid: GridDefinition,
    time_point: str,
    window: Window,
) -> ghsl.FloatArray:
    """Read a window of an input layer at a time point."""
    if isinstance(source, GHSL_SOURCES):
        return ghsl.load(data, source, grid, time_point, window)
    path = data.grid_path(
        grid.name, layer, source, puc.STATIC if layer == "land" else time_point
    )
    check_on_grid(path, grid)
    with rasterio.open(path) as src:
        values: ghsl.FloatArray = src.read(1, window=window).astype(np.float64)
    return values
