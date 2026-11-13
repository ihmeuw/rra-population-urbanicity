"""GHSL input layers, read from the reference downloads.

The GHSL rasters are already on `moll_1km`, so they are read as they are, not
aggregated. Population is in people per cell; land and built-up, published as
m² per cell, are converted to shares of the cell. Missing values are read the
way GHS-SMOD reads them: no people, no built-up, and no land (water).
"""

import numpy as np
import numpy.typing as npt
import rasterio
from rasterio.windows import Window

from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.grid import GridDefinition
from rra_population_urbanicity.reference.ghsl import GhslRaster, check_on_grid
from rra_population_urbanicity.specification import (
    GhsBuiltSSource,
    GhsLandSource,
    GhsPopSource,
    MicrosoftDensitySource,
    MicrosoftWaterSource,
    PopulaceSource,
)

type FloatArray = npt.NDArray[np.float64]
type Source = (
    PopulaceSource
    | GhsPopSource
    | GhsLandSource
    | MicrosoftWaterSource
    | GhsBuiltSSource
    | MicrosoftDensitySource
)

# GHS-LAND R2022A has a single epoch, which serves every time point.
GHS_LAND_EPOCH = 2018
GHSL_EPOCH_YEARS = 5


def epoch(time_point: str) -> int:
    """The GHSL epoch of a time point: GHSL epochs are labelled YYYYq1."""
    year, quarter = time_point.split("q")
    if quarter != "1" or int(year) % GHSL_EPOCH_YEARS:
        msg = f"GHSL layers exist for five-yearly epochs labelled YYYYq1, not {time_point}."
        raise ValueError(msg)
    return int(year)


def ghsl_raster(source: Source, time_point: str) -> GhslRaster:
    if isinstance(source, GhsPopSource):
        return GhslRaster(product="POP", release="R2023A", epoch=epoch(time_point))
    if isinstance(source, GhsLandSource):
        return GhslRaster(product="LAND", release="R2022A", epoch=GHS_LAND_EPOCH)
    if isinstance(source, GhsBuiltSSource):
        return GhslRaster(product="BUILT_S", release="R2023A", epoch=epoch(time_point))
    msg = f"There is no provider for {source.source} yet."
    raise NotImplementedError(msg)


def load(
    data: UrbanicityData,
    source: Source,
    grid: GridDefinition,
    time_point: str,
    window: Window,
) -> FloatArray:
    """Read a window of a GHSL layer in the units the classifiers use."""
    raster = ghsl_raster(source, time_point)
    root = data.reference_root("ghsl", raster.product_dir, raster.release_dir)
    path = root / f"{raster.stem}.tif"
    check_on_grid(path, grid)
    with rasterio.open(path) as src:
        values: FloatArray = src.read(1, window=window).astype(np.float64, copy=False)
        nodata = src.nodata
    values[values == nodata] = 0.0
    if raster.product != "POP":
        values /= grid.cell_size**2
    return values
