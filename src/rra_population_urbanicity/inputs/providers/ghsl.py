"""GHSL input layers, read from the reference downloads.

The GHSL rasters are already on `moll_1km`, so they are read as they are, not
aggregated. Population is in people per cell; land and built-up, published as
m² per cell, are converted to shares of the cell. Missing values are read the
way GHS-SMOD reads them: no people, no built-up, and no land (water).
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import rasterio
from rasterio.windows import Window

from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.grid import GridDefinition
from rra_population_urbanicity.reference.ghsl import (
    MOLL_100M,
    GhslRaster,
    check_on_grid,
)
from rra_population_urbanicity.specification import (
    GhsBuiltSSource,
    GhsLandOrMicrosoftWaterSource,
    GhsLandSource,
    GhsPopSource,
    MicrosoftDensitySource,
    MicrosoftWaterSource,
    PopulaceSource,
)

type FloatArray = npt.NDArray[np.float64]
type Float32Array = npt.NDArray[np.float32]
type IntArray = npt.NDArray[np.int32]
type Source = (
    PopulaceSource
    | GhsPopSource
    | GhsLandSource
    | MicrosoftWaterSource
    | GhsLandOrMicrosoftWaterSource
    | GhsBuiltSSource
    | MicrosoftDensitySource
)

# Points of GHS-LAND at 100 m read from one window at a time: 512 rows of a
# block of the model grid.
SAMPLE_RUN_POINTS = 2**22

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


def sample_land_100m(
    data: UrbanicityData, x: FloatArray, y: FloatArray
) -> Float32Array:
    """GHS-LAND's land share of the 100 m cell containing each point.

    Points are in World Mollweide. They are read in runs of consecutive
    points, each from a window just covering them, so points next to each
    other in the arrays should be near each other on the ground, as a
    raster's pixels read row by row are. One window for a whole block of the
    model grid would not do: near the poles, far from the central meridian,
    a block is a long slanting sliver of Mollweide, and its bounding window
    billions of cells. A missing cell has no land, as at 1 km.
    """
    raster = GhslRaster(
        product="LAND", release="R2022A", epoch=GHS_LAND_EPOCH, resolution=100
    )
    root = data.reference_root("ghsl", raster.product_dir, raster.release_dir)
    path = root / f"{raster.stem}.tif"
    check_on_grid(path, MOLL_100M)
    rows, cols = MOLL_100M.cell_index(x.ravel(), y.ravel())
    result = np.empty(rows.size, dtype=np.float32)
    half = MOLL_100M.n_cols // 2
    for start in range(0, rows.size, SAMPLE_RUN_POINTS):
        run = slice(start, start + SAMPLE_RUN_POINTS)
        run_rows, run_cols = rows[run], cols[run]
        # Points either side of the antimeridian are read as two windows,
        # not one spanning the world.
        if run_cols.max() - run_cols.min() > half:
            west = run_cols < half
            for part in (west, ~west):
                result[run][part] = read_cells(path, run_rows[part], run_cols[part])
        else:
            result[run] = read_cells(path, run_rows, run_cols)
    return result.reshape(x.shape)


def read_cells(path: Path, rows: IntArray, cols: IntArray) -> Float32Array:
    """The land share of each cell given by its row and column, from one window."""
    row_off, col_off = int(rows.min()), int(cols.min())
    window = Window(
        col_off, row_off, int(cols.max()) - col_off + 1, int(rows.max()) - row_off + 1
    )
    with rasterio.open(path) as src:
        land = src.read(1, window=window)
        nodata = src.nodata
    share = land.astype(np.float32) / np.float32(MOLL_100M.cell_size**2)
    share[land == nodata] = 0.0
    result: Float32Array = share[rows - row_off, cols - col_off]
    return result
