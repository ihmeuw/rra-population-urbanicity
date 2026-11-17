"""Administrative units on a classification grid.

A cell belongs to the unit whose shape contains its centre, as a pixel belongs
to the cell containing its centre (`grid.pixel_to_cell`). Cells whose centres
are in no shape, such as the open ocean, belong to none (0).

The shapes are the ones rra-population-model rakes to, read from its admin
inputs. The raster of a grid and shapes version is made once and kept under
admin-rasters/.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import rasterio
from rasterio import features
from rra_tools.shell_tools import mkdir

from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.grid import GridDefinition

type LocationArray = npt.NDArray[np.int32]

# No unit: the cell's centre is in no shape.
NO_UNIT = 0
# Tiles of the raster, in cells.
BLOCK_SIZE = 512


def shapes_path(shapes: str) -> Path:
    """The shapes of a version such as lsae_1578_a0, among the model's admin inputs."""
    return (
        puc.POPULATION_MODEL_ROOT
        / "admin-inputs"
        / "raking"
        / "gbd-inputs"
        / f"shapes_{shapes}.parquet"
    )


def rasterize(shapes: gpd.GeoDataFrame, grid: GridDefinition) -> LocationArray:
    """Each cell's location_id, from the shape containing its centre."""
    projected = shapes.to_crs(grid.crs)
    values: LocationArray = features.rasterize(
        zip(projected.geometry, projected.location_id.astype(int), strict=True),
        out_shape=grid.shape,
        transform=grid.transform,
        fill=NO_UNIT,
        dtype="int32",
    )
    return values


def location_raster(
    data: UrbanicityData, grid: GridDefinition, shapes: str
) -> LocationArray:
    """Each cell's location_id under a shapes version, made once and kept."""
    path = data.admin_raster_path(grid.name, shapes)
    if not path.exists():
        values = rasterize(gpd.read_parquet(shapes_path(shapes)), grid)
        mkdir(path.parent, parents=True, exist_ok=True)
        partial = path.with_name(f"{path.stem}.part.tif")
        with rasterio.open(
            partial,
            "w",
            driver="GTiff",
            height=grid.n_rows,
            width=grid.n_cols,
            count=1,
            dtype="int32",
            crs=grid.crs,
            transform=grid.transform,
            nodata=None,
            tiled=True,
            blockxsize=BLOCK_SIZE,
            blockysize=BLOCK_SIZE,
            compress="deflate",
        ) as dst:
            dst.write(values, 1)
        partial.rename(path)
        return values
    with rasterio.open(path) as src:
        stored: LocationArray = src.read(1)
    return stored
