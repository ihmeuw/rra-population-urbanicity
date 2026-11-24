"""Administrative units on a classification grid.

A cell belongs to the unit whose shape contains its centre, as a pixel belongs
to the cell containing its centre (`grid.pixel_to_cell`). Cells whose centres
are in no shape, such as the open ocean, belong to none (0).

The shapes are the ones rra-population-model rakes to, read from its admin
inputs. The raster of a grid and shapes version is made once and kept under
admin-rasters/.
"""

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
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


def admin_inputs() -> Path:
    return puc.POPULATION_MODEL_ROOT / "admin-inputs" / "raking" / "gbd-inputs"


def shapes_path(shapes: str) -> Path:
    """The shapes of a version such as lsae_1578_a0, among the model's admin inputs."""
    return admin_inputs() / f"shapes_{shapes}.parquet"


@dataclass(frozen=True)
class ShapesVersion:
    """A set of the most detailed admin units, and what they should add up to."""

    units: str  # the shapes of the most detailed units, such as lsae_1578_a2
    countries: str  # the country shapes the units should cover
    hierarchy: str  # the hierarchy the units belong to

    @property
    def hierarchy_path(self) -> Path:
        return admin_inputs() / f"hierarchy_{self.hierarchy}.parquet"


SHAPES = {
    # LSAE 1578's admin 2 units, the units of the RT deliverable.
    "lsae_1578_a2": ShapesVersion(
        units="lsae_1578_a2", countries="lsae_1578_a0", hierarchy="lsae_1578"
    ),
}


def unit_parents(version: ShapesVersion) -> pd.DataFrame:
    """Each unit's admin 1 and country, from its path in the hierarchy.

    Paths run from the top: 1 (the world), the country, then the admin 1.
    """
    hierarchy = pd.read_parquet(
        version.hierarchy_path, columns=["location_id", "path_to_top_parent"]
    )
    path = hierarchy["path_to_top_parent"].str.split(",")
    # Everything below the world.
    below = path.str.len() > 1
    hierarchy, path = hierarchy[below], path[below]
    return pd.DataFrame(
        {
            "location_id": hierarchy["location_id"].astype(np.int64),
            "country_id": path.str[1].astype(np.int64),
            "admin1_id": path.str[2].fillna(path.str[1]).astype(np.int64),
        }
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
