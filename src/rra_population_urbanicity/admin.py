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
import shapely
from rasterio import features
from rra_tools.shell_tools import mkdir

from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.grid import GridDefinition

type LocationArray = npt.NDArray[np.int32]

# No unit: the cell's centre is in no shape.
NO_UNIT = 0
# The world in longitude and latitude.
WORLD_MIN = np.array([-180.0, -90.0])
WORLD_MAX = np.array([180.0, 90.0])
# Tiles of the raster, in cells.
BLOCK_SIZE = 512


def admin_inputs() -> Path:
    return puc.POPULATION_MODEL_ROOT / "admin-inputs" / "raking" / "gbd-inputs"


def shapes_path(shapes: str) -> Path:
    """The shapes of a version such as lsae_1578_a0, among the model's admin inputs."""
    return admin_inputs() / f"shapes_{shapes}.parquet"


@dataclass(frozen=True)
class ShapesVersion:
    """A set of the most detailed admin units, and the countries they make up.

    The units are the hierarchy's most detailed locations, and the countries
    its locations at `country_level`: the position of the country in a path
    from the top of the hierarchy. A shapes file may hold more than either,
    as GBD's holds every level; only their shapes are used.
    """

    units: str  # the shapes file holding the units, such as lsae_1578_a2
    countries: str  # the shapes file holding the countries
    hierarchy: str  # the hierarchy the units belong to
    country_level: int = 1  # 1 when countries are just below the world

    @property
    def hierarchy_path(self) -> Path:
        return admin_inputs() / f"hierarchy_{self.hierarchy}.parquet"

    def paths(self) -> pd.DataFrame:
        hierarchy = pd.read_parquet(
            self.hierarchy_path,
            columns=["location_id", "path_to_top_parent", "most_detailed"],
        )
        return hierarchy.assign(path=hierarchy["path_to_top_parent"].str.split(","))

    def unit_ids(self) -> list[int]:
        hierarchy = self.paths()
        ids: list[int] = hierarchy.loc[
            hierarchy["most_detailed"] == 1, "location_id"
        ].tolist()
        return ids

    def country_ids(self) -> list[int]:
        hierarchy = self.paths()
        at_level = hierarchy["path"].str.len() == self.country_level + 1
        ids: list[int] = hierarchy.loc[at_level, "location_id"].tolist()
        return ids


SHAPES = {
    # LSAE 1578's admin 2 units, the units of the RT deliverable.
    "lsae_1578_a2": ShapesVersion(
        units="lsae_1578_a2", countries="lsae_1578_a0", hierarchy="lsae_1578"
    ),
    # GBD 2023's most detailed locations, which Populace is raked to. Its
    # paths run world, super-region, region, country.
    "gbd_2023": ShapesVersion(
        units="gbd_2023", countries="gbd_2023", hierarchy="gbd_2023", country_level=3
    ),
}


def unit_parents(version: ShapesVersion) -> pd.DataFrame:
    """Each location's country and admin 1, from its path in the hierarchy.

    The admin 1 is the location just below the country in the path, or the
    country itself for a country and for anything above it, which have no
    country of their own and are left out.
    """
    hierarchy = version.paths()
    path = hierarchy["path"]
    level = version.country_level
    has_country = path.str.len() > level
    hierarchy, path = hierarchy[has_country], path[has_country]
    return pd.DataFrame(
        {
            "location_id": hierarchy["location_id"].astype(np.int64),
            "country_id": path.str[level].astype(np.int64),
            "admin1_id": path.str[level + 1].fillna(path.str[level]).astype(np.int64),
        }
    )


def read_shapes(
    shapes: str, crs: str, keep: list[int] | None = None
) -> gpd.GeoDataFrame:
    """A shapes file's `location_id` and geometry, projected to `crs`.

    With `keep`, only the shapes of those locations.

    Some shapes have vertices a hair past ±180° of longitude. A projection
    would wrap each one to the other side of the world, and the edges to it
    would cross the world, so a unit would be drawn across every country in
    its band of latitude. Coordinates are clamped to the world's bounds first,
    which moves those vertices by far less than a metre.
    """
    frame = gpd.read_parquet(shapes_path(shapes), columns=["location_id", "geometry"])
    if keep is not None:
        frame = frame[frame["location_id"].isin(keep)]
    if frame.crs is None or not frame.crs.is_geographic:
        msg = f"The {shapes} shapes are not in longitude and latitude."
        raise ValueError(msg)
    clamped = shapely.transform(
        frame.geometry.to_numpy(), lambda xy: np.clip(xy, WORLD_MIN, WORLD_MAX)
    )
    frame = frame.set_geometry(gpd.GeoSeries(clamped, index=frame.index, crs=frame.crs))
    return frame.to_crs(crs)


def rasterize(shapes: gpd.GeoDataFrame, grid: GridDefinition) -> LocationArray:
    """Each cell's location_id, from the shape containing its centre.

    The shapes are in the grid's CRS.
    """
    values: LocationArray = features.rasterize(
        zip(shapes.geometry, shapes.location_id.astype(int), strict=True),
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
        values = rasterize(read_shapes(shapes, grid.crs), grid)
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
