"""Administrative units on a classification grid.

A cell belongs to the unit whose shape contains its centre, as a pixel belongs
to the cell containing its centre (`grid.pixel_to_cell`). Cells whose centres
are in no shape, such as the open ocean, belong to none (0).

The shapes are read from rra-population-model's admin inputs: the GBD and
LSAE files it caches, and the processed set it rakes Populace to. The raster of
a grid and shapes version is made once and kept under admin-rasters/.
"""

import hashlib
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
from scipy import ndimage

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


def raking_inputs() -> Path:
    """rra-population-model's processed raking units: the shapes and populations it rakes to."""
    return puc.POPULATION_MODEL_ROOT / "admin-inputs" / "raking"


def admin_inputs() -> Path:
    """The GBD and LSAE files rra-population-model caches, as they come."""
    return raking_inputs() / "gbd-inputs"


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

    A raking set's units are instead the locations rra-population-model rakes
    to, from its processed shapes and population files of the same name. Those
    not in the hierarchy, the places GBD does not estimate, are put under their
    GBD region, at a country's level.
    """

    units: str  # the shapes file holding the units, such as lsae_1578_a2
    countries: str | None  # the shapes file holding the countries, if there is one
    hierarchy: str  # the hierarchy the units belong to
    country_level: int = 1  # 1 when countries are just below the world
    raking: bool = False  # the units are the model's processed raking units

    @property
    def hierarchy_path(self) -> Path:
        return admin_inputs() / f"hierarchy_{self.hierarchy}.parquet"

    @property
    def units_path(self) -> Path:
        root = raking_inputs() if self.raking else admin_inputs()
        return root / f"shapes_{self.units}.parquet"

    @property
    def countries_path(self) -> Path | None:
        return None if self.countries is None else shapes_path(self.countries)

    @property
    def raking_population_path(self) -> Path:
        return raking_inputs() / f"population_{self.units}.parquet"

    def input_files(self) -> list[Path]:
        """The files the version is read from."""
        files = [self.units_path, self.hierarchy_path]
        if self.countries_path is not None:
            files.append(self.countries_path)
        if self.raking:
            files.append(self.raking_population_path)
        return files

    def paths(self) -> pd.DataFrame:
        hierarchy = pd.read_parquet(
            self.hierarchy_path,
            columns=["location_id", "path_to_top_parent", "most_detailed"],
        )
        if self.raking:
            hierarchy = self._raking_hierarchy(hierarchy)
        return hierarchy.assign(path=hierarchy["path_to_top_parent"].str.split(","))

    def _raking_hierarchy(self, hierarchy: pd.DataFrame) -> pd.DataFrame:
        raked = pd.read_parquet(
            self.raking_population_path, columns=["location_id", "region_id"]
        ).drop_duplicates("location_id")
        known = hierarchy.set_index("location_id")["path_to_top_parent"]
        added = raked[~raked["location_id"].isin(known.index)]
        region_path = added["region_id"].map(known)
        if region_path.isna().any():
            missing = sorted(added.loc[region_path.isna(), "location_id"])
            msg = f"Raking units {missing} have no region in the {self.hierarchy} hierarchy."
            raise ValueError(msg)
        added = added.assign(
            path_to_top_parent=region_path + "," + added["location_id"].astype(str)
        )
        combined = pd.concat(
            [hierarchy[["location_id", "path_to_top_parent"]], added],
            ignore_index=True,
        )
        is_unit = combined["location_id"].isin(raked["location_id"])
        return combined.assign(most_detailed=is_unit.astype(int))[
            ["location_id", "path_to_top_parent", "most_detailed"]
        ]

    def ihme_loc_ids(self) -> pd.Series:
        """Each location's ihme_loc_id, by location_id."""
        frames = [
            pd.read_parquet(self.hierarchy_path, columns=["location_id", "ihme_loc_id"])
        ]
        if self.raking:
            frames.append(
                pd.read_parquet(
                    self.raking_population_path, columns=["location_id", "ihme_loc_id"]
                )
            )
        combined = pd.concat(frames, ignore_index=True).drop_duplicates("location_id")
        ids: pd.Series = combined.set_index("location_id")["ihme_loc_id"]
        return ids

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
    # The units rra-population-model rakes Populace to: GBD 2023's most
    # detailed locations, except that India's states and New Zealand stand in
    # for their urban and rural parts and their Maori and non-Maori
    # populations, whose shapes are not good places; with northern Cyprus in
    # Cyprus; and with the places GBD does not estimate. Its paths run world,
    # super-region, region, country. It has no separate country shapes, so its
    # units are checked for overlaps instead.
    "raking_gbd_2023": ShapesVersion(
        units="gbd_2023",
        countries=None,
        hierarchy="gbd_2023",
        country_level=3,
        raking=True,
    ),
}


def checksums(version: ShapesVersion) -> dict[str, str]:
    """The SHA-256 of each file the version is read from, by path."""
    sums = {}
    for path in version.input_files():
        digest = hashlib.sha256()
        with path.open("rb") as f:
            while chunk := f.read(1 << 24):
                digest.update(chunk)
        sums[str(path)] = digest.hexdigest()
    return sums


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
    path: Path, crs: str, keep: list[int] | None = None
) -> gpd.GeoDataFrame:
    """A shapes file's `location_id` and geometry, projected to `crs`.

    With `keep`, only the shapes of those locations.

    Some shapes have vertices a hair past ±180° of longitude. A projection
    would wrap each one to the other side of the world, and the edges to it
    would cross the world, so a unit would be drawn across every country in
    its band of latitude. Coordinates are clamped to the world's bounds first,
    which moves those vertices by far less than a metre.
    """
    frame = gpd.read_parquet(path, columns=["location_id", "geometry"])
    if keep is not None:
        frame = frame[frame["location_id"].isin(keep)]
    if frame.crs is None or not frame.crs.is_geographic:
        msg = f"The shapes in {path} are not in longitude and latitude."
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
        values = rasterize(read_shapes(shapes_path(shapes), grid.crs), grid)
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


def nearest_country_raster(
    grid: GridDefinition, version: ShapesVersion
) -> LocationArray:
    """Each cell's country, with cells in no unit given the nearest unit's country.

    A cell's unit is the one containing its centre. A cell whose centre is in
    no unit, such as a coastal cell whose centre is at sea, takes the country
    of the nearest cell that is in one, so every cell has a country and the
    people on the coast are counted with their country.
    """
    units = rasterize(
        read_shapes(version.units_path, grid.crs, version.unit_ids()), grid
    )
    nearest = ndimage.distance_transform_edt(
        units == NO_UNIT, return_distances=False, return_indices=True
    )
    units = units[nearest[0], nearest[1]]
    parents = unit_parents(version)
    size = max(int(units.max()), int(parents["location_id"].max())) + 1
    lookup = np.zeros(size, dtype=np.int32)
    lookup[parents["location_id"]] = parents["country_id"]
    countries: LocationArray = lookup[units]
    return countries
