"""Paths under the pipeline's output root.

reference/<provider>/<product>/<release>/
    External inputs as downloaded. Read-only once downloaded: a new release
    goes in a new directory.
grids/<grid>/population_by_admin/<source>/<version components...>/<shapes>/<time point>.parquet
    People by grid cell and admin unit: row, col, location_id, population.
grids/<grid>/<layer>/<source>/<version components...>/<time point>.tif
    Input layers on a classification grid, shared by every classification
    that uses the same source version. Layers that do not vary in time use
    the time point "static".
admin-rasters/<grid>/<shapes>.tif
    Each cell's administrative unit (location_id) under a shapes version,
    such as lsae_1578_a0, from the shape containing the cell's centre.
admin-rasters/40m/<shapes>/<block key>.tif
    The same for each pixel of a block of the 40 m model grid.
classifications/<version>/
    One run of a classification specification: the saved specification,
    the class scheme, a manifest of what the run resolved, and per time point
    a class raster under classes/, an urban-centre id raster under
    urban_centres/ and a cluster table under clusters/.
"""

from pathlib import Path
from typing import Literal, Protocol

from rra_population_urbanicity import constants as puc

Layer = Literal["population", "land", "built"]


class GridSource(Protocol):
    @property
    def source(self) -> str: ...

    @property
    def version_components(self) -> tuple[str, ...]: ...


class UrbanicityData:
    def __init__(self, root: str | Path = puc.URBANICITY_ROOT) -> None:
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    def reference_root(self, provider: str, product: str, release: str) -> Path:
        return self.root / "reference" / provider / product / release

    @property
    def classifications_root(self) -> Path:
        return self.root / "classifications"

    def classification_root(self, version: str) -> Path:
        return self.classifications_root / version

    def class_raster_path(self, version: str, time_point: str) -> Path:
        return self.classification_root(version) / "classes" / f"{time_point}.tif"

    def urban_centre_raster_path(self, version: str, time_point: str) -> Path:
        return self.classification_root(version) / "urban_centres" / f"{time_point}.tif"

    def cluster_table_path(self, version: str, time_point: str) -> Path:
        return self.classification_root(version) / "clusters" / f"{time_point}.parquet"

    def admin_raster_path(self, grid: str, shapes: str) -> Path:
        return self.root / "admin-rasters" / grid / f"{shapes}.tif"

    def admin_block_root(self, shapes: str) -> Path:
        return self.root / "admin-rasters" / "40m" / shapes

    def admin_block_path(self, shapes: str, block_key: str) -> Path:
        return self.admin_block_root(shapes) / f"{block_key}.tif"

    def population_by_admin_root(
        self, grid: str, source: GridSource, shapes: str
    ) -> Path:
        return (
            self.root
            / "grids"
            / grid
            / "population_by_admin"
            / source.source
            / Path(*source.version_components)
            / shapes
        )

    def grid_path(
        self, grid: str, layer: Layer, source: GridSource, time_point: str
    ) -> Path:
        return (
            self.root
            / "grids"
            / grid
            / layer
            / source.source
            / Path(*source.version_components)
            / f"{time_point}.tif"
        )
