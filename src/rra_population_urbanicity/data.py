"""Paths under the pipeline's output root.

reference/<provider>/<product>/<release>/
    External inputs as downloaded. Read-only once downloaded: a new release
    goes in a new directory.
grids/<grid>/<layer>/<source>/<version components...>/<time point>.tif
    Input layers on a classification grid, shared by every classification
    that uses the same source version. Layers that do not vary in time use
    the time point "static".
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
