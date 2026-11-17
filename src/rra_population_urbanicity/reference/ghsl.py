"""Download the GHSL inputs and documentation for reproducing GHS-SMOD.

The rasters come from the JRC open-data site, where each GHS-SMOD epoch is
published in two versions (V1-0 and V2-0) with different level-2 rules. Files
already on disk are never downloaded again.
"""

import zipfile
from pathlib import Path
from typing import Literal

import click
import pyproj
import rasterio
import requests
from pydantic import BaseModel, ConfigDict
from rra_tools.cli_tools import with_output_directory
from rra_tools.shell_tools import mkdir

from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.grid import GRIDS, GridDefinition

GHSL_FTP_ROOT = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL"
# The 2024 edition of the GHSL Data Package 2023 (PDF dated 2024-09-24), which
# documents GHS-SMOD R2023A V2.0.
DATA_PACKAGE_URL = "https://human-settlement.emergency.copernicus.eu/documents/GHSL_Data_Package_2023.pdf"
DATA_PACKAGE_RELEASE = "2023_2024-09-24"
TIMEOUT_SECONDS = 60
CHUNK_BYTES = 2**20

# The 100 m grid of GHSL's layers, which nests 10 x 10 in `moll_1km`.
MOLL_100M = GridDefinition(
    name="moll_100m",
    crs=puc.CRSES["mollweide"],
    cell_size=100.0,
    x_min=-18_041_000.0,
    y_max=9_000_000.0,
    n_rows=180_000,
    n_cols=360_820,
)


class GhslRaster(BaseModel):
    """A GHSL raster on the World Mollweide grid at 1 km or 100 m."""

    model_config = ConfigDict(frozen=True)

    product: Literal["POP", "BUILT_S", "SMOD", "LAND"]
    release: Literal["R2022A", "R2023A"]
    epoch: int
    version: Literal[1, 2] = 1
    resolution: Literal[100, 1000] = 1000

    @property
    def stem(self) -> str:
        return (
            f"GHS_{self.product}_E{self.epoch}_GLOBE_{self.release}"
            f"_54009_{self.resolution}_V{self.version}_0"
        )

    @property
    def url(self) -> str:
        return (
            f"{GHSL_FTP_ROOT}/GHS_{self.product}_GLOBE_{self.release}"
            f"/GHS_{self.product}_E{self.epoch}_GLOBE_{self.release}_54009_{self.resolution}"
            f"/V{self.version}-0/{self.stem}.zip"
        )

    @property
    def grid(self) -> GridDefinition:
        return {1000: GRIDS["moll_1km"], 100: MOLL_100M}[self.resolution]

    @property
    def product_dir(self) -> str:
        return f"ghs_{self.product.lower()}"

    @property
    def release_dir(self) -> str:
        # GHS-SMOD R2023A has two versions with different rules, so its
        # version is part of the release; the other products have one.
        if self.product == "SMOD":
            return f"{self.release.lower()}_v{self.version}"
        return self.release.lower()


# Everything the GHS-SMOD reproduction needs: the inputs and both versions of
# GHS-SMOD itself, for 2020 and 2025. GHS-LAND has a single epoch, and is also
# downloaded at 100 m.
GHSL_RASTERS = [
    GhslRaster(product="POP", release="R2023A", epoch=2020),
    GhslRaster(product="POP", release="R2023A", epoch=2025),
    GhslRaster(product="BUILT_S", release="R2023A", epoch=2020),
    GhslRaster(product="BUILT_S", release="R2023A", epoch=2025),
    GhslRaster(product="SMOD", release="R2023A", epoch=2020, version=1),
    GhslRaster(product="SMOD", release="R2023A", epoch=2020, version=2),
    GhslRaster(product="SMOD", release="R2023A", epoch=2025, version=1),
    GhslRaster(product="SMOD", release="R2023A", epoch=2025, version=2),
    GhslRaster(product="LAND", release="R2022A", epoch=2018),
    # For a land layer on the 40 m model grid (the land-source investigation).
    GhslRaster(product="LAND", release="R2022A", epoch=2018, resolution=100),
]


def fetch(url: str, path: Path) -> None:
    """Stream a URL to a file, under a temporary name until it is complete."""
    partial = path.with_name(f"{path.name}.part")
    with requests.get(url, stream=True, timeout=TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        with partial.open("wb") as f:
            for chunk in response.iter_content(chunk_size=CHUNK_BYTES):
                f.write(chunk)
    partial.rename(path)


def check_on_grid(path: Path, grid: GridDefinition) -> None:
    """Raise unless a raster has exactly the grid's CRS, transform and shape."""
    with rasterio.open(path) as src:
        on_grid = (
            pyproj.CRS.from_user_input(src.crs.to_wkt()).equals(grid.crs)
            and src.transform == grid.transform
            and src.shape == grid.shape
        )
    if not on_grid:
        msg = f"{path} is not on the {grid.name} grid."
        raise ValueError(msg)


def download_raster(raster: GhslRaster, out_dir: Path, grid: GridDefinition) -> Path:
    """Download and unpack a raster unless it is already there, and check its grid."""
    path = out_dir / f"{raster.stem}.tif"
    if not path.exists():
        mkdir(out_dir, parents=True, exist_ok=True)
        archive = out_dir / f"{raster.stem}.zip"
        fetch(raster.url, archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
        archive.unlink()
    check_on_grid(path, grid)
    return path


def download_data_package(out_dir: Path) -> Path:
    path = out_dir / "GHSL_Data_Package_2023.pdf"
    if not path.exists():
        mkdir(out_dir, parents=True, exist_ok=True)
        fetch(DATA_PACKAGE_URL, path)
    return path


@click.command()
@with_output_directory(puc.URBANICITY_ROOT)
def download_ghsl(output_dir: str) -> None:
    """Download the GHSL rasters and documentation for reproducing GHS-SMOD."""
    data = UrbanicityData(output_dir)
    for raster in GHSL_RASTERS:
        out_dir = data.reference_root("ghsl", raster.product_dir, raster.release_dir)
        print(download_raster(raster, out_dir, raster.grid))
    out_dir = data.reference_root("ghsl", "data_package", DATA_PACKAGE_RELEASE)
    print(download_data_package(out_dir))
