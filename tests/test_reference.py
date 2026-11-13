import io
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import Self

import numpy as np
import pytest
import rasterio
import requests
from click.testing import CliRunner

from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.cli import purun
from rra_population_urbanicity.grid import GridDefinition
from rra_population_urbanicity.reference import ghsl

TEST_GRID = GridDefinition(
    name="test",
    crs=puc.CRSES["mollweide"],
    cell_size=1000.0,
    x_min=-2000.0,
    y_max=3000.0,
    n_rows=3,
    n_cols=4,
)
POP_2020 = ghsl.GhslRaster(product="POP", release="R2023A", epoch=2020)
SMOD_2025_V2 = ghsl.GhslRaster(product="SMOD", release="R2023A", epoch=2025, version=2)


class FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> Iterator[bytes]:
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start : start + chunk_size]


class FakeServer:
    """Serves the same content at every URL and records the URLs requested."""

    def __init__(self, content: bytes) -> None:
        self.content = content
        self.urls: list[str] = []

    def get(self, url: str, **kwargs: object) -> FakeResponse:
        self.urls.append(url)
        return FakeResponse(self.content)


def zipped_raster(tmp_path: Path, stem: str, grid: GridDefinition) -> bytes:
    """A zip archive holding one GeoTIFF on the grid, as GHSL publishes them."""
    path = tmp_path / f"{stem}.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=grid.n_cols,
        height=grid.n_rows,
        count=1,
        dtype="uint8",
        crs=grid.crs,
        transform=grid.transform,
    ) as dst:
        dst.write(np.zeros(grid.shape, dtype=np.uint8), 1)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.write(path, path.name)
    return archive.getvalue()


def test_raster_urls_and_directories() -> None:
    assert POP_2020.url == (
        "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/GHS_POP_GLOBE_R2023A"
        "/GHS_POP_E2020_GLOBE_R2023A_54009_1000"
        "/V1-0/GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0.zip"
    )
    assert (POP_2020.product_dir, POP_2020.release_dir) == ("ghs_pop", "r2023a")
    assert SMOD_2025_V2.url.endswith(
        "/GHS_SMOD_E2025_GLOBE_R2023A_54009_1000"
        "/V2-0/GHS_SMOD_E2025_GLOBE_R2023A_54009_1000_V2_0.zip"
    )
    assert (SMOD_2025_V2.product_dir, SMOD_2025_V2.release_dir) == (
        "ghs_smod",
        "r2023a_v2",
    )


def test_the_reproduction_needs_nine_rasters() -> None:
    assert len({raster.stem for raster in ghsl.GHSL_RASTERS}) == 9


def test_download_raster(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeServer(zipped_raster(tmp_path, POP_2020.stem, TEST_GRID))
    monkeypatch.setattr(requests, "get", server.get)
    out_dir = tmp_path / "reference" / "ghs_pop"

    path = ghsl.download_raster(POP_2020, out_dir, TEST_GRID)
    # The archive and the partial download are gone; only the raster is left.
    assert [p.name for p in out_dir.iterdir()] == [f"{POP_2020.stem}.tif"]
    assert path == out_dir / f"{POP_2020.stem}.tif"

    # A raster already on disk is not downloaded again.
    ghsl.download_raster(POP_2020, out_dir, TEST_GRID)
    assert server.urls == [POP_2020.url]


def test_rasters_off_the_grid_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    shifted = TEST_GRID.model_copy(update={"x_min": -1500.0})
    server = FakeServer(zipped_raster(tmp_path, POP_2020.stem, shifted))
    monkeypatch.setattr(requests, "get", server.get)
    with pytest.raises(ValueError, match="is not on the test grid"):
        ghsl.download_raster(POP_2020, tmp_path / "out", TEST_GRID)


def test_download_data_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    server = FakeServer(b"%PDF-1.4")
    monkeypatch.setattr(requests, "get", server.get)

    path = ghsl.download_data_package(tmp_path)
    ghsl.download_data_package(tmp_path)

    assert path.read_bytes() == b"%PDF-1.4"
    assert server.urls == [ghsl.DATA_PACKAGE_URL]


def test_ghsl_command_downloads_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    downloads: list[Path] = []

    def download_raster(
        raster: ghsl.GhslRaster, out_dir: Path, grid: GridDefinition
    ) -> Path:
        downloads.append(out_dir / raster.stem)
        return out_dir

    def download_data_package(out_dir: Path) -> Path:
        downloads.append(out_dir)
        return out_dir

    monkeypatch.setattr(ghsl, "download_raster", download_raster)
    monkeypatch.setattr(ghsl, "download_data_package", download_data_package)
    result = CliRunner().invoke(
        purun, ["reference", "ghsl", "--output-dir", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    reference = tmp_path / "reference" / "ghsl"
    assert len(downloads) == 10
    assert reference / "ghs_smod" / "r2023a_v2" / SMOD_2025_V2.stem in downloads
    assert downloads[-1] == reference / "data_package" / ghsl.DATA_PACKAGE_RELEASE
