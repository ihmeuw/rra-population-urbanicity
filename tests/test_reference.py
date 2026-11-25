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
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.grid import GridDefinition
from rra_population_urbanicity.reference import download, ghsl, us_census, wup

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


def test_wup_rasters() -> None:
    raster = ghsl.GhslRaster(product="WUP_DEGURBA", release="R2025A", epoch=2025)
    assert raster.url == (
        "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL"
        "/GHS_WUP_DEGURBA_GLOBE_R2025A/GHS_WUP_DEGURBA_E2025_GLOBE_R2025A_54009_1000"
        "/V1-0/GHS_WUP_DEGURBA_E2025_GLOBE_R2025A_54009_1000_V1_0.zip"
    )
    assert (raster.product_dir, raster.release_dir) == ("ghs_wup_degurba", "r2025a")


def test_sixteen_distinct_rasters_are_downloaded() -> None:
    # Nine for the reproduction, GHS-LAND at 100 m, and GHS-WUP's population,
    # built-up and classes for two epochs.
    assert len({raster.stem for raster in ghsl.GHSL_RASTERS}) == 16


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


def test_an_archive_must_hold_the_expected_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("readme.txt", "hello")
    server = FakeServer(archive.getvalue())
    monkeypatch.setattr(requests, "get", server.get)
    url = "https://example.org/files/thing.zip"

    assert download.download_archive(url, tmp_path, "readme.txt").read_text() == "hello"
    assert [p.name for p in tmp_path.iterdir()] == ["readme.txt"]
    with pytest.raises(FileNotFoundError, match="did not hold thing.gpkg"):
        download.download_archive(url, tmp_path / "other", "thing.gpkg")


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
    grids: dict[str, GridDefinition] = {}

    def download_raster(
        raster: ghsl.GhslRaster, out_dir: Path, grid: GridDefinition
    ) -> Path:
        downloads.append(out_dir / raster.stem)
        grids[raster.stem] = grid
        return out_dir

    def download_data_package(out_dir: Path) -> Path:
        downloads.append(out_dir)
        return out_dir

    def download_archive(url: str, out_dir: Path, expected: str) -> Path:
        downloads.append(out_dir / expected)
        return out_dir

    monkeypatch.setattr(ghsl, "download_raster", download_raster)
    monkeypatch.setattr(ghsl, "download_data_package", download_data_package)
    monkeypatch.setattr(ghsl, "download_archive", download_archive)
    result = CliRunner().invoke(
        purun, ["reference", "ghsl", "--output-dir", str(tmp_path)]
    )

    assert result.exit_code == 0, result.output
    reference = tmp_path / "reference" / "ghsl"
    assert len(downloads) == 19
    assert reference / "ghs_smod" / "r2023a_v2" / SMOD_2025_V2.stem in downloads
    # GHS-LAND comes at 100 m too, checked against GHSL's 100 m grid.
    land_100m = "GHS_LAND_E2018_GLOBE_R2022A_54009_100_V1_0"
    assert reference / "ghs_land" / "r2022a" / land_100m in downloads
    assert grids[land_100m] == ghsl.MOLL_100M
    assert grids[SMOD_2025_V2.stem].name == "moll_1km"
    assert downloads[17] == (
        reference
        / "ghs_wup_mtuc"
        / "r2025a_v1_1"
        / "GHS_WUP_MTUC_MT_GLOBE_R2025A_v1_1.shp"
    )
    assert downloads[-1] == reference / "data_package" / ghsl.DATA_PACKAGE_RELEASE


def test_wup_tables_are_copied_once(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for name in wup.FILES.values():
        (source / name).write_text(name)
    data = UrbanicityData(tmp_path / "out")

    paths = wup.copy_tables(data, source)
    assert [p.read_text() for p in paths] == list(wup.FILES.values())
    assert (
        paths[0].parent == tmp_path / "out" / "reference" / "un_desa" / "wup" / "2025"
    )
    (source / wup.FILES["level1"]).write_text("changed")
    wup.copy_tables(data, source)
    assert paths[0].read_text() == wup.FILES["level1"]


def test_wup_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    copied: list[UrbanicityData] = []

    def copy_tables(data: UrbanicityData) -> list[Path]:
        copied.append(data)
        return [data.root / "table.csv"]

    monkeypatch.setattr(wup, "copy_tables", copy_tables)
    result = CliRunner().invoke(
        purun, ["reference", "wup", "--output-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert copied[0].root == tmp_path
    assert result.output == f"{tmp_path / 'table.csv'}\n"


def test_us_census_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, Path, str]] = []
    fetched: list[str] = []

    def download_archive(url: str, out_dir: Path, expected: str) -> Path:
        calls.append((url, out_dir, expected))
        return out_dir / expected

    def fetch(url: str, path: Path) -> None:
        fetched.append(url)
        path.write_text("table")

    monkeypatch.setattr(us_census, "download_archive", download_archive)
    monkeypatch.setattr(us_census, "fetch", fetch)
    # The county table is fetched once; the shapes are left to download_archive.
    for _ in range(2):
        result = CliRunner().invoke(
            purun, ["reference", "us-census", "--output-dir", str(tmp_path)]
        )
        assert result.exit_code == 0, result.output
    out_dir = tmp_path / "reference" / "us_census" / "urban_areas" / "2020"
    assert calls[0] == (us_census.URBAN_AREAS_URL, out_dir, us_census.URBAN_AREAS_FILE)
    assert fetched == [us_census.COUNTY_TABLE_URL]
    assert (out_dir / "2020_UA_COUNTY.xlsx").read_text() == "table"
