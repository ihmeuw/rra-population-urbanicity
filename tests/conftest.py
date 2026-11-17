from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import rasterio
from affine import Affine

from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.grid import GRIDS, GridDefinition
from rra_population_urbanicity.inputs import blocks
from rra_population_urbanicity.inputs.blocks import MODEL_CRS
from rra_population_urbanicity.reference.ghsl import GhslRaster

# A grid of 12 x 12 cells of 1 km at the origin of World Mollweide.
TINY = GridDefinition(
    name="tiny_1km",
    crs=puc.CRSES["mollweide"],
    cell_size=1000.0,
    x_min=-6000.0,
    y_max=6000.0,
    n_rows=12,
    n_cols=12,
)
CELL_M2 = 1e6
UINT32_NODATA = 4294967295


def write_raster(
    path: Path, values: npt.NDArray[Any], nodata: float, grid: GridDefinition = TINY
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=grid.n_rows,
        width=grid.n_cols,
        count=1,
        dtype=values.dtype,
        crs=grid.crs,
        transform=grid.transform,
        nodata=nodata,
    ) as dst:
        dst.write(values, 1)


def write_ghsl(
    root: Path, raster: GhslRaster, values: npt.NDArray[Any], nodata: float
) -> None:
    out_dir = UrbanicityData(root).reference_root(
        "ghsl", raster.product_dir, raster.release_dir
    )
    write_raster(out_dir / f"{raster.stem}.tif", values, nodata)


@pytest.fixture
def ghsl_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An output root whose reference/ holds GHSL layers on the `tiny_1km` grid.

    A town of 50,000 people at 2,000/km² sits in rows and columns 2-6, half
    built up, in both epochs; built-up grows to 60% by 2025. Column 9 is a
    river. One population cell and one land cell are missing.
    """
    monkeypatch.setitem(GRIDS, TINY.name, TINY)
    root = tmp_path / "output"
    land = np.full(TINY.shape, CELL_M2, dtype=np.uint32)
    land[:, 9] = 0
    land[0, 11] = UINT32_NODATA
    write_ghsl(
        root,
        GhslRaster(product="LAND", release="R2022A", epoch=2018),
        land,
        UINT32_NODATA,
    )
    for epoch, built_share in [(2020, 0.5), (2025, 0.6)]:
        population = np.zeros(TINY.shape)
        population[2:7, 2:7] = 2000
        population[11, 11] = -200
        write_ghsl(
            root,
            GhslRaster(product="POP", release="R2023A", epoch=epoch),
            population,
            -200,
        )
        built = np.zeros(TINY.shape, dtype=np.uint32)
        built[2:7, 2:7] = built_share * CELL_M2
        write_ghsl(
            root,
            GhslRaster(product="BUILT_S", release="R2023A", epoch=epoch),
            built,
            UINT32_NODATA,
        )
    return root


def ghsl_spec(**parameters: Any) -> dict[str, Any]:
    """A specification of the V2.0 reproduction on the `tiny_1km` grid."""
    return {
        "population": {"source": "ghs_pop", "release": "r2023a"},
        "grid": TINY.name,
        "land": {"source": "ghs_land", "release": "r2022a"},
        "built": {"source": "ghs_built_s", "release": "r2023a"},
        "time_points": ["2020q1", "2025q1"],
        "method": {
            "name": "degurba_r2023a",
            "parameter_set": "ghsl_r2023a",
            "parameters": parameters,
        },
    }


@pytest.fixture
def small_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    """Shrink the model grid's blocks to 4 pixels and its group tiles to 2 x 2 blocks.

    The raster is 10 pixels wide and 7 high, so there are 3 x 2 blocks, and the
    last column and row of blocks are cut short. Its top-left corner is 120 m
    west and 80 m north of the origin, so the first block straddles the equator
    and the prime meridian.
    """
    monkeypatch.setattr(blocks, "X_MIN", -120.0)
    monkeypatch.setattr(blocks, "Y_MAX", 80.0)
    monkeypatch.setattr(
        blocks, "MODEL_TRANSFORM", Affine(40.0, 0.0, -120.0, 0.0, -40.0, 80.0)
    )
    monkeypatch.setattr(blocks, "BLOCK_PIXELS", 4)
    monkeypatch.setattr(blocks, "GROUP_BLOCKS", 2)
    monkeypatch.setattr(blocks, "WIDTH", 10)
    monkeypatch.setattr(blocks, "HEIGHT", 7)
    monkeypatch.setattr(blocks, "N_BLOCK_COLS", 3)
    monkeypatch.setattr(blocks, "N_BLOCK_ROWS", 2)


def write_model_raster(
    path: Path, values: npt.NDArray[Any], transform: Affine, crs: str = MODEL_CRS
) -> None:
    """Write a float32 raster with NaN as nodata, as the model's outputs are."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=np.nan,
    ) as dst:
        dst.write(values.astype(np.float32), 1)
