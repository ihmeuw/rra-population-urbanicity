from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import rasterio
from affine import Affine
from rasterio.windows import Window
from rra_tools import jobmon

from rra_population_urbanicity import admin
from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.classify import kernel
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.grid import GRIDS, GridDefinition
from rra_population_urbanicity.inputs import blocks
from rra_population_urbanicity.inputs.blocks import MODEL_CRS
from rra_population_urbanicity.inputs.providers import populace
from rra_population_urbanicity.reference.ghsl import GhslRaster
from rra_population_urbanicity.specification import ClassificationSpecification

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


def frame_units() -> npt.NDArray[np.int32]:
    """Unit 11 west of the meridian, 12 north-east, 21 south-east, and one pixel in none."""
    units = np.zeros((blocks.HEIGHT, blocks.WIDTH), dtype=np.int32)
    units[:, :3] = 11
    units[:2, 3:] = 12
    units[2:, 3:] = 21
    units[6, 9] = admin.NO_UNIT
    return units


def write_units(
    data: UrbanicityData, units: npt.NDArray[np.int32], shapes: str
) -> None:
    """A set of units' raster for each block of the model grid."""
    for block in blocks.all_blocks():
        path = data.admin_block_path(shapes, block.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        values = units[block.window.toslices()]
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=values.shape[0],
            width=values.shape[1],
            count=1,
            dtype="int32",
            crs=MODEL_CRS,
            transform=block.transform,
        ) as dst:
            dst.write(values, 1)


KERNEL_UNIT_SETS = ("units_a", "units_b")
KERNEL_TIME_POINTS = ["2020q1", "2023q1"]


def frame_people(time_point: str) -> npt.NDArray[np.float64]:
    """625 people per km² west of column 5 and 62.5 east of it; block (2, 1) empty.

    In 2023 the top-left pixel has 100 people.
    """
    people = np.full((blocks.HEIGHT, blocks.WIDTH), 0.1)
    people[:, :5] = 1.0
    people[4:, 8:] = 0.0
    if time_point == "2023q1":
        people[0, 0] = 100.0
    return people


def frame_land(data: UrbanicityData, window: Window) -> npt.NDArray[np.float32]:
    """All land on the frame, and none beyond it."""
    land = np.zeros((int(window.height), int(window.width)), dtype=np.float32)
    rows = slice(max(-int(window.row_off), 0), blocks.HEIGHT - int(window.row_off))
    cols = slice(max(-int(window.col_off), 0), blocks.WIDTH - int(window.col_off))
    land[rows, cols] = 1.0
    return land


def kernel_spec() -> ClassificationSpecification:
    # A disc of one pixel's radius: the pixel and its four neighbours.
    return ClassificationSpecification.model_validate(
        {
            "population": {
                "source": "populace",
                "model_version": "2026_10_02.001",
                "resolution": "40",
            },
            "grid": TINY.name,
            "land": {"source": "ghs_land", "release": "r2022a"},
            "time_points": KERNEL_TIME_POINTS,
            "method": {
                "name": "kernel_density",
                "parameter_set": "kernel_default",
                "parameters": {"radius_m": 40.0},
            },
        }
    )


@pytest.fixture
def kernel_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_frame: None
) -> Path:
    monkeypatch.setitem(GRIDS, TINY.name, TINY)
    model_root = tmp_path / "population-model"
    monkeypatch.setattr(puc, "POPULATION_MODEL_ROOT", model_root)
    for time_point in KERNEL_TIME_POINTS:
        write_model_raster(
            model_root / f"{time_point}.tif",
            frame_people(time_point),
            blocks.MODEL_TRANSFORM,
        )
    monkeypatch.setattr(
        populace, "vrt_path", lambda root, source, tp: root / f"{tp}.tif"
    )
    monkeypatch.setattr(kernel, "read_land", frame_land)
    monkeypatch.setattr(kernel, "UNIT_SETS", KERNEL_UNIT_SETS)
    root = tmp_path / "output"
    data = UrbanicityData(root)
    for shapes in KERNEL_UNIT_SETS:
        write_units(data, frame_units(), shapes)

    def run_parallel(**kwargs: Any) -> str:
        for key in kwargs["node_args"]["block-key"]:
            kernel.run_block(data, kwargs["task_args"]["version"], key)
        return "done"

    monkeypatch.setattr(jobmon, "run_parallel", run_parallel)
    return root
