"""Microsoft building layers and the water mask behind them.

Building density is the share of each 40 m pixel covered by buildings, in one
file per block and quarter, as rra-building-density puts it on the model
grid. Pixels outside the tiles' coverage hold no buildings.

The water mask is the static mask that decides where the building predictions,
and so Populace, can have anything. It is kept as rra-building-density
downloaded it: tiles of 512 x 512 pixels in longitude and latitude, one per
Web Mercator tile at zoom 11, named by the tile's column and its row counted
from the south. A pixel is 1 for water (permanent or seasonal) and 0 for land.
"""

import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyproj
import rasterio
from rasterio.windows import Window

from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.inputs.blocks import MODEL_CRS, Block
from rra_population_urbanicity.inputs.providers.model_grid import (
    Float32Array,
    read_block,
)
from rra_population_urbanicity.specification import (
    MicrosoftDensitySource,
    MicrosoftWaterSource,
)

type BoolArray = npt.NDArray[np.bool_]
type IntArray = npt.NDArray[np.int64]
type FloatArray = npt.NDArray[np.float64]

WATER_TILES = 2**11  # tiles around the world, and from pole to pole
WATER_TILE_PIXELS = 512

# Microsoft v8 starts in 2020q2. Like rra-population-model, which builds its
# 2020q1 surface from 2020q2 buildings, 2020q1 reads 2020q2.
DENSITY_TIME_POINTS = {"2020q1": "2020q2"}


def density_time_point(time_point: str) -> str:
    """The quarter of building density read for a time point."""
    return DENSITY_TIME_POINTS.get(time_point, time_point)


def density_path(
    root: Path, source: MicrosoftDensitySource, block: Block, time_point: str
) -> Path:
    return (
        root
        / "tiles"
        / f"{source.resolution}m"
        / f"microsoft_{source.building_version}"
        / density_time_point(time_point)
        / f"{block.key}_density.tif"
    )


def load_density_block(
    source: MicrosoftDensitySource,
    block: Block,
    time_point: str,
    root: Path = puc.BUILDING_DENSITY_ROOT,
) -> Float32Array:
    """The built-up share of each pixel of a block."""
    path = density_path(root, source, block, time_point)
    height, width = block.shape
    return read_block(path, block, Window(0, 0, width, height))


def water_tile_path(root: Path, tile_x: int, tile_y: int) -> Path:
    """The mask tile in column `tile_x` and row `tile_y`, counted from the south."""
    return (
        root
        / "raw_tiles"
        / "microsoft_water_mask"
        / f"L15-{tile_x:04d}E-{tile_y:04d}N.tif"
    )


def tile_edge_latitude(row: FloatArray) -> FloatArray:
    """The latitude of the top edge of Web Mercator tile rows, counted from the north."""
    return np.degrees(np.arctan(np.sinh(math.pi * (1 - 2 * row / WATER_TILES))))


def runs(values: IntArray) -> list[slice]:
    """The slices over which a 1-D array keeps one value."""
    edges = [0, *(np.flatnonzero(np.diff(values)) + 1).tolist(), len(values)]
    return [
        slice(start, stop) for start, stop in zip(edges[:-1], edges[1:], strict=True)
    ]


def load_land_block(
    source: MicrosoftWaterSource,  # noqa: ARG001
    block: Block,
    root: Path = puc.BUILDING_DENSITY_ROOT,
) -> BoolArray:
    """Whether the mask has land at the centre of each pixel of a block.

    The model grid is cylindrical, so every pixel in a column shares a
    longitude and every pixel in a row a latitude, and each pixel takes the
    mask pixel at its centre by indexing alone. Where there is no mask tile,
    such as the open ocean, or the tile is an empty file, there is no land.
    """
    height, width = block.shape
    transform = block.transform
    to_lonlat = pyproj.Transformer.from_crs(MODEL_CRS, "EPSG:4326", always_xy=True)
    x = transform.c + (np.arange(width) + 0.5) * transform.a
    y = transform.f + (np.arange(height) + 0.5) * transform.e
    lon, _ = to_lonlat.transform(x, np.zeros(width))
    _, lat = to_lonlat.transform(np.zeros(height), y)
    # The model grid is a little wider than the world: its last centres wrap.
    lon = (np.asarray(lon) + 180.0) % 360.0 - 180.0
    lat = np.asarray(lat)

    tile_col = (lon + 180.0) / 360.0 * WATER_TILES
    tile_x = np.floor(tile_col).astype(np.int64)
    pixel_col = np.floor((tile_col - tile_x) * WATER_TILE_PIXELS)
    pixel_col = np.clip(pixel_col, 0, WATER_TILE_PIXELS - 1).astype(np.int64)

    # Rows counted from the north. Beyond Web Mercator's ±85.05° there are none.
    tile_row = (1 - np.arcsinh(np.tan(np.radians(lat))) / math.pi) / 2 * WATER_TILES
    in_tiles = (tile_row >= 0) & (tile_row < WATER_TILES)
    top = np.floor(np.where(in_tiles, tile_row, 0)).astype(np.int64)
    north = tile_edge_latitude(top.astype(np.float64))
    south = tile_edge_latitude((top + 1).astype(np.float64))
    pixel_row = np.floor((north - lat) / (north - south) * WATER_TILE_PIXELS)
    pixel_row = np.clip(pixel_row, 0, WATER_TILE_PIXELS - 1).astype(np.int64)
    tile_y = np.where(in_tiles, WATER_TILES - 1 - top, -1)

    land = np.zeros((height, width), dtype=bool)
    # The tiles share one directory of almost 900,000 files, which GDAL would
    # otherwise list on every open, looking for sidecar files.
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        for rows in runs(tile_y):
            if tile_y[rows.start] < 0:
                continue
            for cols in runs(tile_x):
                path = water_tile_path(
                    root, int(tile_x[cols.start]), int(tile_y[rows.start])
                )
                # Two tiles on disk are empty files, and are read as missing.
                if not path.exists() or path.stat().st_size == 0:
                    continue
                with rasterio.open(path) as src:
                    water = src.read(1)
                land[rows, cols] = water[np.ix_(pixel_row[rows], pixel_col[cols])] == 0
    return land
