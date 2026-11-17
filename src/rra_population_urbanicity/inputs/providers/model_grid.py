"""Reading a block of a raster on the Populace model grid."""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import rasterio
from rasterio import windows
from rasterio.windows import Window

from rra_population_urbanicity.inputs.blocks import MODEL_CRS, Block

type Float32Array = npt.NDArray[np.float32]


def read_block(path: Path, block: Block, window: Window) -> Float32Array:
    """Read a block from a window of a raster on the model grid, with no data as 0.

    Raises if the window is not where the block is on the model grid.
    """
    # Skip GDAL's listing of the file's directory, which can hold thousands of
    # tiles, when it looks for sidecar files.
    with (
        rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"),
        rasterio.open(path) as src,
    ):
        if src.crs != MODEL_CRS or not windows.transform(
            window, src.transform
        ).almost_equals(block.transform):
            msg = f"{path} does not hold block {block.key} where expected."
            raise ValueError(msg)
        values: Float32Array = src.read(1, window=window).astype(np.float32, copy=False)
    if values.shape != block.shape:
        msg = f"{path} holds {values.shape} pixels of block {block.key}, not {block.shape}."
        raise ValueError(msg)
    np.copyto(values, np.float32(0.0), where=np.isnan(values))
    return values
