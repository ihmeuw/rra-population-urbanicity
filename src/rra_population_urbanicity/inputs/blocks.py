"""The blocks of the Populace model grid, the unit of work for aggregation.

The model grid is World Cylindrical Equal Area at 40 m. Its compiled
predictions form a raster of 1,001,876 x 318,194 pixels whose top-left corner
is (-20,037,508.34, 6,363,885.33). The raster is cut into blocks of 8,192
pixels on a side (16 tiles of 512), stored in group tiles of 10 x 10 blocks.
Blocks in the last column and the last row are cut short where the raster
ends, as are their files.
"""

import math
import re
from dataclasses import dataclass

from affine import Affine
from rasterio.windows import Window

from rra_population_urbanicity import constants as puc

MODEL_CRS = puc.CRSES["equal_area"]
PIXEL_SIZE = 40.0
X_MIN = -20_037_508.34
Y_MAX = 6_363_885.33
WIDTH = 1_001_876
HEIGHT = 318_194
BLOCK_PIXELS = 8192
GROUP_BLOCKS = 10
N_BLOCK_COLS = math.ceil(WIDTH / BLOCK_PIXELS)
N_BLOCK_ROWS = math.ceil(HEIGHT / BLOCK_PIXELS)

MODEL_TRANSFORM = Affine(PIXEL_SIZE, 0.0, X_MIN, 0.0, -PIXEL_SIZE, Y_MAX)

BLOCK_KEY_PATTERN = re.compile(r"^B-(\d{4})X-(\d{4})Y$")


@dataclass(frozen=True)
class Block:
    """A block of the model grid, by its column (x) and row (y) of blocks."""

    x: int
    y: int

    def __post_init__(self) -> None:
        if not (0 <= self.x < N_BLOCK_COLS and 0 <= self.y < N_BLOCK_ROWS):
            msg = f"There is no block at x={self.x}, y={self.y}."
            raise ValueError(msg)

    @classmethod
    def from_key(cls, key: str) -> "Block":
        match = BLOCK_KEY_PATTERN.match(key)
        if match is None:
            msg = f"{key} is not a block key like B-0091X-0011Y."
            raise ValueError(msg)
        return cls(int(match[1]), int(match[2]))

    @property
    def key(self) -> str:
        return f"B-{self.x:04d}X-{self.y:04d}Y"

    @property
    def window(self) -> Window:
        """The block's pixels in the model raster."""
        col, row = self.x * BLOCK_PIXELS, self.y * BLOCK_PIXELS
        width = min(BLOCK_PIXELS, WIDTH - col)
        height = min(BLOCK_PIXELS, HEIGHT - row)
        return Window(col, row, width, height)

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.window.height), int(self.window.width)

    @property
    def transform(self) -> Affine:
        return MODEL_TRANSFORM @ Affine.translation(
            self.x * BLOCK_PIXELS, self.y * BLOCK_PIXELS
        )

    @property
    def group_key(self) -> str:
        return f"G-{self.x // GROUP_BLOCKS:04d}X-{self.y // GROUP_BLOCKS:04d}Y"

    @property
    def group_window(self) -> Window:
        """The block's pixels in its group tile."""
        height, width = self.shape
        return Window(
            (self.x % GROUP_BLOCKS) * BLOCK_PIXELS,
            (self.y % GROUP_BLOCKS) * BLOCK_PIXELS,
            width,
            height,
        )


def all_blocks() -> list[Block]:
    return [Block(x, y) for y in range(N_BLOCK_ROWS) for x in range(N_BLOCK_COLS)]
