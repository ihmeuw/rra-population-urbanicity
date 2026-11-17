import pytest
from affine import Affine
from rasterio.windows import Window

from rra_population_urbanicity.inputs.blocks import Block, all_blocks


def test_a_block_is_named_by_its_column_and_row() -> None:
    block = Block.from_key("B-0091X-0011Y")
    assert (block.x, block.y) == (91, 11)
    assert block.key == "B-0091X-0011Y"
    # As in the block's file of building density.
    assert block.transform.almost_equals(
        Affine(40.0, 0.0, 9_781_371.66, 0.0, -40.0, 2_759_405.33)
    )
    assert block.shape == (8192, 8192)
    assert block.group_key == "G-0009X-0001Y"
    assert block.group_window == Window(8192, 8192, 8192, 8192)


def test_the_last_blocks_are_cut_short() -> None:
    block = Block.from_key("B-0122X-0038Y")
    assert block.shape == (6898, 2452)
    assert block.window == Window(999_424, 311_296, 2452, 6898)
    assert block.group_key == "G-0012X-0003Y"
    assert block.group_window == Window(16_384, 65_536, 2452, 6898)


def test_the_blocks_cover_the_model_grid() -> None:
    blocks = all_blocks()
    assert len(blocks) == 123 * 39
    assert sum(b.shape[0] * b.shape[1] for b in blocks) == 1_001_876 * 318_194


@pytest.mark.parametrize("key", ["B-91X-11Y", "G-0009X-0001Y"])
def test_block_keys_are_checked(key: str) -> None:
    with pytest.raises(ValueError, match="not a block key"):
        Block.from_key(key)


def test_blocks_outside_the_grid_are_rejected() -> None:
    with pytest.raises(ValueError, match="no block at x=123"):
        Block(123, 0)
