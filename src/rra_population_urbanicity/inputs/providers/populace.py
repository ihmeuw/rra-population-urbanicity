"""The Populace surface: the compiled predictions of rra-population-model.

They are read by path, not through rra-population-model, which would bring in
its modelling dependencies. Pixels with no prediction hold no people.
"""

from pathlib import Path

from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.inputs.blocks import Block
from rra_population_urbanicity.inputs.providers.model_grid import (
    Float32Array,
    read_block,
)
from rra_population_urbanicity.specification import PopulaceSource


def compiled_path(
    root: Path, source: PopulaceSource, time_point: str, group_key: str
) -> Path:
    return (
        root
        / "modeling"
        / f"{source.resolution}m"
        / "models"
        / source.model_version
        / "compiled_predictions"
        / "population"
        / time_point
        / f"{group_key}.tif"
    )


def load_block(
    source: PopulaceSource,
    block: Block,
    time_point: str,
    root: Path = puc.POPULATION_MODEL_ROOT,
) -> Float32Array:
    """The people in each pixel of a block."""
    path = compiled_path(root, source, time_point, block.group_key)
    return read_block(path, block, block.group_window)
