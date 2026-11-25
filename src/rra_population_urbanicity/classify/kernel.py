"""Run the kernel density method, one task per block of the 40 m model grid.

The classes are made and used at 40 m and not stored (principle P4). Each
block task classifies every time point and adds up the block's people by admin
unit and class, for each set of units; the reduce adds the blocks together.
The tabulations are those the joins write for a grid method, so the same
deliverable, cut and comparisons read them.

Each block is read with a margin as wide as the disc, so a pixel at its edge
averages over its neighbours in the next block. Land is GHS-LAND's share of
the 100 m cell holding each pixel's centre, read once per block.

A run writes, under classifications/<version>/:
- specification.yaml, scheme.yaml and manifest.yaml, as a grid method's run;
- tabulations/<shapes>/<time_point>.parquet: location_id, class_code and
  population, for each set of units in UNIT_SETS, with location 0 for people
  in no unit;
- .blocks/<block key>.parquet: one block's rows, removed after the reduce.
"""

import datetime
import math
import shutil
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import click
import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterio
import yaml
from affine import Affine
from rasterio.windows import Window
from rra_tools import jobmon
from rra_tools.shell_tools import mkdir

from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.classifiers.kernel_density import rules
from rra_population_urbanicity.classifiers.kernel_density.scheme import BINARY
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.grid import pixel_centres_in
from rra_population_urbanicity.inputs import blocks
from rra_population_urbanicity.inputs.aggregate import TASK_RESOURCES, blocks_for_bounds
from rra_population_urbanicity.inputs.blocks import MODEL_CRS, Block
from rra_population_urbanicity.inputs.providers import ghsl, populace
from rra_population_urbanicity.specification import (
    ClassificationSpecification,
    KernelDensityMethod,
    PopulaceSource,
)
from rra_population_urbanicity.tabulate.population import read_units

# The sets of units each run tabulates by.
UNIT_SETS = ("lsae_1578_a2", "raking_gbd_2023")
CONCURRENCY_LIMIT = 400
EARTH_RADIUS_M = 6_378_137.0
PIXEL_AREA_KM2 = (blocks.PIXEL_SIZE / 1000) ** 2
CODES = [c.code for c in BINARY.classes]
# Summed over units and classes, each time point must keep its pixels' people
# to within this share.
TOLERANCE = 1e-9

type FloatArray = npt.NDArray[np.float32]


def block_latitude(block: Block) -> float:
    """The latitude of a block's middle row, from y on the equal-area grid."""
    _, y = block.transform @ (0, block.shape[0] / 2)
    return math.degrees(math.asin(y / EARTH_RADIUS_M))


def margin_window(block: Block, kernel: FloatArray) -> Window:
    """The block's window on the model grid, widened by half the kernel."""
    rows, cols = kernel.shape[0] // 2, kernel.shape[1] // 2
    window = block.window
    return Window(
        window.col_off - cols,
        window.row_off - rows,
        window.width + 2 * cols,
        window.height + 2 * rows,
    )


def read_population(
    source: PopulaceSource, time_point: str, window: Window, root: Path
) -> tuple[FloatArray, npt.NDArray[np.bool_]]:
    """People in each pixel of a window of the model grid, and where there is data.

    Pixels beyond the grid's edges, and those with no prediction, have no
    data and no people.
    """
    path = populace.vrt_path(root, source, time_point)
    with (
        rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"),
        rasterio.open(path) as src,
    ):
        values = src.read(1, window=window, boundless=True, fill_value=np.nan)
    people = values.astype(np.float32, copy=False)
    has_data = ~np.isnan(people)
    people[~has_data] = 0.0
    return people, has_data


def read_land(data: UrbanicityData, window: Window) -> FloatArray:
    """GHS-LAND's land share at each pixel of a window of the model grid."""
    transform = blocks.MODEL_TRANSFORM @ Affine.translation(
        window.col_off, window.row_off
    )
    shape = (int(window.height), int(window.width))
    x, y = pixel_centres_in(transform, shape, MODEL_CRS, puc.CRSES["mollweide"])
    return ghsl.sample_land_100m(data, x, y)


def tabulate(
    units: npt.NDArray[np.int64], codes: npt.NDArray[np.uint8], people: FloatArray
) -> pd.DataFrame:
    """People by (unit, class) among a block's pixels."""
    has_people = people > 0
    ids, where = np.unique(units[has_people], return_inverse=True)
    index = np.full(256, -1, dtype=np.int64)
    index[CODES] = np.arange(len(CODES))
    keys = where * len(CODES) + index[codes[has_people]]
    totals = np.bincount(
        keys, weights=people[has_people].astype(np.float64), minlength=ids.size * len(CODES)
    )
    table = pd.DataFrame(
        {
            "location_id": np.repeat(ids, len(CODES)),
            "class_code": np.tile(CODES, ids.size),
            "population": totals,
        }
    )
    return table[table["population"] > 0]


def run_block(data: UrbanicityData, version: str, block_key: str) -> Path:
    """Classify a block at every time point and tabulate it by unit and class."""
    root = data.classification_root(version)
    spec = ClassificationSpecification.from_yaml(root / "specification.yaml")
    if not isinstance(spec.method, KernelDensityMethod) or not isinstance(
        spec.population, PopulaceSource
    ):
        msg = f"{version} is not a kernel density run on Populace."
        raise TypeError(msg)
    block = Block.from_key(block_key)
    params = spec.method.parameters
    kernel = rules.disc_kernel(params.radius_m, blocks.PIXEL_SIZE, block_latitude(block))
    window = margin_window(block, kernel)
    inner = (
        slice(kernel.shape[0] // 2, kernel.shape[0] // 2 + block.shape[0]),
        slice(kernel.shape[1] // 2, kernel.shape[1] // 2 + block.shape[1]),
    )
    land: FloatArray | None = None
    units: dict[str, npt.NDArray[np.int64]] = {}
    tables = []
    totals = {}
    for time_point in spec.time_points:
        people, has_data = read_population(
            spec.population, time_point, window, puc.POPULATION_MODEL_ROOT
        )
        block_people = people[inner]
        totals[time_point] = float(block_people.sum(dtype=np.float64))
        if totals[time_point] == 0:
            continue
        if land is None:
            land = read_land(data, window)
            units = {shapes: read_units(data, shapes, block) for shapes in UNIT_SETS}
        codes, _ = rules.classify(
            people, land, has_data, kernel, params, PIXEL_AREA_KM2
        )
        for shapes, block_units in units.items():
            table = tabulate(block_units, codes[inner], block_people)
            tables.append(table.assign(shapes=shapes, time_point=time_point))
    columns = ["shapes", "time_point", "location_id", "class_code", "population"]
    result = (
        pd.concat(tables)[columns] if tables else pd.DataFrame(columns=columns)
    ).astype({"location_id": np.int64, "class_code": np.int64, "population": float})
    path = block_path(root, block_key)
    mkdir(path.parent, parents=True, exist_ok=True)
    result.to_parquet(path, index=False)
    (path.with_suffix(".yaml")).write_text(yaml.safe_dump(totals))
    return path


def block_path(root: Path, block_key: str) -> Path:
    return root / ".blocks" / f"{block_key}.parquet"


def reduce(root: Path, block_keys: list[str]) -> dict[str, Any]:
    """Add the blocks up into the tabulations, and check they keep everyone."""
    spec = ClassificationSpecification.from_yaml(root / "specification.yaml")
    rows = pd.concat([pd.read_parquet(block_path(root, key)) for key in block_keys])
    pixels = {tp: 0.0 for tp in spec.time_points}
    for key in block_keys:
        for time_point, total in yaml.safe_load(
            block_path(root, key).with_suffix(".yaml").read_text()
        ).items():
            pixels[time_point] += total
    checks: dict[str, Any] = {}
    for (shapes, time_point), group in rows.groupby(["shapes", "time_point"]):
        table = (
            group.groupby(["location_id", "class_code"], as_index=False)["population"]
            .sum()
            .sort_values(["location_id", "class_code"])
        )
        total = float(table["population"].sum())
        if abs(total - pixels[time_point]) > TOLERANCE * max(pixels[time_point], 1.0):
            msg = (
                f"The {shapes} tabulation at {time_point} holds {total:,.0f} people "
                f"and the pixels {pixels[time_point]:,.0f}."
            )
            raise ValueError(msg)
        path = root / "tabulations" / str(shapes) / f"{time_point}.parquet"
        mkdir(path.parent, parents=True, exist_ok=True)
        table.to_parquet(path, index=False)
        urban = float(table.loc[table["class_code"] == BINARY.code("urban"), "population"].sum())
        checks.setdefault(str(shapes), {})[str(time_point)] = {
            "people": total,
            "people_in_no_unit": float(
                table.loc[table["location_id"] == 0, "population"].sum()
            ),
            "urban_share": urban / total if total else None,
        }
    shutil.rmtree(root / ".blocks")
    return checks


def run(
    spec: ClassificationSpecification,
    data: UrbanicityData,
    version: str,
    region: str | None,
) -> None:
    """Run a kernel density specification as `version`, already reserved."""
    root = data.classification_root(version)
    spec.model_copy(update={"version": version}).to_yaml(root / "specification.yaml")
    BINARY.to_yaml(root / "scheme.yaml")
    selected = (
        blocks.all_blocks()
        if region is None
        else blocks_for_bounds(*puc.REGIONS[region])
    )
    keys = [block.key for block in selected]
    if region is None:
        jobmon.run_parallel(
            runner="putask classify",
            task_name="kernel-block",
            node_args={"block-key": keys},
            task_args={"version": version, "output-dir": str(data.root)},
            task_resources=TASK_RESOURCES,
            concurrency_limit=CONCURRENCY_LIMIT,
            max_attempts=1,
            log_root=data.root / "logs" / "classify",
        )
    else:
        for key in keys:
            run_block(data, version, key)
    checks = reduce(root, keys)
    manifest = {
        "version": version,
        "created": datetime.datetime.now(tz=datetime.UTC).isoformat(timespec="seconds"),
        "package_version": package_version("rra-population-urbanicity"),
        "region": None
        if region is None
        else {"name": region, "bounds": list(puc.REGIONS[region]), "blocks": keys},
        "outputs": {"tabulations": "tabulations/<shapes>/<time_point>.parquet"},
        "tabulations": checks,
    }
    (root / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))


@click.command()
@click.option("--version", required=True)
@click.option("--block-key", required=True)
@click.option(
    "--output-dir", required=True, type=click.Path(path_type=Path, file_okay=False)
)
def kernel_block_task(version: str, block_key: str, output_dir: Path) -> None:
    """Classify one block with the kernel method and tabulate it."""
    run_block(UrbanicityData(output_dir), version, block_key)
