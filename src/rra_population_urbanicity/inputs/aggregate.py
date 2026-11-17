"""Aggregate the inputs on the 40 m model grid onto a classification grid.

Each 40 m pixel goes to the grid cell containing its centre, through the same
transform the tabulation and the pixel view use (`grid.pixel_to_cell`).
Population is summed. Built-up and land are averaged over the pixels a cell
is given, which is their share of the cell, since both grids are equal-area.

The work is done in two parts:
- One task per block of the model grid computes every layer and time point,
  since finding each pixel's cell is the costly part and is shared. It writes
  the block's sums for the cells it touches to a work directory.
- The reduce adds the blocks into
  grids/<grid>/<layer>/<source>/<version...>/<time point>.tif, checks that
  population is conserved, records the check in a manifest in each version
  directory, and removes the work directory.

The classification grid stops short of the poles (`moll_1km` at 89.09°), so
pixels beyond it are left out, and the people in them are recorded.
"""

import datetime
import shutil
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Annotated, Any

import click
import numpy as np
import numpy.typing as npt
import pyproj
import rasterio
import yaml
from affine import Affine
from pydantic import BaseModel, ConfigDict, Field
from rra_tools import jobmon
from rra_tools.cli_tools import with_output_directory
from rra_tools.shell_tools import mkdir

from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.data import Layer, UrbanicityData
from rra_population_urbanicity.grid import GRIDS, GridDefinition, pixel_centres_in
from rra_population_urbanicity.inputs import blocks
from rra_population_urbanicity.inputs.blocks import MODEL_CRS, Block
from rra_population_urbanicity.inputs.providers import ghsl, microsoft, populace
from rra_population_urbanicity.specification import (
    ClassificationSpecification,
    GhsLandOrMicrosoftWaterSource,
    MicrosoftDensitySource,
    MicrosoftWaterSource,
    PopulaceSource,
)

type Arrays = dict[str, npt.NDArray[Any]]

# Population is conserved if the grid's total and the people beyond the grid
# add up to the pixels' total to within this share.
CONSERVATION_TOLERANCE = 1e-6
# Tiles of the output rasters, in cells.
BLOCK_SIZE = 512

ModelGridSource = Annotated[
    PopulaceSource
    | MicrosoftDensitySource
    | MicrosoftWaterSource
    | GhsLandOrMicrosoftWaterSource,
    Field(discriminator="source"),
]


class GridLayer(BaseModel):
    """One layer of one source at one time point, aggregated from the model grid."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    layer: Layer
    source: ModelGridSource
    time_point: str

    @property
    def key(self) -> str:
        return "/".join(
            [
                self.layer,
                self.source.source,
                *self.source.version_components,
                self.time_point,
            ]
        )

    @property
    def summed(self) -> bool:
        return self.layer == "population"


class Plan(BaseModel):
    """What an aggregation run computes, saved in its work directory for the tasks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    output_dir: str
    grid: str
    layers: list[GridLayer]
    blocks: list[str]
    region: str | None


def grid_layers(spec: ClassificationSpecification) -> list[GridLayer]:
    """The layers of a specification that come from the model grid."""
    layers = []
    if isinstance(spec.population, PopulaceSource):
        layers += [
            GridLayer(layer="population", source=spec.population, time_point=tp)
            for tp in spec.time_points
        ]
    if isinstance(spec.built, MicrosoftDensitySource):
        layers += [
            GridLayer(layer="built", source=spec.built, time_point=tp)
            for tp in spec.time_points
        ]
    if isinstance(spec.land, MicrosoftWaterSource | GhsLandOrMicrosoftWaterSource):
        layers.append(GridLayer(layer="land", source=spec.land, time_point=puc.STATIC))
    return layers


def blocks_for_bounds(
    west: float, south: float, east: float, north: float
) -> list[Block]:
    """The blocks of the model grid that cover a longitude/latitude box."""
    to_model = pyproj.Transformer.from_crs("EPSG:4326", MODEL_CRS, always_xy=True)
    x0, y0, x1, y1 = to_model.transform_bounds(west, south, east, north)
    size = blocks.BLOCK_PIXELS * blocks.PIXEL_SIZE
    first_x = max(int((x0 - blocks.X_MIN) // size), 0)
    last_x = min(int((x1 - blocks.X_MIN) // size), blocks.N_BLOCK_COLS - 1)
    first_y = max(int((blocks.Y_MAX - y1) // size), 0)
    last_y = min(int((blocks.Y_MAX - y0) // size), blocks.N_BLOCK_ROWS - 1)
    return [
        Block(x, y)
        for y in range(first_y, last_y + 1)
        for x in range(first_x, last_x + 1)
    ]


def rows_on_grid(block: Block, grid: GridDefinition) -> slice:
    """The rows of a block whose pixel centres are on the grid.

    Every pixel in a row of the model grid has the same latitude, and so, in
    a pseudo-cylindrical projection such as Mollweide, the same y.
    """
    _, y = pixel_centres_in(block.transform, (block.shape[0], 1), MODEL_CRS, grid.crs)
    y_min = grid.y_max - grid.n_rows * grid.cell_size
    on_grid = np.flatnonzero((y[:, 0] <= grid.y_max) & (y[:, 0] > y_min))
    return slice(int(on_grid[0]), int(on_grid[-1]) + 1)


def aggregate_block(
    block: Block, layers: list[GridLayer], grid: GridDefinition, data: UrbanicityData
) -> Arrays:
    """A block's sums of every layer over the grid cells its pixels fall in.

    Returns the flat index of each cell kept, the number of the block's
    pixels in it, and for each layer `sum:<key>`. A cell is kept if any layer
    is not zero there, or if it may also hold pixels of another block, so the
    reduce can count every cell's pixels. For population, `total:<key>` and
    `beyond:<key>` hold the people in the whole block and beyond the grid.
    """
    rows = rows_on_grid(block, grid)
    height, width = rows.stop - rows.start, block.shape[1]
    transform = block.transform @ Affine.translation(0, rows.start)
    x, y = pixel_centres_in(transform, (height, width), MODEL_CRS, grid.crs)
    cell_row, cell_col = grid.cell_index(x, y)
    if any(isinstance(layer.source, GhsLandOrMicrosoftWaterSource) for layer in layers):
        ghs_land = ghsl.sample_land_100m(data, x, y)
    del x, y
    # Number the cells the block's pixels fall in 0, 1, ..., in grid order.
    first_row = int(cell_row.min())
    span = ((cell_row - first_row).astype(np.int64) * grid.n_cols + cell_col).ravel()
    del cell_row, cell_col
    occupied = np.flatnonzero(np.bincount(span))
    number = np.zeros(int(occupied[-1]) + 1, dtype=np.int32)
    number[occupied] = np.arange(occupied.size, dtype=np.int32)
    local = number[span]
    del span, number
    size = occupied.size
    pixels = np.bincount(local, minlength=size)

    # Cells on the block's edges may hold pixels of its neighbours too.
    keep = np.zeros(size, dtype=bool)
    edges = local.reshape(height, width)
    for edge in (edges[0], edges[-1], edges[:, 0], edges[:, -1]):
        keep[edge] = True

    result: Arrays = {}
    water_mask: npt.NDArray[np.bool_] | None = None
    sums: Arrays = {}
    for layer in layers:
        source = layer.source
        values: npt.NDArray[Any]
        if isinstance(source, PopulaceSource):
            people = populace.load_block(
                source, block, layer.time_point, root=puc.POPULATION_MODEL_ROOT
            )
            total = people.sum(dtype=np.float64)
            values = people[rows]
            result[f"total:{layer.key}"] = np.array(total)
            result[f"beyond:{layer.key}"] = np.array(
                total - values.sum(dtype=np.float64)
            )
        elif isinstance(source, MicrosoftDensitySource):
            values = microsoft.load_density_block(
                source, block, layer.time_point, root=puc.BUILDING_DENSITY_ROOT
            )[rows]
        else:
            if water_mask is None:
                water_mask = microsoft.load_land_block(
                    MicrosoftWaterSource(
                        source="microsoft_water",
                        building_version=source.building_version,
                    ),
                    block,
                    root=puc.BUILDING_DENSITY_ROOT,
                )[rows]
            values = water_mask
            if isinstance(source, GhsLandOrMicrosoftWaterSource):
                values = np.maximum(water_mask, ghs_land.reshape(height, width))
        sums[layer.key] = np.bincount(local, weights=values.ravel(), minlength=size)
        keep |= sums[layer.key] != 0

    cells = np.flatnonzero(keep)
    result["cells"] = (occupied[cells] + first_row * grid.n_cols).astype(np.int32)
    result["pixels"] = pixels[cells].astype(np.uint16)
    for layer in layers:
        dtype = np.float64 if layer.summed else np.float32
        result[f"sum:{layer.key}"] = sums[layer.key][cells].astype(dtype)
    return result


def block_path(work_dir: Path, key: str) -> Path:
    return work_dir / "blocks" / f"{key}.npz"


def read_plan(work_dir: Path) -> Plan:
    return Plan.model_validate(yaml.safe_load((work_dir / "plan.yaml").read_text()))


def run_block(work_dir: Path, key: str) -> None:
    """Aggregate one block of a plan and save it, under a temporary name until complete."""
    plan = read_plan(work_dir)
    arrays = aggregate_block(
        Block.from_key(key),
        plan.layers,
        GRIDS[plan.grid],
        UrbanicityData(plan.output_dir),
    )
    path = block_path(work_dir, key)
    partial = path.with_name(f"{key}.part.npz")
    np.savez_compressed(partial, allow_pickle=False, **arrays)
    partial.rename(path)


def raster_profile(grid: GridDefinition) -> dict[str, Any]:
    return {
        "driver": "GTiff",
        "height": grid.n_rows,
        "width": grid.n_cols,
        "count": 1,
        "dtype": "float32",
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": None,
        "tiled": True,
        "blockxsize": BLOCK_SIZE,
        "blockysize": BLOCK_SIZE,
        "compress": "deflate",
        "predictor": 3,
    }


def update_manifest(path: Path, layer: GridLayer, entry: dict[str, Any]) -> None:
    """Record a time point in its version directory's manifest, keeping the others."""
    manifest: dict[str, Any] = (
        yaml.safe_load(path.read_text())
        if path.exists()
        else {
            "layer": layer.layer,
            "source": layer.source.model_dump(mode="json"),
            "aggregation": "summed"
            if layer.summed
            else "averaged over the 40 m pixels whose centres are in the cell",
            "time_points": {},
        }
    )
    manifest["time_points"][layer.time_point] = entry
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))


def reduce(work_dir: Path) -> None:
    """Add a plan's blocks into its grids, check them, and write the manifests."""
    plan = read_plan(work_dir)
    grid = GRIDS[plan.grid]
    data = UrbanicityData(plan.output_dir)
    paths = [block_path(work_dir, key) for key in plan.blocks]
    if missing := [path.stem for path in paths if not path.exists()]:
        msg = f"{len(missing)} blocks have not been aggregated: {missing[:5]}..."
        raise RuntimeError(msg)

    pixels = np.zeros(grid.n_rows * grid.n_cols, dtype=np.uint16)
    for path in paths:
        with np.load(path) as block:
            pixels[block["cells"]] += block["pixels"]
    for layer in plan.layers:
        sums = np.zeros(grid.n_rows * grid.n_cols)
        total = beyond = 0.0
        for path in paths:
            with np.load(path) as block:
                sums[block["cells"]] += block[f"sum:{layer.key}"]
                if layer.summed:
                    total += float(block[f"total:{layer.key}"])
                    beyond += float(block[f"beyond:{layer.key}"])
        entry: dict[str, Any] = {
            "created": datetime.datetime.now(tz=datetime.UTC).isoformat(
                timespec="seconds"
            ),
            "package_version": package_version("rra-population-urbanicity"),
            "blocks": len(paths),
            "region": plan.region,
        }
        if layer.summed:
            on_grid = float(sums.sum())
            error = abs(on_grid + beyond - total) / max(total, 1.0)
            entry["conservation"] = {
                "pixels": total,
                "grid": on_grid,
                "beyond_grid": beyond,
                "relative_error": error,
            }
            if error > CONSERVATION_TOLERANCE:
                msg = f"{layer.key} is not conserved: relative error {error:.2e}."
                raise RuntimeError(msg)
            values = sums
        else:
            values = np.divide(sums, pixels, out=np.zeros_like(sums), where=pixels > 0)
        if isinstance(layer.source, MicrosoftDensitySource):
            entry["quarter_read"] = microsoft.density_time_point(layer.time_point)

        path = data.grid_path(grid.name, layer.layer, layer.source, layer.time_point)
        mkdir(path.parent, parents=True, exist_ok=True)
        with rasterio.open(path, "w", **raster_profile(grid)) as dst:
            dst.write(values.reshape(grid.shape).astype(np.float32), 1)
        update_manifest(path.parent / "manifest.yaml", layer, entry)
    shutil.rmtree(work_dir)


# Profiled on six blocks (dense, high latitude, polar, ocean, cut short): at
# most 133 s when reading cold, and at most 4.7 GB.
TASK_RESOURCES: dict[str, str | int] = {
    "queue": "all.q",
    "project": "proj_rapidresponse",
    "cores": 1,
    "memory": "8G",
    "runtime": "30m",
}
# Block tasks running at once, to spare the shared file system they read from.
CONCURRENCY_LIMIT = 200


def run(
    specs: list[ClassificationSpecification],
    data: UrbanicityData,
    region: str | None,
) -> Path | None:
    """Aggregate every model-grid layer of the specifications that is not on disk.

    Returns the work directory the blocks were aggregated in, or None if
    there was nothing to do.
    """
    grids = {spec.grid.name for spec in specs}
    if len(grids) != 1:
        msg = f"The specifications must share one grid, not {sorted(grids)}."
        raise ValueError(msg)
    grid = specs[0].grid
    layers = list(dict.fromkeys(layer for spec in specs for layer in grid_layers(spec)))
    layers = [
        layer
        for layer in layers
        if not data.grid_path(
            grid.name, layer.layer, layer.source, layer.time_point
        ).exists()
    ]
    if not layers:
        return None
    selected = (
        blocks.all_blocks()
        if region is None
        else blocks_for_bounds(*puc.REGIONS[region])
    )
    stamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y_%m_%d_%H%M%S")
    work_dir = data.root / "grids" / grid.name / ".aggregate" / stamp
    mkdir(work_dir / "blocks", parents=True, exist_ok=True)
    plan = Plan(
        output_dir=str(data.root),
        grid=grid.name,
        layers=layers,
        blocks=[block.key for block in selected],
        region=region,
    )
    (work_dir / "plan.yaml").write_text(
        yaml.safe_dump(plan.model_dump(mode="json"), sort_keys=False)
    )
    jobmon.run_parallel(
        runner="putask inputs",
        task_name="aggregate-block",
        node_args={"block-key": plan.blocks},
        task_args={"work-dir": str(work_dir)},
        task_resources=TASK_RESOURCES,
        concurrency_limit=CONCURRENCY_LIMIT,
        max_attempts=1,
        log_root=data.root / "logs" / "aggregate",
    )
    reduce(work_dir)
    return work_dir


@click.command()
@click.argument(
    "specifications",
    nargs=-1,
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--region",
    type=click.Choice(sorted(puc.REGIONS)),
    help="Aggregate only the blocks covering this region, for development.",
)
@with_output_directory(puc.URBANICITY_ROOT)
def aggregate_run(
    specifications: tuple[Path, ...], region: str | None, output_dir: str
) -> None:
    """Aggregate the model-grid inputs of classification specifications."""
    if region is not None and Path(output_dir) == puc.URBANICITY_ROOT:
        # The grids are shared by every classification that uses them, so a
        # grid covering only a region must not be written where they live.
        msg = "A run limited to a region needs its own --output-dir."
        raise click.UsageError(msg)
    specs = [ClassificationSpecification.from_yaml(path) for path in specifications]
    work_dir = run(specs, UrbanicityData(output_dir), region)
    print("Nothing to aggregate." if work_dir is None else "Done.")


@click.command()
@click.option("--work-dir", required=True, type=click.Path(path_type=Path))
@click.option("--block-key", required=True)
def aggregate_block_task(work_dir: Path, block_key: str) -> None:
    """Aggregate one block of the model grid."""
    run_block(work_dir, block_key)
