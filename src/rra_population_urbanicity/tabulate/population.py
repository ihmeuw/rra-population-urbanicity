"""People by 1 km cell and admin unit, tabulated once for a surface and shapes version.

For each time point, the table holds the people in each pair of a grid cell
and an admin unit: `row`, `col`, `location_id`, `population`. A cell that a
boundary crosses has one row for each unit, with that unit's own people.
People outside every shape are kept under location 0. Tabulations by class,
the units of the cluster table and cuts are all joins against this table, so
the pass over the 40 m surface is made once, not once per classification.

As in aggregation, one task per block of the model grid handles every time
point, since finding each pixel's cell is the costly part, and the reduce adds
up the rows that blocks share. The reduce checks the table three ways:
- summed over units, it equals the population grid, cell by cell;
- summed over cells, it equals the block's pixels, unit by unit;
- each country's total is compared with the population Populace is raked to,
  which is reported and not checked, since the shapes may differ from the ones
  Populace is raked to at borders and coasts.
"""

import datetime
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
from pydantic import BaseModel, ConfigDict
from rasterio import windows
from rra_tools import jobmon
from rra_tools.cli_tools import with_output_directory
from rra_tools.shell_tools import mkdir

from rra_population_urbanicity import admin
from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.grid import GRIDS, GridDefinition, pixel_centres_in
from rra_population_urbanicity.inputs import blocks
from rra_population_urbanicity.inputs.aggregate import (
    TASK_RESOURCES,
    blocks_for_bounds,
    rows_on_grid,
)
from rra_population_urbanicity.inputs.blocks import MODEL_CRS, Block
from rra_population_urbanicity.inputs.providers import populace
from rra_population_urbanicity.specification import (
    ClassificationSpecification,
    PopulaceSource,
)
from rra_population_urbanicity.tabulate import admin_rasters

type Arrays = dict[str, npt.NDArray[Any]]

# A cell and a unit make one key: the cell's flat index, shifted past the
# largest location_id. LSAE 1578's largest is 97,860.
LOCATION_BITS = 17
LOCATION_MASK = (1 << LOCATION_BITS) - 1
# The table summed over units must equal the grid in every cell to within
# this share of the cell's people (or of one person, in a near-empty cell).
GRID_TOLERANCE = 1e-6
# Summed over cells, each unit must keep its pixels' people to within this share.
UNIT_TOLERANCE = 1e-9
CONCURRENCY_LIMIT = 200
# The units Populace is raked to, whose populations the countries are compared with.
RAKING_SHAPES = "raking_gbd_2023"
# How close countries come to GBD's population, for the report.
GBD_BANDS = (0.001, 0.01)


class Plan(BaseModel):
    """What a tabulation run computes, saved in its work directory for the tasks."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    output_dir: str
    grid: str
    source: PopulaceSource
    shapes: str
    time_points: list[str]
    blocks: list[str]
    region: str | None


def read_plan(work_dir: Path) -> Plan:
    return Plan.model_validate(yaml.safe_load((work_dir / "plan.yaml").read_text()))


def block_path(work_dir: Path, key: str) -> Path:
    return work_dir / "blocks" / f"{key}.npz"


def read_units(
    data: UrbanicityData, shapes: str, block: Block
) -> npt.NDArray[np.int64]:
    path = data.admin_block_path(shapes, block.key)
    with rasterio.open(path) as src:
        if not src.transform.almost_equals(block.transform) or src.shape != block.shape:
            msg = f"{path} does not hold block {block.key} where expected."
            raise ValueError(msg)
        units: npt.NDArray[np.int64] = src.read(1).astype(np.int64)
    if units.max(initial=0) > LOCATION_MASK:
        msg = f"{path} has a location_id over {LOCATION_MASK}, too large for the table's keys."
        raise ValueError(msg)
    return units


def tabulate_block(block: Block, plan: Plan, data: UrbanicityData) -> Arrays:
    """A block's people by (cell, unit) key, for every time point.

    Returns the sorted keys of the pairs its pixels fall in, and for each time
    point `population:<tp>`, the people in each pair. Also, for the checks:
    `units:<tp>` and `unit_people:<tp>`, the people in each unit among the
    pixels on the grid; and `total:<tp>` and `beyond:<tp>`, the people in the
    whole block and beyond the grid.
    """
    grid = GRIDS[plan.grid]
    rows = rows_on_grid(block, grid)
    height, width = rows.stop - rows.start, block.shape[1]
    transform = block.transform @ Affine.translation(0, rows.start)
    x, y = pixel_centres_in(transform, (height, width), MODEL_CRS, grid.crs)
    cell_row, cell_col = grid.cell_index(x, y)
    del x, y
    units = read_units(data, plan.shapes, block)[rows].ravel()
    cells = cell_row.astype(np.int64).ravel() * grid.n_cols + cell_col.ravel()
    del cell_row, cell_col
    keys, pair = np.unique((cells << LOCATION_BITS) | units, return_inverse=True)
    del cells
    unit_ids = np.flatnonzero(np.bincount(units))

    result: Arrays = {"keys": keys}
    for tp in plan.time_points:
        people = populace.load_block(
            plan.source, block, tp, root=puc.POPULATION_MODEL_ROOT
        )
        total = people.sum(dtype=np.float64)
        on_grid = people[rows].ravel()
        result[f"population:{tp}"] = np.bincount(
            pair, weights=on_grid, minlength=keys.size
        )
        result[f"units:{tp}"] = unit_ids
        result[f"unit_people:{tp}"] = np.bincount(units, weights=on_grid)[unit_ids]
        result[f"total:{tp}"] = np.array(total)
        result[f"beyond:{tp}"] = np.array(total - on_grid.sum(dtype=np.float64))
    return result


def run_block(work_dir: Path, key: str) -> None:
    """Tabulate one block of a plan and save it, under a temporary name until complete."""
    plan = read_plan(work_dir)
    arrays = tabulate_block(Block.from_key(key), plan, UrbanicityData(plan.output_dir))
    path = block_path(work_dir, key)
    partial = path.with_name(f"{key}.part.npz")
    np.savez_compressed(partial, allow_pickle=False, **arrays)
    partial.rename(path)


def gbd_population(year: int) -> pd.Series:
    """The population Populace is raked to for a year, by country.

    rra-population-model rakes to each of its units' populations. Past their
    last year, it projects each unit's last population forward by its growth
    over the same number of years before it. The units' populations are then
    added up to their countries. Places with no population, or none known, are
    left out.
    """
    version = admin.SHAPES[RAKING_SHAPES]
    population = pd.read_parquet(
        version.raking_population_path,
        columns=["location_id", "year_id", "population"],
    )
    by_year = population.set_index(["year_id", "location_id"])["population"]
    last = int(population["year_id"].max())
    if year <= last:
        units = by_year.loc[year]
    else:
        last_population = by_year.loc[last]
        earlier = by_year.loc[last - (year - last)]
        units = last_population * (last_population / earlier)
    country_of = admin.unit_parents(version).set_index("location_id")["country_id"]
    countries: pd.Series = units.groupby(country_of.reindex(units.index)).sum(
        min_count=1
    )
    return countries[countries > 0]


def check_grid(
    table: pd.DataFrame, grid: GridDefinition, grid_path: Path
) -> dict[str, float]:
    """Compare the table summed over units with the population grid, cell by cell."""
    cells = table["row"].to_numpy(np.int64) * grid.n_cols + table["col"].to_numpy()
    summed = np.bincount(
        cells,
        weights=table["population"].to_numpy(),
        minlength=grid.n_rows * grid.n_cols,
    )
    with rasterio.open(grid_path) as src:
        expected = src.read(1, window=windows.Window(0, 0, grid.n_cols, grid.n_rows))
    difference = np.abs(summed - expected.ravel())
    relative = difference / np.maximum(np.abs(expected.ravel()), 1.0)
    return {
        "max_difference": float(difference.max()),
        "max_relative": float(relative.max()),
    }


def compare_countries(
    by_unit: pd.Series, parents: pd.DataFrame, time_point: str
) -> dict[str, Any]:
    """Each country's people in the table against GBD's, for the report."""
    country_of = parents.set_index("location_id")["country_id"]
    countries = by_unit.groupby(country_of.reindex(by_unit.index).to_numpy()).sum()
    gbd = gbd_population(int(time_point[:4])).reindex(countries.index)
    ratio = (countries / gbd).dropna()
    gap = (ratio - 1).abs().sort_values(ascending=False)
    return {
        "countries": int(ratio.size),
        **{f"within_{band:.1%}": int((gap <= band).sum()) for band in GBD_BANDS},
        "furthest": {int(c): round(float(ratio[c]), 5) for c in gap.index[:10]},
    }


def reduce(work_dir: Path) -> Path:
    """Add a plan's blocks into its tables, check them, and write the manifest."""
    plan = read_plan(work_dir)
    grid = GRIDS[plan.grid]
    data = UrbanicityData(plan.output_dir)
    paths = [block_path(work_dir, key) for key in plan.blocks]
    if missing := [path.stem for path in paths if not path.exists()]:
        msg = f"{len(missing)} blocks have not been tabulated: {missing[:5]}..."
        raise RuntimeError(msg)
    parents = admin.unit_parents(admin.SHAPES[plan.shapes])
    root = data.population_by_admin_root(plan.grid, plan.source, plan.shapes)
    mkdir(root, parents=True, exist_ok=True)

    entries: dict[str, Any] = {}
    for tp in plan.time_points:
        keys, people, unit_ids, unit_people = [], [], [], []
        total = beyond = 0.0
        for path in paths:
            with np.load(path) as block:
                keys.append(block["keys"])
                people.append(block[f"population:{tp}"])
                unit_ids.append(block[f"units:{tp}"])
                unit_people.append(block[f"unit_people:{tp}"])
                total += float(block[f"total:{tp}"])
                beyond += float(block[f"beyond:{tp}"])
        pairs = pd.Series(np.concatenate(people)).groupby(np.concatenate(keys)).sum()
        pairs = pairs[pairs > 0]
        key = pairs.index.to_numpy(np.int64)
        cell = key >> LOCATION_BITS
        table = pd.DataFrame(
            {
                "row": (cell // grid.n_cols).astype(np.int32),
                "col": (cell % grid.n_cols).astype(np.int32),
                "location_id": (key & LOCATION_MASK).astype(np.int32),
                "population": pairs.to_numpy(),
            }
        )
        by_unit = table.groupby("location_id")["population"].sum()
        pixels = (
            pd.Series(np.concatenate(unit_people))
            .groupby(np.concatenate(unit_ids))
            .sum()
        )
        pixels = pixels[pixels > 0]
        unit_error = float(
            (
                (by_unit.reindex(pixels.index, fill_value=0.0) - pixels).abs() / pixels
            ).max()
        )
        grid_check = check_grid(
            table, grid, data.grid_path(grid.name, "population", plan.source, tp)
        )
        entries[tp] = {
            "rows": len(table),
            "cells": int(table[["row", "col"]].drop_duplicates().shape[0]),
            "people": float(table["population"].sum()),
            "people_in_no_unit": float(by_unit.get(admin.NO_UNIT, 0.0)),
            "people_beyond_grid": beyond,
            "pixels_total": total,
            "grid_check": grid_check,
            "worst_unit_relative_error": unit_error,
            "countries_against_gbd": compare_countries(
                by_unit.drop(admin.NO_UNIT, errors="ignore"), parents, tp
            ),
        }
        if grid_check["max_relative"] > GRID_TOLERANCE or unit_error > UNIT_TOLERANCE:
            msg = f"The {tp} table fails its checks: {grid_check}, units {unit_error:.2e}."
            raise RuntimeError(msg)
        table.to_parquet(root / f"{tp}.parquet", index=False)

    manifest_path = root / "manifest.yaml"
    manifest: dict[str, Any] = (
        yaml.safe_load(manifest_path.read_text()) if manifest_path.exists() else {}
    )
    manifest.update(
        {
            "source": plan.source.model_dump(mode="json"),
            "shapes": plan.shapes,
            "columns": "row, col, location_id, population; location 0 is no unit",
            "created": datetime.datetime.now(tz=datetime.UTC).isoformat(
                timespec="seconds"
            ),
            "package_version": package_version("rra-population-urbanicity"),
            "blocks": len(paths),
            "region": plan.region,
        }
    )
    manifest.setdefault("time_points", {}).update(entries)
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    shutil.rmtree(work_dir)
    return manifest_path


def run(
    spec: ClassificationSpecification,
    data: UrbanicityData,
    shapes: str,
    region: str | None,
) -> Path | None:
    """Tabulate a specification's population by cell and unit, for the time points not on disk.

    Returns the manifest written, or None if there was nothing to do.
    """
    if not isinstance(spec.population, PopulaceSource):
        msg = "Only a population surface on the model grid can be tabulated by unit."
        raise TypeError(msg)
    admin_rasters.check_unchanged(data, shapes)
    root = data.population_by_admin_root(spec.grid.name, spec.population, shapes)
    time_points = [
        tp for tp in spec.time_points if not (root / f"{tp}.parquet").exists()
    ]
    if not time_points:
        return None
    selected = (
        blocks.all_blocks()
        if region is None
        else blocks_for_bounds(*puc.REGIONS[region])
    )
    stamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y_%m_%d_%H%M%S")
    work_dir = data.root / "grids" / spec.grid.name / ".tabulate" / stamp
    mkdir(work_dir / "blocks", parents=True, exist_ok=True)
    plan = Plan(
        output_dir=str(data.root),
        grid=spec.grid.name,
        source=spec.population,
        shapes=shapes,
        time_points=time_points,
        blocks=[block.key for block in selected],
        region=region,
    )
    (work_dir / "plan.yaml").write_text(
        yaml.safe_dump(plan.model_dump(mode="json"), sort_keys=False)
    )
    jobmon.run_parallel(
        runner="putask tabulate",
        task_name="population-block",
        node_args={"block-key": plan.blocks},
        task_args={"work-dir": str(work_dir)},
        task_resources=TASK_RESOURCES,
        concurrency_limit=CONCURRENCY_LIMIT,
        max_attempts=1,
        log_root=data.root / "logs" / "tabulate",
    )
    return reduce(work_dir)


@click.command()
@click.argument(
    "specification", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--shapes",
    type=click.Choice(sorted(admin.SHAPES)),
    required=True,
    help="The shapes version whose units to tabulate by.",
)
@click.option(
    "--region",
    type=click.Choice(sorted(puc.REGIONS)),
    help="Tabulate only the blocks covering this region, for development.",
)
@with_output_directory(puc.URBANICITY_ROOT)
def population_run(
    specification: Path, shapes: str, region: str | None, output_dir: str
) -> None:
    """Tabulate a specification's population by 1 km cell and admin unit."""
    if region is not None and Path(output_dir) == puc.URBANICITY_ROOT:
        msg = "A run limited to a region needs its own --output-dir."
        raise click.UsageError(msg)
    spec = ClassificationSpecification.from_yaml(specification)
    manifest = run(spec, UrbanicityData(output_dir), shapes, region)
    print("Nothing to tabulate." if manifest is None else f"Done: {manifest}")


@click.command()
@click.option("--work-dir", required=True, type=click.Path(path_type=Path))
@click.option("--block-key", required=True)
def population_block_task(work_dir: Path, block_key: str) -> None:
    """Tabulate one block's population by cell and admin unit."""
    run_block(work_dir, block_key)
