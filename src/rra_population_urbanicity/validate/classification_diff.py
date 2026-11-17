"""Compare two classifications on the same grid, weighted by population.

For each time point both classify, it writes, under
classifications/<version>/validation/classification_diff/<other version>/:
- summary.yaml: the cells and people that change class, and each class's
  cells and people under both classifications;
- confusion_<time_point>.csv: cells and people by (class in <version>, class
  in <other version>);
- by_country_<time_point>.parquet: the same by country (location_id under
  the LSAE admin-0 shapes, by cell centre; 0 for none), without the empty
  rows.

People are counted twice, by each classification's own population layer:
`population` is the first's and `other_population` the second's. They are
the same when the two differ only in their land layer or rules, and differ
when the population surfaces do. Cells either classification gives no class
are left out.
"""

from pathlib import Path
from typing import Any

import click
import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterio
import yaml
from rasterio.windows import Window
from rra_tools.cli_tools import with_output_directory
from rra_tools.shell_tools import mkdir

from rra_population_urbanicity import admin
from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.classifiers.degurba.scheme import DEGURBA_L2
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.inputs import grids
from rra_population_urbanicity.specification import ClassificationSpecification

SHAPES = "lsae_1578_a0"
STRIP_ROWS = 2048
CODES = [c.code for c in DEGURBA_L2.classes]
# Position of each class code in CODES, for codes 0-255; -1 for anything else.
INDEX = np.full(256, -1, dtype=np.int64)
INDEX[CODES] = np.arange(len(CODES))
MEASURES = ("cells", "population", "other_population")


def class_window(path: Path, spec: ClassificationSpecification) -> Window:
    """The window of the grid a class raster covers."""
    grid = spec.grid
    with rasterio.open(path) as src:
        col_off = round((src.transform.c - grid.x_min) / grid.cell_size)
        row_off = round((grid.y_max - src.transform.f) / grid.cell_size)
        return Window(col_off, row_off, src.width, src.height)


def compare_time_point(
    data: UrbanicityData,
    specs: tuple[ClassificationSpecification, ClassificationSpecification],
    versions: tuple[str, str],
    time_point: str,
    locations: admin.LocationArray,
    out_dir: Path,
) -> dict[str, Any]:
    spec, other_spec = specs
    grid = spec.grid
    paths = [data.class_raster_path(v, time_point) for v in versions]
    window = class_window(paths[0], spec)
    if class_window(paths[1], other_spec) != window:
        msg = f"{versions[0]} and {versions[1]} cover different windows of the grid."
        raise ValueError(msg)

    # Each location's position among those present, so (location, class,
    # other class) can be counted with one bincount.
    location_ids = np.flatnonzero(np.bincount(locations.ravel()))
    position = np.zeros(int(location_ids[-1]) + 1, dtype=np.int64)
    position[location_ids] = np.arange(location_ids.size)
    n = len(CODES)
    size = location_ids.size * n * n
    totals = {measure: np.zeros(size) for measure in MEASURES}
    with rasterio.open(paths[0]) as src, rasterio.open(paths[1]) as other_src:
        for start in range(0, int(window.height), STRIP_ROWS):
            rows = min(STRIP_ROWS, int(window.height) - start)
            strip = Window(0, start, window.width, rows)
            grid_strip = Window(
                window.col_off, window.row_off + start, window.width, rows
            )
            codes = INDEX[src.read(1, window=strip)]
            other_codes = INDEX[other_src.read(1, window=strip)]
            compared = (codes >= 0) & (other_codes >= 0)
            where = position[locations[grid_strip.toslices()]]
            keys = (where * n + codes) * n + other_codes
            weights = {
                "cells": None,
                "population": grids.load(
                    data, "population", spec.population, grid, time_point, grid_strip
                ),
                "other_population": grids.load(
                    data,
                    "population",
                    other_spec.population,
                    grid,
                    time_point,
                    grid_strip,
                ),
            }
            for measure, weight in weights.items():
                totals[measure] += np.bincount(
                    keys[compared],
                    weights=None if weight is None else weight[compared],
                    minlength=size,
                )

    shape = (location_ids.size, n, n)
    table = pd.DataFrame(
        {
            "location_id": np.repeat(location_ids, n * n),
            "class_code": np.tile(np.repeat(CODES, n), location_ids.size),
            "other_class_code": np.tile(CODES, location_ids.size * n),
            **{m: totals[m] for m in MEASURES},
        }
    )
    table["cells"] = table["cells"].astype(np.int64)
    table[table["cells"] > 0].reset_index(drop=True).to_parquet(
        out_dir / f"by_country_{time_point}.parquet", index=False
    )
    world = {m: totals[m].reshape(shape).sum(axis=0) for m in MEASURES}
    pd.DataFrame(
        [
            (CODES[i], CODES[j], *(world[m][i, j] for m in MEASURES))
            for i in range(n)
            for j in range(n)
        ],
        columns=["class_code", "other_class_code", *MEASURES],
    ).astype({"cells": np.int64}).to_csv(
        out_dir / f"confusion_{time_point}.csv", index=False
    )
    return summarise(world)


def summarise(world: dict[str, npt.NDArray[np.float64]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for measure in MEASURES:
        matrix = world[measure]
        total, same = float(matrix.sum()), float(np.trace(matrix))
        summary[measure] = {
            "total": total,
            "changed": total - same,
            "changed_share": (total - same) / total if total else None,
        }
    summary["classes"] = {
        code: {
            measure: [float(world[measure][i].sum()), float(world[measure][:, i].sum())]
            for measure in MEASURES
        }
        for i, code in enumerate(CODES)
    }
    return summary


def compare(data: UrbanicityData, version: str, other: str) -> dict[str, Any]:
    specs = tuple(
        ClassificationSpecification.from_yaml(
            data.classification_root(v) / "specification.yaml"
        )
        for v in (version, other)
    )
    spec, other_spec = specs
    if spec.grid != other_spec.grid:
        msg = f"{version} and {other} are on different grids."
        raise ValueError(msg)
    time_points = [tp for tp in spec.time_points if tp in other_spec.time_points]
    if not time_points:
        msg = f"{version} and {other} share no time point."
        raise ValueError(msg)
    locations = admin.location_raster(data, spec.grid, SHAPES)
    out_dir = (
        data.classification_root(version) / "validation" / "classification_diff" / other
    )
    mkdir(out_dir, parents=True, exist_ok=True)
    summary = {
        "version": version,
        "other_version": other,
        "shapes": SHAPES,
        "time_points": {
            tp: compare_time_point(
                data, (spec, other_spec), (version, other), tp, locations, out_dir
            )
            for tp in time_points
        },
    }
    (out_dir / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    return summary


@click.command()
@click.argument("version")
@click.argument("other_version")
@with_output_directory(puc.URBANICITY_ROOT)
def validate_diff(version: str, other_version: str, output_dir: str) -> None:
    """Compare two classification versions, weighted by population."""
    summary = compare(UrbanicityData(output_dir), version, other_version)
    for time_point, result in summary["time_points"].items():
        people = result["population"]
        print(
            f"{time_point}: {people['changed']:,.0f} people "
            f"({people['changed_share']:.2%}) and {result['cells']['changed']:,.0f} "
            "cells change class"
        )
