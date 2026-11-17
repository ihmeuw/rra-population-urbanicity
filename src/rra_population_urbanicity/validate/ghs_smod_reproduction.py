"""Compare a classification with GHS-SMOD, cell by cell.

For each time point it writes, under
classifications/<version>/validation/ghs_smod_reproduction/<release>/:
- summary.yaml: agreement over GHS-SMOD's cells with data, and over those
  with population or land (the open ocean, class 10 in both, would otherwise
  dominate); recall, precision and overlap for each class; population by class;
  and the number of urban centres;
- confusion_<time_point>.csv: cells and population by (GHS-SMOD class, our
  class), for both sets of cells;
- disagreement_<time_point>.tif: GHS-SMOD's code times 100 plus ours where they
  differ, and 0 where they agree or either has no data.
"""

from pathlib import Path
from typing import Any, Literal

import click
import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterio
import yaml
from rasterio.windows import Window
from rra_tools.cli_tools import with_output_directory
from rra_tools.shell_tools import mkdir

from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.classifiers.degurba import rules
from rra_population_urbanicity.classifiers.degurba.scheme import (
    DEGURBA_L2,
    URBAN_CENTRE,
)
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.inputs import grids
from rra_population_urbanicity.inputs.providers import ghsl
from rra_population_urbanicity.reference.ghsl import GhslRaster, check_on_grid
from rra_population_urbanicity.specification import ClassificationSpecification

Release = Literal["r2023a_v2", "r2023a_v1"]
SMOD_NODATA = -200
STRIP_ROWS = 2048
CODES = [c.code for c in DEGURBA_L2.classes]
# Position of each class code in CODES, for codes 0-255; -1 for anything else.
INDEX = np.full(256, -1, dtype=np.int64)
INDEX[CODES] = np.arange(len(CODES))
SUBSETS = ("all", "populated_or_land")


def smod_path(data: UrbanicityData, release: Release, time_point: str) -> Path:
    raster = GhslRaster(
        product="SMOD",
        release="R2023A",
        epoch=ghsl.epoch(time_point),
        version=2 if release == "r2023a_v2" else 1,
    )
    root = data.reference_root("ghsl", raster.product_dir, raster.release_dir)
    return root / f"{raster.stem}.tif"


def confusion_counts(
    reference: npt.NDArray[np.integer[Any]],
    ours: npt.NDArray[np.integer[Any]],
    weights: npt.NDArray[np.float64] | None,
) -> npt.NDArray[np.float64]:
    """Counts by (reference class, our class), indexed by position in CODES."""
    n = len(CODES)
    pairs = INDEX[reference.astype(np.int64) % 256] * n + INDEX[ours]
    totals = np.bincount(
        pairs.ravel(),
        weights=None if weights is None else weights.ravel(),
        minlength=n * n,
    )
    return totals.reshape(n, n).astype(np.float64)


def class_summary(
    cells: npt.NDArray[np.float64], population: npt.NDArray[np.float64]
) -> dict[int, dict[str, float | None]]:
    summary = {}
    for i, code in enumerate(CODES):
        reference, ours, both = cells[i].sum(), cells[:, i].sum(), cells[i, i]
        union = reference + ours - both
        summary[code] = {
            "reference_cells": int(reference),
            "our_cells": int(ours),
            "agreeing_cells": int(both),
            "recall": float(both / reference) if reference else None,
            "precision": float(both / ours) if ours else None,
            "overlap": float(both / union) if union else None,
            "reference_population": float(population[i].sum()),
            "our_population": float(population[:, i].sum()),
        }
    return summary


def compare_time_point(
    data: UrbanicityData,
    spec: ClassificationSpecification,
    version: str,
    time_point: str,
    release: Release,
    out_dir: Path,
) -> dict[str, Any]:
    grid = spec.grid
    ours_path = data.class_raster_path(version, time_point)
    reference_path = smod_path(data, release, time_point)
    check_on_grid(reference_path, grid)
    with rasterio.open(ours_path) as src:
        col_off = round((src.transform.c - grid.x_min) / grid.cell_size)
        row_off = round((grid.y_max - src.transform.f) / grid.cell_size)
        window = Window(col_off, row_off, src.width, src.height)
        profile = src.profile | {"dtype": "uint16", "nodata": None}

    n = len(CODES)
    cells = {subset: np.zeros((n, n)) for subset in SUBSETS}
    population = {subset: np.zeros((n, n)) for subset in SUBSETS}
    centres = {}
    with (
        rasterio.open(ours_path) as ours_src,
        rasterio.open(reference_path) as reference_src,
        rasterio.open(
            out_dir / f"disagreement_{time_point}.tif", "w", **profile
        ) as diff_dst,
    ):
        for start in range(0, int(window.height), STRIP_ROWS):
            rows = min(STRIP_ROWS, int(window.height) - start)
            strip = Window(0, start, window.width, rows)
            grid_strip = Window(
                window.col_off, window.row_off + start, window.width, rows
            )
            ours = ours_src.read(1, window=strip)
            reference = reference_src.read(1, window=grid_strip)
            people = grids.load(
                data, "population", spec.population, grid, time_point, grid_strip
            )
            land = grids.load(data, "land", spec.land, grid, time_point, grid_strip)
            compared = (reference != SMOD_NODATA) & (ours != DEGURBA_L2.nodata)
            masks = {
                "all": compared,
                "populated_or_land": compared & ((people > 0) | (land > 0)),
            }
            for subset, mask in masks.items():
                cells[subset] += confusion_counts(reference[mask], ours[mask], None)
                population[subset] += confusion_counts(
                    reference[mask], ours[mask], people[mask]
                )
            differ = compared & (reference != ours)
            diff = np.where(differ, reference.astype(np.int64) * 100 + ours, 0)
            diff_dst.write(diff.astype(np.uint16), 1, window=strip)
        reference_window = reference_src.read(1, window=window)
        ours_window = ours_src.read(1)
    for name, codes in (("reference", reference_window), ("ours", ours_window)):
        _, centres[name] = rules.label(codes == URBAN_CENTRE, 4)

    rows_out = []
    summary: dict[str, Any] = {"urban_centres": centres}
    for subset in SUBSETS:
        total = cells[subset].sum()
        summary[subset] = {
            "cells": int(total),
            "agreement": float(np.trace(cells[subset]) / total) if total else None,
            "population_agreement": float(
                np.trace(population[subset]) / population[subset].sum()
            )
            if population[subset].sum()
            else None,
            "classes": class_summary(cells[subset], population[subset]),
        }
        for (i, j), value in np.ndenumerate(cells[subset]):
            rows_out.append(
                (
                    subset,
                    CODES[i],
                    CODES[j],
                    int(value),
                    float(population[subset][i, j]),
                )
            )
    pd.DataFrame(
        rows_out, columns=["cells_in", "reference", "ours", "cells", "population"]
    ).to_csv(out_dir / f"confusion_{time_point}.csv", index=False)
    return summary


def validate(data: UrbanicityData, version: str, release: Release) -> dict[str, Any]:
    root = data.classification_root(version)
    spec = ClassificationSpecification.from_yaml(root / "specification.yaml")
    out_dir = root / "validation" / "ghs_smod_reproduction" / release
    mkdir(out_dir, parents=True, exist_ok=True)
    summary = {
        "version": version,
        "reference": release,
        "time_points": {
            time_point: compare_time_point(
                data, spec, version, time_point, release, out_dir
            )
            for time_point in spec.time_points
        },
    }
    (out_dir / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    return summary


@click.command()
@click.argument("version")
@click.option(
    "--reference",
    type=click.Choice(["r2023a_v2", "r2023a_v1"]),
    default="r2023a_v2",
    show_default=True,
    help="The GHS-SMOD release to compare with.",
)
@with_output_directory(puc.URBANICITY_ROOT)
def validate_ghs_smod(version: str, reference: Release, output_dir: str) -> None:
    """Compare a classification version with GHS-SMOD."""
    summary = validate(UrbanicityData(output_dir), version, reference)
    for time_point, result in summary["time_points"].items():
        agreement = [f"{s} {result[s]['agreement']:.4%}" for s in SUBSETS]
        overlaps = " ".join(
            f"{code}:{values['overlap']:.3f}"
            for code, values in result["populated_or_land"]["classes"].items()
            if values["overlap"] is not None
        )
        print(" | ".join([time_point, *agreement, f"overlap {overlaps}"]))
