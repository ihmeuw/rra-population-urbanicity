"""Run a classification specification and write it as a new version.

A run writes, under classifications/<version>/:
- specification.yaml: the specification with its version, resolved grid and
  resolved parameters;
- scheme.yaml: what the class codes mean;
- manifest.yaml: what the run resolved from the data (the built-up threshold
  and the built-up per person, for each time point), the region if the run
  was limited to one, and the number of cells in each class;
- classes/<time_point>.tif: the class raster for each time point;
- clusters/<time_point>.parquet: the clusters of each time point (see
  `clusters.cluster_table`).
"""

import datetime
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import click
import numpy as np
import rasterio
import yaml
from rasterio import windows
from rasterio.windows import Window
from rra_tools.cli_tools import with_output_directory
from rra_tools.shell_tools import mkdir

from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.classify.base import ClassScheme
from rra_population_urbanicity.classify.degurba import clusters, rules
from rra_population_urbanicity.classify.degurba.scheme import DEGURBA_L2
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.grid import BoolArray, GridDefinition
from rra_population_urbanicity.inputs.providers import ghsl
from rra_population_urbanicity.specification import ClassificationSpecification

# Tiles of the class rasters, in cells.
BLOCK_SIZE = 512


def reserve_version(root: Path) -> str:
    """Claim the next version for today, YYYY_MM_DD.NNN, by creating its directory.

    Creating a directory is atomic, so two runs started together cannot claim
    the same version.
    """
    mkdir(root, parents=True, exist_ok=True)
    # The local date, as rra-population-model uses for its versions.
    today = datetime.datetime.now(tz=datetime.UTC).astimezone().strftime("%Y_%m_%d")
    taken = [int(path.name.split(".")[-1]) for path in root.glob(f"{today}.*")]
    number = max(taken, default=0) + 1
    while True:
        version = f"{today}.{number:03d}"
        try:
            mkdir(root / version)
        except FileExistsError:
            number += 1
        else:
            return version


def load_inputs(
    data: UrbanicityData,
    spec: ClassificationSpecification,
    time_point: str,
    domain: BoolArray,
) -> rules.DegurbaInputs:
    """The inputs of one time point on the whole grid.

    `domain` is the cells the grid's projection covers. Cells just outside it
    that hold people, land or built-up are classified too, so nobody is left
    without a class.
    """
    grid, window = spec.grid, spec.grid.full_window
    population = ghsl.load(data, spec.population, grid, time_point, window)
    land = ghsl.load(data, spec.land, grid, time_point, window)
    if spec.built is None:
        built = np.zeros_like(population)
    else:
        built = ghsl.load(data, spec.built, grid, time_point, window)
    domain = domain | (population > 0) | (land > 0) | (built > 0)
    return rules.DegurbaInputs(population, land, built, domain, grid.cell_area_km2)


def crop(inputs: rules.DegurbaInputs, window: Window) -> rules.DegurbaInputs:
    rows, cols = window.toslices()
    return rules.DegurbaInputs(
        inputs.population[rows, cols],
        inputs.land[rows, cols],
        inputs.built[rows, cols],
        inputs.domain[rows, cols],
        inputs.cell_area_km2,
    )


def built_per_person(
    inputs: rules.DegurbaInputs, spec: ClassificationSpecification
) -> float | None:
    if spec.method.parameters.density_on_land != "built_up_and_population":
        return None
    return rules.built_per_person(inputs.population, inputs.built)


def built_threshold(
    inputs: rules.DegurbaInputs, spec: ClassificationSpecification
) -> float:
    """The optimal built-up threshold computed on one time point's inputs."""
    params = spec.method.parameters
    prepared = rules.prepare(inputs, params, built_per_person(inputs, spec))
    return rules.optimal_built_threshold(inputs.population, prepared, params)


def fixed_built_threshold(
    data: UrbanicityData, spec: ClassificationSpecification, domain: BoolArray
) -> float | None:
    """The built-up threshold, if one value serves every time point.

    None if the built-up criterion is off, or if each time point computes its
    own threshold.
    """
    params = spec.method.parameters
    if not (params.uc_built_criterion or params.duc_built_criterion):
        return None
    if params.built_threshold != "optimal":
        return params.built_threshold
    if params.built_threshold_epoch == "each_time_point":
        return None
    inputs = load_inputs(data, spec, params.built_threshold_epoch, domain)
    return built_threshold(inputs, spec)


def write_classes(
    path: Path,
    codes: rules.CodeArray,
    grid: GridDefinition,
    window: Window,
    scheme: ClassScheme,
) -> None:
    mkdir(path.parent, parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": codes.shape[0],
        "width": codes.shape[1],
        "count": 1,
        "dtype": "uint8",
        "crs": grid.crs,
        "transform": windows.transform(window, grid.transform),
        "nodata": scheme.nodata,
        "tiled": True,
        "blockxsize": BLOCK_SIZE,
        "blockysize": BLOCK_SIZE,
        "compress": "deflate",
    }
    colours = {
        c.code: (*(int(c.colour[i : i + 2], 16) for i in (1, 3, 5)), 255)
        for c in scheme.classes
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(codes, 1)
        dst.write_colormap(1, colours)


def run(
    spec: ClassificationSpecification, data: UrbanicityData, region: str | None
) -> str:
    """Run a specification and return the version it was written as."""
    grid = spec.grid
    params = spec.method.parameters
    version = reserve_version(data.classifications_root)
    root = data.classification_root(version)
    window = grid.full_window
    if region is not None:
        window = grid.window_for_bounds(*puc.REGIONS[region])

    domain = grid.domain(grid.full_window)
    fixed_threshold = fixed_built_threshold(data, spec, domain)
    resolved: dict[str, dict[str, Any]] = {}
    for time_point in spec.time_points:
        inputs = load_inputs(data, spec, time_point, domain)
        per_person = built_per_person(inputs, spec)
        threshold = fixed_threshold
        if threshold is None and (
            params.uc_built_criterion or params.duc_built_criterion
        ):
            threshold = built_threshold(inputs, spec)
        cropped = crop(inputs, window)
        del inputs
        codes = rules.classify(
            cropped,
            params,
            built_threshold=threshold,
            built_per_person=per_person,
        )
        write_classes(
            data.class_raster_path(version, time_point), codes, grid, window, DEGURBA_L2
        )
        table_path = data.cluster_table_path(version, time_point)
        mkdir(table_path.parent, parents=True, exist_ok=True)
        transform = windows.transform(window, grid.transform)
        clusters.cluster_table(codes, cropped, params, transform, grid.crs).to_parquet(
            table_path, index=False
        )
        del cropped
        values, counts = np.unique(codes, return_counts=True)
        resolved[time_point] = {
            "built_threshold": threshold,
            "built_per_person": per_person,
            "cells_by_class": {
                int(v): int(n) for v, n in zip(values, counts, strict=True)
            },
        }

    spec.model_copy(update={"version": version}).to_yaml(root / "specification.yaml")
    DEGURBA_L2.to_yaml(root / "scheme.yaml")
    manifest = {
        "version": version,
        "created": datetime.datetime.now(tz=datetime.UTC).isoformat(timespec="seconds"),
        "package_version": package_version("rra-population-urbanicity"),
        "region": None
        if region is None
        else {
            "name": region,
            "bounds": list(puc.REGIONS[region]),
            "window": [window.row_off, window.col_off, window.height, window.width],
        },
        # With no built-up layer, the on-land test reads built-up as zero.
        "built_up_layer": "none: read as zero"
        if spec.built is None
        else "from the specification",
        "time_points": resolved,
    }
    (root / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    return version


@click.command()
@click.argument(
    "specification", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--region",
    type=click.Choice(sorted(puc.REGIONS)),
    help=(
        "Classify only this region, for development. The built-up threshold "
        "and built-up per person are still computed on the whole grid."
    ),
)
@with_output_directory(puc.URBANICITY_ROOT)
def classify_run(specification: Path, region: str | None, output_dir: str) -> None:
    """Run a classification specification and write it as a new version."""
    spec = ClassificationSpecification.from_yaml(specification)
    print(run(spec, UrbanicityData(output_dir), region))
