"""Admin units on the 40 m model grid, one raster per block.

Each pixel takes the unit whose shape contains its centre, as everywhere else
in the pipeline, or 0 where no shape does. The shapes are read and projected
once, and the blocks are rasterized in threads on the machine the command runs
on, since a block takes a second or two.

As a check, each block's country shapes are rasterized too, and the manifest
counts the pixels where the two disagree: in a country and in no unit, in a
unit and in no country, and in a unit of another country.
"""

import datetime
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import click
import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterio
import shapely
import yaml
from rasterio import features
from rasterio.errors import NotGeoreferencedWarning
from rra_tools.cli_tools import with_output_directory
from rra_tools.shell_tools import mkdir

from rra_population_urbanicity import admin
from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.inputs import blocks
from rra_population_urbanicity.inputs.aggregate import blocks_for_bounds
from rra_population_urbanicity.inputs.blocks import MODEL_CRS, Block

type IdArray = npt.NDArray[np.int32]

# The east edge of the world on the model grid. The grid is a little wider,
# and pixels past this edge take the shapes at the west edge.
WORLD_X_MAX = 20_037_508.34
# Tiles of the block rasters, in pixels.
TILE_SIZE = 512
KINDS = ("in_a_country_and_no_unit", "in_a_unit_and_no_country", "in_another_country")


@dataclass(frozen=True)
class Shapes:
    """Shapes projected to the model grid, with an index to find them by place."""

    geometry: npt.NDArray[Any]
    location_id: npt.NDArray[np.int64]
    tree: shapely.STRtree


def load_shapes(name: str, keep: list[int]) -> Shapes:
    frame = admin.read_shapes(name, MODEL_CRS, keep)
    geometry = frame.geometry.to_numpy()
    return Shapes(
        geometry, frame["location_id"].to_numpy(np.int64), shapely.STRtree(geometry)
    )


def burn(shapes: Shapes, block: Block) -> IdArray:
    """Each pixel's location_id, from the shape containing its centre."""
    height, width = block.shape
    transform = block.transform
    west, north = transform.c, transform.f
    east, south = west + width * transform.a, north + height * transform.e
    pieces, ids = [], []
    for shift in (0.0, 2 * WORLD_X_MAX):
        if shift and east <= WORLD_X_MAX:
            continue
        found = shapes.tree.query(shapely.box(west - shift, south, east - shift, north))
        clipped = shapely.clip_by_rect(
            shapes.geometry[found], west - shift, south, east - shift, north
        )
        if shift:
            offset = np.array([shift, 0.0])
            clipped = shapely.transform(clipped, lambda xy, o=offset: xy + o)
        kept = ~shapely.is_empty(clipped)
        pieces.append(clipped[kept])
        ids.append(shapes.location_id[found][kept])
    geometry, values = np.concatenate(pieces), np.concatenate(ids)
    if not geometry.size:
        return np.full(block.shape, admin.NO_UNIT, dtype=np.int32)
    burned: IdArray = features.rasterize(
        zip(geometry, values.tolist(), strict=True),
        out_shape=block.shape,
        transform=transform,
        fill=admin.NO_UNIT,
        dtype="int32",
    )
    return burned


def country_lookup(parents: pd.DataFrame) -> npt.NDArray[np.int64]:
    """Each location_id's country, indexed by location_id; -1 for ids not in the hierarchy."""
    lookup = np.full(int(parents["location_id"].max()) + 1, -1, dtype=np.int64)
    lookup[parents["location_id"].to_numpy()] = parents["country_id"].to_numpy()
    return lookup


def coverage(
    units: IdArray, countries: IdArray, country_of: npt.NDArray[np.int64]
) -> pd.DataFrame:
    """Pixels where the units and the country shapes disagree, by country.

    `country_of` gives each unit's country (`country_lookup`). A pixel in a
    unit of another country is counted under the country whose shape holds it.
    """
    in_unit, in_country = units != admin.NO_UNIT, countries != admin.NO_UNIT
    known = units < country_of.size
    unit_country = np.where(known, country_of[np.where(known, units, 0)], -1)
    unit_country[~in_unit] = admin.NO_UNIT
    masks = {
        KINDS[0]: (in_country & ~in_unit, countries),
        KINDS[1]: (in_unit & ~in_country, unit_country),
        KINDS[2]: (in_unit & in_country & (unit_country != countries), countries),
    }
    frames = []
    for kind, (mask, country) in masks.items():
        ids, pixels = np.unique(country[mask], return_counts=True)
        frames.append(pd.DataFrame({"country_id": ids, "kind": kind, "pixels": pixels}))
    return pd.concat(frames, ignore_index=True)


def write_block(path: Path, values: IdArray, block: Block) -> None:
    """Write a block's raster, under a temporary name until it is complete."""
    partial = path.with_name(f"{path.stem}.part.tif")
    with rasterio.open(
        partial,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="int32",
        crs=MODEL_CRS,
        transform=block.transform,
        nodata=None,
        tiled=True,
        blockxsize=TILE_SIZE,
        blockysize=TILE_SIZE,
        compress="deflate",
    ) as dst:
        dst.write(values, 1)
    partial.rename(path)


def run_block(
    block: Block,
    units: Shapes,
    countries: Shapes,
    country_of: npt.NDArray[np.int64],
    data: UrbanicityData,
    shapes: str,
) -> pd.DataFrame:
    unit_ids = burn(units, block)
    write_block(data.admin_block_path(shapes, block.key), unit_ids, block)
    country_ids = burn(countries, block)
    return coverage(unit_ids, country_ids, country_of).assign(block_key=block.key)


def run(data: UrbanicityData, shapes: str, region: str | None) -> Path | None:
    """Rasterize a shapes version onto every block, unless it already has been.

    Returns the manifest written, or None if there was nothing to do.
    """
    root = data.admin_block_root(shapes)
    manifest_path = root / "manifest.yaml"
    if manifest_path.exists():
        return None
    version = admin.SHAPES[shapes]
    units = load_shapes(version.units, version.unit_ids())
    countries = load_shapes(version.countries, version.country_ids())
    country_of = country_lookup(admin.unit_parents(version))
    selected = (
        blocks.all_blocks()
        if region is None
        else blocks_for_bounds(*puc.REGIONS[region])
    )
    mkdir(root, parents=True, exist_ok=True)
    with (
        ThreadPoolExecutor(len(os.sched_getaffinity(0))) as pool,
        warnings.catch_warnings(),
    ):
        # From worker threads, rasterio warns that the in-memory dataset it
        # rasterizes into has no transform. The rasters come out the same as
        # from one thread, and are written with their blocks' transforms.
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        results = list(
            pool.map(
                lambda b: run_block(b, units, countries, country_of, data, shapes),
                selected,
            )
        )
    table = pd.concat(results, ignore_index=True)
    table.to_csv(root / "coverage.csv", index=False)
    totals = table.groupby("kind")["pixels"].sum()
    by_country = table.groupby(["kind", "country_id"])["pixels"].sum()
    manifest = {
        "shapes": shapes,
        "units": version.units,
        "countries": version.countries,
        "hierarchy": version.hierarchy,
        "rule": "each pixel takes the unit whose shape contains its centre; 0 for none",
        "created": datetime.datetime.now(tz=datetime.UTC).isoformat(timespec="seconds"),
        "package_version": package_version("rra-population-urbanicity"),
        "blocks": len(selected),
        "region": region,
        "coverage_pixels": {kind: int(totals.get(kind, 0)) for kind in KINDS},
        "coverage_largest": {
            kind: {
                int(country): int(pixels)
                for country, pixels in by_country[kind].nlargest(10).items()
            }
            for kind in KINDS
            if kind in by_country.index.get_level_values(0)
        },
    }
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return manifest_path


@click.command()
@click.option(
    "--shapes",
    type=click.Choice(sorted(admin.SHAPES)),
    required=True,
    help="The shapes version to rasterize.",
)
@click.option(
    "--region",
    type=click.Choice(sorted(puc.REGIONS)),
    help="Rasterize only the blocks covering this region, for development.",
)
@with_output_directory(puc.URBANICITY_ROOT)
def admin_rasters_run(shapes: str, region: str | None, output_dir: str) -> None:
    """Rasterize admin units onto the 40 m model grid, block by block."""
    if region is not None and Path(output_dir) == puc.URBANICITY_ROOT:
        # The rasters are shared by every tabulation that uses the shapes, so
        # rasters covering only a region must not be written where they live.
        msg = "A run limited to a region needs its own --output-dir."
        raise click.UsageError(msg)
    manifest = run(UrbanicityData(output_dir), shapes, region)
    print("Already made." if manifest is None else f"Done: {manifest}")
