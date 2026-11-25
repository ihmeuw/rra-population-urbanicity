"""Compare a classification's urban centres with GHSL's, centre by centre.

Two references, both for 2025 and both GHSL's urban centres on the same grid:
- `ghs_ucdb_r2024a`: GHS-UCDB R2024A, the urban centres of GHS-SMOD R2023A
  split at GADM 4.1's borders, with names and a plausibility score;
- `ghs_wup_r2025a`: GHS-WUP-DEGURBA's urban centres, the settlements behind
  WUP 2025's figures for cities, split at the UN's borders.
Their polygons follow the grid's cell edges, so each is put back on the grid
by cell centre. A reference centre is matched with the urban centre of ours
that holds most of the people in its cells, and the other way round; a pair
that are each other's match is mutual. A centre's population is the whole
centre's: ours from the classification's own population layer, the
reference's as GHSL published it from its own surface.

Written under classifications/<version>/validation/settlements/<reference>/:
- matches_<time_point>.parquet: one row per reference centre, with its id,
  name, country and population; the urban centre of ours it matches, with that
  centre's population, the people of ours in the cells the two share, and
  whether the match is mutual;
- summary.yaml: how many centres match each way; for mutual pairs, the ratio
  of our population to the reference's, overall and by the reference's size;
  our centres that match several reference centres; and the largest ratios
  among reference centres of a million people or more.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio import features, windows
from rra_tools.cli_tools import with_output_directory
from rra_tools.shell_tools import mkdir

from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.inputs import grids
from rra_population_urbanicity.specification import ClassificationSpecification
from rra_population_urbanicity.validate.classification_diff import class_window

TIME_POINT = "2025q1"
# Bounds of the reference's size classes, in people.
SIZE_BOUNDS = [0, 1e5, 5e5, 1e6, 5e6, np.inf]
LARGE = 1e6
LARGEST_RATIOS = 10
# A mutual pair whose populations are within this factor of each other agrees.
CLOSE_RATIO = 1.25


@dataclass(frozen=True)
class Reference:
    """A file of reference urban centres, and its columns."""

    product_dir: str
    release_dir: str
    file: str
    layer: str | None
    population: str
    country: str
    name: str | None
    id: str = "ID_UC_G0"


REFERENCES = {
    "ghs_ucdb_r2024a": Reference(
        product_dir="ghs_ucdb",
        release_dir="r2024a_v1_2",
        file="GHS_UCDB_GLOBE_R2024A.gpkg",
        layer="GHSL_UCDB_THEME_GENERAL_CHARACTERISTICS_GLOBE_R2024A",
        population="GC_POP_TOT_2025",
        country="GC_CNT_GAD_2025",
        name="GC_UCN_MAI_2025",
    ),
    "ghs_wup_r2025a": Reference(
        product_dir="ghs_wup_degurba",
        release_dir="r2025a",
        file="GHS_WUP_DEGURBA_E2025_GLOBE_R2025A_54009_1000_UC_V1_0.shp",
        layer=None,
        population="POP_2025",
        country="UNLocID",
        name=None,
    ),
}


def read_reference(data: UrbanicityData, reference: Reference, crs: str) -> gpd.GeoDataFrame:
    """The reference centres: id, name, country, population and shape."""
    path = data.reference_root("ghsl", reference.product_dir, reference.release_dir)
    columns = [reference.id, reference.population, reference.country]
    if reference.name is not None:
        columns.append(reference.name)
    frame = gpd.read_file(path / reference.file, layer=reference.layer, columns=columns)
    frame = frame.to_crs(crs)
    return gpd.GeoDataFrame(
        {
            "reference_id": frame[reference.id].astype(np.int64),
            "name": frame[reference.name] if reference.name else None,
            "country": frame[reference.country].astype(str),
            "reference_population": frame[reference.population].astype(np.float64),
        },
        geometry=frame.geometry,
        crs=frame.crs,
    )


def overlaps(
    ours: np.ndarray, theirs: np.ndarray, population: np.ndarray
) -> pd.DataFrame:
    """Cells and our people shared by each pair of our centre and theirs."""
    shared = (ours > 0) & (theirs > 0)
    return (
        pd.DataFrame(
            {
                "our_id": ours[shared].astype(np.int64),
                "reference_id": theirs[shared].astype(np.int64),
                "overlap_population": population[shared],
                "overlap_cells": 1,
            }
        )
        .groupby(["our_id", "reference_id"], as_index=False)
        .sum()
    )


def best(pairs: pd.DataFrame, by: str) -> pd.DataFrame:
    """For each value of `by`, the pair sharing most people (then most cells)."""
    ordered = pairs.sort_values(
        ["overlap_population", "overlap_cells"], ascending=False, kind="stable"
    )
    return ordered.drop_duplicates(by)


def match(
    reference: pd.DataFrame, our_population: pd.Series, pairs: pd.DataFrame
) -> pd.DataFrame:
    """Each reference centre with the centre of ours it matches, if any."""
    theirs = best(pairs, "reference_id")
    ours = best(pairs, "our_id")
    mutual = theirs.merge(ours[["our_id", "reference_id"]], how="left", indicator=True)
    theirs = theirs.assign(mutual=(mutual["_merge"] == "both").to_numpy())
    table = reference.merge(theirs, on="reference_id", how="left")
    table["our_population"] = table["our_id"].map(our_population)
    table["mutual"] = table["mutual"].fillna(False).astype(bool)
    return table


def summarise(table: pd.DataFrame, our_ids: pd.Index, pairs: pd.DataFrame) -> dict[str, Any]:
    matched = table["our_id"].notna()
    people = table["reference_population"]
    matched_ours = pd.Index(best(pairs, "our_id")["our_id"])
    mutual = table[table["mutual"]]
    ratio = mutual["our_population"] / mutual["reference_population"]
    sizes = pd.cut(mutual["reference_population"], SIZE_BOUNDS, right=False)
    shared = table.loc[matched, "our_id"].value_counts()
    large = table[table["mutual"] & (people >= LARGE)]
    log_ratio = np.log(large["our_population"] / large["reference_population"])
    largest = large.loc[log_ratio.abs().sort_values(ascending=False).index[:LARGEST_RATIOS]]
    return {
        "reference_centres": int(len(table)),
        "our_centres": int(our_ids.size),
        "reference_matched": int(matched.sum()),
        "reference_people_matched_share": float(people[matched].sum() / people.sum()),
        "ours_matched": int(our_ids.isin(matched_ours).sum()),
        "mutual_pairs": int(len(mutual)),
        "ratio": ratio_summary(ratio),
        "ratio_by_reference_size": {
            f"{interval.left:.0f}-{interval.right:.0f}": ratio_summary(group)
            for interval, group in ratio.groupby(sizes, observed=False)
        },
        "ours_matching_several": {
            "centres": int((shared > 1).sum()),
            "reference_centres": int(shared[shared > 1].sum()),
        },
        "largest_ratios": [
            {
                "name": None if pd.isna(row["name"]) else str(row["name"]),
                "country": row["country"],
                "reference_population": round(float(row["reference_population"])),
                "our_population": round(float(row["our_population"])),
            }
            for _, row in largest.iterrows()
        ],
    }


def ratio_summary(ratio: pd.Series) -> dict[str, Any]:
    if ratio.empty:
        return {"pairs": 0}
    return {
        "pairs": int(ratio.size),
        "median": float(ratio.median()),
        "quartiles": [float(ratio.quantile(0.25)), float(ratio.quantile(0.75))],
        "share_close": float(
            ((ratio <= CLOSE_RATIO) & (ratio >= 1 / CLOSE_RATIO)).mean()
        ),
    }


def compare(
    data: UrbanicityData, version: str, reference_name: str
) -> dict[str, Any]:
    root = data.classification_root(version)
    spec = ClassificationSpecification.from_yaml(root / "specification.yaml")
    if TIME_POINT not in spec.time_points:
        msg = f"{version} has no {TIME_POINT}, the year of the reference centres."
        raise ValueError(msg)
    grid = spec.grid
    path = data.urban_centre_raster_path(version, TIME_POINT)
    window = class_window(path, spec)
    with rasterio.open(path) as src:
        ours = src.read(1)
    population = grids.load(data, "population", spec.population, grid, TIME_POINT, window)
    reference = read_reference(data, REFERENCES[reference_name], grid.crs)
    theirs = features.rasterize(
        zip(reference.geometry, reference["reference_id"], strict=True),
        out_shape=ours.shape,
        transform=windows.transform(window, grid.transform),
        fill=0,
        dtype="int32",
    )
    clusters = pd.read_parquet(data.cluster_table_path(version, TIME_POINT))
    centres = clusters[clusters["cluster_type"] == "urban_centre"].set_index("cluster_id")
    pairs = overlaps(ours, theirs, population)
    table = match(
        pd.DataFrame(reference.drop(columns="geometry")),
        centres["population"],
        pairs,
    )
    out_dir = root / "validation" / "settlements" / reference_name
    mkdir(out_dir, parents=True, exist_ok=True)
    table.to_parquet(out_dir / f"matches_{TIME_POINT}.parquet", index=False)
    summary = {
        "version": version,
        "reference": reference_name,
        "time_point": TIME_POINT,
        **summarise(table, centres.index, pairs),
    }
    (out_dir / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    return summary


@click.command()
@click.argument("version")
@click.option(
    "--reference",
    "references",
    type=click.Choice(list(REFERENCES)),
    multiple=True,
    default=list(REFERENCES),
    show_default=True,
    help="The reference urban centres to compare with.",
)
@with_output_directory(puc.URBANICITY_ROOT)
def validate_settlements(
    version: str, references: tuple[str, ...], output_dir: str
) -> None:
    """Compare a classification's urban centres with GHSL's, centre by centre."""
    data = UrbanicityData(output_dir)
    for reference in references:
        summary = compare(data, version, reference)
        ratio = summary["ratio"]
        print(
            f"{reference}: {summary['reference_matched']:,} of "
            f"{summary['reference_centres']:,} matched, {summary['mutual_pairs']:,} "
            f"mutual; population ratio median {ratio.get('median', float('nan')):.2f}"
        )
