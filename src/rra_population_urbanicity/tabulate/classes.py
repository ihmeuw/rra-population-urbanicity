"""A classification tabulated by admin unit: joins against the population table.

Every 40 m pixel has the class of its 1 km cell, so joining the table of
people by cell and unit to a class raster gives each unit's people by class
exactly. For each time point of a classification, it writes, under
classifications/<version>/tabulations/<shapes>/:
- <time point>.parquet: `location_id`, `class_code`, `population`. Location 0
  holds the people in no unit, and class `nodata` the people in cells the
  classification does not cover (a run limited to a region).
- stage2_<time point>.parquet: every admin 2, admin 1 and country with people,
  its shares of people in urban centres and in rural cells, and its Degree of
  Urbanisation class as a unit ("Stage II"): a city if at least half its
  people are in urban centres, rural if more than half are in rural cells,
  and towns and semi-dense areas otherwise. For schemes with a `degurba_l1`
  group.
- cluster_people_<time point>.parquet: each cluster's people by admin 2 unit;
  and cluster_units_<time point>.parquet: each cluster's country and admin 1,
  the ones holding most of its people, with their shares. Urban centres and
  level-1 urban clusters are found through their id rasters, and dense,
  semi-dense and rural clusters by labelling the class raster as the cluster
  table does. A classification made before urban-cluster ids were saved
  cannot give its urban clusters units, since they cannot be labelled again
  from its classes; they are left out, and the manifest says so.
- manifest.yaml: what was tabulated, and the check that a join gives the same
  totals as tabulating the 40 m pixels of a few blocks through the pixel view.
"""

import datetime
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import click
import numpy as np
import numpy.typing as npt
import pandas as pd
import rasterio
import yaml
from rra_tools.cli_tools import with_output_directory
from rra_tools.shell_tools import mkdir

from rra_population_urbanicity import admin
from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.classifiers.base import ClassScheme
from rra_population_urbanicity.classifiers.degurba import clusters, rules
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.grid import GridDefinition
from rra_population_urbanicity.inputs.aggregate import rows_on_grid
from rra_population_urbanicity.inputs.blocks import Block
from rra_population_urbanicity.inputs.providers import populace
from rra_population_urbanicity.specification import (
    ClassificationSpecification,
    PopulaceSource,
)
from rra_population_urbanicity.tabulate import pixel_view, population

# Blocks whose 40 m pixels are tabulated through the pixel view, to check the
# join: Dhaka; Lagos; Seattle; the Low Countries, where many units meet; and
# Fiji, cut short at the antimeridian.
CHECK_BLOCKS = (
    "B-0091X-0011Y",
    "B-0062X-0017Y",
    "B-0019X-0005Y",
    "B-0062X-0004Y",
    "B-0122X-0025Y",
)
CHECK_TOLERANCE = 1e-9
# Stage II: a unit is a city if at least this share of its people are in urban
# centres, and rural if more than this share are in rural cells.
STAGE2_SHARE = 0.5


def read_window(path: Path, grid: GridDefinition) -> tuple[npt.NDArray[Any], int, int]:
    """A raster on the grid, with the row and column of its first cell."""
    with rasterio.open(path) as src:
        row_off = round((grid.y_max - src.transform.f) / grid.cell_size)
        col_off = round((src.transform.c - grid.x_min) / grid.cell_size)
        return src.read(1), row_off, col_off


def at_cells(
    values: npt.NDArray[Any],
    row_off: int,
    col_off: int,
    rows: npt.NDArray[np.int64],
    cols: npt.NDArray[np.int64],
    fill: int,
) -> npt.NDArray[Any]:
    """The values at grid cells, or `fill` for cells outside the raster."""
    local_row, local_col = rows - row_off, cols - col_off
    inside = (
        (local_row >= 0)
        & (local_row < values.shape[0])
        & (local_col >= 0)
        & (local_col < values.shape[1])
    )
    result = np.full(rows.shape, fill, dtype=values.dtype)
    result[inside] = values[local_row[inside], local_col[inside]]
    return result


def stage2(
    by_class: pd.DataFrame, scheme: ClassScheme, parents: pd.DataFrame
) -> pd.DataFrame | None:
    """Each unit's Degree of Urbanisation class as a unit, at three levels."""
    groups = scheme.groups.get("degurba_l1")
    if groups is None:
        return None
    city = [scheme.code(k) for k in groups["city"]]
    rural = [scheme.code(k) for k in groups["rural"]]
    units = by_class[by_class["location_id"] != admin.NO_UNIT].merge(
        parents, on="location_id"
    )
    frames = []
    for level, column in (
        ("admin2", "location_id"),
        ("admin1", "admin1_id"),
        ("country", "country_id"),
    ):
        people = units.groupby(column)["population"].sum()
        in_city = (
            units[units["class_code"].isin(city)].groupby(column)["population"].sum()
        )
        in_rural = (
            units[units["class_code"].isin(rural)].groupby(column)["population"].sum()
        )
        frame = pd.DataFrame(
            {
                "population": people,
                "urban_centre_share": in_city.reindex(people.index, fill_value=0.0)
                / people,
                "rural_share": in_rural.reindex(people.index, fill_value=0.0) / people,
            }
        )
        frame["stage2"] = np.where(
            frame["urban_centre_share"] >= STAGE2_SHARE,
            "city",
            np.where(frame["rural_share"] > STAGE2_SHARE, "rural", "town_semi_dense"),
        )
        frames.append(
            frame.rename_axis("location_id").reset_index().assign(level=level)
        )
    return pd.concat(frames, ignore_index=True)[
        [
            "location_id",
            "level",
            "population",
            "urban_centre_share",
            "rural_share",
            "stage2",
        ]
    ]


def cluster_ids(
    data: UrbanicityData,
    version: str,
    spec: ClassificationSpecification,
    time_point: str,
    codes: tuple[npt.NDArray[np.uint8], int, int],
) -> dict[str, tuple[npt.NDArray[np.int32], int, int]]:
    """Each cluster type's ids, as rasters with the row and column of their first cell."""
    ids = {
        "urban_centre": read_window(
            data.urban_centre_raster_path(version, time_point), spec.grid
        )
    }
    urban_clusters = data.urban_cluster_raster_path(version, time_point)
    if urban_clusters.exists():
        ids["urban_cluster"] = read_window(urban_clusters, spec.grid)
    values, row_off, col_off = codes
    for cluster_type, (class_codes, contiguity) in clusters.cluster_types(
        spec.method.parameters
    ).items():
        labels, _ = rules.label(np.isin(values, class_codes), contiguity)
        ids[cluster_type] = (labels, row_off, col_off)
    return ids


def cluster_units(
    table: pd.DataFrame,
    ids: dict[str, tuple[npt.NDArray[np.int32], int, int]],
    parents: pd.DataFrame,
    iso3: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each cluster's people by unit, and its main country and admin 1."""
    rows, cols = table["row"].to_numpy(np.int64), table["col"].to_numpy(np.int64)
    frames = []
    for cluster_type, (labels, row_off, col_off) in ids.items():
        cluster_id = at_cells(labels, row_off, col_off, rows, cols, 0)
        found = cluster_id > 0
        frames.append(
            table.loc[found, ["location_id", "population"]]
            .assign(cluster_type=cluster_type, cluster_id=cluster_id[found])
            .groupby(["cluster_type", "cluster_id", "location_id"], as_index=False)[
                "population"
            ]
            .sum()
        )
    people = pd.concat(frames, ignore_index=True)

    with_parents = people.merge(parents, on="location_id", how="left")
    keys = ["cluster_type", "cluster_id"]
    total = with_parents.groupby(keys)["population"].sum().rename("population")
    main = [total]
    for column in ("country_id", "admin1_id"):
        by = with_parents.groupby([*keys, column])["population"].sum()
        top = by.sort_values(ascending=False).groupby(level=keys).head(1)
        top = top.reset_index(level=column)
        main.append(top[column])
        main.append(
            (top["population"] / total.reindex(top.index)).rename(
                f"{column[:-3]}_share"
            )
        )
    units = pd.concat(main, axis=1).reset_index()
    units["iso3"] = units["country_id"].map(iso3)
    return people, units


def check_pixels(
    data: UrbanicityData,
    spec: ClassificationSpecification,
    source: PopulaceSource,
    version: str,
    shapes: str,
    time_point: str,
    scheme: ClassScheme,
) -> float:
    """The largest relative gap, over a few blocks, between a join and the pixels.

    For each block, its people by (cell, unit) are joined to the class raster,
    and its pixels are tabulated directly through the pixel view. The two
    must agree by (unit, class).
    """
    plan = population.Plan(
        output_dir=str(data.root),
        grid=spec.grid.name,
        source=source,
        shapes=shapes,
        time_points=[time_point],
        blocks=list(CHECK_BLOCKS),
        region=None,
    )
    class_path = data.class_raster_path(version, time_point)
    codes, row_off, col_off = read_window(class_path, spec.grid)
    worst = 0.0
    for key in CHECK_BLOCKS:
        block = Block.from_key(key)
        pairs = population.tabulate_block(block, plan, data)
        cell = pairs["keys"] >> population.LOCATION_BITS
        joined = (
            pd.DataFrame(
                {
                    "location_id": pairs["keys"] & population.LOCATION_MASK,
                    "class_code": at_cells(
                        codes,
                        row_off,
                        col_off,
                        cell // spec.grid.n_cols,
                        cell % spec.grid.n_cols,
                        scheme.nodata,
                    ),
                    "population": pairs[f"population:{time_point}"],
                }
            )
            .groupby(["location_id", "class_code"])["population"]
            .sum()
        )

        pixel_codes = pixel_view.pixel_classes(
            block, spec.grid, class_path, scheme.nodata
        )
        people = populace.load_block(
            source, block, time_point, root=puc.POPULATION_MODEL_ROOT
        )
        units = population.read_units(data, shapes, block)
        rows = rows_on_grid(block, spec.grid)
        direct = (
            pd.DataFrame(
                {
                    "location_id": units[rows].ravel(),
                    "class_code": pixel_codes[rows].ravel(),
                    "population": people[rows].ravel().astype(np.float64),
                }
            )
            .groupby(["location_id", "class_code"])["population"]
            .sum()
        )
        gap = (joined - direct.reindex(joined.index, fill_value=0.0)).abs()
        scale = max(float(direct.sum()), 1.0)
        worst = max(worst, float(gap.sum()) / scale)
    return worst


def run(data: UrbanicityData, version: str, shapes: str) -> Path:
    """Tabulate a classification by the units of a shapes version."""
    root = data.classification_root(version)
    spec = ClassificationSpecification.from_yaml(root / "specification.yaml")
    scheme = ClassScheme.from_yaml(root / "scheme.yaml")
    if not isinstance(spec.population, PopulaceSource):
        msg = "Only a Populace classification has a population table."
        raise TypeError(msg)
    shapes_version = admin.SHAPES[shapes]
    parents = admin.unit_parents(shapes_version)
    iso3 = shapes_version.ihme_loc_ids()
    table_root = data.population_by_admin_root(spec.grid.name, spec.population, shapes)
    out_dir = root / "tabulations" / shapes
    mkdir(out_dir, parents=True, exist_ok=True)

    entries: dict[str, Any] = {}
    for tp in spec.time_points:
        table = pd.read_parquet(table_root / f"{tp}.parquet")
        codes = read_window(data.class_raster_path(version, tp), spec.grid)
        table["class_code"] = at_cells(
            *codes,
            table["row"].to_numpy(np.int64),
            table["col"].to_numpy(np.int64),
            scheme.nodata,
        )
        by_class = table.groupby(["location_id", "class_code"], as_index=False)[
            "population"
        ].sum()
        by_class.to_parquet(out_dir / f"{tp}.parquet", index=False)
        units = stage2(by_class, scheme, parents)
        if units is not None:
            units.to_parquet(out_dir / f"stage2_{tp}.parquet", index=False)
        people, cluster_main = cluster_units(
            table, cluster_ids(data, version, spec, tp, codes), parents, iso3
        )
        people.to_parquet(out_dir / f"cluster_people_{tp}.parquet", index=False)
        cluster_main.to_parquet(out_dir / f"cluster_units_{tp}.parquet", index=False)
        entries[tp] = {
            "people": float(by_class["population"].sum()),
            "people_in_no_unit": float(
                by_class.loc[
                    by_class["location_id"] == admin.NO_UNIT, "population"
                ].sum()
            ),
            "units": int(
                by_class.loc[
                    by_class["location_id"] != admin.NO_UNIT, "location_id"
                ].nunique()
            ),
        }

    first = spec.time_points[0]
    worst = check_pixels(data, spec, spec.population, version, shapes, first, scheme)
    if worst > CHECK_TOLERANCE:
        msg = f"A join and the pixels disagree by {worst:.2e} of the people checked."
        raise RuntimeError(msg)
    manifest = {
        "version": version,
        "shapes": shapes,
        "table": str(table_root),
        "created": datetime.datetime.now(tz=datetime.UTC).isoformat(timespec="seconds"),
        "package_version": package_version("rra-population-urbanicity"),
        "pixel_check": {
            "time_point": first,
            "blocks": list(CHECK_BLOCKS),
            "worst_relative_gap": worst,
        },
        "urban_clusters": "tabulated by unit through their id raster"
        if data.urban_cluster_raster_path(version, first).exists()
        else "not tabulated by unit: this classification does not save their ids",
        "time_points": entries,
    }
    manifest_path = out_dir / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    return manifest_path


@click.command()
@click.argument("version")
@click.option(
    "--shapes",
    type=click.Choice(sorted(admin.SHAPES)),
    required=True,
    help="The shapes version whose units to tabulate by.",
)
@with_output_directory(puc.URBANICITY_ROOT)
def classes_run(version: str, shapes: str, output_dir: str) -> None:
    """Tabulate a classification by admin unit, from the population table."""
    print(f"Done: {run(UrbanicityData(output_dir), version, shapes)}")
