"""Compare a classification's people by class and country with WUP 2025.

WUP 2025's Degree of Urbanisation figures come from GHSL's rules applied to
its own population grid, GHS-WUP-POP. So three sets of figures are compared,
all by country and level-2 class, for each time point:
- `population`: the classification's people, from its own class raster and
  population layer;
- `ghsl_population`: GHS-WUP-POP's people in GHS-WUP-DEGURBA's classes, on the
  same countries, for time points that are GHSL epochs;
- `wup_population`: WUP 2025's published figures, for 1 January of the year.
The first two are tabulated on the same countries, so the gap between them is
the classification's own. The gap between the last two is what WUP's country
borders and anything else in its tables add.

Countries are those of the units Populace is raked to. Their ISO3 codes match
WUP's, except that WUP has Hong Kong and Macao apart from China, and Kosovo
apart from Serbia; those are added to China and Serbia. A cell's country is
that of the unit containing its centre or, for a cell in no unit, of the
nearest unit (`admin.nearest_country_raster`). Places WUP does not list are
left out of the comparison of shares.

Written under classifications/<version>/validation/wup2025_shares/:
- by_country_<time_point>.parquet: iso3, class_code and the three populations;
- summary.yaml: for each time point, the world's shares by class and by level-1
  group under the three; the urban share under the scheme's default cut
  against WUP's urban share under national definitions; and the median gaps
  by country, with the largest.
"""

from collections.abc import Callable
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
from rra_population_urbanicity.classifiers.base import ClassScheme
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.grid import GridDefinition
from rra_population_urbanicity.inputs import grids
from rra_population_urbanicity.reference import wup
from rra_population_urbanicity.specification import (
    ClassificationSpecification,
    GhsWupPopSource,
)
from rra_population_urbanicity.validate.classification_diff import class_window
from rra_population_urbanicity.validate.ghs_smod_reproduction import smod_path

SHAPES = "raking_gbd_2023"
# WUP's places that the raking units hold inside another country.
WUP_MERGES = {"HKG": "CHN", "MAC": "CHN", "XKX": "SRB"}
WUP_TO_PEOPLE = 1000.0
STRIP_ROWS = 2048
# Countries this large, in WUP's figures, enter the median gaps.
MIN_COUNTRY_POPULATION = 5e6
LARGEST_GAPS = 10
SOURCES = ("population", "ghsl_population", "wup_population")
# The gaps reported by country: the classification against WUP's figures and
# against GHSL's classes on the same countries, and GHSL's against WUP's.
PAIRS = {
    "ours_vs_wup": ("population", "wup_population"),
    "ours_vs_ghsl": ("population", "ghsl_population"),
    "ghsl_vs_wup": ("ghsl_population", "wup_population"),
}
GHS_WUP_POP = GhsWupPopSource(source="ghs_wup_pop", release="r2025a")

type FloatArray = npt.NDArray[np.float64]


def tabulate(
    classes: Path,
    window: Window,
    population: Callable[[Window], FloatArray],
    countries: admin.LocationArray,
    country_ids: npt.NDArray[np.int64],
    codes: list[int],
) -> FloatArray:
    """People by (country, class), in the order of `country_ids` and `codes`.

    `classes` covers `window` of the grid; `population` reads a window of the
    grid. Cells with a code outside `codes`, or in no listed country, are left
    out.
    """
    index = np.full(2**16, -1, dtype=np.int64)
    index[codes] = np.arange(len(codes))
    size = max(int(countries.max()), int(country_ids.max())) + 1
    position = np.full(size, -1, dtype=np.int64)
    position[country_ids] = np.arange(country_ids.size)
    n = len(codes)
    totals = np.zeros(country_ids.size * n)
    with rasterio.open(classes) as src:
        for start in range(0, int(window.height), STRIP_ROWS):
            rows = min(STRIP_ROWS, int(window.height) - start)
            strip = Window(0, start, window.width, rows)
            grid_strip = Window(
                window.col_off, window.row_off + start, window.width, rows
            )
            where = position[countries[grid_strip.toslices()]]
            what = index[src.read(1, window=strip).astype(np.int64) % 2**16]
            counted = (where >= 0) & (what >= 0)
            totals += np.bincount(
                (where * n + what)[counted],
                weights=population(grid_strip)[counted],
                minlength=totals.size,
            )
    return totals.reshape(country_ids.size, n)


def wup_tables(data: UrbanicityData, year: int) -> tuple[pd.DataFrame, pd.Series]:
    """WUP's people by country and level-2 class, and its national urban shares."""
    level2 = pd.read_csv(
        wup.wup_path(data, "level2"), encoding="utf-8-sig", dtype={"ISO3_Code": str}
    )
    level2 = level2[
        (level2["LocTypeName"] == "Country/Area")
        & (level2["Year"] == year)
        & (level2["Category_order"] < 50)  # noqa: PLR2004 (50 is their total)
    ]
    classes = (
        level2.assign(
            iso3=level2["ISO3_Code"].replace(WUP_MERGES),
            class_code=level2["Category_order"],
            wup_population=level2["Pop_1Jan"] * WUP_TO_PEOPLE,
        )
        .groupby(["iso3", "class_code"])["wup_population"]
        .sum()
        .reset_index()
    )
    national = pd.read_csv(
        wup.wup_path(data, "national_definitions"),
        encoding="utf-8-sig",
        dtype={"ISO3_Code": str},
    )
    national = national[
        (national["LocTypeName"] == "Country/Area") & (national["Year"] == year)
    ]
    people = (
        national.assign(iso3=national["ISO3_Code"].replace(WUP_MERGES))
        .groupby(["iso3", "Category"])["Pop1Jan"]
        .sum()
        .unstack()
    )
    urban_share = (people["Urban"] / people["Total"]).rename("wup_national_urban_share")
    return classes, urban_share


def shares(table: pd.DataFrame, groups: dict[str, list[int]]) -> pd.DataFrame:
    """Each country's share of people in each group, for each source."""
    by_group = pd.concat(
        {
            group: table[table["class_code"].isin(codes)]
            .groupby("iso3")[list(SOURCES)]
            .sum()
            for group, codes in groups.items()
        },
        axis=1,
    )
    total = table.groupby("iso3")[list(SOURCES)].sum()
    result = by_group.div(pd.concat({g: total for g in groups}, axis=1))
    return result


def gaps(ours: pd.Series, theirs: pd.Series, weights: pd.Series) -> dict[str, Any]:
    """The median gap by country, in points, and the largest gaps."""
    gap = ((ours - theirs) * 100).dropna()
    large = gap[weights.reindex(gap.index) >= MIN_COUNTRY_POPULATION]
    largest = large.reindex(large.abs().sort_values(ascending=False).index)
    return {
        "countries": int(large.size),
        "median_absolute_gap": float(large.abs().median()) if large.size else None,
        "largest": {
            iso3: round(float(value), 2)
            for iso3, value in largest[:LARGEST_GAPS].items()
        },
    }


def summarise(
    table: pd.DataFrame,
    national_share: pd.Series,
    scheme: ClassScheme,
) -> dict[str, Any]:
    groups = {
        group: [scheme.code(key) for key in keys]
        for group, keys in scheme.groups["degurba_l1"].items()
        if group != "water"
    }
    urban = [scheme.code(key) for key in scheme.default_cut]
    # Only the countries WUP lists.
    compared = table[table["iso3"].isin(national_share.index)]
    world = compared.groupby("class_code")[list(SOURCES)].sum()
    total = world.sum()
    present = [s for s in SOURCES if total[s] > 0]
    by_group = shares(compared, groups)
    by_urban = shares(compared, {"urban": urban})[("urban", "population")]
    wup_people = compared.groupby("iso3")["wup_population"].sum()
    national = national_share.reindex(by_urban.index)
    return {
        "people": {s: float(total[s]) for s in SOURCES},
        "world_class_shares": {
            s: {int(c): float(v) for c, v in (world[s] / total[s]).items()}
            for s in present
        },
        "world_group_shares": {
            s: {
                g: float(world.loc[world.index.isin(c), s].sum() / total[s])
                for g, c in groups.items()
            }
            for s in present
        },
        "national_definitions": {
            "world_urban_share": float(
                world.loc[world.index.isin(urban), "population"].sum()
                / total["population"]
            ),
            "wup_world_urban_share": float(
                (national * wup_people).sum() / wup_people[national.notna()].sum()
            ),
            "gap_by_country": gaps(by_urban, national, wup_people),
        },
        "gaps_by_country": {
            group: {
                name: gaps(by_group[(group, a)], by_group[(group, b)], wup_people)
                for name, (a, b) in PAIRS.items()
            }
            for group in groups
        },
    }


def compare_time_point(
    data: UrbanicityData,
    spec: ClassificationSpecification,
    version: str,
    time_point: str,
    countries: admin.LocationArray,
    iso3: pd.Series,
    scheme: ClassScheme,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    grid = spec.grid
    codes = [c.code for c in scheme.classes]
    country_ids = np.asarray(iso3.index, dtype=np.int64)
    path = data.class_raster_path(version, time_point)
    ours = tabulate(
        path,
        class_window(path, spec),
        lambda w: grids.load(data, "population", spec.population, grid, time_point, w),
        countries,
        country_ids,
        codes,
    )
    reference = ghsl_classes(data, time_point)
    if reference is None:
        theirs = np.full_like(ours, np.nan)
    else:
        theirs = tabulate(
            reference,
            grid.full_window,
            lambda w: grids.load(data, "population", GHS_WUP_POP, grid, time_point, w),
            countries,
            country_ids,
            codes,
        )
    table = pd.DataFrame(
        {
            "iso3": np.repeat(iso3.to_numpy(), len(codes)),
            "class_code": np.tile(codes, country_ids.size),
            "population": ours.ravel(),
            "ghsl_population": theirs.ravel(),
        }
    )
    table = table.groupby(["iso3", "class_code"], as_index=False).sum(min_count=1)
    wup_classes, national_share = wup_tables(data, int(time_point[:4]))
    table = table.merge(wup_classes, on=["iso3", "class_code"], how="outer")
    table["wup_population"] = table["wup_population"].fillna(0.0)
    return table, summarise(table, national_share, scheme)


def country_layer(grid: GridDefinition) -> tuple[admin.LocationArray, pd.Series]:
    """Each cell's country, and each country's ISO3 code by location_id."""
    shapes = admin.SHAPES[SHAPES]
    countries = admin.nearest_country_raster(grid, shapes)
    return countries, shapes.ihme_loc_ids().reindex(shapes.country_ids())


def ghsl_classes(data: UrbanicityData, time_point: str) -> Path | None:
    """GHS-WUP-DEGURBA's classes at a time point, if it is a GHSL epoch."""
    try:
        return smod_path(data, "wup_r2025a", time_point)
    except ValueError:
        return None


def validate(data: UrbanicityData, version: str) -> dict[str, Any]:
    root = data.classification_root(version)
    spec = ClassificationSpecification.from_yaml(root / "specification.yaml")
    scheme = ClassScheme.from_yaml(root / "scheme.yaml")
    if "degurba_l1" not in scheme.groups:
        msg = f"{version} has no degurba_l1 group to compare with WUP 2025."
        raise ValueError(msg)
    countries, iso3 = country_layer(spec.grid)
    out_dir = root / "validation" / "wup2025_shares"
    mkdir(out_dir, parents=True, exist_ok=True)
    summary: dict[str, Any] = {"version": version, "shapes": SHAPES, "time_points": {}}
    for time_point in spec.time_points:
        table, result = compare_time_point(
            data, spec, version, time_point, countries, iso3, scheme
        )
        table.to_parquet(out_dir / f"by_country_{time_point}.parquet", index=False)
        summary["time_points"][time_point] = result
    (out_dir / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    return summary


@click.command()
@click.argument("version")
@with_output_directory(puc.URBANICITY_ROOT)
def validate_wup2025(version: str, output_dir: str) -> None:
    """Compare a classification's shares by class and country with WUP 2025."""
    summary = validate(UrbanicityData(output_dir), version)
    for time_point, result in summary["time_points"].items():
        groups = result["world_group_shares"]
        line = " | ".join(
            f"{source} "
            + " ".join(f"{group} {share:.1%}" for group, share in shares_.items())
            for source, shares_ in groups.items()
        )
        print(f"{time_point}: {line}")
