"""Compare a classification's urban shares in the United States with the Census's.

The 2020 Census counted people inside and outside its urban areas, county by
county. Its definition is national: territory built up from census blocks of
enough housing units or impervious surface, in areas of at least 2,000 housing
units or 5,000 people. This diagnostic sets each county's urban share under the
classification's default cut, from its 2020q1 tabulation by LSAE 1578's admin
2 units, which in the United States are the counties, against the Census's
share. It needs only a cut, so it runs for any method.

Counties are matched by state and name: LSAE's short name is the Census's
county name. Counties either side has and the other lacks are listed.

Written under classifications/<version>/validation/us_census/:
- by_county.parquet: state, county, the Census's people and urban share, and
  the classification's;
- summary.yaml: the national and state urban shares under both, and how close
  the counties come, weighted by the Census's people.
"""

from pathlib import Path
from typing import Any

import click
import numpy as np
import pandas as pd
import yaml
from rra_tools.cli_tools import with_output_directory
from rra_tools.shell_tools import mkdir

from rra_population_urbanicity import admin
from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.classifiers.base import ClassScheme
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.reference import us_census

SHAPES = "lsae_1578_a2"
TIME_POINT = "2020q1"
COUNTRY = "USA"
# Bands of the gap in a county's urban share, in points, for the summary.
BANDS = (5, 10, 20)
LARGE_COUNTY = 250_000
LARGEST_GAPS = 10


def census_counties(path: Path) -> pd.DataFrame:
    """The Census's people and urban share by county."""
    table = pd.read_excel(
        path,
        sheet_name="2020_UA_COUNTY",
        usecols=["STATE_NAME", "COUNTY_NAME", "POP_COU", "POP_URB"],
    )
    return pd.DataFrame(
        {
            "state": table["STATE_NAME"],
            "county": table["COUNTY_NAME"],
            "census_population": table["POP_COU"].astype(float),
            "census_urban": table["POP_URB"].astype(float),
        }
    )


def our_counties(data: UrbanicityData, version: str, scheme: ClassScheme) -> pd.DataFrame:
    """The classification's people and urban people by US county."""
    hierarchy = pd.read_parquet(
        admin.SHAPES[SHAPES].hierarchy_path,
        columns=[
            "location_id",
            "parent_id",
            "location_name",
            "location_name_short",
            "ihme_loc_id",
            "most_detailed",
        ],
    )
    states = hierarchy.set_index("location_id")["location_name"]
    counties = hierarchy[
        (hierarchy["most_detailed"] == 1)
        & hierarchy["ihme_loc_id"].str.startswith(f"{COUNTRY}_")
    ]
    path = (
        data.classification_root(version)
        / "tabulations"
        / SHAPES
        / f"{TIME_POINT}.parquet"
    )
    table = pd.read_parquet(path)
    urban = [scheme.code(key) for key in scheme.default_cut]
    people = table.groupby("location_id")["population"].sum()
    urban_people = (
        table[table["class_code"].isin(urban)].groupby("location_id")["population"].sum()
    )
    return pd.DataFrame(
        {
            "state": counties["parent_id"].map(states).to_numpy(),
            "county": counties["location_name_short"].to_numpy(),
            "location_id": counties["location_id"].to_numpy(),
            "population": people.reindex(counties["location_id"], fill_value=0.0).to_numpy(),
            "urban": urban_people.reindex(counties["location_id"], fill_value=0.0).to_numpy(),
        }
    )


def summarise(table: pd.DataFrame) -> dict[str, Any]:
    matched = table.dropna(subset=["census_population", "population"])
    gap = (matched["urban_share"] - matched["census_urban_share"]) * 100
    weights = matched["census_population"]
    large = matched[weights >= LARGE_COUNTY]
    large_gap = gap[large.index]
    largest = large_gap.reindex(large_gap.abs().sort_values(ascending=False).index)
    states = matched.groupby("state")[
        ["census_population", "census_urban", "population", "urban"]
    ].sum()
    return {
        "counties": {
            "matched": int(len(matched)),
            "census_only": sorted(
                f"{r.county}, {r.state}"
                for r in table[table["population"].isna()].itertuples()
            ),
            "ours_only": sorted(
                f"{r.county}, {r.state}"
                for r in table[table["census_population"].isna()].itertuples()
            ),
        },
        "national_urban_share": {
            "census": float(states["census_urban"].sum() / states["census_population"].sum()),
            "ours": float(states["urban"].sum() / states["population"].sum()),
        },
        "state_urban_share": {
            str(state): [
                round(float(row["census_urban"] / row["census_population"]), 4),
                round(float(row["urban"] / row["population"]), 4),
            ]
            for state, row in states.iterrows()
        },
        "county_gap_points": {
            "mean_absolute": float(np.average(gap.abs(), weights=weights)),
            **{
                f"people_within_{band}": float(weights[gap.abs() <= band].sum() / weights.sum())
                for band in BANDS
            },
            "largest_in_large_counties": {
                f"{matched.loc[i, 'county']}, {matched.loc[i, 'state']}": round(float(v), 1)
                for i, v in largest[:LARGEST_GAPS].items()
            },
        },
    }


def compare(data: UrbanicityData, version: str) -> dict[str, Any]:
    root = data.classification_root(version)
    scheme = ClassScheme.from_yaml(root / "scheme.yaml")
    census = census_counties(us_census.county_table_path(data))
    ours = our_counties(data, version, scheme)
    table = census.merge(ours, on=["state", "county"], how="outer")
    table["census_urban_share"] = table["census_urban"] / table["census_population"]
    table["urban_share"] = table["urban"] / table["population"]
    out_dir = root / "validation" / "us_census"
    mkdir(out_dir, parents=True, exist_ok=True)
    table.to_parquet(out_dir / "by_county.parquet", index=False)
    summary = {"version": version, "time_point": TIME_POINT, **summarise(table)}
    (out_dir / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    return summary


@click.command()
@click.argument("version")
@with_output_directory(puc.URBANICITY_ROOT)
def validate_us_census(version: str, output_dir: str) -> None:
    """Compare a classification's US county urban shares with the 2020 Census."""
    summary = compare(UrbanicityData(output_dir), version)
    national = summary["national_urban_share"]
    gap = summary["county_gap_points"]
    print(
        f"United States: {national['ours']:.1%} urban against the Census's "
        f"{national['census']:.1%}; counties off by {gap['mean_absolute']:.1f} points "
        f"on average, {gap['people_within_10']:.0%} of people within 10"
    )
