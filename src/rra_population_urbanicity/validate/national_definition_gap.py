"""How far a classification's urban share is from each country's own definition.

RT's national urban/rural poverty ratios were estimated with each country's
own definition of urban, and reconciled with WUP's urban shares under national
definitions. This diagnostic sets a classification's urban share under its
scheme's default cut, country by country, against WUP 2025's share under
national definitions for 1 January of the year. It needs only a cut, so it
runs for any method: it reads the classification's tabulation by the units
Populace is raked to, which a grid method gets from the joins and a pixel
method from its own run. (The share implied by RT's own rates waits for RT's
figures.)

Countries follow the WUP comparison: the raking units' countries, with WUP's
Hong Kong and Macao in China and Kosovo in Serbia.

Written under classifications/<version>/validation/national_definition_gap/:
- by_country.parquet: time_point, iso3, population, urban_share and
  wup_national_urban_share;
- summary.yaml: by time point, the world's urban share under both, and the
  median and largest gaps by country.
"""

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
from rra_population_urbanicity.specification import ClassificationSpecification
from rra_population_urbanicity.validate.wup2025_shares import (
    SHAPES,
    gaps,
    wup_tables,
)


def by_country(
    data: UrbanicityData, version: str, time_point: str, scheme: ClassScheme
) -> pd.DataFrame:
    """Each country's people and urban share under the default cut."""
    path = data.classification_root(version) / "tabulations" / SHAPES / f"{time_point}.parquet"
    table = pd.read_parquet(path)
    shapes = admin.SHAPES[SHAPES]
    country_of = admin.unit_parents(shapes).set_index("location_id")["country_id"]
    iso3 = shapes.ihme_loc_ids()
    urban = [scheme.code(key) for key in scheme.default_cut]
    table = table.assign(
        iso3=table["location_id"].map(country_of).map(iso3),
        urban=table["population"].where(table["class_code"].isin(urban), 0.0),
    )
    countries = table.groupby("iso3")[["population", "urban"]].sum()
    return countries.assign(urban_share=countries["urban"] / countries["population"])


def compare(data: UrbanicityData, version: str) -> dict[str, Any]:
    root = data.classification_root(version)
    spec = ClassificationSpecification.from_yaml(root / "specification.yaml")
    scheme = ClassScheme.from_yaml(root / "scheme.yaml")
    frames = []
    summary: dict[str, Any] = {"version": version, "shapes": SHAPES, "time_points": {}}
    for time_point in spec.time_points:
        countries = by_country(data, version, time_point, scheme)
        classes, national = wup_tables(data, int(time_point[:4]))
        wup_people = classes.groupby("iso3")["wup_population"].sum()
        compared = countries.join(national, how="inner")
        weights = wup_people.reindex(compared.index)
        summary["time_points"][time_point] = {
            "world_urban_share": float(
                compared["urban"].sum() / compared["population"].sum()
            ),
            "wup_world_urban_share": float(
                np.average(compared["wup_national_urban_share"], weights=weights)
            ),
            "gap_by_country": gaps(
                compared["urban_share"], compared["wup_national_urban_share"], weights
            ),
        }
        frames.append(
            compared.reset_index()[
                ["iso3", "population", "urban_share", "wup_national_urban_share"]
            ].assign(time_point=time_point)
        )
    out_dir = root / "validation" / "national_definition_gap"
    mkdir(out_dir, parents=True, exist_ok=True)
    pd.concat(frames).to_parquet(out_dir / "by_country.parquet", index=False)
    (out_dir / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    return summary


@click.command()
@click.argument("version")
@with_output_directory(puc.URBANICITY_ROOT)
def validate_national_gap(version: str, output_dir: str) -> None:
    """Compare a classification's urban shares with national definitions (WUP)."""
    summary = compare(UrbanicityData(output_dir), version)
    for time_point, result in summary["time_points"].items():
        gap = result["gap_by_country"]
        print(
            f"{time_point}: urban {result['world_urban_share']:.1%} against "
            f"{result['wup_world_urban_share']:.1%} by national definitions; "
            f"median gap {gap['median_absolute_gap']:.1f} points in "
            f"{gap['countries']} countries"
        )
