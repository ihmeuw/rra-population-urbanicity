"""The deliverable for the poverty work: urban and rural people by LSAE admin 2.

For each time point of a classification, from its tabulation by LSAE 1578's
admin 2 units, every admin 2 unit with its people, its urban and rural people
under the scheme's default cut, and its urban share. Units with no people
have zeros and no share. Units for which RT's poverty rates have no row are
flagged, not dropped. People in cells the classification gives no class are
counted in a unit's people and in neither of its parts.

Written to classifications/<version>/deliverables/ as CSV and parquet.
"""

from pathlib import Path

import click
import pandas as pd
from rra_tools.cli_tools import with_output_directory
from rra_tools.shell_tools import mkdir

from rra_population_urbanicity import admin
from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.classifiers.base import ClassScheme
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.specification import ClassificationSpecification

SHAPES = "lsae_1578_a2"
NAME = "admin2_urban_rural_lsae_1578"


def units(version: admin.ShapesVersion) -> pd.DataFrame:
    """The shapes version's units: the most detailed locations of its hierarchy."""
    hierarchy = pd.read_parquet(
        version.hierarchy_path,
        columns=["location_id", "ihme_loc_id", "location_name", "most_detailed"],
    )
    detailed = hierarchy[hierarchy["most_detailed"] == 1]
    return detailed.drop(columns="most_detailed").reset_index(drop=True)


def deliverable(data: UrbanicityData, version: str) -> pd.DataFrame:
    root = data.classification_root(version)
    spec = ClassificationSpecification.from_yaml(root / "specification.yaml")
    scheme = ClassScheme.from_yaml(root / "scheme.yaml")
    urban = [scheme.code(key) for key in scheme.default_cut]
    classified = [c.code for c in scheme.classes]
    all_units = units(admin.SHAPES[SHAPES])
    rated = set(pd.read_csv(puc.RT_RATES_PATH, usecols=["location_id"])["location_id"])

    frames = []
    for tp in spec.time_points:
        by_class = pd.read_parquet(root / "tabulations" / SHAPES / f"{tp}.parquet")
        by_class = by_class[by_class["location_id"] != admin.NO_UNIT]
        people = by_class.groupby("location_id")["population"].sum()
        in_urban = by_class["class_code"].isin(urban)
        in_rural = by_class["class_code"].isin(classified) & ~in_urban
        frame = all_units.assign(
            time_point=tp,
            population=all_units["location_id"].map(people).fillna(0.0),
            urban_population=all_units["location_id"]
            .map(by_class[in_urban].groupby("location_id")["population"].sum())
            .fillna(0.0),
            rural_population=all_units["location_id"]
            .map(by_class[in_rural].groupby("location_id")["population"].sum())
            .fillna(0.0),
        )
        frame["urban_share"] = frame["urban_population"] / frame["population"].where(
            frame["population"] > 0
        )
        frame["has_rt_rates"] = frame["location_id"].isin(rated)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def run(data: UrbanicityData, version: str) -> Path:
    table = deliverable(data, version)
    out_dir = data.classification_root(version) / "deliverables"
    mkdir(out_dir, parents=True, exist_ok=True)
    table.to_parquet(out_dir / f"{NAME}.parquet", index=False)
    table.to_csv(out_dir / f"{NAME}.csv", index=False)
    return out_dir / f"{NAME}.csv"


@click.command()
@click.argument("version")
@with_output_directory(puc.URBANICITY_ROOT)
def deliverable_run(version: str, output_dir: str) -> None:
    """Write the urban and rural people of each LSAE admin 2, for RT."""
    print(f"Done: {run(UrbanicityData(output_dir), version)}")
