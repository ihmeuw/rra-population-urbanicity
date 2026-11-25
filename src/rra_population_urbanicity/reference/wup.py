"""World Urbanization Prospects 2025: the UN's country tables.

Three CSV files published on 2025-11-18: population by Degree of Urbanisation
at levels 1 and 2, and urban and rural population under national definitions,
by country and year. They were downloaded for rra-population-model, and are
copied from there.
"""

import shutil
from pathlib import Path

import click
from rra_tools.cli_tools import with_output_directory
from rra_tools.shell_tools import mkdir

from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.data import UrbanicityData

RELEASE = "2025"
SOURCE_DIR = puc.POPULATION_MODEL_ROOT / "comparison-inputs" / "wup"
FILES = {
    "level1": "WUP2025-DB-DEGURBA-Level1-Population-Surface-Data.csv",
    "level2": "WUP2025-DB-DEGURBA-Level2-Population-Surface-Data.csv",
    "national_definitions": "WUP2025-DB-National-Definitions-Population-Data.csv",
}


def wup_path(data: UrbanicityData, table: str) -> Path:
    return data.reference_root("un_desa", "wup", RELEASE) / FILES[table]


def copy_tables(data: UrbanicityData, source_dir: Path = SOURCE_DIR) -> list[Path]:
    """Copy the tables that are not yet in the reference directory."""
    paths = []
    for table, name in FILES.items():
        path = wup_path(data, table)
        if not path.exists():
            mkdir(path.parent, parents=True, exist_ok=True)
            shutil.copyfile(source_dir / name, path)
        paths.append(path)
    return paths


@click.command()
@with_output_directory(puc.URBANICITY_ROOT)
def copy_wup(output_dir: str) -> None:
    """Copy WUP 2025's country tables into the reference directory."""
    for path in copy_tables(UrbanicityData(output_dir)):
        print(path)
