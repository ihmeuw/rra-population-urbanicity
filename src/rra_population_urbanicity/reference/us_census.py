"""The US Census Bureau's 2020 urban areas: their shapes, and people by county.

The shapes are TIGER/Line's corrected file, with the June 2023 updates for
Atlanta-Gainesville and New Orleans. The county table has each county's people
inside and outside urban areas.
"""

from pathlib import Path

import click
from rra_tools.cli_tools import with_output_directory
from rra_tools.shell_tools import mkdir

from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.reference.download import download_archive, fetch

RELEASE = "2020"
URBAN_AREAS_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2020/UAC/tl_2020_us_uac20_corrected.zip"
)
URBAN_AREAS_FILE = "tl_2020_us_uac20_corrected.shp"
COUNTY_TABLE_URL = "https://www2.census.gov/geo/docs/reference/ua/2020_UA_COUNTY.xlsx"


def urban_areas_path(data: UrbanicityData) -> Path:
    return data.reference_root("us_census", "urban_areas", RELEASE) / URBAN_AREAS_FILE


def county_table_path(data: UrbanicityData) -> Path:
    return urban_areas_path(data).parent / COUNTY_TABLE_URL.rsplit("/", 1)[-1]


@click.command()
@with_output_directory(puc.URBANICITY_ROOT)
def download_us_census(output_dir: str) -> None:
    """Download the US Census Bureau's 2020 urban areas and county table."""
    data = UrbanicityData(output_dir)
    path = urban_areas_path(data)
    print(download_archive(URBAN_AREAS_URL, path.parent, path.name))
    table = county_table_path(data)
    if not table.exists():
        mkdir(table.parent, parents=True, exist_ok=True)
        fetch(COUNTY_TABLE_URL, table)
    print(table)
