"""The US Census Bureau's 2020 urban areas, as TIGER/Line shapes.

The corrected file has the June 2023 updates for Atlanta-Gainesville and New
Orleans.
"""

from pathlib import Path

import click
from rra_tools.cli_tools import with_output_directory

from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.reference.download import download_archive

RELEASE = "2020"
URBAN_AREAS_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2020/UAC/tl_2020_us_uac20_corrected.zip"
)
URBAN_AREAS_FILE = "tl_2020_us_uac20_corrected.shp"


def urban_areas_path(data: UrbanicityData) -> Path:
    return data.reference_root("us_census", "urban_areas", RELEASE) / URBAN_AREAS_FILE


@click.command()
@with_output_directory(puc.URBANICITY_ROOT)
def download_us_census(output_dir: str) -> None:
    """Download the US Census Bureau's 2020 urban areas."""
    path = urban_areas_path(UrbanicityData(output_dir))
    print(download_archive(URBAN_AREAS_URL, path.parent, path.name))
