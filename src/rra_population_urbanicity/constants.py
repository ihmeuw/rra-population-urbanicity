from pathlib import Path
from typing import Annotated

from pydantic import StringConstraints

RRA_ROOT = Path("/mnt/team/rapidresponse/")
URBANICITY_ROOT = RRA_ROOT / "pub" / "population-urbanicity"
# The outputs of rra-population-model and rra-building-density, read as inputs.
POPULATION_MODEL_ROOT = RRA_ROOT / "pub" / "population-model"
BUILDING_DENSITY_ROOT = RRA_ROOT / "pub" / "building-density"
# RT's urban and rural poverty rates by LSAE 1578 admin 2 (2020-2023): the
# deliverable flags the units they have no row for.
RT_RATES_PATH = Path(
    "/mnt/share/geospatial/lsae/economic_indicators/ldi/output/20251022_090128/"
    "raked_admin2_urban_rural_poverty.csv"
)

CRSES = {
    # World Mollweide: the GHSL grid, and the default grid for classification.
    "mollweide": "ESRI:54009",
    # World Cylindrical Equal Area: the grid of the Populace surface.
    "equal_area": "ESRI:54034",
}

# Every source labels its time points YYYYqN. GHSL's five-year epochs are
# labelled with their first quarter (2020q1, 2025q1), as in rra-population-model.
TIME_POINT_PATTERN = r"^\d{4}q[1-4]$"
TimePoint = Annotated[str, StringConstraints(pattern=TIME_POINT_PATTERN)]

# The time point of layers that do not vary in time.
STATIC = "static"

# Regions for development runs, as (west, south, east, north) in degrees. Each
# extends about half a degree past the country, so clusters near its border
# are whole.
REGIONS = {
    # Belgium, as in the flexurba vignette.
    "belgium": (2.0, 49.0, 7.0, 52.0),
    # Bangladesh: Dhaka, the Sylhet haors and the Jamuna chars.
    "bangladesh": (87.5, 20.1, 93.2, 27.1),
}
