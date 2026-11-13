from pathlib import Path
from typing import Annotated

from pydantic import StringConstraints

RRA_ROOT = Path("/mnt/team/rapidresponse/")
URBANICITY_ROOT = RRA_ROOT / "pub" / "population-urbanicity"

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
