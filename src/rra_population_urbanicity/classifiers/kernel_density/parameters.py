"""Parameters of the kernel density method: November's mask with its bugs removed.

A pixel of the 40 m model grid is urban if the people per km² of land in a disc
around it reach a threshold. Against rra-population-model's
`scripts/poverty.py` (results 2025_11_20.001), three things change:
- there is no noise floor: every person counts before the average;
- the average is people over land, not over all area, so water and missing
  pixels do not dilute a coastal neighbourhood;
- the disc is a disc on the ground, not in the cylindrical projection, where
  it would stretch east-west and shrink north-south away from the equator.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, PositiveFloat


class KernelDensityParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # The disc's radius on the ground, in metres.
    radius_m: PositiveFloat
    # People per km² of land at or above which a pixel is urban.
    threshold: PositiveFloat
    # Every pixel in the disc weighs the same.
    kernel: Literal["uniform"]


PARAMETER_SETS = {
    # November's radius and threshold.
    "kernel_default": KernelDensityParameters(
        radius_m=1000.0, threshold=300.0, kernel="uniform"
    ),
}
