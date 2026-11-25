"""The kernel density rule on a block of the 40 m model grid.

The model grid is cylindrical equal-area (ESRI:54034). At latitude φ a metre
east on the ground is 1/cos φ metres on the grid and a metre north is cos φ, so
a disc on the ground is an ellipse on the grid, with the same area. A block's
kernel is drawn at the latitude of the block's centre.
"""

import math

import numpy as np
import numpy.typing as npt
from scipy import signal

from rra_population_urbanicity.classifiers.kernel_density.parameters import (
    KernelDensityParameters,
)
from rra_population_urbanicity.classifiers.kernel_density.scheme import (
    NODATA,
    RURAL,
    URBAN,
    WATER,
)

type FloatArray = npt.NDArray[np.float32]
type CodeArray = npt.NDArray[np.uint8]

# A pixel with less land than this, and nobody in it, is water.
WATER_LAND_MAX = 0.5
EDGE = 1e-9


def disc_kernel(radius_m: float, pixel_m: float, latitude: float) -> FloatArray:
    """A disc of the radius on the ground, in pixels of the grid at the latitude.

    A pixel is in the disc if its centre is. The kernel has an odd number of
    rows and columns, centred on the middle one.
    """
    scale = math.cos(math.radians(latitude))
    half_x, half_y = radius_m / scale / pixel_m, radius_m * scale / pixel_m
    # A tolerance keeps a pixel exactly on the edge, such as at 60°, where
    # cos φ is a hair over 0.5.
    n_x, n_y = math.floor(half_x + EDGE), math.floor(half_y + EDGE)
    x = np.arange(-n_x, n_x + 1) / half_x
    y = np.arange(-n_y, n_y + 1) / half_y
    inside = x[None, :] ** 2 + y[:, None] ** 2 <= 1 + EDGE
    kernel: FloatArray = inside.astype(np.float32)
    return kernel


def density(
    population: FloatArray,
    land: FloatArray,
    kernel: FloatArray,
    pixel_area_km2: float,
) -> FloatArray:
    """People per km² of land in the disc around each pixel.

    A pixel with people in it counts as all land, as a populated cell is on
    land in DEGURBA. Where the disc holds no land the density is 0.
    """
    land = np.where(population > 0, np.float32(1), land)
    people = signal.oaconvolve(population, kernel, mode="same")
    area = signal.oaconvolve(land, kernel, mode="same") * pixel_area_km2
    # The convolution leaves rounding noise where both are zero.
    result: FloatArray = np.divide(
        people, area, out=np.zeros_like(people), where=area > pixel_area_km2 / 2
    ).astype(np.float32)
    return result


def classify(
    population: FloatArray,
    land: FloatArray,
    has_data: npt.NDArray[np.bool_],
    kernel: FloatArray,
    params: KernelDensityParameters,
    pixel_area_km2: float,
) -> tuple[CodeArray, FloatArray]:
    """Each pixel's class, and the density it was classified on."""
    smoothed = density(population, land, kernel, pixel_area_km2)
    codes = np.where(smoothed >= params.threshold, URBAN, RURAL).astype(np.uint8)
    codes[(population <= 0) & (land < WATER_LAND_MAX)] = WATER
    codes[~has_data] = NODATA
    return codes, smoothed
