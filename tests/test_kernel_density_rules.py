import math

import numpy as np
import pytest

from rra_population_urbanicity.classifiers.kernel_density import rules, scheme
from rra_population_urbanicity.classifiers.kernel_density.parameters import (
    PARAMETER_SETS,
)

PIXEL_KM2 = 0.04**2
PARAMS = PARAMETER_SETS["kernel_default"]


def test_the_disc_is_round_on_the_ground() -> None:
    equator = rules.disc_kernel(1000.0, 40.0, 0.0)
    assert equator.shape == (51, 51)
    assert equator[25, 0] == 1
    assert equator[0, 0] == 0
    # At 60° the grid stretches a metre east to two and shrinks a metre north
    # to half: the disc is 4 times as wide as it is tall, with the same area.
    north = rules.disc_kernel(1000.0, 40.0, 60.0)
    assert north.shape == (25, 101)
    for kernel in (equator, north):
        assert kernel.sum() == pytest.approx(math.pi * 25**2, rel=0.01)


def test_water_does_not_dilute_a_coastal_neighbourhood() -> None:
    # One person a pixel on land west of column 30; water east of it.
    population = np.zeros((60, 60), dtype=np.float32)
    population[:, :30] = 1.0
    land = (population > 0).astype(np.float32)
    kernel = rules.disc_kernel(400.0, 40.0, 0.0)
    smoothed = rules.density(population, land, kernel, PIXEL_KM2)
    # 625 people per km² of land on the coast, as inland; none at sea.
    assert smoothed[30, 29] == pytest.approx(625.0, rel=1e-5)
    assert smoothed[30, 10] == pytest.approx(625.0, rel=1e-5)
    assert smoothed[30, 45] == 0.0


def test_a_populated_pixel_counts_as_land() -> None:
    population = np.zeros((21, 21), dtype=np.float32)
    population[10, 10] = 16.0
    land = np.zeros_like(population)
    smoothed = rules.density(population, land, rules.disc_kernel(80.0, 40.0, 0.0), PIXEL_KM2)
    # The disc holds one pixel of land, the populated one.
    assert smoothed[10, 10] == pytest.approx(16.0 / PIXEL_KM2)
    assert smoothed[0, 0] == 0.0


def test_pixels_are_urban_rural_water_or_without_data() -> None:
    population = np.zeros((40, 40), dtype=np.float32)
    population[:, :20] = 0.5  # 312.5 people per km²
    population[:, 10:20] = 0.4  # 250 people per km²
    land = np.ones_like(population)
    land[:, 30:] = 0.0
    has_data = np.ones(population.shape, dtype=bool)
    has_data[0, 0] = False
    codes, smoothed = rules.classify(
        population, land, has_data, rules.disc_kernel(120.0, 40.0, 0.0), PARAMS, PIXEL_KM2
    )
    assert codes[20, 4] == scheme.URBAN
    assert smoothed[20, 4] == pytest.approx(312.5, rel=1e-5)
    assert codes[20, 15] == scheme.RURAL
    assert codes[20, 25] == scheme.RURAL  # empty land
    assert codes[20, 35] == scheme.WATER
    assert codes[0, 0] == scheme.NODATA
