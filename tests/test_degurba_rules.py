from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from rra_population_urbanicity.classify.degurba import rules, scheme
from rra_population_urbanicity.classify.degurba.parameters import (
    GHSL_R2023A,
    DegurbaParameters,
)

type Array = npt.NDArray[np.float64]

# The V2.0 rules without the built-up criterion or the density-on-land rule,
# so a scene needs only population and land. Cells are 1 km², so with all land
# a cell's population is its density.
BASE = GHSL_R2023A.model_copy(
    update={
        "uc_built_criterion": False,
        "duc_built_criterion": False,
        "density_on_land": "none",
    }
)


def params(**overrides: Any) -> DegurbaParameters:
    return BASE.model_copy(update=overrides)


def scene(
    population: Array,
    land: Array | None = None,
    built: Array | None = None,
    domain: npt.NDArray[np.bool_] | None = None,
) -> rules.DegurbaInputs:
    return rules.DegurbaInputs(
        population=population,
        land=np.ones_like(population) if land is None else land,
        built=np.zeros_like(population) if built is None else built,
        domain=np.ones(population.shape, dtype=bool) if domain is None else domain,
        cell_area_km2=1.0,
    )


def classify(
    population: Array,
    land: Array | None = None,
    built: Array | None = None,
    *,
    built_threshold: float | None = None,
    **overrides: Any,
) -> npt.NDArray[np.uint8]:
    return rules.classify(
        scene(population, land, built),
        params(**overrides),
        built_threshold=built_threshold,
    )


def test_lower_bounds_are_inclusive_or_strict() -> None:
    values = np.array([1499.0, 1500.0, 1501.0])
    assert rules.passes(values, 1500, strict=False).tolist() == [False, True, True]
    assert rules.passes(values, 1500, strict=True).tolist() == [False, False, True]


def test_an_urban_centre_needs_50000_people() -> None:
    population = np.zeros((9, 9))
    population[2:7, 2:7] = 2000  # 25 cells, 50,000 people
    assert (classify(population)[2:7, 2:7] == scheme.URBAN_CENTRE).all()
    strict = classify(population, strict_thresholds=True)
    assert (strict[2:7, 2:7] == scheme.DENSE_URBAN_CLUSTER).all()
    population[2, 2] = 1960  # 49,960 people
    assert (classify(population)[2:7, 2:7] == scheme.DENSE_URBAN_CLUSTER).all()


def test_a_cell_at_exactly_the_density_threshold() -> None:
    population = np.zeros((9, 9))
    population[2:7, 2:7] = 2040
    population[2, 2] = 1500  # 50,460 people with this cell, 48,960 without
    assert classify(population)[2, 2] == scheme.URBAN_CENTRE
    strict = classify(population, strict_thresholds=True)
    assert strict[2, 2] == scheme.SUBURBAN
    assert strict[4, 4] == scheme.DENSE_URBAN_CLUSTER


@pytest.mark.parametrize(("people", "expected"), [(5001, 23), (4999, 13)])
def test_a_dense_urban_cluster_needs_5000_people(people: int, expected: int) -> None:
    population = np.zeros((5, 7))
    population[2, 2:5] = [1700, 1700, people - 3400]
    assert (classify(population)[2, 2:5] == expected).all()


def test_dense_areas_touching_at_a_corner_are_separate_under_rook() -> None:
    population = np.zeros((12, 12))
    population[1:5, 1:5] = 2000  # 32,000 people
    population[5:9, 5:9] = 2000  # 32,000 people, touching the first at a corner
    rook = classify(population)
    assert (rook[1:5, 1:5] == scheme.DENSE_URBAN_CLUSTER).all()
    assert (rook[5:9, 5:9] == scheme.DENSE_URBAN_CLUSTER).all()
    queen = classify(population, uc_contiguity=8)
    assert (queen[1:5, 1:5] == scheme.URBAN_CENTRE).all()
    assert (queen[5:9, 5:9] == scheme.URBAN_CENTRE).all()


def test_a_diagonal_string_of_dense_cells_is_only_an_urban_cluster() -> None:
    population = np.zeros((32, 32))
    diagonal = np.arange(1, 31)
    population[diagonal, diagonal] = 2000  # 60,000 people, no two sharing an edge
    assert (classify(population)[diagonal, diagonal] == scheme.SUBURBAN).all()


def test_a_river_splits_an_urban_centre() -> None:
    population = np.full((4, 11), 2000.0)  # 88,000 people
    assert (classify(population) == scheme.URBAN_CENTRE).all()
    land = np.ones_like(population)
    population[:, 5] = 0
    land[:, 5] = 0  # a river one cell wide, leaving 40,000 people on each bank
    codes = classify(population, land)
    assert (codes[:, 5] == scheme.WATER).all()
    assert (codes[:, :5] == scheme.DENSE_URBAN_CLUSTER).all()
    assert (codes[:, 6:] == scheme.DENSE_URBAN_CLUSTER).all()


@pytest.mark.parametrize(("hole", "filled"), [((2, 7), True), ((3, 5), False)])
def test_gaps_under_15_km2_are_filled(hole: tuple[int, int], filled: bool) -> None:
    population = np.zeros((13, 13))
    population[1:12, 1:12] = 2000
    land = np.ones_like(population)
    lake = np.s_[3 : 3 + hole[0], 3 : 3 + hole[1]]  # 14 or 15 cells of water
    population[lake] = 0
    land[lake] = 0
    expected = scheme.URBAN_CENTRE if filled else scheme.WATER
    assert (classify(population, land)[lake] == expected).all()


def test_a_hole_enclosed_only_through_corners() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[[1, 2, 2, 3], [2, 1, 3, 2]] = True  # a diamond whose cells touch at corners
    assert not rules.fill_gaps(mask, 15, 4)[2, 2]  # not a hole under rook contiguity
    assert rules.fill_gaps(mask, 15, 8)[2, 2]


@pytest.mark.parametrize(("other_land", "joins"), [(0.0, True), (1.0, False)])
def test_the_majority_rule_counts_a_draw_as_urban(
    other_land: float, joins: bool
) -> None:
    population = np.zeros((10, 12))
    population[:6, :10] = 2000  # an urban centre of 120,000 people
    land = np.ones_like(population)
    land[6] = 0  # a row of water below it, except for two cells
    land[6, 5] = 1
    land[6, 4] = other_land
    land[7, 6] = 0
    codes = classify(population, land)
    # The window of (6, 5) has six eligible cells, three of them urban: a draw.
    # If (6, 4) is land, there are seven, and three is not a majority.
    expected = scheme.URBAN_CENTRE if joins else scheme.VERY_LOW_DENSITY_RURAL
    assert codes[6, 5] == expected
    assert codes[6, 6] == scheme.WATER  # water is never added


def test_smoothing_fills_a_concave_corner_without_spreading() -> None:
    population = np.zeros((12, 12))
    population[:6, :10] = 2000
    population[6:10, :4] = 2000  # an L-shaped urban centre of 152,000 people
    codes = classify(population)
    # The inner corner has five urban cells in its window of nine and joins.
    # Its neighbours along each edge then have four of nine, not a majority.
    assert codes[6, 4] == scheme.URBAN_CENTRE
    assert codes[6, 5] != scheme.URBAN_CENTRE
    assert codes[7, 4] != scheme.URBAN_CENTRE


def test_each_urban_centre_is_smoothed_on_its_own() -> None:
    seeds = np.zeros((7, 9), dtype=bool)
    seeds[2:5, 2:4] = True  # two centres, one column apart
    seeds[2:5, 5:7] = True
    eligible = np.ones_like(seeds)
    # Counted together, (3, 4) has six urban cells in its window of nine and
    # would join them; each centre alone has three.
    centres = rules.generalise_centres(seeds, eligible, BASE, 1.0)
    np.testing.assert_array_equal(centres, seeds)


def test_a_centre_that_grows_to_its_window_edge_is_redone_in_a_wider_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rules, "CENTRE_PAD", 1)
    seeds = np.zeros((10, 12), dtype=bool)
    seeds[:6, :10] = True
    eligible = np.ones_like(seeds)
    eligible[6] = False  # a row of water below the centre, except for one cell
    eligible[6, 5] = True
    eligible[7, 6] = False
    # (6, 5) joins by a draw, on the last row of the first window.
    centres = rules.generalise_centres(seeds, eligible, BASE, 1.0)
    assert centres[6, 5]
    assert centres.sum() == seeds.sum() + 1


def test_a_dense_urban_cluster_cell_below_300_does_not_join_an_urban_cluster() -> None:
    population = np.zeros((8, 10))
    built = np.zeros_like(population)
    population[1:3, 1:4] = 1000  # a dense urban cluster of 7,000 people
    population[1, 1] = 2000
    built[1:3, 1:4] = 0.6
    population[2, 4] = 100  # built-up only, 100/km²: in the cluster but not at 300
    built[2, 4] = 0.6
    hamlet = np.s_[2:4, 5:7]  # 1,600 people at 400/km², touching only (2, 4)
    population[hamlet] = 400
    codes = classify(
        population, built=built, built_threshold=0.5, duc_built_criterion=True
    )
    assert codes[2, 4] == scheme.DENSE_URBAN_CLUSTER
    assert (codes[hamlet] == scheme.RURAL_CLUSTER).all()


@pytest.mark.parametrize(
    ("order", "filled"), [("smooth_then_fill", True), ("fill_then_smooth", False)]
)
def test_the_order_of_smoothing_and_gap_filling(order: str, filled: bool) -> None:
    population = np.zeros((13, 13))
    population[1:12, 1:12] = 2000
    land = np.ones_like(population)
    gap = np.s_[3:6, 3:8]  # 15 cells: 14 of water, and one of land in a corner
    population[gap] = 0
    land[gap] = 0
    land[3, 3] = 1
    codes = classify(population, land, uc_morphology_order=order)
    assert codes[3, 3] == scheme.URBAN_CENTRE  # smoothing adds the land cell either way
    assert (codes[gap] == scheme.URBAN_CENTRE).all() == filled


@pytest.mark.parametrize(("require", "expected"), [(True, 22), (False, 30)])
def test_a_built_up_only_urban_centre(require: bool, expected: int) -> None:
    population = np.zeros((10, 10))
    population[1:9, 1:9] = 1000  # 64,000 people, no cell at 1,500/km²
    built = np.zeros_like(population)
    built[1:9, 1:9] = 0.6
    codes = classify(
        population,
        built=built,
        built_threshold=0.5,
        uc_built_criterion=True,
        duc_built_criterion=True,
        uc_require_dense_cell=require,
        duc_require_dense_cell=require,
    )
    assert (codes[1:9, 1:9] == expected).all()


@pytest.mark.parametrize(("require", "expected"), [(True, 22), (False, 23)])
def test_a_built_up_only_dense_urban_cluster(require: bool, expected: int) -> None:
    population = np.zeros((7, 7))
    population[2:5, 2:5] = 1000  # 9,000 people, no cell at 1,500/km²
    built = np.zeros_like(population)
    built[2:5, 2:5] = 0.6
    codes = classify(
        population,
        built=built,
        built_threshold=0.5,
        duc_built_criterion=True,
        duc_require_dense_cell=require,
    )
    assert (codes[2:5, 2:5] == expected).all()


@pytest.mark.parametrize(("column", "expected"), [(11, 21), (12, 22)])
def test_semi_dense_cells_within_2_km_of_a_centre(column: int, expected: int) -> None:
    population = np.zeros((10, 16))
    population[:, :10] = 2000  # an urban centre in columns 0-9
    population[3:6, 10:column] = 400  # linking cells, in the same urban cluster
    population[3:6, column] = 1000  # 3,000 semi-dense people, 2 or 3 cells away
    assert (classify(population)[3:6, column] == expected).all()


@pytest.mark.parametrize(
    ("buffer_test", "columns"),
    [
        ("remove_then_cluster", []),
        ("cluster_then_remove", [1, 2, 3, 9, 10, 11]),
        ("drop_cluster", []),
    ],
)
def test_the_class_22_buffer_readings(buffer_test: str, columns: list[int]) -> None:
    shape = (5, 13)
    urban = np.zeros(shape, dtype=bool)
    urban[0, 6] = True  # its buffer is rows 0-2, columns 4-8
    population = np.zeros(shape)
    # A row of semi-dense cells that the buffer cuts in two pieces of 2,400
    # people, and a separate cluster of 3,200 people outside the buffer.
    population[2, 1:12] = 800
    population[4, :4] = 800
    selected = rules.semi_dense(
        population,
        population / 0.8,
        urban,
        np.ones(shape, dtype=bool),
        params(sduc_buffer_test=buffer_test),
    )
    assert np.nonzero(selected[2])[0].tolist() == columns
    assert selected[4, :4].all()


@pytest.mark.parametrize(("within", "expected"), [(True, 13), (False, 22)])
def test_class_22_outside_an_urban_cluster(within: bool, expected: int) -> None:
    population = np.zeros((3, 6))
    population[1, 1:4] = 1000  # 3,000 people: semi-dense, but not an urban cluster
    codes = classify(population, sduc_within_urban_cluster=within)
    assert (codes[1, 1:4] == expected).all()


def test_rural_classes() -> None:
    population = np.zeros((3, 9))
    population[1, [1, 3, 5, 7]] = [500, 499, 50, 49]
    assert classify(population)[1, [1, 3, 5, 7]].tolist() == [13, 12, 12, 11]


def test_built_up_on_water_counts_as_land_only_when_it_reaches_land() -> None:
    land = np.zeros((3, 7))
    land[1, 0] = 1.0
    built = np.zeros_like(land)
    built[1, :2] = 0.2  # a pier, built from the shore
    built[1, 5] = 0.2  # built-up in open water, reaching no land
    population = np.zeros_like(land)
    codes = classify(population, land, built)
    assert codes[1, 1] == scheme.VERY_LOW_DENSITY_RURAL
    assert codes[1, 5] == scheme.WATER
    assert classify(population, land)[1, 1] == scheme.WATER  # with no built-up layer


@pytest.mark.parametrize(
    ("rule", "expected"),
    [
        ("none", [0.0, 0.0, 0.0, 0.1]),
        ("built_up", [0.5, 0.7, 0.0, 0.1]),
        ("built_up_and_population", [0.5, 0.7, 0.5, 0.1]),
    ],
)
def test_density_on_land_readings(rule: str, expected: list[float]) -> None:
    # Only cells with no land are raised; a cell with a little land keeps it.
    population = np.full((1, 4), 200.0)
    land = np.array([[0.0, 0.0, 0.0, 0.1]])
    built = np.array([[0.2, 0.7, 0.0, 0.3]])
    land_for_density = rules.land_for_density(
        scene(population, land, built),
        built,
        params(density_on_land=rule),
        built_per_person=1e-4,
    )
    np.testing.assert_allclose(land_for_density[0], expected)


def test_built_up_per_person() -> None:
    assert (
        rules.built_per_person(np.array([100.0, 300.0]), np.array([0.1, 0.3])) == 1e-3
    )


@pytest.mark.parametrize(
    ("average", "expected"),
    [("cell_mean_land", 0.42), ("area_ratio", 10 / 24), ("cell_mean_total", 0.4)],
)
def test_the_optimal_built_up_threshold(average: str, expected: float) -> None:
    population = np.zeros((7, 12))
    population[1:6, 1:6] = 2000  # 50,000 people
    land = np.ones_like(population)
    land[1, 1:6] = 0.8
    built = np.zeros_like(population)
    built[1:6, 1:6] = 0.4
    population[2:4, 8:10] = 2000  # a smaller dense cluster, which does not count
    built[2:4, 8:10] = 0.9
    prepared = rules.prepare(scene(population, land, built), BASE, None)
    threshold = rules.optimal_built_threshold(
        population, prepared, params(built_threshold_average=average)
    )
    assert threshold == pytest.approx(expected)


def test_the_optimal_threshold_needs_a_large_cluster() -> None:
    population = np.full((2, 2), 2000.0)
    prepared = rules.prepare(scene(population), BASE, None)
    with pytest.raises(ValueError, match="No cluster is large enough"):
        rules.optimal_built_threshold(population, prepared, BASE)


def test_clusters_do_not_join_across_the_edges_of_the_array() -> None:
    population = np.zeros((5, 10))
    population[:, :3] = 2000  # 30,000 people on each side: 60,000 if joined
    population[:, -3:] = 2000
    codes = classify(population)
    assert (codes[:, :3] == scheme.DENSE_URBAN_CLUSTER).all()
    assert (codes[:, -3:] == scheme.DENSE_URBAN_CLUSTER).all()


def test_cells_outside_the_domain_get_nodata() -> None:
    domain = np.array([[True, False], [True, True]])
    codes = rules.classify(scene(np.zeros((2, 2)), domain=domain), BASE)
    assert codes.tolist() == [[11, scheme.NODATA], [11, 11]]  # empty land


def test_missing_resolved_values_are_rejected() -> None:
    inputs = scene(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="built-up threshold is needed"):
        rules.classify(inputs, params(uc_built_criterion=True))
    with pytest.raises(ValueError, match="global built-up per person"):
        rules.classify(inputs, params(density_on_land="built_up_and_population"))


def test_the_ghsl_r2023a_parameters_run_end_to_end() -> None:
    population = np.zeros((14, 14))
    population[6:12, 6:12] = 2000  # 72,000 people
    land = np.ones_like(population)
    village = np.s_[:3, :3]  # 5,400 people on cells with no land
    population[village] = 600
    land[village] = 0.0
    codes = rules.classify(
        scene(population, land), GHSL_R2023A, built_threshold=0.3, built_per_person=1e-4
    )
    assert (codes[6:12, 6:12] == scheme.URBAN_CENTRE).all()
    # On land raised to half of each cell the village is 1,200/km²; on no land
    # its density would be infinite, and it would be a dense urban cluster.
    assert (codes[village] == scheme.SEMI_DENSE_URBAN_CLUSTER).all()
    assert (classify(population, land)[village] == scheme.DENSE_URBAN_CLUSTER).all()
