"""The Degree of Urbanisation grid rules (levels 1 and 2), as array operations.

The rules follow GHS-SMOD R2023A V2.0 (GHSL Data Package 2023, 2024 edition,
section 2.6.2). Where the documentation allows more than one reading, a
parameter chooses between them (see `DegurbaParameters`).

Every function works on 2-D arrays on one grid and reads no files. Inputs are
per cell: population in people, land and built-up as shares of the cell.
Densities are per km² of land. Clusters never join across the edges of the
array, so a settlement split by the antimeridian stays split.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from rra_population_urbanicity.classify.degurba import scheme
from rra_population_urbanicity.classify.degurba.parameters import (
    Contiguity,
    DegurbaParameters,
)

type FloatArray = npt.NDArray[np.float64]
type BoolArray = npt.NDArray[np.bool_]
type LabelArray = npt.NDArray[np.int32]
type CodeArray = npt.NDArray[np.uint8]

STRUCTURES = {
    4: ndimage.generate_binary_structure(2, 1),
    8: ndimage.generate_binary_structure(2, 2),
}
# The contiguity of the background of a mask with the given contiguity.
COMPLEMENT: dict[Contiguity, Contiguity] = {4: 8, 8: 4}
# The eight neighbours of a cell, as (row, column) offsets.
NEIGHBOURS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
# Each urban centre is generalised in a window this many cells beyond its
# bounding box, widened if the centre grows to the window's edge.
CENTRE_PAD = 8


@dataclass(frozen=True)
class DegurbaInputs:
    """The inputs of one classification, all on the same grid."""

    population: FloatArray  # people per cell
    land: FloatArray  # share of the cell that is permanent land, 0-1
    built: FloatArray  # built-up share of the cell, 0-1; zeros without a built-up layer
    domain: BoolArray  # the cells to classify; the rest get the scheme's nodata code
    cell_area_km2: float


def passes(values: FloatArray, threshold: float, *, strict: bool) -> BoolArray:
    """Whether values pass a lower bound: strictly (>) or inclusively (>=)."""
    result: BoolArray = values > threshold if strict else values >= threshold
    return result


def label(mask: BoolArray, contiguity: Contiguity) -> tuple[LabelArray, int]:
    """Label the clusters of a mask: 4 is rook contiguity, 8 is queen."""
    labels: LabelArray
    labels, n = ndimage.label(mask, structure=STRUCTURES[contiguity], output=np.int32)
    return labels, n


def cluster_sums(
    labels: LabelArray, n: int, values: FloatArray | BoolArray
) -> FloatArray:
    """The total of `values` in each cluster, indexed by label (0 is the background)."""
    return np.bincount(labels.ravel(), weights=values.ravel(), minlength=n + 1)


def select(labels: LabelArray, keep: BoolArray) -> BoolArray:
    """The cells of the clusters marked in `keep`, which is indexed by label."""
    keep[0] = False
    result: BoolArray = keep[labels]
    return result


def dilate(mask: BoolArray, cells: int) -> BoolArray:
    """The cells within `cells` cells of the mask, in a square around each cell."""
    size = 2 * cells + 1
    rows = ndimage.maximum_filter1d(mask, size, axis=0, mode="constant", cval=0)
    result: BoolArray = ndimage.maximum_filter1d(
        rows, size, axis=1, mode="constant", cval=0
    )
    return result


def per_land(
    values: FloatArray, land: FloatArray, cell_area_km2: float = 1.0
) -> FloatArray:
    """`values` per unit of land: 0 where `values` is 0, infinite where there is no land."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result: FloatArray = values / (land * cell_area_km2)
    result[values == 0] = 0.0
    return result


def retain_built_on_land(built: FloatArray, land: FloatArray) -> FloatArray:
    """Keep only built-up that is contiguous with built-up at least partly on land (fn 34).

    Built-up detected in open water, in no contiguous area that reaches land,
    is dropped.
    """
    labels, n = label(built > 0, 8)
    reaches_land = cluster_sums(labels, n, land > 0) > 0
    return np.where(select(labels, reaches_land), built, 0.0)


def built_per_person(population: FloatArray, built: FloatArray) -> float:
    """The average built-up per person, in shares of a cell."""
    return float(built.sum() / population.sum())


def land_for_density(
    inputs: DegurbaInputs,
    built: FloatArray,
    params: DegurbaParameters,
    built_per_person: float | None,
) -> FloatArray:
    """The land share densities are computed on, after the density-on-land rule."""
    if params.density_on_land == "none":
        return inputs.land
    fill = built
    if params.density_on_land == "built_up_and_population":
        if built_per_person is None:
            msg = "The density-on-land rule needs the global built-up per person."
            raise ValueError(msg)
        imputed = np.minimum(inputs.population * built_per_person, 1.0)
        fill = np.where((built == 0) & (inputs.population > 0), imputed, built)
    raised = (inputs.land == 0) & (fill > 0)
    return np.where(raised, np.maximum(fill, params.water_land_min), inputs.land)


@dataclass(frozen=True)
class Prepared:
    """The inputs as the rules use them."""

    built: FloatArray  # built-up that reaches land (fn 34)
    land: FloatArray  # land share after the density-on-land rule
    density: FloatArray  # people per km² of that land
    built_on_land: FloatArray  # built-up share of that land


def prepare(
    inputs: DegurbaInputs,
    params: DegurbaParameters,
    built_per_person: float | None,
) -> Prepared:
    built = retain_built_on_land(inputs.built, inputs.land)
    land = land_for_density(inputs, built, params, built_per_person)
    return Prepared(
        built=built,
        land=land,
        density=per_land(inputs.population, land, inputs.cell_area_km2),
        built_on_land=per_land(built, land),
    )


def optimal_built_threshold(
    population: FloatArray,
    prepared: Prepared,
    params: DegurbaParameters,
) -> float:
    """The built-up threshold computed from the data (fn 30).

    The average built-up share in rook clusters of cells at or above the
    urban-centre density that hold at least `built_threshold_cluster_size`
    people, averaged as `built_threshold_average` says.
    """
    strict = params.strict_thresholds
    labels, n = label(
        passes(prepared.density, params.uc_density, strict=strict),
        params.uc_contiguity,
    )
    large = passes(
        cluster_sums(labels, n, population),
        params.built_threshold_cluster_size,
        strict=strict,
    )
    cells = select(labels, large)
    if not cells.any():
        msg = "No cluster is large enough to compute the built-up threshold."
        raise ValueError(msg)
    if params.built_threshold_average == "cell_mean_land":
        return float(prepared.built_on_land[cells].mean())
    if params.built_threshold_average == "area_ratio":
        return float(prepared.built[cells].sum() / prepared.land[cells].sum())
    return float(prepared.built[cells].mean())


def dense_clusters(
    population: FloatArray,
    density: FloatArray,
    built_on_land: FloatArray,
    *,
    density_min: float,
    size_min: float,
    contiguity: Contiguity,
    built_threshold: float | None,
    require_dense_cell: bool,
    strict: bool,
) -> BoolArray:
    """Cells in clusters of dense cells that hold enough people (classes 30 and 23).

    With a `built_threshold`, cells at or above it count as dense too (the
    built-up criterion). With `require_dense_cell`, a cluster must contain at
    least one cell that passes the density test itself.
    """
    dense = passes(density, density_min, strict=strict)
    cells = dense
    if built_threshold is not None:
        cells = dense | passes(built_on_land, built_threshold, strict=strict)
    labels, n = label(cells, contiguity)
    keep = passes(cluster_sums(labels, n, population), size_min, strict=strict)
    if require_dense_cell:
        keep &= cluster_sums(labels, n, dense) > 0
    return select(labels, keep)


def smooth_edges(centres: BoolArray, eligible: BoolArray) -> BoolArray:
    """Grow urban centres by an iterative 3x3 majority filter (Table 22, fn 29).

    The majority is taken over the eligible cells (populated, or mostly land)
    of a cell's 3x3 window, the cell itself included. A cell joins when at
    least half of them are urban centre: with nine cells there is no draw, and
    a draw, possible only when water leaves an even number, counts as urban.
    Only eligible cells can join, so water is never added. All cells are
    tested against the same state, and the test is repeated until no cell
    joins. A cell's answer can change only when a neighbour has just joined,
    so after the first round only those cells are tested again.
    """
    urban = centres.copy()
    n_rows, n_cols = urban.shape
    rows, cols = np.nonzero(dilate(urban, 1) & ~urban & eligible)
    while rows.size:
        n_urban = np.zeros(rows.size, dtype=np.int32)
        n_eligible = np.ones(rows.size, dtype=np.int32)  # the cell itself
        for dr, dc in NEIGHBOURS:
            r, c = rows + dr, cols + dc
            inside = (r >= 0) & (r < n_rows) & (c >= 0) & (c < n_cols)
            r, c = r.clip(0, n_rows - 1), c.clip(0, n_cols - 1)
            counted = inside & eligible[r, c]
            n_eligible += counted
            n_urban += counted & urban[r, c]
        joins = 2 * n_urban >= n_eligible
        urban[rows[joins], cols[joins]] = True
        rows, cols = candidates_near(urban, eligible, rows[joins], cols[joins])
    return urban


def candidates_near(
    urban: BoolArray,
    eligible: BoolArray,
    rows: npt.NDArray[np.intp],
    cols: npt.NDArray[np.intp],
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
    """The eligible cells next to a set of cells that are not yet urban, each once."""
    n_rows, n_cols = urban.shape
    offsets = np.array(NEIGHBOURS)
    r = (rows[:, None] + offsets[:, 0]).ravel()
    c = (cols[:, None] + offsets[:, 1]).ravel()
    inside = (r >= 0) & (r < n_rows) & (c >= 0) & (c < n_cols)
    r, c = r[inside], c[inside]
    keep = eligible[r, c] & ~urban[r, c]
    flat = np.unique(r[keep] * n_cols + c[keep])
    return np.divmod(flat, n_cols)


def fill_gaps(mask: BoolArray, max_cells: float, contiguity: Contiguity) -> BoolArray:
    """Fill the holes in a mask that have fewer than `max_cells` cells.

    A hole is a connected area outside the mask that does not reach the edge
    of the array. Holes are connected with the other contiguity (queen for a
    rook-connected mask), so an area enclosed only through cells that touch
    at a corner, which the mask's own contiguity does not join, is not a hole.
    """
    labels, n = label(~mask, COMPLEMENT[contiguity])
    small = np.bincount(labels.ravel(), minlength=n + 1) < max_cells
    edge = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    small[edge] = False
    result: BoolArray = mask | select(labels, small)
    return result


def urban_centres(
    population: FloatArray,
    density: FloatArray,
    built_on_land: FloatArray,
    eligible: BoolArray,
    params: DegurbaParameters,
    built_threshold: float | None,
    cell_area_km2: float,
) -> BoolArray:
    """Class 30: dense clusters of at least `uc_size` people, each smoothed and gap-filled.

    The clusters are found before the morphological steps (fn 33).
    """
    seeds = dense_clusters(
        population,
        density,
        built_on_land,
        density_min=params.uc_density,
        size_min=params.uc_size,
        contiguity=params.uc_contiguity,
        built_threshold=built_threshold if params.uc_built_criterion else None,
        require_dense_cell=params.uc_require_dense_cell,
        strict=params.strict_thresholds,
    )
    return generalise_centres(seeds, eligible, params, cell_area_km2)


def generalise(
    centre: BoolArray, eligible: BoolArray, params: DegurbaParameters, max_cells: float
) -> BoolArray:
    """Smooth one urban centre's edges and fill its gaps, in the order the parameters give."""
    contiguity = params.uc_contiguity
    if params.uc_morphology_order == "smooth_then_fill":
        return fill_gaps(smooth_edges(centre, eligible), max_cells, contiguity)
    return smooth_edges(fill_gaps(centre, max_cells, contiguity), eligible)


def padded(
    box: tuple[slice, slice], pad: int, shape: tuple[int, ...]
) -> tuple[slice, slice]:
    rows, cols = box
    return (
        slice(max(rows.start - pad, 0), min(rows.stop + pad, shape[0])),
        slice(max(cols.start - pad, 0), min(cols.stop + pad, shape[1])),
    )


def reaches_window_edge(
    mask: BoolArray, window: tuple[slice, slice], shape: tuple[int, ...]
) -> bool:
    """Whether `mask` reaches a side of `window` that is not an edge of the array."""
    rows, cols = window
    return bool(
        (rows.start > 0 and mask[0].any())
        or (rows.stop < shape[0] and mask[-1].any())
        or (cols.start > 0 and mask[:, 0].any())
        or (cols.stop < shape[1] and mask[:, -1].any())
    )


def generalise_centres(
    seeds: BoolArray,
    eligible: BoolArray,
    params: DegurbaParameters,
    cell_area_km2: float,
) -> BoolArray:
    """Generalise each urban centre on its own, then combine them.

    A cell between two centres counts only one centre's cells towards the
    majority, so smoothing does not join centres that lie close together.
    GHS-SMOD V2.0 behaves this way: counted over all of its class 30 cells, the
    majority rule would add cells between neighbouring centres that it leaves out.
    """
    max_cells = params.uc_max_gap_km2 / cell_area_km2
    labels, _ = label(seeds, params.uc_contiguity)
    centres = np.zeros_like(seeds)
    for index, box in enumerate(ndimage.find_objects(labels), start=1):
        pad = CENTRE_PAD
        while True:
            window = padded(box, pad, labels.shape)
            grown = generalise(
                labels[window] == index, eligible[window], params, max_cells
            )
            if not reaches_window_edge(grown, window, labels.shape):
                break
            pad *= 2
        centres[window] |= grown
    return centres


def semi_dense(
    population: FloatArray,
    density: FloatArray,
    urban: BoolArray,
    in_urban_cluster: BoolArray,
    params: DegurbaParameters,
) -> BoolArray:
    """Class 22: semi-dense cells away from classes 30 and 23, in clusters of enough people.

    `urban` is the class 30 and 23 cells; the buffer is the square of
    `sduc_buffer_cells` cells around each of them (fn 32). How the buffer and
    the size test combine is `sduc_buffer_test`.
    """
    strict = params.strict_thresholds
    candidates = passes(density, params.sduc_density, strict=strict) & ~urban
    if params.sduc_within_urban_cluster:
        candidates &= in_urban_cluster
    near = dilate(urban, params.sduc_buffer_cells)
    if params.sduc_buffer_test == "remove_then_cluster":
        labels, n = label(candidates & ~near, params.sduc_contiguity)
        keep = passes(
            cluster_sums(labels, n, population), params.sduc_size, strict=strict
        )
    elif params.sduc_buffer_test == "cluster_then_remove":
        labels, n = label(candidates, params.sduc_contiguity)
        labels[near] = 0
        keep = passes(
            cluster_sums(labels, n, population), params.sduc_size, strict=strict
        )
    else:
        labels, n = label(candidates, params.sduc_contiguity)
        keep = passes(
            cluster_sums(labels, n, population), params.sduc_size, strict=strict
        )
        keep &= cluster_sums(labels, n, near) == 0
    return select(labels, keep)


def classify(
    inputs: DegurbaInputs,
    params: DegurbaParameters,
    *,
    built_threshold: float | None = None,
    built_per_person: float | None = None,
) -> CodeArray:
    """Classify every cell into the `degurba_l2` codes.

    `built_threshold` is the resolved built-up threshold, needed when either
    built-up criterion is on. `built_per_person` is the global average
    built-up per person, needed by the `built_up_and_population` reading of
    the density-on-land rule.
    """
    if (
        params.uc_built_criterion or params.duc_built_criterion
    ) and built_threshold is None:
        msg = "The built-up criterion is on, so a built-up threshold is needed."
        raise ValueError(msg)
    strict = params.strict_thresholds
    population = inputs.population
    prepared = prepare(inputs, params, built_per_person)
    built, density, built_on_land = (
        prepared.built,
        prepared.density,
        prepared.built_on_land,
    )
    # Populated or mostly land: the cells the majority rule counts (fn 29).
    eligible = (population > 0) | (inputs.land >= params.water_land_min)
    on_land = eligible | (built > 0)

    centres = urban_centres(
        population,
        density,
        built_on_land,
        eligible,
        params,
        built_threshold,
        inputs.cell_area_km2,
    )
    dense_urban = ~centres & dense_clusters(
        population,
        density,
        built_on_land,
        density_min=params.duc_density,
        size_min=params.duc_size,
        contiguity=params.duc_contiguity,
        built_threshold=built_threshold if params.duc_built_criterion else None,
        require_dense_cell=params.duc_require_dense_cell,
        strict=strict,
    )
    urban = centres | dense_urban
    # Level-1 urban clusters contain every class 30 and 23 cell.
    labels, n = label(
        passes(density, params.ucl_density, strict=strict) | urban,
        params.ucl_contiguity,
    )
    in_urban_cluster = select(
        labels,
        passes(cluster_sums(labels, n, population), params.ucl_size, strict=strict),
    )
    semi = semi_dense(population, density, urban, in_urban_cluster, params)
    labels, n = label(
        passes(density, params.rc_density, strict=strict)
        & ~(in_urban_cluster | urban | semi),
        params.rc_contiguity,
    )
    rural_cluster = select(
        labels,
        passes(cluster_sums(labels, n, population), params.rc_size, strict=strict),
    )

    codes = np.full(population.shape, scheme.WATER, dtype=np.uint8)
    codes[on_land] = scheme.VERY_LOW_DENSITY_RURAL
    codes[passes(density, params.ldr_density, strict=strict)] = scheme.LOW_DENSITY_RURAL
    codes[rural_cluster] = scheme.RURAL_CLUSTER
    codes[in_urban_cluster] = scheme.SUBURBAN
    codes[semi] = scheme.SEMI_DENSE_URBAN_CLUSTER
    codes[dense_urban] = scheme.DENSE_URBAN_CLUSTER
    codes[centres] = scheme.URBAN_CENTRE
    codes[~inputs.domain] = scheme.NODATA
    return codes
