"""Parameters of the Degree of Urbanisation grid classification (levels 1 and 2).

The rules follow GHS-SMOD R2023A V2.0 as documented in the GHSL Data Package
2023, 2024 edition (PDF dated 2024-09-24), section 2.6.2: the level-2 prose,
Figure 28, Table 22 and footnotes 29-34.
https://human-settlement.emergency.copernicus.eu/documents/GHSL_Data_Package_2023.pdf

Densities are people per km² of permanent land; sizes are people per cluster.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt

from rra_population_urbanicity.constants import TimePoint

# 4 is rook contiguity (cells sharing an edge), 8 is queen (an edge or a corner).
Contiguity = Literal[4, 8]


class DegurbaParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    # Whether lower bounds are strict (>) or inclusive (>=). Table 22 writes ">";
    # the prose says "at least" and Figure 28 "≥".
    strict_thresholds: bool
    # A cell is on land if at least this share of it is permanent land, or if
    # it is populated or built (Table 22, classes 10 and 11).
    water_land_min: float = Field(ge=0, le=1)
    # How land is counted where people or buildings sit on cells with no land
    # at all (Figure 28's "density on land" step). Cells with any land keep
    # their land share; GHS-SMOD V2.0's 11/12 boundary shows this.
    #   none: divide by the land share as it is;
    #   built_up: raise the land share of built-up cells with no land to the
    #     larger of `water_land_min` and their built-up share;
    #   built_up_and_population: also give populated cells with no land and no
    #     built-up a built-up share of population times the global average
    #     built-up per person, and raise them the same way.
    density_on_land: Literal["none", "built_up", "built_up_and_population"]

    # Urban centres (class 30).
    uc_density: PositiveFloat
    uc_size: PositiveFloat
    uc_contiguity: Contiguity
    uc_built_criterion: bool
    # A cluster must contain at least one cell that passes the density test.
    uc_require_dense_cell: bool
    uc_smooth_edge: Literal["majority_rule_r2023a"]
    uc_max_gap_km2: PositiveFloat
    uc_morphology_order: Literal["smooth_then_fill", "fill_then_smooth"]

    # Dense urban clusters (class 23).
    duc_density: PositiveFloat
    duc_size: PositiveFloat
    duc_contiguity: Contiguity
    duc_built_criterion: bool
    duc_require_dense_cell: bool

    # Urban clusters (level 1). They also contain every class 30 and 23 cell.
    ucl_density: PositiveFloat
    ucl_size: PositiveFloat
    ucl_contiguity: Contiguity

    # Semi-dense urban clusters (class 22).
    sduc_density: PositiveFloat
    sduc_size: PositiveFloat
    sduc_contiguity: Contiguity
    # Cells within this many cells of a class 30 or 23 cell are excluded: the
    # buffer is the square of (2n + 1) x (2n + 1) cells centred on each one.
    sduc_buffer_cells: PositiveInt
    sduc_within_urban_cluster: bool
    # How the buffer and the size test combine:
    #   remove_then_cluster: drop cells in the buffer, cluster the rest, size-test each piece;
    #   cluster_then_remove: cluster, drop cells in the buffer, size-test what is left of each cluster;
    #   drop_cluster: drop any cluster that touches the buffer.
    sduc_buffer_test: Literal[
        "remove_then_cluster", "cluster_then_remove", "drop_cluster"
    ]

    # Rural clusters (class 13).
    rc_density: PositiveFloat
    rc_size: PositiveFloat
    rc_contiguity: Contiguity

    # Low density rural cells (class 12).
    ldr_density: PositiveFloat

    # The built-up share above which a cell counts towards classes 30 and 23
    # when the built-up criterion is on. "optimal" computes it from the data
    # (footnote 30): the average built-up share in rook clusters of cells above
    # the urban-centre density that hold at least `built_threshold_cluster_size`
    # people, on the `built_threshold_epoch` time point, or on each time point's
    # own built-up layer.
    built_threshold: Literal["optimal"] | Annotated[float, Field(ge=0, le=1)]
    built_threshold_epoch: TimePoint | Literal["each_time_point"]
    built_threshold_cluster_size: PositiveFloat
    # How the average is taken: the mean over cells of built-up share of land,
    # total built-up area over total land area, or the mean over cells of
    # built-up share of the whole cell.
    built_threshold_average: Literal["cell_mean_land", "area_ratio", "cell_mean_total"]

    def rules_using_built(self) -> list[str]:
        """The enabled rules that need a built-up layer.

        The on-land test also uses built-up, but reads a missing layer as zero.
        """
        rules = [
            name
            for name in ("uc_built_criterion", "duc_built_criterion")
            if getattr(self, name)
        ]
        if self.density_on_land != "none":
            rules.append(f"density_on_land: {self.density_on_land}")
        return rules


GHSL_R2023A = DegurbaParameters(
    strict_thresholds=False,
    water_land_min=0.5,
    # The fullest reading of the Figure 28 caption, which GHS-SMOD V2.0 bears out
    # for cells with no land.
    density_on_land="built_up_and_population",
    uc_density=1500,
    uc_size=50_000,
    uc_contiguity=4,
    uc_built_criterion=True,
    uc_require_dense_cell=True,
    uc_smooth_edge="majority_rule_r2023a",
    uc_max_gap_km2=15,
    uc_morphology_order="smooth_then_fill",
    duc_density=1500,
    duc_size=5000,
    duc_contiguity=4,
    duc_built_criterion=True,
    # Figure 28 checks for a dense cell before both the urban-centre and the
    # dense-urban-cluster size tests; Table 22 states it for class 30 only.
    duc_require_dense_cell=True,
    ucl_density=300,
    ucl_size=5000,
    ucl_contiguity=8,
    sduc_density=900,
    sduc_size=2500,
    sduc_contiguity=4,
    sduc_buffer_cells=2,
    sduc_within_urban_cluster=True,
    sduc_buffer_test="remove_then_cluster",
    rc_density=300,
    rc_size=500,
    rc_contiguity=8,
    ldr_density=50,
    built_threshold="optimal",
    built_threshold_epoch="2020q1",
    built_threshold_cluster_size=50_000,
    built_threshold_average="cell_mean_land",
)

PARAMETER_SETS = {
    # The rules of GHS-SMOD R2023A V2.0, used to reproduce it.
    "ghsl_r2023a": GHSL_R2023A,
    # For Populace runs: the built-up criterion is off until its threshold has
    # been computed on our own built-up data.
    "populace_default": GHSL_R2023A.model_copy(
        update={"uc_built_criterion": False, "duc_built_criterion": False}
    ),
}
