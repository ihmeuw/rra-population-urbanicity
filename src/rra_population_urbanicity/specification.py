"""The classification specification: everything needed to reproduce a classification.

Each input is described by a source-specific model. Every source exposes its
version as an ordered tuple of path components, which is how its aggregated
grids are stored and found (see `UrbanicityData.grid_path`).
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from rra_population_urbanicity.classifiers.degurba.parameters import (
    PARAMETER_SETS,
    DegurbaParameters,
)
from rra_population_urbanicity.classifiers.kernel_density.parameters import (
    PARAMETER_SETS as KERNEL_PARAMETER_SETS,
)
from rra_population_urbanicity.classifiers.kernel_density.parameters import (
    KernelDensityParameters,
)
from rra_population_urbanicity.constants import TimePoint
from rra_population_urbanicity.grid import GRIDS, GridDefinition

# Versions of rra-population-model, e.g. 2026_10_02.001.
ModelVersion = Annotated[str, StringConstraints(pattern=r"^\d{4}_\d{2}_\d{2}\.\d{3}$")]


class _Spec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _GhslSource(_Spec):
    release: str

    @property
    def version_components(self) -> tuple[str, ...]:
        return (self.release,)


class PopulaceSource(_Spec):
    """The Populace surface: the compiled predictions of rra-population-model."""

    source: Literal["populace"]
    model_version: ModelVersion
    resolution: Literal["40"]

    @property
    def version_components(self) -> tuple[str, ...]:
        return f"{self.resolution}m", self.model_version


class GhsPopSource(_GhslSource):
    source: Literal["ghs_pop"]
    release: Literal["r2023a"]


class GhsWupPopSource(_GhslSource):
    """GHS-WUP-POP: the population grid behind WUP 2025's Degree of Urbanisation."""

    source: Literal["ghs_wup_pop"]
    release: Literal["r2025a"]


class GhsLandSource(_GhslSource):
    """GHS-LAND: the share of each cell that is permanent land."""

    source: Literal["ghs_land"]
    release: Literal["r2022a"]


class MicrosoftWaterSource(_Spec):
    """The static water mask behind the Microsoft building predictions."""

    source: Literal["microsoft_water"]
    building_version: Literal["v8"]

    @property
    def version_components(self) -> tuple[str, ...]:
        return (self.building_version,)


class GhsLandOrMicrosoftWaterSource(_Spec):
    """Land wherever GHS-LAND or the Microsoft water mask has it.

    Combined on the 40 m model grid: each pixel takes the larger of the mask's
    land (0 or 1) and GHS-LAND's land share of the 100 m cell containing it.
    """

    source: Literal["ghs_land_or_microsoft_water"]
    ghs_land_release: Literal["r2022a"]
    building_version: Literal["v8"]

    @property
    def version_components(self) -> tuple[str, ...]:
        return self.ghs_land_release, self.building_version


class GhsBuiltSSource(_GhslSource):
    source: Literal["ghs_built_s"]
    release: Literal["r2023a"]


class GhsWupBuiltSSource(_GhslSource):
    """GHS-WUP-BUILT-S: the built-up grid released with GHS-WUP-POP."""

    source: Literal["ghs_wup_built_s"]
    release: Literal["r2025a"]


class MicrosoftDensitySource(_Spec):
    """Building-footprint density from the Microsoft building predictions."""

    source: Literal["microsoft_density"]
    building_version: Literal["v8"]
    resolution: Literal["40"]

    @property
    def version_components(self) -> tuple[str, ...]:
        return self.building_version, f"{self.resolution}m"


PopulationSource = Annotated[
    PopulaceSource | GhsPopSource | GhsWupPopSource, Field(discriminator="source")
]
LandSource = Annotated[
    GhsLandSource | MicrosoftWaterSource | GhsLandOrMicrosoftWaterSource,
    Field(discriminator="source"),
]
BuiltSource = Annotated[
    GhsBuiltSSource | GhsWupBuiltSSource | MicrosoftDensitySource,
    Field(discriminator="source"),
]


class _Method(_Spec):
    """A classification method and its parameters.

    `parameter_set` names a set of defaults and `parameters` overrides any of
    them. Once validated, `parameters` holds the full resolved set, which is
    what a saved specification records. A saved classification therefore does
    not depend on how a named set is defined later.
    """

    parameter_sets: ClassVar[Mapping[str, BaseModel]]
    name: str
    parameter_set: str

    @model_validator(mode="before")
    @classmethod
    def resolve_parameters(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("parameter_set") in cls.parameter_sets:
            defaults = cls.parameter_sets[data["parameter_set"]].model_dump()
            overrides = data.get("parameters") or {}
            data = {**data, "parameters": {**defaults, **overrides}}
        return data

    @field_validator("parameter_set")
    @classmethod
    def known_parameter_set(cls, parameter_set: str) -> str:
        if parameter_set not in cls.parameter_sets:
            msg = f"Unknown parameter set {parameter_set}. Choose from {list(cls.parameter_sets)}."
            raise ValueError(msg)
        return parameter_set


class DegurbaMethod(_Method):
    """The Degree of Urbanisation on a 1 km grid."""

    parameter_sets: ClassVar[Mapping[str, BaseModel]] = PARAMETER_SETS
    name: Literal["degurba_r2023a"]
    parameters: DegurbaParameters


class KernelDensityMethod(_Method):
    """The density of people on land in a disc around each 40 m pixel."""

    parameter_sets: ClassVar[Mapping[str, BaseModel]] = KERNEL_PARAMETER_SETS
    name: Literal["kernel_density"]
    parameters: KernelDensityParameters


MethodSpec = Annotated[DegurbaMethod | KernelDensityMethod, Field(discriminator="name")]


class ClassificationSpecification(_Spec):
    """A classification's inputs, grid, time points and method.

    `grid` names a grid in `GRIDS`. Once validated, it holds the grid's full
    definition, which is what a saved specification records. A saved
    definition must match the named grid, so a specification cannot describe
    a grid the code does not define.
    """

    # Assigned when the specification is run; committed specifications have none.
    version: str | None = None
    population: PopulationSource
    grid: GridDefinition
    land: LandSource
    built: BuiltSource | None = None
    time_points: list[TimePoint] = Field(min_length=1)
    method: MethodSpec

    @field_validator("grid", mode="before")
    @classmethod
    def resolve_grid(cls, grid: Any) -> GridDefinition:
        definition = (
            None if isinstance(grid, str) else GridDefinition.model_validate(grid)
        )
        name = grid if definition is None else definition.name
        if name not in GRIDS:
            msg = f"Unknown grid {name}. Choose from {list(GRIDS)}."
            raise ValueError(msg)
        if definition is not None and definition != GRIDS[name]:
            msg = f"The grid definition does not match the {name} grid."
            raise ValueError(msg)
        return GRIDS[name]

    @field_validator("time_points")
    @classmethod
    def unique_time_points(cls, time_points: list[str]) -> list[str]:
        if len(set(time_points)) != len(time_points):
            msg = f"Time points must be unique: {time_points}."
            raise ValueError(msg)
        return time_points

    @model_validator(mode="after")
    def built_source_if_needed(self) -> Self:
        if not isinstance(self.method, DegurbaMethod):
            return self
        if self.built is None and (rules := self.method.parameters.rules_using_built()):
            msg = f"There is no built-up source, but these rules use it: {', '.join(rules)}."
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def pixel_inputs_for_pixel_methods(self) -> Self:
        """The kernel works on the 40 m model grid: Populace and GHS-LAND at 100 m."""
        if isinstance(self.method, KernelDensityMethod) and not (
            isinstance(self.population, PopulaceSource)
            and isinstance(self.land, GhsLandSource)
        ):
            msg = "The kernel_density method needs Populace and GHS-LAND."
            # Pydantic reports a ValueError as a validation error.
            raise ValueError(msg)  # noqa: TRY004
        return self

    @property
    def degurba_parameters(self) -> DegurbaParameters:
        """The method's parameters, for code that only DEGURBA runs."""
        if not isinstance(self.method, DegurbaMethod):
            msg = f"{self.method.name} is not the Degree of Urbanisation."
            raise TypeError(msg)
        return self.method.parameters

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)
        )
