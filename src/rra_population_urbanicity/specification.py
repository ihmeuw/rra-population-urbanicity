"""The classification specification: everything needed to reproduce a classification.

Each input is described by a source-specific model. Every source exposes its
version as an ordered tuple of path components, which is how its aggregated
grids are stored and found (see `UrbanicityData.grid_path`).
"""

from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from rra_population_urbanicity.classify.degurba.parameters import (
    PARAMETER_SETS,
    DegurbaParameters,
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


class GhsBuiltSSource(_GhslSource):
    source: Literal["ghs_built_s"]
    release: Literal["r2023a"]


class MicrosoftDensitySource(_Spec):
    """Building-footprint density from the Microsoft building predictions."""

    source: Literal["microsoft_density"]
    building_version: Literal["v8"]
    resolution: Literal["40"]

    @property
    def version_components(self) -> tuple[str, ...]:
        return self.building_version, f"{self.resolution}m"


PopulationSource = Annotated[
    PopulaceSource | GhsPopSource, Field(discriminator="source")
]
LandSource = Annotated[
    GhsLandSource | MicrosoftWaterSource, Field(discriminator="source")
]
BuiltSource = Annotated[
    GhsBuiltSSource | MicrosoftDensitySource, Field(discriminator="source")
]


class MethodSpec(_Spec):
    """A classification method and its parameters.

    `parameter_set` names a set of defaults and `parameters` overrides any of
    them. Once validated, `parameters` holds the full resolved set, which is
    what a saved specification records. A saved classification therefore does
    not depend on how a named set is defined later.
    """

    name: Literal["degurba_r2023a"]
    parameter_set: str
    parameters: DegurbaParameters

    @model_validator(mode="before")
    @classmethod
    def resolve_parameters(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("parameter_set") in PARAMETER_SETS:
            defaults = PARAMETER_SETS[data["parameter_set"]].model_dump()
            overrides = data.get("parameters") or {}
            data = {**data, "parameters": {**defaults, **overrides}}
        return data

    @field_validator("parameter_set")
    @classmethod
    def known_parameter_set(cls, parameter_set: str) -> str:
        if parameter_set not in PARAMETER_SETS:
            msg = f"Unknown parameter set {parameter_set}. Choose from {list(PARAMETER_SETS)}."
            raise ValueError(msg)
        return parameter_set


class ClassificationSpecification(_Spec):
    """A classification's inputs, grid, time points and method.

    `grid` names a grid in `GRIDS`. Once validated, it holds the grid's full
    definition, which is what a saved specification records. A saved
    definition must match the named grid, so a specification cannot describe
    a grid the code does not define.
    """

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
        parameters = self.method.parameters
        if self.built is None and (
            parameters.uc_built_criterion or parameters.duc_built_criterion
        ):
            msg = "The built-up criterion is on, but there is no built-up source."
            raise ValueError(msg)
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False)
        )
