"""Class schemes: what the codes of a classification mean.

A scheme is saved next to each class raster, so a consumer can read the codes
without importing the method that produced them.
"""

from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class ClassDef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    code: int
    key: str
    label: str
    colour: str  # hex RGB, e.g. "#FF0000"


class ClassScheme(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    # Ordered from most to least urban.
    classes: list[ClassDef]
    # The class keys that count as urban unless a cut says otherwise.
    default_cut: list[str]
    # Coarser groupings: group name -> {group class -> class keys}.
    groups: dict[str, dict[str, list[str]]]
    # Whether the order of the classes is meaningful, which allows a cut to be
    # chosen per country by walking the classes from most to least urban.
    ordered: bool
    nodata: int

    @model_validator(mode="after")
    def consistent(self) -> Self:
        keys = [c.key for c in self.classes]
        codes = [c.code for c in self.classes] + [self.nodata]
        if len(set(keys)) != len(keys) or len(set(codes)) != len(codes):
            msg = f"Class keys and codes, and the nodata code, must be unique in {self.name}."
            raise ValueError(msg)
        referenced = set(self.default_cut).union(
            *(members for group in self.groups.values() for members in group.values())
        )
        if unknown := referenced - set(keys):
            msg = f"Unknown class keys in {self.name}: {sorted(unknown)}."
            raise ValueError(msg)
        return self

    def code(self, key: str) -> int:
        return next(c.code for c in self.classes if c.key == key)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.model_dump(), sort_keys=False))
