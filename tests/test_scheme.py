from pathlib import Path

import pytest
from pydantic import ValidationError

from rra_population_urbanicity.classify.base import ClassDef, ClassScheme
from rra_population_urbanicity.classify.degurba.scheme import DEGURBA_L2


def test_a_scheme_round_trips_through_yaml(tmp_path: Path) -> None:
    DEGURBA_L2.to_yaml(tmp_path / "scheme.yaml")
    assert ClassScheme.from_yaml(tmp_path / "scheme.yaml") == DEGURBA_L2


def test_codes_are_looked_up_by_key() -> None:
    assert DEGURBA_L2.code("urban_centre") == 30
    assert DEGURBA_L2.code("water") == 10


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"nodata": 30}, "must be unique"),
        ({"default_cut": ["urban"]}, "Unknown class keys"),
        ({"groups": {"l1": {"city": ["city_centre"]}}}, "Unknown class keys"),
    ],
)
def test_inconsistent_schemes_are_rejected(
    changes: dict[str, object], message: str
) -> None:
    raw = {**DEGURBA_L2.model_dump(), **changes}
    with pytest.raises(ValidationError, match=message):
        ClassScheme.model_validate(raw)


def test_class_keys_must_be_unique() -> None:
    first = ClassDef(code=1, key="urban", label="Urban", colour="#FF0000")
    second = ClassDef(code=2, key="urban", label="Also urban", colour="#00FF00")
    with pytest.raises(ValidationError, match="must be unique"):
        ClassScheme(
            name="twice",
            classes=[first, second],
            default_cut=[],
            groups={},
            ordered=False,
            nodata=0,
        )
