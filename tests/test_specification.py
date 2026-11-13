from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from rra_population_urbanicity.classify.degurba.parameters import PARAMETER_SETS
from rra_population_urbanicity.grid import GRIDS
from rra_population_urbanicity.specification import ClassificationSpecification

SPECIFICATION_DIR = Path(__file__).parents[1] / "specifications"
V1_REPRODUCTION = SPECIFICATION_DIR / "ghs_smod_r2023a_v1_reproduction.yaml"
V2_REPRODUCTION = SPECIFICATION_DIR / "ghs_smod_r2023a_v2_reproduction.yaml"


def changed_parameters(
    spec: ClassificationSpecification, parameter_set: str
) -> set[str]:
    resolved = spec.method.parameters.model_dump()
    defaults = PARAMETER_SETS[parameter_set].model_dump()
    return {name for name, value in resolved.items() if value != defaults[name]}


@pytest.mark.parametrize(
    "path", sorted(SPECIFICATION_DIR.glob("*.yaml")), ids=lambda path: path.stem
)
def test_specifications_round_trip_through_yaml(path: Path, tmp_path: Path) -> None:
    spec = ClassificationSpecification.from_yaml(path)
    spec.to_yaml(tmp_path / "specification.yaml")
    assert (
        ClassificationSpecification.from_yaml(tmp_path / "specification.yaml") == spec
    )


def test_a_saved_specification_records_the_full_grid_definition(
    tmp_path: Path,
) -> None:
    spec = ClassificationSpecification.from_yaml(V2_REPRODUCTION)
    spec.to_yaml(tmp_path / "specification.yaml")
    saved = yaml.safe_load((tmp_path / "specification.yaml").read_text())
    assert spec.grid == GRIDS["moll_1km"]
    assert saved["grid"] == GRIDS["moll_1km"].model_dump(mode="json")


def test_a_parameter_set_resolves_to_the_same_parameters_every_time() -> None:
    first = ClassificationSpecification.from_yaml(V2_REPRODUCTION)
    second = ClassificationSpecification.from_yaml(V2_REPRODUCTION)
    assert first.method.parameters == second.method.parameters
    assert first.method.parameters == PARAMETER_SETS["ghsl_r2023a"]


def test_populace_default_only_turns_off_the_built_up_criterion() -> None:
    populace = PARAMETER_SETS["populace_default"].model_dump()
    ghsl = PARAMETER_SETS["ghsl_r2023a"].model_dump()
    changed = {name for name, value in ghsl.items() if value != populace[name]}
    assert changed == {"uc_built_criterion", "duc_built_criterion"}
    assert not populace["uc_built_criterion"]


def test_the_v1_reproduction_overrides_nine_parameters() -> None:
    spec = ClassificationSpecification.from_yaml(V1_REPRODUCTION)
    assert changed_parameters(spec, "ghsl_r2023a") == {
        "sduc_density",
        "sduc_size",
        "sduc_contiguity",
        "sduc_buffer_cells",
        "sduc_buffer_test",
        "uc_require_dense_cell",
        "duc_require_dense_cell",
        "built_threshold_cluster_size",
        "built_threshold_epoch",
    }


@pytest.mark.parametrize(
    ("keys", "value", "message"),
    [
        (("method", "parameters"), {"no_such_parameter": 1}, "Extra inputs"),
        (("method", "parameter_set"), "no_such_set", "Unknown parameter set"),
        (("time_points",), ["2020"], "String should match pattern"),
        (("time_points",), ["2020q1", "2020q1"], "Time points must be unique"),
        (("built",), None, "no built-up source"),
        (("grid",), "no_such_grid", "Unknown grid"),
        (
            ("grid",),
            {**GRIDS["moll_1km"].model_dump(), "cell_size": 500.0},
            "does not match the moll_1km grid",
        ),
    ],
)
def test_invalid_specifications_are_rejected(
    keys: tuple[str, ...], value: Any, message: str
) -> None:
    raw = yaml.safe_load(V2_REPRODUCTION.read_text())
    target = raw
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    with pytest.raises(ValidationError, match=message):
        ClassificationSpecification.model_validate(raw)
