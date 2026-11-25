from click.testing import CliRunner

from rra_population_urbanicity.cli import purun, putask


def test_purun_lists_the_stages() -> None:
    result = CliRunner().invoke(purun, ["--help"])
    assert result.exit_code == 0
    assert "reference" in result.output


def test_putask_lists_the_stages_with_tasks() -> None:
    result = CliRunner().invoke(putask, ["--help"])
    assert result.exit_code == 0
    assert "inputs" in result.output
    assert "classify" in result.output
    assert "validate" not in result.output
