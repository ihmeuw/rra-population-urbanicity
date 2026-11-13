from click.testing import CliRunner

from rra_population_urbanicity.cli import purun, putask


def test_purun_lists_the_stages() -> None:
    result = CliRunner().invoke(purun, ["--help"])
    assert result.exit_code == 0
    assert "reference" in result.output


def test_putask_has_no_stages_yet() -> None:
    result = CliRunner().invoke(putask, ["--help"])
    assert result.exit_code == 0
    assert "Commands:" not in result.output
