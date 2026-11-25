from pathlib import Path


import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from rra_population_urbanicity import admin
from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.classifiers.degurba import scheme
from rra_population_urbanicity.cli import purun
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.validate import us_census

VERSION = "2026_01_01.001"


def census_table(path: Path, sheet_name: str, usecols: list[str]) -> pd.DataFrame:
    """The Census's county table, as pandas would read it from the workbook."""
    assert path.name == "2020_UA_COUNTY.xlsx"
    assert sheet_name == "2020_UA_COUNTY"
    return pd.DataFrame(
        {
            "STATE_NAME": ["Washington", "Washington", "Texas", "Texas"],
            "COUNTY_NAME": ["King", "Ferry", "Harris", "Loving"],
            "POP_COU": [100, 10, 200, 1],
            "POP_URB": [90, 0, 190, 0],
        }
    )[usecols]


@pytest.fixture
def census_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """King and Ferry counties in Washington, Harris in Texas, Kalawao in Hawaii."""
    model_root = tmp_path / "population-model"
    monkeypatch.setattr(puc, "POPULATION_MODEL_ROOT", model_root)
    admin.admin_inputs().mkdir(parents=True)
    pd.DataFrame(
        {
            "location_id": [102, 600, 601, 602, 610, 611, 620, 621],
            "parent_id": [1, 102, 600, 600, 102, 610, 102, 620],
            "location_name": [
                "United States",
                "Washington",
                "King County",
                "Ferry County",
                "Hawaii",
                "Kalawao County",
                "Texas",
                "Harris County",
            ],
            "location_name_short": [
                "USA",
                "Washington",
                "King",
                "Ferry",
                "Hawaii",
                "Kalawao",
                "Texas",
                "Harris",
            ],
            "ihme_loc_id": ["USA", *[f"USA_{i}" for i in (600, 601, 602, 610, 611, 620, 621)]],
            "most_detailed": [0, 0, 1, 1, 0, 1, 0, 1],
        }
    ).to_parquet(admin.admin_inputs() / "hierarchy_us.parquet")
    monkeypatch.setitem(
        admin.SHAPES,
        us_census.SHAPES,
        admin.ShapesVersion(units="us", countries=None, hierarchy="us"),
    )
    monkeypatch.setattr(pd, "read_excel", census_table)
    root = tmp_path / "output"
    data = UrbanicityData(root)
    version_root = data.classification_root(VERSION)
    table_path = version_root / "tabulations" / us_census.SHAPES / "2020q1.parquet"
    table_path.parent.mkdir(parents=True)
    scheme.DEGURBA_L2.to_yaml(version_root / "scheme.yaml")
    pd.DataFrame(
        {
            "location_id": [601, 601, 602, 611, 621, 621],
            "class_code": [30, 12, 12, 12, 30, 12],
            "population": [95.0, 5.0, 10.0, 1.0, 150.0, 50.0],
        }
    ).to_parquet(table_path)
    return root


def test_county_urban_shares_against_the_census(
    census_world: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(us_census, "LARGE_COUNTY", 50)
    summary = us_census.compare(UrbanicityData(census_world), VERSION)
    assert summary["counties"] == {
        "matched": 3,
        "census_only": ["Loving, Texas"],
        "ours_only": ["Kalawao, Hawaii"],
    }
    assert summary["national_urban_share"] == {
        "census": pytest.approx(280 / 310),
        "ours": pytest.approx(245 / 310),
    }
    assert summary["state_urban_share"]["Texas"] == [0.95, 0.75]
    gap = summary["county_gap_points"]
    assert gap["mean_absolute"] == pytest.approx(4500 / 310)
    assert gap["people_within_5"] == pytest.approx(110 / 310)
    assert gap["people_within_20"] == 1.0
    assert gap["largest_in_large_counties"] == {
        "Harris, Texas": -20.0,
        "King, Washington": 5.0,
    }
    out_dir = UrbanicityData(census_world).classification_root(VERSION) / "validation"
    table = pd.read_parquet(out_dir / "us_census" / "by_county.parquet")
    assert len(table) == 5
    stored = yaml.safe_load((out_dir / "us_census" / "summary.yaml").read_text())
    assert stored["time_point"] == "2020q1"


def test_us_census_command(census_world: Path) -> None:
    result = CliRunner().invoke(
        purun, ["validate", "us-census", VERSION, "-o", str(census_world)]
    )
    assert result.exit_code == 0, result.output
    assert result.output == (
        "United States: 79.0% urban against the Census's 90.3%; counties off by "
        "14.5 points on average, 35% of people within 10\n"
    )
