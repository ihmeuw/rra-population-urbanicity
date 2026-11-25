from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import yaml
from click.testing import CliRunner

from rra_population_urbanicity.classifiers.base import ClassScheme
from rra_population_urbanicity.classifiers.degurba import scheme
from rra_population_urbanicity.classify import runner
from rra_population_urbanicity.cli import purun
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.grid import GridDefinition
from rra_population_urbanicity.reference import wup
from rra_population_urbanicity.reference.ghsl import GhslRaster
from rra_population_urbanicity.specification import ClassificationSpecification
from rra_population_urbanicity.validate import wup2025_shares
from tests.conftest import TINY, ghsl_spec, write_ghsl

WEST, EAST = 101, 102


def two_countries(grid: GridDefinition) -> tuple[np.ndarray, pd.Series]:
    """West of column 4 is AAA and the rest SRB, which cuts the town 2:3."""
    countries = np.full(grid.shape, EAST, dtype=np.int32)
    countries[:, :4] = WEST
    return countries, pd.Series({WEST: "AAA", EAST: "SRB"})


def write_wup_tables(root: Path) -> None:
    """WUP's figures, in thousands: AAA has 20 and SRB 20 plus Kosovo's 5."""
    level2 = {
        "AAA": {30: 15, 12: 5},
        "SRB": {30: 10, 23: 10},
        "XKX": {13: 5},
    }
    national = {"AAA": (12, 20), "SRB": (10, 20), "XKX": (1, 5)}
    rows = [
        {
            "LocTypeName": "Country/Area",
            "Year": year,
            "ISO3_Code": iso3,
            "Category_order": code,
            "Pop_1Jan": people,
        }
        for year in (2020, 2025)
        for iso3, classes in level2.items()
        for code, people in {**classes, 50: sum(classes.values())}.items()
    ]
    rows.append(
        {
            "LocTypeName": "World",
            "Year": 2020,
            "ISO3_Code": "WLD",
            "Category_order": 30,
            "Pop_1Jan": 99,
        }
    )
    data = UrbanicityData(root)
    path = wup.wup_path(data, "level2")
    path.parent.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    pd.DataFrame(
        [
            {
                "LocTypeName": "Country/Area",
                "Year": year,
                "ISO3_Code": iso3,
                "Category": category,
                "Pop1Jan": people,
            }
            for year in (2020, 2025)
            for iso3, (urban, total) in national.items()
            for category, people in [
                ("Urban", urban),
                ("Rural", total - urban),
                ("Total", total),
            ]
        ]
    ).to_csv(wup.wup_path(data, "national_definitions"), index=False)


@pytest.fixture
def wup_world(ghsl_world: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    """A classification of the town, GHSL's WUP layers and WUP's tables.

    GHSL's classes split the town at the border, as GHS-WUP-DEGURBA splits
    clusters by country: the SRB side is a dense urban cluster.
    """
    monkeypatch.setattr(wup2025_shares, "country_layer", two_countries)
    monkeypatch.setattr(wup2025_shares, "MIN_COUNTRY_POPULATION", 0)
    data = UrbanicityData(ghsl_world)
    version = runner.run(
        ClassificationSpecification.model_validate(ghsl_spec()), data, None
    )
    for epoch in (2020, 2025):
        with rasterio.open(data.class_raster_path(version, f"{epoch}q1")) as src:
            codes = src.read(1).astype(np.int16)
        codes[codes == scheme.NODATA] = -200
        codes[2:7, 4:7] = scheme.DENSE_URBAN_CLUSTER
        write_ghsl(
            ghsl_world,
            GhslRaster(product="WUP_DEGURBA", release="R2025A", epoch=epoch),
            codes,
            -200,
        )
        population = np.zeros(TINY.shape)
        population[2:7, 2:7] = 2000
        write_ghsl(
            ghsl_world,
            GhslRaster(product="WUP_POP", release="R2025A", epoch=epoch),
            population,
            -200,
        )
    write_wup_tables(ghsl_world)
    return ghsl_world, version


def test_shares_are_compared_three_ways(wup_world: tuple[Path, str]) -> None:
    root, version = wup_world
    summary = wup2025_shares.validate(UrbanicityData(root), version)

    out_dir = UrbanicityData(root).classification_root(version) / "validation"
    table = pd.read_parquet(out_dir / "wup2025_shares" / "by_country_2020q1.parquet")
    people = table.set_index(["iso3", "class_code"])
    # Our town is one urban centre across the border; GHSL's is split; WUP's
    # Kosovo is counted with Serbia.
    assert people.loc[("AAA", 30), "population"] == pytest.approx(20_000)
    assert people.loc[("SRB", 30), "population"] == pytest.approx(30_000)
    assert people.loc[("SRB", 30), "ghsl_population"] == 0
    assert people.loc[("SRB", 23), "ghsl_population"] == pytest.approx(30_000)
    assert people.loc[("SRB", 13), "wup_population"] == 5_000
    assert people["population"].sum() == pytest.approx(50_000)

    result = summary["time_points"]["2020q1"]
    assert result["people"]["wup_population"] == 45_000
    groups = result["world_group_shares"]
    assert groups["population"]["city"] == pytest.approx(1.0)
    assert groups["ghsl_population"]["town"] == pytest.approx(0.6)
    assert groups["wup_population"]["rural"] == pytest.approx(10 / 45)
    national = result["national_definitions"]
    assert national["world_urban_share"] == pytest.approx(1.0)
    assert national["wup_world_urban_share"] == pytest.approx(23 / 45)
    # AAA is 60% urban by its definition and SRB 44%, both 100% here.
    assert national["gap_by_country"]["median_absolute_gap"] == pytest.approx(48.0)
    city = result["gaps_by_country"]["city"]
    assert city["ours_vs_wup"]["largest"] == {"SRB": 60.0, "AAA": 25.0}
    assert city["ours_vs_ghsl"]["largest"] == {"SRB": 100.0, "AAA": 0.0}

    stored = yaml.safe_load((out_dir / "wup2025_shares" / "summary.yaml").read_text())
    assert list(stored["time_points"]) == ["2020q1", "2025q1"]


def test_without_ghsl_classes_only_ours_and_wups_are_compared(
    wup_world: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, version = wup_world
    monkeypatch.setattr(wup2025_shares, "ghsl_classes", lambda data, tp: None)
    summary = wup2025_shares.validate(UrbanicityData(root), version)
    result = summary["time_points"]["2025q1"]
    assert "ghsl_population" not in result["world_group_shares"]
    gaps = result["gaps_by_country"]["town"]["ours_vs_ghsl"]
    assert gaps == {"countries": 0, "median_absolute_gap": None, "largest": {}}


def test_ghsl_classes_exist_for_epochs_only(tmp_path: Path) -> None:
    data = UrbanicityData(tmp_path)
    assert wup2025_shares.ghsl_classes(data, "2021q1") is None
    path = wup2025_shares.ghsl_classes(data, "2025q1")
    assert path is not None
    assert path.name.startswith("GHS_WUP_DEGURBA_E2025")


def test_a_scheme_without_level_one_cannot_be_compared(
    wup_world: tuple[Path, str],
) -> None:
    root, version = wup_world
    path = UrbanicityData(root).classification_root(version) / "scheme.yaml"
    ClassScheme.from_yaml(path).model_copy(update={"groups": {}}).to_yaml(path)
    with pytest.raises(ValueError, match="no degurba_l1 group"):
        wup2025_shares.validate(UrbanicityData(root), version)


def test_wup2025_command(wup_world: tuple[Path, str]) -> None:
    root, version = wup_world
    result = CliRunner().invoke(
        purun, ["validate", "wup2025", version, "-o", str(root)]
    )
    assert result.exit_code == 0, result.output
    assert result.output.startswith(
        "2020q1: population city 100.0% town 0.0% rural 0.0% | "
        "ghsl_population city 40.0% town 60.0% rural 0.0% | "
        "wup_population city 55.6% town 22.2% rural 22.2%"
    )
