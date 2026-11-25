from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner
from shapely.geometry import box

from rra_population_urbanicity.classify import runner
from rra_population_urbanicity.cli import purun
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.specification import ClassificationSpecification
from rra_population_urbanicity.validate import settlements
from tests.conftest import TINY, ghsl_spec


def cells(row0: int, row1: int, col0: int, col1: int) -> box:
    """The shape of a block of cells of `tiny_1km`, rows and columns half-open."""
    return box(
        TINY.x_min + col0 * TINY.cell_size,
        TINY.y_max - row1 * TINY.cell_size,
        TINY.x_min + col1 * TINY.cell_size,
        TINY.y_max - row0 * TINY.cell_size,
    )


def write_reference(root: Path, name: str, frame: gpd.GeoDataFrame) -> None:
    reference = settlements.REFERENCES[name]
    path = UrbanicityData(root).reference_root(
        "ghsl", reference.product_dir, reference.release_dir
    )
    path.mkdir(parents=True)
    frame.to_file(path / reference.file, layer=reference.layer)


@pytest.fixture
def classified(ghsl_world: Path) -> tuple[Path, str]:
    """The town of 50,000 classified as one urban centre, and GHSL's centres.

    GHS-UCDB has it as two centres, West (columns 2-4) and East (columns 5-6),
    and a centre elsewhere with nobody of ours. WUP has only the one elsewhere.
    """
    spec = ClassificationSpecification.model_validate(ghsl_spec())
    version = runner.run(spec, UrbanicityData(ghsl_world), None)
    write_reference(
        ghsl_world,
        "ghs_ucdb_r2024a",
        gpd.GeoDataFrame(
            {
                "ID_UC_G0": [7, 8, 9],
                "GC_POP_TOT_2025": [30_000.0, 15_000.0, 60_000.0],
                "GC_CNT_GAD_2025": ["A", "A", "B"],
                "GC_UCN_MAI_2025": ["West", "East", "Elsewhere"],
            },
            geometry=[cells(2, 7, 2, 5), cells(2, 7, 5, 7), cells(9, 11, 9, 11)],
            crs=TINY.crs,
        ),
    )
    write_reference(
        ghsl_world,
        "ghs_wup_r2025a",
        gpd.GeoDataFrame(
            {"ID_UC_G0": [3], "POP_2025": [60_000.0], "UNLocID": [12]},
            geometry=[cells(9, 11, 9, 11)],
            crs=TINY.crs,
        ),
    )
    return ghsl_world, version


def test_centres_are_matched_by_the_people_they_share(
    classified: tuple[Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, version = classified
    monkeypatch.setattr(settlements, "LARGE", 20_000)
    data = UrbanicityData(root)
    summary = settlements.compare(data, version, "ghs_ucdb_r2024a")

    out_dir = data.classification_root(version) / "validation" / "settlements"
    table = pd.read_parquet(
        out_dir / "ghs_ucdb_r2024a" / "matches_2025q1.parquet"
    ).set_index("name")
    # Our one centre holds 30,000 of its people in West's cells and 20,000 in
    # East's; West is its match, so only that pair is mutual.
    assert table.loc["West", "our_id"] == 1
    assert table.loc["West", "overlap_population"] == pytest.approx(30_000)
    assert table.loc["East", "overlap_cells"] == 10
    assert table.loc["West", "mutual"]
    assert not table.loc["East", "mutual"]
    assert pd.isna(table.loc["Elsewhere", "our_id"])
    assert table.loc["East", "our_population"] == pytest.approx(50_000)

    assert summary["reference_centres"] == 3
    assert summary["reference_matched"] == 2
    assert summary["reference_people_matched_share"] == pytest.approx(45 / 105)
    assert (summary["our_centres"], summary["ours_matched"]) == (1, 1)
    assert summary["mutual_pairs"] == 1
    assert summary["ratio"]["median"] == pytest.approx(50 / 30)
    assert summary["ratio"]["share_close"] == 0.0
    assert summary["ratio_by_reference_size"]["0-100000"]["pairs"] == 1
    assert summary["ratio_by_reference_size"]["100000-500000"] == {"pairs": 0}
    assert summary["ours_matching_several"] == {"centres": 1, "reference_centres": 2}
    assert summary["largest_ratios"] == [
        {
            "name": "West",
            "country": "A",
            "reference_population": 30_000,
            "our_population": 50_000,
        }
    ]
    stored = yaml.safe_load((out_dir / "ghs_ucdb_r2024a" / "summary.yaml").read_text())
    assert stored["mutual_pairs"] == 1


def test_a_reference_with_no_shared_centres(classified: tuple[Path, str]) -> None:
    root, version = classified
    summary = settlements.compare(UrbanicityData(root), version, "ghs_wup_r2025a")
    assert summary["reference_matched"] == 0
    assert summary["ratio"] == {"pairs": 0}
    assert summary["largest_ratios"] == []


def test_the_reference_year_is_needed(classified: tuple[Path, str]) -> None:
    root, version = classified
    path = UrbanicityData(root).classification_root(version) / "specification.yaml"
    spec = yaml.safe_load(path.read_text())
    path.write_text(yaml.safe_dump({**spec, "time_points": ["2020q1"]}))
    with pytest.raises(ValueError, match="has no 2025q1"):
        settlements.compare(UrbanicityData(root), version, "ghs_ucdb_r2024a")


def test_settlements_command(classified: tuple[Path, str]) -> None:
    root, version = classified
    result = CliRunner().invoke(
        purun, ["validate", "settlements", version, "-o", str(root)]
    )
    assert result.exit_code == 0, result.output
    assert result.output.splitlines() == [
        "ghs_ucdb_r2024a: 2 of 3 matched, 1 mutual; population ratio median 1.67",
        "ghs_wup_r2025a: 0 of 1 matched, 0 mutual; population ratio median nan",
    ]
