from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import rasterio
import yaml
from affine import Affine
from click.testing import CliRunner
from rra_tools import jobmon

from rra_population_urbanicity import admin
from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.classifiers.degurba import scheme
from rra_population_urbanicity.classifiers.degurba.scheme import DEGURBA_L2
from rra_population_urbanicity.cli import purun, putask
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.grid import GRIDS
from rra_population_urbanicity.inputs import blocks
from rra_population_urbanicity.inputs.blocks import MODEL_CRS, Block
from rra_population_urbanicity.inputs.providers import populace
from rra_population_urbanicity.specification import ClassificationSpecification
from rra_population_urbanicity.tabulate import classes, pixel_view, population
from tests.conftest import TINY, write_model_raster, write_raster
from tests.test_providers import POPULACE

SHAPES = "test_a2"
TIME_POINTS = ["2020q1", "2023q1"]
# The small frame's 10 x 7 pixels fall in four cells of `tiny_1km` (see
# test_aggregate): rows 5 and 6, columns 5 and 6.
NW, NE, SW, SE = (5, 5), (5, 6), (6, 5), (6, 6)


def spec(grid: str = TINY.name) -> ClassificationSpecification:
    return ClassificationSpecification.model_validate(
        {
            "population": POPULACE.model_dump(),
            "grid": grid,
            "land": {"source": "ghs_land", "release": "r2022a"},
            "built": {
                "source": "microsoft_density",
                "building_version": "v8",
                "resolution": "40",
            },
            "time_points": TIME_POINTS,
            "method": {"name": "degurba_r2023a", "parameter_set": "populace_default"},
        }
    )


def frame_units() -> npt.NDArray[np.int32]:
    """Unit 11 west of the meridian, 12 north-east, 21 south-east, and one pixel in none."""
    units = np.zeros((blocks.HEIGHT, blocks.WIDTH), dtype=np.int32)
    units[:, :3] = 11
    units[:2, 3:] = 12
    units[2:, 3:] = 21
    units[6, 9] = admin.NO_UNIT
    return units


def write_people(model_root: Path, time_point: str, people: npt.NDArray[Any]) -> None:
    size = blocks.GROUP_BLOCKS * blocks.BLOCK_PIXELS
    for group_x in range(2):
        path = populace.compiled_path(
            model_root, POPULACE, time_point, f"G-{group_x:04d}X-0000Y"
        )
        transform = blocks.MODEL_TRANSFORM @ Affine.translation(group_x * size, 0)
        write_model_raster(
            path, people[:, group_x * size : (group_x + 1) * size], transform
        )


def write_units(data: UrbanicityData, units: npt.NDArray[np.int32]) -> None:
    for block in blocks.all_blocks():
        path = data.admin_block_path(SHAPES, block.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        values = units[block.window.toslices()]
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=values.shape[0],
            width=values.shape[1],
            count=1,
            dtype="int32",
            crs=MODEL_CRS,
            transform=block.transform,
        ) as dst:
            dst.write(values, 1)


def write_grid(data: UrbanicityData, time_point: str, cells: dict[Any, float]) -> None:
    values = np.zeros(TINY.shape, dtype=np.float32)
    for cell, people in cells.items():
        values[cell] = people
    path = data.grid_path(TINY.name, "population", POPULACE, time_point)
    write_raster(path, values, np.nan)


@pytest.fixture
def table_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_frame: None
) -> Path:
    """Populace, the units' rasters and M2's grid on the small frame.

    One person in every pixel, and 100 in the top-left pixel in 2023. Units 11
    and 12 are in country 100, unit 21 in country 200.
    """
    monkeypatch.setitem(GRIDS, TINY.name, TINY)
    model_root = tmp_path / "population-model"
    monkeypatch.setattr(puc, "POPULATION_MODEL_ROOT", model_root)
    root = tmp_path / "output"
    root.mkdir()
    data = UrbanicityData(root)
    people = np.ones((blocks.HEIGHT, blocks.WIDTH))
    write_people(model_root, "2020q1", people)
    people[0, 0] = 100
    write_people(model_root, "2023q1", people)
    write_units(data, frame_units())
    write_grid(data, "2020q1", {NW: 6, NE: 14, SW: 15, SE: 35})
    write_grid(data, "2023q1", {NW: 105, NE: 14, SW: 15, SE: 35})

    inputs = admin.admin_inputs()
    inputs.mkdir(parents=True)
    pd.DataFrame(
        {
            "location_id": [1, 100, 200, 11, 12, 21],
            "path_to_top_parent": [
                "1",
                "1,100",
                "1,200",
                "1,100,1001,11",
                "1,100,1001,12",
                "1,200,2001,21",
            ],
            "ihme_loc_id": ["G", "AAA", "BBB", "AAA_11", "AAA_12", "BBB_21"],
            "most_detailed": [0, 0, 0, 1, 1, 1],
        }
    ).to_parquet(inputs / "hierarchy_test.parquet")
    pd.DataFrame(
        {
            "location_id": [100, 200] * 3,
            "year_id": [2020, 2020, 2023, 2023, 2024, 2024],
            "population": [35.0, 40.0, 134.0, 40.0, 134.0, 40.0],
        }
    ).to_parquet(inputs / population.GBD_POPULATION)
    monkeypatch.setitem(
        admin.SHAPES,
        SHAPES,
        admin.ShapesVersion(units=SHAPES, countries="test_a0", hierarchy="test"),
    )

    def run_parallel(**kwargs: Any) -> str:
        for key in kwargs["node_args"]["block-key"]:
            population.run_block(Path(kwargs["task_args"]["work-dir"]), key)
        return "done"

    monkeypatch.setattr(jobmon, "run_parallel", run_parallel)
    return root


def read_table(root: Path, time_point: str) -> pd.DataFrame:
    table_root = UrbanicityData(root).population_by_admin_root(
        TINY.name, POPULACE, SHAPES
    )
    return pd.read_parquet(table_root / f"{time_point}.parquet")


def test_people_by_cell_and_unit(table_world: Path) -> None:
    manifest_path = population.run(spec(), UrbanicityData(table_world), SHAPES, None)
    assert manifest_path is not None

    table = read_table(table_world, "2020q1")
    rows = [tuple(r) for r in table[["row", "col", "location_id"]].to_numpy()]
    # The south-east cell is split between unit 21 and no unit.
    assert rows == [(5, 5, 11), (5, 6, 12), (6, 5, 11), (6, 6, 0), (6, 6, 21)]
    assert table["population"].tolist() == [6, 14, 15, 1, 34]
    assert read_table(table_world, "2023q1")["population"].iloc[0] == 105

    manifest = yaml.safe_load(manifest_path.read_text())
    assert manifest["shapes"] == SHAPES
    entry = manifest["time_points"]["2020q1"]
    assert entry["rows"] == 5
    assert entry["cells"] == 4
    assert entry["people"] == entry["pixels_total"] == 70
    assert entry["people_in_no_unit"] == 1
    assert entry["people_beyond_grid"] == 0
    assert entry["grid_check"] == {"max_difference": 0.0, "max_relative": 0.0}
    assert entry["worst_unit_relative_error"] == 0
    # Country 100 has GBD's 35 people; country 200 has 34 of GBD's 40.
    assert entry["countries_against_gbd"] == {
        "countries": 2,
        "within_0.1%": 1,
        "within_1.0%": 1,
        "furthest": {200: 0.85, 100: 1.0},
    }
    # The work directory is removed once the tables are written.
    assert not any((table_world / "grids" / TINY.name / ".tabulate").iterdir())
    assert population.run(spec(), UrbanicityData(table_world), SHAPES, None) is None


def test_the_table_must_match_the_grid(table_world: Path) -> None:
    write_grid(UrbanicityData(table_world), "2020q1", {NW: 6, NE: 14, SW: 15, SE: 36})
    with pytest.raises(RuntimeError, match="The 2020q1 table fails its checks"):
        population.run(spec(), UrbanicityData(table_world), SHAPES, None)


def test_every_block_must_be_tabulated(
    table_world: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jobmon, "run_parallel", lambda **_: "done")
    with pytest.raises(RuntimeError, match="6 blocks have not been tabulated"):
        population.run(spec(), UrbanicityData(table_world), SHAPES, None)


def test_only_populace_is_tabulated_by_unit(table_world: Path) -> None:
    ghs = ClassificationSpecification.from_yaml(
        Path("specifications/ghs_smod_r2023a_v2_reproduction.yaml")
    )
    with pytest.raises(TypeError, match="model grid"):
        population.run(ghs, UrbanicityData(table_world), SHAPES, None)


def test_unit_rasters_are_checked(table_world: Path) -> None:
    data = UrbanicityData(table_world)
    units = frame_units()
    units[0, 0] = population.LOCATION_MASK + 1
    write_units(data, units)
    with pytest.raises(ValueError, match="too large"):
        population.read_units(data, SHAPES, Block(0, 0))
    path = data.admin_block_path(SHAPES, Block(1, 0).key)
    with rasterio.open(path, "r+") as dst:
        dst.transform = Block(0, 0).transform
    with pytest.raises(ValueError, match="where expected"):
        population.read_units(data, SHAPES, Block(1, 0))


def test_gbd_population_is_projected_past_its_last_year(table_world: Path) -> None:
    # As rra-population-model does: 2024's population times its growth since
    # 2023, for 2025.
    path = admin.admin_inputs() / population.GBD_POPULATION
    pd.DataFrame(
        {
            "location_id": [100] * 3,
            "year_id": [2022, 2023, 2024],
            "population": [80.0, 90.0, 100.0],
        }
    ).to_parquet(path)
    assert population.gbd_population(2023)[100] == 90
    assert population.gbd_population(2025)[100] == pytest.approx(100 * 100 / 90)
    assert population.gbd_population(2026)[100] == pytest.approx(100 * 100 / 80)


def test_the_commands_tabulate_and_run_one_block(
    table_world: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(admin.SHAPES, "lsae_1578_a2", admin.SHAPES[SHAPES])
    spec_path = tmp_path / "spec.yaml"
    spec().to_yaml(spec_path)
    # The command's choices are fixed when it is defined, so the test's units
    # stand in for a registered shapes version, by writing their rasters there.
    data = UrbanicityData(table_world)
    for block in blocks.all_blocks():
        source = data.admin_block_path(SHAPES, block.key)
        target = data.admin_block_path("lsae_1578_a2", block.key)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
    args = ["tabulate", "population", str(spec_path), "--shapes", "lsae_1578_a2"]
    result = CliRunner().invoke(purun, [*args, "-o", str(table_world)])
    assert result.exit_code == 0, result.output
    assert result.output.startswith("Done: ")
    result = CliRunner().invoke(purun, [*args, "-o", str(table_world)])
    assert result.output.strip() == "Nothing to tabulate."
    result = CliRunner().invoke(purun, [*args, "--region", "belgium"])
    assert "needs its own --output-dir" in result.output

    work_dir = tmp_path / "work"
    (work_dir / "blocks").mkdir(parents=True)
    plan = population.Plan(
        output_dir=str(table_world),
        grid=TINY.name,
        source=POPULACE,
        shapes=SHAPES,
        time_points=["2020q1"],
        blocks=["B-0002X-0000Y"],
        region=None,
    )
    (work_dir / "plan.yaml").write_text(yaml.safe_dump(plan.model_dump(mode="json")))
    result = CliRunner().invoke(
        putask,
        [
            "tabulate",
            "population-block",
            "--work-dir",
            str(work_dir),
            "--block-key",
            "B-0002X-0000Y",
        ],
    )
    assert result.exit_code == 0, result.output
    with np.load(work_dir / "blocks" / "B-0002X-0000Y.npz") as block:
        # Block (2, 0)'s 8 pixels: 4 in cell NE (unit 12) and 4 in cell SE (unit 21).
        assert block["population:2020q1"].tolist() == [4, 4]
        assert block["units:2020q1"].tolist() == [12, 21]


VERSION = "2026_11_24.001"


def write_classification(root: Path) -> None:
    """A classification of the small frame's four cells, without running one.

    One urban centre covers the cells north of the equator and the south-east
    one, so it crosses from country 100 into country 200; the south-west cell
    is a rural cluster. Every other cell is water.
    """
    data = UrbanicityData(root)
    version_root = data.classification_root(VERSION)
    version_root.mkdir(parents=True)
    spec().model_copy(update={"version": VERSION}).to_yaml(
        version_root / "specification.yaml"
    )
    DEGURBA_L2.to_yaml(version_root / "scheme.yaml")
    codes = np.full(TINY.shape, scheme.WATER, dtype=np.uint8)
    centres = np.zeros(TINY.shape, dtype=np.int32)
    for cell in (NW, NE, SE):
        codes[cell] = scheme.URBAN_CENTRE
        centres[cell] = 1
    codes[SW] = scheme.RURAL_CLUSTER
    for tp in TIME_POINTS:
        write_raster(data.class_raster_path(VERSION, tp), codes, scheme.NODATA)
        write_raster(data.urban_centre_raster_path(VERSION, tp), centres, 0)
        # The urban centre is also the one level-1 urban cluster.
        write_raster(data.urban_cluster_raster_path(VERSION, tp), centres, 0)


@pytest.fixture
def classified(table_world: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    population.run(spec(), UrbanicityData(table_world), SHAPES, None)
    write_classification(table_world)
    # The pixel check's blocks, on the small frame.
    monkeypatch.setattr(classes, "CHECK_BLOCKS", ("B-0000X-0000Y", "B-0002X-0001Y"))
    return table_world


def test_people_by_unit_and_class(classified: Path) -> None:
    manifest_path = classes.run(UrbanicityData(classified), VERSION, SHAPES)
    out_dir = manifest_path.parent
    by_class = pd.read_parquet(out_dir / "2020q1.parquet")
    assert {
        (int(r.location_id), int(r.class_code)): r.population
        for r in by_class.itertuples()
    } == {(0, 30): 1, (11, 13): 15, (11, 30): 6, (12, 30): 14, (21, 30): 34}

    manifest = yaml.safe_load(manifest_path.read_text())
    assert manifest["pixel_check"]["worst_relative_gap"] == 0
    assert manifest["urban_clusters"] == "tabulated by unit through their id raster"
    assert manifest["time_points"]["2020q1"]["people_in_no_unit"] == 1
    # Units 11, 12 and 21; location 0 is no unit and is not counted.
    assert manifest["time_points"]["2020q1"]["units"] == 3
    assert manifest["time_points"]["2023q1"]["people"] == 70 + 99


def test_stage_2_classes_each_unit(classified: Path) -> None:
    out_dir = classes.run(UrbanicityData(classified), VERSION, SHAPES).parent
    units = pd.read_parquet(out_dir / "stage2_2020q1.parquet").set_index(
        ["level", "location_id"]
    )
    # Unit 11: 6 of its 21 people in the urban centre and 15 in a rural cluster.
    assert units.loc[("admin2", 11), "stage2"] == "rural"
    assert units.loc[("admin2", 11), "urban_centre_share"] == pytest.approx(6 / 21)
    assert units.loc[("admin2", 12), "stage2"] == "city"
    # Admin 1 1001 has units 11 and 12: 20 of 35 in the urban centre.
    assert units.loc[("admin1", 1001), "stage2"] == "city"
    assert units.loc[("country", 200), "population"] == 34
    assert set(units.index.get_level_values("level")) == {"admin2", "admin1", "country"}


def test_each_cluster_has_a_main_country_and_admin_1(classified: Path) -> None:
    out_dir = classes.run(UrbanicityData(classified), VERSION, SHAPES).parent
    units = pd.read_parquet(out_dir / "cluster_units_2020q1.parquet").set_index(
        ["cluster_type", "cluster_id"]
    )
    centre = units.loc[("urban_centre", 1)]
    # 34 of its 55 people are in country 200, and one is in no unit.
    assert centre["population"] == 55
    assert centre["country_id"] == 200
    assert centre["iso3"] == "BBB"
    assert centre["country_share"] == pytest.approx(34 / 55)
    assert centre["admin1_id"] == 2001
    assert units.loc[("urban_cluster", 1), "country_id"] == 200
    rural = units.loc[("rural_cluster", 1)]
    assert rural[["population", "country_id", "country_share"]].tolist() == [15, 100, 1]
    people = pd.read_parquet(out_dir / "cluster_people_2020q1.parquet")
    centre_people = people[people.cluster_type == "urban_centre"]
    assert dict(
        zip(centre_people.location_id, centre_people.population, strict=True)
    ) == {
        0: 1,
        11: 6,
        12: 14,
        21: 34,
    }


def test_a_join_must_agree_with_the_pixels(
    classified: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = pixel_view.pixel_classes

    def shifted(*args: Any) -> npt.NDArray[np.uint8]:
        codes = real(*args)
        codes[0, 0] = scheme.WATER  # one pixel given the wrong class
        return codes

    monkeypatch.setattr(pixel_view, "pixel_classes", shifted)
    with pytest.raises(RuntimeError, match="disagree"):
        classes.run(UrbanicityData(classified), VERSION, SHAPES)


def test_the_pixel_view_gives_each_pixel_its_cells_class(classified: Path) -> None:
    data = UrbanicityData(classified)
    path = data.class_raster_path(VERSION, "2020q1")
    codes = pixel_view.pixel_classes(Block(0, 0), TINY, path, scheme.NODATA)
    # Block (0, 0): columns 0-2 west of the meridian, rows 0-1 north of the equator.
    assert codes.tolist() == [
        [30, 30, 30, 30],
        [30, 30, 30, 30],
        [13, 13, 13, 30],
        [13, 13, 13, 30],
    ]
    # A class raster covering only part of the grid leaves the rest as nodata.
    with rasterio.open(path) as src:
        part = src.read(1, window=rasterio.windows.Window(6, 0, 6, 12))
        profile = src.profile | {
            "width": 6,
            "transform": rasterio.windows.transform(
                rasterio.windows.Window(6, 0, 6, 12), src.transform
            ),
        }
    east = classified / "east.tif"
    with rasterio.open(east, "w", **profile) as dst:
        dst.write(part, 1)
    codes = pixel_view.pixel_classes(Block(0, 0), TINY, east, scheme.NODATA)
    assert (codes[:, :3] == scheme.NODATA).all()
    assert (codes[:, 3] == scheme.URBAN_CENTRE).all()
    assert (
        pixel_view.pixel_classes(Block(0, 0), TINY, east, scheme.NODATA).dtype
        == np.uint8
    )


def test_only_populace_classifications_are_tabulated(classified: Path) -> None:
    root = UrbanicityData(classified).classification_root(VERSION)
    ghs = ClassificationSpecification.from_yaml(
        Path("specifications/ghs_smod_r2023a_v2_reproduction.yaml")
    )
    ghs.to_yaml(root / "specification.yaml")
    with pytest.raises(TypeError, match="population table"):
        classes.run(UrbanicityData(classified), VERSION, SHAPES)


def test_a_scheme_without_levels_has_no_stage_2() -> None:
    plain = DEGURBA_L2.model_copy(update={"groups": {}})
    by_class = pd.DataFrame(
        {"location_id": [11], "class_code": [30], "population": [1.0]}
    )
    assert classes.stage2(by_class, plain, pd.DataFrame()) is None


def test_the_classes_command(classified: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The command's choices are fixed when it is defined, so the test's units
    # stand in for a registered shapes version, with their rasters and table.
    monkeypatch.setitem(admin.SHAPES, "lsae_1578_a2", admin.SHAPES[SHAPES])
    data = UrbanicityData(classified)
    for block in blocks.all_blocks():
        target = data.admin_block_path("lsae_1578_a2", block.key)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data.admin_block_path(SHAPES, block.key).read_bytes())
    population.run(spec(), data, "lsae_1578_a2", None)
    result = CliRunner().invoke(
        purun,
        [
            "tabulate",
            "classes",
            VERSION,
            "--shapes",
            "lsae_1578_a2",
            "-o",
            str(classified),
        ],
    )
    assert result.exit_code == 0, result.output
    assert result.output.startswith("Done: ")


def test_the_rt_deliverable(
    classified: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The test's units stand in for LSAE 1578's admin 2 units, with a fourth
    # where nobody lives; RT has rates for units 11 and 12 only.
    data = UrbanicityData(classified)
    monkeypatch.setitem(admin.SHAPES, "lsae_1578_a2", admin.SHAPES[SHAPES])
    hierarchy_path = admin.SHAPES[SHAPES].hierarchy_path
    hierarchy = pd.read_parquet(hierarchy_path)
    hierarchy["location_name"] = hierarchy["ihme_loc_id"]
    empty = {
        "location_id": 22,
        "path_to_top_parent": "1,200,2001,22",
        "ihme_loc_id": "BBB_22",
        "most_detailed": 1,
        "location_name": "BBB_22",
    }
    pd.concat([hierarchy, pd.DataFrame([empty])]).to_parquet(hierarchy_path)
    rates = tmp_path / "rates.csv"
    pd.DataFrame({"year_id": [2020, 2020], "location_id": [11, 12]}).to_csv(
        rates, index=False
    )
    monkeypatch.setattr(puc, "RT_RATES_PATH", rates)
    for block in blocks.all_blocks():
        target = data.admin_block_path("lsae_1578_a2", block.key)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data.admin_block_path(SHAPES, block.key).read_bytes())
    population.run(spec(), data, "lsae_1578_a2", None)
    classes.run(data, VERSION, "lsae_1578_a2")

    result = CliRunner().invoke(
        purun, ["tabulate", "deliverable", VERSION, "-o", str(classified)]
    )
    assert result.exit_code == 0, result.output
    table = pd.read_parquet(
        data.classification_root(VERSION)
        / "deliverables"
        / "admin2_urban_rural_lsae_1578.parquet"
    )
    first = table[table.time_point == "2020q1"].set_index("location_id")
    assert first.loc[
        11, ["population", "urban_population", "rural_population"]
    ].tolist() == [21, 6, 15]
    assert first.loc[11, "urban_share"] == pytest.approx(6 / 21)
    assert first.loc[21, "urban_share"] == 1
    assert first.loc[22, "population"] == 0
    assert pd.isna(first.loc[22, "urban_share"])
    assert first["has_rt_rates"].to_dict() == {11: True, 12: True, 21: False, 22: False}
    assert len(table) == 4 * len(TIME_POINTS)
    assert (
        data.classification_root(VERSION)
        / "deliverables"
        / "admin2_urban_rural_lsae_1578.csv"
    ).exists()


def test_a_classification_without_urban_cluster_ids(classified: Path) -> None:
    # Made before urban-cluster ids were saved: their units are left out.
    data = UrbanicityData(classified)
    for tp in TIME_POINTS:
        data.urban_cluster_raster_path(VERSION, tp).unlink()
    manifest_path = classes.run(data, VERSION, SHAPES)
    manifest = yaml.safe_load(manifest_path.read_text())
    assert manifest["urban_clusters"].startswith("not tabulated by unit")
    units = pd.read_parquet(manifest_path.parent / "cluster_units_2020q1.parquet")
    assert "urban_cluster" not in set(units.cluster_type)
