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
from rra_population_urbanicity.cli import purun, putask
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.grid import GRIDS
from rra_population_urbanicity.inputs import blocks
from rra_population_urbanicity.inputs.blocks import MODEL_CRS, Block
from rra_population_urbanicity.inputs.providers import populace
from rra_population_urbanicity.specification import ClassificationSpecification
from rra_population_urbanicity.tabulate import population
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
