from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import rasterio
import yaml
from affine import Affine
from click.testing import CliRunner
from rra_tools import jobmon

from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.classifiers.degurba import scheme
from rra_population_urbanicity.classify import runner as classify_runner
from rra_population_urbanicity.cli import purun
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.grid import GRIDS, GridDefinition
from rra_population_urbanicity.inputs import aggregate, blocks, grids
from rra_population_urbanicity.inputs.aggregate import GridLayer
from rra_population_urbanicity.inputs.blocks import Block
from rra_population_urbanicity.inputs.providers import ghsl, populace
from rra_population_urbanicity.reference.ghsl import GhslRaster
from rra_population_urbanicity.specification import (
    ClassificationSpecification,
    MicrosoftWaterSource,
)
from tests.conftest import TINY, write_model_raster, write_raster
from tests.test_providers import POPULACE, write_water_tile

# The small frame's 10 x 7 pixels fall in four cells of `tiny_1km`, split by
# the prime meridian and the equator. Rows 5 and 6, columns 5 and 6:
#   (5, 5): 6 pixels, all in block (0, 0);
#   (5, 6): 14 pixels, from blocks (0, 0), (1, 0) and (2, 0);
#   (6, 5): 15 pixels, from blocks (0, 0) and (0, 1);
#   (6, 6): 35 pixels, from all six blocks.
NW, NE, SW, SE = (5, 5), (5, 6), (6, 5), (6, 6)
PIXELS = {NW: 6, NE: 14, SW: 15, SE: 35}
TIME_POINTS = ["2020q1", "2023q1"]
TINY_100M = GridDefinition(
    name="tiny_100m",
    crs=TINY.crs,
    cell_size=100.0,
    x_min=TINY.x_min,
    y_max=TINY.y_max,
    n_rows=120,
    n_cols=120,
)


def model_spec(land: dict[str, str], grid: str = TINY.name) -> dict[str, Any]:
    return {
        "population": POPULACE.model_dump(),
        "grid": grid,
        "land": land,
        "built": {
            "source": "microsoft_density",
            "building_version": "v8",
            "resolution": "40",
        },
        "time_points": TIME_POINTS,
        "method": {"name": "degurba_r2023a", "parameter_set": "populace_default"},
    }


WATER_LAND = {"source": "microsoft_water", "building_version": "v8"}
EITHER_LAND = {
    "source": "ghs_land_or_microsoft_water",
    "ghs_land_release": "r2022a",
    "building_version": "v8",
}


@pytest.fixture
def model_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_frame: None
) -> Path:
    """The model grid's inputs on the small frame, and an output root.

    One person in every pixel, and 100 in the top-left pixel in 2023. Block
    (0, 0) is half built up and the others have no buildings. The water mask
    has land everywhere north of the equator and west of the meridian, and
    no tile south-east of the origin. GHS-LAND has half land everywhere.
    """
    monkeypatch.setitem(GRIDS, TINY.name, TINY)
    monkeypatch.setattr(ghsl, "MOLL_100M", TINY_100M)
    model_root = tmp_path / "population-model"
    building_root = tmp_path / "building-density"
    monkeypatch.setattr(puc, "POPULATION_MODEL_ROOT", model_root)
    monkeypatch.setattr(puc, "BUILDING_DENSITY_ROOT", building_root)

    size = blocks.GROUP_BLOCKS * blocks.BLOCK_PIXELS
    for time_point in TIME_POINTS:
        people = np.ones((blocks.HEIGHT, blocks.WIDTH))
        if time_point == "2023q1":
            people[0, 0] = 100
        for group_x in range(2):
            path = populace.compiled_path(
                model_root, POPULACE, time_point, f"G-{group_x:04d}X-0000Y"
            )
            transform = blocks.MODEL_TRANSFORM @ Affine.translation(group_x * size, 0)
            write_model_raster(
                path, people[:, group_x * size : (group_x + 1) * size], transform
            )
    for block in blocks.all_blocks():
        density = np.full(block.shape, 0.5 if block == Block(0, 0) else 0.0)
        for quarter in ["2020q2", "2023q1"]:
            path = (
                building_root
                / "tiles"
                / "40m"
                / "microsoft_v8"
                / quarter
                / f"{block.key}_density.tif"
            )
            write_model_raster(path, density, block.transform)
    for tile_x, tile_y in [(1023, 1024), (1024, 1024), (1023, 1023)]:
        write_water_tile(building_root, tile_x, tile_y, np.zeros((512, 512)))

    root = tmp_path / "output"
    land = np.full(TINY_100M.shape, 5000, dtype=np.uint16)
    raster = GhslRaster(product="LAND", release="R2022A", epoch=2018, resolution=100)
    out_dir = UrbanicityData(root).reference_root(
        "ghsl", raster.product_dir, raster.release_dir
    )
    write_raster(out_dir / f"{raster.stem}.tif", land, 65535, TINY_100M)

    def run_parallel(**kwargs: Any) -> str:
        # The tasks, run here one after the other instead of on the cluster.
        for key in kwargs["node_args"]["block-key"]:
            aggregate.run_block(Path(kwargs["task_args"]["work-dir"]), key)
        return "done"

    monkeypatch.setattr(jobmon, "run_parallel", run_parallel)
    return root


def run(root: Path, *lands: dict[str, str]) -> Path | None:
    specs = [
        ClassificationSpecification.model_validate(model_spec(land)) for land in lands
    ]
    return aggregate.run(specs, UrbanicityData(root), None)


def read(root: Path, *parts: str) -> npt.NDArray[np.float32]:
    with rasterio.open(root / "grids" / TINY.name / Path(*parts)) as src:
        assert src.transform == TINY.transform
        values: npt.NDArray[np.float32] = src.read(1)
    return values


def cells(values: np.ndarray) -> dict[tuple[int, int], float]:
    return {cell: float(values[cell]) for cell in PIXELS}


def test_population_is_summed_from_every_block_into_each_cell(
    model_world: Path,
) -> None:
    work_dir = run(model_world, WATER_LAND)
    assert work_dir is not None
    assert not work_dir.exists()  # removed once the grids are written

    population = Path("population", "populace", "40m", "2026_10_02.001")
    assert cells(read(model_world, *population.parts, "2020q1.tif")) == PIXELS
    assert cells(read(model_world, *population.parts, "2023q1.tif"))[NW] == 6 + 99
    assert read(model_world, *population.parts, "2023q1.tif").sum() == 70 + 99

    manifest = yaml.safe_load(
        (model_world / "grids" / TINY.name / population / "manifest.yaml").read_text()
    )
    assert manifest["layer"] == "population"
    assert manifest["aggregation"] == "summed"
    check = manifest["time_points"]["2023q1"]["conservation"]
    assert check["pixels"] == check["grid"] == 169
    assert check["beyond_grid"] == 0
    assert check["worst_block_relative_error"] == 0
    assert manifest["time_points"]["2023q1"]["blocks"] == 6


def test_built_up_and_land_are_averaged_over_each_cells_pixels(
    model_world: Path,
) -> None:
    run(model_world, WATER_LAND, EITHER_LAND)

    # Block (0, 0) is half built up: 2 of the 14 pixels in (5, 6), for example.
    built = Path("built", "microsoft_density", "v8", "40m")
    assert cells(read(model_world, *built.parts, "2020q1.tif")) == pytest.approx(
        {NW: 0.5, NE: 0.5 * 2 / 14, SW: 0.5 * 6 / 15, SE: 0.5 * 2 / 35}
    )
    manifest = yaml.safe_load(
        (model_world / "grids" / TINY.name / built / "manifest.yaml").read_text()
    )
    assert manifest["time_points"]["2020q1"]["quarter_read"] == "2020q2"

    water = read(model_world, "land", "microsoft_water", "v8", "static.tif")
    assert cells(water) == {NW: 1.0, NE: 1.0, SW: 1.0, SE: 0.0}
    # No mask tile south-east of the origin, but GHS-LAND has half land there.
    either = read(
        model_world, "land", "ghs_land_or_microsoft_water", "r2022a", "v8", "static.tif"
    )
    assert cells(either) == {NW: 1.0, NE: 1.0, SW: 1.0, SE: 0.5}


def test_people_beyond_the_grid_are_recorded(
    model_world: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A grid that ends between the first and second rows of pixels.
    cut = TINY.model_copy(update={"name": "tiny_cut", "y_max": 40.0})
    monkeypatch.setitem(GRIDS, cut.name, cut)
    spec = ClassificationSpecification.model_validate(model_spec(WATER_LAND, cut.name))
    aggregate.run([spec], UrbanicityData(model_world), None)

    path = (
        model_world
        / "grids"
        / cut.name
        / "population"
        / "populace"
        / "40m"
        / "2026_10_02.001"
    )
    with rasterio.open(path / "2023q1.tif") as src:
        assert src.read(1).sum() == 169 - 100 - 9
    check = yaml.safe_load((path / "manifest.yaml").read_text())["time_points"][
        "2023q1"
    ]["conservation"]
    assert check["beyond_grid"] == 109
    assert check["grid"] + check["beyond_grid"] == check["pixels"]


def test_population_must_be_conserved_in_every_block(
    model_world: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A block that loses its people's sums: the total over all blocks is off
    # too, but by less than the block itself.
    aggregate_block = aggregate.aggregate_block

    def lossy(block: Block, *args: Any) -> dict[str, Any]:
        arrays = aggregate_block(block, *args)
        if block == Block(1, 0):
            for key in arrays:
                if key.startswith("sum:population"):
                    arrays[key] = arrays[key] * 0
        return arrays

    monkeypatch.setattr(aggregate, "aggregate_block", lossy)
    with pytest.raises(
        RuntimeError, match=r"overall and 1\.00e\+00 in the worst block"
    ):
        run(model_world, WATER_LAND)


def test_every_block_must_be_aggregated(
    model_world: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(jobmon, "run_parallel", lambda **_: "done")
    with pytest.raises(RuntimeError, match="6 blocks have not been aggregated"):
        run(model_world, WATER_LAND)


def test_grids_already_written_are_not_redone(model_world: Path) -> None:
    run(model_world, WATER_LAND)
    assert run(model_world, WATER_LAND) is None
    # Only the new land layer is aggregated.
    work_dir = run(model_world, EITHER_LAND)
    assert work_dir is not None


def test_specifications_must_share_a_grid(
    model_world: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = TINY.model_copy(update={"name": "tiny_other"})
    monkeypatch.setitem(GRIDS, other.name, other)
    specs = [
        ClassificationSpecification.model_validate(model_spec(WATER_LAND, name))
        for name in [TINY.name, other.name]
    ]
    with pytest.raises(ValueError, match="share one grid"):
        aggregate.run(specs, UrbanicityData(model_world), None)


def test_only_model_grid_layers_are_aggregated() -> None:
    reproduction = ClassificationSpecification.from_yaml(
        Path("specifications/ghs_smod_r2023a_v2_reproduction.yaml")
    )
    assert aggregate.grid_layers(reproduction) == []
    example = ClassificationSpecification.from_yaml(
        Path("specifications/populace_example.yaml")
    )
    layers = aggregate.grid_layers(example)
    assert [layer.layer for layer in layers] == ["population"] * 7 + ["built"] * 7
    assert layers[0].key == "population/populace/40m/2026_10_02.001/2020q1"


def test_a_layer_has_a_key_like_its_path() -> None:
    layer = GridLayer(
        layer="land",
        source=MicrosoftWaterSource(source="microsoft_water", building_version="v8"),
        time_point=puc.STATIC,
    )
    assert layer.key == "land/microsoft_water/v8/static"
    assert not layer.summed


def test_the_blocks_covering_a_region() -> None:
    keys = [b.key for b in aggregate.blocks_for_bounds(*puc.REGIONS["bangladesh"])]
    assert "B-0091X-0011Y" in keys  # Dhaka
    assert len(keys) == 3 * 3
    # A box past the edges of the model grid is cut to it.
    corner = aggregate.blocks_for_bounds(-180, -90, -179, -89)
    assert [b.key for b in corner] == ["B-0000X-0038Y"]


def test_the_command_runs_the_blocks_and_the_reduce(
    model_world: Path, tmp_path: Path
) -> None:
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(model_spec(WATER_LAND)))
    args = ["inputs", "aggregate", str(spec_path), "-o", str(model_world)]
    result = CliRunner().invoke(purun, args)
    assert result.exit_code == 0, result.output
    assert result.output.strip() == "Done."
    result = CliRunner().invoke(purun, args)
    assert result.output.strip() == "Nothing to aggregate."


def test_a_region_needs_its_own_output_directory(tmp_path: Path) -> None:
    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(model_spec(WATER_LAND)))
    result = CliRunner().invoke(
        purun, ["inputs", "aggregate", str(spec_path), "--region", "bangladesh"]
    )
    assert result.exit_code != 0
    assert "needs its own --output-dir" in result.output


def test_ghs_land_at_100_m_is_read_either_side_of_the_antimeridian(
    model_world: Path,
) -> None:
    # Points in the first and last columns of the grid are read as two windows.
    x = np.array([TINY.x_min + 50.0, -TINY.x_min - 50.0])
    y = np.array([0.0, 0.0])
    share = ghsl.sample_land_100m(UrbanicityData(model_world), x, y)
    assert share.tolist() == [0.5, 0.5]


def test_ghs_land_at_100_m_is_read_in_runs_of_points(
    model_world: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    land = np.full(TINY_100M.shape, 5000, dtype=np.uint16)
    land[60:, :60] = 10_000  # south-west of the origin is all land
    land[59, 59] = 65535  # and one missing cell north-west of it has none
    raster = GhslRaster(product="LAND", release="R2022A", epoch=2018, resolution=100)
    out_dir = UrbanicityData(model_world).reference_root(
        "ghsl", raster.product_dir, raster.release_dir
    )
    write_raster(out_dir / f"{raster.stem}.tif", land, 65535, TINY_100M)
    x = np.array([[-50.0, 50.0], [-50.0, 50.0]])
    y = np.array([[50.0, 50.0], [-50.0, -50.0]])
    expected = [[0.0, 0.5], [1.0, 0.5]]
    data = UrbanicityData(model_world)
    assert ghsl.sample_land_100m(data, x, y).tolist() == expected
    monkeypatch.setattr(ghsl, "SAMPLE_RUN_POINTS", 3)
    assert ghsl.sample_land_100m(data, x, y).tolist() == expected


def test_a_block_task_aggregates_one_block(model_world: Path) -> None:
    spec = ClassificationSpecification.model_validate(model_spec(WATER_LAND))
    work_dir = model_world / "work"
    plan = aggregate.Plan(
        output_dir=str(model_world),
        grid=TINY.name,
        layers=aggregate.grid_layers(spec),
        blocks=["B-0002X-0000Y"],
        region=None,
    )
    (work_dir / "blocks").mkdir(parents=True)
    (work_dir / "plan.yaml").write_text(yaml.safe_dump(plan.model_dump(mode="json")))
    result = CliRunner().invoke(
        aggregate.aggregate_block_task,
        ["--work-dir", str(work_dir), "--block-key", "B-0002X-0000Y"],
    )
    assert result.exit_code == 0, result.output
    with np.load(work_dir / "blocks" / "B-0002X-0000Y.npz") as block:
        # Block (2, 0)'s 8 pixels all fall in (5, 6) and (6, 6).
        assert block["pixels"].tolist() == [4, 4]
        assert float(block["total:population/populace/40m/2026_10_02.001/2023q1"]) == 8


def test_a_populace_specification_classifies_from_the_aggregated_grids(
    model_world: Path,
) -> None:
    spec = ClassificationSpecification.model_validate(model_spec(EITHER_LAND))
    data = UrbanicityData(model_world)
    aggregate.run([spec], data, None)

    land = grids.load(data, "land", spec.land, TINY, "2023q1", TINY.full_window)
    assert land.dtype == np.float64
    assert cells(land) == {NW: 1.0, NE: 1.0, SW: 1.0, SE: 0.5}

    version = classify_runner.run(spec, data, None)
    codes = {}
    for time_point in TIME_POINTS:
        with rasterio.open(data.class_raster_path(version, time_point)) as src:
            codes[time_point] = src.read(1)[NW]
    # 6 people on 1 km² of land in 2020, and 105 in 2023.
    assert codes == {
        "2020q1": scheme.VERY_LOW_DENSITY_RURAL,
        "2023q1": scheme.LOW_DENSITY_RURAL,
    }
