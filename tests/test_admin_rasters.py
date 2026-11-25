from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import shapely
import yaml
from click.testing import CliRunner
from shapely.geometry import box

from rra_population_urbanicity import admin
from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.cli import purun
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.inputs import blocks
from rra_population_urbanicity.inputs.blocks import MODEL_CRS, Block
from rra_population_urbanicity.tabulate import admin_rasters

TEST_SHAPES = "test_a2"


def write_shapes(root: Path, name: str, shapes: dict[int, shapely.Polygon]) -> None:
    """Shapes given on the model grid, saved in longitude and latitude as the inputs are."""
    frame = gpd.GeoDataFrame(
        {"location_id": list(shapes)}, geometry=list(shapes.values()), crs=MODEL_CRS
    ).to_crs("EPSG:4326")
    path = root / "admin-inputs" / "raking" / "gbd-inputs" / f"shapes_{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)


@pytest.fixture
def admin_world(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, small_frame: None
) -> Path:
    """Three units in two countries on the small frame, which spans x -120 to 280
    and y 80 to -200 (pixel centres -100 ... 260 and 60 ... -180).

    Units: 11 (country 100) west of x = 0; 12 (country 100) north-east, up to
    x = 200; 21 (country 200) south-east. Countries: 100 north of y = 0; 200
    south of it, up to x = 200. So the north-east corner is in a country and in
    no unit (4 pixels), the south-east corner in a unit and in no country (10),
    and the south-west in a unit of the other country (15).
    """
    model_root = tmp_path / "population-model"
    monkeypatch.setattr(puc, "POPULATION_MODEL_ROOT", model_root)
    write_shapes(
        model_root,
        TEST_SHAPES,
        {11: box(-120, -200, 0, 80), 12: box(0, 0, 200, 80), 21: box(0, -200, 280, 0)},
    )
    write_shapes(
        model_root,
        "test_a0",
        {100: box(-120, 0, 280, 80), 200: box(-120, -200, 200, 0)},
    )
    hierarchy = pd.DataFrame(
        {
            "location_id": [1, 100, 200, 1001, 2001, 11, 12, 21],
            "path_to_top_parent": [
                "1",
                "1,100",
                "1,200",
                "1,100,1001",
                "1,200,2001",
                "1,100,1001,11",
                "1,100,1001,12",
                "1,200,2001,21",
            ],
            "most_detailed": [0, 0, 0, 0, 0, 1, 1, 1],
        }
    )
    hierarchy.to_parquet(admin.admin_inputs() / "hierarchy_test.parquet")
    monkeypatch.setitem(
        admin.SHAPES,
        TEST_SHAPES,
        admin.ShapesVersion(units=TEST_SHAPES, countries="test_a0", hierarchy="test"),
    )
    output = tmp_path / "output"
    output.mkdir()
    return output


def read_frame(data: UrbanicityData) -> np.ndarray:
    """The whole small frame, put together from its block rasters."""
    frame = np.zeros((blocks.HEIGHT, blocks.WIDTH), dtype=np.int32)
    for block in blocks.all_blocks():
        with rasterio.open(data.admin_block_path(TEST_SHAPES, block.key)) as src:
            assert src.transform.almost_equals(block.transform)
            frame[block.window.toslices()] = src.read(1)
    return frame


def test_each_pixel_takes_the_unit_containing_its_centre(admin_world: Path) -> None:
    data = UrbanicityData(admin_world)
    manifest_path = admin_rasters.run(data, TEST_SHAPES, None)
    assert manifest_path is not None

    expected = np.zeros((7, 10), dtype=np.int32)
    expected[:, :3] = 11
    expected[:2, 3:8] = 12
    expected[2:, 3:] = 21
    np.testing.assert_array_equal(read_frame(data), expected)

    manifest = yaml.safe_load(manifest_path.read_text())
    assert manifest["blocks"] == 6
    assert manifest["coverage_pixels"] == {
        "in_a_country_and_no_unit": 4,
        "in_a_unit_and_no_country": 10,
        "in_another_country": 15,
    }
    assert manifest["coverage_largest"]["in_another_country"] == {200: 15}
    # Made once: a second run has nothing to do.
    assert admin_rasters.run(data, TEST_SHAPES, None) is None


def test_units_take_their_admin_1_and_country_from_the_hierarchy(
    admin_world: Path,
) -> None:
    parents = admin.unit_parents(admin.SHAPES[TEST_SHAPES]).set_index("location_id")
    assert parents.loc[21].tolist() == [200, 2001]
    assert parents.loc[100].tolist() == [100, 100]  # a country is its own admin 1
    assert 1 not in parents.index  # the world has no country


@pytest.mark.usefixtures("small_frame")
def test_pixels_past_the_east_edge_take_the_shapes_at_the_west_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A world 400 m wide, whose east edge, x = 200, cuts the frame's last block.
    monkeypatch.setattr(admin_rasters, "WORLD_X_MAX", 200.0)
    geometry = np.array([box(-400, 0, -120, 80)])
    shapes = admin_rasters.Shapes(geometry, np.array([31]), shapely.STRtree(geometry))
    last = admin_rasters.burn(shapes, Block(2, 0))
    assert last.tolist() == [[31, 31], [31, 31], [0, 0], [0, 0]]
    # A block with no shape near it has no unit anywhere.
    assert not admin_rasters.burn(shapes, Block(0, 1)).any()


def test_a_region_needs_its_own_output_directory() -> None:
    result = CliRunner().invoke(
        purun,
        [
            "tabulate",
            "admin-rasters",
            "--shapes",
            "lsae_1578_a2",
            "--region",
            "belgium",
        ],
    )
    assert result.exit_code != 0
    assert "needs its own --output-dir" in result.output


def test_the_command_rasterizes_once(
    admin_world: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The command's choices are fixed when it is defined, so the test's shapes
    # stand in for a registered version.
    monkeypatch.setitem(admin.SHAPES, "lsae_1578_a2", admin.SHAPES[TEST_SHAPES])
    args = [
        "tabulate",
        "admin-rasters",
        "--shapes",
        "lsae_1578_a2",
        "-o",
        str(admin_world),
    ]
    result = CliRunner().invoke(purun, args)
    assert result.exit_code == 0, result.output
    assert result.output.startswith("Done: ")
    result = CliRunner().invoke(purun, args)
    assert result.output.strip() == "Already made."
    root = admin_world / "admin-rasters" / "40m" / "lsae_1578_a2"
    assert (root / "coverage.csv").exists()


def test_shapes_a_hair_past_the_antimeridian_stay_on_their_side(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # As in some of LSAE 1578's shapes in Fiji and Chukotka: a vertex at
    # 180.0000001°, which a projection would wrap to the far west.
    monkeypatch.setattr(puc, "POPULATION_MODEL_ROOT", tmp_path)
    past = shapely.Polygon([(179, 0), (180.0000001, 0), (180.0000001, 1), (179, 1)])
    path = admin.shapes_path("edge")
    path.parent.mkdir(parents=True)
    gpd.GeoDataFrame({"location_id": [1]}, geometry=[past], crs="EPSG:4326").to_parquet(
        path
    )

    west, _, east, _ = admin.read_shapes("edge", MODEL_CRS).total_bounds
    assert west > 19_900_000  # 179° east, not near -20,037,508
    assert east == pytest.approx(admin_rasters.WORLD_X_MAX)

    gpd.GeoDataFrame({"location_id": [1]}, geometry=[past], crs=MODEL_CRS).to_parquet(
        path
    )
    with pytest.raises(ValueError, match="not in longitude and latitude"):
        admin.read_shapes("edge", MODEL_CRS)


@pytest.mark.usefixtures("small_frame")
def test_units_and_countries_from_one_file_of_every_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # As GBD's: paths run world, super-region, region, country, unit, and one
    # file holds the shapes of every level. Only the units are drawn, and
    # only the country is checked against them.
    model_root = tmp_path / "population-model"
    monkeypatch.setattr(puc, "POPULATION_MODEL_ROOT", model_root)
    write_shapes(
        model_root,
        "deep",
        {
            800: box(-400, -400, 400, 400),  # a region, larger than everything
            300: box(-120, -200, 280, 80),
            31: box(-120, -200, 0, 80),
            32: box(0, -200, 280, 80),
        },
    )
    pd.DataFrame(
        {
            "location_id": [1, 700, 800, 300, 31, 32],
            "path_to_top_parent": [
                "1",
                "1,700",
                "1,700,800",
                "1,700,800,300",
                "1,700,800,300,31",
                "1,700,800,300,32",
            ],
            "most_detailed": [0, 0, 0, 0, 1, 1],
        }
    ).to_parquet(admin.admin_inputs() / "hierarchy_deep.parquet")
    version = admin.ShapesVersion(
        units="deep", countries="deep", hierarchy="deep", country_level=3
    )
    monkeypatch.setitem(admin.SHAPES, "deep", version)
    assert version.unit_ids() == [31, 32]
    assert version.country_ids() == [300]

    data = UrbanicityData(tmp_path / "output")
    manifest_path = admin_rasters.run(data, "deep", None)
    assert manifest_path is not None
    manifest = yaml.safe_load(manifest_path.read_text())
    assert manifest["coverage_pixels"] == dict.fromkeys(admin_rasters.KINDS, 0)
    with rasterio.open(data.admin_block_path("deep", Block(0, 0).key)) as src:
        assert src.read(1)[0].tolist() == [31, 31, 31, 32]

    parents = admin.unit_parents(version).set_index("location_id")
    assert parents.loc[31].tolist() == [300, 31]
    assert parents.loc[300].tolist() == [300, 300]
    assert 800 not in parents.index  # above the countries
