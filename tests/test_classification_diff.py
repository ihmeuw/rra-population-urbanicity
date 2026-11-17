import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import yaml
from click.testing import CliRunner
from shapely.geometry import box

from rra_population_urbanicity import admin
from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.classifiers.degurba import scheme
from rra_population_urbanicity.classify import runner
from rra_population_urbanicity.cli import purun
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.grid import GRIDS
from rra_population_urbanicity.specification import ClassificationSpecification
from rra_population_urbanicity.validate import classification_diff
from tests.conftest import TINY, ghsl_spec

WEST, EAST = 101, 102


@pytest.fixture
def countries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Two countries splitting `tiny_1km` at the central meridian.

    The eastern one stops 2 km south of the equator, so the cells south-east
    of it are in no country.
    """
    model_root = tmp_path / "population-model"
    monkeypatch.setattr(puc, "POPULATION_MODEL_ROOT", model_root)
    shapes = gpd.GeoDataFrame(
        {"location_id": [WEST, EAST]},
        geometry=[box(-6000, -6000, 0, 6000), box(0, -2000, 6000, 6000)],
        crs=TINY.crs,
    ).to_crs("EPSG:4326")
    path = admin.shapes_path(classification_diff.SHAPES)
    path.parent.mkdir(parents=True)
    shapes.to_parquet(path)
    return path


def test_each_cell_is_in_the_country_containing_its_centre(
    ghsl_world: Path, countries: Path
) -> None:
    data = UrbanicityData(ghsl_world)
    locations = admin.location_raster(data, TINY, classification_diff.SHAPES)
    assert locations.dtype == np.int32
    assert (locations[:, :6] == WEST).all()
    assert (locations[:8, 6:] == EAST).all()
    assert (locations[8:, 6:] == admin.NO_UNIT).all()

    # Made once and kept: later calls read it back.
    countries.unlink()
    again = admin.location_raster(data, TINY, classification_diff.SHAPES)
    np.testing.assert_array_equal(again, locations)


def classify_twice(root: Path) -> tuple[str, str]:
    """A classification, and a copy of it with the town as class 23."""
    data = UrbanicityData(root)
    spec = ClassificationSpecification.model_validate(ghsl_spec())
    version = runner.run(spec, data, None)
    other = runner.reserve_version(data.classifications_root)
    shutil.copytree(
        data.classification_root(version),
        data.classification_root(other),
        dirs_exist_ok=True,
    )
    path = data.class_raster_path(other, "2020q1")
    with rasterio.open(path, "r+") as dst:
        codes = dst.read(1)
        codes[2:7, 2:7] = scheme.DENSE_URBAN_CLUSTER
        dst.write(codes, 1)
    return version, other


@pytest.mark.usefixtures("countries")
def test_the_people_and_cells_that_change_class(ghsl_world: Path) -> None:
    version, other = classify_twice(ghsl_world)
    summary = classification_diff.compare(UrbanicityData(ghsl_world), version, other)

    first = summary["time_points"]["2020q1"]
    # The town: 25 cells of 2,000 people.
    assert first["cells"]["changed"] == 25
    assert first["population"]["changed"] == 50_000
    assert first["population"]["changed_share"] == 1.0
    assert first["other_population"]["changed"] == 50_000
    assert first["classes"][scheme.URBAN_CENTRE]["population"] == [50_000, 0]
    assert first["classes"][scheme.DENSE_URBAN_CLUSTER]["cells"] == [0, 25]
    assert summary["time_points"]["2025q1"]["cells"]["changed"] == 0

    out_dir = (
        ghsl_world
        / "classifications"
        / version
        / "validation"
        / "classification_diff"
        / other
    )
    saved = yaml.safe_load((out_dir / "summary.yaml").read_text())
    assert saved["shapes"] == "lsae_1578_a0"
    confusion = pd.read_csv(out_dir / "confusion_2020q1.csv").set_index(
        ["class_code", "other_class_code"]
    )
    town = confusion.loc[(scheme.URBAN_CENTRE, scheme.DENSE_URBAN_CLUSTER)]
    assert town["cells"] == 25
    # The town is in rows and columns 2-6: column 6 is east of the meridian.
    by_country = pd.read_parquet(out_dir / "by_country_2020q1.parquet")
    changed = by_country[by_country.class_code != by_country.other_class_code]
    assert changed.set_index("location_id")["cells"].to_dict() == {WEST: 20, EAST: 5}
    assert (by_country.cells > 0).all()


@pytest.mark.usefixtures("countries")
def test_the_command_prints_the_changes(ghsl_world: Path) -> None:
    version, other = classify_twice(ghsl_world)
    result = CliRunner().invoke(
        purun, ["validate", "diff", version, other, "-o", str(ghsl_world)]
    )
    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[0] == (
        "2020q1: 50,000 people (100.00%) and 25 cells change class"
    )


@pytest.mark.usefixtures("countries")
def test_classifications_must_be_comparable(
    ghsl_world: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = UrbanicityData(ghsl_world)
    version = runner.run(
        ClassificationSpecification.model_validate(ghsl_spec()), data, None
    )

    other_grid = TINY.model_copy(update={"name": "tiny_other"})
    monkeypatch.setitem(GRIDS, other_grid.name, other_grid)
    spec = ghsl_spec() | {"grid": other_grid.name}
    other = runner.reserve_version(data.classifications_root)
    ClassificationSpecification.model_validate(spec).to_yaml(
        data.classification_root(other) / "specification.yaml"
    )
    with pytest.raises(ValueError, match="different grids"):
        classification_diff.compare(data, version, other)

    later = ghsl_spec() | {"time_points": ["2030q1"]}
    ClassificationSpecification.model_validate(later).to_yaml(
        data.classification_root(other) / "specification.yaml"
    )
    with pytest.raises(ValueError, match="share no time point"):
        classification_diff.compare(data, version, other)

    monkeypatch.setitem(puc.REGIONS, "east", (0.001, -0.06, 0.06, 0.06))
    region = runner.run(
        ClassificationSpecification.model_validate(ghsl_spec()), data, "east"
    )
    with pytest.raises(ValueError, match="different windows"):
        classification_diff.compare(data, version, region)
