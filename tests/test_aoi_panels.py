from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import yaml
from click.testing import CliRunner
from shapely.geometry import box

from rra_population_urbanicity.classifiers.degurba import scheme
from rra_population_urbanicity.classify import runner
from rra_population_urbanicity.cli import purun
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.inputs.blocks import MODEL_CRS
from rra_population_urbanicity.reference import us_census
from rra_population_urbanicity.specification import ClassificationSpecification
from rra_population_urbanicity.validate import aoi_panels
from tests.conftest import TINY, kernel_spec, write_raster

# The small frame, in degrees: 120 m west to 280 m east, 80 m north to 200 m south.
FRAME = (-0.0010, -0.0017, 0.0025, 0.0007)


def degurba_version(root: Path) -> str:
    """A DEGURBA run on `tiny_1km` with the cell north-west of the origin an urban centre."""
    data = UrbanicityData(root)
    version = runner.reserve_version(data.classifications_root)
    spec = kernel_spec().model_dump()
    spec["method"] = {"name": "degurba_r2023a", "parameter_set": "populace_default"}
    spec["built"] = {
        "source": "microsoft_density",
        "building_version": "v8",
        "resolution": "40",
    }
    ClassificationSpecification.model_validate(spec).to_yaml(
        data.classification_root(version) / "specification.yaml"
    )
    scheme.DEGURBA_L2.to_yaml(data.classification_root(version) / "scheme.yaml")
    codes = np.full(TINY.shape, scheme.LOW_DENSITY_RURAL, dtype=np.uint8)
    codes[5, 5] = scheme.URBAN_CENTRE
    write_raster(data.class_raster_path(version, "2023q1"), codes, scheme.NODATA)
    return version


def test_panels_show_each_classification_on_the_same_pixels(
    kernel_world: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aoi_panels, "AREAS", {"frame": FRAME})
    monkeypatch.setattr(aoi_panels, "US_AREAS", {"frame"})
    data = UrbanicityData(kernel_world)
    path = us_census.urban_areas_path(data)
    path.parent.mkdir(parents=True)
    gpd.GeoDataFrame(
        {"NAME20": ["Town"]}, geometry=[box(-120, -40, 40, 80)], crs=MODEL_CRS
    ).to_crs("EPSG:4269").to_file(path)
    degurba = degurba_version(kernel_world)
    kernel = runner.run(kernel_spec(), data, None)

    window = aoi_panels.area_window(FRAME)
    assert (window.col_off, window.row_off) == (0, 0)
    codes = aoi_panels.classes(
        data,
        degurba,
        ClassificationSpecification.from_yaml(
            data.classification_root(degurba) / "specification.yaml"
        ),
        scheme.DEGURBA_L2,
        window,
        "2023q1",
    )
    # West of the meridian and north of the equator is the urban centre's cell.
    assert codes[0, 0] == scheme.URBAN_CENTRE
    assert codes[0, 4] == scheme.LOW_DENSITY_RURAL
    # The urban area covers columns 0-3 and rows 0-2.
    edges = aoi_panels.urban_area_outlines(data, window)
    assert edges[3, :4].all()
    assert edges[:3, 4].all()
    assert edges.sum() == 7

    result = CliRunner().invoke(
        purun,
        [
            "validate",
            "aoi-panels",
            kernel,
            degurba,
            "--time-point",
            "2023q1",
            "-o",
            str(kernel_world),
        ],
    )
    assert result.exit_code == 0, result.output
    figure = (
        data.classification_root(kernel)
        / "validation"
        / "aoi_panels"
        / "2023q1"
        / "frame.png"
    )
    assert figure.stat().st_size > 0


def test_panels_need_a_populace_classification(
    kernel_world: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(aoi_panels, "AREAS", {"frame": FRAME})
    data = UrbanicityData(kernel_world)
    version = degurba_version(kernel_world)
    path = data.classification_root(version) / "specification.yaml"
    spec = yaml.safe_load(path.read_text())
    spec["population"] = {"source": "ghs_pop", "release": "r2023a"}
    spec["land"] = {"source": "ghs_land", "release": "r2022a"}
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match="need a classification of the Populace"):
        aoi_panels.draw(data, [version], "frame", "2023q1", kernel_world / "panel.png")


def test_unknown_areas_are_refused(kernel_world: Path) -> None:
    result = CliRunner().invoke(
        purun,
        ["validate", "aoi-panels", "v", "--area", "atlantis", "-o", str(kernel_world)],
    )
    assert result.exit_code == 2
    assert "Unknown areas ['atlantis']" in result.output
