"""Panels of fixed areas: where the people are, and how each classification divides them.

For each area and time point, one figure. Its first panel is the people in each
40 m pixel. Then, for each classification listed, its classes on the same
pixels and its default cut. A grid method's 1 km classes are shown through the
pixel view, so each pixel has its cell's class; a pixel method's are computed
for the area. In the United States, the Census's 2020 urban areas are drawn on
the cut.

Written to classifications/<first version>/validation/aoi_panels/<time
point>/<area>.png.
"""

from pathlib import Path

import click
import geopandas as gpd
import matplotlib as mpl
import numpy as np
import numpy.typing as npt
import pyproj
from affine import Affine
from matplotlib import colors, patches
from matplotlib import pyplot as plt
from rasterio import features
from rasterio.windows import Window
from rra_tools.cli_tools import with_output_directory
from rra_tools.shell_tools import mkdir
from shapely.geometry import box

from rra_population_urbanicity import constants as puc
from rra_population_urbanicity.classifiers.base import ClassScheme
from rra_population_urbanicity.classify import kernel
from rra_population_urbanicity.data import UrbanicityData
from rra_population_urbanicity.inputs import blocks
from rra_population_urbanicity.inputs.blocks import MODEL_CRS
from rra_population_urbanicity.reference import us_census
from rra_population_urbanicity.specification import (
    ClassificationSpecification,
    KernelDensityMethod,
    PopulaceSource,
)
from rra_population_urbanicity.tabulate.pixel_view import window_classes

mpl.use("Agg")

# Each area as (west, south, east, north) in degrees.
AREAS = {
    # The motivating example for counting density over land.
    "west_seattle": (-122.45, 47.48, -122.25, 47.62),
    "lusaka": (28.15, -15.55, 28.45, -15.30),
    "zinder": (8.92, 13.73, 9.06, 13.87),
    "kano": (8.40, 11.88, 8.65, 12.12),
    "dhaka": (90.0, 23.5, 90.8, 24.2),
    "nile_delta": (30.90, 30.60, 31.30, 30.95),
    "yogyakarta": (110.25, -7.95, 110.55, -7.65),
}
# Areas whose Census urban areas are drawn.
US_AREAS = {"west_seattle"}
URBAN_COLOUR, RURAL_COLOUR = "#E31A1C", "#E8E8E8"
PANEL_INCHES = 4.0

type CodeArray = npt.NDArray[np.uint8]


def area_window(bounds: tuple[float, float, float, float]) -> Window:
    """The window of the model grid covering a longitude/latitude box."""
    to_model = pyproj.Transformer.from_crs("EPSG:4326", MODEL_CRS, always_xy=True)
    x0, y0, x1, y1 = to_model.transform_bounds(*bounds)
    col0, row0 = ~blocks.MODEL_TRANSFORM @ (x0, y1)
    col1, row1 = ~blocks.MODEL_TRANSFORM @ (x1, y0)
    return Window(
        int(col0), int(row0), int(np.ceil(col1)) - int(col0), int(np.ceil(row1)) - int(row0)
    )


def classes(
    data: UrbanicityData,
    version: str,
    spec: ClassificationSpecification,
    scheme: ClassScheme,
    window: Window,
    time_point: str,
) -> CodeArray:
    """A classification's classes on the pixels of a window."""
    if isinstance(spec.method, KernelDensityMethod) and isinstance(
        spec.population, PopulaceSource
    ):
        codes, _ = kernel.classify_window(
            data, spec.method.parameters, spec.population, window, time_point
        )
        return codes
    transform = blocks.MODEL_TRANSFORM @ Affine.translation(window.col_off, window.row_off)
    return window_classes(
        transform,
        (int(window.height), int(window.width)),
        spec.grid,
        data.class_raster_path(version, time_point),
        scheme.nodata,
    )


def urban_area_outlines(data: UrbanicityData, window: Window) -> npt.NDArray[np.bool_]:
    """The edges of the Census's urban areas on the pixels of a window."""
    transform = blocks.MODEL_TRANSFORM @ Affine.translation(window.col_off, window.row_off)
    shape = (int(window.height), int(window.width))
    west, north = transform @ (0, 0)
    east, south = transform @ (shape[1], shape[0])
    area = gpd.GeoSeries([box(west, south, east, north)], crs=MODEL_CRS)
    shapes = gpd.read_file(
        us_census.urban_areas_path(data), bbox=area.to_crs("EPSG:4269").iloc[0]
    ).to_crs(MODEL_CRS)
    inside = features.rasterize(
        shapes.geometry, out_shape=shape, transform=transform, fill=0, default_value=1
    ).astype(bool)
    edges = np.zeros(shape, dtype=bool)
    edges[1:] |= inside[1:] != inside[:-1]
    edges[:, 1:] |= inside[:, 1:] != inside[:, :-1]
    return edges


def draw(
    data: UrbanicityData,
    versions: list[str],
    area: str,
    time_point: str,
    out_path: Path,
) -> None:
    window = area_window(AREAS[area])
    specs = [
        ClassificationSpecification.from_yaml(
            data.classification_root(v) / "specification.yaml"
        )
        for v in versions
    ]
    schemes = [
        ClassScheme.from_yaml(data.classification_root(v) / "scheme.yaml") for v in versions
    ]
    populace = [s.population for s in specs if isinstance(s.population, PopulaceSource)]
    if not populace:
        msg = "The panels need a classification of the Populace surface."
        raise ValueError(msg)
    people, _ = kernel.read_population(
        populace[0], time_point, window, puc.POPULATION_MODEL_ROOT
    )
    outlines = urban_area_outlines(data, window) if area in US_AREAS else None

    height = PANEL_INCHES * window.height / window.width
    figure, axes = plt.subplots(
        len(versions) + 1,
        2,
        figsize=(2 * PANEL_INCHES, (len(versions) + 1) * (height + 0.5)),
        squeeze=False,
    )
    axes[0, 0].imshow(
        np.where(people > 0, people, np.nan),
        norm=colors.LogNorm(vmin=0.1, vmax=max(float(people.max()), 1.0)),
        cmap="viridis",
        interpolation="nearest",
    )
    axes[0, 0].set_title(f"People per 40 m pixel, {time_point}")
    axes[0, 1].axis("off")
    for row, (version, spec, scheme) in enumerate(
        zip(versions, specs, schemes, strict=True), start=1
    ):
        codes = classes(data, version, spec, scheme, window, time_point)
        palette = np.full((256, 3), 1.0)
        for c in scheme.classes:
            palette[c.code] = colors.to_rgb(c.colour)
        axes[row, 0].imshow(palette[codes], interpolation="nearest")
        axes[row, 0].set_title(f"{version}: {spec.method.name} classes")
        present = [c for c in scheme.classes if (codes == c.code).any()]
        axes[row, 0].legend(
            handles=[patches.Patch(color=c.colour, label=c.label) for c in present],
            fontsize="xx-small",
            loc="lower left",
        )
        urban = np.isin(codes, [scheme.code(key) for key in scheme.default_cut])
        cut = np.where(
            urban[..., None], colors.to_rgb(URBAN_COLOUR), colors.to_rgb(RURAL_COLOUR)
        )
        if outlines is not None:
            cut[outlines] = (0.0, 0.0, 0.0)
        axes[row, 1].imshow(cut, interpolation="nearest")
        axes[row, 1].set_title(f"{version}: default cut")
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(area.replace("_", " ").title())
    figure.tight_layout()
    mkdir(out_path.parent, parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    plt.close(figure)


@click.command()
@click.argument("versions", nargs=-1, required=True)
@click.option("--time-point", default="2025q1", show_default=True)
@click.option(
    "--area", "areas", multiple=True, help="An area to draw; every area if none."
)
@with_output_directory(puc.URBANICITY_ROOT)
def validate_aoi_panels(
    versions: tuple[str, ...], time_point: str, areas: tuple[str, ...], output_dir: str
) -> None:
    """Draw panels of fixed areas for one or more classifications."""
    data = UrbanicityData(output_dir)
    out_dir = data.classification_root(versions[0]) / "validation" / "aoi_panels" / time_point
    if unknown := set(areas) - set(AREAS):
        msg = f"Unknown areas {sorted(unknown)}. Choose from {list(AREAS)}."
        raise click.BadParameter(msg)
    for area in areas or list(AREAS):
        path = out_dir / f"{area}.png"
        draw(data, list(versions), area, time_point, path)
        print(path)
