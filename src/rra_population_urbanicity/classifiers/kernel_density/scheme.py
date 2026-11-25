"""The `binary` scheme: urban or rural, and water."""

from rra_population_urbanicity.classifiers.base import ClassDef, ClassScheme

URBAN = 1
RURAL = 2
WATER = 10
NODATA = 255

BINARY = ClassScheme(
    name="binary",
    classes=[
        ClassDef(code=URBAN, key="urban", label="Urban", colour="#E31A1C"),
        ClassDef(code=RURAL, key="rural", label="Rural", colour="#33A02C"),
        ClassDef(code=WATER, key="water", label="Water", colour="#7AB6F5"),
    ],
    default_cut=["urban"],
    groups={},
    ordered=False,
    nodata=NODATA,
)
