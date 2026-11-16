"""The `degurba_l2` scheme: the level-2 classes of the Degree of Urbanisation.

Codes and colours are GHS-SMOD's own (its .clr colour table), so our class
rasters and figures read like GHS-SMOD's.
"""

from rra_population_urbanicity.classifiers.base import ClassDef, ClassScheme

URBAN_CENTRE = 30
DENSE_URBAN_CLUSTER = 23
SEMI_DENSE_URBAN_CLUSTER = 22
SUBURBAN = 21
RURAL_CLUSTER = 13
LOW_DENSITY_RURAL = 12
VERY_LOW_DENSITY_RURAL = 11
WATER = 10
NODATA = 255

DEGURBA_L2 = ClassScheme(
    name="degurba_l2",
    classes=[
        ClassDef(
            code=URBAN_CENTRE,
            key="urban_centre",
            label="Urban centre",
            colour="#FF0000",
        ),
        ClassDef(
            code=DENSE_URBAN_CLUSTER,
            key="dense_urban_cluster",
            label="Dense urban cluster",
            colour="#732600",
        ),
        ClassDef(
            code=SEMI_DENSE_URBAN_CLUSTER,
            key="semi_dense_urban_cluster",
            label="Semi-dense urban cluster",
            colour="#A87000",
        ),
        ClassDef(
            code=SUBURBAN,
            key="suburban",
            label="Suburban or peri-urban",
            colour="#FFFF00",
        ),
        ClassDef(
            code=RURAL_CLUSTER,
            key="rural_cluster",
            label="Rural cluster",
            colour="#375623",
        ),
        ClassDef(
            code=LOW_DENSITY_RURAL,
            key="low_density_rural",
            label="Low density rural",
            colour="#ABCD66",
        ),
        ClassDef(
            code=VERY_LOW_DENSITY_RURAL,
            key="very_low_density_rural",
            label="Very low density rural",
            colour="#CDF57A",
        ),
        ClassDef(code=WATER, key="water", label="Water", colour="#7AB6F5"),
    ],
    # Level-1 urban: urban centres and urban clusters.
    default_cut=[
        "urban_centre",
        "dense_urban_cluster",
        "semi_dense_urban_cluster",
        "suburban",
    ],
    groups={
        "degurba_l1": {
            "city": ["urban_centre"],
            "town": ["dense_urban_cluster", "semi_dense_urban_cluster", "suburban"],
            "rural": ["rural_cluster", "low_density_rural", "very_low_density_rural"],
            "water": ["water"],
        }
    },
    ordered=True,
    nodata=NODATA,
)
