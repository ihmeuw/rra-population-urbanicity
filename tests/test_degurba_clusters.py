import numpy as np
import pytest
from affine import Affine
from rasterio import warp

from rra_population_urbanicity.classify.degurba import clusters, rules
from tests.test_degurba_rules import BASE, scene

TRANSFORM = Affine(1000.0, 0.0, 0.0, 0.0, -1000.0, 0.0)
CRS = "ESRI:54009"


def test_the_cluster_table() -> None:
    population = np.zeros((22, 22))
    population[2:8, 2:8] = 2000  # two urban centres touching at a corner
    population[8:14, 8:14] = 2000
    population[17:19, 2:5] = 400  # a rural cluster of 2,400 people
    inputs = scene(population)
    codes = rules.classify(inputs, BASE)
    table = clusters.cluster_table(codes, inputs, BASE, TRANSFORM, CRS)

    assert table.groupby("cluster_type").size().to_dict() == {
        "urban_centre": 2,  # rook contiguity keeps them apart
        "urban_cluster": 1,  # queen contiguity joins them
        "rural_cluster": 1,
    }
    rows = table.set_index(["cluster_type", "cluster_id"])
    first = rows.loc[("urban_centre", 1)]
    assert first[["class_code", "population", "cells", "land_km2"]].tolist() == [
        30,
        72_000,
        36,
        36,
    ]
    assert first[["row_start", "row_stop", "col_start", "col_stop"]].tolist() == [
        2,
        8,
        2,
        8,
    ]
    lon, lat = warp.transform(CRS, "EPSG:4326", [5000.0], [-5000.0])
    assert first["centroid_lon"] == pytest.approx(lon[0])
    assert first["centroid_lat"] == pytest.approx(lat[0])

    urban = rows.loc[("urban_cluster", 1)]
    assert urban[["class_code", "population", "cells"]].tolist() == [30, 144_000, 72]
    assert urban[["row_start", "row_stop", "col_start", "col_stop"]].tolist() == [
        2,
        14,
        2,
        14,
    ]
    rural = rows.loc[("rural_cluster", 1)]
    assert rural[["class_code", "population", "cells"]].tolist() == [13, 2400, 6]


def test_a_scene_without_clusters_gives_an_empty_table() -> None:
    inputs = scene(np.zeros((3, 3)))
    codes = rules.classify(inputs, BASE)
    table = clusters.cluster_table(codes, inputs, BASE, TRANSFORM, CRS)
    assert table.empty
    assert "centroid_lon" in table.columns
