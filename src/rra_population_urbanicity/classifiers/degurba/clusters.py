"""DEGURBA's clusters as a table, one row per cluster.

Urban centres and level-1 urban clusters are the method's own entities, as
`classify` identifies them. Each urban centre is generalised on its own, so two
can meet in the class raster and still be two centres, as in GHS-SMOD's
urban-centre entities. An urban cluster leaves out class 23 cells below
`ucl_density`, so it can't always be labelled again from the class raster:
such a cell is not counted, and two urban clusters that meet only through one
stay two. The other types are read off the final class raster: each type's
cells, labelled with the contiguity of the rule that forms them. Cluster ids
run from 1 within each type and time point.
"""

import numpy as np
import pandas as pd
import rasterio.transform
from affine import Affine
from rasterio import warp
from scipy import ndimage

from rra_population_urbanicity.classifiers.degurba import rules, scheme
from rra_population_urbanicity.classifiers.degurba.parameters import (
    Contiguity,
    DegurbaParameters,
)


def cluster_types(
    params: DegurbaParameters,
) -> dict[str, tuple[list[int], Contiguity]]:
    """The types read off the class raster: their class codes and contiguity."""
    return {
        "dense_urban_cluster": ([scheme.DENSE_URBAN_CLUSTER], params.duc_contiguity),
        "semi_dense_urban_cluster": (
            [scheme.SEMI_DENSE_URBAN_CLUSTER],
            params.sduc_contiguity,
        ),
        "rural_cluster": ([scheme.RURAL_CLUSTER], params.rc_contiguity),
    }


def summarise(
    cluster_type: str,
    labels: rules.LabelArray,
    n: int,
    codes: rules.CodeArray,
    inputs: rules.DegurbaInputs,
    transform: Affine,
    crs: str,
) -> pd.DataFrame:
    """One row for each of the clusters labelled 1 to `n` in `labels`."""
    rows, cols = np.nonzero(labels)
    ids = labels[rows, cols]
    people = inputs.population[rows, cols]
    row, col = rows.astype(np.float64), cols.astype(np.float64)

    def total(values: rules.FloatArray) -> rules.FloatArray:
        result: rules.FloatArray = np.bincount(ids, weights=values, minlength=n + 1)[1:]
        return result

    cells, population = total(np.ones_like(people)), total(people)
    # Population-weighted, as GHS-SMOD's entity centroids are; a cluster with
    # no population takes the mean of its cells' centres.
    weighted = population > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        row_mean = np.where(
            weighted, total(row * people) / population, total(row) / cells
        )
        col_mean = np.where(
            weighted, total(col * people) / population, total(col) / cells
        )
    xs, ys = rasterio.transform.xy(transform, row_mean, col_mean, offset="center")
    lon, lat = warp.transform(crs, "EPSG:4326", xs, ys)
    class_code = np.zeros(n + 1, dtype=np.uint8)
    np.maximum.at(class_code, ids, codes[rows, cols])
    boxes = ndimage.find_objects(labels)
    return pd.DataFrame(
        {
            "cluster_type": cluster_type,
            "cluster_id": np.arange(1, n + 1, dtype=np.int32),
            "class_code": class_code[1:],
            "population": population,
            "cells": cells.astype(np.int64),
            "land_km2": total(inputs.land[rows, cols]) * inputs.cell_area_km2,
            "centroid_lon": np.asarray(lon, dtype=np.float64),
            "centroid_lat": np.asarray(lat, dtype=np.float64),
            "row_start": np.array([box[0].start for box in boxes], dtype=np.int64),
            "row_stop": np.array([box[0].stop for box in boxes], dtype=np.int64),
            "col_start": np.array([box[1].start for box in boxes], dtype=np.int64),
            "col_stop": np.array([box[1].stop for box in boxes], dtype=np.int64),
        }
    )


def cluster_table(
    classification: rules.Classification,
    inputs: rules.DegurbaInputs,
    params: DegurbaParameters,
    transform: Affine,
    crs: str,
) -> pd.DataFrame:
    """One row per cluster: its type, id, class, population, size and extent.

    `class_code` is the cluster's class or, for an urban cluster, the most
    urban class it contains. The extent is in cells of the class raster and
    half-open: rows `row_start` to `row_stop` - 1, and likewise for columns.
    `transform` and `crs` place the class raster.
    """
    codes = classification.codes
    frames = [
        summarise(cluster_type, ids, int(ids.max()), codes, inputs, transform, crs)
        for cluster_type, ids in (
            ("urban_centre", classification.urban_centres),
            ("urban_cluster", classification.urban_clusters),
        )
    ]
    for cluster_type, (class_codes, contiguity) in cluster_types(params).items():
        labels, n = rules.label(np.isin(codes, class_codes), contiguity)
        frames.append(summarise(cluster_type, labels, n, codes, inputs, transform, crs))
    return pd.concat(frames, ignore_index=True)
