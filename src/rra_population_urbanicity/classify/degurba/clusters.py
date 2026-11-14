"""DEGURBA's clusters as a table, one row per cluster.

The clusters are read off the final class raster: each type's cells, labelled
with the contiguity of the rule that forms them. The table therefore matches
the raster, and an urban centre includes the cells that smoothing and gap
filling added. Cluster ids run from 1 within each type and time point.
"""

import numpy as np
import pandas as pd
import rasterio.transform
from affine import Affine
from rasterio import warp
from scipy import ndimage

from rra_population_urbanicity.classify.degurba import rules, scheme
from rra_population_urbanicity.classify.degurba.parameters import (
    Contiguity,
    DegurbaParameters,
)

URBAN_CLUSTER_CODES = [
    scheme.URBAN_CENTRE,
    scheme.DENSE_URBAN_CLUSTER,
    scheme.SEMI_DENSE_URBAN_CLUSTER,
    scheme.SUBURBAN,
]


def cluster_types(
    params: DegurbaParameters,
) -> dict[str, tuple[list[int], Contiguity]]:
    """Each cluster type's class codes and contiguity."""
    return {
        "urban_centre": ([scheme.URBAN_CENTRE], params.uc_contiguity),
        "dense_urban_cluster": ([scheme.DENSE_URBAN_CLUSTER], params.duc_contiguity),
        "semi_dense_urban_cluster": (
            [scheme.SEMI_DENSE_URBAN_CLUSTER],
            params.sduc_contiguity,
        ),
        "urban_cluster": (URBAN_CLUSTER_CODES, params.ucl_contiguity),
        "rural_cluster": ([scheme.RURAL_CLUSTER], params.rc_contiguity),
    }


def cluster_table(
    codes: rules.CodeArray,
    inputs: rules.DegurbaInputs,
    params: DegurbaParameters,
    transform: Affine,
    crs: str,
) -> pd.DataFrame:
    """One row per cluster: its type, id, class, population, size and extent.

    `class_code` is the cluster's class or, for an urban cluster, the most
    urban class it contains. The centroid is the mean of its cells' centres.
    The extent is in cells of the class raster and half-open: rows `row_start`
    to `row_stop` - 1, and likewise for columns. `transform` and `crs` place
    the class raster.
    """
    frames = []
    for cluster_type, (class_codes, contiguity) in cluster_types(params).items():
        labels, n = rules.label(np.isin(codes, class_codes), contiguity)
        rows, cols = np.nonzero(labels)
        ids = labels[rows, cols]
        cells = np.bincount(ids, minlength=n + 1)[1:]
        class_code = np.zeros(n + 1, dtype=np.uint8)
        np.maximum.at(class_code, ids, codes[rows, cols])
        row_mean = np.bincount(ids, rows, minlength=n + 1)[1:] / cells
        col_mean = np.bincount(ids, cols, minlength=n + 1)[1:] / cells
        xs, ys = rasterio.transform.xy(transform, row_mean, col_mean, offset="center")
        lon, lat = warp.transform(crs, "EPSG:4326", xs, ys)
        boxes = ndimage.find_objects(labels)
        frames.append(
            pd.DataFrame(
                {
                    "cluster_type": cluster_type,
                    "cluster_id": np.arange(1, n + 1, dtype=np.int32),
                    "class_code": class_code[1:],
                    "population": np.bincount(
                        ids, inputs.population[rows, cols], minlength=n + 1
                    )[1:],
                    "cells": cells,
                    "land_km2": np.bincount(
                        ids, inputs.land[rows, cols], minlength=n + 1
                    )[1:]
                    * inputs.cell_area_km2,
                    "centroid_lon": np.asarray(lon, dtype=np.float64),
                    "centroid_lat": np.asarray(lat, dtype=np.float64),
                    "row_start": np.array(
                        [box[0].start for box in boxes], dtype=np.int64
                    ),
                    "row_stop": np.array(
                        [box[0].stop for box in boxes], dtype=np.int64
                    ),
                    "col_start": np.array(
                        [box[1].start for box in boxes], dtype=np.int64
                    ),
                    "col_stop": np.array(
                        [box[1].stop for box in boxes], dtype=np.int64
                    ),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
