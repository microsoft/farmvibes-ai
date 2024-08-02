# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, List, Union

from shapely import geometry as shpg

from vibe_core.data import Raster


def callback(
    rasters1: List[Raster], rasters2: List[Raster]
) -> Dict[str, Union[List[Raster], List[Raster]]]:
    paired_rasters1 = []
    paired_rasters2 = []
    for r1 in rasters1:
        geom_n = shpg.shape(r1.geometry)
        for r2 in rasters2:
            geom_d = shpg.shape(r2.geometry)
            if geom_n.intersects(geom_d):
                paired_rasters1.append(r1)
                paired_rasters2.append(r2)

    if not paired_rasters1:
        raise ValueError("No intersecting rasters could be paired")
    return {"paired_rasters1": paired_rasters1, "paired_rasters2": paired_rasters2}


def callback_builder():
    return callback
