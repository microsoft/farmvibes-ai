# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import hashlib
from datetime import datetime
from typing import Any, Dict, List, Tuple

from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from vibe_core.data import Raster
from vibe_core.data.rasters import RasterSequence


def time_range_union(list_rasters: List[Raster]) -> Tuple[datetime, datetime]:
    return (
        min([r.time_range[0] for r in list_rasters]),
        max([r.time_range[1] for r in list_rasters]),
    )


def geometry_union(list_rasters: List[Raster]) -> Dict[str, Any]:
    return mapping(unary_union([shape(r.geometry) for r in list_rasters]))


def callback_builder():
    def callback(list_rasters: List[Raster]) -> Dict[str, RasterSequence]:
        res = RasterSequence.clone_from(
            list_rasters[0],
            id=hashlib.sha256(
                ("sequence" + "".join(r.id for r in list_rasters)).encode()
            ).hexdigest(),
            time_range=time_range_union(list_rasters),
            geometry=geometry_union(list_rasters),
            assets=[],
        )
        for r in list_rasters:
            res.add_item(r)

        return {"rasters_seq": res}

    return callback
