# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional

import pandas as pd
from shapely import geometry as shpg

from vibe_core.data import DataSummaryStatistics, DataVibe, Raster, gen_guid
from vibe_core.data.core_types import AssetVibe
from vibe_lib.raster import load_raster_from_url


def summarize_raster(
    raster: Raster, mask: Optional[Raster], geometry: Dict[str, Any]
) -> Dict[str, float]:
    geom = shpg.shape(geometry).intersection(shpg.shape(raster.geometry))
    data_ar = load_raster_from_url(raster.raster_asset.url, geometry=geom, geometry_crs="epsg:4326")
    data_ma = data_ar.to_masked_array()
    if mask is not None:
        mask_ma = load_raster_from_url(
            mask.raster_asset.url,
            crs=data_ar.rio.crs,
            geometry=geom,
            geometry_crs="epsg:4326",
        ).to_masked_array()
        # Update mask
        data_ma.mask = data_ma.mask | (mask_ma.data > 0 & ~mask_ma.mask)
        masked_ratio = mask_ma.mean()
    else:
        masked_ratio = 0.0
    return {
        "mean": data_ma.mean(),
        "std": data_ma.std(),
        "min": data_ma.min(),
        "max": data_ma.max(),
        "masked_ratio": masked_ratio,
    }


class CallbackBuilder:
    def __init__(self):
        self.tmp_dir = TemporaryDirectory()

    def __call__(self):
        def callback(
            raster: Raster, input_geometry: DataVibe, mask: Optional[Raster] = None
        ) -> Dict[str, DataSummaryStatistics]:
            geom = input_geometry.geometry
            stats = summarize_raster(raster, mask, geom)
            guid = gen_guid()
            filepath = os.path.join(self.tmp_dir.name, f"{guid}.csv")
            pd.DataFrame(stats, index=pd.Index([raster.time_range[0]], name="date")).to_csv(
                filepath
            )
            summary = DataSummaryStatistics.clone_from(
                raster,
                geometry=geom,
                id=gen_guid(),
                assets=[AssetVibe(reference=filepath, type="text/csv", id=guid)],
            )
            return {"summary": summary}

        return callback

    def __del__(self):
        self.tmp_dir.cleanup()
