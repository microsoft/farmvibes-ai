# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pyright: reportUnknownMemberType=false
from typing import Dict, List, Union, cast

import rasterio

from vibe_core.data import RasterIlluminance, Sentinel2CloudMask, Sentinel2Raster
from vibe_lib.spaceeye.illumination import MIN_CLEAR_RATIO, masked_average_illuminance
from vibe_lib.spaceeye.utils import QUANTIFICATION_VALUE


def compute_illuminance(item: Sentinel2Raster, cloud_mask: Sentinel2CloudMask):
    """
    Compute illuminance values one band at a time to save memory
    """
    data_filepath = item.raster_asset.url
    mask_filepath = cloud_mask.raster_asset.url
    illuminance: List[float] = []
    with rasterio.open(mask_filepath) as src:
        mask = src.read(1).astype(bool)
    if mask.mean() < MIN_CLEAR_RATIO:
        return None
    with rasterio.open(data_filepath) as src:
        # rasterio indexes bands starting with 1
        for i in range(1, cast(int, src.count + 1)):
            x = src.read(i) / QUANTIFICATION_VALUE
            illuminance.append(float(masked_average_illuminance(x, mask)))

    return RasterIlluminance.clone_from(item, id=item.id, assets=[], illuminance=illuminance)


class CallbackBuilder:
    def __init__(self, num_workers: int):
        self.num_workers = num_workers

    def __call__(self):
        def callback(
            rasters: List[Sentinel2Raster], cloud_masks: List[Sentinel2CloudMask]
        ) -> Dict[str, List[RasterIlluminance]]:
            results = [compute_illuminance(item, mask) for item, mask in zip(rasters, cloud_masks)]
            results = cast(List[Union[RasterIlluminance, None]], results)
            results = [r for r in results if r is not None]

            return {"illuminance": results}

        return callback
