# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import mimetypes
import os
from tempfile import TemporaryDirectory
from typing import Dict, List, Union

from rasterio.merge import merge

from vibe_core.data import (
    AssetVibe,
    Sentinel2CloudMask,
    Sentinel2CloudMaskOrbitGroup,
    Sentinel2Raster,
    Sentinel2RasterOrbitGroup,
    gen_guid,
)
from vibe_core.uri import uri_to_filename


def merge_rasters(path_list: List[str], dst_dir: str) -> str:
    filename = uri_to_filename(path_list[0])
    dst_path = os.path.join(dst_dir, filename)
    # Rasterio is merging by keeping the first pixel while GDAL was keeping the
    # last. There seems to be no advantage to either, but the new behavior is
    # different.
    merge(path_list, dst_path=dst_path, dst_kwds={"zstd_level": 9, "predictor": 2})
    return dst_path


class CallbackBuilder:
    def __init__(self):
        self.tmp_dir = TemporaryDirectory()

    def __call__(self):
        def merge_orbits(
            raster_group: Sentinel2RasterOrbitGroup, mask_group: Sentinel2CloudMaskOrbitGroup
        ) -> Dict[str, Union[Sentinel2Raster, Sentinel2CloudMask]]:
            raster_list = [a.url for a in raster_group.get_ordered_assets()]
            mask_list = [a.url for a in mask_group.get_ordered_assets()]

            if len(raster_list) > 1:
                merged_img = merge_rasters(raster_list, self.tmp_dir.name)
                merged_cloud = merge_rasters(mask_list, self.tmp_dir.name)

                raster_asset = AssetVibe(
                    reference=merged_img, type=mimetypes.types_map[".tif"], id=gen_guid()
                )
                mask_asset = AssetVibe(
                    reference=merged_cloud, type=mimetypes.types_map[".tif"], id=gen_guid()
                )
            else:
                raster_asset = raster_group.get_ordered_assets()[0]
                mask_asset = mask_group.get_ordered_assets()[0]

            # Update item geometry
            new_raster = Sentinel2Raster.clone_from(
                raster_group,
                id=gen_guid(),
                assets=[raster_asset],
            )

            new_mask = Sentinel2CloudMask.clone_from(
                mask_group,
                id=gen_guid(),
                assets=[mask_asset],
            )

            return {"output_raster": new_raster, "output_mask": new_mask}

        return merge_orbits

    def __del__(self):
        self.tmp_dir.cleanup()
