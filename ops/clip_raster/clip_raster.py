# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import os
from tempfile import TemporaryDirectory
from typing import Dict, TypeVar, cast

import rioxarray as rio
import xarray as xr
from shapely import geometry as shpg

from vibe_core.data import AssetVibe, DataVibe, Raster, gen_guid, gen_hash_id

LOGGER = logging.getLogger(__name__)
T = TypeVar("T", bound=Raster)


class CallbackBuilder:
    def __init__(self, hard_clip: bool):
        self.tmp_dir = TemporaryDirectory()
        self.hard_clip = hard_clip

    def __call__(self):
        def operator_callback(input_item: DataVibe, raster: T) -> Dict[str, T]:
            ref_geometry = shpg.shape(input_item.geometry)

            raster_shpg = shpg.shape(raster.geometry)
            if raster_shpg.intersects(ref_geometry):
                intersecting_geometry = raster_shpg.intersection(ref_geometry)

                if not self.hard_clip:
                    out_raster = type(raster).clone_from(
                        raster,
                        id=gen_hash_id(
                            f"{raster.id}_soft_clip", intersecting_geometry, raster.time_range
                        ),
                        geometry=shpg.mapping(intersecting_geometry),
                        assets=raster.assets,
                    )
                else:
                    da = cast(xr.DataArray, rio.open_rasterio(raster.raster_asset.path_or_url))
                    fpath = os.path.join(self.tmp_dir.name, "clip.tif")
                    da.rio.clip(
                        [intersecting_geometry], crs="EPSG:4326", from_disk=True
                    ).rio.to_raster(fpath)
                    new_raster_asset = AssetVibe(reference=fpath, type="image/tiff", id=gen_guid())
                    assets = raster.assets.copy()
                    assets.remove(raster.raster_asset)
                    assets.append(new_raster_asset)
                    out_raster = type(raster).clone_from(
                        raster,
                        id=gen_hash_id(
                            f"{raster.id}_hard_clip", intersecting_geometry, raster.time_range
                        ),
                        geometry=shpg.mapping(intersecting_geometry),
                        assets=assets,
                    )

                return {"clipped_raster": out_raster}
            else:
                raise ValueError(
                    "Input reference geometry does not intersect with raster geometry."
                )

        return operator_callback

    def __del__(self):
        self.tmp_dir.cleanup()
