# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Tuple

import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.vrt import WarpedVRT
from shapely import geometry as shpg

from vibe_core.data import AssetVibe, Sentinel1Raster, Sentinel1RasterOrbitGroup, gen_guid
from vibe_lib.raster import FLOAT_COMPRESSION_KWARGS, tile_to_utm


def merge_rasters(
    filepaths: List[str],
    bounds: Tuple[float, float, float, float],
    resampling: Resampling,
    out_path: str,
    **kwargs: Any,
):
    src = []
    vrt = []
    try:
        src = [rasterio.open(i) for i in filepaths]
        vrt = [WarpedVRT(i, **kwargs) for i in src]
        dst_kwds = FLOAT_COMPRESSION_KWARGS
        dst_kwds["driver"] = "GTiff"
        dst_kwds.update({"blockxsize": 512, "blockysize": 512})
        return merge(
            vrt, bounds=bounds, resampling=resampling, dst_path=out_path, dst_kwds=dst_kwds
        )
    finally:
        for i in src + vrt:
            i.close()  # type:ignore


def process_orbit(
    orbit_group: Sentinel1RasterOrbitGroup, output_dir: str, resampling: Resampling
) -> Sentinel1Raster:
    out_id = gen_guid()
    filepath = os.path.join(output_dir, f"{out_id}.tif")
    geom = orbit_group.geometry
    tile_id = orbit_group.tile_id
    crs = f"epsg:{tile_to_utm(tile_id)}"
    bounds = tuple(
        gpd.GeoSeries(shpg.shape(geom), crs="epsg:4326").to_crs(crs).bounds.round().iloc[0]
    )
    merge_rasters(
        [i.url for i in orbit_group.get_ordered_assets()],
        bounds=bounds,
        resampling=resampling,
        out_path=filepath,
        crs=crs,
    )

    asset = AssetVibe(reference=filepath, type="image/tiff", id=out_id)
    product = Sentinel1Raster.clone_from(orbit_group, id=gen_guid(), assets=[asset])
    return product


class CallbackBuilder:
    def __init__(self, resampling: str):
        self.tmp_dir = TemporaryDirectory()
        self.resampling = Resampling[resampling]

    def __call__(self):
        def callback(
            raster_group: Sentinel1RasterOrbitGroup,
        ) -> Dict[str, Sentinel1Raster]:
            return {
                "merged_product": process_orbit(raster_group, self.tmp_dir.name, self.resampling)
            }

        return callback

    def __del__(self):
        self.tmp_dir.cleanup()
