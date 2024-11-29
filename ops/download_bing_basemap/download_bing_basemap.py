# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import hashlib
import os
from tempfile import TemporaryDirectory
from typing import Dict

import rasterio
from rasterio.transform import from_bounds

from vibe_core.data import AssetVibe, BBox, Raster, gen_guid
from vibe_core.data.products import BingMapsProduct
from vibe_lib.bing_maps import BingMapsCollection


def build_raster_asset(tile_path: str, tile_bbox: BBox, output_path: str):
    """Build a GeoTIFF raster asset from a tile downloaded from BingMaps."""
    with rasterio.open(tile_path) as src:
        img = src.read()

    transform = from_bounds(*tile_bbox, img.shape[2], img.shape[1])

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=img.shape[1],
        width=img.shape[2],
        count=3,
        dtype=img.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(img)


class CallbackBuilder:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("BingMaps API key was not provided.")

        self.collection = BingMapsCollection(api_key)
        self.tmp_dir = TemporaryDirectory()

    def download_basemap(self, product: BingMapsProduct) -> AssetVibe:
        img_id = gen_guid()
        tile_path = os.path.join(self.tmp_dir.name, f"{img_id}.jpeg")
        raster_path = os.path.join(self.tmp_dir.name, f"{img_id}.tiff")

        try:
            self.collection.download_tile(product.url, tile_path)
        except (RuntimeError, ValueError) as e:
            raise type(e)(
                f"Failed to download tile {product.id} at zoom level {product.zoom_level}. {e}"
            ) from e

        build_raster_asset(tile_path, product.bbox, raster_path)
        asset = AssetVibe(
            reference=raster_path,
            type="image/tiff",
            id=gen_guid(),
        )
        return asset

    def __call__(self):
        def download_bing_basemap(
            input_product: BingMapsProduct,
        ) -> Dict[str, Raster]:
            asset = self.download_basemap(input_product)

            basemap = Raster.clone_from(
                input_product,
                id=hashlib.sha256(f"downloaded_basemap_{input_product.id}".encode()).hexdigest(),
                assets=[asset],
                bands={"red": 0, "green": 1, "blue": 2},
            )

            return {"basemap": basemap}

        return download_bing_basemap

    def __del__(self):
        self.tmp_dir.cleanup()
