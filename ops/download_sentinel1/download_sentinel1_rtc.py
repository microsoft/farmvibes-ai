import logging
import os
from concurrent.futures import TimeoutError
from tempfile import TemporaryDirectory
from typing import Dict

import planetary_computer as pc
import rasterio
from pystac import Item
from rasterio.enums import Resampling
from rasterio.windows import Window

from vibe_core.data import AssetVibe, Sentinel1Product, Sentinel1Raster, gen_guid
from vibe_lib.planetary_computer import Sentinel1RTCCollection
from vibe_lib.raster import FLOAT_COMPRESSION_KWARGS, get_profile_from_ref, serial_stack_bands

LOGGER = logging.getLogger(__name__)


def read_block(raster_url: str, win: Window):
    with rasterio.open(raster_url) as src:
        return src.read(window=win), win


class CallbackBuilder:
    def __init__(self, api_key: str, num_workers: int, block_size: int, timeout_s: float):
        self.api_key = api_key
        self.num_workers = num_workers
        self.block_size = block_size
        self.timeout_s = timeout_s
        self.tmp_dir = TemporaryDirectory()

    def stack_bands(self, col: Sentinel1RTCCollection, item: Item) -> AssetVibe:
        asset_guid = gen_guid()
        out_path = os.path.join(self.tmp_dir.name, f"{asset_guid}.tif")
        LOGGER.debug(f"Downloading Sentinel-1 RTC bands for product {item.id}")
        band_hrefs = col.download_item(item, os.path.join(self.tmp_dir.name, item.id))
        LOGGER.debug(f"Done downloading Sentinel-1 RTC bands for product {item.id}")
        kwargs = get_profile_from_ref(
            band_hrefs[0], count=len(band_hrefs), **FLOAT_COMPRESSION_KWARGS
        )
        LOGGER.debug(f"Stacking Sentinel-1 RTC bands for product {item.id}")
        serial_stack_bands(
            band_hrefs,
            out_path,
            (self.block_size, self.block_size),
            Resampling.bilinear,
            **kwargs,
        )
        LOGGER.debug(f"Done stacking Sentinel-1 RTC bands for product {item.id}")
        return AssetVibe(reference=out_path, type="image/tiff", id=asset_guid)

    def __call__(self):
        def callback(sentinel_product: Sentinel1Product) -> Dict[str, Sentinel1Raster]:
            pc.set_subscription_key(self.api_key)
            col = Sentinel1RTCCollection()
            item = pc.sign(col.query_by_id(sentinel_product.id))
            try:
                asset = self.stack_bands(col, item)
            except TimeoutError as e:
                raise TimeoutError(
                    f"Timeout while stacking bands for products {sentinel_product.product_name}"
                ) from e
            raster = Sentinel1Raster.clone_from(
                sentinel_product,
                sentinel_product.id,
                assets=[asset],
                bands={k.upper(): i for i, k in enumerate(col.asset_keys)},
                tile_id="",
            )
            return {"downloaded_product": raster}

        return callback

    def __del__(self):
        self.tmp_dir.cleanup()
