# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
from tempfile import TemporaryDirectory
from typing import Dict, Optional

import planetary_computer as pc

from vibe_core.data import LandsatProduct, gen_hash_id
from vibe_lib.planetary_computer import LandsatCollection

LOGGER = logging.getLogger(__name__)


class CallbackBuilder:
    def __init__(self, api_key: str):
        self.tmp_dir = TemporaryDirectory()
        self.api_key = api_key

    def __call__(self):
        def download_product(
            landsat_product: LandsatProduct,
        ) -> Dict[str, Optional[LandsatProduct]]:
            pc.set_subscription_key(self.api_key)
            collection = LandsatCollection()
            item = collection.query_by_id(landsat_product.tile_id)

            downloaded_product = LandsatProduct.clone_from(
                landsat_product,
                id=gen_hash_id(
                    f"{landsat_product.id}_download_landsat_product",
                    landsat_product.geometry,
                    landsat_product.time_range,
                ),
                assets=[],
            )

            for k in collection.asset_keys:
                try:
                    asset_path = collection.download_asset(item.assets[k], self.tmp_dir.name)
                    downloaded_product.add_downloaded_band(k, asset_path)
                except KeyError as e:
                    LOGGER.warning(f"No band {k} found. Original exception {e}")

            return {"downloaded_product": downloaded_product}

        return download_product

    def __del__(self):
        self.tmp_dir.cleanup()
