# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from tempfile import TemporaryDirectory
from typing import Dict

import planetary_computer as pc

from vibe_core.data import AlosProduct, AssetVibe, CategoricalRaster, gen_guid, gen_hash_id
from vibe_lib.planetary_computer import AlosForestCollection


class CallbackBuilder:
    def __init__(self, pc_key: str):
        self.tmp_dir = TemporaryDirectory()
        pc.set_subscription_key(pc_key)

    def __call__(self):
        def callback(product: AlosProduct) -> Dict[str, CategoricalRaster]:
            collection = AlosForestCollection()
            item = collection.query_by_id(product.id)
            if not item:
                raise Exception(f"Product {product.id} not found in ALOS Forest collection")
            assets = collection.download_item(item, os.path.join(self.tmp_dir.name, product.id))
            if not assets:
                raise Exception(f"No assets found for product {product.id}")
            assets = [AssetVibe(reference=a, type="image/tiff", id=gen_guid()) for a in assets]
            return {
                "raster": CategoricalRaster.clone_from(
                    product,
                    id=gen_hash_id(
                        f"{product.id}_download_alos_product",
                        product.geometry,
                        product.time_range,
                    ),
                    assets=assets,
                    bands={"forest_non_forest": 0},
                    categories=AlosForestCollection.categories,
                )
            }

        return callback
