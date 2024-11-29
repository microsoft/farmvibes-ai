# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import mimetypes
import os
from tempfile import TemporaryDirectory
from typing import Dict, cast

import planetary_computer as pc

from vibe_core.data import AssetVibe, CategoricalRaster, gen_guid, gen_hash_id
from vibe_core.data.products import EsriLandUseLandCoverProduct
from vibe_lib.planetary_computer import EsriLandUseLandCoverCollection
from vibe_lib.raster import json_to_asset


class CallbackBuilder:
    def __init__(self, api_key: str):
        self.tmp_dir = TemporaryDirectory()
        self.api_key = api_key

    def __call__(self):
        def op(input_product: EsriLandUseLandCoverProduct) -> Dict[str, CategoricalRaster]:
            pc.set_subscription_key(self.api_key)
            collection = EsriLandUseLandCoverCollection()
            item = collection.query_by_id(input_product.id)
            assets = collection.download_item(
                item, os.path.join(self.tmp_dir.name, input_product.id)
            )
            vibe_assets = [
                AssetVibe(reference=a, type=cast(str, mimetypes.guess_type(a)[0]), id=gen_guid())
                for a in assets
            ]
            vis_asset = json_to_asset({"bands": list(range(1))}, self.tmp_dir.name)
            vibe_assets.append(vis_asset)
            downloaded_product = CategoricalRaster(
                id=gen_hash_id(
                    f"{input_product.id}_download_esri_landuse_landcover_product",
                    input_product.geometry,
                    input_product.time_range,
                ),
                time_range=input_product.time_range,
                geometry=input_product.geometry,
                assets=vibe_assets,
                bands={"data": 0},
                categories=EsriLandUseLandCoverCollection.categories,
            )

            return {"downloaded_product": downloaded_product}

        return op

    def __del__(self):
        self.tmp_dir.cleanup()
