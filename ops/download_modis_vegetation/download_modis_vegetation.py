# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from tempfile import TemporaryDirectory
from typing import Dict, Optional

import planetary_computer as pc

from vibe_core.data import AssetVibe, ModisProduct, Raster, gen_guid
from vibe_lib.planetary_computer import Modis16DayVICollection

VALID_INDICES = ("evi", "ndvi")


class CallbackBuilder:
    def __init__(self, index: str, pc_key: Optional[str]):
        self.tmp_dir = TemporaryDirectory()
        if index not in VALID_INDICES:
            raise ValueError(f"Expected index to be one of {VALID_INDICES}, got '{index}'.")
        self.index = index
        pc.set_subscription_key(pc_key)  # type: ignore

    def __call__(self):
        def callback(product: ModisProduct) -> Dict[str, Raster]:
            col = Modis16DayVICollection(product.resolution)
            items = col.query(
                roi=product.bbox,
                time_range=product.time_range,
                ids=[product.id],
            )
            assert len(items) == 1
            item = items[0]
            assets = [v for k, v in item.assets.items() if self.index.upper() in k]
            assert len(assets) == 1
            asset = assets[0]
            assets = [
                AssetVibe(
                    reference=col.download_asset(asset, self.tmp_dir.name),
                    type="image/tiff",
                    id=gen_guid(),
                )
            ]
            return {
                "index": Raster.clone_from(
                    product, id=gen_guid(), assets=assets, bands={self.index: 0}
                )
            }

        return callback

    def __del__(self):
        self.tmp_dir.cleanup()
