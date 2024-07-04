import os
from tempfile import TemporaryDirectory
from typing import Dict, Optional

import planetary_computer as pc

from vibe_core.data import GNATSGOProduct, gen_hash_id
from vibe_core.data.core_types import AssetVibe, gen_guid
from vibe_core.data.rasters import GNATSGORaster
from vibe_lib.planetary_computer import GNATSGOCollection
from vibe_lib.raster import FLOAT_COMPRESSION_KWARGS, compress_raster


def download_asset(input_product: GNATSGOProduct, variable: str, dir_path: str) -> AssetVibe:
    """
    Downloads the raster asset of the selected variable and compresses it
    """
    collection = GNATSGOCollection()
    item = collection.query_by_id(input_product.id)

    uncompressed_asset_path = collection.download_asset(item.assets[variable], dir_path)

    asset_id = gen_guid()
    asset_path = os.path.join(dir_path, f"{asset_id}.tif")

    compress_raster(uncompressed_asset_path, asset_path, **FLOAT_COMPRESSION_KWARGS)

    return AssetVibe(reference=asset_path, type="image/tiff", id=asset_id)


class CallbackBuilder:
    def __init__(self, api_key: str, variable: str):
        self.tmp_dir = TemporaryDirectory()
        self.api_key = api_key

        if variable not in GNATSGOCollection.asset_keys:
            raise ValueError(
                f"Requested variable '{variable}' not valid. "
                f"Valid values are {', '.join(GNATSGOCollection.asset_keys)}"
            )
        self.variable = variable

    def __call__(self):
        def download_gnatsgo_raster(
            gnatsgo_product: GNATSGOProduct,
        ) -> Dict[str, Optional[GNATSGORaster]]:
            pc.set_subscription_key(self.api_key)

            asset = download_asset(gnatsgo_product, self.variable, self.tmp_dir.name)

            downloaded_raster = GNATSGORaster.clone_from(
                gnatsgo_product,
                id=gen_hash_id(
                    f"{gnatsgo_product.id}_{self.variable}_downloaded_gnatsgo_product",
                    gnatsgo_product.geometry,
                    gnatsgo_product.time_range,
                ),
                assets=[asset],
                bands={self.variable: 0},
                variable=self.variable,
            )

            return {"downloaded_raster": downloaded_raster}

        return download_gnatsgo_raster

    def __del__(self):
        self.tmp_dir.cleanup()
