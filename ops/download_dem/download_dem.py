import mimetypes
import os
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, cast

import planetary_computer as pc

from vibe_core.data import AssetVibe, DemProduct, DemRaster, gen_guid, gen_hash_id
from vibe_lib.planetary_computer import validate_dem_provider
from vibe_lib.raster import RGBA, interpolated_cmap_from_colors, json_to_asset

ELEVATION_CMAP_INTERVALS: List[float] = [0.0, 4000.0]

ELEVATION_CMAP_COLORS: List[RGBA] = [
    RGBA(0, 0, 0, 255),
    RGBA(255, 255, 255, 255),
]


class CallbackBuilder:
    def __init__(self, api_key: str):
        self.tmp_dir = TemporaryDirectory()
        self.api_key = api_key

    def __call__(self):
        def op(input_product: DemProduct) -> Dict[str, DemRaster]:
            pc.set_subscription_key(self.api_key)
            collection = validate_dem_provider(
                input_product.provider.upper(), input_product.resolution
            )
            item = collection.query_by_id(input_product.tile_id)
            assets = collection.download_item(
                item, os.path.join(self.tmp_dir.name, input_product.id)
            )
            assets = [
                AssetVibe(reference=a, type=cast(str, mimetypes.guess_type(a)[0]), id=gen_guid())
                for a in assets
            ]
            vis_dict: Dict[str, Any] = {
                "bands": [0],
                "colormap": interpolated_cmap_from_colors(
                    ELEVATION_CMAP_COLORS, ELEVATION_CMAP_INTERVALS
                ),
                "range": (0, 4000),
            }
            assets.append(json_to_asset(vis_dict, self.tmp_dir.name))

            downloaded_product = DemRaster(
                id=gen_hash_id(
                    f"{input_product.id}_download_dem_product",
                    input_product.geometry,
                    input_product.time_range,
                ),
                time_range=input_product.time_range,
                geometry=input_product.geometry,
                assets=assets,
                bands={"elevation": 0},
                tile_id=input_product.tile_id,
                resolution=input_product.resolution,
                provider=input_product.provider,
            )

            return {"downloaded_product": downloaded_product}

        return op

    def __del__(self):
        self.tmp_dir.cleanup()
