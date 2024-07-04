import mimetypes
import os
from tempfile import TemporaryDirectory
from typing import Dict

from vibe_core.data import AssetVibe, HansenProduct
from vibe_core.data.core_types import gen_guid, gen_hash_id
from vibe_core.data.rasters import Raster
from vibe_core.file_downloader import download_file


class CallbackBuilder:
    def __init__(self):
        self.tmp_dir = TemporaryDirectory()

    def __call__(self):
        def download_product(hansen_product: HansenProduct) -> Dict[str, Raster]:
            fname = (
                f"hansen_{hansen_product.layer_name}_{hansen_product.tile_name}_"
                f"{hansen_product.last_year}.tif"
            )
            fpath = os.path.join(self.tmp_dir.name, fname)
            download_file(hansen_product.asset_url, fpath)

            asset = AssetVibe(reference=fpath, type=mimetypes.types_map[".tif"], id=gen_guid())
            downloaded_product = Raster.clone_from(
                hansen_product,
                id=gen_hash_id(
                    f"{hansen_product.id}_downloaded_hansen_product",
                    hansen_product.geometry,
                    hansen_product.time_range,
                ),
                assets=[asset],
                bands={hansen_product.layer_name: 0},
            )

            return {"raster": downloaded_product}

        return download_product

    def __del__(self):
        self.tmp_dir.cleanup()
