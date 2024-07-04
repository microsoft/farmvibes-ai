import mimetypes
import os
from tempfile import TemporaryDirectory
from typing import Dict

from vibe_core.data import AssetVibe, CategoricalRaster, gen_hash_id
from vibe_core.data.core_types import gen_guid
from vibe_core.data.products import GLADProduct
from vibe_core.file_downloader import download_file


class CallbackBuilder:
    def __init__(self):
        self.tmp_dir = TemporaryDirectory()

    def __call__(self):
        def download_product(glad_product: GLADProduct) -> Dict[str, CategoricalRaster]:
            fname = f"{glad_product.tile_name}_{glad_product.time_range[0].year}.tif"
            fpath = os.path.join(self.tmp_dir.name, fname)
            download_file(glad_product.url, fpath)

            asset = AssetVibe(reference=fpath, type=mimetypes.types_map[".tif"], id=gen_guid())

            downloaded_product = CategoricalRaster.clone_from(
                glad_product,
                id=gen_hash_id(fname, glad_product.geometry, glad_product.time_range),
                assets=[asset],
                bands={"forest_extent": 0},
                categories=["Non-Forest", "Forest"],
            )

            return {"downloaded_product": downloaded_product}

        return download_product

    def __del__(self):
        self.tmp_dir.cleanup()
