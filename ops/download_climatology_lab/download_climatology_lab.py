# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import mimetypes
import os
from tempfile import TemporaryDirectory
from typing import Dict

from vibe_core.data import AssetVibe, gen_guid, gen_hash_id
from vibe_core.data.products import ClimatologyLabProduct
from vibe_core.file_downloader import download_file


class CallbackBuilder:
    def __init__(self):
        self.tmp_dir = TemporaryDirectory()

    def __call__(self):
        def download_climatology_lab(
            input_product: ClimatologyLabProduct,
        ) -> Dict[str, ClimatologyLabProduct]:
            asset_id = gen_guid()
            filepath = os.path.join(self.tmp_dir.name, f"{asset_id}.nc")
            download_file(input_product.url, filepath)
            new_asset = AssetVibe(reference=filepath, type=mimetypes.types_map[".nc"], id=asset_id)

            product = ClimatologyLabProduct.clone_from(
                input_product,
                id=gen_hash_id(
                    f"{input_product.id}_downloaded",
                    input_product.geometry,
                    input_product.time_range,
                ),
                assets=[new_asset],
            )

            return {"downloaded_product": product}

        return download_climatology_lab

    def __del__(self):
        self.tmp_dir.cleanup()
