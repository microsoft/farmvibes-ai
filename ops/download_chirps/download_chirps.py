# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import os
import re
from tempfile import TemporaryDirectory
from typing import Dict, Optional

from vibe_core.data import AssetVibe, gen_hash_id
from vibe_core.data.core_types import gen_guid
from vibe_core.data.products import ChirpsProduct
from vibe_core.file_downloader import download_file

LOGGER = logging.getLogger(__name__)


class CallbackBuilder:
    def __init__(self):
        self.tmp_dir = TemporaryDirectory()

    def __call__(self):
        def download_product(
            chirps_product: ChirpsProduct,
        ) -> Dict[str, Optional[ChirpsProduct]]:
            fname = re.search("chirps-.*cog", chirps_product.url)
            if fname is not None:
                fname = fname.group()
            else:
                raise ValueError(f"URL for chirps product has no COG. url: {chirps_product.url}")
            fpath = os.path.join(self.tmp_dir.name, fname)
            download_file(chirps_product.url, fpath)

            asset = AssetVibe(reference=fpath, type="image/tiff", id=gen_guid())

            downloaded_product = ChirpsProduct.clone_from(
                chirps_product,
                id=gen_hash_id(fname, chirps_product.geometry, chirps_product.time_range),
                assets=[asset],
            )

            return {"downloaded_product": downloaded_product}

        return download_product

    def __del__(self):
        self.tmp_dir.cleanup()
