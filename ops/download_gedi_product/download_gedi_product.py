# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import os
from tempfile import TemporaryDirectory
from typing import Dict

from vibe_core.data import AssetVibe, GEDIProduct, gen_guid
from vibe_core.file_downloader import download_file
from vibe_lib.earthdata import EarthDataAPI

LOGGER = logging.getLogger(__name__)


class CallbackBuilder:
    def __init__(self, token: str):
        self.token = token
        self.tmp_dir = TemporaryDirectory()

    def __call__(self):
        def callback(gedi_product: GEDIProduct) -> Dict[str, GEDIProduct]:
            api = EarthDataAPI(gedi_product.processing_level)
            LOGGER.info(f"Querying EarthData API for product {gedi_product.product_name}")
            items = api.query(id=gedi_product.product_name)
            if len(items) != 1:
                raise RuntimeError(
                    f"Query for GEDI product {gedi_product.product_name} "
                    "returned {len(items)} items, expected one item"
                )
            url = items[0]["links"][0]["href"]
            asset_guid = gen_guid()
            out_path = os.path.join(self.tmp_dir.name, f"{asset_guid}")
            h5_path = f"{out_path}.h5"
            headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
            LOGGER.info(f"Downloading data from {url}")
            download_file(url, h5_path, headers=headers)
            asset = AssetVibe(reference=h5_path, type="application/x-hdf5", id=asset_guid)
            dl_product = GEDIProduct.clone_from(gedi_product, id=gen_guid(), assets=[asset])
            return {"downloaded_product": dl_product}

        return callback
