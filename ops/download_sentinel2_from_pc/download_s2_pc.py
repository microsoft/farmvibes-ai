import logging
import os
import re
from tempfile import TemporaryDirectory
from typing import Dict, Optional

import planetary_computer as pc
from azure.storage.blob import BlobClient

from vibe_core.data import gen_guid
from vibe_core.data.sentinel import DownloadedSentinel2Product, Sentinel2Product, discriminator_date
from vibe_core.file_downloader import download_file
from vibe_lib.planetary_computer import Sentinel2Collection

LOGGER = logging.getLogger(__name__)


def get_partial_id(product_id: str) -> str:
    return "_".join(re.sub(r"_N[\d]{4}_", "_", product_id).split("_")[:-1])


class CallbackBuilder:
    def __init__(self, api_key: str):
        self.tmp_dir = TemporaryDirectory()
        self.api_key = api_key

    def __call__(self):
        def download_product(
            sentinel_product: Sentinel2Product,
        ) -> Dict[str, Optional[DownloadedSentinel2Product]]:
            pc.set_subscription_key(self.api_key)
            collection = Sentinel2Collection()
            items = collection.query(
                roi=sentinel_product.bbox, time_range=sentinel_product.time_range
            )
            partial_id = get_partial_id(sentinel_product.product_name)
            matches = [item for item in items if get_partial_id(item.id) == partial_id]
            if not matches:
                raise RuntimeError(
                    f"Could not find matches for sentinel 2 product "
                    f"{sentinel_product.product_name}"
                )
            if len(matches) > 1:
                matches = sorted(matches, key=lambda x: discriminator_date(x.id), reverse=True)
                LOGGER.warning(
                    f"Found {len(matches)} > 1 matches for product "
                    f"{sentinel_product.product_name}: {', '.join([m.id for m in matches])}. "
                    f"Picking newest one ({matches[0].id})."
                )

            item = matches[0]
            downloaded_product = DownloadedSentinel2Product.clone_from(
                sentinel_product, sentinel_product.id, []
            )
            # Adding bands
            for k in collection.asset_keys:  # where actual download happens
                asset_path = collection.download_asset(item.assets[k], self.tmp_dir.name)
                downloaded_product.add_downloaded_band(k, asset_path)

            # Adding cloud mask
            gml_out_path = os.path.join(self.tmp_dir.name, f"{gen_guid()}.gml")
            mask_pc_path = collection.get_cloud_mask(item)
            if BlobClient.from_blob_url(mask_pc_path).exists():
                download_file(mask_pc_path, gml_out_path)
                downloaded_product.add_downloaded_cloudmask(gml_out_path)
            else:
                LOGGER.warning(
                    f"GML file is not available for product {sentinel_product.product_name}"
                )

            return {"downloaded_product": downloaded_product}

        return download_product

    def __del__(self):
        self.tmp_dir.cleanup()
