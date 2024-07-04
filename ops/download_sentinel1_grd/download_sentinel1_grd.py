import logging
import os
import shutil
import time
from tempfile import TemporaryDirectory
from typing import Final, cast

import planetary_computer as pc
from requests import RequestException

from vibe_core.data import DownloadedSentinel1Product, Sentinel1Product
from vibe_core.file_downloader import download_file
from vibe_lib.planetary_computer import (
    get_complete_s1_prefix,
    get_sentinel1_container_client,
    get_sentinel1_scene_files,
)

RETRY_WAIT: Final[int] = 10
MAX_RETRIES: Final[int] = 5
LOGGER: Final[logging.Logger] = logging.getLogger(__name__)
READ_TIMEOUT_S: Final[int] = 90
MAX_CONCURRENCY: Final[int] = 3


def download_from_blob(item: Sentinel1Product, save_path: str) -> str:
    container_client = get_sentinel1_container_client()
    scene_files = get_sentinel1_scene_files(item)
    LOGGER.debug(f"Obtained {len(scene_files)} scene files for product '{item.product_name}'")

    if not scene_files:
        # No scene files found!
        raise RuntimeError(
            f"Failed to download sentinel 1 product {item.product_name}, no scene files found."
        )

    blob_prefix = get_complete_s1_prefix(scene_files)
    LOGGER.debug(f"Obtained blob prefix '{blob_prefix}' for product name '{item.product_name}'")
    product_name = blob_prefix.split("/")[-1]

    zip_name = os.path.join(save_path, product_name)
    base_dir = f"{zip_name}.SAFE"

    LOGGER.debug(f"Downloading scene files for product '{item.product_name}'")
    for blob in scene_files:
        out_path = os.path.join(base_dir, os.path.relpath(cast(str, blob.name), blob_prefix))
        save_dir = os.path.dirname(out_path)
        os.makedirs(save_dir, exist_ok=True)
        for retry in range(MAX_RETRIES):
            try:
                url = container_client.get_blob_client(blob.name).url
                download_file(url, out_path)
                break
            except RequestException as e:
                LOGGER.warning(
                    f"Exception {e} downloading from blob {blob.name}."
                    f" Retrying after {RETRY_WAIT}s ({retry+1}/{MAX_RETRIES})."
                )
                time.sleep(RETRY_WAIT)
        else:
            raise RuntimeError(f"Failed asset {blob.name} after {MAX_RETRIES} retries.")
    LOGGER.debug(f"Making zip archive '{zip_name}' for root dir '{save_path}'")
    zip_path = shutil.make_archive(
        zip_name, "zip", root_dir=save_path, base_dir=f"{product_name}.SAFE"
    )
    return zip_path


class CallbackBuilder:
    def __init__(self, api_key: str):
        self.tmp_dir = TemporaryDirectory()
        self.api_key = api_key

    def __call__(self):
        def download_sentinel1_from_pc(sentinel_product: Sentinel1Product):
            pc.set_subscription_key(self.api_key)
            save_path = os.path.join(self.tmp_dir.name, sentinel_product.id)
            zip_path = download_from_blob(sentinel_product, save_path)
            new_item = DownloadedSentinel1Product.clone_from(
                sentinel_product, sentinel_product.id, assets=[]
            )
            new_item.add_zip_asset(zip_path)
            return {"downloaded_product": new_item}

        return download_sentinel1_from_pc

    def __del__(self):
        self.tmp_dir.cleanup()
