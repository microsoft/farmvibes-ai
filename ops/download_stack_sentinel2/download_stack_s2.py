import hashlib
import logging
import os
import re
from tempfile import TemporaryDirectory
from typing import Dict, Union

import geopandas as gpd
import numpy as np
import planetary_computer as pc
from azure.storage.blob import BlobClient
from rasterio.enums import Resampling
from rasterio.features import rasterize

from vibe_core.data import AssetVibe, gen_guid
from vibe_core.data.sentinel import (
    Sentinel2CloudMask,
    Sentinel2Product,
    Sentinel2Raster,
    discriminator_date,
)
from vibe_lib.planetary_computer import Sentinel2Collection
from vibe_lib.raster import (
    INT_COMPRESSION_KWARGS,
    get_profile_from_ref,
    open_raster_from_ref,
    serial_stack_bands,
)

LOGGER = logging.getLogger(__name__)

CLOUD_CATEGORIES = ["NO-CLOUD", "OPAQUE", "CIRRUS", "OTHER"]

SENTINEL2_SPYNDEX: Dict[str, str] = {
    "B01": "A",
    "B02": "B",
    "B03": "G",
    "B04": "R",
    "B05": "RE1",
    "B06": "RE2",
    "B07": "RE3",
    "B08": "N",
    "B8A": "N2",
    "B09": "WV",
    "B11": "S1",
    "B12": "S2",
}


def get_partial_id(product_id: str) -> str:
    return "_".join(re.sub(r"_N[\d]{4}_", "_", product_id).split("_")[:-1])


def rasterize_clouds(cloud_gml_ref: str, ref_file: str, out_path: str, product_name: str) -> None:
    """
    Rasterize cloud shapes and save compressed tiff file.
    """
    with open_raster_from_ref(ref_file) as src:
        meta = src.meta
    meta.update(
        {"nodata": 100, "count": 1, "driver": "GTiff", "dtype": "uint8", **INT_COMPRESSION_KWARGS}
    )
    out = np.zeros((meta["height"], meta["width"]))
    try:
        # The file might not exist, in this case we generate empty cloud masks (no clouds)
        if BlobClient.from_blob_url(cloud_gml_ref).exists():
            df = gpd.read_file(cloud_gml_ref, WRITE_GFS="NO")
            cloud_map = {
                "OPAQUE": CLOUD_CATEGORIES.index("OPAQUE"),
                "CIRRUS": CLOUD_CATEGORIES.index("CIRRUS"),
            }
            values = (
                df["maskType"]  # type: ignore
                .map(cloud_map)  # type: ignore
                .fillna(CLOUD_CATEGORIES.index("OTHER"))
            )
            rasterize(
                ((g, v) for g, v in zip(df["geometry"], values)),  # type: ignore
                out=out,
                transform=meta["transform"],
            )
        else:
            LOGGER.debug(
                f"Cloud GML file is not available for product {product_name}, generating empty mask"
            )
    except ValueError:
        # Empty file means no clouds
        LOGGER.debug(
            "ValueError when opening cloud GML file. Assuming there are no clouds and ignoring",
            exc_info=True,
        )
        pass
    with open_raster_from_ref(out_path, "w", **meta) as dst:
        dst.write(out, 1)


class CallbackBuilder:
    def __init__(self, api_key: str, num_workers: int, block_size: int, timeout_s: float):
        self.tmp_dir = TemporaryDirectory()
        self.api_key = api_key
        self.num_workers = num_workers
        self.block_size = block_size
        self.timeout_s = timeout_s

    def __call__(self):
        def callback(
            sentinel_product: Sentinel2Product,
        ) -> Dict[str, Union[Sentinel2Raster, Sentinel2CloudMask]]:
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
            item = pc.sign(item)
            LOGGER.debug(
                f"Downloading Sentinel-2 bands for product {sentinel_product.product_name}"
            )
            band_hrefs = collection.download_item(
                item, os.path.join(self.tmp_dir.name, sentinel_product.product_name)
            )
            LOGGER.debug(
                f"Done downloading Sentinel-2 bands for product {sentinel_product.product_name}"
            )
            tiff_args = get_profile_from_ref(
                band_hrefs[collection.asset_keys.index("B02")],
                count=len(band_hrefs),
                nodata=0,
                **INT_COMPRESSION_KWARGS,
            )
            bands_id = gen_guid()
            bands_path = os.path.join(self.tmp_dir.name, f"{bands_id}.tif")
            LOGGER.debug(f"Stacking Sentinel-2 bands for product {sentinel_product.product_name}")
            serial_stack_bands(
                band_hrefs,
                bands_path,
                block_size=(self.block_size, self.block_size),
                resampling=Resampling.bilinear,
                **tiff_args,
            )
            LOGGER.debug(f"Done stacking bands for product {sentinel_product.product_name}")

            # Adding cloud mask
            mask_id = gen_guid()
            mask_path = os.path.join(self.tmp_dir.name, f"{mask_id}.tif")

            rasterize_clouds(
                collection.get_cloud_mask(item),
                bands_path,
                mask_path,
                sentinel_product.product_name,
            )
            band_idx = {name: idx for idx, name in enumerate(collection.asset_keys)}
            # Add band aliases for spyndex
            for k, v in SENTINEL2_SPYNDEX.items():
                band_idx[v] = band_idx[k]
            bands_raster = Sentinel2Raster.clone_from(
                sentinel_product,
                bands=band_idx,
                id=hashlib.sha256(f"stacked bands {sentinel_product.id}".encode()).hexdigest(),
                assets=[AssetVibe(reference=bands_path, type="image/tiff", id=bands_id)],
            )
            cloud_raster = Sentinel2CloudMask.clone_from(
                sentinel_product,
                bands={"cloud": 0},
                categories=CLOUD_CATEGORIES,
                id=hashlib.sha256(f"clouds {sentinel_product.id}".encode()).hexdigest(),
                assets=[AssetVibe(reference=mask_path, type="image/tiff", id=mask_id)],
            )
            return {"raster": bands_raster, "cloud": cloud_raster}

        return callback

    def __del__(self):
        self.tmp_dir.cleanup()
