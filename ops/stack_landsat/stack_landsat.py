# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from tempfile import TemporaryDirectory
from typing import Dict, Tuple

import numpy as np
import rioxarray as rio
import xarray as xr

from vibe_core.data import AssetVibe, LandsatProduct, gen_hash_id
from vibe_core.data.rasters import LandsatRaster
from vibe_lib.raster import save_raster_to_asset

LANDSAT_SPYNDEX: Dict[str, str] = {
    "blue": "B",
    "green": "G",
    "red": "R",
    "nir08": "N",
    "swir16": "S1",
    "swir22": "S2",
}


def stack_landsat(
    input: LandsatProduct,
    tmp_folder: str,
    qa_mask: int,
) -> Tuple[AssetVibe, Dict[str, int]]:
    bands2stack = list(input.asset_map.keys())
    band_filepaths = [input.get_downloaded_band(band).path_or_url for band in bands2stack]

    band_idx = {k: v for v, k in enumerate(bands2stack)}
    band_idx["nir"] = band_idx["nir08"]
    # Add band aliases for spyndex
    for k in LANDSAT_SPYNDEX.keys():
        band_idx[LANDSAT_SPYNDEX[k]] = band_idx[k]

    da = (
        xr.open_mfdataset(band_filepaths, engine="rasterio", combine="nested", concat_dim="bands")
        .to_array()
        .squeeze()
    )

    if qa_mask:
        try:
            qa_pixel = (
                rio.open_rasterio(input.get_downloaded_band("qa_pixel").path_or_url)
                .squeeze()  # type: ignore
                .values.astype(int)
            )
            mask = np.bitwise_and(qa_pixel, qa_mask)
            del qa_pixel
            da = da.where(mask)
        except Exception as e:
            raise ValueError(f"qa_pixel not found {e}")

    asset = save_raster_to_asset(da, tmp_folder)
    return asset, band_idx


class CallbackBuilder:
    def __init__(self, qa_mask_value: int):
        self.tmp_dir = TemporaryDirectory()
        self.qa_mask = qa_mask_value

    def __call__(self):
        def process_landsat(
            landsat_product: LandsatProduct,
        ) -> Dict[str, LandsatRaster]:
            img_asset, band_idx = stack_landsat(landsat_product, self.tmp_dir.name, self.qa_mask)

            bands = LandsatRaster.clone_from(
                landsat_product,
                id=gen_hash_id(
                    f"{landsat_product.tile_id}_stacked_landsat",
                    landsat_product.geometry,
                    landsat_product.time_range,
                ),
                assets=[img_asset],
                bands=band_idx,
            )

            return {"landsat_raster": bands}

        return process_landsat

    def __del__(self):
        self.tmp_dir.cleanup()
