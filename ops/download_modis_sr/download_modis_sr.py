# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from tempfile import TemporaryDirectory
from typing import Dict, Optional

import numpy as np
import planetary_computer as pc
import rioxarray as rio
import xarray as xr

from vibe_core.data import ModisProduct, ModisRaster, gen_guid
from vibe_lib.planetary_computer import Modis8DaySRCollection
from vibe_lib.raster import save_raster_to_asset

MODIS_SPYNDEX: Dict[str, str] = {
    "sur_refl_b01": "R",
    "sur_refl_b02": "N",
    "sur_refl_b03": "B",
    "sur_refl_b04": "G",
    "sur_refl_b06": "S1",
    "sur_refl_b07": "S2",
}


class CallbackBuilder:
    def __init__(self, qa_mask_value: int, pc_key: Optional[str]):
        self.tmp_dir = TemporaryDirectory()
        self.qa_mask_value = qa_mask_value
        pc.set_subscription_key(pc_key)  # type: ignore

    def __call__(self):
        def callback(product: ModisProduct) -> Dict[str, ModisRaster]:
            col = Modis8DaySRCollection(product.resolution)
            items = col.query(
                roi=product.bbox,
                time_range=product.time_range,
                ids=[product.id],
            )
            assert len(items) == 1
            item = items[0]
            bands = sorted([k for k in item.assets if k.find("sur_refl") >= 0])
            tifs = [col.download_asset(item.assets[k], self.tmp_dir.name) for k in bands]
            da = (
                xr.open_mfdataset(tifs, engine="rasterio", combine="nested", concat_dim="bands")
                .to_array()
                .squeeze()
            )

            if self.qa_mask_value:
                if np.any([b.find("sur_refl_state_") >= 0 for b in bands]):
                    idx = next(
                        filter(lambda b: b[1].find("sur_refl_state_") >= 0, enumerate(bands))
                    )[0]
                    qa_pixel = rio.open_rasterio(tifs[idx]).squeeze().values.astype(int)  # type: ignore
                    mask = np.logical_not(np.bitwise_and(qa_pixel, self.qa_mask_value))
                    del qa_pixel
                    da = da.where(mask)
                else:
                    raise ValueError("sur_refl_state not found")

            asset = save_raster_to_asset(da, self.tmp_dir.name)

            band_idx = {name: idx for idx, name in enumerate(bands)}
            # Add Spyndex aliases to available bands
            for k, v in MODIS_SPYNDEX.items():
                if k in bands:
                    band_idx[v] = band_idx[k]

            return {
                "raster": ModisRaster.clone_from(
                    product,
                    id=gen_guid(),
                    assets=[asset],
                    bands=band_idx,
                )
            }

        return callback

    def __del__(self):
        self.tmp_dir.cleanup()
