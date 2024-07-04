import os
from tempfile import TemporaryDirectory
from typing import Dict

import numpy as np
import rioxarray as rio
import xarray as xr

from vibe_core.data import AssetVibe, LandsatRaster, Raster, gen_guid
from vibe_lib.raster import load_raster_match

# QA_PIXEL mask for cloud cover
CLOUD_DILATED_CLOUD_BIT = 6


class CallbackBuilder:
    def __init__(self, ndvi_threshold: float):
        # Create temporary directory to store our new data, which will be transfered to our storage
        # automatically when the op is run in a workflow
        self.tmp_dir = TemporaryDirectory()
        # Define the parameters
        self.ndvi_threshold = ndvi_threshold

    def __call__(self):
        def callback(landsat_raster: LandsatRaster, ndvi_raster: Raster) -> Dict[str, Raster]:
            # Get QA band from the Landsat raster
            qa_pixel = rio.open_rasterio(landsat_raster.raster_asset.path_or_url)[
                landsat_raster.bands["qa_pixel"]
            ]
            qa_pixel = qa_pixel.astype(np.uint16)

            # Calculate the cloud mask
            cloud_mask = (qa_pixel & (1 << CLOUD_DILATED_CLOUD_BIT)) > 0
            # Assign pixels without cloud contamination as 1 and nan for pixels with cloud
            cloud_mask = xr.where(cloud_mask > 0, 1, np.nan)

            # Retrieve ndvi layer
            ndvi = load_raster_match(ndvi_raster, landsat_raster)[0]

            # Assign pixel value of water bodies as nan and rest as 1
            ndvi_mask = xr.where(ndvi > self.ndvi_threshold, 1, np.nan)

            # Merge cloud and ndvi mask
            cloud_water_mask = cloud_mask * ndvi_mask

            # Save final mask
            filepath = os.path.join(self.tmp_dir.name, "cloud_water_mask.tif")
            cloud_water_mask.rio.to_raster(filepath)
            cwm_asset = AssetVibe(reference=filepath, type="image/tiff", id=gen_guid())

            return {
                "cloud_water_mask": Raster.clone_from(
                    landsat_raster,
                    id=gen_guid(),
                    assets=[cwm_asset],
                    bands={"cloud_water_mask": 0},
                ),
            }

        return callback

    def __del__(self):
        self.tmp_dir.cleanup()
