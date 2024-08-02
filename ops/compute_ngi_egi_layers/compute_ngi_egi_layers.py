# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from tempfile import TemporaryDirectory
from typing import Dict

import rioxarray as rio

from vibe_core.data import AssetVibe, LandsatRaster, Raster, gen_guid
from vibe_lib.raster import load_raster, load_raster_match

# Scale and Offset Constants of LST and Rest of the Landsat Bands
SCALE_LST = 0.00341802
OFFSET_LST = 149
SCALE_BAND = 0.0000275
OFFSET_BAND = 0.2


class CallbackBuilder:
    def __init__(self):
        # Create temporary directory to store our new data, which will be transfered to our storage
        # automatically when the op is run in a workflow
        self.tmp_dir = TemporaryDirectory()

    def __call__(self):
        def callback(
            landsat_raster: LandsatRaster,
            ndvi_raster: Raster,
            evaporative_fraction: Raster,
            cloud_water_mask_raster: Raster,
        ) -> Dict[str, Raster]:
            # LAYERS PREPARATION
            lst = rio.open_rasterio(landsat_raster.raster_asset.path_or_url)[
                landsat_raster.bands["lwir11"]
            ]

            # Apply scale and offset value to the band lst band
            lst = load_raster(landsat_raster, bands=["lwir11"])[0]
            lst = (lst * SCALE_LST) + OFFSET_LST

            # Apply scale and offset value to the band lst band
            green = rio.open_rasterio(landsat_raster.raster_asset.path_or_url)[
                landsat_raster.bands["green"]
            ]
            green = (green * SCALE_BAND) - OFFSET_BAND

            # Apply scale and offset value to the band lst band
            nir = rio.open_rasterio(landsat_raster.raster_asset.path_or_url)[
                landsat_raster.bands["nir"]
            ]
            nir = (nir * SCALE_BAND) - OFFSET_BAND

            # Get ndvi index
            ndvi = load_raster_match(ndvi_raster, landsat_raster)[0]

            # Get evaporative fraction raster
            evap_fraxn = load_raster_match(evaporative_fraction, landsat_raster)[0]

            # Get cloud water mask raster
            cloud_water_mask = load_raster_match(cloud_water_mask_raster, landsat_raster)[0]

            # Calculate Green Index
            gi = nir / green

            # Calculate ngi layer from Green Index and ndvi index
            ngi = ndvi * gi

            # Calculate egi layer from Green Index and evaporative fraction layer
            egi = evap_fraxn / gi

            # Apply cloud water mask to ngi, egi, and lst layers
            ngi = ngi * cloud_water_mask
            egi = egi * cloud_water_mask
            lst = lst * cloud_water_mask

            # Save the DataArray to a raster file
            filepath = os.path.join(self.tmp_dir.name, "ngi.tif")
            ngi.rio.to_raster(filepath)
            ngi_asset = AssetVibe(reference=filepath, type="image/tiff", id=gen_guid())

            filepath1 = os.path.join(self.tmp_dir.name, "egi.tif")
            egi.rio.to_raster(filepath1)
            egi_asset = AssetVibe(reference=filepath1, type="image/tiff", id=gen_guid())

            filepath2 = os.path.join(self.tmp_dir.name, "lst.tif")
            lst.rio.to_raster(filepath2)
            lst_asset = AssetVibe(reference=filepath2, type="image/tiff", id=gen_guid())

            return {
                "ngi": Raster.clone_from(
                    landsat_raster, id=gen_guid(), assets=[ngi_asset], bands={"ngi": 0}
                ),
                "egi": Raster.clone_from(
                    landsat_raster, id=gen_guid(), assets=[egi_asset], bands={"egi": 0}
                ),
                "lst": Raster.clone_from(
                    landsat_raster, id=gen_guid(), assets=[lst_asset], bands={"lst": 0}
                ),
            }

        return callback

    def __del__(self):
        self.tmp_dir.cleanup()
