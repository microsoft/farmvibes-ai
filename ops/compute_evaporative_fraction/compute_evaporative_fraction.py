# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from tempfile import TemporaryDirectory
from typing import Any, Dict, cast

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from scipy import ndimage

from vibe_core.data import AssetVibe, LandsatRaster, Raster, gen_guid
from vibe_lib.raster import load_raster, load_raster_match

# DEFINE CONSTANTS
# source: Senay et. al (2013)
K1 = 0.35
K2 = 0.7
LP = 0.65
# Set threshold of minimum pixel size
PIXEL_SIZE_THRESHOLD = 9


class CallbackBuilder:
    def __init__(self, ndvi_hot_threshold: float):
        self.tmp_dir = TemporaryDirectory()
        self.ndvi_hot_threshold = ndvi_hot_threshold

    def __call__(self):
        def calculate_hot_pixels(
            lst_elev_m: xr.DataArray, ndvi_hot_mask: NDArray[Any]
        ) -> NDArray[Any]:
            # Calculate percentile value of lst_elev
            lst_elev_p90 = np.nanpercentile(lst_elev_m, 90)
            lst_elev_p95 = np.nanpercentile(lst_elev_m, 95)

            lst_hot_mask = np.where(lst_elev_m > lst_elev_p90, lst_elev_m, np.nan)
            lst_hot_mask = np.where(lst_hot_mask < lst_elev_p95, lst_hot_mask, np.nan)

            ndvi_hot_mask = np.where(ndvi_hot_mask > self.ndvi_hot_threshold, ndvi_hot_mask, np.nan)
            ndvi_hot_mask = np.where(ndvi_hot_mask > 0, 1, np.nan)

            hot_pixels = lst_hot_mask * ndvi_hot_mask
            return hot_pixels

        def calculate_cold_pixels(
            lst_elev_m: xr.DataArray, ndvi_cold_mask: NDArray[Any]
        ) -> NDArray[Any]:
            # Calculate percentile value of lst_elev
            lst_elev_p02 = np.nanpercentile(lst_elev_m, 2)
            lst_elev_p04 = np.nanpercentile(lst_elev_m, 4)

            lst_cold_mask = np.where(lst_elev_m > lst_elev_p02, lst_elev_m, np.nan)
            lst_cold_mask = np.where(lst_cold_mask < lst_elev_p04, lst_cold_mask, np.nan)

            ndvi_cold_mask = np.where(ndvi_cold_mask > 0, 1, np.nan)

            cold_pixels = lst_cold_mask * ndvi_cold_mask
            return cold_pixels

        def calculate_evap_frxn(
            etrf: xr.DataArray, lst: xr.DataArray, hot_pixel_value: float, cold_pixel_value: float
        ) -> NDArray[Any]:
            etf_nom = hot_pixel_value - lst
            etf_dom = hot_pixel_value - cold_pixel_value
            etf = etf_nom / etf_dom
            evap_frxn = etrf * etf
            evap_frxn = np.where(evap_frxn < 0, 0, evap_frxn)
            return evap_frxn

        def main_processing(
            landsat_raster: LandsatRaster,
            dem_raster: Raster,
            ndvi_raster: Raster,
            cloud_water_mask_raster: Raster,
        ) -> xr.DataArray:
            lst = load_raster(landsat_raster, bands=["lwir11"])[0]
            lst = (lst * 0.00341802) + 149

            dem = load_raster_match(dem_raster, landsat_raster)[0]
            ndvi = load_raster_match(ndvi_raster, landsat_raster)[0]

            lst_elev = lst + (0.0065 * dem)
            cloud_water_mask = load_raster_match(cloud_water_mask_raster, landsat_raster)[0]

            lst_elev_m = lst_elev * cloud_water_mask
            ndvi_m = ndvi * cloud_water_mask

            # Calculate percentile value of ndvi
            ndvi_p01 = np.nanpercentile(ndvi_m, 1)
            ndvi_p90 = np.nanpercentile(ndvi_m, 90)
            ndvi_p95 = np.nanpercentile(ndvi_m, 95)

            # Define ndvi_hot_mask and ndvi_cold_mask here
            ndvi_hot_mask = np.where(ndvi_m < ndvi_p01, ndvi_m, np.nan)
            ndvi_hot_mask = np.where(ndvi_hot_mask > self.ndvi_hot_threshold, ndvi_hot_mask, np.nan)
            ndvi_hot_mask = np.where(ndvi_hot_mask > 0, 1, np.nan)

            ndvi_cold_mask = np.where(ndvi_m > ndvi_p90, ndvi_m, np.nan)
            ndvi_cold_mask = np.where(ndvi_cold_mask < ndvi_p95, ndvi_cold_mask, np.nan)
            ndvi_cold_mask = np.where(ndvi_cold_mask > 0, 1, np.nan)

            hot_pixels = calculate_hot_pixels(lst_elev_m, ndvi_hot_mask)
            cold_pixels = calculate_cold_pixels(lst_elev_m, ndvi_cold_mask)

            hot_pixels_binary = (hot_pixels > 0).astype(int)
            labels, _ = ndimage.label(hot_pixels_binary)  # type: ignore
            sizes = np.bincount(labels.ravel())
            mask_sizes = sizes > PIXEL_SIZE_THRESHOLD
            hot_pixels[~mask_sizes[labels]] = 0  # type: ignore
            hot_pixels = np.where(hot_pixels > 0, hot_pixels, np.nan)
            hot_pixel_value = cast(float, np.nanmedian(hot_pixels))

            cold_pixels_binary = (cold_pixels > 0).astype(int)
            labels, _ = ndimage.label(cold_pixels_binary)  # type: ignore
            sizes = np.bincount(labels.ravel())
            mask_sizes = sizes > PIXEL_SIZE_THRESHOLD
            cold_pixels[~mask_sizes[labels]] = 0  # type: ignore
            cold_pixels = np.where(cold_pixels > 0, cold_pixels, np.nan)
            cold_pixel_value = cast(float, np.nanmin(cold_pixels))

            etrf = ndvi * K1
            etrf = etrf / K2
            etrf = etrf + LP

            evap_frxn = calculate_evap_frxn(etrf, lst, hot_pixel_value, cold_pixel_value)

            evap_frxn_xr = xr.DataArray(
                evap_frxn, dims=cloud_water_mask.dims, coords=cloud_water_mask.coords
            )

            return evap_frxn_xr

        def callback(
            landsat_raster: LandsatRaster,
            dem_raster: Raster,
            ndvi_raster: Raster,
            cloud_water_mask_raster: Raster,
        ) -> Dict[str, Raster]:
            evap_frxn_xr_result = main_processing(
                landsat_raster, dem_raster, ndvi_raster, cloud_water_mask_raster
            )

            filepath = os.path.join(self.tmp_dir.name, "evaporative_fraction.tif")
            evap_frxn_xr_result.rio.to_raster(filepath)
            etrf_asset = AssetVibe(reference=filepath, type="image/tiff", id=gen_guid())

            return {
                "evaporative_fraction": Raster.clone_from(
                    landsat_raster,
                    id=gen_guid(),
                    assets=[etrf_asset],
                    bands={"evaporative_fraction": 0},
                )
            }

        return callback

    def __del__(self):
        self.tmp_dir.cleanup()
