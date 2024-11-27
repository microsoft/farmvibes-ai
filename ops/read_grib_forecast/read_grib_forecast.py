# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import mimetypes
import os
from tempfile import TemporaryDirectory
from typing import Dict, List

import xarray as xr
from shapely import geometry as shpg

from vibe_core.data import AssetVibe, DataVibe, GfsForecast, gen_forecast_time_hash_id, gen_guid


def parse_grib_file(grib_file: str, lat: float, lon: float, output_dir: str) -> AssetVibe:
    """Extracts the local data from a global forecast.

    Args:
        grib_file: the path to the grib file for the given time of interest
        lat: the latitude of the forecast [-90, 90]
        lon: the longitude of the forecast [-180, 180]
        output_dir: directory in which to save csv data for this forecast

    Returns:
        VibeAsset containging the forecast for the time and location specified
    """
    # GFS stores longitude in a range from 0-360
    # compute unsigned value from [-180,180] scale
    gfs_lon = (lon + 360) % 360

    keys = {"typeOfLevel": "surface"}
    if not grib_file.endswith("f000.grib"):
        keys["stepType"] = "instant"

    ds = xr.load_dataset(grib_file, engine="cfgrib", filter_by_keys=keys)
    forecast = ds.sel(latitude=lat, longitude=gfs_lon, method="nearest")

    data_file = "{file}_{lat}_{lon}.csv".format(file=grib_file[:-5], lat=lat, lon=lon)

    file_path = os.path.join(output_dir, data_file)

    with open(file_path, "w") as forecast_file:
        forecast_file.write(forecast.to_pandas().to_csv())  # type: ignore

    return AssetVibe(reference=file_path, type=mimetypes.types_map[".csv"], id=gen_guid())


class CallbackBuilder:
    def __init__(self):
        self.temp_dir = TemporaryDirectory()

    def __call__(self):
        def read_forecast(
            location: List[DataVibe], global_forecast: List[GfsForecast]
        ) -> Dict[str, List[GfsForecast]]:
            loc = location[0]
            forecast_data = global_forecast[0]
            # wkt format is (lon, lat)
            lon, lat = shpg.shape(loc.geometry).centroid.coords[0]
            grib_file = forecast_data.assets[0].local_path
            forecast_asset = parse_grib_file(
                grib_file=grib_file, lat=lat, lon=lon, output_dir=self.temp_dir.name
            )

            local_forecast = GfsForecast(
                id=gen_forecast_time_hash_id(
                    "local_forecast", loc.geometry, forecast_data.publish_time, loc.time_range
                ),
                geometry=loc.geometry,
                time_range=loc.time_range,
                assets=[forecast_asset],
                publish_time=forecast_data.publish_time,
            )

            output = {"local_forecast": [local_forecast]}
            return output

        return read_forecast

    def __del__(self):
        self.temp_dir.cleanup()
