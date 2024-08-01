# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import os
from datetime import datetime
from tempfile import TemporaryDirectory
from typing import Dict, List

from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import ContainerClient

from vibe_core.data import AssetVibe, GfsForecast, gen_forecast_time_hash_id, gen_guid
from vibe_lib.gfs_blob_utils import blob_url_from_offset, get_sas_uri

LOGGER = logging.getLogger(__name__)


def get_noaa_data(time: GfsForecast, output_dir: str, sas_token: str) -> GfsForecast:
    """Get the global forecast for the given input time data

    Args:
        time: GfsForecast containing forecast publish time and desired forecast time
        output_dir: directory in which to save the grib file
        sas_token: token used to access Azure blob storage

    Returns:
        GfsForecast containing global forecast for the specified time

    Raises:
        azure.core.exceptions.ResourceNotFoundError if forecast file cannot be found
    """
    container_client: ContainerClient = ContainerClient.from_container_url(get_sas_uri(sas_token))
    publish_time = datetime.fromisoformat(time.publish_time)
    forecast_time = time.time_range[0]
    forecast_offset = (forecast_time - publish_time).seconds // 3600

    blob_url = blob_url_from_offset(publish_time, forecast_offset)
    grib_file = "{date}T{cycle_runtime:02}-f{offset:03}.grib".format(
        date=publish_time.date().isoformat(),
        cycle_runtime=publish_time.hour,
        offset=forecast_offset,
    )

    file_path = os.path.join(output_dir, grib_file)

    try:
        with open(file_path, "wb") as blob_file:
            blob_file.write(container_client.download_blob(blob_url).readall())
    except ResourceNotFoundError as e:
        # the specified forecast date has no publications
        LOGGER.exception("Failed to download blob {}".format(blob_url))
        raise e

    return GfsForecast(
        id=gen_forecast_time_hash_id(
            "GlobalForecast", time.geometry, publish_time, time.time_range
        ),
        time_range=time.time_range,
        geometry=time.geometry,
        assets=[grib_to_asset(file_path)],
        publish_time=time.publish_time,
    )


def grib_to_asset(file_path: str) -> AssetVibe:
    """Convert the given file to an VibeAsset"""
    return AssetVibe(reference=file_path, type=None, id=gen_guid())


class CallbackBuilder:
    def __init__(self, sas_token: str):
        self.sas_token = sas_token
        self.temp_dir = TemporaryDirectory()

    def __call__(self):
        def get_weather_forecast(time: List[GfsForecast]) -> Dict[str, List[GfsForecast]]:
            global_forecast = get_noaa_data(time[0], self.temp_dir.name, self.sas_token)
            return {"global_forecast": [global_forecast]}

        return get_weather_forecast

    def __del__(self):
        self.temp_dir.cleanup()
