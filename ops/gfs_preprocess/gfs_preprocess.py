# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
from datetime import datetime, time, timedelta, timezone
from typing import Any, Dict, List, Tuple, Union

from azure.storage.blob import ContainerClient
from shapely.geometry import Point, mapping

from vibe_core.data import DataVibe, GfsForecast, gen_forecast_time_hash_id, gen_hash_id
from vibe_lib.gfs_blob_utils import blob_url_from_offset, get_sas_uri

LOGGER = logging.getLogger(__name__)

# Geometry pointing to Null Island
NULL_ISLAND: Dict[str, Any] = mapping(Point(0, 0))

# The number of hours between model cycle runtimes for GFS data
CC_GAP: int = 6


def datetime_to_query_date(
    user_input: DataVibe, sas_token: str
) -> Tuple[datetime, Tuple[datetime, datetime]]:
    """Gets the most relevant model date and forecast hour of product for the given day and time

    Input:
        user_input: EwyaData representing the day and hour of interest
        sas_token: token used to access Azure blob storage

    Output:
        published_datetime: datetime representing the publish date and
                time of the most relevant forecast data
        forecast_datetime: datetime representing the date and time reflected in the forecast
    """
    container_client: ContainerClient = ContainerClient.from_container_url(get_sas_uri(sas_token))
    # get the forecast for the beginning of the time range in UTC
    input_utc = user_input.time_range[0].astimezone(timezone.utc)
    now_utc = datetime.now(tz=timezone.utc)

    if input_utc > now_utc:
        # forecast is for a future time; get the latest data
        publish_date = now_utc
    else:
        # forecast is for a past time; fetch old forecasts
        publish_date = input_utc

    # modify time to be one of 00, 06, 12, 18 hours
    time_utc = publish_date.time()
    query_hour = (time_utc.hour // CC_GAP) * CC_GAP

    published_datetime = datetime.combine(
        publish_date.date(), time.min.replace(hour=query_hour), tzinfo=timezone.utc
    )

    # compute the difference between the forecast publish time and the target forecast time
    forecast_offset = int((input_utc - published_datetime).total_seconds() // 3600)

    # Find the most relevant blob
    blob_found = False
    valid_duration = 1
    while not blob_found:
        blob_url = blob_url_from_offset(published_datetime, forecast_offset)
        blob_client = container_client.get_blob_client(blob=blob_url)
        if blob_client.exists():
            blob_found = True
        else:
            # Try the previous cycle runtime
            published_datetime -= timedelta(hours=CC_GAP)
            forecast_offset += CC_GAP
            if forecast_offset > 120 and forecast_offset <= 384:
                valid_duration = 3
                # forecasts this far into the future are made with 3 hour granularity
                forecast_offset -= forecast_offset % 3
            elif forecast_offset > 384:
                # forecasts are not made this far out
                LOGGER.exception(
                    "Could not find valid forecast for time {}".format(input_utc.isoformat)
                )
                raise RuntimeError("Forecast not found")

    forecast_datetime = published_datetime + timedelta(hours=forecast_offset)
    forecast_end = forecast_datetime + timedelta(hours=valid_duration)
    return published_datetime, (forecast_datetime, forecast_end)


class CallbackBuilder:
    def __init__(self, sas_token: str):
        self.sas_token = sas_token

    def __call__(self):
        def preprocess_input(
            user_input: List[DataVibe],
        ) -> Dict[str, List[Union[GfsForecast, DataVibe]]]:
            publish_time, time_valid = datetime_to_query_date(user_input[0], self.sas_token)
            location = user_input[0].geometry
            time_data = GfsForecast(
                id=gen_forecast_time_hash_id(
                    "forecast_time", NULL_ISLAND, publish_time, time_valid
                ),
                time_range=time_valid,
                geometry=NULL_ISLAND,
                assets=[],
                publish_time=publish_time.isoformat(),
            )
            location_data = DataVibe(
                gen_hash_id("forecast_location", location, time_valid), time_valid, location, []
            )

            return {"time": [time_data], "location": [location_data]}

        return preprocess_input
