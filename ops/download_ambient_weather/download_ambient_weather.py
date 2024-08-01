# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import mimetypes
import os
import time
from datetime import timedelta
from random import randint
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, Final, List, cast

import pandas as pd
from ambient_api.ambientapi import AmbientAPI, AmbientWeatherStation
from shapely.geometry import shape

from vibe_core.data import AssetVibe, DataVibe, gen_guid, gen_hash_id
from vibe_core.data.weather import WeatherVibe

# Ambient Weather Station API endpoint
ENDPOINT: Final[str] = "https://api.ambientweather.net/v1"

# time to sleep between API calls to avoid rate limits
ONE_SECOND: Final[int] = 1

# in minutes
SKIP_DATA_FOR_PERIOD = 360

# data points
MAX_FETCH = 288

# data points
MIN_FETCH = 2

# allows failed
FAILED_COUNT = 25

LOGGER = logging.getLogger(__name__)


def get_weather(
    user_input: DataVibe,
    output_dir: str,
    api_key: str,
    app_key: str,
    limit: int,
    feed_interval: int,
) -> WeatherVibe:
    """Gets the Ambient Weather Station data at the location and time specified

    Args:
        user_input: Specifies location and time for data query
        output_dir: directory in which to save data
        api_key: API key used to access Ambient Weather Station API
        app_key: App key used to access Ambient Weather Station API
        limit: Number of data points to be downloaded from ambient service
    Returns:
        Weather data at specified location and time
    Raises:
        RuntimeError: if API service, devices, or data is unreachable
    """
    api = AmbientAPI(
        AMBIENT_ENDPOINT=ENDPOINT,
        AMBIENT_API_KEY=api_key,
        AMBIENT_APPLICATION_KEY=app_key,
    )

    devices = call_ambient_api(api.get_devices)
    assert devices is not None, "No devices found"
    device = get_device(devices, user_input.geometry)

    # create a closure to simplify retries
    def get_data() -> List[Dict[str, Any]]:
        out = device.get_data(end_date=end_date, limit=delta)
        assert out is not None, "No data found"
        return out

    start_date = user_input.time_range[0]
    end_date = user_input.time_range[1]

    delta = end_date - start_date
    delta_sec = (delta.seconds // 60) // feed_interval

    if delta.days > 0:
        delta = delta_sec + delta.days * 24 * 60 // feed_interval
    else:
        delta = delta_sec

    out = []

    # split request into chunks if number of data points is greater than MAX_FETCH
    if limit > MAX_FETCH or delta > MAX_FETCH:
        limit = max(limit, delta)
        lnt = 0
        failed_count = 0

        # for lnt in range(0, limit, MAX_FETCH):
        while end_date > start_date:
            try:
                if (limit - lnt) < MAX_FETCH:
                    delta = limit - lnt
                else:
                    delta = MAX_FETCH

                time.sleep(ONE_SECOND)
                out.extend(cast(List[Any], call_ambient_api(get_data)))
                end_date -= timedelta(minutes=delta * feed_interval)
                lnt += MAX_FETCH
                failed_count = 0
            except Exception:
                # skip from weation station malfunction by every 60 minutes
                end_date -= timedelta(minutes=SKIP_DATA_FOR_PERIOD)
                start_date -= timedelta(minutes=SKIP_DATA_FOR_PERIOD)
                lnt += SKIP_DATA_FOR_PERIOD // feed_interval
                failed_count += 1

                # stop execution if not able to access api 25 times continuously
                if failed_count > FAILED_COUNT:
                    raise RuntimeError("Weather station not responding.")
    else:
        if limit > 0:
            delta = limit
        else:
            delta = MIN_FETCH if delta == 0 else delta

        out = call_ambient_api(get_data)

    file_path = os.path.join(output_dir, "weather.csv")
    pd.DataFrame(out).to_csv(file_path)

    asset = AssetVibe(reference=file_path, type=mimetypes.types_map[".csv"], id=gen_guid())
    return WeatherVibe(
        gen_hash_id(
            f"AmbientWeather_{device.mac_address}",
            user_input.geometry,
            user_input.time_range,
        ),
        user_input.time_range,
        user_input.geometry,
        [asset],
    )


# In the following, pyright fails to detect that we are raising an exception
def get_device(
    devices: List[AmbientWeatherStation], geometry: Dict[str, Any]
) -> AmbientWeatherStation:  # type: ignore
    """Returns a weather device within the bounding box

    Args:
        devices: list of weather stations in this subscription
        geometry: location of interest

    Returns:
        A device within the region

    Raises:
        RuntimteError if no matching device is found
    """
    search_area = shape(geometry)
    for device in devices:
        try:
            device_loc = shape(device.info["coords"]["geo"])  # type: ignore
        except KeyError:
            LOGGER.error("Device info did not contain geolocation for device {}".format(device))
            continue
        if device_loc.within(search_area):
            return device

    log_and_raise_error("No devices found in given geometry {}".format(search_area))


def log_and_raise_error(message: str):
    LOGGER.error(message)
    raise RuntimeError(message)


def call_ambient_api(
    api_call: Callable[[], List[Any]], max_attempts: int = 3, backoff: int = ONE_SECOND
):
    """Call the given function with retries.

    Args:
        api_call: function to call
        max_attempts: tries to make before quitting
        backoff: seconds to wait before first retry. Wait increases between each call.

    Returns:
        result of function call

    Raises:
        RuntimeError if function does not return a non-empty result after max_attempts calls
    """
    # use 1 based counting
    for attempt in range(1, max_attempts + 1):
        result = api_call()
        if result:
            return result
        else:
            LOGGER.warning(
                f"Ambient Weather API call {api_call.__name__} "
                f"failed on try {attempt}/{max_attempts}"
            )
            if attempt < max_attempts:
                time.sleep(backoff + randint(0, 10))
                backoff *= randint(2, 5)
    log_and_raise_error("Could not get data from Ambient Weather API")


class CallbackBuilder:
    def __init__(self, api_key: str, app_key: str, limit: int, feed_interval: int):
        """
        Args:
            api_key: API key used to access Ambient Weather Station API
            app_key: App key used to access Ambient Weather Station API
            limit: Number of data points to be downloaded from ambient service
        """
        self.temp_dir = TemporaryDirectory()
        self.api_key = api_key
        self.app_key = app_key
        self.limit = limit
        self.feed_interval = feed_interval

    def __call__(self):
        def get_weather_data(user_input: List[DataVibe]) -> Dict[str, WeatherVibe]:
            measured_weather = get_weather(
                user_input[0],
                output_dir=self.temp_dir.name,
                api_key=self.api_key,
                app_key=self.app_key,
                limit=self.limit,
                feed_interval=self.feed_interval,
            )
            return {"weather": measured_weather}

        return get_weather_data

    def __del__(self):
        self.temp_dir.cleanup()
