import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Tuple, Union

from shapely import wkt
from shapely.geometry import shape

from .core_types import DataVibe


def gen_forecast_time_hash_id(
    name: str,
    geometry: Dict[str, Any],
    publish_time: Union[str, datetime],
    time_range: Tuple[datetime, datetime],
):
    """Generates a SHA-256 hash ID for a forecast time, based on the input parameters.

    :param name: The name of the forecast.

    :param geometry: The geometry associated with the forecast, as a dictionary.

    :param publish_time: The time when the forecast was published, as a string or a datetime object.

    :param time_range: The time range of the forecast, as a tuple of two datetime objects.

    :return: The SHA-256 hash ID of the forecast time.
    """
    if type(publish_time) is datetime:
        publish_time_str = publish_time.isoformat()
    else:
        publish_time_str = str(publish_time)

    return hashlib.sha256(
        (
            name
            + wkt.dumps(shape(geometry))
            + publish_time_str
            + time_range[0].isoformat()
            + time_range[1].isoformat()
        ).encode()
    ).hexdigest()


@dataclass
class GfsForecast(DataVibe):
    """Represents a Global Forecast System (GFS) forecast."""

    publish_time: str
    """The publication time of the forecast in ISO format."""


@dataclass
class WeatherVibe(DataVibe):
    """Represents weather data."""

    pass
