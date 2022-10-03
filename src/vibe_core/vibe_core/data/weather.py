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
    # Publication time of the forecast in ISO format
    publish_time: str


@dataclass
class WeatherVibe(DataVibe):
    pass
