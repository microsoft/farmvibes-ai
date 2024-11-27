# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from datetime import datetime
from functools import partial
from typing import Dict

from shapely.geometry import Polygon, box, mapping

from vibe_core.data import DataVibe
from vibe_core.data.core_types import gen_hash_id


def datavibe_filter(input_item: DataVibe, filter_out: str) -> Dict[str, DataVibe]:
    geometry = input_item.geometry
    time_range = input_item.time_range
    if filter_out in ("all", "geometry"):
        bbox = [0.0, -90.0, 360.0, 90.0]
        polygon: Polygon = box(*bbox, ccw=True)
        geometry = mapping(polygon)  # dummy geometry
    if filter_out in ("all", "time_range"):
        time_range = (datetime(2022, 1, 1), datetime(2022, 1, 1))  # dummy dates
    return {
        "output_item": DataVibe.clone_from(
            input_item,
            id=gen_hash_id("datavibe_filter", geometry=geometry, time_range=time_range),
            geometry=geometry,
            time_range=time_range,
            assets=[],
        )
    }


def callback_builder(filter_out: str):
    filter_out_options = ["all", "time_range", "geometry"]
    if filter_out not in filter_out_options:
        raise ValueError(
            f"Invalid filter_out parameter: {filter_out}. "
            f"Valid values are: {', '.join(filter_out_options)}"
        )
    return partial(datavibe_filter, filter_out=filter_out)
