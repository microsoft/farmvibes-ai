# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import partial
from itertools import groupby
from typing import Dict, List

from vibe_core.data import Raster, RasterSequence
from vibe_core.data.core_types import gen_guid


def callback(rasters: List[Raster], criterion: str) -> Dict[str, List[RasterSequence]]:
    key_func = {
        "day_of_year": lambda x: x.time_range[0].timetuple().tm_yday,
        "week": lambda x: x.time_range[0].isocalendar()[1],
        "month": lambda x: x.time_range[0].month,
        "year": lambda x: x.time_range[0].year,
        "month_and_year": lambda x: (x.time_range[0].year, x.time_range[0].month),
    }
    criterion_func = key_func.get(criterion)
    if criterion_func is None:
        raise ValueError(f"Invalid group criterion {criterion}")

    res = []
    for key, group in groupby(sorted(rasters, key=criterion_func), criterion_func):
        group = list(group)
        if isinstance(key, list):
            key = "_".join([str(k) for k in key])

        raster_seq = RasterSequence.clone_from(group[0], f"group_{key}_{gen_guid()}", [])
        for r in group:
            raster_seq.add_item(r)
        res.append(raster_seq)

    return {"raster_groups": res}


def callback_builder(criterion: str):
    return partial(callback, criterion=criterion)
