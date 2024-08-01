# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import hashlib
from datetime import datetime
from functools import partial
from typing import Dict, List

from shapely import geometry as shpg

from vibe_core.data import DataVibe, Era5Product

VARS = {
    "msl": "mean_sea_level_pressure",
    "2t": "2m_temperature",
    "2d": "2m_dewpoint_temperature",
    "100u": "100m_u_component_of_wind",
    "10u": "10m_u_component_of_wind",
    "ssrd": "surface_solar_radiation_downwards",
    "100v": "100m_v_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "tp": "total_precipitation",
    "sst": "sea_surface_temperature",
    "sp": "surface_pressure",
}


def list_era5(input_item: DataVibe, variable: str) -> Dict[str, List[Era5Product]]:
    # Currently only listing the era5 variable that we have on PC in the monthly
    # aggregates (instead of hourly). This should speedup statistics computation
    # (and addition to save these assets in our cache). We may add the much richer
    # set of variables available on CDS (all Era5 variables, Wildfire reanalysis, etc)
    if variable not in VARS.keys():
        raise ValueError(
            f"Requested variable '{variable}' not valid. "
            f"Valid values are {', '.join(VARS.keys())}"
        )

    year_ini = input_item.time_range[0].year
    year_end = input_item.time_range[1].year

    dataset = "reanalysis-era5-single-levels-monthly-means"
    request = {
        "format": "netcdf",
        "variable": [VARS[variable]],
        "product_type": "monthly_averaged_reanalysis",
        "time": "00:00",
        "month": [f"{i:02d}" for i in range(1, 13)],
        "year": [f"{i}" for i in range(year_ini, year_end + 1)],
    }

    res = Era5Product(
        id=hashlib.sha256((dataset + str(request)).encode()).hexdigest(),
        time_range=(datetime(year_ini, 1, 1), datetime(year_end, 12, 31)),
        geometry=shpg.mapping(shpg.box(-180, -90, 180, 90)),
        assets=[],
        item_id="",
        var=VARS[variable],
        cds_request={dataset: request},
    )

    return {"era5_products": [res]}


def callback_builder(variable: str):
    return partial(list_era5, variable=variable)
