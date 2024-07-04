from functools import partial
from typing import Any, Dict, List

from dateutil.parser import isoparse
from shapely import geometry as shpg
from shapely.geometry import mapping

from vibe_core.data import DataVibe, Era5Product
from vibe_lib.planetary_computer import Era5Collection

VARS = {
    "msl": "air_pressure_at_mean_sea_level",
    "2t": "air_temperature_at_2_metres",
    "mx2t": "air_temperature_at_2_metres_1hour_Maximum",
    "mn2t": "air_temperature_at_2_metres_1hour_Minimum",
    "2d": "dew_point_temperature_at_2_metres",
    "100u": "eastward_wind_at_100_metres",
    "10u": "eastward_wind_at_10_metres",
    "ssrd": "integral_wrt_time_of_surface_direct_downwelling"
    "_shortwave_flux_in_air_1hour_Accumulation",
    "100v": "northward_wind_at_100_metres",
    "10v": "northward_wind_at_10_metres",
    "tp": "precipitation_amount_1hour_Accumulation",
    "sst": "sea_surface_temperature",
    "sp": "surface_air_pressure",
}


def convert_product(item: Dict[str, Any], var: str) -> Era5Product:
    start_datetime = isoparse(item["properties"]["start_datetime"])
    end_datetime = isoparse(item["properties"]["end_datetime"])
    x_extend = item["properties"]["cube:dimensions"]["lon"]["extent"]
    y_extend = item["properties"]["cube:dimensions"]["lat"]["extent"]
    geometry = mapping(shpg.box(x_extend[0], y_extend[0], x_extend[1], y_extend[1]))

    output = Era5Product(
        id=f"{item['id']}_{var}",
        time_range=(start_datetime, end_datetime),
        geometry=geometry,
        assets=[],
        item_id=str(item["id"]),
        var=VARS[var],
    )

    return output


def list_era5(input_item: DataVibe, variable: str) -> Dict[str, List[Era5Product]]:
    if variable not in VARS.keys():
        raise ValueError(
            f"Requested variable '{variable}' not valid. "
            f"Valid values are {', '.join(VARS.keys())}"
        )
    collection = Era5Collection()
    items = collection.query(roi=input_item.bbox, time_range=input_item.time_range)
    items = filter(lambda item: VARS[variable] in item.assets.keys(), items)
    products = [convert_product(item.to_dict(), variable) for item in items]
    if not products:
        raise RuntimeError(
            f"No product found for time range {input_item.time_range} "
            f"and geometry {input_item.geometry}"
        )
    return {"era5_products": products}


def callback_builder(variable: str):
    return partial(list_era5, variable=variable)
