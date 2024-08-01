# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This operator receives a region and a date range and obtains the respective
# NAIP items, returning a list of NaipProduct.
from typing import Any, Dict, List, Tuple, cast

from dateutil.parser import isoparse
from shapely.geometry import shape

from vibe_core.data import DataVibe, NaipProduct
from vibe_lib.planetary_computer import NaipCollection


def convert_product(item: Dict[str, Any]) -> NaipProduct:
    date = isoparse(item["properties"]["datetime"])
    output = NaipProduct(
        id=str(item["id"]),
        time_range=(date, date),
        geometry=item["geometry"],
        assets=[],
        tile_id=str(item["id"]),
        resolution=float(item["properties"]["gsd"]),
        year=int(item["properties"]["naip:year"]),
    )

    return output


def list_naip_products(input_item: DataVibe) -> Dict[str, List[NaipProduct]]:
    collection = NaipCollection()
    input_geometry = shape(input_item.geometry)
    time_range = input_item.time_range
    bbox = cast(Tuple[Any, Any, Any, Any], input_geometry.bounds)
    items = collection.query(roi=bbox, time_range=time_range)
    products = [convert_product(item.to_dict()) for item in items]

    if not products:
        raise RuntimeError(
            f"No product found for time range {input_item.time_range} "
            f"and geometry {input_item.geometry}"
        )

    return {"naip_products": products}


def callback_builder():
    return list_naip_products
