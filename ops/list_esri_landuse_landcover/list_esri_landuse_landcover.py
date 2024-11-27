# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Dict, List, cast

from dateutil.parser import isoparse
from shapely.geometry import shape

from vibe_core.data import BBox, DataVibe, EsriLandUseLandCoverProduct
from vibe_lib.planetary_computer import EsriLandUseLandCoverCollection


def convert_product(item: Dict[str, Any]) -> EsriLandUseLandCoverProduct:
    start_date = isoparse(item["properties"]["start_datetime"])
    end_date = isoparse(item["properties"]["end_datetime"])
    output = EsriLandUseLandCoverProduct(
        id=str(item["id"]),
        time_range=(start_date, end_date),
        geometry=item["geometry"],
        assets=[],
    )

    return output


def list_products(input_item: DataVibe) -> Dict[str, List[EsriLandUseLandCoverProduct]]:
    collection = EsriLandUseLandCoverCollection()
    input_geometry = shape(input_item.geometry)
    time_range = input_item.time_range
    bbox = cast(BBox, input_geometry.bounds)
    items = collection.query(roi=bbox, time_range=time_range)
    products = [convert_product(item.to_dict()) for item in items]

    if not products:
        raise RuntimeError(
            f"No product found for time range {input_item.time_range} "
            f"and geometry {input_item.geometry}"
        )

    return {"listed_products": products}


def callback_builder():
    return list_products
