# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict, List

from pystac import Item

from vibe_core.data import DataVibe, GNATSGOProduct
from vibe_lib.planetary_computer import GNATSGOCollection


def convert_product(item: Item) -> GNATSGOProduct:
    assert item.geometry is not None, "Input item has no geometry"
    assert item.datetime is not None, "Input item has no datetime"

    output = GNATSGOProduct(
        id=item.id,
        time_range=(item.datetime, item.datetime),
        geometry=item.geometry,
        assets=[],
    )
    return output


def callback_builder():
    def callback(input_item: DataVibe) -> Dict[str, List[GNATSGOProduct]]:
        collection = GNATSGOCollection()
        items = collection.query(roi=input_item.bbox)
        products = [convert_product(item) for item in items]
        if not products:
            raise RuntimeError(
                f"No product found for geometry {input_item.geometry}. "
                f"Please, make sure the geometry is within Continental USA"
            )
        return {"gnatsgo_products": products}

    return callback
