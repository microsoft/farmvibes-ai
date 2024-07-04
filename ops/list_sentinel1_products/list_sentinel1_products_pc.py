import logging
from typing import Dict, List

import planetary_computer as pc
from shapely import geometry as shpg

from vibe_core.data import DataVibe, Sentinel1Product
from vibe_lib.planetary_computer import (
    Sentinel1GRDCollection,
    Sentinel1RTCCollection,
    convert_to_s1_product,
)

LOGGER = logging.getLogger(__name__)
COLLECTIONS = {"grd": Sentinel1GRDCollection, "rtc": Sentinel1RTCCollection}


def callback_builder(pc_key: str, collection: str):
    collection = collection.lower()
    if collection not in COLLECTIONS:
        col_names = ", ".join(f"'{c}'" for c in COLLECTIONS)
        raise ValueError(
            f"Invalid Sentinel-1 collection '{collection}', expected one of {col_names}"
        )

    def list_sentinel1_products(input_item: DataVibe) -> Dict[str, List[Sentinel1Product]]:
        pc.set_subscription_key(pc_key)

        input_range = input_item.time_range
        input_geom = shpg.shape(input_item.geometry)

        col = COLLECTIONS[collection]()
        items = col.query(geometry=input_geom, time_range=input_range)
        LOGGER.debug(f"Planetary Computer query returned {len(items)} STAC items")
        products = [convert_to_s1_product(item) for item in items]
        if not products:
            raise RuntimeError(
                f"No product found for time range {input_range} and "
                f"and geometry {input_item.geometry}"
            )
        return {"sentinel_products": products}

    return list_sentinel1_products
