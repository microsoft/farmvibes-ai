from typing import Dict, List

from dateutil.parser import parse
from pystac import Item
from shapely import geometry as shpg

from vibe_core.data import DataVibe, ModisProduct
from vibe_lib.planetary_computer import Modis8DaySRCollection


def convert_product(item: Item, resolution: int) -> ModisProduct:
    time_range = tuple(parse(item.properties[k]) for k in ("start_datetime", "end_datetime"))
    assert item.geometry is not None, f"Item {item.id} is missing geometry field"
    return ModisProduct(
        id=item.id, geometry=item.geometry, time_range=time_range, assets=[], resolution=resolution
    )


def callback_builder(resolution: int):
    available_res = Modis8DaySRCollection.collections.keys()
    if resolution not in available_res:
        raise ValueError(f"Valid resolutions are {available_res}, got {resolution}.")

    def callback(input_data: List[DataVibe]) -> Dict[str, List[ModisProduct]]:
        collection = Modis8DaySRCollection(resolution)
        items: Dict[str, Item] = {}
        for input_datum in input_data:
            input_geom = shpg.shape(input_datum.geometry)
            datum_items = collection.query(geometry=input_geom, time_range=input_datum.time_range)
            for i in datum_items:
                items[i.id] = i
        return {"modis_products": [convert_product(i, resolution) for i in items.values()]}

    return callback
