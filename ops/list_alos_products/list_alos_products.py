from typing import Any, Dict, List, cast

from dateutil.parser import parse
from pystac import Item
from shapely import geometry as shpg

from vibe_core.data import AlosProduct, DataVibe
from vibe_lib.planetary_computer import AlosForestCollection


class CallbackBuilder:
    def __init__(self):
        pass

    def validate_item(self, item: Item):
        if item.geometry is None:
            raise ValueError(f"Item {item.id} is missing geometry field")
        if not isinstance(item.geometry, dict):
            raise ValueError(f"Item {item.id} geometry is not a dict")

    def convert_product(self, item: Item) -> AlosProduct:
        self.validate_item(item)
        time_range = tuple(parse(item.properties[k]) for k in ("start_datetime", "end_datetime"))
        geometry = cast(Dict[str, Any], item.geometry)
        return AlosProduct(id=item.id, geometry=geometry, time_range=time_range, assets=[])

    def __call__(self):
        def callback(input_data: DataVibe) -> Dict[str, List[AlosProduct]]:
            collection = AlosForestCollection()
            items = collection.query(
                geometry=shpg.shape(input_data.geometry), time_range=input_data.time_range
            )

            if not items:
                raise ValueError(
                    f"No items found for geometry {input_data.geometry} "
                    f"and time range {input_data.time_range}"
                )

            return {"alos_products": [self.convert_product(i) for i in items]}

        return callback
