from typing import Any, Dict, List

from dateutil.parser import isoparse

from vibe_core.data import DataVibe, LandsatProduct
from vibe_lib.planetary_computer import LandsatCollection


def convert_product(item: Dict[str, Any]) -> LandsatProduct:
    date = isoparse(item["properties"]["datetime"])
    output = LandsatProduct(
        id=str(item["id"]),
        time_range=(date, date),
        geometry=item["geometry"],
        assets=[],
        tile_id=str(item["id"]),
    )

    return output


def callback_builder():
    def list_landsat_products(
        input_item: DataVibe,
    ) -> Dict[str, List[LandsatProduct]]:
        collection = LandsatCollection()
        items = collection.query(roi=input_item.bbox, time_range=input_item.time_range)

        products = [convert_product(item.to_dict()) for item in items]

        if not products:
            raise RuntimeError(
                f"No product found for time range {input_item.time_range} "
                f"and geometry {input_item.geometry}"
            )
        return {"landsat_products": products}

    return list_landsat_products
