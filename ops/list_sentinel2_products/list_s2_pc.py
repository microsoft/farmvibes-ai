from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from vibe_core.data import DataVibe, Sentinel2Product
from vibe_lib.planetary_computer import Sentinel2Collection, convert_to_s2_product


def callback_builder(num_workers: int):
    def list_sentinel_2_products(
        input_item: DataVibe,
    ) -> Dict[str, List[Sentinel2Product]]:
        collection = Sentinel2Collection()
        items = collection.query(roi=input_item.bbox, time_range=input_item.time_range)

        # We convert products in parallel otherwise this becomes a huge
        # bottleneck due to needing to fetch the absolute orbit from the SAFE file
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            products = list(executor.map(convert_to_s2_product, items))

        if not products:
            raise RuntimeError(
                f"No product found for time range {input_item.time_range} "
                f"and geometry {input_item.geometry}"
            )
        return {"sentinel_products": products}

    return list_sentinel_2_products
