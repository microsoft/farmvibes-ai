# This operator receives a region and obtains the digital elevation model
# items associated with the input region. The collection 3dep-seamless
# only covers CONUS (continental us) and contains tiles with distinct
# spatial resolutions (10 and 30 meters). This operator returns a list of
# DemProduct.
from functools import partial
from typing import Any, Dict, List

from dateutil.parser import isoparse
from shapely import geometry as shpg
from shapely import ops as shpo

from vibe_core.data import DataVibe, DemProduct
from vibe_lib.planetary_computer import validate_dem_provider


def convert_product(item: Dict[str, Any], provider: str) -> DemProduct:
    date = isoparse(item["properties"]["datetime"])
    output = DemProduct(
        id=str(item["id"]),
        time_range=(date, date),
        geometry=item["geometry"],
        assets=[],
        tile_id=str(item["id"]),
        resolution=int(item["properties"]["gsd"]),
        provider=provider,
    )

    return output


def list_dem_products(
    input_items: List[DataVibe], resolution: int, provider: str
) -> Dict[str, List[DemProduct]]:
    collection = validate_dem_provider(provider.upper(), resolution)

    geom = shpo.unary_union([shpg.shape(i.geometry) for i in input_items])
    items = collection.query(geometry=geom)

    products = [
        convert_product(item.to_dict(), provider)
        for item in items
        if item.properties["gsd"] == resolution
    ]

    if not products:
        raise RuntimeError("No product found on provider '{provider}' for geometry {geom}")

    return {"dem_products": products}


def callback_builder(resolution: int, provider: str):
    return partial(list_dem_products, resolution=resolution, provider=provider)
