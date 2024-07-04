from datetime import datetime
from typing import Dict, List

from pystac.item import Item

from vibe_core.data import DataVibe
from vibe_core.data.products import ClimatologyLabProduct
from vibe_lib.climatology_lab import (
    ClimatologyLabCollection,
    GridMETCollection,
    TerraClimateCollection,
)


class CallbackBuilder:
    collection: ClimatologyLabCollection

    def __init__(self, variable: str):
        if variable not in self.collection.asset_keys:
            raise ValueError(
                f"Requested variable '{variable}' not valid.\n"
                f"Available properties: {', '.join(self.collection.asset_keys)}"
            )
        self.variable = variable

    def convert_product(self, item: Item) -> ClimatologyLabProduct:
        assert item.geometry is not None, "input Item has no geometry"
        assert item.datetime is not None, "input Item has no datetime"
        time_range = (datetime(item.datetime.year, 1, 1), datetime(item.datetime.year, 12, 31))

        product = ClimatologyLabProduct(
            id=item.id,
            time_range=time_range,
            geometry=item.geometry,
            assets=[],
            url=item.properties["url"],
            variable=item.properties["variable"],
        )
        return product

    def __call__(self):
        def list_climatology_lab(
            input_item: DataVibe,
        ) -> Dict[str, List[ClimatologyLabProduct]]:
            items = self.collection.query(variable=self.variable, time_range=input_item.time_range)

            if not items:
                raise RuntimeError(f"No products found for time range {input_item.time_range}")

            products = [self.convert_product(item) for item in items]
            return {"products": products}

        return list_climatology_lab


class CallbackBuilderGridMET(CallbackBuilder):
    collection = GridMETCollection()


class CallbackBuilderTerraClimate(CallbackBuilder):
    collection = TerraClimateCollection()
