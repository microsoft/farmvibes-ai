# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from datetime import datetime
from tempfile import TemporaryDirectory
from typing import Dict, List

from shapely import geometry as shpg
from shapely.ops import unary_union

from vibe_core.data import AirbusPrice, AirbusProduct, gen_guid
from vibe_lib.airbus import AirBusAPI, Constellation
from vibe_lib.geometry import norm_intersection

AMOUNT_UNIT = "kB"


class CallbackBuilder:
    def __init__(self, api_key: str, projected_crs: bool, iou_threshold: float):
        self.api_key = api_key
        self.projected_crs = projected_crs
        self.iou_thr = iou_threshold
        self.tmp_dir = TemporaryDirectory()

    def __call__(self):
        def price_product(api: AirBusAPI, product: AirbusProduct) -> float:
            geom = shpg.shape(product.geometry)
            owned = api.query_owned(geom, product.acquisition_id)
            owned = sorted(
                owned,
                key=lambda o: norm_intersection(geom, shpg.shape(o["geometry"])),
                reverse=True,
            )
            if (
                not owned
                or norm_intersection(geom, shpg.shape(owned[0]["geometry"])) < self.iou_thr
            ):
                # We choose the envelope to avoid having images with a lot of nodata in the library
                quote = api.get_price([product.extra_info["id"]], geom.envelope)["price"]
                if quote["amountUnit"] != AMOUNT_UNIT:
                    raise ValueError(f"Expected amount in kB, got {quote['amountUnit']}")
                return quote["amount"]
            return 0  # We already have it so price is 0

        def price_products(
            airbus_products: List[AirbusProduct],
        ) -> Dict[str, AirbusPrice]:
            api = AirBusAPI(self.api_key, self.projected_crs, [c for c in Constellation])
            total_price = sum(price_product(api, p) for p in airbus_products)
            print(total_price)
            date = datetime.now()
            geom = unary_union([shpg.shape(p.geometry) for p in airbus_products])
            return {
                "products_price": AirbusPrice(
                    id=gen_guid(),
                    time_range=(date, date),
                    geometry=shpg.mapping(geom),
                    assets=[],
                    price=total_price,
                )
            }

        return price_products
