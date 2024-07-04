import re
from datetime import datetime
from tempfile import TemporaryDirectory
from typing import Any, Dict, List

from shapely import geometry as shpg

from vibe_core.data import AirbusProduct, AirbusRaster, AssetVibe, gen_guid
from vibe_lib.airbus import IMAGE_FORMAT, AirBusAPI, Constellation
from vibe_lib.geometry import norm_intersection
from vibe_lib.raster import json_to_asset


def convert_product(product: Dict[str, Any], out_dir: str) -> AirbusRaster:
    dt = datetime.fromisoformat(product["acquisitionDate"].replace("Z", "+00:00"))
    filepath = product.pop("filepath")
    geom = product.pop("geometry")

    asset = AssetVibe(
        reference=filepath,
        type=IMAGE_FORMAT,
        id=gen_guid(),
    )
    vis_asset = json_to_asset({"bands": list(range(3))}, out_dir)
    # Get actual bounds from the raster
    return AirbusRaster(
        id=gen_guid(),
        time_range=(dt, dt),
        geometry=geom,
        assets=[asset, vis_asset],
        bands={k: v for v, k in enumerate(("red", "green", "blue", "nir"))},
        acquisition_id=product.pop("acquisitionIdentifier"),
        extra_info=product,
    )


class CallbackBuilder:
    def __init__(
        self,
        api_key: str,
        projected_crs: bool,
        iou_threshold: float,
        delay: float,
        timeout: float,
    ):
        self.api_key = api_key
        self.projected_crs = projected_crs
        self.iou_thr = iou_threshold
        self.delay = delay
        self.timeout = timeout
        self.tmp_dir = TemporaryDirectory()

    def __call__(self):
        def download_product(api: AirBusAPI, product: AirbusProduct) -> AirbusRaster:
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
                # We need to purchase the product
                # We choose the envelope to avoid having images with a lot of nodata in the library
                order = api.place_order([product.extra_info["id"]], geom.envelope)
                order = api.block_until_order_delivered(order["id"])
                product_id = re.findall(
                    r"items/(.*)/", order["deliveries"][0]["_links"]["download"]["href"]
                )[0]
                owned = api.get_product_by_id(product_id)
            else:
                owned = owned[0]
                product_id = owned["id"]
            owned["filepath"] = api.download_product(product_id, self.tmp_dir.name)
            return convert_product(owned, self.tmp_dir.name)

        def download_products(
            airbus_products: List[AirbusProduct],
        ) -> Dict[str, List[AirbusRaster]]:
            api = AirBusAPI(
                self.api_key,
                self.projected_crs,
                [c for c in Constellation],
                self.delay,
                self.timeout,
            )
            return {"downloaded_products": [download_product(api, p) for p in airbus_products]}

        return download_products
