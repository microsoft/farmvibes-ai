from datetime import datetime
from tempfile import TemporaryDirectory
from typing import Any, Dict, List

from shapely import geometry as shpg
from shapely.geometry.base import BaseGeometry

from vibe_core.data import AirbusProduct, DataVibe, gen_guid
from vibe_lib.airbus import AirBusAPI, Constellation


def convert_product(product: Dict[str, Any], geom: BaseGeometry) -> AirbusProduct:
    dt = datetime.fromisoformat(product["acquisitionDate"].replace("Z", "+00:00"))
    # This is the geometry for the whole product
    product["product_geometry"] = product.pop("geometry")

    # Get actual bounds from the raster
    return AirbusProduct(
        id=gen_guid(),
        time_range=(dt, dt),
        geometry=shpg.mapping(geom),
        assets=[],
        acquisition_id=product.pop("acquisitionIdentifier"),
        extra_info=product,
    )


class CallbackBuilder:
    def __init__(
        self,
        api_key: str,
        constellations: List[str],
        max_cloud_cover: int,
    ):
        self.api_key = api_key
        self.constellations = [Constellation(c) for c in constellations]
        self.max_cloud_cover = max_cloud_cover
        self.tmp_dir = TemporaryDirectory()

    def __call__(self):
        def download_products(
            input_item: DataVibe,
        ) -> Dict[str, List[AirbusProduct]]:
            api = AirBusAPI(
                self.api_key,
                projected_crs=False,
                constellations=self.constellations,
            )
            geom = shpg.shape(input_item.geometry)

            search_results = api.query(
                geom, input_item.time_range, self.max_cloud_cover, my_workspace=False
            )

            return {"airbus_products": [convert_product(p, geom) for p in search_results]}

        return download_products
