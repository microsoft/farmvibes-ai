import hashlib
import itertools
from datetime import datetime
from typing import Dict, List, cast

import geopandas as gpd

from vibe_core.data import DataVibe, GLADProduct
from vibe_lib import glad


class CallbackBuilder:
    def __init__(self, tile_geometry: str):
        self.tiles_gdf: gpd.GeoDataFrame = cast(gpd.GeoDataFrame, gpd.read_file(tile_geometry))

    def __call__(self):
        def list_glad_products(input_item: DataVibe) -> Dict[str, List[GLADProduct]]:
            geom_tiles = glad.intersecting_tiles(self.tiles_gdf, input_item.geometry)
            years_range = range(input_item.time_range[0].year, input_item.time_range[1].year + 1)
            intersection_years = itertools.product(geom_tiles, years_range)

            out_glad_products = [
                GLADProduct.clone_from(
                    input_item,
                    id=hashlib.sha256((f"glad-product-{tile_name}-{year}").encode()).hexdigest(),
                    assets=[],
                    time_range=(datetime(year, 1, 1), datetime(year, 12, 31)),
                    geometry=glad.get_tile_geometry(self.tiles_gdf, tile_name),
                    url=glad.GLAD_DOWNLOAD_URL.format(year=year, tile_name=tile_name),
                )
                for tile_name, year in intersection_years
                if glad.check_glad_for_year(tile_name, year)
            ]
            if len(out_glad_products) == 0:
                raise RuntimeError(
                    f"No Glad products found for time range {input_item.time_range}"
                    f" and geometry {input_item.geometry}"
                )

            return {"glad_products": out_glad_products}

        return list_glad_products
