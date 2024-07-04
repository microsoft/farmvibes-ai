import hashlib
from datetime import datetime
from typing import Dict, List, cast
from urllib.parse import urljoin

import geopandas as gpd

from vibe_core.data import DataVibe, HansenProduct
from vibe_core.file_downloader import verify_url
from vibe_lib import glad

DATASET_START_YEAR = 2000


class CallbackBuilder:
    def __init__(
        self,
        layer_name: str,
        tile_geometry: str,
        tiles_folder_url: str,
    ):
        self.layer_name = layer_name
        self.tiles_gdf: gpd.GeoDataFrame = cast(gpd.GeoDataFrame, gpd.read_file(tile_geometry))
        # Base urls are expected to be in the format:
        # 'https://storage.googleapis.com/earthenginepartners-hansen/GFC-2022-v1.10/'
        self.tiles_folder_url = tiles_folder_url

        # Make sure folder url ends with a slash
        self.tiles_folder_url = (
            self.tiles_folder_url
            if self.tiles_folder_url.endswith("/")
            else f"{self.tiles_folder_url}/"
        )

        self.final_year = HansenProduct.extract_last_year(self.tiles_folder_url)
        self.version = HansenProduct.extract_version(self.tiles_folder_url)

        # Create an asset template for the products, this will be used to check if the tif files are
        # compatible to 'https://storage.googleapis.com/.../Hansen_GFC-2022-v1.10_50N_000E.tif'
        template = f"Hansen_GFC-{self.final_year}-{self.version}_{{asset_key}}_{{tile_name}}.tif"
        self.asset_template = urljoin(self.tiles_folder_url, template)

    def is_product_available(self, layer_name: str, tile_name: str) -> bool:
        return verify_url(self.asset_template.format(asset_key=layer_name, tile_name=tile_name))

    def validate_time_range(self, input_item: DataVibe):
        start_year = input_item.time_range[0].year
        if start_year != DATASET_START_YEAR:
            raise ValueError(
                f"Start year must be {DATASET_START_YEAR} for Hansen dataset "
                f"version {self.version}-{self.final_year}, received {start_year}"
            )

        end_year = input_item.time_range[1].year
        if end_year > self.final_year:
            raise ValueError(
                f"End year must be <= {self.final_year} for Hansen dataset "
                f"version {self.version}-{self.final_year}, received {end_year}"
            )

    def __call__(self):
        def list_hansen_products(input_item: DataVibe) -> Dict[str, List[HansenProduct]]:
            self.validate_time_range(input_item)
            geom_tiles = glad.intersecting_tiles(self.tiles_gdf, input_item.geometry)

            first_year = input_item.time_range[0].year
            last_year = input_item.time_range[1].year

            out_hansen_products = [
                HansenProduct.clone_from(
                    input_item,
                    id=hashlib.sha256(
                        (
                            f"hansen-product-{self.layer_name}-{tile_name}"
                            f"{first_year}-{last_year}-{self.version}"
                        ).encode()
                    ).hexdigest(),
                    assets=[],
                    time_range=(datetime(first_year, 1, 1), datetime(last_year, 12, 31)),
                    geometry=glad.get_tile_geometry(self.tiles_gdf, tile_name),
                    asset_url=self.asset_template.format(
                        asset_key=self.layer_name, tile_name=tile_name
                    ),
                )
                for tile_name in geom_tiles
                if self.is_product_available(self.layer_name, tile_name)
            ]

            if len(out_hansen_products) == 0:
                raise RuntimeError(
                    f"No Hansen products found for time range {input_item.time_range}"
                    f" and geometry {input_item.geometry}"
                )

            return {"hansen_products": out_hansen_products}

        return list_hansen_products
