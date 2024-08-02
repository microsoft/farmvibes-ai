# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# This op receives a date range and geometry and list the respective CDL products
from datetime import datetime
from typing import Dict, List

from shapely import geometry as shpg
from shapely import wkt

from vibe_core.data import DataVibe
from vibe_core.data.core_types import gen_hash_id
from vibe_core.data.products import CDL_DOWNLOAD_URL, CDLProduct
from vibe_core.file_downloader import verify_url


def check_cdl_for_year(year: int) -> bool:
    """Verify if there is a CDL file available for that year"""
    url = CDL_DOWNLOAD_URL.format(year)
    return verify_url(url)


class CallbackBuilder:
    def __init__(self, cdl_geometry_wkt: str):
        with open(cdl_geometry_wkt, "r") as wkt_file:
            self.cdl_geometry = wkt.load(wkt_file)

    def convert_product(self, year: int) -> CDLProduct:
        """Given the year, builds the CDLProduct"""

        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        time_range = (start_date, end_date)

        cdl_geom = shpg.mapping(self.cdl_geometry)

        product = CDLProduct(
            id=gen_hash_id(f"cdl_product_{year}", cdl_geom, time_range),
            time_range=time_range,
            geometry=cdl_geom,
            assets=[],
        )

        return product

    def __call__(self):
        def list_cdl_products(input_item: DataVibe) -> Dict[str, List[CDLProduct]]:
            """List all years for the input time range and create a product for each of them"""

            # Verify if input geometry intersects with cdl geometry
            input_geom = shpg.shape(input_item.geometry)
            if input_geom.intersects(self.cdl_geometry):
                # List all years
                start_date, end_date = input_item.time_range
                input_years = range(start_date.year, end_date.year + 1)

                # Create a product for each year that has a CDL map available
                products = [
                    self.convert_product(year) for year in input_years if check_cdl_for_year(year)
                ]
            else:
                raise ValueError(
                    "Input geometry does not intersect with CDL coverage area (continental US)."
                )

            return {"cdl_products": products}

        return list_cdl_products
