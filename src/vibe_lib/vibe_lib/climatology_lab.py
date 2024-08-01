# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import hashlib
from datetime import datetime
from typing import List, Tuple

import shapely.geometry as shpg
from pystac.item import Item

from vibe_core.file_downloader import verify_url


class ClimatologyLabCollection:
    asset_keys: List[str]
    download_url: str
    geometry_box: Tuple[float, float, float, float]

    def check_url_variable_year(self, variable: str, year: int) -> bool:
        url = self.download_url.format(variable, year)
        return verify_url(url)

    def query(self, variable: str, time_range: Tuple[datetime, datetime]) -> List[Item]:
        start_date, end_date = time_range
        year_range = range(start_date.year, end_date.year + 1)

        items = [
            self._create_item(variable, year)
            for year in year_range
            if self.check_url_variable_year(variable, year)
        ]
        return items

    def _create_item(self, variable: str, year: int) -> Item:
        url = self.download_url.format(variable, year)

        item = Item(
            id=hashlib.sha256(f"{variable}_{year}".encode()).hexdigest(),
            geometry=shpg.mapping(shpg.box(*self.geometry_box)),
            bbox=self.geometry_box,  # type: ignore
            datetime=datetime(year, 1, 1),
            properties={"variable": variable, "url": url},
        )

        return item


class TerraClimateCollection(ClimatologyLabCollection):
    asset_keys: List[str] = [
        "aet",
        "def",
        "pet",
        "ppt",
        "q",
        "soil",
        "srad",
        "swe",
        "tmax",
        "tmin",
        "vap",
        "ws",
        "vpd",
        "PDSI",
    ]

    download_url = "https://climate.northwestknowledge.net/TERRACLIMATE-DATA/TerraClimate_{}_{}.nc"
    geometry_box = (-180, -90, 180, 90)


class GridMETCollection(ClimatologyLabCollection):
    asset_keys: List[str] = [
        "bi",
        "erc",
        "etr",
        "fm1000",
        "fm100",
        "pet",
        "pr",
        "rmax",
        "rmin",
        "sph",
        "srad",
        "th",
        "tmmn",
        "tmmx",
        "vpd",
        "vs",
    ]

    download_url = "https://www.northwestknowledge.net/metdata/data/{}_{}.nc"
    geometry_box = (
        -124.76666663333334,
        25.066666666666666,
        -67.05833330000002,
        49.400000000000006,
    )  # Geometry for contiguous US (from gridMET products)
