"""
Planetary computer model for TerraVibes. Helps query and download items and assets.
"""

import io
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from itertools import product
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import planetary_computer as pc
import requests
from azure.storage.blob import BlobProperties, ContainerClient
from planetary_computer.sas import get_token
from pystac.asset import Asset
from pystac.item import Item
from pystac_client import Client
from requests.exceptions import RequestException
from shapely import geometry as shpg
from shapely.geometry.base import BaseGeometry

from vibe_core.data import S2ProcessingLevel, Sentinel1Product, Sentinel2Product
from vibe_core.data.core_types import BBox
from vibe_core.file_downloader import download_file

CATALOG_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
DATE_FORMAT = "%Y-%m-%d"
RETRY_WAIT = 10
MAX_RETRIES = 5

# https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar/naming-conventions
MODE_SLICE = slice(4, 6)
POLARIZATION_SLICE = slice(14, 16)
YEAR_SLICE = slice(17, 21)
MONTH_SLICE = slice(21, 23)
DAY_SLICE = slice(23, 25)
LOGGER = logging.getLogger(__name__)


class PlanetaryComputerCollection:
    collection: str = ""
    filename_regex: str = r".*/(.*\.\w{3,4})(?:\?|$)"
    asset_keys: List[str] = ["image"]

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.available_collections = get_available_collections()

        if self.collection not in self.available_collections:
            message = (
                f"Invalid collection '{self.collection}'. "
                f"Available collections: {self.available_collections}"
            )
            self.logger.error(message)
            raise ValueError(message)

    def query_by_id(self, id: str) -> Item:
        items = query_catalog_by_ids([self.collection], [id])
        if not items:
            message = f"There is no item with id {id} on collection {self.collection}."
            self.logger.error(message)
            raise KeyError(message)
        return items[0]

    def query(
        self,
        geometry: Optional[BaseGeometry] = None,
        roi: Optional[BBox] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        ids: Optional[List[str]] = None,
        query: Optional[Dict[str, Any]] = None,
    ) -> List[Item]:
        return query_catalog(
            [self.collection],
            geometry=geometry,
            roi=roi,
            time_range=time_range,
            ids=ids,
            query=query,
        )

    def download_asset(self, asset: Asset, out_path: str) -> str:
        """
        Download asset from the planetary computer and save it into the desired path.
        If the output path is a directory, try to infer the filename from the asset href.
        """
        if os.path.isdir(out_path):
            # Resolve name from href
            match = re.match(self.filename_regex, asset.href)
            if match is None:
                raise ValueError(f"Unable to parse filename from asset href: {asset.href}")
            filename = match.groups()[0]
            out_path = os.path.join(out_path, filename)
        for retry in range(MAX_RETRIES):
            href = pc.sign(asset.href)
            try:
                download_file(href, out_path)
                return out_path
            except RequestException as e:
                LOGGER.warning(
                    f"Exception {e} downloading from {href}."
                    f" Retrying after {RETRY_WAIT}s ({retry+1}/{MAX_RETRIES})."
                )
                time.sleep(RETRY_WAIT)
        raise RuntimeError(f"Failed asset {asset.href} after {MAX_RETRIES} retries.")

    def download_item(self, item: Item, out_dir: str):
        """
        Download assets from planetary computer.
        """
        os.makedirs(out_dir)
        asset_paths: List[str] = []
        for k in self.asset_keys:
            asset_paths.append(self.download_asset(item.assets[k], out_dir))
        return asset_paths


class Sentinel2Collection(PlanetaryComputerCollection):
    collection = "sentinel-2-l2a"
    filename_regex = r".*/(.*\.\w{3,4})(?:\?|$)"
    asset_keys: List[str] = [
        "B01",
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B09",
        "B11",
        "B12",
    ]

    def get_cloud_mask(self, item: Item) -> str:
        return pc.sign(urljoin(item.assets["granule-metadata"].href, "QI_DATA/MSK_CLOUDS_B00.gml"))


class Sentinel1GRDCollection(PlanetaryComputerCollection):
    collection: str = "sentinel-1-grd"


class Sentinel1RTCCollection(PlanetaryComputerCollection):
    collection: str = "sentinel-1-rtc"
    asset_keys: List[str] = ["vh", "vv"]


class USGS3DEPCollection(PlanetaryComputerCollection):
    collection = "3dep-seamless"
    asset_keys: List[str] = ["data"]


class CopernicusDEMCollection(PlanetaryComputerCollection):
    collection = "cop-dem-glo-30"
    asset_keys: List[str] = ["data"]


class NaipCollection(PlanetaryComputerCollection):
    collection = "naip"
    asset_keys: List[str] = ["image"]


class LandsatCollection(PlanetaryComputerCollection):
    collection = "landsat-c2-l2"
    asset_keys: List[str] = [
        "qa",
        "red",
        "blue",
        "drad",
        "emis",
        "emsd",
        "trad",
        "urad",
        "atran",
        "cdist",
        "green",
        "nir08",
        "swir16",
        "swir22",
        "qa_pixel",
        "qa_radsat",
        "lwir11",
    ]


class Era5Collection(PlanetaryComputerCollection):
    collection = "era5-pds"
    asset_keys: List[str] = [
        "msl",
        "2t",
        "mx2t",
        "mn2t",
        "2d",
        "100u",
        "10u",
        "ssrd",
        "100v",
        "10v",
        "t0",
        "sst",
        "sp",
    ]


class Modis8DaySRCollection(PlanetaryComputerCollection):
    """
    MODIS Surface Reflectance generated every 8 days.
    Available resolutions are 250m and 500m.
    https://planetarycomputer.microsoft.com/dataset/modis-09Q1-061
    https://planetarycomputer.microsoft.com/dataset/modis-09A1-061
    """

    collections: Dict[int, str] = {250: "modis-09Q1-061", 500: "modis-09A1-061"}

    def __init__(self, resolution: int):
        if resolution not in self.collections:
            raise ValueError(
                f"Expected resolution to be one of {list(self.collections)}, got {resolution}."
            )
        self.collection = self.collections[resolution]
        super().__init__()


class Modis16DayVICollection(PlanetaryComputerCollection):
    """
    MODIS Vegetation Indices generated every 16 days.
    Pixels are chosen from all acquisitions in the 16-day period.
    Available resolutions are 250m and 500m.
    https://planetarycomputer.microsoft.com/dataset/modis-13Q1-061
    """

    collections: Dict[int, str] = {250: "modis-13Q1-061", 500: "modis-13A1-061"}

    def __init__(self, resolution: int):
        if resolution not in self.collections:
            raise ValueError(
                f"Expected resolution to be one of {list(self.collections)}, got {resolution}."
            )
        self.collection = self.collections[resolution]
        super().__init__()


class AlosForestCollection(PlanetaryComputerCollection):
    """
    ALOS Forest/Non-Forest Classification is derived from the ALOS PALSAR Annual
    Mosaic, and classifies the pixels to detect forest cover.
    """

    collection = "alos-fnf-mosaic"
    asset_keys: List[str] = ["C"]
    categories: List[str] = [
        "No data",
        "Forest (>90% canopy cover)",
        "Forest (10-90% canopy cover)",
        "Non-forest",
        "Water",
    ]


class GNATSGOCollection(PlanetaryComputerCollection):
    collection = "gnatsgo-rasters"
    depth_variables = ["aws{}", "soc{}", "tk{}a", "tk{}s"]
    soil_depths = [
        "0_5",
        "0_20",
        "0_30",
        "5_20",
        "0_100",
        "0_150",
        "0_999",
        "20_50",
        "50_100",
        "100_150",
        "150_999",
    ]

    soil_assets = [d.format(v) for (d, v) in product(depth_variables, soil_depths)]

    additional_assets = [
        "mukey",
        "droughty",
        "nccpi3sg",
        "musumcpct",
        "nccpi3all",
        "nccpi3cot",
        "nccpi3soy",
        "pwsl1pomu",
        "rootznaws",
        "rootznemc",
        "musumcpcta",
        "musumcpcts",
        "nccpi3corn",
        "pctearthmc",
    ]

    asset_keys: List[str] = soil_assets + additional_assets


class EsriLandUseLandCoverCollection(PlanetaryComputerCollection):
    collection = "io-lulc-9-class"
    asset_keys: List[str] = ["data"]
    categories: List[str] = [
        "No Data",
        "Water",
        "Trees",
        "Flooded vegetation",
        "Crops",
        "Built area",
        "Bare ground",
        "Snow/ice",
        "Clouds",
        "Rangeland",
    ]


def query_catalog(
    collections: List[str],
    geometry: Optional[BaseGeometry] = None,
    roi: Optional[BBox] = None,
    time_range: Optional[Tuple[datetime, datetime]] = None,
    ids: Optional[List[str]] = None,
    query: Optional[Dict[str, Any]] = None,
) -> List[Item]:
    """
    Query the planetary computer for items that intersect with the desired RoI in the time range
    """
    catalog = Client.open(CATALOG_URL)
    datetime = (
        "/".join(i.strftime(DATE_FORMAT) for i in time_range) if time_range is not None else None
    )
    search = catalog.search(
        collections=collections,
        intersects=shpg.mapping(geometry) if geometry is not None else None,
        bbox=roi,
        datetime=datetime,
        ids=ids,
        query=query,
    )

    items = [item for item in list(search.get_items())]
    return items


def query_catalog_by_ids(collections: List[str], ids: List[str]) -> List[Item]:
    """
    Query the planetary computer for items given a list of ids
    """
    catalog = Client.open(CATALOG_URL)
    search = catalog.search(collections=collections, ids=ids)
    items = [item for item in list(search.get_items())]
    return items


def get_available_collections() -> List[str]:
    cat = Client.open(CATALOG_URL)
    return [collection.id for collection in cat.get_collections()]


def map_sentinel_product_args(item: Item) -> Dict[str, Any]:
    props = item.properties
    kwargs = {
        "geometry": item.geometry,
        "time_range": (item.datetime, item.datetime),
        "relative_orbit_number": props["sat:relative_orbit"],
        "orbit_direction": props["sat:orbit_state"],
        "platform": props["platform"].upper().replace("SENTINEL-", ""),
        "extra_info": {},
        "assets": [],
    }
    return kwargs


def map_s1_product_args(item: Item) -> Dict[str, Any]:
    kwargs = map_sentinel_product_args(item)
    props = item.properties
    kwargs.update(
        {
            "id": item.id,
            "product_name": item.id,  # Name without the unique identifier
            "orbit_number": props["sat:absolute_orbit"],
            "sensor_mode": props["sar:instrument_mode"],
            "polarisation_mode": " ".join(props["sar:polarizations"]),
        }
    )
    return kwargs


def convert_to_s1_product(item: Item) -> Sentinel1Product:
    kwargs = map_s1_product_args(item)
    return Sentinel1Product(**kwargs)


def convert_to_s2_product(item: Item) -> Sentinel2Product:
    kwargs = map_sentinel_product_args(item)
    props = item.properties
    product_name = props["s2:product_uri"].replace(".SAFE", "")
    kwargs.update(
        {
            "id": product_name,
            "product_name": product_name,
            "orbit_number": get_absolute_orbit(item),
            "tile_id": props["s2:mgrs_tile"],
            "processing_level": S2ProcessingLevel.L2A,
        }
    )
    return Sentinel2Product(**kwargs)


def get_absolute_orbit(item: Item) -> int:
    href = item.assets["safe-manifest"].href
    signed_href = pc.sign(href)
    response = requests.get(signed_href)
    tree = ET.parse(io.BytesIO(response.content))
    orbit_element = [e for e in tree.iter() if "orbitNumber" in e.tag]
    if not orbit_element:
        raise RuntimeError(
            f"Could not find orbit element when parsing manifest XML for item {item.id}"
        )
    orbit = orbit_element[0].text
    assert orbit is not None
    return int(orbit)


def get_sentinel1_scene_name(item: Sentinel1Product) -> str:
    collection = Sentinel1GRDCollection()
    stac_item = collection.query_by_id(item.product_name)
    scene_name = stac_item.assets["safe-manifest"].href.split("/")[-2]
    return scene_name


# From example in:
# https://nbviewer.org/github/microsoft/AIforEarthDataSets/blob/main/data/sentinel-1-grd.ipynb
def generate_sentinel1_blob_path(item: Sentinel1Product) -> str:
    scene_name = get_sentinel1_scene_name(item)
    root = "GRD"
    mode = scene_name[MODE_SLICE]
    polarization = scene_name[POLARIZATION_SLICE]  # "DV", for example, is "dual VV/VH"
    year = scene_name[YEAR_SLICE]
    month = scene_name[MONTH_SLICE].lstrip("0")
    day = scene_name[DAY_SLICE].lstrip("0")

    azure_scene_prefix = "/".join([root, year, month, day, mode, polarization, scene_name])

    return azure_scene_prefix


def get_sentinel1_container_client() -> ContainerClient:
    storage_account_name = "sentinel1euwest"
    container_name = "s1-grd"

    storage_account_url = "https://" + storage_account_name + ".blob.core.windows.net/"

    token = get_token(storage_account_name, container_name).token
    container_client = ContainerClient(
        account_url=storage_account_url, container_name=container_name, credential=token
    )
    return container_client


def get_sentinel1_scene_files(item: Sentinel1Product) -> List[BlobProperties]:
    blob_prefix = generate_sentinel1_blob_path(item)

    container_client = get_sentinel1_container_client()
    blob_generator = container_client.list_blobs(name_starts_with=blob_prefix)
    return list(blob_generator)


def get_complete_s1_prefix(scene_files: List[BlobProperties]) -> str:
    prefixes = {"/".join(f["name"].split("/")[:7]) for f in scene_files}
    if len(prefixes) > 1:
        base_pref = next(iter(prefixes))[:-5]
        raise RuntimeError(f"Found multiple prefixes matching '{base_pref}': {prefixes}")
    prefix = next(iter(prefixes))
    return prefix


def validate_dem_provider(name: str, resolution: int) -> PlanetaryComputerCollection:
    valid_providers = {
        "USGS3DEP": {
            "class": USGS3DEPCollection,
            "resolutions": [10, 30],
        },
        "COPERNICUSDEM30": {
            "class": CopernicusDEMCollection,
            "resolutions": [30],
        },
    }
    if name in valid_providers:
        if resolution in valid_providers[name]["resolutions"]:
            return valid_providers[name]["class"]()
        else:
            raise RuntimeError(
                f"Wrong resolution for dem provider {name}. "
                f"Valid resolution(s) is/are {valid_providers[name]['resolutions']}"
            )
    else:
        raise RuntimeError(
            f"Invalid DEM parameter 'provider': {name}. "
            f"Valid providers are {', '.join(valid_providers.keys())}"
        )
