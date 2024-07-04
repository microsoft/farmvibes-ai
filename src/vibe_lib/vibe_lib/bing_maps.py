"""
BingMaps API interface and auxiliary method to query tiles, download basemaps,
and manipulate between lat-lon coordinates and tile x-y coordinates. Part of the code
is adapted from the following source:
https://learn.microsoft.com/en-us/bingmaps/articles/bing-maps-tile-system
"""

import logging
from datetime import datetime
from typing import Dict, List, Tuple, cast

import numpy as np
import requests
import shapely.geometry as shpg
from pystac.item import Item

from vibe_core.data import BBox

MIN_LATITUDE = -85.05112878
MAX_LATITUDE = 85.05112878
MIN_LONGITUDE = -180
MAX_LONGITUDE = 180
MIN_ZOOM_LEVEL = 1
MAX_ZOOM_LEVEL = 20
NO_TILE_AVALABILITY_KEY, NO_TILE_AVAILABILITY_VALUE = "X-VE-Tile-Info", "no-tile"
LOGGER = logging.getLogger(__name__)


def tile_xy_from_latlon(lat: float, lon: float, zoom_level: int) -> Tuple[int, int]:
    """
    Get the tile x-y coordinates given a lat/lon pair and a zoom level.
    """
    # Clip lat/lon to the valid range
    lat = min(max(lat, MIN_LATITUDE), MAX_LATITUDE)
    lon = min(max(lon, MIN_LONGITUDE), MAX_LONGITUDE)

    # Compute the world map size in pixels for a zoom level
    map_size = 256 * (2**zoom_level)

    # Calculate x-y coordinates from the lat/lon (x-y are float values
    # representing positions as ratio of the map size)
    x = (lon + 180) / 360
    sin_lat = np.sin(lat * np.pi / 180)
    y = 0.5 - np.log((1 + sin_lat) / (1 - sin_lat)) / (4 * np.pi)

    # Transform x-y coordinates to pixel positions and clip to a valid range
    pixel_x = min(max(x * map_size, 0), map_size - 1)
    pixel_y = min(max(y * map_size, 0), map_size - 1)

    # As each tile is 256x256 pixels, get tile x-y coordinates from pixel coordinates
    tile_x = int(np.floor(pixel_x / 256))
    tile_y = int(np.floor(pixel_y / 256))

    return tile_x, tile_y


def latlon_from_tile_xy(tile_x: int, tile_y: int, zoom_level: int) -> Tuple[float, float]:
    """
    Given a tile x-y coordinates and a zoom level, return the lat/lon pair of the
    tile's upper-left corner.
    """

    # Compute the world map size in pixels for a zoom level
    map_size = 256 * (2**zoom_level)

    # Get upper-left corner pixel coordinates for the tile
    pixel_x = tile_x * 256
    pixel_y = tile_y * 256

    # Calculate x-y coordinates from pixel coordinates (x-y are float values
    # representing positions as ratio of the map size)
    x = min(max(pixel_x, 0), map_size - 1) / map_size - 0.5
    y = 0.5 - min(max(pixel_y, 0), map_size - 1) / map_size

    # Convert x-y coordinates to lat/lon
    lat = 90 - 360 * np.arctan(np.exp(-y * 2 * np.pi)) / np.pi
    lon = 360 * x

    return lat, lon


def tiles_from_bbox(bbox: BBox, zoom_level: int) -> List[Tuple[int, int]]:
    """
    Get a list of tile x-y coordinates for all tiles covering the given bounding box
    for a given zoom level.
    """
    lon_bottom_left, lat_bottom_left, lon_top_right, lat_top_right = bbox

    # Get tile x-y coordinates for the bottom-left and top-right corners of the bbox
    tile_x_bottom_left, tile_y_bottom_left = tile_xy_from_latlon(
        lat_bottom_left, lon_bottom_left, zoom_level
    )

    # Do the same for the top-right corner of the bbox
    tile_x_top_right, tile_y_top_right = tile_xy_from_latlon(
        lat_top_right, lon_top_right, zoom_level
    )

    tiles = [
        (tile_x, tile_y)
        for tile_x in range(tile_x_bottom_left, tile_x_top_right + 1)
        for tile_y in range(
            tile_y_top_right, tile_y_bottom_left + 1
        )  # top-right to bottom-left instead because y-axis is inverted
    ]
    return tiles


def quadkey_from_tile_xy(tile_x: int, tile_y: int, zoom_level: int) -> str:
    """
    Build the quadkey string that uniquely identifies a tile with x-y coordinates
    for a given zoom level.

    For more information, please refer to the 'Tile Coordinates and Quadkeys' section of
    https://learn.microsoft.com/en-us/bingmaps/articles/bing-maps-tile-system
    """
    quadkey = ""
    for i in range(zoom_level, 0, -1):
        digit = 0
        mask = 1 << (i - 1)
        if tile_x & mask:
            digit += 1
        if tile_y & mask:
            digit += 2
        quadkey += str(digit)
    return quadkey


def get_geometry_for_tile(tile_x: int, tile_y: int, zoom_level: int) -> shpg.Polygon:
    """
    Get the geometry of the tile with x-y coordinates for a given zoom level.
    """
    # Max lat, min lon because it is the upper-left corner of the tile
    max_lat, min_lon = latlon_from_tile_xy(tile_x, tile_y, zoom_level)
    # Min lat, max lon because it is the bottom-right corner of the tile
    # (computed as the upper-left of x+1, y+1)
    min_lat, max_lon = latlon_from_tile_xy(tile_x + 1, tile_y + 1, zoom_level)
    bbox = shpg.box(min_lon, min_lat, max_lon, max_lat)
    return bbox


def tile_is_available(url: str) -> bool:
    """
    Make a request to BingMaps API to verify if tile represented by url is available for download.
    """
    with requests.get(url, stream=True) as r:
        try:
            r.raise_for_status()
            headers = cast(Dict[str, str], r.headers)
            return (NO_TILE_AVALABILITY_KEY not in headers) or (
                headers[NO_TILE_AVALABILITY_KEY] != NO_TILE_AVAILABILITY_VALUE
            )
        except requests.HTTPError:
            error_details = r.json()["errorDetails"]
            raise ValueError("Error when verifying tile availablity: " + "\n".join(error_details))


class BingMapsCollection:
    """
    BingMaps collection interface to query tiles and download basemaps.
    Reference: https://learn.microsoft.com/en-us/bingmaps/articles/bing-maps-tile-system
    """

    METADATA_URL: str = (
        "http://dev.virtualearth.net/REST/V1/Imagery/Metadata/Aerial"
        "?output=json&include=ImageryProviders&key={BING_MAPS_API_KEY}"
    )

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("No API key provided.")
        self.api_key = api_key
        self.tile_download_url, self.subdomains = self.get_download_url_and_subdomains()

    def get_download_url_and_subdomains(self) -> Tuple[str, List[str]]:
        """Fetch the download URL and subdomains using BingMaps API."""
        try:
            with requests.get(self.METADATA_URL.format(BING_MAPS_API_KEY=self.api_key)) as r:
                r.raise_for_status()
                metadata = r.json()
                url = metadata["resourceSets"][0]["resources"][0]["imageUrl"]
                subdomains = metadata["resourceSets"][0]["resources"][0]["imageUrlSubdomains"]
                return url, subdomains
        except (requests.HTTPError, requests.ConnectionError) as e:
            raise ValueError("Error when retrieving Bing Maps metadata.") from e

    def query_tiles(self, roi: BBox, zoom_level: int) -> List[Item]:
        """Query the collection for tiles that intersect with the given bounding box."""
        tiles = tiles_from_bbox(roi, zoom_level)

        items = []
        for subdomain_idx, tile in enumerate(tiles):
            tile_x, tile_y = tile
            subdomain = self.subdomains[subdomain_idx % len(self.subdomains)]

            quadkey = quadkey_from_tile_xy(tile_x, tile_y, zoom_level)
            url = self.tile_download_url.format(
                quadkey=quadkey,
                api_key=self.api_key,
                subdomain=subdomain,
            )
            if tile_is_available(url):
                geometry = get_geometry_for_tile(tile_x, tile_y, zoom_level)
                item = Item(
                    id=quadkey,
                    geometry=shpg.mapping(geometry),
                    bbox=list(geometry.bounds),
                    datetime=datetime.now(),
                    properties={"url": url},
                )
                items.append(item)
            else:
                LOGGER.info(
                    f"Tile {quadkey} (x {tile_x}, y {tile_y}, ZL {zoom_level}) "
                    "is not available for download. Skipping it."
                )
        return items

    def download_tile(self, url: str, out_path: str):
        """Download a tile from the given URL."""
        with requests.get(url, stream=True) as r:
            try:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            except requests.HTTPError:
                error_details = r.json()["errorDetails"]
                raise ValueError("Error when downloading basemap: " + "\n".join(error_details))
