from typing import Any, Dict, Iterable, List

import geopandas as gpd
from shapely import geometry as shpg

from vibe_core.file_downloader import verify_url

GLAD_DOWNLOAD_URL = (
    "https://glad.umd.edu/users/Potapov/GLCLUC2020/Forest_extent_{year}/{tile_name}.tif"
)


def check_glad_for_year(tile_name: str, year: int) -> bool:
    """Verify if there is a GLAD file available for that year"""
    url = GLAD_DOWNLOAD_URL.format(year=year, tile_name=tile_name)
    return verify_url(url)


def get_tile_geometry(tiles_gdf: gpd.GeoDataFrame, tile_name: str) -> Dict[str, Any]:
    selected_tile = tiles_gdf[tiles_gdf["NAME"] == tile_name]
    if not isinstance(selected_tile, gpd.GeoDataFrame) or "geometry" not in selected_tile.columns:
        raise RuntimeError(f"Tile {tile_name} not found in GLAD/Hansen tiles shapefile.")

    selected_geometries = selected_tile["geometry"]

    if not isinstance(selected_geometries, Iterable):
        raise RuntimeError(
            "Failed to load the GLAD/Hansen tiles shapefile. 'geometry' field is not iterable."
        )

    if len(selected_geometries) != 1:
        raise RuntimeError(
            f"Failed to load the GLAD/Hansen tiles shapefile. "
            f"Expected 1 geometry for tile {tile_name}, found {len(selected_geometries)}."
        )

    return shpg.mapping(selected_geometries.iloc[0])


def intersecting_tiles(tiles_gdf: gpd.GeoDataFrame, user_polygon: Dict[str, Any]) -> List[str]:
    user_gdf = gpd.GeoDataFrame({"geometry": [shpg.shape(user_polygon)]})
    intersection = gpd.overlay(user_gdf, tiles_gdf, how="intersection")

    name_intersections = intersection["NAME"]

    if not isinstance(name_intersections, Iterable):
        raise RuntimeError(
            "Failed to load the GLAD/Hansen tiles shapefile. 'NAME' field is not iterable."
        )

    return [str(name) for name in name_intersections]
