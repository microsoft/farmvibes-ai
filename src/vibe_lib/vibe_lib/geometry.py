# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from enum import auto
from functools import reduce
from operator import add
from typing import Any, Dict, List, cast

import geopandas as gpd
import numpy as np
from geopandas import GeoDataFrame
from shapely import geometry as shpg
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry
from strenum import StrEnum

FEATURE = "feature"
FEATURE_COLLECTION = "featurecollection"


class SimplifyBy(StrEnum):
    simplify = auto()
    convex = auto()
    none = auto()


def geojson_to_wkt(json: Dict[str, Any]) -> List[str]:
    "Recursively extracts WKTs from geojson features"

    if "type" not in json:
        return []

    if json["type"].lower() == FEATURE:
        return [shpg.shape(json["geometry"]).wkt]

    if json["type"].lower() == FEATURE_COLLECTION:
        return reduce(add, [geojson_to_wkt(f) for f in json["features"]])

    raise ValueError("Unable to parse GeoJSON input")


def norm_intersection(g1: BaseGeometry, g2: BaseGeometry) -> float:
    """
    Compute normalized intersection area between two geometries
    Area(G1 ∩ G2) / Area(G1)
    """
    return g1.intersection(g2).area / g1.area


def is_approx_within(small_geom: BaseGeometry, big_geom: BaseGeometry, threshold: float) -> bool:
    """
    Maybe not within, but close enough
    """
    return norm_intersection(small_geom, big_geom) > threshold


def is_approx_equal(geom1: BaseGeometry, geom2: BaseGeometry, threshold: float) -> bool:
    return is_approx_within(geom1, geom2, threshold) and is_approx_within(geom2, geom1, threshold)


def wgs_to_utm(geometry: BaseGeometry) -> str:
    """
    Compute UTM sector for a geometry in WGS84 (EPSG:4326)
    """
    c = cast(Point, geometry.centroid)
    lon, lat = c.x, c.y
    assert abs(lon) < 180.0 and abs(lat) < 90.0
    utm_band = str(int(lon + 180 + 6) // 6).zfill(2)
    if lat >= 0:
        epsg_code = "326" + utm_band
    else:
        epsg_code = "327" + utm_band
    return epsg_code


def create_mesh_grid(boundary: Polygon, resolution: int, raster_crs: int = 32611) -> GeoDataFrame:
    boundary_df = gpd.GeoDataFrame(geometry=[boundary], crs=4326).to_crs(raster_crs)  # type: ignore

    if boundary_df is not None and not boundary_df.empty and boundary_df.bounds is not None:
        # Extract the bounds of the polygon
        xmin, ymin, xmax, ymax = list(boundary_df.bounds.itertuples(index=False, name=None))[0]

        # Calculate the number of points in each dimension
        num_x = int((xmax - xmin) / resolution) + 1
        num_y = int((ymax - ymin) / resolution) + 1

        # Generate the coordinate arrays
        x = np.linspace(xmin, xmax, num_x)
        y = np.linspace(ymin, ymax, num_y)

        # Create the mesh grid
        x_, y_ = np.meshgrid(x, y)

        g_df = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(x_.flatten(), y_.flatten()), crs=raster_crs
        ).to_crs(4326)  # type: ignore
        if g_df is not None and not g_df.empty:
            intersecting_locations = cast(GeoDataFrame, g_df[g_df.intersects(boundary)])  # type: ignore
            return intersecting_locations

    raise Exception("Unable to create mesh grid")
