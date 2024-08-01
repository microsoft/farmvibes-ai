# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import hashlib
import logging
from tempfile import TemporaryDirectory
from typing import Dict, List, Union, cast, overload

import fiona
import geopandas as gpd
from shapely import geometry as shpg
from shapely.geometry.base import BaseGeometry

from vibe_core.data import (
    DownloadedSentinel1Product,
    Sentinel1Raster,
    Sentinel2Product,
    TiledSentinel1Product,
)

LOGGER = logging.getLogger(__name__)
S1List = Union[List[DownloadedSentinel1Product], List[Sentinel1Raster]]
TiledList = Union[List[TiledSentinel1Product], List[Sentinel1Raster]]
KML_DRIVER_NAMES = "kml KML libkml LIBKML".split()


@overload
def prepare_items(
    s1_products: List[DownloadedSentinel1Product], tiles_df: gpd.GeoDataFrame
) -> List[TiledSentinel1Product]: ...


@overload
def prepare_items(
    s1_products: List[Sentinel1Raster], tiles_df: gpd.GeoDataFrame
) -> List[Sentinel1Raster]: ...


def prepare_items(
    s1_products: S1List,
    tiles_df: gpd.GeoDataFrame,
) -> TiledList:
    processing_items = []
    for s1_item in s1_products:
        s1_geom = shpg.shape(s1_item.geometry)
        intersecting_df = cast(gpd.GeoDataFrame, tiles_df[tiles_df.intersects(s1_geom)])
        for _, intersecting_tile in intersecting_df.iterrows():
            geom = cast(BaseGeometry, intersecting_tile["geometry"]).buffer(0)
            tile_id = cast(str, intersecting_tile["Name"])
            id = hashlib.sha256((s1_item.id + tile_id).encode()).hexdigest()
            out_type = (
                TiledSentinel1Product
                if isinstance(s1_item, DownloadedSentinel1Product)
                else Sentinel1Raster
            )
            tiled_s1 = out_type.clone_from(
                s1_item,
                id=id,
                assets=s1_item.assets,
                geometry=shpg.mapping(geom),
                tile_id=tile_id,
            )
            processing_items.append(tiled_s1)
    return processing_items


class CallbackBuilder:
    def __init__(self, tile_geometry: str):
        self.tmp_dir = TemporaryDirectory()
        self.tile_geometry = tile_geometry

    def __call__(self):
        def preprocess_items(
            sentinel1_products: S1List,
            sentinel2_products: List[Sentinel2Product],
        ) -> Dict[str, TiledList]:
            tile_ids = set(p.tile_id for p in sentinel2_products)
            # Make fiona read the file: https://gis.stackexchange.com/questions/114066/
            for driver in KML_DRIVER_NAMES:
                fiona.drvsupport.supported_drivers[driver] = "rw"  # type: ignore

            df = gpd.read_file(self.tile_geometry)
            # Filter only tiles for which we have products
            df = cast(gpd.GeoDataFrame, df[df["Name"].isin(tile_ids)])  # type: ignore

            # Prepare items for preprocessing with the s1 item, target geometry and tile id
            processing_items = prepare_items(sentinel1_products, df)

            return {"tiled_products": processing_items}

        return preprocess_items

    def __del__(self):
        self.tmp_dir.cleanup()
