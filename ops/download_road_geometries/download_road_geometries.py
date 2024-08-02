# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from tempfile import TemporaryDirectory
from typing import Dict, cast

import geopandas as gpd
import osmnx as ox
from shapely import geometry as shpg

from vibe_core.data import DataVibe, GeometryCollection
from vibe_core.data.core_types import AssetVibe, gen_guid
from vibe_lib.geometry import wgs_to_utm


def get_road_geometries(geom: shpg.Polygon, network_type: str) -> gpd.GeoDataFrame:
    graph = ox.graph_from_polygon(
        geom, network_type=network_type, truncate_by_edge=True, retain_all=True
    )
    df_edges = cast(gpd.GeoDataFrame, ox.graph_to_gdfs(graph, nodes=False))
    df_edges = cast(gpd.GeoDataFrame, df_edges[df_edges.intersects(geom)])
    # Encode Metadata as strings to avoid lists
    for k in df_edges.columns:
        if k == "geometry":
            continue
        df_edges[k] = df_edges[k].apply(  # type: ignore
            lambda x: ",".join([str(i) for i in x]) if isinstance(x, list) else str(x)
        )
    return cast(gpd.GeoDataFrame, df_edges)


class CallbackBuilder:
    def __init__(self, network_type: str, buffer_size: float):
        self.network_type = network_type
        self.buffer_size = buffer_size
        self.tmp_dir = TemporaryDirectory()

    def __call__(self):
        def callback(input_region: DataVibe) -> Dict[str, GeometryCollection]:
            geom = shpg.box(*input_region.bbox)
            crs = "epsg:4326"
            proj_crs = f"epsg:{wgs_to_utm(geom)}"
            buffered_geom = (
                gpd.GeoSeries(geom, crs=crs)
                .to_crs(proj_crs)
                .buffer(self.buffer_size)
                .to_crs(crs=crs)
                .iloc[0]
                .envelope
            )
            df = get_road_geometries(buffered_geom, self.network_type)
            guid = gen_guid()
            filepath = os.path.join(self.tmp_dir.name, f"{guid}.gpkg")
            df.to_file(filepath, driver="GPKG")
            asset = AssetVibe(reference=filepath, type="application/geopackage+sqlite3", id=guid)

            out = GeometryCollection.clone_from(input_region, id=gen_guid(), assets=[asset])

            return {"roads": out}

        return callback

    def __del__(self):
        self.tmp_dir.cleanup()
