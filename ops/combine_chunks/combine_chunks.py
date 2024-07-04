import hashlib
import logging
import mimetypes
import os
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Tuple

import geopandas as gpd
import rasterio
from rasterio.windows import Window
from shapely import geometry as shpg

from vibe_core.data import ChunkLimits, RasterChunk
from vibe_core.data.core_types import AssetVibe, BBox, gen_guid
from vibe_core.data.rasters import Raster
from vibe_lib.raster import FLOAT_COMPRESSION_KWARGS, INT_COMPRESSION_KWARGS

LOGGER = logging.getLogger(__name__)


def get_abs_write_limits(
    read_abs_limits: ChunkLimits, write_rel_limits: ChunkLimits
) -> ChunkLimits:
    return (
        read_abs_limits[0] + write_rel_limits[0],
        read_abs_limits[1] + write_rel_limits[1],
        write_rel_limits[2],
        write_rel_limits[3],
    )


def get_structure_and_meta(
    chunks: List[RasterChunk],
) -> Tuple[Dict[Tuple[int, int], Any], Dict[str, Any]]:
    cs = {}
    for c in chunks:
        cs[(c.chunk_pos)] = dict(
            chunk=c, write_limits=get_abs_write_limits(c.limits, c.write_rel_limits)
        )
    with rasterio.open(cs[(0, 0)]["chunk"].raster_asset.path_or_url) as src:
        meta = src.meta
    ncol, nrow = cs[(0, 0)]["chunk"].num_chunks
    meta["width"] = (
        cs[(ncol - 1, nrow - 1)]["write_limits"][0] + cs[(ncol - 1, nrow - 1)]["write_limits"][2]
    )
    meta["height"] = (
        cs[(ncol - 1, nrow - 1)]["write_limits"][1] + cs[(ncol - 1, nrow - 1)]["write_limits"][3]
    )
    meta["mode"] = "w"
    if meta["dtype"].lower().find("float") >= 0:
        meta.update(FLOAT_COMPRESSION_KWARGS)
    else:
        meta.update(INT_COMPRESSION_KWARGS)
    return cs, meta


def get_combined_tif_and_bounds(
    cs: Dict[Tuple[int, int], Any],
    meta: Dict[str, Any],
    path: str,
) -> Tuple[str, BBox]:
    fname = "combined_image.tif"
    path = os.path.join(path, fname)
    with rasterio.open(path, **meta) as dst:
        bounds = dst.bounds
        for v in cs.values():
            c = v["chunk"]
            write_limits = v["write_limits"]
            window_out = Window(*write_limits)
            window_in = Window(*c.write_rel_limits)
            with rasterio.open(c.raster_asset.path_or_url) as src:
                arr = src.read(window=window_in)
            dst.write(arr, window=window_out)
    return path, bounds


class CallbackBuilder:
    def __init__(self):
        self.tmp_dir = TemporaryDirectory()

    def __call__(self):
        def combine_chunks_callback(chunks: List[RasterChunk]) -> Dict[str, Raster]:
            cs, meta = get_structure_and_meta(chunks)

            path, bounds = get_combined_tif_and_bounds(cs, meta, self.tmp_dir.name)

            asset = AssetVibe(reference=path, type=mimetypes.types_map[".tif"], id=gen_guid())
            res_id = hashlib.sha256("".join(i.id for i in chunks).encode()).hexdigest()
            proj_geom = shpg.box(*bounds)
            proj_crs = meta.get("crs")
            if proj_crs is not None:
                geom = gpd.GeoSeries(proj_geom, crs=proj_crs).to_crs("epsg:4326").iloc[0]
            else:
                LOGGER.warning(
                    "Could not find projected coordinate system for combined raster,"
                    " using geometry as is"
                )
                geom = proj_geom
            res = Raster(
                id=res_id,
                time_range=chunks[0].time_range,
                geometry=shpg.mapping(geom),
                assets=[asset],
                bands=chunks[0].bands,
            )

            return {"raster": res}

        return combine_chunks_callback

    def __del__(self):
        self.tmp_dir.cleanup()
