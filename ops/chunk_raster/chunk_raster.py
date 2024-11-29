# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import hashlib
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pyproj
import rioxarray
import xarray as xr
from numpy.typing import NDArray
from rasterio.windows import Window, bounds
from shapely import geometry as shpg
from shapely.geometry import mapping
from shapely.ops import transform

from vibe_core.data import ChunkLimits, Raster, RasterChunk, RasterSequence, gen_guid
from vibe_lib.spaceeye.dataset import get_read_intervals, get_write_intervals

PosChunk = Tuple[int, int]


def get_geometry(limits: ChunkLimits, ref: xr.DataArray) -> Dict[str, Any]:
    """
    return geojson with the geometry of the particular chunk
    """
    p = shpg.box(*bounds(Window(*limits), ref.rio.transform()))  # type: ignore

    # convert polygon to lat lon
    if ref.rio.crs is not None and str(ref.rio.crs) != "EPSG:4326":
        crs = str(ref.rio.crs)
        origin = pyproj.CRS(crs)
        dest = pyproj.CRS("EPSG:4326")
        project = pyproj.Transformer.from_crs(origin, dest, always_xy=True).transform
        return mapping(transform(project, p))
    else:
        return mapping(p)


def make_chunk(
    pos: PosChunk,
    size: Tuple[int, int],
    limits: ChunkLimits,
    write_rel_limits: ChunkLimits,
    rasters: List[Raster],
) -> RasterChunk:
    chunk_id = hashlib.sha256(
        (f"chunk-{str(limits)}" + "".join(i.id for i in rasters)).encode()
    ).hexdigest()

    # instead of using the geometry of the rasters, using the computed geometry of
    # the specific chunk
    geom = get_geometry(
        limits,  # type: ignore
        rioxarray.open_rasterio(rasters[0].raster_asset.path_or_url),  # type: ignore
    )

    time_range = [rasters[0].time_range[0], rasters[-1].time_range[0]]
    res = RasterChunk.clone_from(
        rasters[0],
        id=chunk_id,
        assets=[],
        time_range=time_range,
        geometry=geom,
        limits=limits,
        chunk_pos=pos,
        num_chunks=size,
        write_rel_limits=write_rel_limits,
    )
    return res


def meshgrid_1d_array(
    y: NDArray[np.int_], x: NDArray[np.int_]
) -> Tuple[NDArray[np.int_], NDArray[np.int_]]:
    return tuple(i.reshape(-1) for i in np.meshgrid(y, x, indexing="ij"))


def get_limits(
    start_col: NDArray[np.int_],
    start_row: NDArray[np.int_],
    width: NDArray[np.int_],
    height: NDArray[np.int_],
) -> List[ChunkLimits]:
    Y, X = meshgrid_1d_array(start_row, start_col)
    H, W = meshgrid_1d_array(height, width)
    return [tuple(i) for i in np.stack((X, Y, W, H)).T.tolist()]


def make_chunks(
    shape: Tuple[int, ...], step_y: int, step_x: int, rasters: List[Raster]
) -> List[RasterChunk]:
    if len(shape) == 2 or len(shape) == 3:
        # assuming the spatial dimensions are the last two
        end_y, end_x = shape[-2:]
    else:
        raise ValueError(f"Chunk assumes rasters have dimension 2 or 3, but {len(shape)} found")

    start_abs_read_y, end_abs_read_y = get_read_intervals(end_y, step_y, step_y, 0)
    start_abs_read_x, end_abs_read_x = get_read_intervals(end_x, step_x, step_x, 0)
    _, rel_write_y = get_write_intervals(end_y, step_y, step_y, 0)
    _, rel_write_x = get_write_intervals(end_x, step_x, step_x, 0)
    start_rel_write_y, end_rel_write_y = rel_write_y
    start_rel_write_x, end_rel_write_x = rel_write_x

    size = (len(start_abs_read_y), len(start_abs_read_x))
    abs_read_limits = get_limits(
        start_abs_read_x,
        start_abs_read_y,
        end_abs_read_x - start_abs_read_x,
        end_abs_read_y - start_abs_read_y,
    )
    rel_write_limits = get_limits(
        start_rel_write_x,
        start_rel_write_y,
        end_rel_write_x - start_rel_write_x,
        end_rel_write_y - start_rel_write_y,
    )
    Y, X = meshgrid_1d_array(np.arange(size[0]), np.arange(size[1]))
    positions = [tuple(i) for i in np.stack((Y, X)).T.tolist()]

    res = []
    for position, read_limits, write_limits in zip(positions, abs_read_limits, rel_write_limits):
        res.append(make_chunk(position, size, read_limits, write_limits, rasters))

    return res


class CallbackBuilder:
    def __init__(self, step_y: int, step_x: int):
        self.step_y = step_y
        self.step_x = step_x

    def __call__(self):
        def chunk_callback(
            rasters: Union[List[Raster], RasterSequence],
        ) -> Dict[str, List[RasterChunk]]:
            # the latest raster is the reference for shape and for (later) to warp all images
            if isinstance(rasters, RasterSequence):
                rasters = [
                    Raster.clone_from(rasters, gen_guid(), assets=[i])
                    for i in rasters.get_ordered_assets()  # type: ignore
                ]
            else:
                rasters = sorted(rasters, key=lambda x: x.time_range[0], reverse=True)

            ref = rasters[0]

            shape = rioxarray.open_rasterio(ref.raster_asset.path_or_url).shape  # type: ignore

            chunks = make_chunks(shape, self.step_y, self.step_x, rasters)

            return {"chunk_series": chunks}

        return chunk_callback
