import logging
import mimetypes
import os
from collections import defaultdict
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Set, Tuple, cast

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from shapely import geometry as shpg

from vibe_core.data import AssetVibe, Raster, RasterSequence, gen_guid
from vibe_lib.raster import FLOAT_COMPRESSION_KWARGS, INT_COMPRESSION_KWARGS

FIELDS = ("crs", "dtype", "count")
RESOLUTION_METHODS = {
    "equal": None,
    "average": lambda resolutions: tuple(np.mean(resolutions, axis=0)),
    "lowest": lambda resolutions: tuple(np.min(resolutions, axis=0)),
    "highest": lambda resolutions: tuple(np.max(resolutions, axis=0)),
}
LOGGER = logging.getLogger(__name__)


def get_resolution(
    raster_sequence: RasterSequence, resolution_method: str
) -> Optional[Tuple[float, float]]:
    resolutions = []
    for r in raster_sequence.get_ordered_assets():
        with rasterio.open(r.url) as src:
            resolutions.append((src.res[0], src.res[1]))

    if resolution_method == "equal":
        if len(set(resolutions)) > 1:
            raise ValueError(
                "Found multiple resolutions when merging RasterSequence, "
                "but expected all resolutions to be equal."
            )
        return None
    elif resolution_method in ["average", "lowest", "highest"]:
        if len(set(resolutions)) > 1:
            LOGGER.warning(
                "Found multiple resolutions when merging RasterSequence, "
                f"using the {resolution_method} of {len(resolutions)} resolutions."
            )
        return cast(Tuple[float, float], RESOLUTION_METHODS[resolution_method](resolutions))
    else:
        raise ValueError(
            f"Expected resolution method to be in {list(RESOLUTION_METHODS.keys())}. "
            f"Found {resolution_method}."
        )


def merge_rasters(
    raster_sequence: RasterSequence, output_dir: str, resampling: Resampling, resolution: str
) -> Dict[str, Raster]:
    out_id = gen_guid()
    file_path = os.path.join(output_dir, f"{out_id}.tif")
    # All rasters should have the same CRS
    assets_meta: Dict[str, Set[Any]] = defaultdict(set)
    for r in raster_sequence.get_ordered_assets():
        with rasterio.open(r.url) as src:
            for field in FIELDS:
                assets_meta[field].add(src.meta[field])
    for field, field_set in assets_meta.items():
        if len(field_set) > 1:
            raise ValueError(
                f"Expected all rasters in RasterSequence to have the same '{field}', "
                f"found {field_set}"
            )
    crs = assets_meta["crs"].pop()
    dtype = assets_meta["dtype"].pop()

    compression_kwargs = (
        INT_COMPRESSION_KWARGS if np.issubdtype(dtype, np.integer) else FLOAT_COMPRESSION_KWARGS
    )
    if not (np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.floating)):
        ValueError(f"Expected raster with int or float subtype, found {dtype}")

    bounds = tuple(
        gpd.GeoSeries(shpg.shape(raster_sequence.geometry), crs="epsg:4326")
        .to_crs(crs)
        .bounds.iloc[0]
    )

    merge(
        [i.url for i in raster_sequence.get_ordered_assets()],
        bounds=bounds,
        res=get_resolution(raster_sequence, resolution),
        resampling=resampling,
        dst_path=file_path,
        dst_kwds=compression_kwargs,
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Merged raster not found in {file_path}.")

    asset = AssetVibe(reference=file_path, type=mimetypes.types_map[".tif"], id=out_id)
    product = Raster.clone_from(raster_sequence, id=gen_guid(), assets=[asset])
    return {"raster": product}


class CallbackBuilder:
    def __init__(self, resampling: str, resolution: str):
        self.tmp_dir = TemporaryDirectory()
        self.resampling = Resampling[resampling]
        self.resolution = resolution

    def __call__(self):
        def callback(raster_sequence: RasterSequence):
            return merge_rasters(
                raster_sequence,
                output_dir=self.tmp_dir.name,
                resampling=self.resampling,
                resolution=self.resolution,
            )

        return callback

    def __del__(self):
        self.tmp_dir.cleanup()
