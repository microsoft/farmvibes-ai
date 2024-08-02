# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Raster data processing utilities."""

from typing import Any, List, Optional

import geopandas as gpd
import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.mask import mask

from vibe_core.data import Raster
from vibe_core.data.core_types import BaseGeometry


def s2_to_img(ar: NDArray[Any], rgb_idx: List[int] = [3, 2, 1]):
    """Normalize S2 raster values and reordering bands for visualization."""
    return (ar[rgb_idx] / 10000).clip(0, 0.3).transpose((1, 2, 0)) / 0.3


def s1_to_img(ar: NDArray[Any]):
    """Compute Sentinel-1 RGB-composite image for display."""
    nodata = ar.sum(axis=0) == 0
    ar = np.stack((ar[1], ar[0], ar[1] - ar[0]), axis=-1)
    qmin, qmax = np.quantile(ar[~nodata], (0.01, 0.99), axis=0)
    ar = np.clip((ar - qmin[None, None]) / (qmax - qmin)[None, None], 0, 1)
    ar[nodata] = 0
    return ar


def spaceeye_to_img(ar: NDArray[Any]):
    """Normalize SpaceEye raster values and reordering bands for visualization."""
    return s2_to_img(ar, rgb_idx=[2, 1, 0])


def read_raster(
    raster: Raster,
    geometry: Optional[BaseGeometry] = None,
    projected_geometry: Optional[BaseGeometry] = None,
    window: Optional[BaseGeometry] = None,
    **kwargs: Any,
):
    """Read raster data and mask it to the geometry."""
    with rasterio.open(raster.raster_asset.url, **kwargs) as src:
        if geometry is not None or projected_geometry is not None:
            proj_geom = (
                projected_geometry
                if projected_geometry is not None
                else gpd.GeoSeries(geometry, crs="epsg:4326").to_crs(src.crs).iloc[0].envelope  # type: ignore
            )

            return mask(src, [proj_geom], crop=True, **kwargs)
        return src.read(window=window), None


def read_clip_index(raster: Raster, geometry: BaseGeometry):
    """Read raster, mask it to the geometry, and clip values to the 1-99 percentile range."""
    ar = read_raster(raster, geometry, filled=False)[0]
    ar = ar.filled(0)[0]
    qmin, qmax = np.nanquantile(ar, (0.01, 0.99))
    return ar.clip(qmin, qmax)
