from collections import defaultdict
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, cast

import numpy as np
import spyndex
import xarray as xr
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors

from vibe_core.data import Raster
from vibe_lib.raster import (
    RGBA,
    compute_index,
    get_cmap,
    interpolated_cmap_from_colors,
    json_to_asset,
    load_raster,
    save_raster_from_ref,
)

NDVI_CMAP_INTERVALS: List[float] = [
    -1.0,
    -0.2,
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
]
NDVI_CMAP_COLORS: List[RGBA] = [
    RGBA(0, 0, 255, 255),
    RGBA(0, 0, 38, 255),
    RGBA(166, 0, 38, 255),
    RGBA(214, 48, 38, 255),
    RGBA(242, 110, 66, 255),
    RGBA(252, 173, 97, 255),
    RGBA(252, 224, 140, 255),
    RGBA(255, 255, 191, 255),
    RGBA(217, 240, 140, 255),
    RGBA(166, 217, 107, 255),
    RGBA(102, 189, 99, 255),
    RGBA(26, 153, 79, 255),
    RGBA(0, 102, 54, 255),
]


def compute_ndre(bands: xr.DataArray) -> xr.DataArray:
    """
    Normalized difference red edge index
    """
    re, nir = bands
    ndre: xr.DataArray = (nir - re) / (nir + re)
    ndre.rio.write_nodata(100, encoded=True, inplace=True)
    return ndre


def compute_pri(bands: xr.DataArray) -> xr.DataArray:
    """
    Photochemical reflectance index
    """
    re, nir = bands
    pri: xr.DataArray = (re) / (nir + re)
    pri.rio.write_nodata(100, encoded=True, inplace=True)
    return pri


def compute_reci(bands: xr.DataArray) -> xr.DataArray:
    """
    Red-Edge Chlorophyll Vegetation Index
    """
    re, nir = bands
    reci: xr.DataArray = (nir / re) - 1
    reci.rio.write_nodata(100, encoded=True, inplace=True)
    return reci


def compute_methane(bands: xr.DataArray, neighbors: int = 6, sigma: float = 1.8) -> xr.DataArray:
    b12 = bands[-1].to_masked_array()
    m = b12.mask
    b12 = b12.filled(b12.mean())
    other_bands = bands[:-1].to_masked_array()
    m = m | other_bands.mask.any(axis=0)
    other_bands = other_bands.filled(other_bands.mean())
    b12 = gaussian_filter(b12, sigma).squeeze()
    b12_f = b12.flatten()
    other_bands = gaussian_filter(other_bands, sigma)
    x = other_bands.reshape(other_bands.shape[0], -1).T
    nn = NearestNeighbors(n_neighbors=neighbors).fit(x)
    ref_b12_values = np.median(
        b12_f[nn.kneighbors(x, return_distance=False)],  # type: ignore
        axis=1,
    ).reshape(b12.shape)
    index = (b12 - ref_b12_values) / ref_b12_values
    methane_xr = bands[0].astype(np.float32).copy(data=np.ma.masked_array(index, mask=m))
    return methane_xr


def default_vis():
    return {
        "colormap": interpolated_cmap_from_colors(NDVI_CMAP_COLORS, NDVI_CMAP_INTERVALS),
        "range": (-1, 1),
    }


class CallbackBuilder:
    custom_indices: Dict[str, Callable[..., xr.DataArray]] = {
        "methane": compute_methane,
        "ndre": compute_ndre,
        "pri": compute_pri,
        "reci": compute_reci,
    }
    custom_index_bands: Dict[str, List[str]] = {
        "methane": ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B12"],
        "ndre": ["RE1", "N"],
        "pri": ["R", "N"],
        "reci": ["RE1", "N"],
    }
    index_vis: Dict[str, Dict[str, Any]] = defaultdict(
        default_vis, {"methane": {"colormap": get_cmap("gray"), "range": (-0.2, 0.2)}}
    )

    def __init__(self, index: str):
        # the indices ndvi, evi, msevi and ndmi are now computed with spyndex
        if (
            index not in spyndex.indices
            and index.upper() not in spyndex.indices
            and index not in self.custom_indices
        ):
            raise ValueError(
                f"Operation compute_index called with unknown index {index}. "
                f"Available indices are {list(spyndex.indices) + list(self.custom_indices.keys())}."
            )
        self.tmp_dir = TemporaryDirectory()
        if index in self.custom_indices.keys():
            self.name = index
            self.index_fn = self.custom_indices[index]
        else:
            self.name = {i.upper(): i for i in spyndex.indices}[index.upper()]

    def check_raster_bands(self, raster: Raster, bands: List[str]) -> None:
        if not set(bands).issubset(set(raster.bands)):
            raise ValueError(
                f"Raster does not contain bands {bands} needed to compute index {self.name}. "
                f"Bands in input raster are: {', '.join(raster.bands.keys())}."
            )

    def check_constants(self, constants: Dict[str, Any]) -> None:
        unsupported_constants = []
        for k, v in constants.items():
            if v is None or not isinstance(v, (int, float)):
                unsupported_constants.append(k)

        if unsupported_constants:
            raise ValueError(
                f"Index {self.name} still not supported. "
                "Spyndex does not define a default int or float value "
                f"for constants {unsupported_constants}."
            )

    def __call__(self):
        def index_callback(raster: Raster) -> Dict[str, Raster]:
            output_dir = self.tmp_dir.name

            # compute index using spyndex
            if self.name in spyndex.indices:
                bands_spyndex = list(set(spyndex.indices[self.name].bands) - set(spyndex.constants))
                # TODO allow user to use different values for the constants
                const_spyndex = {
                    i: spyndex.constants[i].default
                    for i in set(spyndex.indices[self.name].bands).intersection(
                        set(spyndex.constants)
                    )
                }
                self.check_constants(const_spyndex)
                self.check_raster_bands(raster, bands_spyndex)
                raster_da = load_raster(
                    raster, bands=cast(List[str], bands_spyndex), use_geometry=True
                )
                # Convert to reflectance values, add minimum value to avoid division by zero
                raster_da = (raster_da.astype(np.float32) * raster.scale + raster.offset).clip(
                    min=1e-6
                )
                params = {j: raster_da[i] for i, j in enumerate(bands_spyndex)}
                params.update(const_spyndex)
                idx = spyndex.computeIndex(index=self.name, params=params)
                index_raster = save_raster_from_ref(idx, output_dir, raster)
                index_raster.bands = {self.name: 0}
            else:
                self.check_raster_bands(raster, self.custom_index_bands[self.name])
                index_raster = compute_index(
                    raster,
                    self.custom_index_bands[self.name],
                    self.index_fn,
                    self.name,
                    output_dir,
                )

            vis_dict = {"bands": [0], **self.index_vis[self.name]}
            index_raster.assets.append(json_to_asset(vis_dict, output_dir))

            return {"index": index_raster}

        return index_callback

    def __del__(self):
        self.tmp_dir.cleanup()
