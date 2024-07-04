import json
import logging
import mimetypes
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rioxarray as rio
import scipy.ndimage
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap, to_rgba_array
from numpy.lib.stride_tricks import as_strided
from numpy.typing import NDArray
from rasterio import Affine
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.io import DatasetWriter
from rasterio.vrt import WarpedVRT
from rasterio.warp import reproject
from rasterio.windows import Window
from rio_cogeo.cogeo import cog_translate, cog_validate
from rio_cogeo.profiles import cog_profiles

from vibe_core.data import AssetVibe, CategoricalRaster, Raster, gen_guid
from vibe_core.data.rasters import ChunkLimits

if TYPE_CHECKING:
    MaskedArrayType = np.ma.MaskedArray[Any, np.dtype[Any]]
else:
    MaskedArrayType = np.ma.MaskedArray

LOGGER = logging.getLogger(__name__)
# https://kokoalberti.com/articles/geotiff-compression-optimization-guide/
COMPRESSION_KWARGS: Dict[str, Any] = {
    "tiled": True,
    "compress": "ZSTD",
    "zstd_level": 9,
}

FLOAT_COMPRESSION_KWARGS: Dict[str, Any] = {**COMPRESSION_KWARGS, "predictor": 3}

INT_COMPRESSION_KWARGS: Dict[str, Any] = {**COMPRESSION_KWARGS, "predictor": 2}

DEFAULT_NODATA = 100


class RGBA(NamedTuple):
    """
    Int RGBA
    """

    red: int
    green: int
    blue: int
    alpha: int


class FRGB(NamedTuple):
    """
    Float RGB
    """

    red: float
    green: float
    blue: float


class FRGBA(FRGB):
    """
    Float RGBA
    """

    alpha: float


def get_crs(raster: Raster) -> CRS:
    with rasterio.open(raster.raster_asset.url) as src:
        crs = src.crs
    return crs


def open_raster(raster: Raster, *args: Any, **kwargs: Any) -> rasterio.DatasetReader:
    return open_raster_from_ref(raster.raster_asset.url, *args, **kwargs)


def open_raster_from_ref(raster_ref: str, *args: Any, **kwargs: Any) -> rasterio.DatasetReader:
    return rasterio.open(raster_ref, *args, **kwargs)  # type: ignore


def load_raster_from_url(
    raster_url: str,
    band_indices: Optional[Sequence[int]] = None,
    crs: Optional[Any] = None,
    transform: Optional[rasterio.Affine] = None,
    shape: Optional[Tuple[int, int]] = None,
    resampling: Resampling = Resampling.nearest,
    geometry: Optional[Any] = None,
    geometry_crs: Optional[Any] = None,
    dtype: Optional[Any] = None,
) -> xr.DataArray:
    with rasterio.open(raster_url) as src:
        if crs or transform or shape:
            if shape:
                height, width = shape
                if not transform:
                    # Fix bug from rasterio https://github.com/rasterio/rasterio/issues/2346
                    scale_x, scale_y = src.meta["width"] / width, src.meta["height"] / height
                    transform = src.transform * Affine.scale(scale_x, scale_y)
            else:
                height, width = None, None
            dtype = dtype if dtype is not None else src.meta["dtype"]
            src = WarpedVRT(
                src,
                crs=crs,
                transform=transform,
                height=height,
                width=width,
                resampling=resampling,
                dtype=dtype,
            )
        with src:
            data = rio.open_rasterio(src, masked=True)
            if band_indices:  # Read only the desired bands
                data = data[band_indices]
            if geometry:
                data = data.rio.clip([geometry], crs=geometry_crs, all_touched=True, from_disk=True)
    return data


def load_raster(
    raster: Raster,
    bands: Optional[Sequence[Union[int, str]]] = None,
    use_geometry: bool = False,
    crs: Optional[Any] = None,
    transform: Optional[rasterio.Affine] = None,
    shape: Optional[Tuple[int, int]] = None,
    resampling: Resampling = Resampling.nearest,
) -> xr.DataArray:
    """
    Open file and read desired raster bands.
    Bands may be specified as integers (band indices from the TIFF) or strings (band names).
    Band names are mapped to indices by looking up the Raster metadata.
    If desired CRS, transform, and/or shape are defined, the raster will be lazily resampled using
    rasterio's WarpedVRT according to the chosen resampling algorithm.
    Finally, if `use_geometry` is True, the transformed raster will be clipped to the geometry
    in the Raster.
    """
    raster_url = raster.raster_asset.url
    if bands:
        # Map band names to indices if necessary
        band_indices = [raster.bands[b] if isinstance(b, str) else b for b in bands]
    else:
        band_indices = None
    if use_geometry:
        geometry = raster.geometry
        geometry_crs = "epsg:4326"
    else:
        geometry = None
        geometry_crs = None
    data = load_raster_from_url(
        raster_url,
        band_indices,
        crs=crs,
        transform=transform,
        shape=shape,
        resampling=resampling,
        geometry=geometry,
        geometry_crs=geometry_crs,
    )
    return data


def load_raster_match(
    raster: Raster,
    match_raster: Raster,
    bands: Optional[Sequence[Union[int, str]]] = None,
    use_geometry: bool = False,
    resampling: Resampling = Resampling.nearest,
) -> xr.DataArray:
    """
    Load a resampled raster that matches the `match_raster`'s CRS, shape, and transform.
    """
    match_file = match_raster.raster_asset.url
    with rasterio.open(match_file) as ref:
        meta = ref.meta
    return load_raster(
        raster,
        bands,
        use_geometry=use_geometry,
        crs=meta["crs"],
        transform=meta["transform"],
        shape=(meta["height"], meta["width"]),
        resampling=resampling,
    )


def get_profile_from_ref(ref_filepath: str, **kwargs: int) -> Dict[str, Any]:
    """
    Get the TIFF profile from a reference file and update it with the given kwargs.
    """
    with rasterio.open(ref_filepath) as src:
        profile = src.profile
    # We'll store all bands in the same file
    profile.update(kwargs)
    return profile


def check_valid_cog_raster(output_path: str):
    is_valid, errors, warnings = cog_validate(output_path, strict=False)
    if not is_valid and errors:
        message = f"Raster is not a valid COG. Errors: {errors}"
        LOGGER.warning(message)
        return
    if is_valid and warnings:
        message = f"Raster is valid COG, but there are the following warnings {warnings}"
        LOGGER.info(message)
        return
    if is_valid:
        LOGGER.info(f"{output_path} is a valid COG Raster. No Warnings")


def save_raster_to_path(array: xr.DataArray, output_path: str) -> None:
    """
    Save raster to file
    """
    dtype = array.encoding.get("dtype", str(array.dtype))
    if np.issubdtype(dtype, np.floating):
        predictor = 3
    else:
        # For integers
        predictor = 2

    array.rio.to_raster(output_path, tiled=True, compress="ZSTD", zstd_level=9, predictor=predictor)


def save_raster_to_asset(array: xr.DataArray, output_dir: str) -> AssetVibe:
    """
    Save raster to file and return the corresponding asset
    """
    out_id = gen_guid()
    filepath = os.path.join(output_dir, f"{out_id}.tif")
    save_raster_to_path(array, filepath)
    new_asset = AssetVibe(reference=filepath, type=mimetypes.types_map[".tif"], id=out_id)
    return new_asset


def save_raster_from_ref(array: xr.DataArray, output_dir: str, ref_raster: Raster) -> Raster:
    """
    Save raster to file and create a Raster type by copying metadata from a reference raster.
    """
    new_asset = save_raster_to_asset(array, output_dir)
    # Instantiate Raster by copying metadata from reference raster
    return Raster.clone_from(ref_raster, id=gen_guid(), assets=[new_asset])


def get_cmap(cmap_name: str) -> List[RGBA]:
    color_map = plt.get_cmap(cmap_name.lower())
    return [RGBA(*color_map(i)) for i in range(256)]  # type: ignore


def get_categorical_cmap(cmap_name: str, num_classes: int) -> List[RGBA]:
    colors = plt.get_cmap(cmap_name).colors  # type: ignore
    intervals = np.linspace(0, 255, num_classes + 1).round().astype(int)[1:-1]
    return step_cmap_from_colors(colors, intervals)


def step_cmap_from_colors(
    colors: Union[Sequence[Union[FRGB, FRGBA]], NDArray[Any]],
    intervals: Union[Sequence[int], NDArray[Any]],
) -> List[RGBA]:
    interval_array = np.asarray(intervals)
    idx = interval_array.shape - (np.arange(256) < interval_array[:, None]).sum(axis=0)
    # Get RGBA values
    rgba = to_rgba_array(np.asarray(colors)[idx])
    # Convert to RGBA in range 0 - 255
    rgba = np.round(255 * rgba).astype(int).tolist()
    rgba = [RGBA(*c) for c in rgba]
    return rgba


def interpolated_cmap_from_colors(colors: Sequence[RGBA], intervals: Sequence[float]) -> List[RGBA]:
    colors = np.asarray(colors) / 255  # type: ignore
    intervals = np.asarray(intervals)  # type: ignore
    imin, imax = intervals.min(), intervals.max()  # type: ignore
    norm_int = (intervals - imin) / (imax - imin)
    ndvi_cmap = LinearSegmentedColormap.from_list(
        "interpolated_cmap",
        [(i, c) for (i, c) in zip(norm_int, colors)],  # type: ignore
    )
    rgba = np.round(ndvi_cmap(np.linspace(0, 1, 256)) * 255).astype(int).tolist()  # type: ignore
    return [RGBA(*c) for c in rgba]


def json_to_asset(json_dict: Dict[str, Any], output_dir: str) -> AssetVibe:
    uid = gen_guid()
    filepath = os.path.join(output_dir, f"{uid}.json")
    with open(filepath, "w") as f:
        json.dump(json_dict, f)
    return AssetVibe(reference=filepath, type=mimetypes.types_map[".json"], id=uid)


def load_vis_dict(raster: Raster) -> Dict[str, Any]:
    local_path = raster.visualization_asset.local_path
    with open(local_path) as f:
        vis_dict = json.load(f)
    vis_dict["colormap"] = {i: c for i, c in enumerate(vis_dict["colormap"])}
    if isinstance(raster, CategoricalRaster):
        vis_dict["labels"] = raster.categories
        # Position ticks in the middle of the class section
        ticks = np.linspace(0, 255, len(raster.categories) + 1)
        ticks = as_strided(
            ticks,
            shape=(len(raster.categories), 2),
            strides=(ticks.strides[0], ticks.strides[0]),
            writeable=False,
        )
        ticks = ticks.mean(axis=1)  # type: ignore
        vis_dict["ticks"] = ticks
    else:
        num_ticks = 5
        vis_dict["ticks"] = np.linspace(0, 255, num_ticks)
        vis_dict["labels"] = np.linspace(
            vis_dict["range"][0], vis_dict["range"][1], num_ticks
        ).round(1)
    return vis_dict


def compute_index(
    raster: Raster,
    bands: Optional[Sequence[Union[int, str]]],
    index_fun: Callable[[xr.DataArray], xr.DataArray],
    index_name: str,
    output_dir: str,
) -> Raster:
    """
    Open raster, load specified bands, compute index, save a 1-band raster with indices.
    bands can be a sequence of integers (direct band indices) or strings (band names).
    """
    bands_array = load_raster(raster, bands, use_geometry=True)
    # Convert to reflectance values, add minimum value to avoid division by zero
    bands_array = (bands_array.astype(np.float32) * raster.scale + raster.offset).clip(min=1e-6)
    index_array = index_fun(bands_array)

    index_raster = save_raster_from_ref(index_array, output_dir, raster)
    index_raster.bands = {index_name: 0}
    return index_raster


def compute_sobel_gradient(x: NDArray[Any]) -> NDArray[Any]:
    """Use a Sobel filter to compute the magnitude of the gradient in input

    Args:
        x (np.array): Input image (height, width)
    Returns:
        grad_mag (np.array): Gradient magnitude of input
    """
    if len(x.shape) > 2:
        x = np.squeeze(x)

    if len(x.shape) != 2:
        raise ValueError(
            "Invalid NumPy array. Valid arrays have two dimensions or more dimensions of "
            "length 1. E.g. (100, 100) or (1, 100, 100) or (1, 1, 100, 100)"
        )

    grad_y: NDArray[Any] = cast(NDArray[Any], scipy.ndimage.sobel(x, axis=1))
    grad_x: NDArray[Any] = cast(NDArray[Any], scipy.ndimage.sobel(x, axis=0))

    return np.sqrt(grad_x**2 + grad_y**2)


def tile_to_utm(tile_id: str) -> str:
    """
    Get EPSG for a sentinel 2 tile
    """
    utm_band = tile_id[:2]
    is_north = tile_id[2] > "M"
    epsg_code = f"32{'6' if is_north else '7'}{utm_band}"
    return epsg_code


def write_window_to_file(
    data_ar: NDArray[Any],
    mask_ar: Optional[NDArray[Any]],
    write_window: Window,
    filepath: str,
    meta: Dict[str, Any],
) -> None:
    """Helper function to write a window of data to file.

    The function will create the file if it does not exist or will open it in
    `r+` mode if it does. The data array will then be written in the window.
    """
    if mask_ar is not None:
        data_ar[:, mask_ar] = meta["nodata"]
    if os.path.exists(filepath):
        kwargs = {"mode": "r+"}
    else:
        kwargs = {
            "mode": "w",
            **meta,
        }
        kwargs["count"] = data_ar.shape[0]
    with rasterio.open(filepath, **kwargs) as dst:
        dst.write(data_ar, window=write_window)


def read_chunk_series(limits: ChunkLimits, rasters: List[Raster]) -> xr.Dataset:
    rasters = sorted(rasters, key=lambda x: x.time_range[0], reverse=True)
    ref_path = rasters[0].raster_asset.path_or_url

    with rasterio.open(ref_path) as src:
        meta = src.meta

    vrt_options = {
        "resampling": Resampling.bilinear,
        "crs": meta["crs"],
        "transform": meta["transform"],
        "height": meta["height"],
        "width": meta["width"],
    }

    col_off, row_off, width, height = limits
    s0 = row_off
    e0 = row_off + height
    s1 = col_off
    e1 = col_off + width
    res = []
    time = []
    for raster in rasters:
        asset = raster.raster_asset
        t = raster.time_range[0]
        path = asset.path_or_url
        time.append(t)
        with rasterio.open(path) as src:
            with WarpedVRT(src, **vrt_options) as vrt:
                res.append(rio.open_rasterio(vrt, masked=True)[:, s0:e0, s1:e1])
    return xr.concat(res, xr.DataArray(time, name="time", dims="time"))


def get_meta(
    in_path: str,
    width: int,
    height: int,
    transform: Affine,
    nodata: Optional[Union[int, float]] = None,
) -> Dict[str, Any]:
    """
    Get input metadata from input raster and adjust width, height, and transform
    """
    with rasterio.open(in_path) as src:
        kwargs = src.meta.copy()
        if nodata is not None:
            kwargs["nodata"] = nodata
        compression_kwargs = (
            INT_COMPRESSION_KWARGS
            if np.issubdtype(src.meta["dtype"], np.integer)
            else FLOAT_COMPRESSION_KWARGS
        )
        kwargs.update(
            {
                "width": width,
                "height": height,
                "transform": transform,
                "BIGTIFF": "IF_SAFER",
                **compression_kwargs,
            }
        )
        return kwargs


def resample_raster(
    in_path: str,
    out_dir: str,
    width: int,
    height: int,
    transform: Affine,
    resampling: Resampling,
    nodata: Optional[Union[int, float]] = None,
) -> str:
    """
    Compress file and resample (if necessary) to the desired resolution
    """
    kwargs = get_meta(in_path, width, height, transform, nodata)
    out_path = os.path.join(out_dir, f"{gen_guid()}.tif")
    with rasterio.open(in_path) as src:
        with rasterio.open(out_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                if width != src.width or height != src.height:
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=src.crs,
                        resampling=resampling,
                    )
                else:
                    dst.write(src.read(i), i)

    return out_path


def compress_raster(
    src_path: str, dst_path: str, num_threads: Union[int, str] = "all_cpus", **kwargs: Any
) -> None:
    """Load a tif raster and save it in compressed format"""
    with rasterio.open(src_path) as src:
        with rasterio.open(dst_path, "w", **src.meta, **kwargs, num_threads=num_threads) as dst:
            for _, win in src.block_windows():
                dst.write(src.read(window=win), window=win)


def include_raster_overviews(src_path: str):
    """Convert image to COG."""

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmpfile_name = os.path.join(tmp_dir, "tmp_file.tif")
        # Format creation option (see gdalwarp `-co` option)
        output_profile = cog_profiles.get("deflate")
        output_profile.update(dict(BIGTIFF="IF_SAFER"))

        # Dataset Open option (see gdalwarp `-oo` option)
        config = dict(
            GDAL_NUM_THREADS="ALL_CPUS",
            GDAL_TIFF_OVR_BLOCKSIZE="128",
        )

        LOGGER.info("Starting raster COG translation")
        cog_translate(
            src_path,
            tmpfile_name,
            output_profile,
            config=config,
            in_memory=False,
            quiet=True,
        )

        LOGGER.info("Finished raster COG translation")
        shutil.move(tmpfile_name, src_path)


def get_windows(width: int, height: int, win_width: int, win_height: int):
    """
    Returns non-overlapping windows that cover the raster
    """
    wins = []
    for start_r in range(0, height, win_height):
        for start_c in range(0, width, win_width):
            end_c = min(start_c + win_width, width)
            end_r = min(start_r + win_height, height)
            wins.append(Window.from_slices(rows=(start_r, end_r), cols=(start_c, end_c)))
    return wins


def parallel_stack_bands(
    raster_refs: Sequence[str],
    out_path: str,
    num_workers: int,
    block_size: Tuple[int, int],
    resampling: Resampling,
    timeout_s: float = 120.0,
    **kwargs: Any,
):
    """
    Stack bands by reading different band files and writing them into a single file.
    All bands are resampled to the output CRS and affine transform.

    Arguments:
        raster_refs: sequence of references for the files containing band data
        out_path: output filepath
        num_workers: number of threads used to read data
        block_size: size of the block (width, height) that is read by each thread
        resampling: rasterio resampling method used to resample band data
        timeout_s: timeout in seconds for each band read operation (default: 120)
        **kwargs: other keyword arguments will be used to create the output raster.
    Should include things like driver, height, width, transform, crs
    """

    def read_block(raster_url: str, win: Window):
        LOGGER.debug(f"Reading block {win} from {raster_url}")
        with rasterio.open(raster_url) as src:
            with WarpedVRT(
                src,
                crs=kwargs["crs"],
                width=kwargs["width"],
                height=kwargs["height"],
                transform=kwargs["transform"],
                resampling=resampling,
            ) as vrt:
                win_data = vrt.read(window=win)
        LOGGER.debug(f"Done reading block {win} from {raster_url}")
        return win_data, win

    def write_bands(raster_ref: str, wins: List[Window], band_idx: List[int], dst: DatasetWriter):
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            futures = [pool.submit(read_block, raster_ref, win) for win in wins]
            for future in as_completed(futures, timeout=timeout_s):
                try:
                    ar, w = future.result()
                    LOGGER.debug(f"Writing block {w}, bands {band_idx}, to {out_path}")
                    dst.write(ar, band_idx, window=w)
                    LOGGER.debug(f"Done writing block {w}, bands {band_idx}, to {out_path}")
                except Exception as e:
                    LOGGER.exception(f"Exception while processing block from {raster_ref}: {e}")
                    raise e

    wins = [w for w in get_windows(kwargs["width"], kwargs["height"], *block_size)]
    with rasterio.open(out_path, "w", **kwargs, num_threads="all_cpus") as dst:
        offset = 1
        for raster_ref in raster_refs:
            with rasterio.open(raster_ref) as src:
                band_idx = [i + offset for i in range(src.count)]
            try:
                write_bands(raster_ref, wins, band_idx, dst)
                offset = band_idx[-1] + 1
            except TimeoutError:
                msg = f"Timeout while reading raster data from {raster_ref}"
                LOGGER.exception(msg)
                raise TimeoutError(msg)


def serial_stack_bands(
    raster_refs: Sequence[str],
    out_path: str,
    block_size: Tuple[int, int],
    resampling: Resampling,
    **kwargs: Any,
):
    def read_block(raster_ref: str, win: Window):
        LOGGER.debug(f"Reading block {win} from {raster_ref}")
        with rasterio.open(raster_ref) as src:
            with WarpedVRT(
                src,
                crs=kwargs["crs"],
                width=kwargs["width"],
                height=kwargs["height"],
                transform=kwargs["transform"],
                resampling=resampling,
            ) as vrt:
                win_data = vrt.read(window=win)
        LOGGER.debug(f"Done reading block {win} from {raster_ref}")
        return win_data

    def write_bands(raster_ref: str, wins: List[Window], band_idx: List[int], dst: DatasetWriter):
        for w in wins:
            try:
                ar = read_block(raster_ref, w)
                LOGGER.debug(f"Writing block {w}, bands {band_idx}, to {out_path}")
                dst.write(ar, band_idx, window=w)
                LOGGER.debug(f"Done writing block {w}, bands {band_idx}, to {out_path}")
            except Exception as e:
                LOGGER.exception(f"Exception while processing block from {raster_ref}: {e}")
                raise e

    with rasterio.open(out_path, "w", **kwargs, num_threads="all_cpus") as dst:
        offset = 1
        wins = [w for w in get_windows(kwargs["width"], kwargs["height"], *block_size)]
        for raster_ref in raster_refs:
            with rasterio.open(raster_ref) as src:
                band_idx = [i + offset for i in range(src.count)]
            write_bands(raster_ref, wins, band_idx, dst)
            offset = band_idx[-1] + 1


def write_to_raster(data: NDArray[Any], tr: Affine, raster_path: str, raster_crs: CRS) -> AssetVibe:
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        transform=tr,
        dtype=rasterio.float32,
        count=1,
        width=data.shape[1],
        height=data.shape[0],
        crs=raster_crs,
    ) as dst:
        dst.write(data, indexes=1)
    return AssetVibe(reference=raster_path, type="image/tiff", id=gen_guid())
