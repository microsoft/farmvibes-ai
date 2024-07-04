import datetime
import gc
import mimetypes
import os
from itertools import chain
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Tuple, cast

import numpy as np
from numpy.typing import NDArray
from osgeo import gdal, gdalconst
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, disk

from vibe_core.data import AssetVibe, Sentinel2CloudMask, Sentinel2CloudProbability, gen_guid
from vibe_lib.raster import load_raster_from_url
from vibe_lib.spaceeye.utils import find_s2_product

TileData = List[Tuple[Sentinel2CloudMask, Sentinel2CloudProbability]]


def write_tiff(
    x: NDArray[Any],
    tiff_file: str,
    ref_file: str,
    gdal_type: int = gdalconst.GDT_Float32,
    predictor: int = 3,
):
    """
    USAGE: write_tiff(array, tiff_file, ref_file)
    Use predictor=3 for float types and predictor=2 for integer types.
    """
    gtiff_flags = [
        "COMPRESS=ZSTD",  # also LZW and DEFLATE works well
        "ZSTD_LEVEL=9",  # should be between 1-22, and 22 is highest compression.
        # 9 is default and gets essentially the same compression-rate
        "PREDICTOR=%d" % predictor,  # default is 1, use 2 for ints, and 3 for floats
        "TILED=YES",  # so that we can read sub-arrays efficiently
        "BIGTIFF=YES",  # in case resulting file is >4GB
    ]

    assert x.ndim == 2 or x.ndim == 3
    if x.ndim == 3:
        nx, ny, nbands = x.shape
    else:
        nx, ny = x.shape
        nbands = 1

    if not os.path.exists(ref_file):
        raise (FileNotFoundError("<%s> doesn't exist" % ref_file))
    ds = gdal.Open(ref_file)
    if (ds.RasterYSize != nx) and (ds.RasterXSize != ny):
        print("Size mismatch between reference file and input array")
        print("x: %s, ref_file: %d, %d" % (x.shape, ds.RasterYSize, ds.RasterXSize))

    outDrv = gdal.GetDriverByName("GTiff")
    out = outDrv.Create(tiff_file, ny, nx, nbands, gdal_type, gtiff_flags)
    out.SetProjection(ds.GetProjection())
    out.SetGeoTransform(ds.GetGeoTransform())
    if x.ndim == 3:
        for i in range(nbands):
            out.GetRasterBand(i + 1).WriteArray(x[:, :, i])
    else:
        out.GetRasterBand(1).WriteArray(x)
    out.FlushCache()
    del out  # guarantee the flush
    del ds


def read_s2_bands(
    tif_file: str, bands: List[int], transpose: bool = False, dtype: type = np.uint16
) -> NDArray[Any]:
    """
    USAGE: x = read_s2_bands(s2_file, [2,3,4])
    The command above reads in the RGB bands of the sentinel-2 tif file.
    """
    ds = gdal.Open(tif_file)
    nb = ds.RasterCount
    nx = ds.RasterYSize
    ny = ds.RasterXSize
    for i in bands:
        if i >= nb:
            print("Band %d does not exist, only %d bands in %s" % (i, nb, tif_file))
            assert i < nb
    if not transpose:
        x = np.zeros((len(bands), nx, ny), dtype=dtype)
        for i, b in enumerate(bands):
            band = ds.GetRasterBand(b + 1)
            x[i, :, :] = band.ReadAsArray()
    else:
        x = np.zeros((nx, ny, len(bands)), dtype=dtype)
        for i, b in enumerate(bands):
            band = ds.GetRasterBand(b + 1)
            x[:, :, i] = band.ReadAsArray()
    return x


def compute_missing_mask(s2_file: str, dilation: int = 1):
    # TCI is no longer explicitly stored
    bands_10m = read_s2_bands(s2_file, [1, 2, 3, 7])

    # A dicey proposition, but it seems like 0 == NO_DATA in all bands.
    missing_mask = np.min(bands_10m, axis=0) == 0

    # Takes lots of memory, free up fast
    del bands_10m

    # Try hard to free it up
    gc.collect()

    # Compute missing mask using binary dilation
    if dilation > 1 and np.max(missing_mask) == 0:
        selem = disk(dilation)
        missing_mask = binary_dilation(missing_mask, selem)

    return missing_mask


def kill_labels_(clabel: NDArray[Any], min_area: int) -> List[Any]:
    """
    USAGE: kill_list = kill_labels(clabel, min_area)
    Make a list of regions with area below min_area and return the list of regions.
    """
    props = regionprops(clabel)
    kill_list = []
    for p in props:
        if p.area < min_area:
            kill_list.append(p.label)
    return kill_list


def remove_small_components(cmask: NDArray[Any], min_area: int = 400):
    """
    USAGE: new_mask = remove_small_components(cmask, min_area=400)
    First removes small connected cloud components, then fill in small
    connected holes in clouds to make for a smoother cloud mask.
    """
    assert cmask.ndim == 2
    cm2_comp = label(cmask)  # remove small clouds
    tmp = cmask.copy()

    kill_list = kill_labels_(cm2_comp, min_area)  # type: ignore
    small_clouds = np.isin(cm2_comp, kill_list)  # type: ignore

    tmp[small_clouds] = False
    cm2_inv = label(~tmp)  # fill small holes in clouds
    kill_list = kill_labels_(cm2_inv, min_area)  # type: ignore
    small_cloud_holes = np.isin(cm2_inv, kill_list)  # type: ignore
    tmp[small_cloud_holes] = True

    return tmp


def shift_arr(
    cloud_probs: List[str],
    cloud_masks: List[str],
    T: int,
    w2: int,
    cm1_arr: List[NDArray[Any]],
    cm2_arr: List[NDArray[Any]],
    min_prob: float,
) -> Tuple[List[NDArray[Any]], List[NDArray[Any]]]:
    """
    USAGE: cm1_arr, cm2_arr = shift_arr(s2_files, T, w2, cm1_arr, cm2_arr, min_prob)
    Remove the first mask in the cm1_arr and cm2_arr and read the next masks in.
    This is used to maintain a window (in time) of cloud-masks without having to read
    in masks that have already been read in.
    """
    c1_new = [cm1_arr[i + 1] for i in range(2 * T)]
    c2_new = [cm2_arr[i + 1] for i in range(2 * T)]

    cm1, cm2 = load_cloud_masks(cloud_probs[w2], cloud_masks[w2], min_prob)
    c1_new.append(cm1)
    c2_new.append(cm2)

    return c1_new, c2_new


def compute_mask_with_missing_clouds(
    cm1_arr: List[NDArray[Any]],
    cm2_arr: List[NDArray[Any]],
    idx: int,
    max_extra_cloud: float,
    min_area: int,
    dilation: int,
) -> NDArray[Any]:
    cm1 = np.dstack(cm1_arr)
    cm2 = np.dstack(cm2_arr)
    x = np.sum(np.logical_and(cm2, np.logical_not(cm1)), axis=2)
    suspect = np.logical_and(x > max_extra_cloud, cm2[:, :, idx])
    suspect = np.logical_and(suspect, np.logical_not(cm1[:, :, idx]))

    new_mask = cm2[:, :, idx].copy()
    new_mask[suspect] = cm1[suspect, idx]  # i.e. = False

    new_mask = remove_small_components(new_mask, min_area=min_area)
    old_mask = cm1[:, :, idx]
    # don't switch off clouds in original built in mask
    new_mask = np.logical_or(old_mask, new_mask)

    if dilation > 1:
        selem = disk(dilation)
        new_mask = binary_dilation(new_mask, selem)

    return new_mask


def fill_missing_pixels(ref_file: str, new_mask: NDArray[Any], tmp_dir: str) -> str:
    """
    Since part of the region may be outside the footprint of the orbit
    we need to handle missing pixels in some way.  Here we choose to
    simply mark them as clouds and let the reconstruction algorithm
    handle it.  We detect missing pixels by looking for TCI pixels where
    the RGB bands are all zero.
    """

    # Add missing pixels as clouds
    out_file = os.path.join(tmp_dir, f"{gen_guid()}.tif")
    write_tiff(
        new_mask.astype(np.uint8), out_file, ref_file, gdal_type=gdalconst.GDT_Byte, predictor=2
    )

    return out_file


def load_cloud_masks(
    cloudless_prob_path: str, l1c_cloud_path: str, min_prob: float
) -> Tuple[NDArray[Any], NDArray[Any]]:
    cmask = load_raster_from_url(l1c_cloud_path).to_numpy()[0]
    # Open it and fill masked values as clouds
    cprob = load_raster_from_url(cloudless_prob_path).to_masked_array()[0]
    cmask[cprob.mask] = 1.0
    cprob = cprob.filled(1.0)
    cprob_thr = cprob > min_prob

    return cmask, cprob_thr


def cloud_masks_for_time_window(
    cloudless_files: List[str], mask_files: List[str], min_prob: float
) -> Tuple[List[NDArray[Any]], List[NDArray[Any]]]:
    """
    Populate temporal window of cloud masks
    """

    cm1_arr: List[NDArray[Any]] = []
    cm2_arr: List[NDArray[Any]] = []
    for prob, mask in zip(cloudless_files, mask_files):
        cm1, cm2 = load_cloud_masks(prob, mask, min_prob)
        cm1_arr.append(cm1)
        cm2_arr.append(cm2)

    return cm1_arr, cm2_arr


# This script should take as input only the cloud masks.
def clean_clouds_for_tile(
    probs_files: List[str],
    mask_files: List[str],
    out_dir: str,
    T: int,
    min_prob: float,
    min_area: int,
    max_extra_cloud: int,
    dilation: int,
) -> List[str]:
    """
    USAGE: clean_clouds_for_tile(tile, start, end, save=True, T=10, min_prob=0.7,
    min_area=400, max_extra_cloud=5) reads in all the cloud masks in the directory
     and cleans it based on two rules.
    1. If in a time window of length 2*T+1 there are max_extra_cloud pixels that
       became cloudy in the s2cloudless mask and were not in the built in cloud
       mask, then we back off to the built in mask.
    2. We remove connected cloud components with less than min_area pixels and
       fill in holes in clouds with less than min_area pixels.
    Finally we take the union of these cloud pixels and the built in cloud mask and
    write it to a file named cloud_mask_merged.ny.
    """

    # Window of cloud masks to process
    window_start = 0
    window_end = 2 * T + 1

    selected_probs_files = probs_files[window_start:window_end]
    selected_mask_files = mask_files[window_start:window_end]

    cm1_arr, cm2_arr = cloud_masks_for_time_window(
        selected_probs_files, selected_mask_files, min_prob
    )

    N = len(probs_files)
    saved_masks: List[str] = []
    for i in range(N):
        if i + T > window_end and window_end < N:
            cm1_arr, cm2_arr = shift_arr(
                probs_files, mask_files, T, window_end, cm1_arr, cm2_arr, min_prob
            )
            gc.collect()
            window_start += 1
            window_end += 1
        idx = i - window_start
        new_mask = compute_mask_with_missing_clouds(
            cm1_arr, cm2_arr, idx, max_extra_cloud, min_area, dilation
        )
        saved_masks.append(fill_missing_pixels(mask_files[i], new_mask, out_dir))
        gc.collect()

    return saved_masks


def prepare_tile_data(
    items: TileData,
) -> Tuple[List[str], List[str]]:
    date_list: List[datetime.datetime] = []
    cloud_masks: List[str] = []
    cloud_probs: List[str] = []
    for mask, prob in items:
        cloud_probs.append(prob.raster_asset.local_path)
        cloud_masks.append(mask.raster_asset.local_path)
        date_list.append(mask.time_range[0])

    ind = np.argsort(cast(NDArray[Any], date_list))
    out_cloud_probs = [cloud_probs[i] for i in ind]
    out_cloud_masks = [cloud_masks[i] for i in ind]

    return out_cloud_probs, out_cloud_masks


class CallbackBuilder:
    def __init__(
        self,
        num_workers: int,
        window_size: int,
        cloud_prob_threshold: float,
        min_area: int,
        max_extra_cloud: int,
        dilation: int,
    ):
        self.num_workers = num_workers
        self.tmp_dir = TemporaryDirectory()
        self.window_size = window_size
        self.threshold = cloud_prob_threshold
        self.min_area = min_area
        self.max_extra_cloud = max_extra_cloud
        self.dilation = dilation

    def __call__(self):
        def compute_cloud_prob(
            masks: List[Sentinel2CloudMask],
            cloud_probabilities: List[Sentinel2CloudProbability],
        ) -> Dict[str, List[Sentinel2CloudMask]]:
            def process_single_tile(items: TileData) -> List[Sentinel2CloudMask]:
                items = sorted(items, key=lambda x: x[0].time_range[0])
                probs_files, mask_files = prepare_tile_data(items)

                out_files = clean_clouds_for_tile(
                    probs_files,
                    mask_files,
                    self.tmp_dir.name,
                    T=self.window_size,
                    min_prob=self.threshold,
                    min_area=self.min_area,
                    max_extra_cloud=self.max_extra_cloud,
                    dilation=self.dilation,
                )

                # Generating output items
                output_items: List[Sentinel2CloudMask] = []
                for (
                    mask,
                    _,
                ), new_asset in zip(items, out_files):
                    merged_cloud = AssetVibe(
                        reference=new_asset, type=mimetypes.types_map[".tif"], id=gen_guid()
                    )
                    new_mask = Sentinel2CloudMask.clone_from(mask, gen_guid(), [merged_cloud])
                    output_items.append(new_mask)

                return output_items

            # Grouping by tile_id
            tile_dict: Dict[str, TileData] = {}

            for mask in masks:
                tile_id = mask.tile_id
                prob = find_s2_product(mask.product_name, cloud_probabilities)
                if tile_id in tile_dict:
                    tile_dict[tile_id].append((mask, prob))
                else:
                    tile_dict[tile_id] = [(mask, prob)]

            results = [process_single_tile(tile) for tile in tile_dict.values()]
            results = cast(List[List[Sentinel2CloudMask]], results)

            consolidated_result = [result for result in chain(*results)]

            return {"merged_cloud_masks": consolidated_result}

        return compute_cloud_prob

    def __del__(self):
        self.tmp_dir.cleanup()
