import os
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Tuple

import numpy as np
import rasterio
from numpy.typing import NDArray
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, disk

from vibe_core.data import AssetVibe, Sentinel2CloudMask, Sentinel2CloudProbability, gen_guid
from vibe_lib.raster import INT_COMPRESSION_KWARGS

TileData = List[Tuple[Sentinel2CloudMask, Sentinel2CloudProbability]]


def kill_labels(clabel: NDArray[Any], min_area: int) -> List[Any]:
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


def remove_small_components(cmask: NDArray[Any], min_area: int):
    """
    USAGE: new_mask = remove_small_components(cmask, min_area=400)
    First removes small connected cloud components, then fill in small
    connected holes in clouds to make for a smoother cloud mask.
    """
    # Get cloud components
    cloud_comp = label(cmask)
    # Mark small components
    kill_list = kill_labels(cloud_comp, min_area)  # type: ignore
    small_clouds = np.isin(cloud_comp, kill_list)  # type: ignore
    # Remove them
    cmask[small_clouds] = False

    # Do the same for small components of clear sky
    holes_comp = label(~cmask)
    kill_list = kill_labels(holes_comp, min_area)  # type: ignore
    small_cloud_holes = np.isin(holes_comp, kill_list)  # type: ignore
    cmask[small_cloud_holes] = True

    return cmask


def merge_masks(
    product_mask: Sentinel2CloudMask,
    cloud_probability: Sentinel2CloudProbability,
    shadow_probability: Sentinel2CloudProbability,
    cloud_threshold: float,
    shadow_threshold: float,
    closing_size: int,
    min_area: int,
) -> Tuple[NDArray[np.uint8], Dict[str, Any]]:
    with rasterio.open(cloud_probability.raster_asset.url) as src:
        meta = src.meta
        cloud_p = src.read(1) > cloud_threshold
    with rasterio.open(shadow_probability.raster_asset.url) as src:
        shadow_p = src.read(1) > shadow_threshold
    with rasterio.open(product_mask.raster_asset.url) as src:
        cloud_m = src.read(1).astype(bool)
    # Do the most conservative thing we can, and pick cloud if any model classifies as cloud/shadow
    merged = cloud_p | shadow_p | cloud_m
    # Remove small holes and keep a buffer
    merged = binary_dilation(merged, disk(closing_size)).astype(np.uint8)
    if min_area > 0:
        merged = remove_small_components(merged, min_area)
    meta["dtype"] = "uint8"
    return merged[None], meta


class CallbackBuilder:
    def __init__(
        self,
        cloud_prob_threshold: float,
        shadow_prob_threshold: float,
        closing_size: int,
        min_area: int,
    ):
        self.tmp_dir = TemporaryDirectory()
        self.cloud_threshold = cloud_prob_threshold
        self.shadow_threshold = shadow_prob_threshold
        self.closing_size = closing_size
        self.min_area = min_area

    def __call__(self):
        def compute_cloud_prob(
            product_mask: Sentinel2CloudMask,
            cloud_probability: Sentinel2CloudProbability,
            shadow_probability: Sentinel2CloudProbability,
        ) -> Dict[str, Sentinel2CloudMask]:
            merged, meta = merge_masks(
                product_mask,
                cloud_probability,
                shadow_probability,
                self.cloud_threshold,
                self.shadow_threshold,
                self.closing_size,
                self.min_area,
            )
            id = gen_guid()
            out_path = os.path.join(self.tmp_dir.name, f"{id}.tif")
            with rasterio.open(out_path, "w", **meta, **INT_COMPRESSION_KWARGS) as dst:
                dst.write(merged)
            return {
                "merged_cloud_mask": Sentinel2CloudMask.clone_from(
                    cloud_probability,
                    id=gen_guid(),
                    bands={"cloud": 0},
                    categories=["Clear", "Cloud"],
                    assets=[AssetVibe(id=id, type="image/tiff", reference=out_path)],
                )
            }

        return compute_cloud_prob

    def __del__(self):
        self.tmp_dir.cleanup()
