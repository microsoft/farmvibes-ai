import os
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple, cast

import geopandas as gpd
import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.features import shapes
from rasterio.mask import mask
from rasterio.transform import Affine
from scipy.ndimage import convolve
from shapely import geometry as shpg
from shapely import ops as shpo
from shapely.geometry.base import BaseGeometry
from skimage.measure import label, regionprops
from skimage.transform import rotate

from vibe_core.data import CategoricalRaster, DataVibe, Raster
from vibe_core.data.core_types import AssetVibe, GeometryCollection, gen_guid
from vibe_lib.raster import MaskedArrayType


def read_raster(filepath: str, geometry: BaseGeometry) -> Tuple[MaskedArrayType, Affine]:
    with rasterio.open(filepath) as src:
        return mask(src, [geometry], crop=True, filled=False)


def get_kernels(kernel_size: Tuple[int, int], n_kernels: int) -> List[NDArray[Any]]:
    y, x = kernel_size
    k_max = max(kernel_size)

    base_kernel = np.zeros((k_max, k_max))
    off_y = (k_max - y) // 2
    off_x = (k_max - x) // 2
    base_kernel[off_y : k_max - off_y, off_x : k_max - off_x] = 1

    angles = np.linspace(0, 180, n_kernels + 1)[:-1]
    return [rotate(base_kernel, a, order=0) for a in angles]


def can_park(mask: NDArray[Any], car_size: Tuple[int, int], n_kernels: int, thr: float):
    mask = mask.astype(np.float32)
    kernels = get_kernels(car_size, n_kernels)
    for kernel in kernels:
        ks = kernel.sum()
        if np.any(convolve(mask, kernel, mode="constant") / ks >= thr):
            return True
    return False


class DrivewayDetector:
    def __init__(
        self,
        img_filepath: str,
        pred_filepath: str,
        road_df: gpd.GeoDataFrame,
        min_region_area: float,
        ndvi_thr: float,
        car_size: Tuple[int, int],
        num_kernels: int,
        car_thr: float,
    ) -> None:
        self.img_filepath = img_filepath
        self.pred_filepath = pred_filepath

        with rasterio.open(img_filepath) as src:
            pixel_area = src.res[0] * src.res[1]
            self.raster_geom = shpg.box(*src.bounds)
            self.raster_crs = src.crs
        self.min_area = min_region_area / pixel_area

        self.road_df = cast(gpd.GeoDataFrame, road_df.to_crs(self.raster_crs))

        self.ndvi_thr = ndvi_thr
        self.car_size = car_size
        self.num_kernels = num_kernels
        self.car_thr = car_thr

    def _get_region_near_road(
        self, pred_mask: MaskedArrayType, tr: Affine
    ) -> Optional[NDArray[np.bool_]]:
        pred_labels = label(pred_mask.filled(0))
        pred_regions = sorted(
            [p for p in regionprops(pred_labels) if p.area > self.min_area],
            key=lambda x: self.road_df.geometry.distance(shpg.Point(tr * x.centroid[::-1])).min(),
        )
        if not pred_regions:
            # No region that is large enough
            return None

        region = pred_regions[0]  # Get region closest to the road
        mask = pred_labels == region.label
        return mask

    def detect(self, geom: BaseGeometry) -> Optional[BaseGeometry]:
        bands, tr = read_raster(self.img_filepath, geom)
        pred_mask = read_raster(self.pred_filepath, geom)[0][0] > 0

        red, nir = bands[[0, 3]]
        ndvi = (nir - red) / (nir + red)
        not_green = (ndvi < self.ndvi_thr).filled(0)

        region_mask = self._get_region_near_road(pred_mask, tr)
        if region_mask is None:
            # Not region large enough
            return None

        region_mask = not_green * region_mask
        region_labels = label(region_mask)

        # Find regions where we could fit a car
        dw_regions = [
            p
            for p in regionprops(region_labels)
            if can_park(p.image, self.car_size, self.num_kernels, self.car_thr)
        ]
        if not dw_regions:
            # No region that can fit a car
            return None
        # Estimate total region of the driveway
        dw_mask = np.sum([region_labels == p.label for p in dw_regions], axis=0).astype(bool)
        dw_geom = shpo.unary_union(
            [
                shpg.shape(s).convex_hull
                for s, _ in shapes(
                    dw_mask.astype(np.uint8), mask=dw_mask, connectivity=8, transform=tr
                )
            ]
        )
        return dw_geom


class CallbackBuilder:
    def __init__(
        self,
        min_region_area: float,
        ndvi_thr: float,
        car_size: Tuple[int, int],
        num_kernels: int,
        car_thr: float,
    ):
        self.min_region_area = min_region_area
        self.ndvi_thr = ndvi_thr
        self.car_size = car_size
        self.num_kernels = num_kernels
        self.car_thr = car_thr
        self.tmp_dir = TemporaryDirectory()

    def __call__(self):
        def callback(
            input_raster: Raster,
            segmentation_raster: CategoricalRaster,
            property_boundaries: GeometryCollection,
            roads: GeometryCollection,
        ) -> Dict[str, DataVibe]:
            road_df = cast(gpd.GeoDataFrame, gpd.read_file(roads.assets[0].url))
            detector = DrivewayDetector(
                input_raster.raster_asset.url,
                segmentation_raster.raster_asset.url,
                road_df=road_df,
                min_region_area=self.min_region_area,
                ndvi_thr=self.ndvi_thr,
                car_size=self.car_size,
                num_kernels=self.num_kernels,
                car_thr=self.car_thr,
            )
            properties_df = cast(
                gpd.GeoDataFrame,
                gpd.read_file(property_boundaries.assets[0].url).to_crs(detector.raster_crs),  # type: ignore
            )
            properties_df = properties_df[properties_df.intersects(detector.raster_geom)]
            driveway = []
            dw_geoms = []
            assert properties_df is not None, "There are no intersections with properties"
            for _, row in properties_df.iterrows():
                geom = row.geometry.buffer(0)
                dw_geom = detector.detect(geom)
                is_dw = dw_geom is not None
                driveway.append(is_dw)
                if is_dw:
                    dw_geoms.append(dw_geom)  # type: ignore
            full_df = properties_df[driveway].copy()  # type: ignore
            dw_df = full_df.copy()
            dw_df["geometry"] = dw_geoms  # type: ignore
            out = {}
            for out_name, df in zip(("properties_with_driveways", "driveways"), (full_df, dw_df)):
                asset_id = gen_guid()
                filepath = os.path.join(self.tmp_dir.name, f"{asset_id}.geojson")
                df.to_file(filepath, driver="GeoJSON")  # type: ignore
                asset = AssetVibe(reference=filepath, type="application/geo+json", id=asset_id)
                out[out_name] = DataVibe.clone_from(input_raster, id=gen_guid(), assets=[asset])

            return out

        return callback

    def __del__(self):
        self.tmp_dir.cleanup()
