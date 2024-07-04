import os
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Tuple, cast

import geopandas as gpd
import numpy as np
import rasterio
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from rasterio import Affine, features
from rasterio.crs import CRS
from rasterio.enums import MergeAlg
from rasterio.mask import mask
from shapely.geometry import Polygon, shape

from vibe_core.data import DataVibe, gen_hash_id
from vibe_core.data.core_types import AssetVibe, GeometryCollection
from vibe_core.data.rasters import Raster
from vibe_lib.geometry import create_mesh_grid
from vibe_lib.heatmap_neighbor import (
    run_cluster_overlap,
    run_kriging_model,
    run_nearest_neighbor,
)
from vibe_lib.raster import write_to_raster
from vibe_lib.shapefile import write_shapefile


class CallbackBuilder:
    def __init__(
        self,
        attribute_name: str,
        simplify: str,
        tolerance: float,
        algorithm: str,
        resolution: int,
        bins: int,
    ):
        self.temp_shapefile_dir = TemporaryDirectory()
        self.temp_tiff_dir = TemporaryDirectory()
        self.attribute_name = attribute_name
        self.simplify = simplify
        self.tolerance = tolerance
        self.algorithm = algorithm
        self.resolution = resolution
        self.bins = bins

    def create_heatmap(
        self,
        raster: Raster,
        samples: GeometryCollection,
        samples_boundary: GeometryCollection,
    ) -> DataVibe:
        with rasterio.open(raster.assets[0].path_or_url) as src:
            self.raster_crs = src.crs
        # Get reduced samples
        samples_df = gpd.read_file(samples.assets[0].url)
        samples_df = cast(GeoDataFrame, samples_df[["geometry", self.attribute_name]])
        # Get reduced sample boundaries (clusters)
        samples_boundary_df = cast(
            GeoDataFrame,
            gpd.read_file(samples_boundary.assets[0].url),
        )
        samples_boundary_df = cast(GeoDataFrame, samples_boundary_df[["geometry"]])
        boundary = cast(Polygon, shape(samples.geometry))
        # Get mesh grid geo locations for farm boundary
        geo_locations = create_mesh_grid(boundary, self.resolution, self.raster_crs)
        # Run nutrient algorithm and create heatmap
        farm_boundary_df = GeoDataFrame(geometry=[boundary], crs=4326)  # type: ignore
        nutrients_df = self.run_algorithm(samples_df, samples_boundary_df, geo_locations)
        assetVibe = self.generate_samples_heat_map(
            nutrients_df, raster.assets[0].url, farm_boundary_df
        )
        return DataVibe(
            gen_hash_id(
                f"heatmap_nutrients_{self.attribute_name}",
                raster.geometry,
                raster.time_range,
            ),
            raster.time_range,
            raster.geometry,
            assetVibe,
        )

    def run_algorithm(
        self,
        samples_df: GeoDataFrame,
        samples_boundary_df: GeoDataFrame,
        geo_locations: GeoDataFrame,
    ) -> GeoDataFrame:
        if self.algorithm == "cluster overlap":
            return run_cluster_overlap(
                attribute_name=self.attribute_name,
                reduced_samples=samples_df,
                minimum_sample_polygons=samples_boundary_df,
                geo_locations=geo_locations,
            )
        elif self.algorithm == "nearest neighbor":
            return run_nearest_neighbor(
                attribute_name=self.attribute_name,
                reduced_samples=samples_df,
                geo_locations=geo_locations,
            )
        elif self.algorithm == "kriging neighbor":
            return run_kriging_model(
                attribute_name=self.attribute_name,
                reduced_samples=samples_df,
                geo_locations=geo_locations,
            )
        else:
            raise RuntimeError(f"Unknown algorithm: {self.algorithm}")

    def rasterize_heatmap(
        self,
        shapes: Tuple[Any],
        ar: NDArray[Any],
        tr: Affine,
        raster_mask: NDArray[Any],
    ):
        # Rasterize the nutrient boundaries
        raster_output = features.rasterize(
            shapes=shapes,
            out_shape=ar[0].shape,
            transform=tr,
            all_touched=True,
            fill=-1,  # background value
            merge_alg=MergeAlg.replace,
            dtype=rasterio.float32,
        )
        raster_output[ar.sum(axis=0) == 0] = 0
        out_path = os.path.join(self.temp_tiff_dir.name, "raster_output.tif")
        raster_output = self.group_to_nearest(raster_output, raster_mask)
        out = raster_output * raster_mask.astype(np.uint16)
        asset_vibe = write_to_raster(out, tr, out_path, self.raster_crs)
        return out, asset_vibe

    def group_to_nearest(self, raster_output: NDArray[Any], raster_mask: NDArray[Any]):
        raster_output[raster_output <= 0] = raster_output[raster_output > 0].mean()

        intervals = np.histogram(raster_output[raster_mask], bins=self.bins)[1]
        intervals[0] = -1
        index = np.searchsorted(intervals, raster_output) - 1
        out_grouped_raster = np.zeros(raster_output.shape)

        for i in range(len(intervals)):
            out_grouped_raster[index == i] = raster_output[index == i].mean()

        return out_grouped_raster

    def generate_samples_heat_map(
        self,
        nutrients_df: GeoDataFrame,
        src_image_path: str,
        farm_boundary_df: GeoDataFrame,
    ) -> List[AssetVibe]:
        with rasterio.open(src_image_path, "r") as o_raster:
            # change spatial projection of inputs matching to sentinel image
            nutrients_df = cast(GeoDataFrame, nutrients_df.to_crs(o_raster.crs))
            farm_boundary_df = cast(GeoDataFrame, farm_boundary_df.to_crs(o_raster.crs))
            # create mask for farm boundary
            if not farm_boundary_df.empty:
                boundary = farm_boundary_df[:1].geometry[0]  # type: ignore
                ar, tr = mask(o_raster, [boundary], crop=True, nodata=0)
                mask1 = (ar != 0).any(axis=0)
                shapes = []
                # collect shapes for rasterization
                nutrients_df["geometry"] = nutrients_df.buffer(self.resolution, cap_style=3)
                nutrients_df["shapes"] = nutrients_df.apply(
                    lambda row: (row.geometry, row[self.attribute_name]), axis=1
                )
                if not nutrients_df.empty:
                    shapes = tuple(nutrients_df["shapes"].values)  # type: ignore
                    # rasterize shapes
                    out, raster_vibe = self.rasterize_heatmap(shapes, ar, tr, mask1)
                    shape_vibe = self.export_to_shapeFile(out, o_raster.crs, tr, mask1)

                    vibes = [shape_vibe, raster_vibe]
                    return vibes

                raise RuntimeError("Model didn't identified nutrient locations")

        raise RuntimeError("No farm boundary found")

    def export_to_shapeFile(
        self,
        data: NDArray[Any],
        crs: CRS,
        tr: Affine,
        mask1: NDArray[Any],
    ):
        asset = write_shapefile(
            data,
            crs,
            tr,
            mask1,
            self.temp_shapefile_dir.name,
            self.simplify,
            self.tolerance,
            "cluster",
        )
        return asset

    def __call__(self):
        def create_heatmap_init(
            raster: Raster,
            samples: GeometryCollection,
            samples_boundary: GeometryCollection,
        ) -> Dict[str, DataVibe]:
            out_vibe = self.create_heatmap(raster, samples, samples_boundary)
            return {"result": out_vibe}

        return create_heatmap_init

    def __del__(self):
        self.temp_shapefile_dir.cleanup()
        self.temp_tiff_dir.cleanup()
