# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Tuple, cast

import numpy as np
import rasterio
from geopandas import GeoDataFrame, GeoSeries, clip
from numpy._typing import NDArray
from rasterio.features import shapes, sieve
from rasterio.mask import mask
from shapely import geometry as shpg
from shapely.geometry import shape
from shapely.validation import make_valid
from sklearn.mixture import GaussianMixture

from vibe_core.data import DataVibe, gen_hash_id
from vibe_core.data.core_types import AssetVibe, gen_guid
from vibe_core.data.rasters import Raster
from vibe_lib.archive import create_flat_archive


class CallbackBuilder:
    def __init__(self, n_clusters: int, sieve_size: int):
        self.temp_dir = []
        self.n_clusters = n_clusters
        self.random_state = 45
        self.sieve_size = sieve_size

    def find_minimum_samples(self, raster: Raster, user_input: DataVibe) -> DataVibe:
        self.geometry_mask = GeoSeries([shape(user_input.geometry)], crs="EPSG:4326")
        # read input files
        with rasterio.open(raster.raster_asset.url, "r") as r_obj:
            p = self.geometry_mask.to_crs(r_obj.crs)[0]
            ar, tr = mask(r_obj, [p], crop=True, nodata=0)
            self.raster_crs = r_obj.crs
            self.tr = tr
            x = ar[0]

        asset_vibes = self.get_samples(x)
        return DataVibe(
            gen_hash_id("heatmap_nutrients", raster.geometry, raster.time_range),
            raster.time_range,
            raster.geometry,
            asset_vibes,
        )

    def get_samples(self, x: NDArray[Any]) -> List[AssetVibe]:
        model = self.train_model(x)
        geo_clusters, geo_locations = self.inference(model=model, input=x)
        asset_vibes = []
        asset_vibes.append(self.write_samples(geo_clusters, "geo_cluster_boundaries"))
        asset_vibes.append(self.write_samples(geo_locations, "geo_sample_locations"))
        return asset_vibes

    def train_model(
        self,
        input: NDArray[Any],
    ):
        x_ = input.reshape(-1, 1)
        x_ = np.nan_to_num(x_)
        model = GaussianMixture(
            n_components=self.n_clusters, covariance_type="full", random_state=self.random_state
        )
        model.fit(x_)
        return model

    def inference(
        self,
        model: GaussianMixture,
        input: NDArray[Any],
    ) -> Tuple[GeoDataFrame, GeoDataFrame]:
        # convert input to 2D array
        x_ = input.reshape(-1, 1)
        x_ = np.nan_to_num(x_)

        # predict clusters
        d = model.predict(x_)
        blocks = d.reshape(input.shape)

        # group small clusters
        blocks = sieve(blocks.astype(np.uint8), self.sieve_size)

        # converting clusters generated to a GeoDataFrame
        out = []
        for segment in range(self.n_clusters):
            polygons = (blocks == segment).astype(np.uint8)
            geoms = [
                make_valid(shpg.shape(s))
                for s, _ in shapes(polygons, mask=polygons, transform=self.tr)
            ]
            out.extend(geoms)

        if len(out) > 0:
            # get lat lon of center of each polygon, the center will be inside the polygon
            gdf = GeoDataFrame(data=out, columns=["geometry"], crs=self.raster_crs)  # type: ignore
            gdf = cast(GeoDataFrame, gdf.to_crs("EPSG:4326"))
            gdf = cast(GeoDataFrame, clip(gdf, self.geometry_mask, keep_geom_type=True))

            if gdf is not None and not gdf.empty:
                gdf_locations = gdf.geometry.representative_point()
                return (gdf, gdf_locations)

        raise RuntimeError("No samples found")

    def write_samples(self, geo_df: GeoDataFrame, geo_type: str) -> AssetVibe:
        temp_d = TemporaryDirectory()
        output_path = os.path.join(temp_d.name, f"minimum_samples_location_{geo_df.shape[0]}.shp")
        geo_df.to_file(output_path)
        self.temp_dir.append(temp_d)

        # Create zip archive containing all output
        archive_path = create_flat_archive(temp_d.name, geo_type)
        return AssetVibe(reference=archive_path, type="application/zip", id=gen_guid())

    def __call__(self):
        def find_minimum_samples_init(raster: Raster, user_input: DataVibe) -> Dict[str, DataVibe]:
            out_vibe = self.find_minimum_samples(raster, user_input)
            return {"locations": out_vibe}

        return find_minimum_samples_init

    def __del__(self):
        for temp_d in self.temp_dir:
            temp_d.cleanup()
