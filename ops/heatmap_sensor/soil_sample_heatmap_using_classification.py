# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, cast

import geopandas as gpd
import numpy as np
import rasterio
from geopandas.geodataframe import GeoDataFrame, GeoSeries
from pyproj.crs import crs
from rasterio.features import sieve
from rasterio.io import DatasetReader
from rasterio.mask import mask
from shapely.geometry import shape
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from vibe_core.data import DataVibe, gen_hash_id
from vibe_core.data.core_types import GeometryCollection
from vibe_core.data.rasters import Raster
from vibe_lib.shapefile import write_shapefile


class CallbackBuilder:
    def __init__(
        self,
        attribute_name: str,
        buffer: int,
        bins: int,
        simplify: str,
        tolerance: float,
        data_scale: bool,
        max_depth: int,
        n_estimators: int,
        random_state: int,
    ):
        self.temp_dir = TemporaryDirectory()
        self.attribute_name = attribute_name
        self.buffer = buffer
        self.bins = bins
        self.simplify = simplify
        self.tolerance = tolerance
        self.data_scale = data_scale
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.random_state = random_state

    def create_heatmap(self, raster: Raster, samples: GeometryCollection) -> DataVibe:
        # Read and filter GeoDataFrame using input attribute name
        samples_df = cast(
            gpd.GeoDataFrame,
            gpd.read_file(samples.assets[0].url),
        )
        samples_df = cast(GeoDataFrame, samples_df[["geometry", self.attribute_name]])
        assert samples_df.crs, "samples dataframe has no CRS"

        # Train Model
        model, le, scaler = self.train_classifier(
            raster_path=raster.raster_asset.url,
            samples=cast(GeoDataFrame, samples_df),
        )

        # Predict
        assetVibe = self.predict_classifier(
            model=model,
            raster_path=raster.raster_asset.url,
            label_encoder=le,
            scaler=scaler,
            farm_boundary=samples.geometry,
            samples_crs=samples_df.crs,
        )

        return DataVibe(
            gen_hash_id("heatmap_nutrients", raster.geometry, raster.time_range),
            raster.time_range,
            raster.geometry,
            [assetVibe],
        )

    def predict_classifier(
        self,
        model: RandomForestClassifier,
        raster_path: str,
        label_encoder: LabelEncoder,
        scaler: Optional[StandardScaler],
        farm_boundary: Dict[str, Any],
        samples_crs: crs.CRS,
    ):
        # Read input raster and clip it to farm boundary
        with rasterio.open(raster_path) as src:
            p = GeoSeries([shape(farm_boundary)], crs=samples_crs).to_crs(src.crs)[0]
            index_out, tr = mask(src, [p], crop=True, nodata=0)
            crs = src.crs
            mask1 = (index_out != 0).any(axis=0)
            index_out = index_out[0]

        index_out[np.isnan(index_out)] = 0
        index_out[index_out == np.inf] = 0
        s = index_out.reshape(-1, 1)

        # scale indexes
        if scaler is not None:
            s = scaler.transform(s)

        # predict and perform inverse transform
        ck = model.predict(s)
        ck = sieve(ck.reshape(index_out.shape).astype(np.int32), self.bins, mask=mask1)
        ck = label_encoder.inverse_transform(ck.reshape(-1))
        out_ = ck.reshape(index_out.shape)  # type: ignore
        out = out_ * mask1.astype(np.int32)

        asset = write_shapefile(
            out,
            crs,
            tr,
            mask1,
            self.temp_dir.name,
            self.simplify,
            self.tolerance,
            "cluster",
        )
        return asset

    def get_train_data(self, samples: GeoDataFrame, raster: DatasetReader):
        x_, y_, height = [], [], -1
        for _, row in samples.iterrows():
            # clip raster to field boundary
            x, _ = mask(raster, [row["geometry"]], crop=True, nodata=0, filled=True)
            x = x[0]

            x[np.isnan(x)] = 0
            height = x.shape
            x_.extend(x.reshape(-1, 1))

            y_.extend((np.ones(height) * row[self.attribute_name]).reshape(-1, 1))  # type: ignore

        # Scale the data
        scaler = None
        x = x_
        if self.data_scale:
            scaler = StandardScaler()
            x = scaler.fit_transform(x_)  # type: ignore

        # assign data to bins
        intervals = np.histogram(y_, bins=self.bins)[1]
        intervals[0] = -1
        index = np.searchsorted(intervals, y_) - 1
        y = np.zeros(len(y_)).reshape(index.shape)

        for i in range(len(intervals)):
            y[index == i] = np.array(y_)[index == i].mean()

        y = y.reshape(-1)

        # encode labels
        le = LabelEncoder()
        y = le.fit_transform(y)

        return x, y, le, scaler

    def train_classifier(
        self,
        raster_path: str,
        samples: GeoDataFrame,
    ):
        # read input files
        raster_obj = rasterio.open(raster_path, "r")

        # create grid from sample distance
        samples = cast(GeoDataFrame, samples.to_crs(raster_obj.crs))  # type: ignore
        samples["geometry"] = cast(GeoSeries, samples["geometry"]).buffer(self.buffer, cap_style=3)

        x, y, le, scaler = self.get_train_data(samples=samples, raster=raster_obj)

        # train model
        x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2)
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        model.fit(x_train, y_train)
        return model, le, scaler

    def __call__(self):
        def create_heatmap_init(raster: Raster, samples: GeometryCollection) -> Dict[str, DataVibe]:
            out_vibe = self.create_heatmap(raster, samples)
            return {"result": out_vibe}

        return create_heatmap_init

    def __del__(self):
        self.temp_dir.cleanup()
