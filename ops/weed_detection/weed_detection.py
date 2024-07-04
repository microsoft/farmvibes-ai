import os
from dataclasses import dataclass
from enum import auto
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Tuple, Union

import geopandas as gpd
import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.crs import CRS
from rasterio.features import geometry_mask, shapes, sieve
from rasterio.mask import mask
from rasterio.transform import Affine
from shapely import geometry as shpg
from sklearn.mixture import GaussianMixture
from strenum import StrEnum

from vibe_core.data import DataVibe
from vibe_core.data.core_types import AssetVibe, gen_guid
from vibe_core.data.rasters import Raster
from vibe_lib.archive import create_flat_archive


class SimplifyBy(StrEnum):
    simplify = auto()
    convex = auto()
    none = auto()


@dataclass
class OpenedRaster:
    """Load a raster for training and prediction

    Attributes:
        pixels: 1D array of selected data points
        shape: shape of the input raster
        alpha_mask: boolean values indicating which pixels were selected from the input raster
        transform: affine transform of the input raster
        crs: coordinate reference system of the input raster
    """

    def __init__(
        self,
        raster: Raster,
        buffer: int,
        no_data: Union[int, None],
        alpha_index: int,
        bands: List[int],
    ):
        with rasterio.open(raster.raster_asset.url) as src:
            projected_geo = (
                gpd.GeoSeries(shpg.shape(raster.geometry), crs="epsg:4326").to_crs(src.crs).iloc[0]
            )

            if no_data is None:
                no_data = src.nodata
            ar, self.tr = mask(src, [projected_geo], crop=True, nodata=no_data)
            self.input_crs = src.crs

        self.buffer_mask = geometry_mask(
            [projected_geo.buffer(buffer)], ar.shape[1:], self.tr, invert=True
        )

        # Create an alpha mask
        if alpha_index >= 0:
            self._alpha_mask = ar[alpha_index].astype(bool)
        else:  # no alpha band
            self._alpha_mask = np.ones(ar.shape[1:], dtype=bool)

        if not bands:
            bands = [i for i in range(ar.shape[0]) if i != alpha_index]
        self.pixels = ar[bands]

        self.input_shape = ar.shape

    @property
    def shape(self) -> Tuple[int]:
        return self.input_shape

    @property
    def crs(self) -> CRS:
        return self.input_crs

    @property
    def transform(self) -> Affine:
        return self.tr

    @property
    def training_data(self) -> NDArray[Any]:
        mask = self.buffer_mask & self.alpha_mask
        return self.pixels[:, mask]

    @property
    def prediction_data(self) -> NDArray[Any]:
        return self.pixels[:, self.alpha_mask]

    @property
    def alpha_mask(self) -> NDArray[Any]:
        return self._alpha_mask


def train_model(open_raster: OpenedRaster, samples: int, clusters: int) -> GaussianMixture:
    training_data = open_raster.training_data
    idx = np.random.choice(training_data.shape[1], samples)
    xy = training_data[:, idx].T

    gmm = GaussianMixture(n_components=clusters, covariance_type="full")
    gmm.fit(xy)

    return gmm


def predict(
    open_raster: OpenedRaster,
    sieve_size: int,
    clusters: int,
    simplify: SimplifyBy,
    tolerance: float,
    model: GaussianMixture,
    output_dir: str,
) -> AssetVibe:
    prediction_data = open_raster.prediction_data
    classes = model.predict(prediction_data.reshape(prediction_data.shape[0], -1).T)
    result = np.zeros(open_raster.shape[1:], dtype=np.uint8)
    result[open_raster.alpha_mask] = classes
    result = sieve(result, sieve_size)

    file_num = 0
    for segment in range(clusters):
        cluster = (result == segment).astype(np.uint8)

        df_shapes = gpd.GeoSeries(
            [shpg.shape(s) for s, _ in shapes(cluster, mask=cluster, transform=open_raster.tr)],
            crs=open_raster.crs,
        )  # type: ignore

        if df_shapes.empty:
            # Model could not converge with all requested clusters
            continue

        cluster_path = os.path.join(output_dir, f"cluster{file_num}")
        file_num += 1

        if simplify == SimplifyBy.simplify:
            df_shapes.simplify(tolerance).to_file(cluster_path)
        elif simplify == SimplifyBy.convex:
            df_shapes.convex_hull.to_file(cluster_path)
        elif simplify == SimplifyBy.none:
            df_shapes.to_file(cluster_path)  # type: ignore

    # Create zip archive containing all output
    archive_path = create_flat_archive(output_dir, "result")
    return AssetVibe(reference=archive_path, type="application/zip", id=gen_guid())


class CallbackBuilder:
    def __init__(
        self,
        buffer: int,
        no_data: Union[int, None],
        clusters: int,
        sieve_size: int,
        simplify: str,
        tolerance: float,
        samples: int,
        bands: List[int],
        alpha_index: int,
    ):
        self.temp_dir = TemporaryDirectory()
        self.buffer = buffer
        self.no_data = no_data
        self.clusters = clusters
        self.sieve_size = sieve_size
        self.simplify = SimplifyBy(simplify.lower())
        self.tolerance = tolerance
        self.samples = samples
        self.bands = bands
        self.alpha_index = alpha_index

    def __call__(self):
        def detect_weeds(
            raster: Raster,
        ) -> Dict[str, DataVibe]:
            open_raster = OpenedRaster(
                raster=raster,
                buffer=self.buffer,
                no_data=self.no_data,
                alpha_index=self.alpha_index,
                bands=self.bands,
            )

            model = train_model(
                open_raster=open_raster,
                samples=self.samples,
                clusters=self.clusters,
            )

            prediction = predict(
                open_raster=open_raster,
                sieve_size=self.sieve_size,
                clusters=self.clusters,
                simplify=self.simplify,
                tolerance=self.tolerance,
                model=model,
                output_dir=self.temp_dir.name,
            )

            result = DataVibe(
                id=gen_guid(),
                time_range=raster.time_range,
                geometry=raster.geometry,
                assets=[prediction],
            )
            return {"result": result}

        return detect_weeds

    def __del__(self):
        self.temp_dir.cleanup()
