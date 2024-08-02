# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from datetime import datetime
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union, cast

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

from vibe_core.data import CategoricalRaster, Raster, TimeSeries, gen_guid
from vibe_lib.gaussian_mixture import (
    cluster_data,
    mixture_log_likelihood,
    train_mixture_with_component_search,
)
from vibe_lib.raster import (
    get_categorical_cmap,
    get_cmap,
    json_to_asset,
    load_raster,
    save_raster_to_asset,
)
from vibe_lib.timeseries import save_timeseries_to_asset


def compute_outliers(
    curves: NDArray[Any], preprocessing: StandardScaler, thr: float, max_components: int
) -> Tuple[NDArray[np.int32], NDArray[np.float32], NDArray[np.int32], NDArray[Any]]:
    x = preprocessing.fit_transform(curves)  # Preprocess data

    mix = train_mixture_with_component_search(x, max_components=max_components)
    labels = cluster_data(x, mix)  # Assign labels
    labels = labels.astype(np.int32)
    # TODO: How to compute the threshold? Use fixed for now
    likelihood = mixture_log_likelihood(x, mix)
    outliers = likelihood < thr
    likelihood = likelihood.astype(np.float32)
    outliers = cast(NDArray[np.int32], outliers.astype(np.int32))
    # Recover means in the NDVI space
    mix_means = cast(NDArray[Any], preprocessing.inverse_transform(mix.means_))

    return labels, likelihood, outliers, mix_means


def save_mixture_means(
    mix_means: NDArray[Any],
    output_dir: str,
    geom: Dict[str, Any],
    date_list: Sequence[datetime],
) -> TimeSeries:
    # Save timeseries output
    df = pd.DataFrame(date_list, columns=["date"])
    for i, m in enumerate(mix_means):
        df[f"component{i}"] = m

    df.set_index("date", drop=True, inplace=True)

    return TimeSeries(
        id=gen_guid(),
        geometry=geom,
        time_range=(date_list[0], date_list[-1]),
        assets=[save_timeseries_to_asset(df, output_dir)],
    )


def unpack_data(rasters: Sequence[Raster]) -> Tuple[NDArray[np.float32], xr.DataArray]:
    # Sort rasters according to date
    rasters = sorted(rasters, key=lambda x: x.time_range[0])
    # Load one raster to get metadata we need
    band_data = load_raster(rasters[0], use_geometry=True)

    # Get band data and compress masked data into a stack of timeseries
    curves = (
        np.stack(
            [band_data.to_masked_array().compressed()]
            + [
                load_raster(r, use_geometry=True).to_masked_array().compressed()
                for r in rasters[1:]
            ]
        )
        .astype(np.float32)
        .T
    )
    return curves, band_data


def pack_rasters(
    labels: NDArray[np.int32],
    likelihood: NDArray[np.float32],
    outliers: NDArray[np.int32],
    geom: Dict[str, Any],
    date_list: Sequence[datetime],
    threshold: float,
    output_dir: str,
    reshape_fun: Callable[[NDArray[Any]], xr.DataArray],
):
    output: Dict[str, List[Any]] = {}
    time_range = (date_list[0], date_list[-1])

    # Save likelihood raster
    vis_dict = {
        "bands": [0],
        "colormap": get_cmap("viridis"),
        "range": (max(threshold, float(likelihood.min())), float(likelihood.max())),
    }
    heatmap = Raster(
        id=gen_guid(),
        geometry=geom,
        time_range=time_range,
        assets=[
            save_raster_to_asset(reshape_fun(likelihood), output_dir),
            json_to_asset(vis_dict, output_dir),
        ],
        bands={"likelihood": 0},
    )
    output["heatmap"] = [heatmap]

    # Save categorical rasters
    classes = np.unique(labels)
    num_classes = classes.shape[0]
    vis_dict = {
        "bands": [0],
        "colormap": get_categorical_cmap("tab10", num_classes),
        "range": (0, num_classes - 1),
    }
    output["segmentation"] = [
        CategoricalRaster(
            id=gen_guid(),
            geometry=geom,
            time_range=time_range,
            assets=[
                save_raster_to_asset(reshape_fun(labels), output_dir),
                json_to_asset(vis_dict, output_dir),
            ],
            bands={"labels": 0},
            categories=[f"component{i}" for i in range(num_classes)],
        )
    ]
    vis_dict = {
        "bands": [0],
        "colormap": get_categorical_cmap("tab10", 2),
        "range": (0, 1),
    }
    output["outliers"] = [
        CategoricalRaster(
            id=gen_guid(),
            geometry=geom,
            time_range=time_range,
            assets=[
                save_raster_to_asset(reshape_fun(outliers), output_dir),
                json_to_asset(vis_dict, output_dir),
            ],
            bands={"labels": 0},
            categories=["normal", "outlier"],
        )
    ]
    return output


def pack_data(
    labels: NDArray[np.int32],
    likelihood: NDArray[np.float32],
    outliers: NDArray[np.int32],
    mix_means: NDArray[np.float32],
    geom: Dict[str, Any],
    date_list: Sequence[datetime],
    threshold: float,
    output_dir: str,
    reshape_fun: Callable[[NDArray[Any]], xr.DataArray],
):
    output = pack_rasters(
        labels, likelihood, outliers, geom, date_list, threshold, output_dir, reshape_fun
    )
    output["mixture_means"] = [save_mixture_means(mix_means, output_dir, geom, date_list)]
    return output


class CallbackBuilder:
    def __init__(self, threshold: float):
        self.tmp_dir = TemporaryDirectory()
        self.threshold = threshold
        # TODO: Customize preprocessing
        self.preprocessing = StandardScaler()

    def __call__(self):
        def outliers_callback(rasters: List[Raster]) -> Dict[str, List[Union[Raster, TimeSeries]]]:
            curves, band_data = unpack_data(rasters)

            # Get metadata
            geom = rasters[0].geometry
            date_list = [r.time_range[0] for r in rasters]

            # Helper function to obtain masked array from 1D array
            def reshape_to_geom(values: NDArray[Any]) -> xr.DataArray:
                data = np.ma.masked_all(band_data.shape, values.dtype)
                data.mask = band_data.isnull()
                data.data[~data.mask] = values
                data.fill_value = band_data.rio.encoded_nodata  # Unused value
                data = band_data.copy(data=data.filled())
                data.rio.update_encoding({"dtype": str(values.dtype)}, inplace=True)
                return data

            # Gaussian mixtures modeling
            labels, likelihood, outliers, mix_means = compute_outliers(
                curves,
                self.preprocessing,
                self.threshold,
                max_components=1,  # Assume only one component
            )

            # Pack data
            output = pack_data(
                labels,
                likelihood,
                outliers,
                mix_means,
                geom,
                date_list,
                self.threshold,
                self.tmp_dir.name,
                reshape_to_geom,
            )

            return output

        return outliers_callback

    def __del__(self):
        self.tmp_dir.cleanup()
