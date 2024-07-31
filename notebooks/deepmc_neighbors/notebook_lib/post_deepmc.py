import os
import pickle
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from notebook_lib.base_deepmc import inference_deepmc, inference_deepmc_post
from notebook_lib.data_utils import get_file, preprocess_transform
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler

from vibe_notebook.deepmc.utils import get_csv_data, transform_to_array_3D


def write_embeddings_input(
    embeddings_input_path: str,
    data_scaler: StandardScaler,
    mix_yhat: NDArray[Any],
    mix_train_yhat: NDArray[Any],
    mix_yc: NDArray[Any],
    mix_train_yc: NDArray[Any],
    train_y: NDArray[Any],
    test_y: NDArray[Any],
    train_dates_list: NDArray[Any],
    test_dates_list: NDArray[Any],
):
    if os.path.exists(embeddings_input_path):
        os.remove(embeddings_input_path)

    p_path_dir = os.path.dirname(embeddings_input_path)
    if not os.path.exists(p_path_dir):
        os.makedirs(p_path_dir)

    # Inverse transform outputs, save results
    with open(
        embeddings_input_path,
        "wb",
    ) as f:
        mix_yhat = np.expand_dims(np.array(data_scaler.inverse_transform(mix_yhat[:, :])), axis=1)
        mix_yc = np.expand_dims(np.array(data_scaler.inverse_transform(mix_yc[:, 0, :])), axis=1)
        mix_train_yhat = np.expand_dims(
            np.array(data_scaler.inverse_transform(mix_train_yhat[:, :])), axis=1
        )
        mix_train_yc = np.expand_dims(
            np.array(data_scaler.inverse_transform(mix_train_yc[:, 0, :])), axis=1
        )
        train_dates_list = train_dates_list[:, 0]
        test_dates_list = test_dates_list[:, 0]
        train_labels = np.array(data_scaler.inverse_transform(np.rollaxis(train_y, 2, 1)[:, 0, :]))
        test_labels = np.array(data_scaler.inverse_transform(np.rollaxis(test_y, 2, 1)[:, 0, :]))
        train_labels = train_labels[:, 0]
        test_labels = test_labels[:, 0]
        pickle.dump(
            [
                mix_yhat,
                mix_train_yhat,
                mix_yc,
                mix_train_yc,
                train_labels,
                test_labels,
                train_dates_list,
                test_dates_list,
            ],
            f,
        )

    return mix_yhat, mix_train_yhat, mix_yc, mix_train_yc, train_labels, test_labels


def get_date_range(
    stations: List[Dict[str, Any]], infer_station_name: str, root_path: str, model_type: str
):
    for station in stations:
        if station["name"] != infer_station_name:
            model_path = os.path.join(root_path, station["name"], model_type)
            train_data_path = os.path.join(model_path, "train_data_dates.pkl")
            (
                _,
                _,
                _,
                _,
                _,
                _,
                _,
                train_dates_list,
                _,
                test_dates_list,
            ) = get_file(train_data_path)

            return (train_dates_list, test_dates_list)
    raise Exception("No station found to get date range")


def get_station_object(stations: List[Dict[str, Any]], infer_station_name: str):
    station, column_name = None, None
    for stations_dict in stations:
        if stations_dict["name"] == infer_station_name:
            station = stations_dict["name"]
            column_name = stations_dict["column_name"]
            return station, column_name

    raise Exception(f"No station found with name {infer_station_name}")


def dump_forecast_output(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_path: str,
    column_name: str,
    train_dates_list: List[str],
    test_dates_list: List[str],
    inference_hours: int,
):
    train_data = np.array(train_df[column_name].values)
    test_data = np.array(test_df[column_name].values)
    mix_train_yhat = transform_to_array_3D(train_data[:-inference_hours], inference_hours)
    mix_train_y = transform_to_array_3D(train_data[inference_hours:], inference_hours)
    mix_test_yhat = transform_to_array_3D(test_data[:-inference_hours], inference_hours)
    mix_test_y = transform_to_array_3D(test_data[inference_hours:], inference_hours)
    out_dir = os.path.join(model_path, "embeddings")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_path = os.path.join(out_dir, "post_processed_results.pkl")

    # Inverse transform outputs, save results
    with open(out_path, "wb") as f:
        train_labels = mix_train_y.squeeze()
        test_labels = mix_test_y.squeeze()
        train_labels = train_labels[:, 0]
        test_labels = test_labels[:, 0]

        pickle.dump(
            [
                mix_test_yhat,
                mix_train_yhat,
                mix_test_y,
                mix_train_y,
                train_labels,
                test_labels,
                train_dates_list,
                test_dates_list,
            ],
            f,
        )


def embeddings_preprocess_forecast(
    stations: List[Dict[str, Any]],
    infer_station_name: str,
    root_path: str,
    input_data_path: str,
    forecast_interval: int,
    model_type: str,
    column_name: str,
):
    model_path = os.path.join(root_path, infer_station_name, model_type)
    forecast_df = get_csv_data(input_data_path)
    train_dates_list, test_dates_list = get_date_range(
        stations, infer_station_name, root_path, model_type
    )
    train_df = forecast_df[forecast_df.index.isin(train_dates_list[:, 0])]
    test_df = forecast_df[forecast_df.index.isin(test_dates_list[:, 0])]

    train_dates_list = (
        train_df[forecast_interval:].index.strftime("%Y-%m-%d %H:%M:%S").tolist()  # type: ignore
    )
    test_dates_list = (
        test_df[forecast_interval:].index.strftime("%Y-%m-%d %H:%M:%S").tolist()  # type: ignore
    )

    dump_forecast_output(
        train_df,
        test_df,
        model_path,
        column_name,
        train_dates_list,
        test_dates_list,
        forecast_interval,
    )


def embeddings_preprocess_deepmc(
    model_path: str,
    inference_hours: int,
):
    train_data_path = os.path.join(model_path, "train_data_dates.pkl")
    (
        train_X,
        train_y,
        test_X,
        test_y,
        _,
        output_scaler1,
        _,
        train_dates_list,
        _,
        test_dates_list,
    ) = get_file(train_data_path)

    list_train_X = inference_deepmc(model_path, train_X, inference_hours)
    list_test_X = inference_deepmc(model_path, test_X, inference_hours)

    # Train data deepmc inference Post-Processing
    mix_train_yc = preprocess_post_deepmc_gt(list_train_X, train_y, inference_hours)
    mix_train_yhat = inference_deepmc_post(model_path, list_train_X)

    # Test data deepmc inference Post-Processing
    mix_yc = preprocess_post_deepmc_gt(list_test_X, test_y, inference_hours)
    mix_yhat = inference_deepmc_post(model_path, list_test_X)

    mix_train_yhat, train_dates_list = preprocess_transform(
        mix_train_yhat, inference_hours, train_dates_list
    )
    mix_yhat, test_dates_list = preprocess_transform(mix_yhat, inference_hours, test_dates_list)
    embeddings_input_path = os.path.join(model_path, "embeddings", "post_processed_results.pkl")

    # Inverse transform outputs, save results
    write_embeddings_input(
        embeddings_input_path,
        output_scaler1,
        mix_yhat,
        mix_train_yhat,
        mix_yc,
        mix_train_yc,
        train_y,
        test_y,
        train_dates_list,
        test_dates_list,
    )


def preprocess_post_deepmc_gt(
    post_data_x: List[NDArray[Any]], data_y: NDArray[Any], inference_hours: int
):
    data_y = data_y[: data_y.shape[0] - inference_hours]
    mix_data_gt = np.empty([data_y.shape[0], data_y.shape[1], len(post_data_x)])

    idx = 0
    for _, _ in enumerate(post_data_x):
        mix_data_gt[:, :, idx] = mix_data_gt[:, idx, :]
        idx = idx + 1

    return mix_data_gt


def initialize_embeddings_preprocessing(
    infer_station_name: str,
    stations: List[Dict[str, Any]],
    root_path: str,
    infer_forecast_data_path: str,
    infer_interval: int,
    model_type: str,
):
    for station in stations:
        model_path = os.path.join(root_path, station["name"], model_type)
        if station["name"] == infer_station_name:
            embeddings_preprocess_forecast(
                stations,
                infer_station_name,
                root_path,
                infer_forecast_data_path,
                infer_interval,
                model_type,
                station["column_name"],
            )
        else:
            embeddings_preprocess_deepmc(
                model_path,
                inference_hours=24,
            )
