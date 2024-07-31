import os
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from .data_utils import get_file


def construct_neighbor_stations(stations: List[Dict[str, Any]]):
    neighbors = {"stations": [], "coordinates": {}}
    for station in stations:
        neighbors["stations"].append(station["name"])
        neighbors["coordinates"][station["name"]] = station["coordinates"]

    return neighbors


def get_deepmc_post_results(root_path: str, stations: List[Dict[str, Any]], model_type: str):
    predict_out = {}
    for station in stations:
        deepmc_post_path = os.path.join(
            root_path, station["name"], model_type, "embeddings", "post_processed_results.pkl"
        )
        (
            intermediate_test,
            intermediate_train,
            _,
            _,
            train_labels_station,
            test_labels_station,
            out_train_dates,
            out_test_dates,
        ) = get_file(deepmc_post_path)
        predict_out[station["name"]] = (
            intermediate_train,
            intermediate_test,
            train_labels_station,
            test_labels_station,
            out_train_dates,
            out_test_dates,
        )

    return predict_out


def get_date(stations: Dict[str, Any], data_index: int = -2, date_type: int = 0):
    """Retrieves the start date and end date  by comparing data of all stations.
    :param stations: Dictionary with station name as key and values
    with collection of station information used to generate embeddings.

    :param data_index: It defines position of data in array.
    will use -2 for train, -1 for test, 1 for inference.

    :param date_type: 0 for start_date, -1 for end_date.

    return: date.
    """
    station_name = next(iter(stations))
    station_values = stations[station_name]
    date = datetime.strptime(station_values[data_index][date_type], "%Y-%m-%d %H:%M:%S")
    for station_values in stations.values():
        try:
            s_date = datetime.strptime(station_values[data_index][date_type], "%Y-%m-%d %H:%M:%S")
            # for start date
            if date_type == 0 and date < s_date:
                date = s_date
            # for end date
            if date_type == -1 and date > s_date:
                date = s_date
        except Exception as e:
            print(e)
    return date


def create_embeddings(
    stations: List[Dict[str, Any]],
    inference_hours: int,
    root_path: str,
    model_type: str,
):
    neighbor_stations = construct_neighbor_stations(stations)
    predict_out = get_deepmc_post_results(root_path, stations, model_type)

    # get start date
    train_start_date = get_date(predict_out, data_index=-2, date_type=0)
    test_start_date = get_date(predict_out, data_index=-1, date_type=0)

    # get end date
    train_end_date = get_date(predict_out, data_index=-2, date_type=-1)
    test_end_date = get_date(predict_out, data_index=-1, date_type=-1)

    test_start_date = datetime.strptime(
        test_start_date.strftime("%Y-%m-%d") + " " + train_start_date.strftime("%H:%M:%S"),
        "%Y-%m-%d %H:%M:%S",
    )

    df_train_embeddings = process_embeddings(
        predict_out=predict_out,
        inference_hours=inference_hours,
        neighbor_stations=neighbor_stations,
        start_date=train_start_date,
        end_date=train_end_date,
        data_index=0,
        label_index=2,
        timestamp_index=4,
    )

    df_test_embeddings = process_embeddings(
        predict_out=predict_out,
        inference_hours=inference_hours,
        neighbor_stations=neighbor_stations,
        start_date=test_start_date,
        end_date=test_end_date,
        data_index=1,
        label_index=3,
        timestamp_index=5,
    )

    return df_train_embeddings, df_test_embeddings


def create_embeddings_inference(
    stations: List[Dict[str, Any]],
    inference_hours: int,
    deepmc_post_results: Dict[str, Any],
):
    neighbor_stations = construct_neighbor_stations(stations)
    inference_start_date = get_date(deepmc_post_results, data_index=1, date_type=0)
    inference_end_date = get_date(deepmc_post_results, data_index=1, date_type=-1)

    df_embeddings = get_inference_embeddings(
        predict_out=deepmc_post_results,
        inference_hours=inference_hours,
        neighbor_stations=neighbor_stations,
        start_date=inference_start_date,
        end_date=inference_end_date,
    )

    return df_embeddings


def get_inference_embeddings(
    predict_out: Dict[str, Any],
    inference_hours: int,
    neighbor_stations: Dict[str, Any],
    start_date: datetime,
    end_date: datetime,
):
    embeddings = []
    for station in neighbor_stations["stations"]:
        df = pd.DataFrame(
            predict_out[station][0].reshape(
                predict_out[station][0].shape[0], predict_out[station][0].shape[2]
            ),
            columns=list(range(inference_hours)),
        )
        timestamps = predict_out[station][1]

        df["station"] = station
        df["timestamp"] = timestamps
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")

        mask = (df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)
        df = df.loc[mask]

        df.reset_index(drop=True, inplace=True)
        df["forecast_step"] = df.index
        embeddings.append(df)

    df_embeddings = pd.concat(embeddings, axis=0)
    df_embeddings.sort_values(by=["forecast_step", "station"], inplace=True)
    return df_embeddings


def process_embeddings(
    predict_out: Dict[str, Any],
    inference_hours: int,
    neighbor_stations: Dict[str, Any],
    start_date: datetime,
    end_date: datetime,
    data_index: int,
    label_index: int,
    timestamp_index: int,
):
    """
    Process embeddings for train or test data.

    :param predict_out: Dictionary with station name as key and values. It's output of deepmc post processing.
    :param inference_hours: Number of hours to predict.
    :param neighbor_stations: Dictionary with stations and coordinates.
    :param start_date: Start date for embeddings.
    :param end_date: End date for embeddings.
    :param data_index: Index of train or test data in predict_out. The pickle file
        generated by deepmc follows this index train=0, test=1
    :param label_index: Index of train or test labels in predict_out. The pickle file
        generated by deepmc follows this index train=2, test=3
    :param timestamp_index: Index of train or test timestamps in predict_out. The pickle file
        generated by deepmc follows this index train=4, test=5
    """
    embeddings = []
    for station in neighbor_stations["stations"]:
        df = pd.DataFrame(
            predict_out[station][data_index].reshape(
                predict_out[station][data_index].shape[0], predict_out[station][data_index].shape[2]
            ),
            columns=list(range(inference_hours)),
        )

        labels = predict_out[station][label_index]
        timestamps = predict_out[station][timestamp_index]

        df["station"] = station
        if len(timestamps) < len(labels):
            labels = labels[: len(timestamps)]

        df["labels"] = labels

        if len(timestamps) > len(labels):
            timestamps = timestamps[: len(labels)]
        df["timestamp"] = timestamps

        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S")

        mask = (df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)
        df = df.loc[mask]

        df.reset_index(drop=True, inplace=True)
        df["forecast_step"] = df.index

        embeddings.append(df)

    df_embeddings = pd.concat(embeddings, axis=0)
    df_embeddings.sort_values(by=["forecast_step", "station"], inplace=True)
    return df_embeddings
