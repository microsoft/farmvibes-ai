import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from notebook_lib.base_deepmc import inference_deepmc, inference_deepmc_post
from notebook_lib.data_utils import preprocess_transform
from numpy.typing import NDArray
from shapely import geometry
from sklearn.preprocessing import StandardScaler

from vibe_notebook.deepmc import prediction, utils
from vibe_notebook.deepmc.forecast import Forecast
from vibe_notebook.deepmc.utils import get_csv_data, transform_to_array_3D

HRRR_PARAMETERS = [
    {"weather_type": "temperature", "search_text": "TMP:2 m"},
    {"weather_type": "humidity", "search_text": "RH:2 m"},
    {"weather_type": "u-component", "search_text": "UGRD:10 m"},
    {"weather_type": "v-component", "search_text": "VGRD:10 m"},
]


def get_date_range(
    stations: List[Dict[str, Any]],
    infer_station_name: str,
    deepmc_inference_results: Dict[str, Any],
):
    for station in stations:
        if station["name"] != infer_station_name:
            (_, dates_list, _, _) = deepmc_inference_results[station["name"]]
            dates_list = np.squeeze(np.array(dates_list)[:, 0])
            dates_list = dates_list[:, 0]
            return dates_list

    raise Exception("No station found to get date range")


def get_station_object(stations: List[Dict[str, Any]], infer_station_name: str):
    station, column_name = None, None
    for stations_dict in stations:
        if stations_dict["name"] == infer_station_name:
            station = stations_dict["name"]
            column_name = stations_dict["column_name"]
            return station, column_name

    if station is None:
        raise Exception(f"No station found with name {infer_station_name}")


def embeddings_preprocess_forecast(
    stations: List[Dict[str, Any]],
    infer_station_name: str,
    input_data_path: str,
    forecast_interval: int,
    deepmc_inference_results: Dict[str, Any],
    column_name: str,
):
    forecast_df = get_csv_data(input_data_path)
    dates_list = get_date_range(stations, infer_station_name, deepmc_inference_results)
    data_df = forecast_df[forecast_df.index.isin(dates_list)]

    dates_list = (
        data_df[forecast_interval:].index.strftime("%Y-%m-%d %H:%M:%S").tolist()  # type: ignore
    )

    data_forecast = np.array(data_df[column_name].values)
    data_forecast = transform_to_array_3D(data_forecast[:], forecast_interval)

    return data_forecast, dates_list


def embeddings_preprocess_deepmc(
    model_path: str,
    inference_hours: int,
    deepmc_inference_results: Tuple[NDArray[Any], NDArray[Any], StandardScaler, StandardScaler],
):
    (data_x, dates_list, _, output_scaler) = deepmc_inference_results

    deepmc_out = inference_deepmc(model_path, data_x, inference_hours)

    # Train Post-Processing Scaling Models
    mix_yhat = inference_deepmc_post(model_path, deepmc_out)
    mix_yhat, dates_list = preprocess_transform(mix_yhat, inference_hours, dates_list)
    dates_list = np.squeeze(np.array(dates_list)[:, 0])
    dates_list = dates_list[:, 0]
    dates_list = pd.to_datetime(dates_list).strftime("%Y-%m-%d %H:%M:%S")
    mix_yhat = np.expand_dims(np.array(output_scaler.inverse_transform(mix_yhat[:, :])), axis=1)
    return mix_yhat, dates_list


def inference_embeddings_preprocessing(
    infer_station_name: str,
    stations: List[Dict[str, Any]],
    root_path: str,
    infer_forecast_data_path: str,
    infer_interval: int,
    model_type: str,
    deepmc_inference_results: Dict[str, Any],
):
    process_out = {}
    for station in stations:
        model_path = os.path.join(root_path, station["name"], model_type)
        if station["name"] == infer_station_name:
            process_out[station["name"]] = embeddings_preprocess_forecast(
                stations,
                infer_station_name,
                infer_forecast_data_path,
                infer_interval,
                deepmc_inference_results,
                station["column_name"],
            )
        else:
            process_out[station["name"]] = embeddings_preprocess_deepmc(
                model_path,
                infer_interval,
                deepmc_inference_results[station["name"]],
            )
    return process_out


def download_forecast_data(
    stations: List[Dict[str, Any]],
    start_date: datetime,
    end_date: datetime,
):
    parameters = HRRR_PARAMETERS
    hrrr_data_workflow = "data_ingestion/weather/herbie_forecast"
    time_range = (start_date, end_date)
    forecast_dataset = {}

    for station in stations:
        # AGWeatherNet station
        station_name = station["name"]
        station_location = station["coordinates"]
        station_geometry = geometry.Point(station_location)

        forecast_ = Forecast(
            workflow_name=hrrr_data_workflow,
            geometry=station_geometry,
            time_range=time_range,
            parameters=parameters,
        )
        run_list = forecast_.submit_download_request()

        p_forecast_dataset = forecast_.get_downloaded_data(run_list=run_list, offset_hours=-8)
        p_forecast_dataset = utils.convert_forecast_data(p_forecast_dataset)
        forecast_dataset[station_name] = p_forecast_dataset
    return forecast_dataset


def get_historical_data(
    stations: List[Dict[str, Any]],
    historical_data_path: str,
    historical_dataset_features: List[str],
    inference_station: str,
):
    historical_datasets = {}
    for station in stations:
        if station["name"] != inference_station:
            p = historical_data_path % station["name"]
            historical_df = utils.get_csv_data(path=p, interpolate=False, fill_na=False)
            historical_df = historical_df[historical_dataset_features]

            historical_datasets[station["name"]] = historical_df

    return historical_datasets


def concat_historical_forecast(
    stations: List[Dict[str, Any]],
    historical_data_path: str,
    hrrr_datasets: Dict[str, pd.DataFrame],
    start_date: datetime,
    end_date: datetime,
    inference_station: str,
    historical_dataset_features: List[str] = ["humidity", "wind_speed", "temperature"],
    forecast_dataset_features: List[str] = [
        "humidity_forecast",
        "wind_speed_forecast",
        "temperature_forecast",
    ],
    frequency_hour: int = 1,
    number_of_hours: int = 24,
    weather_inference_type: str = "temperature",
):
    historical_datasets = get_historical_data(
        stations, historical_data_path, historical_dataset_features, inference_station
    )

    dataset_variables = historical_dataset_features.copy()
    dataset_variables.extend(forecast_dataset_features)
    dataset_variables.sort()

    out_dataset = {}
    for station, historical_df in historical_datasets.items():
        forecast_df = hrrr_datasets[station]

        input_df = utils.clean_relevant_data_using_hrrr(
            actual_df=historical_df.copy(),
            forecast_df=forecast_df.copy(),
            out_variables=dataset_variables,
            freq_hours=frequency_hour,
            num_of_indices=number_of_hours,
            start_date=start_date,
            end_date=end_date,
        )

        input_df = input_df[dataset_variables]
        input_df = input_df[input_df.columns]
        out_feature_df = input_df[weather_inference_type]
        input_df.drop(columns=[weather_inference_type], inplace=True)
        input_df[weather_inference_type] = out_feature_df
        out_dataset[station] = input_df

    return out_dataset


def run_deepmc_inference(
    root_path: str,
    model_type: str,
    out_features: List[str],
    stations: List[Dict[str, Any]],
    historical_data_path: str,
    hrrr_datasets: Dict[str, pd.DataFrame],
    start_date: datetime,
    end_date: datetime,
    inference_station: str,
    historical_dataset_features: List[str] = ["humidity", "wind_speed", "temperature"],
    forecast_dataset_features: List[str] = [
        "humidity_forecast",
        "wind_speed_forecast",
        "temperature_forecast",
    ],
    frequency_hour: int = 1,
    number_of_hours: int = 24,
    weather_inference_type: str = "temperature",
):
    historical_clean_dataset = concat_historical_forecast(
        stations,
        historical_data_path,
        hrrr_datasets,
        start_date,
        end_date,
        inference_station,
        historical_dataset_features,
        forecast_dataset_features,
        frequency_hour,
        number_of_hours,
        weather_inference_type,
    )

    inference_output = {}
    for station, clean_dataset in historical_clean_dataset.items():
        train_data_export_path = os.path.join(root_path, station, model_type, "train_data.pkl")

        weather_forecast = prediction.InferenceWeather(
            root_path=root_path,
            data_export_path=train_data_export_path,
            station_name=station,
            predicts=out_features,
            relevant=True,
        )

        inference_output[station] = weather_forecast.deepmc_preprocess(clean_dataset, "temperature")

    return inference_output
