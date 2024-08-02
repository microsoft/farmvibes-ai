# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from datetime import datetime, timedelta
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from numpy._typing import NDArray
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


def get_csv_data(
    path: str,
    date_attribute: str = "date",
    columns_rename: Dict[str, str] = {},
    frequency: str = "60min",
    interpolate: bool = True,
    fill_na: bool = True,
):
    """
    Read data from CSV file using Pandas python package.
    """

    data_df = pd.read_csv(path)
    data_df[date_attribute] = pd.to_datetime(data_df[date_attribute])

    if columns_rename:
        data_df.rename(columns=columns_rename, inplace=True)

    # apply index on date
    data_df.reset_index(drop=True, inplace=True)
    data_df.set_index(date_attribute, inplace=True)
    data_df.sort_index(ascending=True, inplace=True)

    if interpolate:
        # interpolate to derive missing data
        data_df = data_df.interpolate(method="from_derivatives")
        assert data_df is not None, "Interpolate deleted all data"
        data_df = data_df.dropna()

    if fill_na:
        # Group rows by frequency, requires date attribute indexed to execute this
        data_df = data_df.fillna(method="ffill")  # type: ignore
        data_df = data_df.fillna(method="bfill")
        data_df = data_df.groupby(pd.Grouper(freq=frequency)).mean()
        data_df = data_df.fillna(method="ffill")
        data_df = data_df.fillna(method="bfill")
    else:
        data_df = data_df.groupby(pd.Grouper(freq=frequency)).mean()

    return data_df


def hour_round(t: datetime):
    # Rounds to nearest hour by adding a timedelta hour if minute >= 30
    return t.replace(second=0, microsecond=0, minute=0, hour=t.hour) + timedelta(
        hours=t.minute // 30
    )


def get_split_scaled_data(data: pd.DataFrame, out_feature: str, split_ratio: float = 0.92):
    split = int(split_ratio * data.shape[0])

    train_data = data.iloc[:split]
    test_data = data.iloc[split:]

    output_scaler = StandardScaler()
    output_scaler.fit_transform(np.expand_dims(data[out_feature].values, axis=1))  # type: ignore

    train_scaler = StandardScaler()
    train_scale_df = pd.DataFrame(
        train_scaler.fit_transform(train_data),
        columns=train_data.columns,
        index=train_data.index,
    )
    test_scale_df = pd.DataFrame(
        train_scaler.transform(test_data),
        columns=test_data.columns,
        index=test_data.index,
    )

    return train_scaler, output_scaler, train_scale_df, test_scale_df


def shift_index(ds_df: pd.DataFrame, freq_minutes: int, num_indices: int, dateColumn: str = "date"):
    ds_df[dateColumn] = ds_df.index.shift(-num_indices, freq=DateOffset(minutes=freq_minutes))
    ds_df = ds_df.reset_index(drop=True)
    ds_df = ds_df.set_index(dateColumn)
    return ds_df


def clean_relevant_data(
    actual_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    out_variables: List[str],
    freq_hours: int,
    num_of_indices: int,
):
    base_data_df = actual_df.copy()
    current_ws_df = forecast_df.add_suffix("Current")
    base_data_df = base_data_df.join(current_ws_df)
    shift_forecast_df = shift_index(forecast_df, freq_hours * 60, num_of_indices)
    base_data_df = base_data_df.join(shift_forecast_df)

    base_data_df = base_data_df[out_variables]
    base_data_df = base_data_df.interpolate(method="from_derivatives")
    assert base_data_df is not None, "Interpolate deleted all data"
    base_data_df = base_data_df.dropna()
    return base_data_df


def smooth(y: List[float], box_pts: int):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


def clean_relevant_data_using_hrrr(
    actual_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    out_variables: List[str],
    freq_hours: int,
    num_of_indices: int,
    start_date: datetime,
    end_date: datetime,
):
    forecast_df = forecast_df.loc[
        (forecast_df.index >= start_date) & (forecast_df.index <= end_date)
    ]
    actual_df = actual_df.loc[(actual_df.index >= start_date) & (actual_df.index <= end_date)]

    for col in actual_df.columns:
        sub_df = actual_df[actual_df[col].isna()]
        if col + "_forecast" in forecast_df.columns:
            actual_df.loc[actual_df.index.isin(sub_df.index.values), col] = forecast_df[
                forecast_df.index.isin(sub_df.index.values)
            ][col + "_forecast"]

    base_data_df = actual_df.copy()
    current_ws_df = forecast_df.add_suffix("Current")
    base_data_df = base_data_df.join(current_ws_df)
    shift_forecast_df = shift_index(forecast_df, freq_hours * 60, num_of_indices)
    base_data_df = base_data_df.join(shift_forecast_df)

    base_data_df = base_data_df[out_variables]
    base_data_df = base_data_df.interpolate(method="from_derivatives")
    assert base_data_df is not None, "Interpolate deleted all data"
    base_data_df = base_data_df.dropna()
    return base_data_df


def calculate_KPI(y: NDArray[Any], yhat: NDArray[Any]):
    mae = float(mean_absolute_error(y, yhat))
    rmse = float(mean_squared_error(y, yhat, squared=False))
    print(f"RMSE: {round(rmse, 2)}")
    print(f"MAE: {round(mae, 2)}")
    print(f"MAE%: {round(100*sum(abs(y-yhat))/sum(y),2)}%")


def convert_forecast_data(data: pd.DataFrame):
    # Temperature
    # convert kelvin to celsius
    # convert celsius to Fahrenheit
    data["temperature_forecast"] = data["temperature_forecast"].apply(
        lambda x: ((x - 273.15) * 9 / 5) + 32
    )

    # wind_speed
    # multiplying with 2.23 to convert wind speed from m/sec to mph
    data["wind_speed_forecast"] = data.apply(
        lambda x: np.sqrt(
            np.square(x["u-component_forecast"]) + np.square(x["v-component_forecast"])
        )
        * 2.23,
        axis=1,
    )
    data.drop(columns=["u-component_forecast", "v-component_forecast"], inplace=True)
    return data


def transform_to_array_3D(data: NDArray[Any], inference_hours: int = 24) -> NDArray[Any]:
    X = transform_to_array(data, inference_hours)
    X = X.reshape(X.shape[0], 1, X.shape[1])
    return X


def transform_to_array(data: NDArray[Any], inference_hours: int = 24) -> NDArray[Any]:
    data = np.array(data)
    X = []
    for in_start in range(len(data)):
        in_end = in_start + inference_hours
        if in_end <= (len(data)):
            X.append(data[in_start:in_end])
        else:
            break

    X = np.array(X)
    # skip rows not in loop
    X = X[: data.shape[0] - inference_hours]
    return X
