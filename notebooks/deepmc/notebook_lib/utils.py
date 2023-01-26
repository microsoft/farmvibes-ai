from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from sklearn.preprocessing import StandardScaler


def get_csv_data(
    path: str,
    date_attribute: str = "date",
    columns_rename: Dict[str, str] = {},
    frequency: str = "60min",
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

    # interpolate to derive missing data
    data_df = data_df.interpolate(method="from_derivatives")
    data_df = data_df.dropna()

    # Group rows by frequency, requires date attribute indexed to execute this
    data_df = data_df.fillna(method="ffill")
    data_df = data_df.fillna(method="bfill")
    data_df = data_df.groupby(pd.Grouper(freq=frequency)).mean()
    data_df = data_df.fillna(method="ffill")
    data_df = data_df.fillna(method="bfill")

    return data_df


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


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
    output_scaler.fit_transform(np.expand_dims(data[out_feature].values, axis=1))

    train_scaler = StandardScaler()
    train_scale_df = pd.DataFrame(
        train_scaler.fit_transform(train_data), columns=train_data.columns, index=train_data.index
    )
    test_scale_df = pd.DataFrame(
        train_scaler.transform(test_data), columns=test_data.columns, index=test_data.index
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
    base_data_df = base_data_df.dropna()
    return base_data_df


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth
