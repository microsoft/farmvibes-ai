from datetime import datetime, timedelta
from typing import Dict

import numpy as np
import pandas as pd
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
    test_data = data.iloc[:split]

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
