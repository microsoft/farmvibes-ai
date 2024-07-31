from datetime import timedelta
from math import ceil
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import pywt
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler


class Preprocess:
    def __init__(
        self,
        train_scaler: StandardScaler,
        output_scaler: StandardScaler,
        is_training: bool,
        is_validation: bool = False,
        ts_lookahead: int = 24,
        ts_lookback: int = 24,
        chunk_size: int = 528,
        wavelet: str = "bior3.5",
        mode: str = "periodic",
        level: int = 5,
        relevant: bool = False,
    ):
        self.train_scaler = train_scaler
        self.output_scaler = output_scaler
        self.trunc = chunk_size
        self.ts_lookback = ts_lookback
        self.wavelet = wavelet
        self.mode = mode
        self.level = level
        self.is_training = is_training
        self.ts_lookahead = ts_lookahead
        self.is_validation = is_validation
        self.relevant = relevant

    def wavelet_transform_predict(
        self, df_in: pd.DataFrame, predict: str
    ) -> Tuple[NDArray[Any], List[Any], List[Any]]:
        i = 1
        start = i
        end = start
        t_test_X = []
        t_x_dates = []
        t_y_dates = []

        test_df = pd.DataFrame(
            self.train_scaler.transform(df_in), columns=df_in.columns, index=df_in.index
        )

        # convert input data to wavelet
        while end < test_df.shape[0]:
            start = i
            end = start + self.trunc
            i = i + 1
            chunkdataDF = test_df.iloc[start:end]

            test_uX, _, test_x_dates, test_y_dates = self.convert_df_wavelet_input(
                data_df=chunkdataDF, predict=predict
            )

            t_test_X.append(test_uX)
            t_x_dates.append(test_x_dates)
            t_y_dates.append(test_y_dates)

        test_X = t_test_X[0].copy()

        for i in range(1, len(t_test_X)):
            for j in range(len(t_test_X[i])):
                test_X[j] = np.append(test_X[j], t_test_X[i][j], axis=0)

        return test_X, t_x_dates, t_y_dates

    def wavelet_transform_train(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, out_feature: str
    ) -> Tuple[NDArray[Any], ...]:
        t_train_X, t_train_y, t_train_X_dates, t_train_y_dates = self.prepare_wavelet_data(
            train_df, out_feature=out_feature
        )

        t_test_X, t_test_y, t_test_X_dates, t_test_y_dates = self.prepare_wavelet_data(
            test_df, out_feature=out_feature
        )

        train_X = t_train_X[0].copy()
        train_y = t_train_y[0].copy()
        train_dates_X = t_train_X_dates[0][0].copy()
        train_dates_y = t_train_y_dates[0].copy()
        for i in range(len(t_train_X)):
            train_y = np.append(train_y, t_train_y[i], axis=0)
            train_dates_X = np.append(train_dates_X, t_train_X_dates[i][0], axis=0)
            train_dates_y = np.append(train_dates_y, t_train_y_dates[i], axis=0)
            for j in range(len(t_train_X[i])):
                train_X[j] = np.append(train_X[j], t_train_X[i][j], axis=0)

        test_X = t_test_X[0].copy()
        test_y = t_test_y[0].copy()
        test_dates_X = t_test_X_dates[0][0].copy()
        test_dates_y = t_test_y_dates[0].copy()
        for i in range(1, len(t_test_X)):
            test_y = np.append(test_y, t_test_y[i], axis=0)
            test_dates_X = np.append(test_dates_X, t_test_X_dates[i][0], axis=0)
            test_dates_y = np.append(test_dates_y, t_test_y_dates[i], axis=0)
            for j in range(len(t_test_X[i])):
                test_X[j] = np.append(test_X[j], t_test_X[i][j], axis=0)

        return (
            train_X,
            train_y,
            test_X,
            test_y,
            train_dates_X,
            train_dates_y,
            test_dates_X,
            test_dates_y,
        )

    def prepare_wavelet_data(self, data_df: pd.DataFrame, out_feature: str):
        i = 0
        start = i * self.trunc
        end = start
        t_data_x = []
        t_data_y = []
        t_dates_x = []
        t_dates_y = []

        while end < data_df.shape[0]:
            start = i
            end = start + self.trunc
            i = i + 1
            o_data_df = data_df.iloc[start:end]

            data_ux, data_uy, data_ux_dates, data_uy_dates = self.convert_df_wavelet_input(
                o_data_df,
                predict=out_feature,
            )
            t_data_x.append(data_ux)
            t_data_y.append(data_uy)
            t_dates_x.append(data_ux_dates)
            t_dates_y.append(data_uy_dates)

        return t_data_x, t_data_y, t_dates_x, t_dates_y

    def dl_preprocess_data(
        self,
        df: pd.DataFrame,
        predict: str,
        per_split: float = 0.8,
        training: bool = False,
    ) -> Tuple[NDArray, Optional[NDArray], Optional[NDArray], Optional[NDArray], Optional[NDArray]]:  # type: ignore
        """
        merge chunk of data as single entity
        Args:
            df: lookback input data chunk based on number of models
            predict: feature to predict
            per_split: percentage data split
        Returns:
            data as single entity
        """

        n_in = self.ts_lookback
        scaled_df = df
        data = scaled_df.values.astype(float)

        if training:
            n_out = self.ts_lookahead
            label_df = df.copy()
            for column in label_df:
                if column != predict:
                    label_df.drop(columns=column, inplace=True)

            label_data = label_df.values

            # label_data = label_df.values
            X, y, dates = list(), list(), list()
            in_start = 0

            # step over the entire history one time step at a time
            # reshape input to be 3D [samples, timesteps, features]
            for _ in range(len(data)):
                # define the end of the input sequence
                in_end = in_start + n_in
                out_end = in_end + n_out
                # ensure we have enough data for this instance
                if out_end <= len(data):
                    X.append(data[in_start:in_end, :])
                    y.append(label_data[in_end:out_end, :])
                    dates.append(df.index[in_end:out_end].strftime("%Y-%m-%d %H:%M:%S").values)
                # move along one time step
                in_start += 1

            X = np.array(X)
            y = np.array(y)
            dates = np.array(dates)

            if self.is_validation is True:
                n_train_split = ceil(len(data) * per_split)
                train_X, train_y = X[:n_train_split, :, :], y[:n_train_split, :, :]
                test_X, test_y = X[n_train_split:, :], y[n_train_split:, :]

                return train_X, train_y, test_X, test_y, dates
            else:
                return X, y, None, None, dates
        else:
            X, dates = list(), list()
            in_start = 0
            for _ in range(len(data) - n_in + 1):
                in_end = in_start + n_in
                if in_end <= len(data):
                    X.append(data[in_start:in_end, :])
                    # shift dates by lookahead to match it with the y
                    dates.append(
                        [t + timedelta(hours=self.ts_lookback) for t in df.index[in_start:in_end]]
                    )
                in_start += 1
            X = np.array(X)
            dates = np.array(dates)
        return X, None, None, None, dates

    def convert_df_wavelet_input(self, data_df: pd.DataFrame, predict: str):
        if self.relevant:
            return self.convert_df_wavelet_input_relevant(data_df, predict)
        else:
            return self.convert_df_wavelet_input_not_relevant(data_df, predict)

    def convert_df_wavelet_input_not_relevant(self, data_df: pd.DataFrame, predict: str):
        level = self.level
        rd = list()
        N = data_df.shape[0]
        test_X, test_X_dates, test_y_dates, test_y = list(), list(), list(), list()

        if self.is_training:
            (_, test_y, _, _, test_y_dates) = self.dl_preprocess_data(
                data_df.iloc[-self.ts_lookback - self.ts_lookahead :],
                predict=predict,
                training=self.is_training,
            )

            assert test_y is not None
            test_y = test_y[[-1], :, :]
            dates = test_y_dates[[-1], :]

            data_df = data_df.iloc[: -self.ts_lookahead]

        wp5 = pywt.wavedec(data=data_df[predict], wavelet=self.wavelet, mode=self.mode, level=level)
        N = data_df.shape[0]
        for i in range(1, level + 1):
            rd.append(pywt.waverec(wp5[:-i] + [None] * i, wavelet=self.wavelet, mode=self.mode)[:N])

        (t_test_X, _, _, _, t_test_X_dates) = self.dl_preprocess_data(
            data_df.iloc[-self.ts_lookback :], predict=predict
        )

        test_X.append(t_test_X[[-1], :, :])
        test_X_dates.append(t_test_X_dates[[-1], :])
        wpt_df = data_df[[]].copy()

        for i in range(0, level):
            wpt_df[predict] = rd[i][:]

            (t_test_X, _, _, _, t_test_X_dates) = self.dl_preprocess_data(
                wpt_df.iloc[-self.ts_lookback :], predict=predict
            )

            test_X.append(t_test_X[[-1], :, :])
            test_X_dates.append(t_test_X_dates)

        return test_X, test_y, test_X_dates, test_y_dates

    def convert_df_wavelet_input_relevant(self, data_df: pd.DataFrame, predict: str):
        rd = list()
        test_X = list()
        test_X, test_X_dates, test_y_dates, test_y = list(), list(), list(), list()

        if self.is_training:
            (_, test_y, _, _, test_y_dates) = self.dl_preprocess_data(
                data_df.iloc[-self.ts_lookback - self.ts_lookahead :],
                predict=predict,
                training=self.is_training,
            )

            assert test_y is not None
            test_y = test_y[[-1], :, :]
            test_y_dates = test_y_dates[[-1], :]

        data_df = data_df.iloc[: -self.ts_lookahead]
        (t_test_X, _, _, _, t_test_X_dates) = self.dl_preprocess_data(
            data_df.iloc[-self.ts_lookback :], predict=predict
        )

        data = data_df[predict]
        data = data.append(data_df[predict + "_forecast"].iloc[-self.ts_lookback :]).values
        wp5 = pywt.wavedec(data=data, wavelet=self.wavelet, mode=self.mode, level=self.level)
        N = data.shape[0]

        for i in range(1, self.level + 1):
            rd.append(
                pywt.waverec(wp5[:-i] + [None] * i, wavelet=self.wavelet, mode=self.mode)[: N - 24]
            )

        test_X.append(t_test_X[[-1], :, :])
        test_X_dates.append(t_test_X_dates[[-1], :])
        wpt_df = data_df[[]].copy()

        for i in range(0, self.level):
            wpt_df[predict] = rd[i]

            (t_test_X, _, _, _, t_test_X_dates) = self.dl_preprocess_data(
                wpt_df.iloc[-self.ts_lookback :], predict=predict
            )

            test_X.append(t_test_X[[-1], :, :])
            test_X_dates.append(t_test_X_dates)

        return test_X, test_y, test_X_dates, test_y_dates
