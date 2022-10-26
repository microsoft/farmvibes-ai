from math import ceil

import numpy as np
import pandas as pd
import pywt
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

    def wavelet_transform_predict(self, df_in: pd.DataFrame, predict: str):
        i = 1
        start = i
        end = start
        t_test_X = []

        test_df = pd.DataFrame(
            self.train_scaler.transform(df_in), columns=df_in.columns, index=df_in.index
        )

        # convert input data to wavelet
        while end < test_df.shape[0]:
            start = i
            end = start + self.trunc
            i = i + 1
            chunkdataDF = test_df.iloc[start:end]

            test_uX, _ = self.convert_df_wavelet_input(data_df=chunkdataDF, predict=predict)

            t_test_X.append(test_uX)

        test_X = t_test_X[0].copy()

        for i in range(1, len(t_test_X)):
            for j in range(len(t_test_X[i])):
                test_X[j] = np.append(test_X[j], t_test_X[i][j], axis=0)

        return test_X

    def wavelet_transform_train(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, out_feature: str
    ):
        t_train_X, t_train_y = self.prepare_wavelet_data(train_df, out_feature=out_feature)

        t_test_X, t_test_y = self.prepare_wavelet_data(test_df, out_feature=out_feature)

        train_X = t_train_X[0].copy()
        train_y = t_train_y[0].copy()
        for i in range(1, len(t_train_X)):
            train_y = np.append(train_y, t_train_y[i], axis=0)
            for j in range(len(t_train_X[i])):
                train_X[j] = np.append(train_X[j], t_train_X[i][j], axis=0)

        test_X = t_test_X[0].copy()
        test_y = t_test_y[0].copy()
        for i in range(1, len(t_test_X)):
            test_y = np.append(test_y, t_test_y[i], axis=0)
            for j in range(len(t_test_X[i])):
                test_X[j] = np.append(test_X[j], t_test_X[i][j], axis=0)

        return train_X, train_y, test_X, test_y

    def prepare_wavelet_data(self, data_df: pd.DataFrame, out_feature: str):
        i = 0
        start = i * self.trunc
        end = start
        t_data_x = []
        t_data_y = []

        while end < data_df.shape[0]:
            start = i
            end = start + self.trunc
            i = i + 1
            o_data_df = data_df.iloc[start:end]

            data_ux, data_uy = self.convert_df_wavelet_input(
                o_data_df,
                predict=out_feature,
            )
            t_data_x.append(data_ux)
            t_data_y.append(data_uy)

        return t_data_x, t_data_y

    def dl_preprocess_data(
        self, df: pd.DataFrame, predict: str, per_split: float = 0.8, training: bool = False
    ):
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
            X, y = list(), list()
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
                # move along one time step
                in_start += 1

            X = np.array(X)
            y = np.array(y)

            if self.is_validation is True:
                n_train_split = ceil(len(data) * per_split)
                train_X, train_y = X[:n_train_split, :, :], y[:n_train_split, :, :]
                test_X, test_y = X[n_train_split:, :], y[n_train_split:, :]

                return train_X, train_y, test_X, test_y
            else:
                return X, y, None, None
        else:

            X = list()
            in_start = 0
            for _ in range(len(data) - n_in + 1):
                in_end = in_start + n_in
                if in_end <= len(data):
                    X.append(data[in_start:in_end, :])
                in_start += 1
            X = np.array(X)
        return X, None, None, None

    def convert_df_wavelet_input(self, data_df: pd.DataFrame, predict: str):

        level = self.level
        rd = list()
        N = data_df.shape[0]
        test_X = list()

        if self.is_training:
            test_y = self.dl_preprocess_data(
                data_df.iloc[-self.ts_lookback - self.ts_lookahead :],
                predict=predict,
                training=self.is_training,
            )[1]

            test_y = test_y[[-1], :, :]

            data_df = data_df.iloc[: -self.ts_lookahead]
        else:
            test_y = []

        wp5 = pywt.wavedec(data=data_df[predict], wavelet=self.wavelet, mode=self.mode, level=level)
        N = data_df.shape[0]
        for i in range(1, level + 1):
            rd.append(pywt.waverec(wp5[:-i] + [None] * i, wavelet=self.wavelet, mode=self.mode)[:N])

        t_test_X = self.dl_preprocess_data(data_df.iloc[-self.ts_lookback :], predict=predict)[0]

        test_X.append(t_test_X[[-1], :, :])
        wpt_df = data_df[[]].copy()

        for i in range(0, level):
            wpt_df[predict] = rd[i][:]

            t_test_X = self.dl_preprocess_data(wpt_df.iloc[-self.ts_lookback :], predict=predict)[0]

            test_X.append(t_test_X[[-1], :, :])

        return test_X, test_y
