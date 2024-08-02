# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import pickle
from datetime import datetime, timedelta
from typing import Any, List, cast

import numpy as np
import onnxruntime
import pandas as pd
from numpy.typing import NDArray

from vibe_notebook.deepmc.preprocess import Preprocess

MODEL_SUFFIX = "deepmc."


class InferenceWeather:
    def __init__(
        self,
        root_path: str,
        data_export_path: str,
        station_name: str,
        predicts: List[str],
        total_models: int = 24,
        feed_interval_minutes: int = 60,
        chunk_size: int = 528,
        ts_lookback: int = 24,
        date_attribute: str = "date",
        wavelet: str = "bior3.5",
        mode: str = "periodic",
        level: int = 5,
        relevant: bool = False,
    ):
        if relevant:
            self.relevant_text = "relevant"
        else:
            self.relevant_text = "not-relevant"

        self.total_models = total_models
        self.ts_lookahead = total_models
        self.feed_interval = feed_interval_minutes
        self.date_attribute = date_attribute
        self.root_path = root_path
        self.model_path = self.root_path + f"{station_name}/{self.relevant_text}/model_%s/"
        self.post_model_path = self.model_path + "post/"
        self.ts_lookback = ts_lookback
        self.chunk_size = chunk_size
        self.wavelet = wavelet
        self.mode = mode
        self.level = level
        self.data_export_path = data_export_path
        self.predicts = predicts
        self.onnx_file = os.path.join(self.model_path, "export.onnx")
        self.post_onnx_file = os.path.join(self.post_model_path, "export.onnx")
        self.relevant = relevant

    def inference(
        self,
        input_df: pd.DataFrame,
        start_datetime: datetime,
    ):
        cols = self.predicts.copy()
        cols.append(self.date_attribute)

        df_out = pd.DataFrame(columns=cols)

        df_in_1 = input_df[(input_df.index <= start_datetime)].tail(
            self.chunk_size + self.total_models
        )

        if df_in_1.shape[0] < self.chunk_size:
            raise RuntimeError(
                f"Forecast not done between {start_datetime.strftime('%m/%d/%Y, %H:%M:%S')},"
                " since number of input data points less than chunk size"
            )

        df_out = pd.DataFrame(columns=self.predicts)
        df_out = self.run_predict(
            df_in=df_in_1,
            df_out=df_out,
        )

        return df_out

    def inference_historical(
        self,
        input_df: pd.DataFrame,
        start_datetime: datetime,
        end_datetime: datetime,
    ):
        cols = self.predicts.copy()
        cols.append(self.date_attribute)

        df_out = pd.DataFrame(columns=cols)

        df_in = input_df[(input_df.index > (start_datetime - timedelta(hours=(self.chunk_size))))]

        if df_in.shape[0] < self.chunk_size:
            raise RuntimeError(
                f"Forecast not done between {start_datetime.strftime('%m/%d/%Y, %H:%M:%S')}"
                f" and {end_datetime.strftime('%m/%d/%Y, %H:%M:%S')}, since number of input data"
                " points less than chunk size",
            )

        y_datetime_out = input_df.index[
            (input_df.index >= start_datetime) & (input_df.index <= end_datetime)
        ]

        df_all_predict = pd.DataFrame()
        # df_out = pd.DataFrame(columns=self.predicts)
        for predict in self.predicts:
            input_order_df = df_in[df_in.columns].copy()
            out_feature_df = input_order_df[predict]
            input_order_df.drop(columns=[predict], inplace=True)
            input_order_df[predict] = out_feature_df

            df_out = self.run_individual_predict_historical(
                df_in=input_order_df,
                df_out=cast(pd.DatetimeIndex, y_datetime_out),
                predict=predict,
            )

            df_all_predict = pd.concat([df_all_predict, df_out], axis=1)

        df_all_predict = df_all_predict.loc[:, ~df_all_predict.columns.duplicated()]  # type: ignore
        return df_all_predict

    def predict(
        self, path: str, predict: str, model_idx: int, inputs: NDArray[Any], is_post: bool = False
    ):
        path = path % (predict, model_idx)
        session = onnxruntime.InferenceSession(path, None)

        if not is_post:
            in_ = {
                out.name: inputs[i].astype(np.float32) for i, out in enumerate(session.get_inputs())
            }
        else:
            in_ = {
                out.name: inputs.astype(np.float32) for i, out in enumerate(session.get_inputs())
            }

        result = session.run(None, input_feed=in_)[0]
        return result

    def run_individual_predict(
        self,
        df_in: pd.DataFrame,
        predict: str,
    ):
        df_predict = pd.DataFrame(columns=[predict, self.date_attribute])
        interval = self.feed_interval
        start_date: datetime = cast(datetime, df_in.index[-1])

        with open(self.data_export_path % (predict, self.relevant_text), "rb") as f:
            train_scaler, output_scaler = pickle.load(f)[4:6]

        preprocess = Preprocess(
            train_scaler=train_scaler,
            output_scaler=output_scaler,
            is_training=False,
            ts_lookahead=self.ts_lookahead,
            ts_lookback=self.ts_lookback,
            chunk_size=self.chunk_size,
            wavelet=self.wavelet,
            mode=self.mode,
            level=self.level,
            relevant=self.relevant,
        )

        test_X, _, _ = preprocess.wavelet_transform_predict(df_in=df_in, predict=predict)
        time_arr = []
        post_yhat = np.empty([1, self.ts_lookahead, self.ts_lookahead])
        for idx in range(0, self.total_models):
            out_x = self.predict(path=self.onnx_file, predict=predict, model_idx=idx, inputs=test_X)
            out_x = preprocess.dl_preprocess_data(pd.DataFrame(out_x), predict=predict)[0]
            out_x = out_x.transpose((0, 2, 1))
            out_x = self.predict(
                path=self.post_onnx_file, predict=predict, model_idx=idx, inputs=out_x
            )
            post_yhat[:, :, idx] = out_x
            hours_added = timedelta(minutes=interval)
            _date = start_date + hours_added
            time_arr.append(_date)
            interval += self.feed_interval

        yhat_final = []
        init_start = 0
        end = post_yhat.shape[0]

        for i in range(init_start, end, self.total_models):
            for j in range(self.total_models):
                yhat_final.append(post_yhat[i, -1, j])

        yhat_final = output_scaler.inverse_transform(np.expand_dims(yhat_final, axis=1))[:, 0]
        df_predict = pd.DataFrame(data=list(zip(time_arr, yhat_final)), columns=["date", predict])
        return df_predict

    def run_predict(
        self,
        df_in: pd.DataFrame,
        df_out: pd.DataFrame,
    ):
        df_all_predict = pd.DataFrame()
        df_in.sort_values(by=[self.date_attribute], ascending=True, inplace=True)

        for predict in self.predicts:
            input_order_df = df_in[df_in.columns].copy()
            out_feature_df = input_order_df[predict]
            input_order_df.drop(columns=[predict], inplace=True)
            input_order_df[predict] = out_feature_df

            df_predict = self.run_individual_predict(
                df_in=df_in,
                predict=predict,
            )

            if df_predict is not None:
                if df_all_predict.empty:
                    df_all_predict[predict] = df_predict[predict]
                    df_all_predict[self.date_attribute] = df_predict[self.date_attribute]
                else:
                    df_all_predict = pd.concat([df_all_predict, df_predict], axis=1)

        df_all_predict = df_all_predict.loc[:, list(~df_all_predict.columns.duplicated())]
        df_out = pd.concat([df_out, df_all_predict], ignore_index=True)
        df_out.reset_index(drop=True, inplace=True)
        return df_out

    def run_individual_predict_historical(
        self,
        df_in: pd.DataFrame,
        df_out: pd.DatetimeIndex,
        predict: str,
    ):
        df_predict = pd.DataFrame(columns=[predict, self.date_attribute])

        with open(self.data_export_path % (predict, self.relevant_text), "rb") as f:
            train_scaler, output_scaler = pickle.load(f)[4:6]

        preprocess = Preprocess(
            train_scaler=train_scaler,
            output_scaler=output_scaler,
            is_training=True,
            ts_lookahead=self.ts_lookahead,
            ts_lookback=self.ts_lookback,
            chunk_size=self.chunk_size,
            wavelet=self.wavelet,
            mode=self.mode,
            level=self.level,
            relevant=self.relevant,
        )

        inshape = self.total_models
        test_X, _, _ = preprocess.wavelet_transform_predict(df_in=df_in, predict=predict)
        post_yhat = np.empty([test_X[0].shape[0] + 1 - inshape, inshape, self.total_models])
        for idx in range(0, self.total_models):
            out_x = self.predict(path=self.onnx_file, predict=predict, model_idx=idx, inputs=test_X)
            out_x = preprocess.dl_preprocess_data(pd.DataFrame(out_x), predict=predict)[0]
            out_x = out_x[..., 0]

            out_x = self.predict(
                path=self.post_onnx_file,
                predict=predict,
                model_idx=idx,
                inputs=out_x,
                is_post=True,
            )

            post_yhat[:, :, idx] = out_x

        yhat_final = []
        init_start = 0

        end = post_yhat.shape[0]
        for i in range(init_start, end, self.total_models):
            for j in range(self.total_models):
                yhat_final.append(post_yhat[i, -1, j])

        yhat_final = output_scaler.inverse_transform(np.expand_dims(yhat_final, axis=1))[:, 0]
        df_predict = pd.DataFrame(data=list(zip(df_out, yhat_final)), columns=["date", predict])
        return df_predict

    def deepmc_preprocess(self, df_in: pd.DataFrame, predict: str):
        with open(self.data_export_path, "rb") as f:
            train_scaler, output_scaler = pickle.load(f)[4:6]

        preprocess = Preprocess(
            train_scaler=train_scaler,
            output_scaler=output_scaler,
            is_training=False,
            ts_lookahead=self.ts_lookahead,
            ts_lookback=self.ts_lookback,
            chunk_size=self.chunk_size,
            wavelet=self.wavelet,
            mode=self.mode,
            level=self.level,
            relevant=self.relevant,
        )

        test_x, test_x_dates, _ = preprocess.wavelet_transform_predict(df_in=df_in, predict=predict)

        return test_x, test_x_dates, train_scaler, output_scaler
