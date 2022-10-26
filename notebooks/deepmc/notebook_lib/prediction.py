import pickle
from datetime import datetime, timedelta
from glob import glob
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model, load_model

from .preprocess import Preprocess

tf.get_logger().setLevel("CRITICAL")


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
    ):
        self.total_models = total_models
        self.ts_lookahead = total_models
        self.feed_interval = feed_interval_minutes
        self.date_attribute = date_attribute
        self.root_path = root_path
        self.model_path = self.root_path + f"{station_name}/model_%s/"
        self.post_model_path = self.model_path + "post/"
        self.ts_lookback = ts_lookback
        self.chunk_size = chunk_size
        self.wavelet = wavelet
        self.mode = mode
        self.level = level
        self.data_export_path = data_export_path

        self.predicts = predicts
        self.models = self.extract_weights()
        self.post_models = self.extract_weights_post()

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
            model=self.models,
            post_model=self.post_models,
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

        df_in = input_df[
            (input_df.index > (start_datetime - timedelta(hours=(self.chunk_size))))
        ]

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
                model=self.models[predict],
                post_model=self.post_models[predict],
                df_in=input_order_df,
                df_out=y_datetime_out,
                predict=predict,
            )

            df_all_predict = pd.concat([df_all_predict, df_out], axis=1)

        df_all_predict = df_all_predict.loc[:, ~df_all_predict.columns.duplicated()]
        return df_all_predict

    def run_individual_predict(
        self,
        model: Model,
        post_model: Model,
        df_in: pd.DataFrame,
        predict: str,
    ):
        df_predict = pd.DataFrame(columns=[predict, self.date_attribute])
        interval = self.feed_interval
        start_date = df_in.index[-1]

        with open(self.data_export_path % predict, "rb") as f:
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
        )

        test_X = preprocess.wavelet_transform_predict(df_in=df_in, predict=predict)

        time_arr = []
        post_yhat = np.empty([1, self.ts_lookahead, self.ts_lookahead])

        for idx in range(0, self.total_models):
            model.load_weights(self.model_path % (predict, str(idx)))

            out_x = model.predict(test_X, verbose="1")[:, :, 0]

            post_model.load_weights(self.post_model_path % (predict, str(idx)))

            out_x = preprocess.dl_preprocess_data(pd.DataFrame(out_x), predict=predict)[
                0
            ]
            post_yhat[:, :, idx] = post_model.predict(out_x, verbose="1")

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

        yhat_final = output_scaler.inverse_transform(
            np.expand_dims(yhat_final, axis=1)
        )[:, 0]
        df_predict = pd.DataFrame(
            data=list(zip(time_arr, yhat_final)), columns=["date", predict]
        )

        return df_predict

    def run_individual_predict_historical(
        self,
        model: Model,
        post_model: Model,
        df_in: pd.DataFrame,
        df_out: pd.DatetimeIndex,
        predict: str,
    ):
        df_predict = pd.DataFrame(columns=[predict, self.date_attribute])

        with open(self.data_export_path % predict, "rb") as f:
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
        )

        inshape = self.total_models
        test_X = preprocess.wavelet_transform_predict(df_in=df_in, predict=predict)

        post_yhat = np.empty(
            [test_X[0].shape[0] + 1 - inshape, inshape, self.total_models]
        )

        for idx in range(0, self.total_models):
            model.load_weights(self.model_path % (predict, str(idx)))

            out_x = model.predict(test_X, verbose="1")[:, 0, 0]

            post_model.load_weights(self.post_model_path % (predict, str(idx)))

            out_x = preprocess.dl_preprocess_data(pd.DataFrame(out_x), predict=predict)[
                0
            ]
            post_yhat[:, :, idx] = post_model.predict(out_x[:, :, 0], verbose="1")

        yhat_final = []
        init_start = 0

        end = post_yhat.shape[0]
        for i in range(init_start, end, self.total_models):
            for j in range(self.total_models):
                yhat_final.append(post_yhat[i, -1, j])

        yhat_final = output_scaler.inverse_transform(
            np.expand_dims(yhat_final, axis=1)
        )[:, 0]
        df_predict = pd.DataFrame(
            data=list(zip(df_out, yhat_final)), columns=["date", predict]
        )

        return df_predict

    def run_predict(
        self,
        model: Dict[str, Model],
        post_model: Dict[str, Model],
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
                model=model[predict],
                post_model=post_model[predict],
                df_in=input_order_df,
                predict=predict,
            )

            if df_predict is not None:
                if df_all_predict.empty:
                    df_all_predict[predict] = df_predict[predict]
                    df_all_predict[self.date_attribute] = df_predict[
                        self.date_attribute
                    ]
                else:
                    df_all_predict = pd.concat([df_all_predict, df_predict], axis=1)

        df_all_predict = df_all_predict.loc[:, ~df_all_predict.columns.duplicated()]
        df_out = pd.concat([df_out, df_all_predict], ignore_index=True)
        df_out.reset_index(drop=True, inplace=True)

        return df_out

    def extract_weights(self) -> Dict[str, Model]:
        """
        Save the weights to path provided if weights doesn't exist.
        Returns:
            returns model architecture
        """
        md_list = {}

        for predict in self.predicts:
            md = load_model(self.model_path % (predict, "0"))

            for i in range(self.total_models):
                root = self.model_path % (
                    predict,
                    str(i),
                )
                weights_path = root + "/weights"

                ls = glob(weights_path + "*")

                if len(ls) == 0:
                    md = load_model(self.model_path % (predict, str(i)))
                    md.save_weights(weights_path)

            md_list[predict] = md

        return md_list

    def extract_weights_post(self) -> Dict[str, Model]:
        """
        Save the weights to path provided if weights doesn't exist.
        Returns:
            returns model architecture
        """
        md_list = {}
        for predict in self.predicts:
            md = load_model(self.post_model_path % (predict, str(0)))

            for i in range(self.total_models):
                root = self.post_model_path % (predict, str(i))
                weights_path = root + "/weights"

                ls = glob(weights_path + "*")

                if len(ls) == 0:
                    md = load_model(self.post_model_path % (predict, str(i)))
                    md.save_weights(weights_path)

            md_list[predict] = md

        return md_list
