import os
import pickle
from typing import List

import pandas as pd
from keras.models import Model

from . import utils
from .models import deepmc_fit_model, moddeepmc_pred_model
from .post_models import fit_model, simple_mixture_model
from .preprocess import Preprocess


class ModelTrainWeather:
    def __init__(
        self,
        root_path: str,
        data_export_path: str,
        station_name: str,
        train_features: List[str],
        out_features: List[str],
        chunk_size: int = 528,
        ts_lookback: int = 24,
        total_models: int = 24,
        is_validation: bool = False,
        wavelet: str = "bior3.5",
        mode: str = "periodic",
        level: int = 5,
        relevant: bool = False,
        batch_size: int = 8,
    ):
        if relevant:
            self.relevant_text = "relevant"
        else:
            self.relevant_text = "not-relevant"

        self.total_models = total_models
        self.root_path = root_path
        self.data_export_path = data_export_path
        self.path_to_station = os.path.join(self.root_path, station_name, self.relevant_text, "")
        self.model_path = os.path.join(self.path_to_station, "model_%s", "")
        self.post_model_path = os.path.join(self.model_path, "post", "")
        self.train_features = train_features
        self.out_features = out_features
        self.ts_lookback = ts_lookback
        self.is_validation = is_validation
        self.ts_lookahead = total_models
        self.chunk_size = chunk_size
        self.wavelet = wavelet
        self.mode = mode
        self.level = level
        self.relevant = relevant
        self.batch_size = batch_size

    def train_model(
        self,
        input_df: pd.DataFrame,
        epochs: int = 20,
        server_mode: bool = True,
    ):

        for out_feature in self.out_features:
            if not os.path.exists(self.path_to_station % out_feature):
                os.makedirs(self.path_to_station % out_feature)

            input_order_df = input_df[self.train_features].copy()
            out_feature_df = input_order_df[out_feature]
            input_order_df.drop(columns=[out_feature], inplace=True)
            input_order_df[out_feature] = out_feature_df

            # data preprocessing
            (train_scaler, output_scaler, train_df, test_df,) = utils.get_split_scaled_data(
                data=input_order_df, out_feature=out_feature, split_ratio=0.92
            )

            if not os.path.exists(self.data_export_path % (out_feature, self.relevant_text)):
                self.preprocess = Preprocess(
                    train_scaler=train_scaler,
                    output_scaler=output_scaler,
                    is_training=True,
                    is_validation=self.is_validation,
                    ts_lookahead=self.ts_lookahead,
                    ts_lookback=self.ts_lookback,
                    chunk_size=self.chunk_size,
                    wavelet=self.wavelet,
                    mode=self.mode,
                    level=self.level,
                    relevant=self.relevant,
                )

                train_X, train_y, test_X, test_y = self.preprocess.wavelet_transform_train(
                    train_df, test_df, out_feature
                )

                with open(self.data_export_path % (out_feature, self.relevant_text), "wb") as f:
                    pickle.dump([train_X, train_y, test_X, test_y, train_scaler, output_scaler], f)
            else:
                with open(self.data_export_path % (out_feature, self.relevant_text), "rb") as f:
                    train_X, train_y, test_X, test_y, train_scaler, output_scaler = pickle.load(f)

                self.preprocess = Preprocess(
                    train_scaler=train_scaler,
                    output_scaler=output_scaler,
                    is_training=True,
                    is_validation=self.is_validation,
                    ts_lookahead=self.ts_lookahead,
                    ts_lookback=self.ts_lookback,
                    chunk_size=self.chunk_size,
                    wavelet=self.wavelet,
                    mode=self.mode,
                    level=self.level,
                    relevant=self.relevant,
                )

            # train model
            for model_idx in range(0, self.total_models):
                model = moddeepmc_pred_model(
                    train_X,
                    train_y[:, [model_idx], :],
                )

                model, _ = deepmc_fit_model(
                    model,
                    train_X,
                    train_y[:, [model_idx], :],
                    validation_data=(test_X, test_y[:, [model_idx], :]),
                    epochs=epochs,
                    server_mode=server_mode,
                    batch_size=self.batch_size,
                )

                model.save(self.model_path % (out_feature, str(model_idx)))
                model.save_weights(self.model_path % (out_feature, str(model_idx)) + "weights")
                self.post_model(model, train_X, train_y, test_X, test_y, out_feature, model_idx)

    def post_model(
        self,
        model: Model,
        train_X,
        train_y,
        test_X,
        test_y,
        out_feature: str,
        model_index: int,
    ):

        train_yhat = model.predict(train_X)[:, 0, 0]
        test_yhat = model.predict(test_X)[:, 0, 0]
        mix_train_X = self.preprocess.dl_preprocess_data(pd.DataFrame(train_yhat), out_feature)[0]

        mix_train_y = self.preprocess.dl_preprocess_data(
            pd.DataFrame(train_y[:, model_index, 0]), out_feature
        )[0]

        mix_test_X = self.preprocess.dl_preprocess_data(pd.DataFrame(test_yhat), out_feature)[0]

        mix_test_y = self.preprocess.dl_preprocess_data(
            pd.DataFrame(test_y[:, model_index, 0]), out_feature
        )[0]

        mix_model = simple_mixture_model(self.total_models)
        post_model, _ = fit_model(
            mix_model,
            mix_train_X[:, :, 0],
            mix_train_y[:, :, 0],
            mix_test_X[:, :, 0],
            mix_test_y[:, :, 0],
            batch_size=self.batch_size,
        )

        post_model_path = self.post_model_path % (out_feature, str(model_index))

        if not os.path.exists(post_model_path):
            os.makedirs(post_model_path)

        post_model.save(post_model_path)

        post_model_weights = post_model_path + "weights"
        post_model.save_weights(post_model_weights)
