import os
import pickle
import shutil
from glob import glob
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from notebook_lib.models import DeepMCModel, DeepMCPostModel
from notebook_lib.modules import DeepMCPostTrain, DeepMCTrain
from numpy import ndarray
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from . import utils
from .preprocess import Preprocess

MODEL_SUFFIX = "deepmc."


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
        batch_size: int = 256,
        relevant: bool = False,
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
        self.batch_size = batch_size
        self.relevant = relevant

    def train_model(
        self,
        input_df: pd.DataFrame,
        start: int = 0,
        end: int = -1,
        epochs: int = 20,
    ):
        m = None
        end = self.total_models if end == -1 else end

        for out_feature in self.out_features:
            if not os.path.exists(self.path_to_station % out_feature):
                os.makedirs(self.path_to_station % out_feature, exist_ok=True)

            input_order_df = input_df[self.train_features].copy()
            out_feature_df = input_order_df[out_feature]
            input_order_df.drop(columns=[out_feature], inplace=True)
            input_order_df[out_feature] = out_feature_df

            # data preprocessing
            (train_scaler, output_scaler, train_df, test_df,) = utils.get_split_scaled_data(
                data=input_order_df, out_feature=out_feature, split_ratio=0.92
            )

            if os.path.exists(self.data_export_path % (out_feature, self.relevant_text)):
                with open(self.data_export_path % (out_feature, self.relevant_text), "rb") as f:
                    (
                        train_X,
                        train_y,
                        test_X,
                        test_y,
                        train_scaler,
                        output_scaler,
                    ) = pickle.load(f)

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
            else:
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

                (
                    train_X,
                    train_y,
                    test_X,
                    test_y,
                ) = self.preprocess.wavelet_transform_train(train_df, test_df, out_feature)

                with open(self.data_export_path % (out_feature, self.relevant_text), "wb") as f:
                    pickle.dump(
                        [train_X, train_y, test_X, test_y, train_scaler, output_scaler],
                        f,
                    )

            m = self.train_models(
                train_X=train_X,
                train_y=train_y,
                test_X=test_X,
                test_y=test_y,
                epochs=epochs,
                out_feature=out_feature,
                start=start,
                end=end,
            )

    def train_models(
        self,
        train_X: List[ndarray],
        train_y: List[ndarray],
        test_X: List[ndarray],
        test_y: List[ndarray],
        epochs: int,
        out_feature: str,
        start: int,
        end: int,
    ):
        first_channels = train_X[0].shape[2]
        rest_channels = train_X[1].shape[2]
        first_encoder_channels = 3
        rest_encoder_channels = (4, 8, 16)
        sequence_length = train_X[0].shape[1]
        kernel_size = 2
        num_inputs = len(train_X)

        for i in range(start, end):
            train_inputs = [
                torch.from_numpy(x.astype(np.float32)) for x in (*train_X, train_y[:, i])
            ]
            train_dataset = TensorDataset(*train_inputs)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

            val_inputs = [torch.from_numpy(x.astype(np.float32)) for x in (*test_X, test_y[:, i])]
            val_dataset = TensorDataset(*val_inputs)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

            m = DeepMCTrain(
                first_channels=first_channels,
                rest_channels=rest_channels,
                first_encoder_channels=first_encoder_channels,
                rest_encoder_channels=rest_encoder_channels,
                sequence_length=sequence_length,
                kernel_size=kernel_size,
                num_inputs=num_inputs,
            )

            model_path = self.model_path % (out_feature, str(i))

            if os.path.exists(model_path):
                shutil.rmtree(model_path, ignore_errors=True)

            os.makedirs(model_path, exist_ok=True)

            t_obj = pl.Trainer(
                logger=True,
                max_epochs=epochs,
                callbacks=[
                    LearningRateMonitor(),
                    ModelCheckpoint(
                        monitor="val_loss/total",
                        save_last=True,
                        dirpath=model_path,
                    ),
                ],
                num_processes=1,
            )

            t_obj.fit(m, train_loader, val_loader)

            self.export_to_onnx(file_path=model_path, model=m.deepmc, inputs=train_inputs)

            self.post_model(
                m.deepmc,
                train_X=train_X,
                train_y=train_y,
                test_X=test_X,
                test_y=test_y,
                out_feature=out_feature,
                model_index=i,
                epochs=epochs,
            )

    def export_to_onnx(
        self,
        file_path: str,
        model: Union[DeepMCModel, DeepMCPostModel],
        inputs: Union[List[Tensor], Tensor],
    ):
        batch_axes = {f"tensor.{str(i)}": {0: "batch_size"} for i in range(len(inputs))}

        onnx_output_path = os.path.join(file_path, "export.onnx")
        if os.path.exists(onnx_output_path):
            os.remove(onnx_output_path)

        # Export the model
        torch.onnx.export(
            model,
            inputs,
            onnx_output_path,
            input_names=list(batch_axes.keys()),
            dynamic_axes=batch_axes,
        )

    def get_dataloader(
        self, gt: ndarray, target: ndarray, o_feature: str
    ) -> Tuple[DataLoader, List[Tensor]]:
        o_x = self.preprocess.dl_preprocess_data(pd.DataFrame(gt), o_feature)[0][:, :, 0].astype(
            np.float32
        )

        o_y = self.preprocess.dl_preprocess_data(pd.DataFrame(target), o_feature)[0][
            :, :, 0
        ].astype(np.float32)

        o_inputs = [torch.from_numpy(x.astype(np.float32)) for x in (o_x, o_y)]
        o_dataset = TensorDataset(*o_inputs)
        o_loader = DataLoader(o_dataset, batch_size=self.batch_size, shuffle=True)
        return o_loader, o_inputs

    def post_model(
        self,
        m: DeepMCModel,
        train_X: List[ndarray],
        train_y: List[ndarray],
        test_X: List[ndarray],
        test_y: List[ndarray],
        out_feature: str,
        model_index: int,
        epochs: int,
    ):
        m.eval()

        xf = lambda a: [torch.from_numpy(x.astype(np.float32)) for x in a]

        train_yhat = m(xf(train_X)).detach().numpy()[:, 0]
        test_yhat = m(xf(test_X)).detach().numpy()[:, 0]
        post_model_path = self.post_model_path % (out_feature, str(model_index))

        if not os.path.exists(post_model_path):
            os.mkdir(post_model_path)

        train_dataloader, _ = self.get_dataloader(
            gt=train_y[:, model_index, 0], target=train_yhat, o_feature=out_feature
        )

        val_dataloader, val_inputs = self.get_dataloader(
            gt=test_y[:, model_index, 0], target=test_yhat, o_feature=out_feature
        )

        p_m = DeepMCPostTrain(first_in_features=self.total_models)

        t_obj = pl.Trainer(
            logger=True,
            max_epochs=epochs,
            callbacks=[
                LearningRateMonitor(),
                ModelCheckpoint(
                    monitor="val_loss/total",
                    save_last=True,
                    dirpath=post_model_path,
                ),
            ],
            num_processes=1,
        )

        t_obj.fit(p_m, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

        self.export_to_onnx(file_path=post_model_path, model=p_m.deepmc, inputs=torch.rand((1, 24)))
