import os
import shutil
import warnings
from datetime import datetime
from typing import Any, Dict, List, Union

import numpy as np
import onnxruntime
import pandas as pd
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from notebook_lib.embeddings import create_embeddings, create_embeddings_inference
from notebook_lib.post_deepmc import initialize_embeddings_preprocessing
from notebook_lib.post_deepmc_inference import (
    inference_embeddings_preprocessing,
    run_deepmc_inference,
)
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from vibe_notebook.deepmc.utils import calculate_KPI, get_csv_data

from .base_dataset import BatchSampler, GNNDataset
from .base_modules import BatchTGCNInputs, BatchTGCNTrain
from .data_utils import (
    build_scaler,
    build_scaler_label,
    export_to_onnx,
    get_batch,
    get_batch_sample,
    get_file,
    get_split_data,
    problem_params,
    smooth,
    train_test_dataset,
    write_to_file,
)


class MC_Neighbors:
    def __init__(
        self,
        root_dir: str,
        hidden_dim: int = 528,
        lookahead_horizon: int = 1,
        lookback_horizon: int = 1,
        learning_rate: float = 0.001,
        use_dropout: bool = False,
        use_edge_weights: bool = False,
        device_type: str = "cpu",  # cuda, cpu
        labels_column: str = "labels",
        weather_type: str = "temperature",
        model_type: str = "relevant",
    ):
        """
        Initialize the MC_Neighbors.

        :param root_dir: Path to trained model and preprocessed files.
        :param hidden_dim: Input dimension transforms it to linear layer.
        :param lookahead_horizon: Number of hours to lookahead.
        :param lookback_horizon: Number of hours to lookback.
        :param learning_rate: The learning rate of the model.
        :param use_dropout: True or False to use dropout layer for model training.
        :param use_edge_weights: True or False. If True consider spatial distance
            between stations for model training.
        :param device_type: The device type of the model.
        :param labels_column: The labels column of the dataset.
        :param weather_type: Purpose of trained model. It can be temperature or wind_speed etc.,.
        :param model_type: relevant or not-relevant.
        """
        self.weather_type = weather_type
        self.root_dir = root_dir
        self.lookahead_horizon = lookahead_horizon
        self.lookback_horizon = lookback_horizon
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.use_dropout = use_dropout
        self.use_edge_weights = use_edge_weights
        self.labels_column = labels_column
        self.device = torch.device(
            device_type if device_type == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.model_type = model_type

    def gnn_output_dir(self, infer_station: str):
        if self.use_edge_weights:
            edge_weights = "edge_weights"
        else:
            edge_weights = "no_edge_weights"
        return os.path.join(
            self.root_dir,
            infer_station,
            self.model_type,
            "gnn_models",
            edge_weights,
        )

    def gnn_preprocess_file(self, infer_station: str):
        output_dir = self.gnn_output_dir(infer_station)
        return os.path.join(output_dir, "pre_process_data_export.json")

    def run_train(
        self,
        train_embeddings: pd.DataFrame,
        test_embeddings: pd.DataFrame,
        neighbor_stations: List[Dict[str, Any]],
        infer_station: str,
        epochs: int,
        batch_size: int,
        forecast_hours: int,
    ) -> None:
        self.output_dir = self.gnn_output_dir(infer_station)
        stations = self.get_neighbor_stations(neighbor_stations)
        scaler_data = build_scaler(train_embeddings.copy(), forecast_hours)
        scaler_label, labels_column_index = build_scaler_label(
            train_embeddings.copy(), self.labels_column
        )
        data_export_path = self.gnn_preprocess_file(infer_station)
        if not os.path.exists(data_export_path):
            os.makedirs(os.path.dirname(data_export_path), exist_ok=True)
            write_to_file(data_export_path, data=[scaler_data, scaler_label, labels_column_index])

        self.initialize_train(
            train_embeddings,
            test_embeddings,
            stations,
            infer_station,
            epochs,
            batch_size,
            forecast_hours,
            scaler_data,
            scaler_label,
            labels_column_index,
        )

    def initialize_train(
        self,
        train_embeddings: pd.DataFrame,
        test_embeddings: pd.DataFrame,
        neighbors_station: Dict[str, Any],
        infer_station: str,
        epochs: int,
        batch_size: int,
        forecast_hours: int,
        scaler_data: StandardScaler,
        scaler_label: StandardScaler,
        labels_column_index: int,
    ):
        for step in range(forecast_hours):
            train_dataset, test_dataset = train_test_dataset(
                train_data=train_embeddings,
                test_data=test_embeddings,
                step=step,
                neighbors_station=neighbors_station,
                scaler_data=scaler_data,
                scaler_label=scaler_label,
                infer_station=infer_station,
                labels_column_index=labels_column_index,
            )

            train_sampler, test_sampler = get_batch_sample(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                batch_size=batch_size,
                lookahead_horizon=self.lookahead_horizon,
                lookback_horizon=self.lookback_horizon,
                device=self.device,
                use_edge_weights=self.use_edge_weights,
            )

            inputs = BatchTGCNInputs(
                **problem_params(
                    train_dataset,
                    batch_size,
                    self.lookback_horizon,
                    self.lookahead_horizon,
                    self.use_edge_weights,
                    self.use_dropout,
                    self.hidden_dim,
                    forecast_hours,
                )
            )
            model = BatchTGCNTrain(inputs, self.learning_rate)
            model.to(self.device)
            self.train_model(model, epochs, train_sampler, test_sampler, step)

    def train_model(
        self,
        model: BatchTGCNTrain,
        epochs: int,
        train_sampler: BatchSampler,
        test_sampler: BatchSampler,
        forecast_step: int,
    ):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        model_path = "{}/model_{}".format(self.output_dir, forecast_step)

        if os.path.exists(model_path):
            shutil.rmtree(model_path, ignore_errors=True)

        os.makedirs(model_path, exist_ok=True)

        # batch_size is set to None to avoid batch size in dataloader
        # batch_size is set when creating the sampler
        train_loader = DataLoader(train_sampler, batch_size=None, collate_fn=lambda x: x)
        val_loader = DataLoader(test_sampler, batch_size=None, collate_fn=lambda x: x)

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
        )
        t_obj.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        export_to_onnx(
            model_path,
            model,
            train_loader,
            self.use_edge_weights,
            train_sampler.dataset.edge_num,
            train_sampler.dataset.stations_count,
        )

    def run_inference(
        self,
        embeddings: pd.DataFrame,
        neighbors_station: List[Dict[str, Any]],
        infer_station: str,
        batch_size: int,
        forecast_hours: int,
    ):
        self.output_dir = self.gnn_output_dir(infer_station)
        stations = self.get_neighbor_stations(neighbors_station)
        scaler_data, scaler_label, labels_column_index = get_file(
            self.gnn_preprocess_file(infer_station)
        )

        pred_data = []
        for step in range(forecast_hours):
            dataset, sampler = self.get_infer_inputs(
                embeddings,
                stations,
                infer_station,
                batch_size,
                forecast_hours,
                step,
                None,
                scaler_data,
                scaler_label,
            )
            loader = DataLoader(sampler, batch_size=None, collate_fn=lambda x: x)
            for index, data in enumerate(loader):
                onnx_file_path = "{}/model_{}/model_output.onnx".format(self.output_dir, step)
                if data[0].shape[0] != batch_size:
                    warnings.warn(
                        f"""Data at step {step} batch index {index} is less than batch size.
                                It will be skipped from running inference."""
                    )
                    continue
                if step == 0:
                    results = np.zeros((batch_size, forecast_hours))
                    results[:, step] = self.inference(onnx_file_path, data)[
                        :, dataset.infer_station_index
                    ].squeeze()
                    pred_data.append(results)
                else:
                    pred_data[index][:, step] = self.inference(onnx_file_path, data)[
                        :, dataset.infer_station_index
                    ].squeeze()
        pred_data = np.concatenate(pred_data, axis=0)
        pred_data = scaler_data.inverse_transform(pred_data)
        timestamps = dataset.timestamps[: pred_data.shape[0]]
        pred_data = get_split_data(pred_data, timestamps, forecast_hours)  # type: ignore
        pred_data_df = pd.DataFrame(
            zip(pred_data, timestamps), columns=[self.weather_type, "timestamp"]
        )
        return pred_data_df

    def get_historical_data(self, data_path: str):
        historical_data_df = get_csv_data(data_path)
        historical_data_df.reset_index(inplace=True)
        historical_data_df.rename(columns={"date": "timestamp"}, inplace=True)
        return historical_data_df

    def get_hrrr_data(
        self,
        data_path: str,
    ):
        df_node = pd.read_csv(data_path, parse_dates=["date"])
        df_node.rename(columns={"date": "timestamp"}, inplace=True)
        return df_node

    def get_infer_inputs(
        self,
        embeddings: pd.DataFrame,
        neighbors_station: Dict[str, Any],
        infer_station: str,
        batch_size: int,
        forecast_hours: int,
        step: int,
        labels_column_index: Union[int, None],
        scaler_data: StandardScaler,
        scaler_label: StandardScaler,
    ):
        dataset = GNNDataset(
            embeddings,
            forecast_step=step,
            scaler_input=scaler_data,
            scaler_label=scaler_label,
            neighbor_station=neighbors_station,
            forecast_hours=forecast_hours,
            infer_station=infer_station,
            label_column_index=labels_column_index,
        )

        sampler = BatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            lookahead_horizon=self.lookahead_horizon,
            lookback_horizon=self.lookback_horizon,
            device=self.device,
            random=False,
            use_edge_weights=self.use_edge_weights,
        )

        return dataset, sampler

    def inference(self, onnx_file_path: str, data: torch.Tensor):
        session = onnxruntime.InferenceSession(onnx_file_path, None)
        node_data, edge_index, edge_data, _ = get_batch(data, self.use_edge_weights)

        inputs = {
            "node_data": node_data.numpy(),
            "edge_index": edge_index.numpy(),
            "edge_data": edge_data.numpy(),
        }

        inputs = {out.name: inputs[out.name] for i, out in enumerate(session.get_inputs())}
        results = session.run(None, input_feed=inputs)[0]
        return results

    def get_embeddings(
        self,
        inference_station: str,
        neighbor_stations: List[Dict[str, Any]],
        inference_hours: int,
        infer_forecast_data_path: str,
    ):
        initialize_embeddings_preprocessing(
            infer_station_name=inference_station,
            stations=neighbor_stations,
            root_path=self.root_dir,
            infer_forecast_data_path=infer_forecast_data_path,
            infer_interval=inference_hours,
            model_type=self.model_type,
        )

        df_train_embeddings, df_test_embeddings = create_embeddings(
            stations=neighbor_stations,
            inference_hours=inference_hours,
            root_path=self.root_dir,
            model_type=self.model_type,
        )

        return df_train_embeddings, df_test_embeddings

    def get_neighbor_stations(
        self,
        neighbor_stations: List[Dict[str, Any]],
    ):
        stations_connection = {}
        stations = []
        station_long_lat = {}
        for station in neighbor_stations:
            stations.append(station["name"])
            station_long_lat[station["name"]] = station["coordinates"]

        stations_connection["stations"] = stations
        stations_connection["long_lat"] = station_long_lat

        return stations_connection

    def filter_data(
        self,
        df_inference: pd.DataFrame,
        df_historical: pd.DataFrame,
        df_forecast: pd.DataFrame,
    ):
        start_date = df_inference["timestamp"].min()
        end_date = df_inference["timestamp"].max()

        df_historical = df_historical[df_historical.timestamp.between(start_date, end_date)]
        df_historical = df_historical[["timestamp", self.weather_type]]

        df_inference = df_inference[df_inference.timestamp.between(start_date, end_date)]
        df_inference = df_inference[["timestamp", self.weather_type]]

        df_forecast = df_forecast[df_forecast.timestamp.between(start_date, end_date)]
        df_forecast.rename(columns={"temperature_forecast": self.weather_type}, inplace=True)
        df_forecast = df_forecast[["timestamp", self.weather_type]]

        return df_inference, df_historical, df_forecast

    def view_plot(
        self,
        df_inference: pd.DataFrame,
        historical_data_path: str,
        hrrr_data_path: str,
    ):
        df_historical = self.get_historical_data(historical_data_path)
        df_forecast = self.get_hrrr_data(hrrr_data_path)

        df_inference, df_historical, df_forecast = self.filter_data(
            df_inference, df_historical, df_forecast
        )

        timestamps = df_inference["timestamp"]
        y_hat = list(df_inference[self.weather_type].values)
        y = list(df_historical[self.weather_type].values)
        hrrr_data_y = list(df_forecast[self.weather_type].values)

        plt.figure(figsize=(18, 6))
        plt.plot(timestamps, smooth(y_hat, 2), label="Predict")
        plt.plot(timestamps, y, label="Ground Truth")
        plt.plot(timestamps, hrrr_data_y, label="HRRR", linestyle="--")
        plt.title("Comparison Ground Truth Vs Inference Results Vs HRRR")
        plt.legend()

    def view_performance(
        self,
        df_inference: pd.DataFrame,
        historical_data_path: str,
        hrrr_data_path: str,
    ):
        df_historical = self.get_historical_data(historical_data_path)
        df_forecast = self.get_hrrr_data(hrrr_data_path)

        df_inference, df_historical, df_forecast = self.filter_data(
            df_inference, df_historical, df_forecast
        )

        y_hat = list(df_inference[self.weather_type].values)
        y = np.array(df_historical[self.weather_type].values)
        hrrr_data_y = list(df_forecast[self.weather_type].values)

        print("GNN ", self.weather_type)
        calculate_KPI(smooth(y_hat, 1), y)
        print("")
        print("Hrrr", self.weather_type)
        calculate_KPI(smooth(hrrr_data_y, 1), y)

    def get_embeddings_inference(
        self,
        inference_station: str,
        neighbor_stations: List[Dict[str, Any]],
        inference_hours: int,
        infer_forecast_data_path: str,
        out_features: List[str],
        historical_data_path: str,
        hrrr_datasets: Dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
        historical_dataset_featues: List[str] = ["humidity", "wind_speed", "temperature"],
        forecast_dataset_features: List[str] = [
            "humidity_forecast",
            "wind_speed_forecast",
            "temperature_forecast",
        ],
        frequency_hour: int = 1,
        number_of_hours: int = 24,
        weather_inference_type: str = "temperature",
    ):
        deepmc_results = run_deepmc_inference(
            self.root_dir,
            self.model_type,
            out_features,
            neighbor_stations,
            historical_data_path,
            hrrr_datasets,
            start_date,
            end_date,
            inference_station,
            historical_dataset_featues,
            forecast_dataset_features,
            frequency_hour,
            number_of_hours,
            weather_inference_type,
        )

        deepmc_post_results = inference_embeddings_preprocessing(
            infer_station_name=inference_station,
            stations=neighbor_stations,
            root_path=self.root_dir,
            infer_forecast_data_path=infer_forecast_data_path,
            infer_interval=inference_hours,
            model_type=self.model_type,
            deepmc_inference_results=deepmc_results,
        )

        df_embeddings = create_embeddings_inference(
            stations=neighbor_stations,
            inference_hours=inference_hours,
            deepmc_post_results=deepmc_post_results,
        )

        return df_embeddings
