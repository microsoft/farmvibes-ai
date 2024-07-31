import os
import pickle
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from vibe_notebook.deepmc.utils import transform_to_array

from .base_dataset import BatchSampler, GNNDataset
from .base_modules import BatchTGCNTrain


def build_scaler(train_embeddings: pd.DataFrame, forecast_hours: int) -> StandardScaler:
    train_data_scaler = StandardScaler()
    train_data_scaler.fit(train_embeddings.to_numpy()[:, :forecast_hours])
    return train_data_scaler


def build_scaler_label(
    train_embeddings: pd.DataFrame, labels_column: str
) -> Tuple[StandardScaler, int]:
    index = -1
    for i, column in enumerate(train_embeddings.columns):
        if column == labels_column:
            index = i

    if index == -1:
        raise ValueError(f"Labels column '{labels_column}' not found")

    train_label_scaler = StandardScaler()
    train_label_scaler.fit(np.expand_dims(train_embeddings.to_numpy()[:, index], axis=-1))
    return train_label_scaler, index


def get_batch_sample(
    train_dataset: GNNDataset,
    test_dataset: GNNDataset,
    batch_size: int,
    lookahead_horizon: int,
    lookback_horizon: int,
    device: torch.device,
    use_edge_weights: bool,
) -> Tuple[BatchSampler, BatchSampler]:
    train_sampler = BatchSampler(
        dataset=train_dataset,
        batch_size=batch_size,
        lookahead_horizon=lookahead_horizon,
        lookback_horizon=lookback_horizon,
        device=device,
        random=False,
        use_edge_weights=use_edge_weights,
    )

    test_sampler = BatchSampler(
        dataset=test_dataset,
        batch_size=batch_size,
        lookahead_horizon=lookahead_horizon,
        lookback_horizon=lookback_horizon,
        device=device,
        random=False,
        use_edge_weights=use_edge_weights,
    )

    return (train_sampler, test_sampler)


def train_test_dataset(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    step: int,
    neighbors_station: Dict[str, Any],
    scaler_data: StandardScaler,
    scaler_label: StandardScaler,
    infer_station: str,
    labels_column_index: int,
) -> Tuple[GNNDataset, GNNDataset]:
    train_dataset = GNNDataset(
        train_data,
        forecast_step=step,
        scaler_input=scaler_data,
        scaler_label=scaler_label,
        neighbor_station=neighbors_station,
        forecast_hours=24,
        infer_station=infer_station,
        label_column_index=labels_column_index,
    )

    test_dataset = GNNDataset(
        test_data,
        forecast_step=step,
        scaler_input=scaler_data,
        scaler_label=scaler_label,
        neighbor_station=neighbors_station,
        forecast_hours=24,
        infer_station=infer_station,
        label_column_index=labels_column_index,
    )

    return (train_dataset, test_dataset)


def problem_params(
    dataset: GNNDataset,
    batch_size: int,
    lookback_horizon: int,
    lookahead_horizon: int,
    use_edge_weights: bool,
    use_dropout: bool,
    hidden_dim: int,
    forecast_hours: int,
) -> Dict[str, Any]:
    problem_params = {
        "lookback_horizon": lookback_horizon,
        "lookahead_horizon": lookahead_horizon,
        "node_num": dataset.node_num,
        "node_in_fea_dim": dataset.node_fea_dim,
        "node_out_fea_dim": 1,
        "edge_in_fea_dim": dataset.edge_fea_dim,
        "edge_out_fea_dim": 1,
        "edge_num": dataset.edge_num,
        "use_edge_weights": use_edge_weights,
        "day_em_dim": 1,
        "hour_em_dim": 1,
        "period": 5,  # for attention model
        "batch_size": batch_size,
        "use_dropout": use_dropout,
        "hidden_dim": hidden_dim,
        "device_count": torch.cuda.device_count(),
        "lookback_indices": list(range(forecast_hours)),
    }

    return problem_params


def export_to_onnx(
    file_path: str,
    model: BatchTGCNTrain,
    inputs: DataLoader,  # type: ignore
    use_edge_weights: bool,
    edge_num: int,
    number_of_stations: int,
):
    data = next(iter(inputs))
    node_data, edge_index, edge_data, _ = get_batch(data, use_edge_weights)
    data = {
        "node_data": node_data[:number_of_stations],
        "edge_index": edge_index[:, : (edge_num * number_of_stations)],
        "edge_data": edge_data[: (edge_num * number_of_stations)],
    }
    keys = list(data.keys())
    batch_axes = {keys[i]: {0: "batch_size"} for i in range(len(keys))}
    onnx_output_path = os.path.join(file_path, "model_output.onnx")
    if os.path.exists(onnx_output_path):
        os.remove(onnx_output_path)

    # Export the model
    torch.onnx.export(
        model,
        list(data.values()),  # type: ignore
        onnx_output_path,
        input_names=list(batch_axes.keys()),
        dynamic_axes=batch_axes,
        opset_version=16,
    )


def write_to_file(output_file: str, data: List[Any]):
    with open(output_file, "wb") as f:
        pickle.dump(data, f)


def get_file(file_path: str) -> List[Any]:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    else:
        raise Exception(f"File {file_path} not found")


def get_batch(batch: Union[Tensor, List[Tensor], TensorDataset], use_edge_weights: bool):
    if type(batch) == TensorDataset:
        batch = batch[:]
    node_data = batch[0]
    edge_index = batch[1]
    # considered for training
    # skipped during inference
    if len(batch) == 5:
        node_labels = batch[4]
    else:
        node_labels = torch.tensor([])

    if use_edge_weights:
        edge_data = batch[2]
    else:
        edge_data = torch.tensor([])
    return node_data, edge_index, edge_data, node_labels


def smooth(y: List[float], box_pts: int):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


def get_split_data(split_data: NDArray[Any], timestamps: NDArray[Any], split_at_index: int):
    split_by_index = []
    for i in range(split_at_index):
        data_at_index = split_data[i::split_at_index][:, i]
        timestamp_at_index = timestamps[i::split_at_index]
        split_by_index.append(
            pd.DataFrame(zip(timestamp_at_index, data_at_index), columns=["timestamp", "label"])
        )

    split_data_df = pd.concat(split_by_index, axis=0, ignore_index=True)
    split_data_df["timestamp"] = pd.to_datetime(split_data_df["timestamp"])
    split_data_df = split_data_df.sort_values(by="timestamp")

    return np.array(split_data_df["label"].values)


def preprocess_transform(
    mix_data_yhat: NDArray[Any],
    inference_hours: int,
    dates_list: NDArray[Any],
):
    init_start = 0
    data_list = []
    end = mix_data_yhat.shape[0]
    for i in range(init_start, end, inference_hours):
        for j in range(inference_hours):
            data_list.append(mix_data_yhat[i, 0, j])

    mix_data_yhat = transform_to_array(np.array(data_list))[: mix_data_yhat.shape[0]]
    dates_list = dates_list[: mix_data_yhat.shape[0]]
    return mix_data_yhat, dates_list
