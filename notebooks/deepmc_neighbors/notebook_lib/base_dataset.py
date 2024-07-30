from math import cos, sin
from typing import Any, Dict, List, Union

import geopy.distance
import numpy as np
import pandas as pd
import torch
import torch.utils
import torch.utils.data
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import IterableDataset


class GNNDataset:
    def __init__(
        self,
        data: pd.DataFrame,
        scaler_input: StandardScaler,
        scaler_label: StandardScaler,
        neighbor_station: Dict[str, Any],
        infer_station: str,
        forecast_hours: int,
        label_column_index: Union[int, None],
        forecast_step: int = 0,
        device_count: int = torch.cuda.device_count(),
    ):
        super().__init__()
        self.data = data
        self.forecast_step = forecast_step
        self.device_count = device_count
        self.scaler_input = scaler_input
        self.scaler_label = scaler_label
        self.neighbor_stations = neighbor_station
        self.stations_count = len(self.neighbor_stations["stations"])
        self.infer_station = infer_station
        self.forecast_hours = forecast_hours
        self.label_column_index = label_column_index
        self.load_nodes()
        self.load_edges()

    def load_node_labels(self, data: pd.DataFrame):
        if "labels" not in data.columns:
            return data

        node_labels = data["labels"].to_numpy()
        node_labels = node_labels.reshape(-1)[
            : int(len(data.index.get_level_values(0)) / self.node_num) * self.node_num * 1
        ]

        self.node_labels = torch.from_numpy(
            node_labels.reshape(
                int(len(data.index.get_level_values(0)) / self.node_num),
                self.node_num,
                1,
            ).astype("float32")
        )
        data.drop(columns=["labels"], inplace=True)
        return data

    def load_nodes(self):
        data = self.node_feature_selection(self.data)
        data["timestamp"] = [pd.Timestamp(a).replace(tzinfo=None) for a in data["timestamp"]]
        data = data.rename(columns={"station": "Node"})
        self.node_names = data["Node"].unique().astype(str)
        self.node_num = len(self.node_names)
        data.set_index(["timestamp", "Node"], inplace=True)
        data = self.load_node_labels(data)
        data.drop(columns=["forecast_step"], inplace=True)

        # Set node variables
        self.lookback_indices = list(range(self.forecast_hours))
        self.target_idx = self.forecast_step
        self.timestamps = data.index.get_level_values(0).unique()
        self.infer_station_index = next(
            (i for i, a in enumerate(self.node_names) if a == self.infer_station), None
        )
        self.node_feas = list(data.columns)
        self.node_fea_dim = len(self.node_feas)
        node_vals = data.values.reshape(-1)[
            : int(len(data.index.get_level_values(0)) / self.node_num)
            * self.node_num
            * self.node_fea_dim
        ]

        self.node_data = torch.from_numpy(
            node_vals.reshape(
                int(len(data.index.get_level_values(0)) / self.node_num),
                self.node_num,
                self.node_fea_dim,
            ).astype("float32")
        )

        self.timestamps = self.timestamps[: self.node_data.shape[0]]

    def get_from_to_nodes(self, neighbor_stations: Dict[str, Any]):
        from_node = []
        to_node = []
        for s in neighbor_stations["stations"]:
            for c in self.neighbor_stations["stations"]:
                if s != c and s != self.infer_station:
                    from_node.append(s)
                    to_node.append(c)
        return from_node, to_node

    def get_edges(self, neighbor_stations: Dict[str, Any]):
        from_node, to_node = self.get_from_to_nodes(neighbor_stations)

        coords = neighbor_stations["long_lat"]
        edges = zip(from_node, to_node)
        distances = []
        turbine_dir_x = []
        turbine_dir_y = []

        for edge in edges:
            coord_1 = coords[edge[0]][::-1]
            coord_2 = coords[edge[1]][::-1]
            distances.append(geopy.distance.geodesic(coord_1, coord_2).km)
            x1, y1 = coord_1
            x2, y2 = coord_2
            turbine_dir_x.append(cos(x1) * sin(y1 - y2))
            turbine_dir_y.append(cos(x2) * sin(x1) - sin(x2) * cos(x1) * cos(y1 - y2))

        data = {
            "from_node": from_node,
            "to_node": to_node,
            "distance": distances,
            "dir_x": turbine_dir_x,
            "dir_y": turbine_dir_y,
        }
        return data

    def load_edges(self):
        data = self.get_edges(self.neighbor_stations)
        data = pd.DataFrame(data)
        data["to_node"] = data["to_node"]
        data["from_node"] = data["from_node"]
        data["edge"] = data.apply(lambda x: "{}->{}".format(x["from_node"], x["to_node"]), axis=1)
        data.loc[:, "distance"] = 1 / data.loc[:, "distance"]
        data.drop(columns=["from_node", "to_node"], inplace=True)
        edge_names = sorted(data["edge"].unique())
        node2id = dict(zip(self.node_names, range(len(self.node_names))))
        edge_index = [
            [node2id[src_node], node2id[tgt_node]]
            for src_node, tgt_node in [edge.split("->") for edge in edge_names]
        ]

        edge_df = data[["distance", "edge"]].set_index(["edge"])
        self.edge_names = edge_names
        self.edge_feas = list(edge_df.columns)
        self.edge_index = torch.LongTensor(edge_index)
        self.edge_num = len(self.edge_names)

        self.edge_fea_dim = len(self.edge_feas)
        self.edge_data = torch.from_numpy(
            edge_df.values.reshape(
                self.edge_num,
                self.edge_fea_dim,
            ).astype("float32")
        )

    def node_feature_selection(self, df_node: pd.DataFrame):
        df_node = df_node.sort_values(["timestamp", "forecast_step", "station"])
        scaled_input_array = self.scaler_input.transform(
            df_node.to_numpy()[:, 0 : self.forecast_hours]
        )
        df_node.iloc[:, 0 : self.forecast_hours] = scaled_input_array  # type: ignore

        if self.label_column_index is not None:
            scaled_label = self.scaler_label.transform(
                np.expand_dims(df_node.to_numpy()[:, self.label_column_index], axis=-1)
            )
            df_node.iloc[:, self.label_column_index] = scaled_label  # type: ignore
        return df_node


class BatchSampler(IterableDataset):  # type: ignore
    def __init__(
        self,
        dataset: GNNDataset,
        batch_size: int,
        lookahead_horizon: int,
        lookback_horizon: int,
        device: Union[str, torch.device],
        random: bool = True,
        noise_parameters: Dict[str, Any] = {},
        use_edge_weights: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.device_count = dataset.device_count
        self.random = random
        self.lookahead_horizon = lookahead_horizon
        self.lookback_horizon = lookback_horizon
        self.device = device
        self.noise_parameters = noise_parameters
        self.use_edge_weights = use_edge_weights
        self.stations_count = dataset.stations_count

    def get_forecast_indices(self):
        forecast_indices = list(range(len(self.dataset.timestamps)))
        if self.random:
            np.random.seed()
            np.random.shuffle(forecast_indices)

        return forecast_indices

    def get_batch_edge_index(self, cur_batch_size: int, num_devices: int):
        edge_num = self.dataset.edge_num
        if num_devices == 0:
            num_devices = 1

        batch_size_each_device = int(cur_batch_size / num_devices)

        # Reshape edge_index to [batch_size, 2, edge_num]
        self.edge_index = torch.cat(
            batch_size_each_device * [self.dataset.edge_index]  # type: ignore
        ).reshape(  # type: ignore
            batch_size_each_device, 2, edge_num
        )

        # Add offset to edge_index
        offset = torch.arange(
            0, batch_size_each_device * self.dataset.node_num, self.dataset.node_num
        ).view(-1, 1, 1)
        self.edge_index = self.edge_index + offset
        self.edge_index = torch.cat(num_devices * [self.edge_index]).reshape(
            cur_batch_size, 2, edge_num
        )

    def get_batch_edge_data(self, cur_batch_size: int, num_devices: int):
        edge_num = self.dataset.edge_num
        if num_devices == 0:
            num_devices = 1
        batch_size_each_device = int(cur_batch_size / num_devices)

        # Reshape edge_index to [batch_size, 2, edge_num]
        self.edge_data = torch.cat(batch_size_each_device * [self.dataset.edge_data]).reshape(
            batch_size_each_device, self.dataset.edge_fea_dim, edge_num
        )  # batch_size, edge_in_fea_dim, num_edges
        # Add offset to edge_index
        offset = torch.arange(
            0, batch_size_each_device * self.dataset.node_num, self.dataset.node_num
        ).view(-1, 1, 1)
        self.edge_data = self.edge_data + offset  # [batch_size, edge_node_dim, num_edges]

        self.edge_data = torch.cat(num_devices * [self.edge_data]).reshape(
            cur_batch_size, self.dataset.edge_fea_dim, edge_num
        )

    def generate(self):
        total_forecast_indices = self.get_forecast_indices()
        num_batches = (len(total_forecast_indices) // (self.batch_size)) + (
            len(total_forecast_indices) % self.batch_size != 0
        )

        for batch_id in range(num_batches):
            lookback_indices = []
            batch_id_s = batch_id * self.batch_size
            batch_id_e = batch_id_s + self.batch_size
            forecast_indices = total_forecast_indices[batch_id_s:batch_id_e]
            cur_batch_size = len(forecast_indices)
            lookback_indices = forecast_indices

            # Collect meta data
            forecast_timestamps = [self.dataset.timestamps[i] for i in forecast_indices]

            # Collect node-level time series
            node_lookback = (
                self.dataset.node_data[lookback_indices]
                .reshape(cur_batch_size, 1, self.dataset.node_num, self.dataset.node_fea_dim)
                .transpose(1, 2)
                .contiguous()
            )

            if self.dataset.label_column_index is not None:
                # Collect node-level time series
                node_lookback_labels = (
                    self.dataset.node_labels[lookback_indices]
                    .reshape(cur_batch_size, 1, self.dataset.node_num, 1)
                    .transpose(1, 2)
                    .contiguous()
                )
            else:
                node_lookback_labels = None

            self.get_batch_edge_index(cur_batch_size, self.device_count)
            self.get_batch_edge_data(cur_batch_size, self.device_count)

            batch = self.get_output(node_lookback, node_lookback_labels, forecast_timestamps)

            yield batch

    def get_output(
        self,
        node_lookback: Tensor,
        node_lookback_labels: Union[Tensor, None],
        forecast_timestamps: List[str],
    ):
        if self.use_edge_weights:
            self.edge_data = torch.squeeze(self.edge_data.reshape(-1, 1))

        self.edge_index = self.edge_index.permute(1, 0, 2).contiguous().view(2, -1)
        # node_lookahead not implemented
        # when we get it in the future, we will implement it
        batch = {}
        batch["node_data"] = node_lookback[:, :, :, :]
        batch["edge_index"] = self.edge_index
        batch["edge_data"] = self.edge_data
        batch["forecast_timestamps"] = forecast_timestamps

        if node_lookback_labels is not None:
            batch["node_labels"] = node_lookback_labels

        return list(batch.values())

    def __iter__(self):
        return iter(self.generate())
