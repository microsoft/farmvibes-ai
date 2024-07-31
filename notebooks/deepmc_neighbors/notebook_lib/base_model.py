from typing import Any, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Conv1d, Linear
from torch.utils.data import TensorDataset
from torch_geometric_temporal.nn.recurrent import TGCN

from .schema import BatchTGCNInputs


def get_batch(batch: Union[Tensor, List[Tensor], TensorDataset], use_edge_weights: bool):
    if isinstance(batch, TensorDataset):
        batch = batch[:]
    node_data = batch[0]
    edge_index = batch[1]
    # used for training
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


class BaseModule(nn.Module):
    def __init__(self, problem_params: Dict[str, Any]):
        super().__init__()
        self.batch_size = problem_params["batch_size"]
        self.lookback_horizon = problem_params["lookback_horizon"]
        self.lookahead_horizon = problem_params["lookahead_horizon"]

        # node
        self.num_nodes = problem_params["node_num"]
        self.node_in_fea_dim = problem_params["node_in_fea_dim"]
        self.node_out_fea_dim = problem_params["node_out_fea_dim"]
        self.node_input_dim = self.lookback_horizon * self.node_in_fea_dim
        self.node_output_dim = self.lookahead_horizon * self.node_out_fea_dim
        self.use_dropout = problem_params["use_dropout"]

        # edge
        self.edge_in_fea_dim = problem_params["edge_in_fea_dim"]
        self.edge_out_fea_dim = problem_params["edge_out_fea_dim"]
        self.edge_input_dim = self.lookback_horizon * self.edge_in_fea_dim
        self.edge_output_dim = self.lookahead_horizon * self.edge_out_fea_dim

        # Add day and hour embeddings
        self.day_em_dim = problem_params["day_em_dim"]
        self.hour_em_dim = problem_params["hour_em_dim"]
        # 7 days
        self.day_em = nn.Embedding(7, self.day_em_dim)
        # 24 hours
        self.hour_em = nn.Embedding(24, self.hour_em_dim)

        # GRU hidden him
        self.hidden_dim = problem_params["hidden_dim"]
        self.dropout = nn.Dropout2d(0.01)

        # linear layer
        self.linear1_node = nn.Linear(self.hidden_dim, self.node_output_dim)
        self.linear2_node = nn.Linear(self.node_in_fea_dim - 1, self.lookahead_horizon)
        self.ar = nn.Linear(self.lookback_horizon, self.lookahead_horizon)

        # Multi-dimensional edge attribute to one dimension
        self.edge_num = problem_params["edge_num"]
        self.use_edge_weights = problem_params["use_edge_weights"]
        self.linear_edge = nn.Linear(self.edge_in_fea_dim, 1)

    def weights_init(self, m: Union[Conv1d, Linear]):
        if isinstance(m, Conv1d) or isinstance(m, Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                nn.init.zeros_(m.bias.data)

    def initialize_weights(self):
        pass

    def forward(self, batch: Dict[str, Any]):
        pass


class BatchTGCN(BaseModule):
    def __init__(
        self,
        inputs: BatchTGCNInputs,
    ):
        super().__init__(inputs.dict())
        self.inputs = inputs.dict()
        self.decoder_in_fea_dim = 2
        self.node_in_fea_dim = self.node_in_fea_dim

        self.tgcn_cell_encoder = TGCN(self.node_in_fea_dim, self.hidden_dim)
        self.tgcn_cell_encoder1 = TGCN(self.node_in_fea_dim, self.hidden_dim)

        self.tgcn_cell_decoder = TGCN(self.decoder_in_fea_dim, self.hidden_dim)
        self.tgcn_cell_decoder1 = TGCN(self.decoder_in_fea_dim, self.hidden_dim)
        # stopping loop reference
        self.get_batch = get_batch
        self.dropout_encoder1 = nn.Dropout(0.05)

    def forward(self, inputs: Union[Tensor, List[Tensor]]):
        node_data, edge_index, edge_data, _ = get_batch(inputs, self.use_edge_weights)
        h = torch.empty
        self.edge_index = edge_index  # 2, num_edges
        # Process edge
        self.batch_size, self.num_nodes, _, _ = node_data.shape
        hh, e = self.process(node_data, edge_data)
        h = F.relu_(hh)
        h = self.linear1_node(h)
        h = h.reshape(self.batch_size, self.num_nodes, self.lookahead_horizon)  # type: ignore
        hh = hh.reshape(self.batch_size, self.num_nodes, self.hidden_dim)  # type: ignore
        return h, e, hh

    def get_hidden_embedding(
        self,
        horizon: int,
        x: Tensor,
        edge_weights: Union[Tensor, None],
    ) -> Tuple[Tensor, Union[Tensor, None]]:
        for i in range(horizon):
            indices_lookback = torch.tensor(self.inputs["lookback_indices"]).to(x.device)
            input = torch.index_select(x[:, :, i, :], 2, indices_lookback)
            input = input.reshape(self.batch_size * self.num_nodes, -1)
            h = self.tgcn_cell_encoder(input, self.edge_index, edge_weights)
            h = F.relu(h)
            h = self.dropout_encoder1(h)
        return h, edge_weights

    def process(
        self,
        node_data: Tensor,
        edge_data: Tensor,
    ) -> Tuple[Tensor, Union[Tensor, None]]:
        #  Add hour and day embedding
        horizon = self.lookback_horizon
        x = node_data

        if self.use_dropout:
            x = self.dropout(x)

        edge_weights = None
        if self.use_edge_weights:
            edge_weights = edge_data

        self.prev_input = x[:, :, -1, :horizon]
        h, e = self.get_hidden_embedding(horizon, x, edge_weights)
        return h, e
