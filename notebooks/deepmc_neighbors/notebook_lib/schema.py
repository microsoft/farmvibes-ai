from typing import List

from pydantic import BaseModel


class BatchTGCNInputs(BaseModel):
    lookback_horizon: int
    lookahead_horizon: int
    node_num: int
    node_in_fea_dim: int
    node_out_fea_dim: int
    edge_in_fea_dim: int
    edge_out_fea_dim: int
    edge_num: int
    use_edge_weights: bool
    day_em_dim: int
    hour_em_dim: int
    period: int
    batch_size: int
    use_dropout: bool
    hidden_dim: int
    device_count: int
    lookback_indices: List[int]
