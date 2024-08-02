# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from numpy._typing import NDArray
from torch import Tensor
from torch.nn import Sequential


def get_angles(pos: NDArray[Any], i: NDArray[Any], d_model: int):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position: int, d_model: int) -> Tensor:
    angle_rads = get_angles(
        np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model
    )

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return torch.tensor(pos_encoding, dtype=torch.float32)


def attn(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> Tensor:
    sim = torch.einsum("b i d, b j d -> b i j", q, k)

    if mask is not None:
        max_neg_value = -torch.finfo(sim.dtype).max
        sim.masked_fill_(~mask, max_neg_value)

    attn = sim.softmax(dim=-1)
    out = torch.einsum("b i j, b j d -> b i d", attn, v)
    return out


def point_wise_feed_forward_network(in_features: int, out_features: int, d_ff: int) -> Sequential:
    return Sequential(
        nn.Linear(in_features, d_ff),
        nn.ReLU(),
        nn.Linear(d_ff, out_features),
    )
