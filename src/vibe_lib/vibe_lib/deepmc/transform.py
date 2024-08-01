# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch.nn as nn
from einops import rearrange
from torch import Tensor

from .helpers import attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.d_head = d_model // self.num_heads
        self.scale = self.d_head**-0.5

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)

        self.dense = nn.Linear(d_model, d_model)

    def forward(self, v: Tensor, k: Tensor, q: Tensor, mask: Tensor):
        # (batch_size, seq_len, d_model)
        q = self.wq(q)
        k = self.wq(k)
        v = self.wq(v)

        # (batch_size, num_heads, seq_len_q, depth)
        q, k, v = (rearrange(x, "b l (h d) -> (b h) l d", h=self.num_heads) for x in (q, k, v))

        q *= self.scale
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention = attn(q, k, v, mask)

        concat_attention = rearrange(scaled_attention, "(b h) l d -> b l (h d)", h=self.num_heads)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output
