# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.types import _dtype


class LocallyConnected1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        seq_len: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        bias: bool = True,
        device: Optional[str] = None,
        dtype: Optional[_dtype] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        out_seq_len = (seq_len + sum(self.padding) - (kernel_size - 1) - 1) // stride + 1
        self.weight = Parameter(
            torch.empty(
                (in_channels, out_channels, kernel_size, out_seq_len),  # type: ignore
                device=device,
                dtype=dtype,  # type: ignore
            )
        )

        if bias:
            self.bias = Parameter((torch.empty(out_channels, out_seq_len)))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # Do normal initialization for now, but can use something smarter
        nn.init.normal_(self.weight, std=0.1)
        if self.bias is not None:
            nn.init.normal_(self.bias, std=0.1)

    def forward(self, x: torch.Tensor):
        x = F.pad(x, self.padding)
        x = x.unfold(-1, self.kernel_size, self.stride)
        x = torch.einsum("b i l k, i o k l -> bol", x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x
