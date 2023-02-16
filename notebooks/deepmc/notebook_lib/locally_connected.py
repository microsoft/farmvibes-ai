from typing import Tuple, Union

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from unfoldNd.utils import _get_conv, _get_kernel_size_numel, _tuple


def _make_weight(in_channels, kernel_size, device, dtype):
    """Create one-hot convolution kernel. ``kernel_size`` must be an ``N``-tuple.
    Details:
        Let ``T`` denote the one-hot weight, then
        ``T[c * i, 0, j] = δᵢⱼ ∀ c = 1, ... C_in``
        (``j`` is a group index of the ``Kᵢ``).
        This can be done by building diagonals ``D[i, j] = δᵢⱼ``, reshaping
        them into ``[∏ᵢ Kᵢ, 1, K]``, and repeat them ``C_in`` times along the
        leading dimension.
    Returns:
        torch.Tensor : A tensor of shape ``[ C_in * ∏ᵢ Kᵢ, 1, K]`` where
            ``K = (K₁, K₂, ..., Kₙ)`` is the kernel size. Filter groups are
            one-hot such that they effectively extract one element of the patch
            the kernel currently overlaps with.
    """
    kernel_size_numel = _get_kernel_size_numel(kernel_size)
    repeat = [in_channels, 1] + [1 for _ in kernel_size]
    return (
        torch.eye(kernel_size_numel, device=device, dtype=dtype)
        .reshape((kernel_size_numel, 1, *kernel_size))
        .repeat(*repeat)
    )


class Unfold1d(torch.nn.Module):
    """Extracts sliding local blocks from a batched input tensor. Also known as im2col.
    PyTorch module that accepts 3d, 4d, and 5d tensors. Acts like ``torch.nn.Unfold``
    for a 4d input. Uses one-hot convolution under the hood.
    See docs at https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html.
    """

    def __init__(
        self,
        in_channels,
        kernel_size,
        dilation=1,
        padding=0,
        stride=1,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride
        # get convolution operation
        batch_size_and_in_channels_dims = 2
        N = 1
        self._conv = _get_conv(N)
        # prepare one-hot convolution kernel
        kernel_size = _tuple(kernel_size, N)
        self.kernel_size_numel = _get_kernel_size_numel(kernel_size)
        self.weight = _make_weight(in_channels, kernel_size, device, dtype)

    def forward(self, input):
        batch_size = input.shape[0]
        unfold = self._conv(
            input,
            self.weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.in_channels,
        )
        return unfold.reshape(batch_size, self.in_channels * self.kernel_size_numel, -1)


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
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_len = seq_len
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        out_seq_len = (
            seq_len + sum(self.padding) - (kernel_size - 1) - 1
        ) // stride + 1
        self.unfold = Unfold1d(self.in_channels, self.kernel_size, stride=stride)
        self.weight = nn.Parameter(
            torch.empty(
                (in_channels, out_channels, kernel_size, out_seq_len),
                device=device,
                dtype=dtype,
            )
        )
        if bias:
            self.bias = nn.Parameter((torch.empty(out_channels, out_seq_len)))
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
        x = self.unfold(x)
        x = rearrange(x, "b (i k) l -> b i l k", i=self.in_channels)
        x = torch.einsum("b i l k, i o k l -> bol", x, self.weight)
        if self.bias is not None:
            x = x + self.bias
        return x
