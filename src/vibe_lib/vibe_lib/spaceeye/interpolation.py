# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Dict

import torch
import torch.nn as nn
from einops import rearrange, repeat

EPS = 1e-6


def generate_delta_matrix(dim: int):
    """
    The matrix returned calculates discrete forward differences (discrete derivative).
    delta * x returns a matrix with elements x[t+1]-x[t] with the last entry being 0.

    The matrix returned looks in general like this:
    delta = [ [-1,  1, 0, ...,  0, 0],
              [ 0, -1, 1, ...,  0, 0],
              ...
              [ 0,  0, 0, ..., -1, 1],
              [ 0,  0, 0, ...,  0, 0]]
    """
    d = torch.zeros((dim, dim), dtype=torch.float32)
    i = torch.arange(dim - 1)
    d[i, i] = -1
    d[i, i + 1] = 1
    return d


def masked_time_average(x: torch.Tensor, m: torch.Tensor):
    n = (x * m).sum(dim=2, keepdim=True)
    d = m.sum(dim=2, keepdim=True)
    return n / (d + EPS)


class DampedInterpolation(nn.Module):
    """
    This algorithm implements interpolation through minimizing an object function, namely:

       F(X) = sum_t || (X_t - S2_t) .* M_t ||_F^2 + alpha sum_t ||X_{t+1}-X_t||_F^2
            = || (X - S2) .* M ||_F^2 + alpha || Delta * X ||_F^2

    The gradient is
       F'(X) = 2 * M**2 .* (X-S2) + 2 * alpha * (Delta^T @ Delta) @ X
    We use || F'(X) ||_F^2 / (nb*nt*nx*ny) as a stoppping criteria for the algorithm.
    Note that M**2=M when M represents a 0/1 cloud-mask.
    In the case of cloud-probabilities it's more complex.

    Using algorithm from SpaceEye paper:
        X <== (I+alpha*Delta^T*Delta)^{-1} ((M.*S2)-(1-M).*X)

    Note that S2, X and M here are assumed to me (nb*nt) x (nx*ny) matrices, while the illumination
    calculation is done on nb x nt x nx x ny tensors.  (Of course we just use different views of the
    same tensors).

    """

    def __init__(
        self,
        num_bands: int,
        time_window: int,
        damping_factor: float = 0.1,
        tol: float = 1e-3,
        max_iter: int = 200,
        check_interval: int = 5,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.time_window = time_window
        self.damping_factor = damping_factor
        self.tol = tol
        self.max_iter = max_iter
        self.check_interval = check_interval
        assert self.damping_factor > 0
        d = generate_delta_matrix(self.time_window)
        self.delta = torch.kron(torch.eye(self.num_bands), d)
        self.w: torch.Tensor = torch.linalg.inv(
            torch.eye(self.time_window) + damping_factor * (d.T @ d)
        )

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        s2, m = inputs["S2"], inputs["cloud_label"] == 1
        x = s2.clone()
        m = m.to(x)
        m_: torch.Tensor = 1 - m
        pixel_avg = masked_time_average(x, m)
        x = x * m + pixel_avg * m_
        b, c, _, h, _ = s2.shape
        s2 = rearrange(s2, "b c t h w -> t (b c h w)").contiguous()
        x = rearrange(x, "b c t h w -> t (b c h w)").contiguous()
        m = repeat(m, "b 1 t h w -> t (b c h w)", c=c).contiguous()
        m_ = repeat(m_, "b 1 t h w -> t (b c h w)", c=c).contiguous()
        f = self.w @ (m * s2)
        for i in range(self.max_iter):
            x1 = f + self.w @ (m_ * x)
            if not (i % self.check_interval) and (
                (x1 - x).abs().mean() / (x1.abs().mean() + EPS) < self.tol
            ):
                return rearrange(x1, "t (b c h w) -> b c t h w", b=b, c=c, h=h)
            x = x1
        return rearrange(x, "t (b c h w) -> b c t h w", b=b, c=c, h=h)
