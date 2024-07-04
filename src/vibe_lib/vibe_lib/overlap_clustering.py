import logging
import math
from typing import Any

import numpy as np
import torch as T
from numpy.typing import NDArray
from torch.nn.functional import avg_pool2d, interpolate

POSTERIOR_SMOOTHING = 0.001

LOGGER = logging.getLogger(__name__)


# compute 2D average pooling of data in squares of side 2*half_side_length+1
def compute_local_average(data: T.Tensor, half_side_length: int, stride: int = 1):
    if half_side_length == 0:
        return data
    w, h = data.shape[-2:]
    mean = avg_pool2d(
        data.reshape(-1, 1, w, h),
        2 * half_side_length + 1,
        stride=stride,
        padding=half_side_length,
        count_include_pad=False,
    )

    # if pooling was strided (for speedup), upsample to original raster size
    if stride > 1:
        mean = interpolate(mean, size=(w, h), mode="bilinear", align_corners=False)
    return mean.view(data.shape)


# compute mean and variance in local windows of data in each cluster c weighted by q[c]
def compute_weighted_average_and_variance(
    data: T.Tensor,
    weights: T.Tensor,
    half_side_length: int,
    stride: int = 1,
    var_min: float = 0.0001,
    mq_min: float = 0.000001,
):
    # compute probability normalization constants per class
    mq = compute_local_average(weights, half_side_length, stride)
    mq.clamp(min=mq_min)

    # instantiate data and data**2 weighted by weights[c] for each c
    # future todo: investigate whether replacing einsum by broadcast ops gives a speedup
    weighted = T.einsum("zij,cij->czij", data, weights)  # class,channel,x,y
    weighted_sq = T.einsum("zij,cij->czij", data**2, weights)

    # mean = E_[x~weights[c]] data[x]
    # var = E_x (data[x]^2) - (E_x data[x])^2
    mean = compute_local_average(weighted, half_side_length, stride) / mq.unsqueeze(1)
    var = compute_local_average(weighted_sq, half_side_length, stride) / mq.unsqueeze(1) - mean**2
    var = var.clamp(min=var_min)

    return mean, var


# batched log-pdf of a diagonal Gaussian
def lp_gaussian(
    data: T.Tensor, mean: T.Tensor, var: T.Tensor, half_side_length: int, stride: int = 1
):
    m0 = -compute_local_average(1 / var, half_side_length, stride)
    m1 = compute_local_average(2 * mean / var, half_side_length, stride)
    m2 = -compute_local_average(mean**2 / var, half_side_length, stride)
    L = compute_local_average(T.log(var), half_side_length, stride)
    return (m0 * data**2 + m1 * data + m2 - 1 * L).sum(1) / 2


# batched posterior over components in a Gaussian mixture
def gaussian_mixture_posterior(
    data: T.Tensor,
    prior: T.Tensor,
    mean: T.Tensor,
    var: T.Tensor,
    half_side_length: int,
    stride: int = 1,
):
    # compute unnormalized log-pdf
    lp = lp_gaussian(data, mean, var, half_side_length, stride)

    # posterior proportional to density*prior
    p = lp.softmax(0) * prior
    p /= p.sum(0)
    p += POSTERIOR_SMOOTHING
    p /= p.sum(0)

    return p


# one iteration of EM algorithm for Gaussian mixture
def perform_iteration_expectation_maximization(
    data: T.Tensor, p: T.Tensor, half_side_length: int, stride: int = 1
):
    # M step: compute optimal GMM parameters in each raster window
    prior = compute_local_average(p, half_side_length, stride)
    mean, var = compute_weighted_average_and_variance(data, p, half_side_length, stride)

    # E step: recompute posteriors
    p_new = gaussian_mixture_posterior(data, prior, mean, var, half_side_length, stride)

    return p_new, mean, var, prior


# run EM algorithm for Gaussian mixture
def run_clustering(
    image: NDArray[Any],
    number_classes: int,
    half_side_length: int,
    number_iterations: int,
    stride: int,
    warmup_steps: int,
    warmup_half_side_length: int,
    window: int,
) -> NDArray[Any]:
    _, x_size, y_size = image.shape
    result = np.zeros(shape=(x_size, y_size), dtype="uint8")

    for row in range(math.ceil(x_size / window)):
        for col in range(math.ceil(y_size / window)):
            xmin = row * window
            xmax = (row + 1) * window
            if xmax > x_size:
                xmax = x_size
            ymin = col * window
            ymax = (col + 1) * window
            if ymax > y_size:
                ymax = y_size

            partial_image = image[:, xmin:xmax, ymin:ymax]

            logging.info(
                f"Computing clusters for row: {row}, col: {col}, [{xmin}, {xmax}, {ymin}, {ymax}]"
            )

            with T.inference_mode():
                # convert image to Torch object
                data = T.as_tensor(partial_image)

                # randomly initialize posterior matrix
                p = T.rand((number_classes,) + partial_image.shape[1:])
                p /= p.sum(0)

                # EM
                for i in range(number_iterations):
                    p.mean().item()  # trigger synchronization
                    p, _, _, _ = perform_iteration_expectation_maximization(
                        data,
                        p,
                        warmup_half_side_length if i < warmup_steps else half_side_length,
                        stride,
                    )

                # return np.argmax(p.numpy(), axis=0)
                result[xmin:xmax, ymin:ymax] = np.argmax(p.numpy(), axis=0)
    return result
