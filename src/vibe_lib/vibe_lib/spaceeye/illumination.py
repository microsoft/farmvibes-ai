"""
Methods for computing, normalizing and interpolation illuminance of
multispectral raster timeseries.
"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

EPS = 1e-10
MIN_CLEAR_RATIO = 0.01
MIN_OVERLAP = 0.01
DEFAULT_LAMBDA_T = 0.5
SPATIAL_AXES = (-2, -1)


def extract_illuminance(
    x: NDArray[np.float32], mask: NDArray[np.float32]
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    numerator = (x * mask).sum(axis=SPATIAL_AXES, keepdims=True)
    denominator = mask.sum(axis=SPATIAL_AXES, keepdims=True)
    illuminance = numerator / (denominator + EPS)
    albedo = x / (illuminance + EPS)
    return albedo, illuminance


def extract_illuminance_simple(
    x: NDArray[np.float32], mask: NDArray[np.float32]
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    illuminance = masked_average_illuminance(x, mask)
    illuminance_mask = (mask.mean(axis=SPATIAL_AXES, keepdims=True) > MIN_CLEAR_RATIO).astype(
        np.float32
    )
    interp_illuminance = interpolate_illuminance(illuminance, illuminance_mask)
    x /= interp_illuminance + EPS  # Modify inplace to save memory
    return x, interp_illuminance


def masked_average_illuminance(
    x: NDArray[np.float32], mask: NDArray[np.float32]
) -> NDArray[np.float32]:
    #           x: C x T x H x W
    #        mask: 1 x T x H x W
    #      output: C x T x 1 x 1
    numerator = (x * mask).sum(axis=SPATIAL_AXES, keepdims=True)
    denominator = mask.sum(axis=SPATIAL_AXES, keepdims=True)
    illuminance = numerator / (denominator + EPS)
    return illuminance


def extract_illuminance_relative(
    x: NDArray[np.float32], mask: NDArray[np.float32]
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    illuminance_mask = (mask.mean(axis=SPATIAL_AXES, keepdims=True) > MIN_CLEAR_RATIO).astype(
        np.float32
    )

    # Relevant inputs for which we have data
    # We'll interpolate the rest
    available = np.squeeze(illuminance_mask).astype(bool)
    x_s = x[:, available]
    mask_s = mask[:, available]

    # find the anchor image
    clear_percentage = mask_s.sum(axis=0).mean(axis=SPATIAL_AXES)
    t_anchor = np.argmax(clear_percentage)

    # compute the anchor illuminance
    anchor_x = x_s[:, t_anchor : t_anchor + 1]
    anchor_mask = mask_s[:, t_anchor : t_anchor + 1]
    anchor_illuminance = masked_average_illuminance(anchor_x, anchor_mask)

    # Compute relative illuminance
    ratio_mask = ((mask_s + anchor_mask) == 2.0).astype(np.float32)
    # Fall back to the old method if there is not enough overlap
    overlap_mask = ratio_mask.mean(axis=(0, *SPATIAL_AXES)) > MIN_OVERLAP
    _, i_old = extract_illuminance(x_s[:, ~overlap_mask], mask_s[:, ~overlap_mask])
    # New method for the rest
    relative_illuminance = masked_average_illuminance(
        x_s[:, overlap_mask], ratio_mask[:, overlap_mask]
    ) / (masked_average_illuminance(anchor_x, ratio_mask[:, overlap_mask]) + EPS)
    # Compute final illuminance
    i_new = anchor_illuminance * relative_illuminance

    available_idx = np.where(available)[0]
    illuminance = np.zeros((*x.shape[:2], 1, 1), dtype=np.float32)
    illuminance[:, available_idx[~overlap_mask]] = i_old
    illuminance[:, available_idx[overlap_mask]] = i_new
    interp_illuminance = interpolate_illuminance(illuminance, illuminance_mask)
    x /= interp_illuminance + EPS  # Modify inplace to save memory
    return x, interp_illuminance


def add_illuminance(
    albedo: NDArray[np.float32], illuminance: NDArray[np.float32]
) -> NDArray[np.float32]:
    return albedo * illuminance


def interpolate_illuminance(
    illuminance: NDArray[np.float32], mask: NDArray[np.float32], lambda_t: float = DEFAULT_LAMBDA_T
) -> NDArray[np.float32]:
    C, T, _, _ = illuminance.shape
    t_tensor = np.arange(T, dtype=np.float32)
    delta_t_matrix = np.abs(t_tensor[None] - t_tensor[:, None])
    weight = np.exp(-lambda_t * delta_t_matrix)
    illuminance_sum = (weight @ illuminance.reshape((C, T, -1))).reshape(illuminance.shape)
    mask_sum = (weight @ mask.reshape((1, T, -1))).reshape(mask.shape)
    weighted_illuminance = illuminance_sum / (mask_sum + EPS)
    return weighted_illuminance * (1 - mask) + illuminance * mask
