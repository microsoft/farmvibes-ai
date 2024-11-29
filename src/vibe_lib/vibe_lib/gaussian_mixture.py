# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.mixture import GaussianMixture


def low_rank_precision(
    cov: NDArray[Any], thr: float
) -> Tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    """
    Compute (pseudo?)inverse of low-rank approximation of covariance matrix.
    Approximation is computed by using considering only
    the top eigenvalues so that total energy is around thr.
    """
    w, v = np.linalg.eigh(cov)
    wi = 1 / w
    mask = np.cumsum(w[::-1] / w.sum())[::-1] < thr
    wi[~mask] = 0
    precision = v @ (wi * v.T)
    return precision, w, mask


def component_log_likelihood(
    x: NDArray[Any], mix: GaussianMixture, idx: int, thr: float = 0.99
) -> NDArray[Any]:
    """
    Pass in the curves (N, T), mixture object, and component index
    Output is size N containing the log-likelihood of each curve under the component
    Does the normalization part make sense? Should check with someone smarter
    """

    x = x - mix.means_[idx]  # type: ignore
    cov = mix.covariances_[idx]  # type: ignore
    # Invert covariance matrix but erasing bad eigenvalues
    precision, w, mask = low_rank_precision(cov, thr)  # type: ignore
    # Numerator
    n = (x * (precision @ x.T).T).sum(axis=1)
    # Denominator
    # We compute the denominator considering only the kept eigenvalues
    d = mask.sum() * np.log(2 * np.pi) + np.sum(np.log(w[mask]))  # type: ignore
    return -(n + d) / 2


def mixture_log_likelihood(
    x: NDArray[Any], mix: GaussianMixture, thr: float = 0.99
) -> NDArray[Any]:
    """
    Compute the mixture log-likelihood (max of each component log-likelihood)
    """
    return np.stack(
        [component_log_likelihood(x, mix, i, thr) for i in range(mix.n_components)]  # type: ignore
    ).max(axis=0)


def cluster_data(x: NDArray[Any], mix: GaussianMixture, thr: float = 0.99) -> NDArray[Any]:
    """
    Assign data to cluster with maximum likelihood
    """
    return np.argmax(
        [component_log_likelihood(x, mix, i, thr) for i in range(mix.n_components)],  # type: ignore
        axis=0,
    )


def train_mixture_with_component_search(
    x: NDArray[Any], max_components: int = 10, thr: float = 0.2
) -> GaussianMixture:
    """
    Train mixture of gaussians with stopping criterion to try and figure out how
    many components should be used
    """

    base_mixture = GaussianMixture(n_components=1).fit(x)
    base_ll = mixture_log_likelihood(x, base_mixture).mean()
    mixture = base_mixture
    ll = base_ll
    for n in range(2, max_components + 1):
        new_mixture = GaussianMixture(n_components=n).fit(x)
        new_ll = mixture_log_likelihood(x, new_mixture).mean()
        if (new_ll - ll) < np.abs(thr * base_ll):
            return mixture
        mixture = new_mixture
        ll = new_ll
    return mixture
