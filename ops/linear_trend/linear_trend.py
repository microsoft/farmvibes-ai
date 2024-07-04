import hashlib
from tempfile import TemporaryDirectory
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from vibe_core.data import RasterChunk
from vibe_core.data.rasters import Raster
from vibe_lib.raster import read_chunk_series, save_raster_to_asset


def fit_model_in_bulk(da: xr.Dataset) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    B, A, ATAinv, beta_hat, trend = linear_fit_in_bulk(da)

    test_stat = compute_test_statistics(da, B, A, ATAinv, beta_hat)

    return trend, test_stat


def compute_test_statistics(
    da: xr.Dataset,
    B: NDArray[np.float64],
    A: NDArray[np.float64],
    ATAinv: NDArray[np.float64],
    beta_hat: NDArray[np.float64],
):
    # estimating test statistic for the trend
    n = np.sum(np.logical_not(np.isnan(B)).astype(int), axis=0)
    gamma = ATAinv[0, 0]
    sig_hat2 = np.nansum((B - A @ beta_hat) ** 2, axis=0) / (n - 2)
    maskout = np.logical_or(np.isnan(sig_hat2), sig_hat2 == 0)
    test_stat = beta_hat[0, :] / np.sqrt(np.where(np.logical_not(maskout), sig_hat2, 1.0) * gamma)

    # make sure we have at least two points to store trend
    test_stat = np.where(n > 1, test_stat, np.nan)

    test_stat = np.where(np.logical_not(maskout), test_stat, np.nan)

    test_stat = test_stat.reshape(da.shape[1:])
    return test_stat


def linear_fit_in_bulk(
    da: xr.Dataset,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    # fitting a linear model in bulk
    n = da.shape[0]
    B = da.values.reshape((n, -1))
    t = da.time.values
    if type(t[0]) is not np.datetime64:
        t = np.array(list(map(lambda x: x.to_datetime64(), da.time.values)))
    t = (t - np.min(t)) / np.timedelta64(1, "D")
    A = np.stack((t, np.ones_like(t))).T
    ATAinv = np.linalg.inv(A.T @ A)

    # this is just A.T@B, but avoing issues with nan, so that even if
    # one pixel/band has a nan in a given time we still estimate the trend
    # by ignoring the particular time (also in test statistic estimation)
    ATB = np.nansum(A.reshape(n, 2, 1) * B.reshape(n, 1, -1), axis=0)

    beta_hat = ATAinv @ ATB
    trend = beta_hat[0, :]

    # make sure we have at least two points to store trend
    trend = np.where(n > 1, trend, np.nan)

    trend = trend.reshape(da.shape[1:])
    return B, A, ATAinv, beta_hat, trend


class CallbackBuilder:
    def __init__(self):
        self.tmp_dir = TemporaryDirectory()

    def __call__(self):
        def linear_trend_callback(
            series: RasterChunk, rasters: List[Raster]
        ) -> Dict[str, RasterChunk]:
            da = read_chunk_series(series.limits, rasters)

            trend, test_stat = fit_model_in_bulk(da)

            # store results
            coords = {k: v for k, v in da.coords.items() if k != "time" and k != "band"}
            data = np.concatenate((trend, test_stat))
            res = xr.DataArray(data=data, dims=list(da.dims)[1:], coords=coords, attrs=da.attrs)
            asset = save_raster_to_asset(res, self.tmp_dir.name)
            bands: Dict[str, int] = {}
            for k, v in series.bands.items():
                bands[f"trend_{k}"] = int(v)
                bands[f"test_stat_{k}"] = int(v) + len(series.bands)
            res = RasterChunk(
                id=hashlib.sha256(f"linear_trend-{series.id}".encode()).hexdigest(),
                time_range=series.time_range,
                geometry=series.geometry,
                assets=[asset],
                bands=bands,
                chunk_pos=series.chunk_pos,
                num_chunks=series.num_chunks,
                limits=series.limits,
                write_rel_limits=series.write_rel_limits,
            )

            return {"trend": res}

        return linear_trend_callback

    def __del__(self):
        self.tmp_dir.cleanup()
