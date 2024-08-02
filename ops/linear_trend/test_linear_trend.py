# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import xarray as xr
from linear_trend import fit_model_in_bulk
from pandas import Timedelta, Timestamp


def _one_test_fit_model_in_bulk(sy: int, sx: int, sz: int):
    TOL = 1e-10
    t = [Timestamp(2001, 1, 1) + Timedelta(days=d) for d in range(sz)]  # type: ignore

    fake_rasters = []
    true_trend = []
    for i in range(sy * sx):
        h = i / (sy * sx - 1)
        true_trend.append(h)
        fake_rasters.append(np.linspace(0, h * (sz - 1), sz))
    fake_rasters = np.stack(fake_rasters).reshape((sy, sx, sz)).transpose((2, 0, 1))
    true_trend = np.array(true_trend).reshape((sy, sx))

    da = xr.DataArray(data=fake_rasters, dims=["time", "y", "x"], coords={"time": t})

    trend_hat, _ = fit_model_in_bulk(da)  # type: ignore

    assert np.max(np.abs(trend_hat - true_trend)) < TOL


def test_fit_model_in_bulk():
    s = [32, 64, 128]
    for sy in s:
        for sx in s:
            for sz in s:
                _one_test_fit_model_in_bulk(sy, sx, sz)
