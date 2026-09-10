# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import numpy as np
import pytest
import xarray as xr
from compute_irrigation_probability import CallbackBuilder
from sklearn.linear_model import LogisticRegression


def test_distinct_ngi_egi_coefficients(monkeypatch: pytest.MonkeyPatch):
    inputs = [object() for _ in range(5)]
    landsat, cloud_mask, ngi, egi, lst = inputs
    rasters = {
        cloud_mask: np.ones((1, 2, 2)),
        ngi: [[[0, 1], [2, 3]]],
        egi: [[[3, 2], [1, 0]]],
        lst: [[[1, 2], [3, 4]]],
    }
    captured = {}

    monkeypatch.setattr(
        "compute_irrigation_probability.load_raster_match",
        lambda raster, _: xr.DataArray(rasters[raster], dims=("band", "y", "x")),
    )

    def capture_coefficients(model: LogisticRegression, _: object):
        captured["coef"] = model.coef_
        raise RuntimeError("stop after coefficient capture")

    monkeypatch.setattr(LogisticRegression, "predict_proba", capture_coefficients)

    builder = CallbackBuilder(1.0, 2.0, 3.0, 4.0)
    try:
        with pytest.raises(RuntimeError, match="stop after coefficient capture"):
            builder()(landsat, ngi, egi, lst, cloud_mask)  # type: ignore
    finally:
        builder.tmp_dir.cleanup()

    np.testing.assert_array_equal(captured["coef"], [[1.0, 2.0, 3.0]])
