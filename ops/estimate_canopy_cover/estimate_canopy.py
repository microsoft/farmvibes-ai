# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from tempfile import TemporaryDirectory
from typing import Any, Dict, List

import numpy as np
import xarray as xr
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from vibe_core.data import Raster
from vibe_lib.raster import get_cmap, json_to_asset, load_raster, save_raster_from_ref

SUPPORTED_INDICES: Dict[str, Dict[str, NDArray[np.float32]]] = {
    "ndvi": {
        "coefficients": np.array([[0.0, 0.28480232, 0.8144678, 0.63961434]], dtype=np.float32),
        "intercept": np.array([-0.10434419], dtype=np.float32),
    },
}


def calibrate(model: Pipeline, index: xr.DataArray):
    """
    Calibrate non-masked values, clip to [0, 1] and copy over the geodata from original array
    """
    index_masked = index.to_masked_array()
    index_compressed = index_masked.compressed()
    calibrated = model.predict(index_compressed[:, None]).squeeze().clip(0, 1)  # type: ignore
    calibrated_masked = index_masked.copy()
    calibrated_masked.data[~calibrated_masked.mask] = calibrated
    return index.copy(data=calibrated_masked)


class CallbackBuilder:
    def __init__(self, index: str):
        self.tmp_dir = TemporaryDirectory()
        if index not in SUPPORTED_INDICES:
            raise ValueError(f"Operation estimate_canopy called with unsupported index {index}")
        self.index = index

    def __call__(self):
        def calibration_callback(index_raster: Raster) -> Raster:
            output_dir = self.tmp_dir.name

            # Create model and copy weights
            model = make_pipeline(PolynomialFeatures(degree=3), Ridge())
            model[0].fit(np.zeros((1, 1)))
            model[1].coef_ = SUPPORTED_INDICES[self.index]["coefficients"].copy()  # type: ignore
            model[1].intercept_ = SUPPORTED_INDICES[self.index]["intercept"].copy()  # type: ignore
            index = load_raster(index_raster, use_geometry=True)
            calibrated = calibrate(model, index)

            vis_dict: Dict[str, Any] = {
                "bands": [0],
                "colormap": get_cmap("viridis"),
                "range": (0, 1),
            }
            calibrated_raster = save_raster_from_ref(
                calibrated, output_dir, ref_raster=index_raster
            )
            calibrated_raster.assets.append(json_to_asset(vis_dict, output_dir))
            return calibrated_raster

        def calibration_callback_list(indices: List[Raster]) -> Dict[str, List[Raster]]:
            return {"estimated_canopy_cover": [calibration_callback(index) for index in indices]}

        return calibration_callback_list

    def __del__(self):
        self.tmp_dir.cleanup()
