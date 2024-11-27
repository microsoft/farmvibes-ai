# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from tempfile import TemporaryDirectory
from typing import Dict, Optional, cast

import numpy as np

from vibe_core.data import Raster
from vibe_lib.raster import MaskedArrayType, load_raster, save_raster_from_ref


class CallbackBuilder:
    def __init__(self, threshold: Optional[float]):
        self.tmp_dir = TemporaryDirectory()
        if threshold is None:
            raise ValueError(
                "Threshold must not be None. "
                "Did you forget to overwrite the value on the workflow definition?"
            )
        self.threshold = threshold

    def __call__(self):
        def callback(raster: Raster) -> Dict[str, Raster]:
            data_ar = load_raster(raster)
            # Make a mess to keep the mask intact
            data_ma = data_ar.to_masked_array()
            thr_ma = cast(MaskedArrayType, (data_ma > self.threshold).astype("float32"))
            thr_ar = data_ar.copy(data=thr_ma.filled(np.nan))
            # Save it as uint8 instead of the original dtype
            thr_ar.rio.update_encoding({"dtype": "uint8"}, inplace=True)
            thr_raster = save_raster_from_ref(thr_ar, self.tmp_dir.name, raster)
            return {"thresholded": thr_raster}

        return callback

    def __del__(self):
        self.tmp_dir.cleanup()
