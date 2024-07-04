from tempfile import TemporaryDirectory
from typing import Dict, List

import numpy as np

from vibe_core.data import Raster
from vibe_lib.raster import load_raster, save_raster_from_ref


class CallbackBuilder:
    def __init__(self, from_values: List[float], to_values: List[float]):
        self.tmp_dir = TemporaryDirectory()

        if len(from_values) != len(to_values):
            raise ValueError(
                f"'from_values' and 'to_values' must have the same length. "
                f"Got {len(from_values)} and {len(to_values)}, respectively."
            )

        self.recode_map = dict(zip(from_values, to_values))

    def __call__(self):
        def callback(raster: Raster) -> Dict[str, Raster]:
            data_ar = load_raster(raster)

            # Return the same pixel value if it is not in the recode map
            transformed_ar = data_ar.copy(
                data=np.vectorize(lambda x: self.recode_map.get(x, x))(data_ar)
            )
            transformed_raster = save_raster_from_ref(transformed_ar, self.tmp_dir.name, raster)

            return {"recoded_raster": transformed_raster}

        return callback

    def __del__(self):
        self.tmp_dir.cleanup()
