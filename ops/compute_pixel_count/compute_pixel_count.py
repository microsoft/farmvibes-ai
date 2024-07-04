import os
from tempfile import TemporaryDirectory
from typing import Any, Dict

import numpy as np
import rasterio
from numpy._typing import NDArray
from rasterio.mask import mask
from shapely import geometry as shpg

from vibe_core.data import Raster, RasterPixelCount, gen_guid
from vibe_core.data.core_types import AssetVibe, BaseGeometry

UNIQUE_VALUES_COLUMN = "unique_values"
COUNTS_COLUMN = "counts"


def read_data(raster: Raster, geom: BaseGeometry) -> NDArray[Any]:
    with rasterio.open(raster.raster_asset.path_or_url) as src:
        raw_data, _ = mask(
            src,
            [geom],
            crop=True,
            filled=False,
        )

        # We are counting the number of pixels
        # for all the raster bands
        return raw_data.compressed()  # type: ignore


def calculate_unique_values(data: NDArray[Any]) -> NDArray[Any]:
    unique_values, counts = np.unique(data, return_counts=True)
    return np.column_stack((unique_values, counts))


class CallbackBuilder:
    def __init__(self):
        self.tmp_dir = TemporaryDirectory()

    def __call__(self):
        def callback(raster: Raster) -> Dict[str, RasterPixelCount]:
            data = read_data(raster, shpg.shape(raster.geometry))
            stack_data = calculate_unique_values(data)
            guid = gen_guid()
            filepath = os.path.join(self.tmp_dir.name, f"{guid}.csv")

            # Save the data to a CSV file
            np.savetxt(
                filepath,
                stack_data,
                delimiter=",",
                fmt="%d",
                header=f"{UNIQUE_VALUES_COLUMN},{COUNTS_COLUMN}",
                comments="",
            )

            raster_pixel_count = RasterPixelCount.clone_from(
                raster,
                id="pixel_count_" + raster.id,
                assets=[AssetVibe(reference=filepath, type="text/csv", id=guid)],
            )

            return {"pixel_count": raster_pixel_count}

        return callback

    def __del__(self):
        self.tmp_dir.cleanup()
