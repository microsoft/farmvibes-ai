# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
from tempfile import TemporaryDirectory
from typing import Dict

from rasterio.enums import Resampling

from vibe_core.data import Raster, gen_guid
from vibe_lib.raster import load_raster_match, save_raster_to_asset

LOGGER = logging.getLogger(__name__)


class CallbackBuilder:
    def __init__(self, resampling: str):
        self.tmp_dir = TemporaryDirectory()
        self.resampling: Resampling = getattr(Resampling, resampling)

    def __call__(self):
        def operator_callback(raster: Raster, ref_raster: Raster) -> Dict[str, Raster]:
            raster_ar = load_raster_match(
                raster, match_raster=ref_raster, resampling=self.resampling
            )
            asset = save_raster_to_asset(raster_ar, self.tmp_dir.name)
            assets = [asset]
            try:
                assets.append(raster.visualization_asset)
            except ValueError as e:
                LOGGER.warning(f"Visualization asset not found {e}")

            out_raster = Raster.clone_from(
                src=raster,
                id=gen_guid(),
                geometry=ref_raster.geometry,
                assets=assets,
            )

            return {"output_raster": out_raster}

        return operator_callback

    def __del__(self):
        self.tmp_dir.cleanup()
