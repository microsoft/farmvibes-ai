# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import mimetypes
import os
from tempfile import TemporaryDirectory
from typing import Any, Dict, List

import numpy as np
import rasterio

from vibe_core.data import AssetVibe, Raster, gen_guid, gen_hash_id
from vibe_lib.raster import (
    RGBA,
    compute_sobel_gradient,
    include_raster_overviews,
    interpolated_cmap_from_colors,
    json_to_asset,
)

GRADIENT_CMAP_INTERVALS: List[float] = [0.0, 100.0, 200.0]

GRADIENT_CMAP_COLORS: List[RGBA] = [
    RGBA(255, 237, 160, 255),
    RGBA(254, 178, 76, 255),
    RGBA(240, 59, 32, 255),
]


class CallbackBuilder:
    def __init__(self):
        self.tmp_dir = TemporaryDirectory()

    def __call__(self):
        def operator_callback(input_raster: Raster) -> Dict[str, Raster]:
            input_band_mapping = input_raster.bands
            output_band_mapping = {}
            output_bands = []
            uid = gen_guid()

            out_path = os.path.join(self.tmp_dir.name, f"{gen_guid()}.tif")

            # Open the original raster and go through the layers computing the gradient.
            with rasterio.open(input_raster.raster_asset.url) as src:
                out_meta = src.meta
                for band_name in input_band_mapping.keys():
                    output_bands.insert(
                        input_band_mapping[band_name],
                        compute_sobel_gradient(src.read(input_band_mapping[band_name] + 1)),
                    )

            # Create a new raster to save the gradient layers.
            with rasterio.open(out_path, "w", **out_meta) as dst:
                dst.write(np.stack(output_bands, axis=0))

            # Update output bands name.
            output_band_mapping = {f"{k}_gradient": v for k, v in input_band_mapping.items()}

            vis_dict: Dict[str, Any] = {
                "bands": [0],
                "colormap": interpolated_cmap_from_colors(
                    GRADIENT_CMAP_COLORS, GRADIENT_CMAP_INTERVALS
                ),
                "range": (0, 200),
            }

            asset = AssetVibe(reference=out_path, type=mimetypes.types_map[".tif"], id=uid)
            include_raster_overviews(asset.local_path)
            out_raster = Raster.clone_from(
                input_raster,
                id=gen_hash_id(
                    f"{input_raster.id}_compute_raster_gradient",
                    input_raster.geometry,
                    input_raster.time_range,
                ),
                assets=[asset, json_to_asset(vis_dict, self.tmp_dir.name)],
                bands=output_band_mapping,
            )

            return {"output_raster": out_raster}

        return operator_callback

    def __del__(self):
        self.tmp_dir.cleanup()
