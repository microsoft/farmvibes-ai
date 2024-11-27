# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
from tempfile import TemporaryDirectory
from typing import Any, Dict

from numpy.typing import NDArray

from vibe_core.data import CategoricalRaster, Raster, gen_guid
from vibe_lib import overlap_clustering
from vibe_lib.raster import get_categorical_cmap, json_to_asset, load_raster, save_raster_to_asset

INT8_MAX_VALUE = 255

LOGGER = logging.getLogger(__name__)


class CallbackBuilder:
    def __init__(
        self,
        clustering_method: str,
        number_classes: int,
        half_side_length: int,
        number_iterations: int,
        stride: int,
        warmup_steps: int,
        warmup_half_side_length: int,
        window: int,
    ):
        self.tmp_dir = TemporaryDirectory()
        self.clustering_method = clustering_method
        self.number_classes = number_classes
        self.half_side_length = half_side_length
        self.number_iterations = number_iterations
        self.stride = stride
        self.warmup_steps = warmup_steps
        self.warmup_half_side_length = warmup_half_side_length
        self.window = window

    def __call__(self):
        def operator_callback(input_raster: Raster) -> Dict[str, Raster]:
            src_xa = load_raster(input_raster, use_geometry=True)
            src_data: NDArray[Any] = src_xa.to_numpy()

            if src_xa.dtype == "uint8":  # overlap clustering requires a float numpy array
                src_data = src_data / float(INT8_MAX_VALUE)

            p: NDArray[Any] = overlap_clustering.run_clustering(
                src_data,
                number_classes=self.number_classes,
                half_side_length=self.half_side_length,
                number_iterations=self.number_iterations,
                stride=self.stride,
                warmup_steps=self.warmup_steps,
                warmup_half_side_length=self.warmup_half_side_length,
                window=self.window,
            )

            vis_dict: Dict[str, Any] = {
                "bands": [0],
                "colormap": get_categorical_cmap("tab10", self.number_classes),
                "range": (0, self.number_classes - 1),
            }

            out_raster = CategoricalRaster(
                id=gen_guid(),
                geometry=input_raster.geometry,
                time_range=input_raster.time_range,
                assets=[
                    save_raster_to_asset(src_xa[0].copy(data=p), self.tmp_dir.name),
                    json_to_asset(vis_dict, self.tmp_dir.name),
                ],
                bands={"cluster": 0},
                categories=[f"cluster{i}" for i in range(self.number_classes)],
            )

            return {"output_raster": out_raster}

        return operator_callback

    def __del__(self):
        self.tmp_dir.cleanup()
