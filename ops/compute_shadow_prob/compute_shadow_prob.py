# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from tempfile import TemporaryDirectory
from typing import Any, Dict

import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from rasterio.enums import Resampling

from vibe_core.data import (
    AssetVibe,
    S2ProcessingLevel,
    Sentinel2CloudProbability,
    Sentinel2Raster,
    gen_guid,
)
from vibe_lib.raster import DEFAULT_NODATA, resample_raster
from vibe_lib.spaceeye.chip import ChipDataset, Dims, InMemoryReader, get_loader, predict_chips
from vibe_lib.spaceeye.utils import verify_processing_level


def pre_process(scale: float):
    def fun(chip_data: NDArray[Any], _):
        return chip_data * scale

    return fun


def post_process(
    chip_data: NDArray[Any], chip_mask: NDArray[Any], model_out: NDArray[Any]
) -> NDArray[Any]:
    """
    After prediction, we set nodata (all zeros) regions as 100% cloud
    """
    nodata_mask = chip_mask.any(axis=1, keepdims=True)
    model_prob = 1 / (1 + np.exp(-model_out))
    model_prob[nodata_mask] = 1
    return model_prob


class CallbackBuilder:
    def __init__(
        self,
        downsampling: int,
        root_dir: str,
        model_path: str,
        window_size: int,
        overlap: float,
        batch_size: int,
        num_workers: int,
        in_memory: bool,
    ):
        self.downsampling = downsampling
        self.root_dir = root_dir
        self.model_path = model_path
        self.window_size = window_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.in_memory = in_memory
        self.tmp_dir = TemporaryDirectory()

    def __call__(self):
        def compute_shadow_prob(
            sentinel_raster: Sentinel2Raster,
        ) -> Dict[str, Sentinel2CloudProbability]:
            verify_processing_level((sentinel_raster,), S2ProcessingLevel.L2A, "FPN Shadow model")

            if self.downsampling < 1:
                raise ValueError(
                    f"Downsampling must be equal or larger than 1, found {self.downsampling}"
                )
            model_path = os.path.join(self.root_dir, self.model_path)
            model = ort.InferenceSession(model_path)
            chip_size = self.window_size
            step_size = int(chip_size * (1 - self.overlap))
            dataset = ChipDataset(
                [sentinel_raster],
                chip_size=Dims(chip_size, chip_size, 1),
                step_size=Dims(step_size, step_size, 1),
                downsampling=self.downsampling,
                nodata=DEFAULT_NODATA,
                reader=InMemoryReader(self.downsampling) if self.in_memory else None,
            )

            dataloader = get_loader(
                dataset, self.batch_size, self.num_workers if not self.in_memory else 0
            )
            pred_filepaths = predict_chips(
                model,
                dataloader,
                self.tmp_dir.name,
                skip_nodata=True,
                pre_process=pre_process(sentinel_raster.scale),
                post_process=post_process,
            )
            assert (
                len(pred_filepaths) == 1
            ), f"Expected one prediction file, found: {len(pred_filepaths)}"
            mask_filepath = resample_raster(
                pred_filepaths[0],
                self.tmp_dir.name,
                dataset.width,
                dataset.height,
                dataset.transform,
                Resampling.bilinear,
            )
            asset = AssetVibe(reference=mask_filepath, type="image/tiff", id=gen_guid())

            shadow_mask = Sentinel2CloudProbability.clone_from(
                sentinel_raster, id=gen_guid(), assets=[asset]
            )

            return {"shadow_probability": shadow_mask}

        return compute_shadow_prob

    def __del__(self):
        self.tmp_dir.cleanup()
