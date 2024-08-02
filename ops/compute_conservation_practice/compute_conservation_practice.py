# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from tempfile import TemporaryDirectory
from typing import Any, Dict

import numpy as np
import onnxruntime as ort
import rasterio
from numpy.typing import NDArray
from rasterio import Affine
from rasterio.enums import Resampling

from vibe_core.data import AssetVibe, gen_guid
from vibe_core.data.rasters import Raster
from vibe_lib.raster import DEFAULT_NODATA, resample_raster
from vibe_lib.spaceeye.chip import Dims, StackOnChannelsChipDataset, get_loader, predict_chips


def post_process(_: NDArray[Any], __: NDArray[Any], model_out: NDArray[Any]) -> NDArray[Any]:
    """
    After prediction, we transform probabilities into classes via argmax
    """
    model_classes = np.argmax(model_out, axis=1, keepdims=True)
    return model_classes


def get_meta(in_path: str, width: int, height: int, transform: Affine) -> Dict[str, Any]:
    with rasterio.open(in_path) as src:
        kwargs = src.meta.copy()
        kwargs.update(
            {
                "nodata": 0,
                "width": width,
                "height": height,
                "transform": transform,
            }
        )
        return kwargs


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
    ):
        self.downsampling = downsampling
        self.root_dir = root_dir
        self.model_path = model_path
        self.window_size = window_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tmp_dir = TemporaryDirectory()

    def __call__(self):
        def compute_conservation_practice(
            elevation_gradient: Raster, average_elevation: Raster
        ) -> Dict[str, Raster]:
            if self.downsampling < 1:
                raise ValueError(
                    f"Downsampling must be equal or larger than 1, found {self.downsampling}"
                )
            model_path = os.path.join(self.root_dir, self.model_path)
            model = ort.InferenceSession(model_path)
            chip_size = self.window_size
            step_size = int(chip_size * (1 - self.overlap))

            dataset = StackOnChannelsChipDataset(
                [[elevation_gradient], [average_elevation]],
                chip_size=Dims(chip_size, chip_size, 1),
                step_size=Dims(step_size, step_size, 1),
                downsampling=self.downsampling,
                nodata=DEFAULT_NODATA,
            )

            dataloader = get_loader(dataset, self.batch_size, self.num_workers)

            pred_filepaths = predict_chips(
                model,
                dataloader,
                self.tmp_dir.name,
                skip_nodata=False,
                post_process=post_process,
            )
            assert (
                len(pred_filepaths) == 1
            ), f"Expected one prediction file, found: {len(pred_filepaths)}"
            out_filepath = resample_raster(
                pred_filepaths[0],
                self.tmp_dir.name,
                dataset.width,
                dataset.height,
                dataset.transform,
                Resampling.nearest,
            )
            asset = AssetVibe(reference=out_filepath, type="image/tiff", id=gen_guid())
            pred = Raster.clone_from(elevation_gradient, id=gen_guid(), assets=[asset])

            return {"output_raster": pred}

        return compute_conservation_practice

    def __del__(self):
        self.tmp_dir.cleanup()
