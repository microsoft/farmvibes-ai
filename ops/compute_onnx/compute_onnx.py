# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from tempfile import TemporaryDirectory
from typing import Dict, List, Optional, Union

import onnxruntime as ort
from rasterio.enums import Resampling

from vibe_core.data import AssetVibe, Raster, gen_guid
from vibe_core.data.rasters import RasterChunk, RasterSequence
from vibe_lib.raster import resample_raster
from vibe_lib.spaceeye.chip import Dims, StackOnChannelsChipDataset, get_loader, predict_chips

ROOT_DIR = "/mnt/onnx_resources/"


class CallbackBuilder:
    def __init__(
        self,
        model_file: str,
        window_size: int,
        overlap: float,
        batch_size: int,
        num_workers: int,
        nodata: Union[float, int],
        skip_nodata: bool,
        resampling: str = "bilinear",
        root_dir: str = ROOT_DIR,
        downsampling: int = 1,
    ):
        self.tmp_dir = TemporaryDirectory()
        self.downsampling = downsampling
        if model_file is None or not os.path.exists(os.path.join(root_dir, model_file)):
            raise ValueError(f"Model file '{model_file}' does not exist.")
        self.root_dir = root_dir
        self.model_file = model_file
        self.window_size = window_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.nodata = nodata
        self.skip_nodata = skip_nodata
        self.resampling = Resampling[resampling]

    def __call__(self):
        def compute_onnx(
            input_raster: Union[Raster, RasterSequence, List[Raster]],
            chunk: Optional[RasterChunk] = None,
        ) -> Dict[str, Union[Raster, RasterChunk]]:
            if self.downsampling < 1:
                raise ValueError(
                    f"Downsampling must be equal or larger than 1, found {self.downsampling}"
                )

            if isinstance(input_raster, RasterSequence):
                input = [
                    Raster.clone_from(input_raster, gen_guid(), assets=[i])
                    for i in input_raster.get_ordered_assets()
                ]
            elif isinstance(input_raster, list):
                input = input_raster
            else:
                input = [input_raster]

            model_path = os.path.join(self.root_dir, self.model_file)
            model = ort.InferenceSession(model_path)
            chip_size = self.window_size
            step_size = int(chip_size * (1 - self.overlap))
            dataset = StackOnChannelsChipDataset(
                [[i] for i in input],
                chip_size=Dims(chip_size, chip_size, 1),
                step_size=Dims(step_size, step_size, 1),
                downsampling=self.downsampling,
                nodata=self.nodata,
                geometry_or_chunk=chunk,
            )

            dataloader = get_loader(dataset, self.batch_size, self.num_workers)
            pred_filepaths = predict_chips(
                model, dataloader, self.tmp_dir.name, skip_nodata=self.skip_nodata
            )
            assert (
                len(pred_filepaths) == 1
            ), f"Expected one prediction file, found: {len(pred_filepaths)}"
            pred_filepath = resample_raster(
                pred_filepaths[0],
                self.tmp_dir.name,
                dataset.width,
                dataset.height,
                dataset.transform,
                self.resampling,
            )
            asset = AssetVibe(reference=pred_filepath, type="image/tiff", id=gen_guid())
            if chunk is None:
                res = Raster.clone_from(input[0], id=gen_guid(), assets=[asset])
            else:
                res = RasterChunk.clone_from(
                    chunk, id=gen_guid(), geometry=chunk.geometry, assets=[asset]
                )

            return {"output_raster": res}

        return compute_onnx

    def __del__(self):
        self.tmp_dir.cleanup()
