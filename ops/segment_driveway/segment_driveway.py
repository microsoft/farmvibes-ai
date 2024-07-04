import os
from tempfile import TemporaryDirectory
from typing import Any, Callable, Dict, Tuple

import numpy as np
import onnxruntime as ort
import rasterio
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from rasterio.enums import Resampling
from rasterio.windows import Window
from shapely import geometry as shpg

from vibe_core.data import AssetVibe, gen_guid
from vibe_core.data.rasters import CategoricalRaster, Raster
from vibe_lib.raster import resample_raster
from vibe_lib.spaceeye.chip import ChipDataset, Dims, get_loader, predict_chips


def reader(
    raster: Raster, window: Window, out_shape: Tuple[int, int]
) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    with rasterio.open(raster.raster_asset.url) as src:
        x = src.read(window=window, out_shape=out_shape, indexes=[4, 1, 2])
        mask = x == src.nodata
        x[mask] = 0
        return x, mask


def contrast_enhance(img: NDArray[Any], low: float = 2, high: float = 98) -> NDArray[np.float32]:
    img_min, img_max = np.nanpercentile(img, (low, high), axis=(-1, -2), keepdims=True)
    return np.clip((img.astype(np.float32) - img_min) / (img_max - img_min), 0, 1)


def pre_process(size: Tuple[int, int]) -> Callable[[NDArray[Any], NDArray[Any]], NDArray[Any]]:
    """
    Preprocess data by normalizing and picking a few bands
    """

    def fn(chip_data: NDArray[Any], _) -> NDArray[np.float32]:
        x = F.interpolate(torch.from_numpy(chip_data), size=size, mode="bilinear").numpy()
        x = contrast_enhance(x).astype(np.float32)
        return x

    return fn


def post_process(
    size: Tuple[int, int],
) -> Callable[[NDArray[Any], NDArray[Any], NDArray[Any]], NDArray[Any]]:
    """
    Get most probable class
    """

    def fn(_, __: NDArray[Any], model_out: NDArray[Any]) -> NDArray[Any]:
        x = F.interpolate(torch.from_numpy(model_out), size=size, mode="bilinear").numpy()
        return x.argmax(axis=1, keepdims=True).astype(np.uint8)

    return fn


class CallbackBuilder:
    def __init__(
        self,
        downsampling: int,
        root_dir: str,
        model_path: str,
        window_size: int,
        model_size: int,
        overlap: float,
        batch_size: int,
        num_workers: int,
    ):
        self.downsampling = downsampling
        self.root_dir = root_dir
        self.model_path = model_path
        self.window_size = window_size
        self.model_size = model_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tmp_dir = TemporaryDirectory()

    def __call__(self):
        def callback(
            input_raster: Raster,
        ) -> Dict[str, CategoricalRaster]:
            if self.downsampling < 1:
                raise ValueError(
                    f"Downsampling must be equal or larger than 1, found {self.downsampling}"
                )
            model_path = os.path.join(self.root_dir, self.model_path)
            model = ort.InferenceSession(model_path)
            chip_size = self.window_size
            step_size = int(chip_size * (1 - self.overlap))
            dataset = ChipDataset(
                [input_raster],
                chip_size=Dims(chip_size, chip_size, 1),
                step_size=Dims(step_size, step_size, 1),
                downsampling=self.downsampling,
                nodata=255,
                geometry_or_chunk=shpg.shape(input_raster.geometry),
                reader=reader,
                dtype="uint8",
            )

            dataloader = get_loader(dataset, self.batch_size, self.num_workers)
            pred_filepaths = predict_chips(
                model,
                dataloader,
                self.tmp_dir.name,
                skip_nodata=False,
                pre_process=pre_process((self.model_size, self.model_size)),
                post_process=post_process((self.window_size, self.window_size)),
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
                Resampling.nearest,
                nodata=255,
            )
            asset = AssetVibe(reference=mask_filepath, type="image/tiff", id=gen_guid())
            out = CategoricalRaster.clone_from(
                input_raster,
                id=gen_guid(),
                assets=[asset],
                categories=["Background", "Driveway", "Unknown"],
            )

            return {"segmentation_raster": out}

        return callback

    def __del__(self):
        self.tmp_dir.cleanup()
