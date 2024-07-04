"""
This module contains code for running a pytorch module in chips extracted from
rasters. Chips are read from disk before inference and predictions are written
to disk as they are computed.
"""

import logging
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union, cast, overload

import geopandas as gpd
import numpy as np
import onnxruntime as ort
import rasterio
from numpy.typing import NDArray
from rasterio import Affine
from rasterio.mask import raster_geometry_mask
from rasterio.windows import Window
from rasterio.windows import bounds as window_bounds
from rasterio.windows import transform as window_transform
from shapely import geometry as shpg
from shapely.geometry.base import BaseGeometry
from torch.utils.data import DataLoader, Dataset

from vibe_core.data import Raster
from vibe_core.data.rasters import RasterChunk

from ..raster import MaskedArrayType, write_window_to_file
from .dataset import Dims, get_read_windows, get_write_windows

LOGGER = logging.getLogger(__name__)
T = TypeVar("T", bound=Raster)

ChipDataType = Tuple[NDArray[Any], NDArray[Any], Dict[str, Any]]

EPS = 1e-6


def affine_all_close(tr1: Affine, tr2: Affine, rel_tol: float = EPS) -> bool:
    return all(abs((a - b) / (a + b + EPS)) < rel_tol for a, b in zip(tr1, tr2))


class InMemoryReader:
    def __init__(self, downsampling: int):
        self.rasters = {}
        self.downsampling = downsampling
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _cache_raster(self, raster: Raster):
        """
        Read the whole raster and keep it in memory for subsequent windows
        """

        self.logger.debug(f"Loading raster id={raster.id} into memory")
        # Read the whole raster and keep it in memory
        with rasterio.open(raster.raster_asset.url) as src:
            ds_shape = (src.height // self.downsampling, src.width // self.downsampling)
            raster_data = src.read(out_shape=ds_shape)
            self.rasters[raster.id] = {
                "data": raster_data,
                "meta": src.meta,
            }
        self.logger.debug(
            f"Loaded raster id={raster.id} into memory as array of shape "
            f"{raster_data.shape} and dtype {raster_data.dtype}"
        )

    def _adjust_window(self, window: Window):
        """Adjust window to downsampled raster"""
        win = Window(*(i // self.downsampling for i in window.flatten()))
        return win

    def _read_data_from_cache(self, raster: Raster, window: Window):
        if raster.id not in self.rasters:
            self._cache_raster(raster)
        # Adjust window to downsampled raster
        win = self._adjust_window(window)
        i, j = win.toslices()
        raster_cache = self.rasters[raster.id]
        x = raster_cache["data"][:, i, j]
        return x.astype(np.float32), x == raster_cache["meta"]["nodata"]

    def __call__(self, raster: Raster, window: Window, out_shape: Tuple[int, int]):
        win_data, win_mask = self._read_data_from_cache(raster, window)
        if win_data.shape[1:] != out_shape:
            raise ValueError(
                f"Requested output shape {out_shape}, got {win_data.shape[1:]} "
                f"for downsampling {self.downsampling}"
            )
        return win_data, win_mask


class ChipDataset(Dataset[ChipDataType]):
    """
    Pytorch dataset that load chips of data for model inference.

    This dataset can be used with a pytorch DataLoader to load data as needed and
    avoid loading the whole raster into memory. Will optionally downsample the
    input to reduce computation requirements.
    """

    def __init__(
        self,
        rasters: List[T],
        chip_size: Dims,
        step_size: Dims,
        downsampling: int = 1,
        nodata: Optional[float] = None,
        geometry_or_chunk: Optional[Union[BaseGeometry, RasterChunk]] = None,
        reader: Optional[
            Callable[[T, Window, Tuple[int, int]], Tuple[NDArray[Any], NDArray[Any]]]
        ] = None,
        dtype: str = "float32",
    ):
        self.rasters = rasters
        self.chip_size = Dims(*chip_size)
        self.step_size = Dims(*step_size)
        self.downsampling = downsampling
        self.read_chip = Dims(
            chip_size.width * downsampling, chip_size.height * downsampling, chip_size.time
        )
        self.read_step = Dims(
            step_size.width * downsampling, step_size.height * downsampling, step_size.time
        )
        self.reader = reader if reader is not None else self._default_reader

        self._read_meta(rasters[0].raster_asset.url, geometry_or_chunk, nodata)

        self.out_width = self.width // self.downsampling
        self.out_height = self.height // self.downsampling
        self.out_transform = self.transform * Affine.scale(self.downsampling, self.downsampling)

        self.read_windows = get_read_windows(
            self.width, self.height, len(self.rasters), self.read_chip, self.read_step, self.offset
        )
        self.write_windows, self.chip_slices = get_write_windows(
            self.out_width, self.out_height, len(self.rasters), self.chip_size, self.step_size
        )

        self.meta = {
            "driver": "GTiff",
            "height": self.out_height,
            "width": self.out_width,
            "crs": self.crs,
            "dtype": dtype,
            "transform": self.out_transform,
            "nodata": self.nodata,
        }

    def _adjust_roi_window(self, window: Window) -> Window:
        width = self.chip_size.width
        height = self.chip_size.height
        if window.width >= width and window.height >= height:
            return window
        width = max(window.width, width)
        height = max(window.height, height)
        LOGGER.warning(
            f"RoI has dimensions {window.width, window.height} and chip size is {self.chip_size},"
            f" adjusting to {width, height}"
        )
        diff_w = width - window.width
        dw = diff_w // 2
        diff_h = height - window.height
        dh = diff_h // 2

        hs, ws = window.toranges()
        min_w = max(ws[0] - dw, 0)
        max_w = min(ws[1] + diff_w - dw, self.raster_width)
        min_h = max(hs[0] - dh, 0)
        max_h = min(hs[1] + diff_h - dh, self.raster_height)

        new_win = Window.from_slices((min_h, max_h), (min_w, max_w))
        LOGGER.info(f"Adjusting from {window} to {new_win}")
        return new_win

    def __len__(self):
        return len(self.read_windows)

    def _read_meta(
        self,
        url: str,
        geometry_or_chunk: Optional[Union[BaseGeometry, RasterChunk]] = None,
        nodata: Optional[float] = None,
    ):
        with rasterio.open(url) as src:
            self.crs = src.crs
            self.raster_width: int = src.width
            self.raster_height: int = src.height
            self.nodata = src.nodata if nodata is None else nodata
            if geometry_or_chunk and isinstance(geometry_or_chunk, BaseGeometry):
                # Compute envelope in native CRS to avoid nodata
                box = cast(
                    shpg.Polygon,
                    gpd.GeoSeries(geometry_or_chunk, crs="epsg:4326")
                    .to_crs(self.crs)
                    .iloc[0]
                    .envelope,
                )
                window = cast(
                    Window, raster_geometry_mask(src, [box], all_touched=True, crop=True)[2]
                )
                # Adjust window to make sure it is not too small
                window = self._adjust_roi_window(window)
                # Compute the transform with the adjusted window
                self.transform: Affine = window_transform(window, src.transform)
                self.roi_window = window
                self.width: int = window.width
                self.height: int = window.height
                self.offset = Dims(window.col_off, window.row_off, 0)
            elif geometry_or_chunk and isinstance(geometry_or_chunk, RasterChunk):
                col_off, row_off, width, height = geometry_or_chunk.limits
                self.transform: Affine = src.transform
                self.width: int = width
                self.height: int = height
                self.offset = Dims(col_off, row_off, 0)
                self.roi_window = Window(*geometry_or_chunk.limits)  # type:ignore
                box = window_bounds(self.roi_window, self.transform)
            else:
                box = shpg.box(*src.bounds)
                self.transform: Affine = src.transform
                self.width: int = src.width
                self.height: int = src.height
                self.offset = Dims(0, 0, 0)
                self.roi_window = Window(0, 0, src.width, src.height)  # type:ignore
            self.roi = box

    @staticmethod
    def _default_reader(
        raster: Raster, window: Window, out_shape: Tuple[int, int]
    ) -> Tuple[NDArray[np.float32], NDArray[np.bool_]]:
        with rasterio.open(raster.raster_asset.url) as src:
            x = src.read(window=window, out_shape=out_shape, masked=True).astype(np.float32)
        x = cast(MaskedArrayType, x)
        return x.data, np.ma.getmaskarray(x)

    def __getitem__(self, idx: int) -> ChipDataType:
        read_window, read_times = self.read_windows[idx]
        write_window, write_times = self.write_windows[idx]
        chip_slices = self.chip_slices[idx]
        # Squeeze to remove singleton dimension if time chip_size is 1
        data = [
            self.reader(self.rasters[i], read_window, self.chip_size[:2])
            for i in range(*read_times)
        ]
        data, mask = (np.squeeze(np.stack(x)) for x in zip(*data))

        write_info = {
            "write_window": write_window,
            "write_times": write_times,
            "chip_slices": chip_slices,
            "meta": self.meta,
        }
        return data, mask, write_info

    def get_filename(self, idx: int):
        return f"pred_{idx}.tif"


class StackOnChannelsChipDataset(ChipDataset):
    def __init__(
        self,
        rasters: List[List[T]],
        chip_size: Dims,
        step_size: Dims,
        downsampling: int = 1,
        nodata: Optional[float] = None,
        geometry_or_chunk: Optional[Union[BaseGeometry, RasterChunk]] = None,
        reader: Optional[
            Callable[[T, Window, Tuple[int, int]], Tuple[NDArray[Any], NDArray[Any]]]
        ] = None,
    ):
        super().__init__(
            rasters[0], chip_size, step_size, downsampling, nodata, geometry_or_chunk, reader
        )
        self.datasets = [
            ChipDataset(r, chip_size, step_size, downsampling, nodata, geometry_or_chunk, reader)
            for r in rasters
        ]
        for attr in ("width", "height", "crs", "transform"):
            for d in self.datasets:
                ref_attr = getattr(self, attr)
                comp_attr = getattr(d, attr)
                if (attr == "transform" and not affine_all_close(ref_attr, comp_attr)) or (
                    attr != "transform" and ref_attr != comp_attr
                ):
                    raise ValueError(
                        f"Expected '{attr}' to be the same for all datasets, found "
                        f"{ref_attr} != {comp_attr}"
                    )

    def __getitem__(self, idx: int) -> ChipDataType:
        # Convert sequence of tuples to tuple of sequences
        # (d, i), (d, i), (d, i) -> (d, d, d), (i, i, i)
        chip_data, chip_mask, chip_info = zip(*(d[idx] for d in self.datasets))
        chip_data = cast(List[NDArray[Any]], chip_data)
        chip_mask = cast(List[NDArray[Any]], chip_mask)
        chip_info = cast(List[Dict[str, str]], chip_info)
        assert all(
            chip_info[0][k] == c[k]
            for c in chip_info
            for k in ("write_window", "write_times", "chip_slices")
        )
        chip_data = np.concatenate([c[None] if c.ndim == 2 else c for c in chip_data])
        chip_mask = np.concatenate([c[None] if c.ndim == 2 else c for c in chip_mask])
        return chip_data, chip_mask, chip_info[0]


def custom_collate(
    samples: List[ChipDataType],
) -> Tuple[NDArray[Any], NDArray[Any], List[Dict[str, Any]]]:
    """Custom function for joining samples from `ChipDataset` into a batch"""
    chip_data, chip_mask, write_info = zip(*samples)
    chip_data = cast(List[NDArray[Any]], chip_data)
    chip_mask = cast(List[NDArray[Any]], chip_mask)
    write_info = cast(List[Dict[str, Any]], write_info)
    return collate_data(chip_data), collate_data(chip_mask), write_info


@overload
def collate_data(data: List[NDArray[Any]]) -> NDArray[Any]: ...


@overload
def collate_data(data: Dict[Any, NDArray[Any]]) -> Dict[Any, NDArray[Any]]: ...


@overload
def collate_data(data: NDArray[Any]) -> NDArray[Any]: ...


def collate_data(
    data: Union[List[NDArray[Any]], Dict[Any, NDArray[Any]], NDArray[Any]],
) -> Union[Dict[Any, NDArray[Any]], NDArray[Any]]:
    if isinstance(data, dict):
        return {k: collate_data(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        if isinstance(data[0], np.ndarray):
            return np.stack(data)
    if isinstance(data, np.ndarray):
        return data

    raise ValueError(f"Invalid type {type(data)} for collate function.")


def get_loader(
    dataset: ChipDataset,
    batch_size: int,
    num_workers: int = 1,
    collate_fn: Callable[
        [List[ChipDataType]], Tuple[NDArray[Any], NDArray[Any], List[Dict[str, Any]]]
    ] = custom_collate,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,  # type: ignore
        num_workers=num_workers,
    )


def predict_chips(
    model: ort.InferenceSession,
    dataloader: DataLoader[ChipDataType],
    out_dir: str,
    skip_nodata: bool,
    pre_process: Callable[[NDArray[Any], NDArray[Any]], NDArray[Any]] = lambda x, _: x,
    post_process: Callable[[NDArray[Any], NDArray[Any], NDArray[Any]], NDArray[Any]] = lambda *x: x[
        -1
    ],
) -> List[str]:
    """
    Function to extract chips, compute model predictions, and save to disk.

    Optionally accepts `pre_process` and `post_process` functions which are
    called before and after model predictions, respectively.
    """
    filepaths: List[str] = []
    dataset = cast(ChipDataset, dataloader.dataset)
    get_filename = dataset.get_filename
    out_shape: Optional[Tuple[int, ...]] = None
    for batch_idx, batch in enumerate(dataloader):
        LOGGER.info(f"Running model for batch ({batch_idx + 1}/{len(dataloader)})")
        chip_data, chip_mask, write_info_list = batch
        if skip_nodata and chip_mask.all():
            if out_shape is None:
                # Run the model to get the output shape
                model_inputs = pre_process(chip_data, chip_mask)
                out_shape = model.run(None, {model.get_inputs()[0].name: model_inputs})[0].shape[1:]
            LOGGER.info(f"Skipping batch of nodata ({batch_idx+1})")
            assert out_shape is not None
            model_out = dataset.nodata * np.ones((chip_data.shape[0], *out_shape))
        else:
            model_inputs = pre_process(chip_data, chip_mask)
            model_out = model.run(None, {model.get_inputs()[0].name: model_inputs})[0]
            out_shape = model_out.shape[1:]  # ignore batch size
        post_out = post_process(chip_data, chip_mask, model_out)
        write_prediction_to_file(
            post_out, chip_mask, write_info_list, out_dir, filepaths, get_filename
        )
    return filepaths


def write_prediction_to_file(
    chip_data: NDArray[Any],
    chip_mask: NDArray[Any],
    write_info_list: List[Dict[str, Any]],
    out_dir: str,
    filepaths: List[str],
    get_filename: Callable[[int], str],
):
    for out, mask, write_info in zip(chip_data, chip_mask, write_info_list):
        if out.ndim == 3:
            out = out[None]  # Create singleton time dimension if necessary
        if mask.ndim == 3:
            mask = mask[None]
        chip_times, chip_rows, chip_cols = write_info["chip_slices"]
        for write_t, chip_t in zip(range(*write_info["write_times"]), range(*chip_times)):
            filename = get_filename(write_t)
            filepath = os.path.join(out_dir, filename)
            if filepath not in filepaths:
                filepaths.append(filepath)
            write_window_to_file(
                out[chip_t, :, slice(*chip_rows), slice(*chip_cols)],
                mask[chip_t, :, slice(*chip_rows), slice(*chip_cols)].any(axis=0),
                write_info["write_window"],
                filepath,
                write_info["meta"],
            )
