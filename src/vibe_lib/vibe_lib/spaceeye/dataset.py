# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Module for generating inputs for the SpaceEye model.

It includes code for splitting the RoI into chips of adequate size, loading and
normalizing Sentinel 1 and 2 data, doing illuminance normalization, and
generating the windows for writing predictions to file.

The main idea is that we only load the necessary data to perform inference, and
write predictions to disk as they are done, to avoid loading the whole thing
into memory.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, TypeVar, Union, cast

import geopandas as gpd
import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio import Affine
from rasterio.mask import raster_geometry_mask
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from shapely.geometry.base import BaseGeometry
from torch.utils.data import Dataset

from vibe_core.data import Sentinel1Raster, Sentinel2Raster
from vibe_core.data.core_types import AssetVibe
from vibe_core.data.sentinel import (
    Sentinel1RasterTileSequence,
    Sentinel2CloudMaskTileSequence,
    Sentinel2RasterTileSequence,
)

from .illumination import interpolate_illuminance, masked_average_illuminance
from .utils import QUANTIFICATION_VALUE

EPS = 1e-10
LOGGER = logging.getLogger(__name__)


class Dims(NamedTuple):
    width: int
    height: int
    time: int


Interval = Tuple[int, int]

DatasetReturnType = Tuple[
    Dict[str, NDArray[Any]],
    Dict[str, Union[Window, Interval, Tuple[Interval, Interval, Interval]]],
]

TileSequenceData = Union[
    Sentinel1RasterTileSequence,
    Sentinel2RasterTileSequence,
    Sentinel2CloudMaskTileSequence,
]

T = TypeVar("T", Sentinel1Raster, Sentinel2Raster)
NDArrayInt = NDArray[np.int_]


def get_read_intervals(
    dim_size: int, chip_size: int, step: int, offset: int
) -> Tuple[NDArrayInt, NDArrayInt]:
    """
    Divide total dim size in intervals by using an approximate step
    Actual step is computed by rounding the step so that the number of windows
        is the rounded number of windows with the desired step
    """
    if dim_size < chip_size:
        raise ValueError(
            f"{dim_size=} cannot be smaller than {chip_size=}. "
            "Please consider reducing the step/chip size or increasing the input geometry."
        )

    # Effects of using round versus ceil for determining step size:
    # With round:
    # This number of blocks should have the step be at most 1.5x the original step
    # Which should only happen when the chip size is quite big compared to the dimension size
    # With ceil: step size should be at most the chosen step
    num_blocks = int(np.ceil((dim_size - chip_size) / step)) + 1
    # Make sure we capture the whole area if dim_size is barely larger
    if dim_size > chip_size:
        num_blocks = max(num_blocks, 2)
    start = np.round(np.linspace(0, dim_size - chip_size, num_blocks)).astype(int)
    end = np.clip(start + chip_size, 0, dim_size)
    assert end[-1] == dim_size, f"{end[-1]=} != {dim_size}"
    return start + offset, end + offset


def get_write_intervals(
    dim_size: int, chip_size: int, step: int, offset: int
) -> Tuple[Tuple[NDArrayInt, NDArrayInt], Tuple[NDArrayInt, NDArrayInt]]:
    """
    Divide total dim size in non-overlapping intervals which divide the overlap
    sections according to proximity to the center of the interval
    """
    read_start, read_end = get_read_intervals(dim_size, chip_size, step, offset)
    edges = np.concatenate((read_start[:1], (read_end[:-1] + read_start[1:]) // 2, read_end[-1:]))
    write_start = edges[:-1].astype(int)
    write_end = edges[1:].astype(int)
    chip_start = write_start - read_start
    chip_end = write_end - read_start
    return (write_start, write_end), (chip_start, chip_end)


def get_read_windows(
    width: int, height: int, time_length: int, chip_size: Dims, step: Dims, offset: Dims
) -> List[Tuple[Window, Interval]]:
    """
    Generate read windows for a tensor with width, height, and time_length.
    The windows are generated according to chip_size, step and offset (for all three dimensions).
    The offset is used to start the first read window in the RoI boundary.
    """
    return [
        (
            Window.from_slices(rows, cols),
            time,
        )
        for time in zip(*get_read_intervals(time_length, chip_size.time, step.time, offset.time))
        for rows in zip(*get_read_intervals(height, chip_size.height, step.height, offset.height))
        for cols in zip(*get_read_intervals(width, chip_size.width, step.width, offset.width))
    ]


def get_write_windows(
    width: int, height: int, time_length: int, chip_size: Dims, step: Dims
) -> Tuple[List[Tuple[Window, Interval]], List[Tuple[Interval, Interval, Interval]]]:
    """
    Generate write windows for a tensor with width, height, and time_length.
    The windows are generated according to chip_size and step (for all three dimensions).
    """
    col_intervals, chip_col_intervals = get_write_intervals(width, chip_size.width, step.width, 0)
    row_intervals, chip_row_intervals = get_write_intervals(
        height, chip_size.height, step.height, 0
    )
    time_intervals, chip_time_intervals = get_write_intervals(
        time_length, chip_size.time, step.time, 0
    )
    return (
        [
            (
                Window.from_slices(rows, cols),
                time,
            )
            for time in zip(*time_intervals)
            for rows in zip(*row_intervals)
            for cols in zip(*col_intervals)
        ],
        [
            (chip_time, chip_rows, chip_cols)
            for chip_time in zip(*chip_time_intervals)
            for chip_rows in zip(*chip_row_intervals)
            for chip_cols in zip(*chip_col_intervals)
        ],
    )


def adjust_dim(
    window_dim: float, window_ranges: Tuple[float, float], chip_dim: float, raster_bounds: float
) -> Tuple[float, float]:
    """
    Adjust a window's dimension (width or height) to make sure the window reaches the chip size
    while still within the raster bounds.

    Args:
        chip_dim: The chip dimension (width or height).
        window_dim: The window dimension (width or height).
        window_ranges: The window ranges (start, end).
        raster_bounds: The raster dimension (width or height).

    Returns:
        The adjusted window ranges.
    """
    diff = chip_dim - window_dim
    offset = diff // 2

    offset_low = offset if window_ranges[0] - offset >= 0 else window_ranges[0]
    offset_high = diff - offset_low
    if offset_high + window_ranges[1] > raster_bounds:
        offset_high = raster_bounds - window_ranges[1]
        offset_low = diff - offset_high

    min_dim = max(window_ranges[0] - offset_low, 0)
    max_dim = window_ranges[1] + offset_high

    return min_dim, max_dim


class SpaceEyeReader(Dataset[DatasetReturnType]):
    """Dataset that lazily reads chips from sentinel 1 and 2 rasters.
    The dataset computes the necessary chips to cover the whole RoI according to
        chip size and overlap, and generates input data, as well as write windows
        for each chip.
    It also includes preprocessing steps such as input standardization,
        discarding very cloud days illuminance normalization
    Input data is a daily tensor with padding on non-available days.
    """

    def __init__(
        self,
        s1_items: Optional[Sentinel1RasterTileSequence],
        s2_items: Sentinel2RasterTileSequence,
        cloud_masks: Sentinel2CloudMaskTileSequence,
        time_range: Tuple[datetime, datetime],
        geometry: BaseGeometry,
        chip_size: Dims,
        overlap: Tuple[float, float, float],
        s2_bands: List[int],
        min_clear_ratio: float,
        normalize_illuminance: bool,
    ):
        self.s1_items = s1_items
        self.s2_items = s2_items
        self.cloud_masks = cloud_masks
        ref_item = s2_items.assets[0]
        self.time_range = time_range
        self.geometry = geometry
        self.chip_size = chip_size
        self.min_clear_ratio = min_clear_ratio
        if any((o < 0) or (o >= 1) for o in overlap):
            raise ValueError(f"Overlap values must be in range [0, 1), found {overlap}")
        self.overlap = overlap
        self.step = Dims(*(int(s * (1 - o)) for s, o in zip(chip_size, overlap)))
        self.s2_bands = s2_bands
        self.normalize_illuminance = normalize_illuminance
        self.time_length = (self.time_range[1] - self.time_range[0]).days + 1
        if self.time_length != self.chip_size.time:
            raise ValueError(
                f"Expected time length = {self.time_length} to be the same as "
                f"chip size = {self.chip_size.time}"
            )
        self.write_range = s2_items.write_time_range
        self.write_indices = (
            (self.write_range[0] - self.time_range[0]).days,
            (self.write_range[1] - self.time_range[0]).days + 1,
        )

        with rasterio.open(ref_item.url) as src:
            # Assuming all products are from the same tile for now
            self.crs = src.crs
            self.raster_width: int = src.width
            self.raster_height: int = src.height
            # Compute envelope in native CRS to avoid nodata
            box = gpd.GeoSeries(geometry, crs="epsg:4326").to_crs(self.crs).iloc[0].envelope
            window = cast(Window, raster_geometry_mask(src, [box], all_touched=True, crop=True)[2])
            # Adjust window to make sure it is not too small
            window = self._adjust_roi_window(window)
            # Compute the transform with the adjusted window
            self.transform: Affine = window_transform(window, src.transform)
            self.width: int = window.width
            self.height: int = window.height
        self.roi = box
        self.offset = Dims(window.col_off, window.row_off, 0)
        self.roi_window = window
        read_windows = get_read_windows(
            self.width, self.height, self.time_length, self.chip_size, self.step, self.offset
        )
        write_windows, chip_slices = get_write_windows(
            self.width, self.height, self.time_length, self.chip_size, self.step
        )
        assert all(i == write_windows[0][1] for _, i in write_windows)
        assert all(i == chip_slices[0][0] for i, _, _ in chip_slices)
        # Overwrite time indices by what we get from the input sequence
        write_windows = [(w, self.write_indices) for w, _ in write_windows]
        chip_slices = [(self.write_indices, h, w) for _, h, w in chip_slices]

        assert len(read_windows) == len(write_windows) == len(chip_slices)
        self.s1_indices = self._get_indices(self.s1_items) if self.s1_items is not None else None
        self.s2_indices = self._get_s2_indices(self.s2_items, self.cloud_masks)

        # Filter out windows without any cloud-free data
        valid_idx = [idx for idx in self.s2_indices if idx != -1]

        if valid_idx:
            self.read_windows = cast(List[Tuple[Window, Interval]], read_windows)
            self.write_windows = cast(List[Tuple[Window, Interval]], write_windows)
            self.chip_slices = cast(List[Tuple[Interval, Interval, Interval]], chip_slices)
        else:
            self.read_windows, self.write_windows, self.chip_slices = [], [], []
        assert len(self.read_windows) == len(self.write_windows) == len(self.chip_slices)

        self.illuminance = self._get_illumination_array()

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

        hs, ws = window.toranges()

        min_h, max_h = adjust_dim(window.height, hs, height, self.raster_height)
        min_w, max_w = adjust_dim(window.width, ws, width, self.raster_width)

        new_win = Window.from_slices((min_h, max_h), (min_w, max_w))
        LOGGER.info(f"Adjusting from {window} to {new_win}")
        return new_win

    def _get_indices(self, sequence: TileSequenceData) -> List[int]:
        """
        Get timestep indices for each asset in the sequence.
        Assuming daily predictions here. Not supporting multiple day intervals.
        For a generic timestep we would need to treat possible collisions, i.e.,
        multiple products on the same timestep index. This is not currently treated here.
        """
        asset_list = sequence.get_ordered_assets()
        start = sequence.asset_time_range[asset_list[0].id][0]
        return [(sequence.asset_time_range[a.id][0] - start).days for a in asset_list]

    def _get_clear_ratio(self, cloud_mask_asset: AssetVibe) -> int:
        mask = self._read_cloud_mask(
            cloud_mask_asset,
            np.zeros(1, dtype=bool),
            self.roi_window,
        )
        return (mask == 1).mean()

    def _get_s2_indices(
        self,
        s2_sequence: Sentinel2RasterTileSequence,
        cloud_mask_sequence: Sentinel2CloudMaskTileSequence,
    ) -> List[int]:
        """
        Get indices and remove items that have too much cloud cover. To do so,
        we consider that each asset in the same (ordered) position in s2_sequence
        and cloud_mask_sequence is associated.
        """
        indices = self._get_indices(s2_sequence)
        return [
            index if self._get_clear_ratio(cloudmask_item) > self.min_clear_ratio else -1
            for index, cloudmask_item in zip(indices, cloud_mask_sequence.get_ordered_assets())
        ]

    def _get_illumination_array(self) -> NDArray[np.float32]:
        """
        Compute the illumance array for each available product in the RoI
        The illuminance for days where there is no data (or not enough cloudless
            data) is obtained through interpolation
        """
        if not self.normalize_illuminance:
            return np.ones((len(self.s2_bands), self.time_length, 1, 1), dtype=np.float32)
        illuminance = np.zeros((len(self.s2_bands), self.time_length, 1, 1), dtype=np.float32)
        mask_ar = np.zeros((1, self.time_length, 1, 1), dtype=np.float32)
        for s2_asset, cloud_mask_asset, index in zip(
            self.s2_items.get_ordered_assets(),
            self.cloud_masks.get_ordered_assets(),
            self.s2_indices,
        ):
            if 0 <= index < self.time_length:
                x, m = self._read_s2(s2_asset, self.roi_window, cloud_mask_asset)
                m = m == 1
                clear_ratio = m.mean()
                if clear_ratio < self.min_clear_ratio:
                    LOGGER.warning(
                        "Discarding sentinel data for illumination computation with date "
                        f"{self.s2_items.asset_time_range[s2_asset.id][0]} (index {index}) because "
                        f"clear_ratio {clear_ratio:.1%} < threshold {self.min_clear_ratio:.1%}"
                    )
                    continue
                illum_ar = masked_average_illuminance(x, m.astype(np.float32))
                illuminance[:, index] = illum_ar
                mask_ar[:, index] = 1
        if mask_ar.sum() == 0:
            LOGGER.warning("No cloudless day available for illuminance calculation.")
            return np.ones((len(self.s2_bands), self.time_length, 1, 1), dtype=np.float32)
        return interpolate_illuminance(illuminance, mask_ar)

    @staticmethod
    def _read_data(
        file_ref: str, window: Window, bands: Optional[List[int]] = None
    ) -> NDArray[Any]:
        """
        Read a window of data from a file
        """
        offset_bands = [b + 1 for b in bands] if bands else None
        with rasterio.open(file_ref) as src:
            return src.read(indexes=offset_bands, window=window)

    def _read_s2(
        self,
        s2_asset: AssetVibe,
        window: Window,
        cloud_mask_asset: AssetVibe,
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        Read a window sentinel 2 data and the associated cloud mask
        """
        # Read s2 data
        s2_data = self._read_data(s2_asset.url, window, self.s2_bands)
        nodata = s2_data.sum(axis=0, keepdims=True) == 0
        s2_data = s2_data.astype(np.float32) / QUANTIFICATION_VALUE
        cloud_mask = self._read_cloud_mask(cloud_mask_asset, nodata, window)
        return s2_data, cloud_mask

    def _read_cloud_mask(
        self, cloud_mask_asset: AssetVibe, nodata: NDArray[np.bool8], window: Window
    ) -> NDArray[np.float32]:
        """
        Read a cloud mask and change the binary mask to the format expected by the model
        """
        # Read cloud mask
        cloud_mask = self._read_data(cloud_mask_asset.url, window, [0])
        # Use this masking for now for compatibility purposes
        # TODO: Change the model to receive a binary mask for Sentinel2 as well
        cloud_mask[cloud_mask == 1] = 2
        cloud_mask[cloud_mask == 0] = 1
        # Add nodata as cloud
        cloud_mask[nodata] = 2
        return cloud_mask.astype(np.float32)

    def _read_s1(
        self, s1_asset: AssetVibe, window: Window, _
    ) -> Tuple[NDArray[np.float32], NDArray[np.bool8]]:
        filepath = s1_asset.url
        s1 = self._read_data(filepath, window, None)
        s1_available = np.sum(np.abs(s1), axis=0) > 0
        s1 = (s1 + 20.0) / 40.0
        s1[:, ~s1_available] = 0.0  # just to make it match the images that are completely missing.
        return s1, s1_available

    def _get_data_array(
        self,
        items: List[AssetVibe],
        mask_items: List[Optional[AssetVibe]],
        indices: List[int],
        read_times: Interval,
        read_window: Window,
        read_callback: Callable[
            [AssetVibe, Window, Optional[AssetVibe]], Tuple[NDArray[np.float32], NDArray[Any]]
        ],
    ) -> Tuple[NDArray[np.float32], NDArray[Any]]:
        """
        Get data array which will be used as input to the network.
        This is done by selecting data inside the time range of the input
            and inserting it in the correct time index
        """
        x = None
        mask = None
        # Closed at beginning, open at ending
        read_start, read_end = read_times
        for item, mask_item, index in zip(items, mask_items, indices):
            if read_start <= index < read_end:
                chip_data, chip_mask = read_callback(item, read_window, mask_item)
                if x is None:
                    x = np.zeros(
                        (
                            chip_data.shape[0],
                            self.chip_size.time,
                            self.chip_size.height,
                            self.chip_size.width,
                        ),
                        dtype=np.float32,
                    )
                if mask is None:
                    mask = np.zeros(
                        (1, self.chip_size.time, self.chip_size.height, self.chip_size.width),
                        dtype=chip_mask.dtype,
                    )
                x[:, index - read_start] = chip_data
                mask[:, index - read_start] = chip_mask
        if x is None or mask is None:
            start_time = (self.time_range[0] + timedelta(days=int(read_start))).isoformat()
            end_time = (self.time_range[0] + timedelta(days=int(read_end))).isoformat()
            raise RuntimeError(
                f"Could not find any cloud-free data from dates {start_time} to {end_time}"
            )
        return x, mask

    def __getitem__(self, idx: int) -> DatasetReturnType:
        # Tensors are C x T x H x W
        read_window, read_times = self.read_windows[idx]

        s2_data, s2_mask = self._get_data_array(
            self.s2_items.get_ordered_assets(),
            self.cloud_masks.get_ordered_assets(),  # type: ignore
            self.s2_indices,
            read_times,
            read_window,
            self._read_s2,  # type: ignore
        )
        # Get data on where to write in the file
        write_window, write_times = self.write_windows[idx]
        # Which part of the predictions will be written
        chip_slices = self.chip_slices[idx]
        # Illuminance values for the chip
        chip_illuminance = self.illuminance[:, read_times[0] : read_times[1]]

        # Data we feed into the network
        chip_data = {
            "S2": s2_data / (chip_illuminance + np.float32(EPS)),
            "cloud_label": s2_mask,
            "illuminance": chip_illuminance,
        }
        if self.s1_items is not None:
            s1_sorted_assets = self.s1_items.get_ordered_assets()
            # Read data
            s1_data, s1_mask = self._get_data_array(
                s1_sorted_assets,
                [None for _ in range(len(s1_sorted_assets))],
                cast(List[int], self.s1_indices),
                read_times,
                read_window,
                self._read_s1,
            )
            chip_data.update({"S1": s1_data, "S1_mask": s1_mask})
        # Information for writing in the files
        write_info = {
            "write_window": write_window,
            "write_times": write_times,
            "chip_slices": chip_slices,
        }

        return chip_data, write_info

    def __len__(self) -> int:
        return len(self.read_windows)
