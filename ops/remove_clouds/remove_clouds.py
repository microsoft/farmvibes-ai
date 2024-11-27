# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pyright: reportUnknownMemberType=false
import logging
import os
from abc import abstractmethod
from datetime import datetime, timedelta
from tempfile import TemporaryDirectory
from typing import Dict, Optional, Union, cast

import geopandas as gpd
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from shapely import geometry as shpg
from torch.utils.data import DataLoader

from vibe_core.data import AssetVibe, gen_guid
from vibe_core.data.sentinel import (
    S2ProcessingLevel,
    Sentinel1RasterTileSequence,
    Sentinel2CloudMaskTileSequence,
    Sentinel2RasterTileSequence,
    SpaceEyeRasterSequence,
)
from vibe_lib.raster import INT_COMPRESSION_KWARGS, compress_raster, write_window_to_file
from vibe_lib.spaceeye.dataset import Dims, SpaceEyeReader
from vibe_lib.spaceeye.illumination import add_illuminance
from vibe_lib.spaceeye.interpolation import DampedInterpolation
from vibe_lib.spaceeye.utils import QUANTIFICATION_VALUE, SPACEEYE_TO_SPYNDEX_BAND_NAMES

S1_NUM_BANDS = 2
S2_NUM_BANDS = 10
L1C_BAND_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]
L2A_BAND_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
FILENAME_TEMPLATE = "preds_{}.tif"

LOGGER = logging.getLogger(__name__)


def get_filename(date: datetime) -> str:
    return FILENAME_TEMPLATE.format(date.strftime("%Y%m%d"))


def remove_clouds(
    model: Union[ort.InferenceSession, nn.Module],
    dataset: SpaceEyeReader,
    out_dir: str,
    num_workers: int,
) -> SpaceEyeRasterSequence:
    # TODO: Add meta to write_info dict
    meta = {
        "driver": "GTiff",
        "height": dataset.height,
        "width": dataset.width,
        "count": S2_NUM_BANDS,
        "crs": dataset.crs,
        "dtype": "uint16",
        "transform": dataset.transform,
        "nodata": 0,
    }
    # Use batch size 1
    dataloader = DataLoader(dataset, collate_fn=lambda x: x, num_workers=num_workers)
    total_chips = len(dataloader)
    start_datetime = dataset.time_range[0]
    for chip_idx, batch in enumerate(dataloader):
        chip_data, write_info = batch[0]
        t1, t2 = (
            (start_datetime + timedelta(days=t)).strftime("%Y-%m-%d")
            for t in write_info["write_times"]
        )
        write_window = write_info["write_window"]
        (r1, r2), (c1, c2) = write_window.toranges()
        LOGGER.info(
            f"Running model for {t1}:{t2}, extent {r1}:{r2}, {c1}:{c2} "
            f"({chip_idx + 1}/{total_chips})"
        )
        inputs = {k: v[None] for k, v in chip_data.items() if k != "illuminance"}
        with torch.inference_mode():
            if isinstance(model, nn.Module):
                inputs = {k: torch.from_numpy(v) for k, v in inputs.items()}
                s2 = cast(nn.Module, model)(inputs).numpy()
            else:
                s2 = cast(ort.InferenceSession, model).run(None, inputs)[0]
            s2 = s2[0, :]
        # Put illumination back
        s2 = (add_illuminance(s2, chip_data["illuminance"]) * QUANTIFICATION_VALUE).astype(
            np.uint16
        )
        chip_times, chip_rows, chip_cols = write_info["chip_slices"]
        for write_t, chip_t in zip(range(*write_info["write_times"]), range(*chip_times)):
            date = start_datetime + timedelta(days=write_t)
            filename = get_filename(date)
            filepath = os.path.join(out_dir, filename)
            write_window_to_file(
                s2[:, chip_t, slice(*chip_rows), slice(*chip_cols)],
                None,
                write_window,
                filepath,
                meta,
            )

    # Create a SpaceEyeRasterSequence with the sequence metadata
    ref_sequence = dataset.s2_items
    geom = shpg.mapping(gpd.GeoSeries(dataset.roi, crs=dataset.crs).to_crs("epsg:4326").iloc[0])
    spaceeye_sequence = SpaceEyeRasterSequence.clone_from(
        ref_sequence,
        assets=[],
        id=gen_guid(),
        geometry=geom,
        time_range=dataset.time_range,
        bands={name: idx for idx, name in enumerate(SPACEEYE_TO_SPYNDEX_BAND_NAMES.values())},
    )

    geom = shpg.shape(geom)

    # Add each raster asset to the sequence
    for time_idx in range(dataset.time_length):
        date = start_datetime + timedelta(days=time_idx)
        filename = get_filename(date)
        filepath = os.path.join(out_dir, filename)
        # Skip file if no predictions were made (not enough data)
        if not os.path.exists(filepath):
            continue
        guid = gen_guid()
        out_path = os.path.join(out_dir, f"{guid}.tif")
        LOGGER.info(f"Compressing raster for {date.strftime('%Y-%m-%d')}")
        compress_raster(filepath, out_path, **INT_COMPRESSION_KWARGS)
        asset = AssetVibe(reference=out_path, type="image/tiff", id=guid)
        spaceeye_sequence.add_asset(asset, (date, date), geom)

    return spaceeye_sequence


class CallbackBuilder:
    def __init__(
        self,
        duration: int,
        window_size: int,
        spatial_overlap: float,
        min_clear_ratio: float,
        normalize_illuminance: bool,
        num_workers: int,
    ):
        self.duration = duration
        self.window_size = window_size
        self.spatial_overlap = spatial_overlap
        self.min_clear_ratio = min_clear_ratio
        self.normalize_illuminance = normalize_illuminance
        self.num_workers = num_workers
        self.tmp_dir = TemporaryDirectory()

    def get_dataset(
        self,
        s1_products: Optional[Sentinel1RasterTileSequence],
        s2_products: Sentinel2RasterTileSequence,
        cloud_masks: Sentinel2CloudMaskTileSequence,
    ) -> SpaceEyeReader:
        s2_bands = (
            L1C_BAND_INDICES
            if s2_products.processing_level == S2ProcessingLevel.L1C
            else L2A_BAND_INDICES
        )
        sequence_geom = shpg.shape(s2_products.geometry)
        sequence_time_range = s2_products.time_range
        dataset = SpaceEyeReader(
            s1_items=s1_products,
            s2_items=s2_products,
            cloud_masks=cloud_masks,
            time_range=sequence_time_range,
            geometry=sequence_geom,
            chip_size=Dims(width=self.window_size, height=self.window_size, time=self.duration),
            overlap=(self.spatial_overlap, self.spatial_overlap, 0),
            s2_bands=s2_bands,
            min_clear_ratio=self.min_clear_ratio,
            normalize_illuminance=self.normalize_illuminance,
        )
        return dataset

    @abstractmethod
    def get_model(self) -> Union[ort.InferenceSession, nn.Module]:
        raise NotImplementedError

    def __call__(self):
        def callback(
            s2_products: Sentinel2RasterTileSequence,
            cloud_masks: Sentinel2CloudMaskTileSequence,
            s1_products: Optional[Sentinel1RasterTileSequence] = None,
        ) -> Dict[str, SpaceEyeRasterSequence]:
            if not s2_products.assets or (s1_products is not None and not s1_products.assets):
                s1_str = (
                    "" if s1_products is None else f"Sentinel-1: {len(s1_products.assets)} assets"
                )
                LOGGER.warning(
                    "Received empty input sequence, output will be empty sequence. "
                    f"Sentinel-2: {len(s2_products.assets)} assets, {s1_str}"
                )
                spaceeye_sequence = SpaceEyeRasterSequence.clone_from(
                    s2_products,
                    assets=[],
                    id=gen_guid(),
                    bands={
                        name: idx
                        for idx, name in enumerate(SPACEEYE_TO_SPYNDEX_BAND_NAMES.values())
                    },
                )
                return {"spaceeye_sequence": spaceeye_sequence}
            model = self.get_model()
            dataset = self.get_dataset(s1_products, s2_products, cloud_masks)
            spaceeye_sequence = remove_clouds(model, dataset, self.tmp_dir.name, self.num_workers)

            return {"spaceeye_sequence": spaceeye_sequence}

        return callback

    def __del__(self):
        self.tmp_dir.cleanup()


class NNCallbackBuilder(CallbackBuilder):
    def __init__(
        self,
        model_path: str,
        duration: int,
        window_size: int,
        spatial_overlap: float,
        min_clear_ratio: float,
        normalize_illuminance: bool,
        num_workers: int,
    ):
        super().__init__(
            duration,
            window_size,
            spatial_overlap,
            min_clear_ratio,
            normalize_illuminance,
            num_workers,
        )
        self.model_path = model_path

    def get_model(self) -> ort.InferenceSession:
        return ort.InferenceSession(self.model_path)


class InterpolationCallbackBuilder(CallbackBuilder):
    def __init__(
        self,
        duration: int,
        window_size: int,
        spatial_overlap: float,
        min_clear_ratio: float,
        normalize_illuminance: bool,
        num_workers: int,
        damping_factor: float,
        tolerance: float,
        max_iterations: int,
        check_interval: int,
    ):
        super().__init__(
            duration,
            window_size,
            spatial_overlap,
            min_clear_ratio,
            normalize_illuminance,
            num_workers,
        )
        self.damping_factor = damping_factor
        self.tol = tolerance
        self.max_iter = max_iterations
        self.check_interval = check_interval

    def get_model(self):
        return DampedInterpolation(
            S2_NUM_BANDS,
            self.duration,
            damping_factor=self.damping_factor,
            tol=self.tol,
            max_iter=self.max_iter,
            check_interval=self.check_interval,
        )
