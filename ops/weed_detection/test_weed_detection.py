# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_bounds
from shapely import geometry as shpg
from weed_detection import OpenedRaster

from vibe_core.data import AssetVibe, Raster

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "weed_detection.yaml")


def test_wgs84_default_buffer(tmp_path: Path):
    bounds = (-47.1, -22.9, -47.0, -22.8)
    raster_path = tmp_path / "raster.tif"
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        width=10,
        height=10,
        count=1,
        dtype=np.uint8,
        crs="EPSG:4326",
        transform=from_bounds(*bounds, 10, 10),
    ) as dataset:
        dataset.write(np.ones((1, 10, 10), dtype=np.uint8))

    raster = Raster(
        id="raster",
        time_range=(datetime(2023, 1, 1), datetime(2023, 1, 1)),
        geometry=shpg.mapping(shpg.box(*bounds)),
        assets=[AssetVibe(reference=str(raster_path), type="image/tiff", id="asset")],
        bands={"band": 0},
    )
    with open(CONFIG_PATH) as config:
        buffer = yaml.safe_load(config)["parameters"]["buffer"]

    assert buffer == 0
    assert OpenedRaster(raster, buffer, None, -1, []).training_data.shape == (1, 100)
    with pytest.raises(ValueError, match="creates an empty or invalid geometry"):
        OpenedRaster(raster, -1, None, -1, [])
