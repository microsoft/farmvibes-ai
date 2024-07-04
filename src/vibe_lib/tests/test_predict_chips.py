from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import rasterio
from rasterio.windows import Window
from shapely import geometry as shpg

from vibe_core.data import AssetVibe, Raster
from vibe_lib.spaceeye import chip

RASTER_SIZE = 256
RASTER_BANDS = 2


class MockDataset:
    spatial_size: int = 256
    channels: int = 3
    nodata: int = 100

    def __init__(self, start_idx: int, length: int):
        self.start_idx = start_idx
        self.length = length
        self.get_filename = None
        self.zeros = np.zeros((MockDataset.channels, self.spatial_size, self.spatial_size))
        self.ones = np.ones((MockDataset.channels, self.spatial_size, self.spatial_size))

    def __getitem__(self, idx: int):
        if idx < self.start_idx:
            return self.ones, self.zeros, {}
        return self.zeros, self.ones, {}

    def __len__(self):
        return self.length


@pytest.fixture()
def test_raster(tmp_path: Path):
    geom = shpg.mapping(shpg.box(0, 0, 1, 1))
    now = datetime.now()
    filepath = tmp_path / "test_raster.tif"
    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        width=RASTER_SIZE,
        height=RASTER_SIZE,
        count=RASTER_BANDS,
        dtype="float32",
        nodata=-1,
    ) as dst:
        dst.write(np.arange(RASTER_SIZE**2 * RASTER_BANDS).reshape(2, RASTER_SIZE, RASTER_SIZE))
    asset = AssetVibe(reference=str(filepath), type="image/tiff", id="asset_id")
    return Raster(id="1", geometry=geom, time_range=(now, now), bands={}, assets=[asset])


@pytest.mark.parametrize("start_idx, length", ((0, 5), (1, 5), (5, 5), (0, 100), (50, 100)))
@patch.object(chip, "write_prediction_to_file")
def test_skip_nodata(write_patch: MagicMock, start_idx: int, length: int):
    dataset = MockDataset(start_idx, length)
    loader = chip.get_loader(dataset, 1, num_workers=0)  # type: ignore
    model = MagicMock()
    model.run.return_value = 10 * np.ones((1, 5, dataset.spatial_size, dataset.spatial_size))
    chip.predict_chips(model, loader, "anything", skip_nodata=True)
    assert model.run.call_count == max(start_idx, 1)


@pytest.mark.filterwarnings("ignore: Dataset has no geotransform")
@pytest.mark.parametrize("downsampling", (1, 2, 8))
def test_in_memory_reader(downsampling: int, test_raster: Raster):
    out_shape = (16, 16)
    reader = chip.InMemoryReader(downsampling)
    reader._cache_raster = MagicMock(side_effect=reader._cache_raster)
    for offset in (0, 0, 1, 2):
        win = Window(
            offset * downsampling,  # type: ignore
            0,
            *(o * downsampling for o in out_shape),
        )
        x, m = reader(test_raster, win, out_shape=out_shape)
        assert x.shape[1:] == out_shape
        assert m.shape[1:] == out_shape
        x, m = reader(test_raster, win, out_shape=out_shape)
    reader._cache_raster.assert_called_once()
    assert reader.rasters[test_raster.id]["data"].shape == (
        RASTER_BANDS,
        RASTER_SIZE // downsampling,
        RASTER_SIZE // downsampling,
    )
