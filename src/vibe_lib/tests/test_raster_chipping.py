from datetime import datetime
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import xarray as xr
from shapely import geometry as shpg

from vibe_core.data import Raster
from vibe_lib.raster import save_raster_to_asset
from vibe_lib.spaceeye.chip import ChipDataset
from vibe_lib.spaceeye.dataset import Dims, get_read_intervals, get_write_intervals

RASTER_SIZE = 256


@pytest.mark.parametrize("dim_size", [500, 10000])
@pytest.mark.parametrize("chip_ratio", [1, 2, 10, 100])
@pytest.mark.parametrize("step_ratio", [0.3, 0.5, 1.0])
@pytest.mark.parametrize("offset", [0, 5000])
def test_read_intervals(dim_size: int, chip_ratio: int, step_ratio: int, offset: int):
    chip_size = dim_size // chip_ratio
    step = int(step_ratio * chip_size)
    read_start, read_end = get_read_intervals(dim_size, chip_size, step, offset)
    assert np.all(read_end > read_start)
    # No empty space in reads
    assert np.all(read_start[1:] <= read_end[:-1])
    # All windows have the correct size
    assert np.all((read_end - read_start) == chip_size)
    # Don't make the step larger when adjusting
    assert np.all((read_start[1:] - read_start[:-1]) <= step)
    # Cover the whole thing
    assert read_start[0] == offset
    assert read_end[-1] == dim_size + offset


@pytest.mark.parametrize("dim_size", [500, 10000])
@pytest.mark.parametrize("chip_ratio", [1, 2, 10, 100])
@pytest.mark.parametrize("step_ratio", [0.3, 0.5, 1.0])
@pytest.mark.parametrize("offset", [0, 5000])
def test_write_intervals(dim_size: int, chip_ratio: int, step_ratio: int, offset: int):
    chip_size = dim_size // chip_ratio
    step = int(step_ratio * chip_size)
    read_start, read_end = get_read_intervals(dim_size, chip_size, step, offset)
    (write_start, write_end), (chip_start, chip_end) = get_write_intervals(
        dim_size, chip_size, step, offset
    )
    assert np.all(write_end > write_start)
    # Chip and window sizes are the same
    assert np.allclose(write_end - write_start, chip_end - chip_start)
    # No empty space and no intersection in writes
    assert np.all(write_start[1:] == write_end[:-1])
    # Don't try to write where we didn't read
    assert np.all(write_start >= read_start)
    assert np.all(write_end <= read_end)
    # Cover the whole thing
    assert write_start[0] == offset
    assert write_end[-1] == dim_size + offset


def test_chip_size_too_large():
    dim_size = 447
    chip_size = 448
    step = 0
    offset = 0
    with pytest.raises(ValueError):
        get_read_intervals(dim_size, chip_size, step, offset)
    with pytest.raises(ValueError):
        get_write_intervals(dim_size, chip_size, step, offset)


@pytest.fixture
def tmp_dir_name():
    _tmp_dir = TemporaryDirectory()
    yield _tmp_dir.name
    _tmp_dir.cleanup()


@pytest.fixture()
def test_raster(tmp_dir_name: str):
    geom = shpg.mapping(shpg.box(0, 0, RASTER_SIZE, RASTER_SIZE))
    now = datetime.now()
    raster_dim = (1, RASTER_SIZE, RASTER_SIZE)

    fake_data = np.zeros(raster_dim).astype(np.float32)
    fake_da = xr.DataArray(
        fake_data,
        coords={
            "bands": np.arange(raster_dim[0]),
            "x": np.linspace(0, 1, raster_dim[1]),
            "y": np.linspace(0, 1, raster_dim[2]),
        },
        dims=["bands", "y", "x"],
    )
    fake_da.rio.write_crs("epsg:4326", inplace=True)

    asset = save_raster_to_asset(fake_da, tmp_dir_name)
    return Raster(id="1", geometry=geom, time_range=(now, now), bands={}, assets=[asset])


def test_window_smaller_than_chip(test_raster: Raster):
    chip_size = RASTER_SIZE // 2

    # window of size 0.25 * RASTER_SIZE, while chip is 0.5 * RASTER_SIZE
    # RoI will need to be adjusted to match chip size
    roi_geometry = shpg.Polygon(shpg.box(0.25, 0.25, 0.5, 0.5))

    dataset = ChipDataset(
        rasters=[test_raster],
        chip_size=Dims(chip_size, chip_size, 1),
        step_size=Dims(chip_size, chip_size, 1),
        geometry_or_chunk=roi_geometry,
    )

    assert (dataset.raster_width, dataset.raster_height) == (RASTER_SIZE, RASTER_SIZE)
    assert (dataset.width, dataset.height) == (chip_size, chip_size)
    assert (dataset.roi_window.width, dataset.roi_window.height) == (chip_size, chip_size)
