import os
from datetime import datetime
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np
import pytest
import xarray as xr
from shapely import geometry as shpg

from vibe_core.data import Raster
from vibe_dev.testing.op_tester import OpTester
from vibe_lib.raster import load_raster, save_raster_to_asset

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "recode_raster.yaml")


@pytest.fixture
def tmp_dir():
    _tmp_dir = TemporaryDirectory()
    yield _tmp_dir.name
    _tmp_dir.cleanup()


@pytest.fixture
def fake_raster(tmp_dir: str):
    nbands = 3
    x = 128
    y = 128

    fake_data = np.random.randint(0, 4, size=(nbands, y, x)).astype(np.float32)
    fake_da = xr.DataArray(
        fake_data,
        coords={"bands": np.arange(nbands), "x": np.linspace(0, 1, x), "y": np.linspace(0, 1, y)},
        dims=["bands", "y", "x"],
    )
    fake_da.rio.write_crs("epsg:4326", inplace=True)

    asset = save_raster_to_asset(fake_da, tmp_dir)
    return Raster(
        id="fake_id",
        time_range=(datetime(2023, 1, 1), datetime(2023, 1, 1)),
        geometry=shpg.mapping(shpg.box(*fake_da.rio.bounds())),
        assets=[asset],
        bands={j: i for i, j in enumerate(["B1", "B2", "B3"])},
    )


def test_recode_raster(fake_raster: Raster):
    op = OpTester(CONFIG_PATH)
    parameters = {
        "from_values": [0, 1, 2, 3],
        "to_values": [4, 5, 6, 7],
    }

    op.update_parameters(parameters)
    output = op.run(raster=fake_raster)
    assert output

    raster = cast(Raster, output["recoded_raster"])
    raster_data = load_raster(raster)
    fake_raster_data = load_raster(fake_raster)

    # Assert that the recoded raster has the same shape as the original
    assert raster_data.shape == fake_raster_data.shape
    # Assert fake_raster_data - raster values is always 4
    assert np.all(raster_data - fake_raster_data == 4)


def test_recode_not_mapped_values(fake_raster: Raster):
    op = OpTester(CONFIG_PATH)

    parameters = {
        "from_values": [10, 11, 12, 13],
        "to_values": [-1, -2, -3, -4],
    }

    op.update_parameters(parameters)
    output = op.run(raster=fake_raster)
    assert output

    raster = cast(Raster, output["recoded_raster"])
    raster_data = load_raster(raster)
    fake_raster_data = load_raster(fake_raster)

    # Assert that the recoded raster has the same shape as the original
    assert raster_data.shape == fake_raster_data.shape

    # Assert fake_raster_data and raster_data are the same
    assert np.all(raster_data == fake_raster_data)

    # Assert raster_data has no negative values
    assert np.all(raster_data >= 0)


def test_recode_raster_different_lengths(fake_raster: Raster):
    op = OpTester(CONFIG_PATH)
    parameters = {
        "from_values": [0, 1, 2],
        "to_values": [4, 5, 6, 7],
    }

    op.update_parameters(parameters)
    with pytest.raises(ValueError):
        op.run(raster=fake_raster)
