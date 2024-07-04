import os
from datetime import datetime
from typing import cast

import numpy as np
import pandas as pd
import pytest
import shapely.geometry as shpg
import xarray as xr
from compute_pixel_count import COUNTS_COLUMN, UNIQUE_VALUES_COLUMN

from vibe_core.data import Raster, RasterPixelCount
from vibe_dev.testing.op_tester import OpTester
from vibe_lib.raster import save_raster_to_asset

NBANDS = 3
FAKE_RASTER_DATA = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]]).astype(np.float32)
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "compute_pixel_count.yaml")

EXPECTED_UNIQUE_VALUES = [0, 1, 2]
# We are using 3 bands, so we expect 9 counts for each unique value
EXPECTED_COUNTS = [9, 9, 9]


@pytest.fixture
def fake_raster(tmpdir: str):
    x = 3
    y = 3

    fake_data = FAKE_RASTER_DATA
    fake_data = [fake_data] * NBANDS

    fake_da = xr.DataArray(
        fake_data,
        coords={"bands": np.arange(NBANDS), "x": np.linspace(0, 1, x), "y": np.linspace(0, 1, y)},
        dims=["bands", "y", "x"],
    )

    fake_da.rio.write_crs("epsg:4326", inplace=True)
    asset = save_raster_to_asset(fake_da, tmpdir)

    return Raster(
        id="fake_id",
        time_range=(datetime(2023, 1, 1), datetime(2023, 1, 1)),
        geometry=shpg.mapping(shpg.box(*fake_da.rio.bounds())),
        assets=[asset],
        bands={j: i for i, j in enumerate(["B1", "B2", "B3"])},
    )


def test_compute_pixel_count(fake_raster: Raster):
    op = OpTester(CONFIG_PATH)

    output = op.run(raster=fake_raster)
    assert output
    assert "pixel_count" in output

    pixel_count = cast(RasterPixelCount, output["pixel_count"])
    assert len(pixel_count.assets) == 1

    asset_path = pixel_count.assets[0].path_or_url
    assert os.path.exists(asset_path)

    # Read the CSV file
    df = pd.read_csv(asset_path)

    # Check the columns
    assert UNIQUE_VALUES_COLUMN in df.columns  # type: ignore
    assert COUNTS_COLUMN in df.columns  # type: ignore

    # Check the values
    assert np.array_equal(df[UNIQUE_VALUES_COLUMN].values, EXPECTED_UNIQUE_VALUES)  # type: ignore
    assert np.array_equal(df[COUNTS_COLUMN].values, EXPECTED_COUNTS)  # type: ignore
