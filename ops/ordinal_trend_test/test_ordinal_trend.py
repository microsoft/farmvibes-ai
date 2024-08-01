# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from datetime import datetime
from tempfile import TemporaryDirectory
from typing import Any, cast

import numpy as np
import pytest
from numpy._typing import NDArray
from shapely import geometry as shpg

from vibe_core.data import AssetVibe, OrdinalTrendTest, RasterPixelCount
from vibe_dev.testing.op_tester import OpTester

SIGNIFICANCE_LEVEL = 0.05
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "ordinal_trend_test.yaml")
CSV_HEADER = "unique_values,counts"


@pytest.fixture
def tmp_dir():
    _tmp_dir = TemporaryDirectory()
    yield _tmp_dir.name
    _tmp_dir.cleanup()


def fake_raster_pixel_count(
    tmp_dir: str, pixel_id: str, fake_stack_data: NDArray[Any]
) -> RasterPixelCount:
    file_path = os.path.join(tmp_dir, f"{pixel_id}.csv")
    time_range = (datetime(2023, 1, 1), datetime(2023, 12, 31))
    np.savetxt(file_path, fake_stack_data, delimiter=",", fmt="%d", comments="", header=CSV_HEADER)

    return RasterPixelCount(
        id=pixel_id,
        time_range=time_range,
        geometry=shpg.mapping(shpg.box(0, 0, 0, 0)),
        assets=[AssetVibe(reference=file_path, type="text/csv", id="fake_asset_id")],
    )


@pytest.fixture
def fake_pixel_count0(tmp_dir: str) -> RasterPixelCount:
    stack_data = np.column_stack(([0, 1, 2], [3, 3, 3]))
    return fake_raster_pixel_count(tmp_dir, "pixel_id_0", stack_data)


@pytest.fixture
def fake_pixel_count1(tmp_dir: str) -> RasterPixelCount:
    stack_data = np.column_stack(([0, 1, 2], [3, 3, 3]))
    return fake_raster_pixel_count(tmp_dir, "pixel_id_1", stack_data)


@pytest.fixture
def fake_pixel_count2(tmp_dir: str) -> RasterPixelCount:
    stack_data = np.column_stack(([0, 1, 2], [0, 1, 8]))
    return fake_raster_pixel_count(tmp_dir, "pixel_id_2", stack_data)


def test_ordinal_trend_no_change(
    fake_pixel_count0: RasterPixelCount, fake_pixel_count1: RasterPixelCount
):
    op = OpTester(CONFIG_PATH)
    output = op.run(pixel_count=[fake_pixel_count0, fake_pixel_count1])
    assert output
    assert "ordinal_trend_result" in output

    ordinal_trend_result = output["ordinal_trend_result"]
    ordinal_trend_result = cast(OrdinalTrendTest, ordinal_trend_result)
    assert ordinal_trend_result.p_value == 1
    assert ordinal_trend_result.z_score == 0


def test_ordinal_trend_increase(
    fake_pixel_count0: RasterPixelCount, fake_pixel_count2: RasterPixelCount
):
    op = OpTester(CONFIG_PATH)
    output = op.run(pixel_count=[fake_pixel_count0, fake_pixel_count2])
    assert output
    assert "ordinal_trend_result" in output

    ordinal_trend_result = output["ordinal_trend_result"]
    ordinal_trend_result = cast(OrdinalTrendTest, ordinal_trend_result)
    assert ordinal_trend_result.p_value < SIGNIFICANCE_LEVEL
    assert ordinal_trend_result.z_score > 0


def test_ordinal_trend_decrease(
    fake_pixel_count2: RasterPixelCount, fake_pixel_count0: RasterPixelCount
):
    op = OpTester(CONFIG_PATH)
    output = op.run(pixel_count=[fake_pixel_count2, fake_pixel_count0])
    assert output
    assert "ordinal_trend_result" in output

    ordinal_trend_result = output["ordinal_trend_result"]
    ordinal_trend_result = cast(OrdinalTrendTest, ordinal_trend_result)
    assert ordinal_trend_result.p_value < SIGNIFICANCE_LEVEL
    assert ordinal_trend_result.z_score < 0
