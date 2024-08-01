# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from datetime import datetime
from typing import cast
from unittest.mock import Mock, patch

import pytest
from shapely import geometry as shpg

from vibe_core import file_downloader
from vibe_core.data import CategoricalRaster, GLADProduct
from vibe_dev.testing.op_tester import OpTester


@pytest.fixture
def glad_product():
    return GLADProduct(
        id="test_id",
        geometry=shpg.mapping(shpg.box(-115.0, 45.0, -105.0, 55.0)),
        time_range=(datetime(2020, 1, 1), datetime(2020, 1, 2)),
        url="https://test.com/test.tif",
        assets=[],
    )


@patch.object(file_downloader, "download_file")
def test_download_glad_product(download: Mock, glad_product: GLADProduct):
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_glad.yaml")

    op_tester = OpTester(config_path)
    out = op_tester.run(glad_product=glad_product)
    assert out
    assert "downloaded_product" in out
    downloaded_product: CategoricalRaster = cast(CategoricalRaster, out["downloaded_product"])
    assert len(downloaded_product.assets) > 0
    asset = downloaded_product.assets[0]
    assert asset.path_or_url.endswith(
        f"{glad_product.tile_name}_{glad_product.time_range[0].year}.tif"
    )
