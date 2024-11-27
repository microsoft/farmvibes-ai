# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from datetime import datetime
from typing import List, cast
from unittest.mock import Mock, patch

import pytest
from shapely import geometry as shpg

from vibe_core import file_downloader
from vibe_core.data import HansenProduct, Raster
from vibe_dev.testing.op_tester import OpTester


@pytest.fixture
def hansen_products():
    return [
        HansenProduct(
            id="test_id",
            geometry=shpg.mapping(shpg.box(-115.0, 45.0, -105.0, 55.0)),
            time_range=(datetime(2000, 1, 1), datetime(2022, 1, 2)),
            asset_url=(
                f"https://storage.googleapis.com/earthenginepartners-hansen/"
                f"GFC-2022-v1.10/Hansen_GFC-2022-v1.10_{asset_key}_00N_000E.tif"
            ),
            assets=[],
        )
        for asset_key in HansenProduct.asset_keys
    ]


@patch.object(file_downloader, "download_file")
def test_download_hansen_product(download: Mock, hansen_products: List[HansenProduct]):
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_hansen.yaml")

    op_tester = OpTester(config_path)

    for hansen_product in hansen_products:
        out = op_tester.run(hansen_product=hansen_product)
        assert out

        raster = cast(Raster, out["raster"])

        assert raster
        assert len(raster.assets) == 1
        assert raster.bands == {hansen_product.layer_name: 0}

        assert raster.time_range == hansen_product.time_range
        assert raster.geometry == hansen_product.geometry
