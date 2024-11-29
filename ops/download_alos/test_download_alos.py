# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from datetime import datetime, timezone
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from pystac import Asset, Item

from vibe_core.data import AlosProduct, Raster
from vibe_dev.testing.op_tester import OpTester
from vibe_lib.planetary_computer import AlosForestCollection

FAKE_TIME_RANGE = (
    datetime(2020, 11, 1, tzinfo=timezone.utc),
    datetime(2020, 11, 2, tzinfo=timezone.utc),
)


@pytest.fixture
def alos_product():
    return AlosProduct(
        id="N15W087_20_FNF",
        geometry={
            "type": "Polygon",
            "coordinates": [
                [
                    [-86.773827, 14.575498],
                    [-86.770459, 14.579301],
                    [-86.764283, 14.575102],
                    [-86.769591, 14.567595],
                    [-86.773827, 14.575498],
                ]
            ],
        },
        time_range=FAKE_TIME_RANGE,
        assets=[],
    )


def fake_items():
    assets = {"N15W087_20_FNF": Asset(href="fake_href", media_type="image/tiff")}
    return Item(
        id="N15W087_20_FNF",
        geometry=None,
        bbox=None,
        datetime=None,
        properties={
            "start_datetime": FAKE_TIME_RANGE[0].isoformat() + "Z",
            "end_datetime": FAKE_TIME_RANGE[1].isoformat() + "Z",
        },
        assets=assets,
    )


@patch.object(AlosForestCollection, "download_item")
@patch.object(AlosForestCollection, "query_by_id")
@patch("vibe_lib.planetary_computer.get_available_collections")
def test_alos_download(
    get_collections: MagicMock,
    query_by_id: MagicMock,
    download_item: MagicMock,
    alos_product: AlosProduct,
):
    get_collections.return_value = [AlosForestCollection.collection]
    query_by_id.return_value = fake_items()
    download_item.side_effect = lambda item, _: [item.assets[item.id].href]

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "download_alos.yaml")

    op = OpTester(config_path)
    output_data = op.run(product=alos_product)
    assert output_data
    assert "raster" in output_data

    output_raster = cast(Raster, output_data["raster"])
    assert len(output_raster.assets) == 1
    assert output_raster.assets[0].type == "image/tiff"
    assert output_raster.assets[0].path_or_url == "fake_href"
    assert output_raster.bands == {"forest_non_forest": 0}
    assert output_raster.time_range == FAKE_TIME_RANGE
    assert output_raster.geometry == alos_product.geometry
