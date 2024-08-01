# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import itertools
import os
from datetime import datetime
from typing import List
from unittest.mock import Mock, patch

import pytest

from vibe_core import file_downloader
from vibe_core.data import DataVibe
from vibe_core.data.products import HansenProduct
from vibe_dev.testing.op_tester import OpTester

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "list_hansen_products.yaml")
DEFAULT_DATASET_FINAL_YEAR = 2022
DEFAULT_DATASET_FOLDER = "https://storage.googleapis.com/earthenginepartners-hansen/"
DEFAULT_DATASET_VERSION = "v1.10"

EXPECTED_TILES = {
    "northwest": ["50N_110W", "50N_120W", "60N_110W", "60N_120W"],
    "northeast": ["50N_060E", "50N_070E", "60N_060E", "60N_070E"],
    "southwest": ["10S_060W", "10S_070W", "20S_060W", "20S_070W"],
    "southeast": ["10S_010E", "00N_010E", "10S_020E", "00N_020E"],
}

MOCK_TILES = set([tile_name for tile_list in EXPECTED_TILES.values() for tile_name in tile_list])


def create_fake_datavibe(coordinates: List[List[float]]) -> DataVibe:
    return DataVibe(
        id=str("test_id"),
        time_range=(datetime(2000, 1, 1), datetime(2022, 1, 1)),
        geometry={
            "type": "Polygon",
            "coordinates": [coordinates],
        },
        assets=[],
    )


MOCK_INPUT_DICT = {
    "northwest": create_fake_datavibe(
        [
            [-115.0, 55.0],
            [-105.0, 55.0],
            [-105.0, 45.0],
            [-115.0, 45.0],
        ]
    ),
    "northeast": create_fake_datavibe(
        [
            [75.0, 55.0],
            [65.0, 55.0],
            [65.0, 45.0],
            [75.0, 45.0],
        ]
    ),
    "southwest": create_fake_datavibe(
        [
            [-65.0, -15.0],
            [-55.0, -15.0],
            [-55.0, -25.0],
            [-65.0, -25.0],
        ]
    ),
    "southeast": create_fake_datavibe(
        [
            [15.0, -5.0],
            [25.0, -5.0],
            [25.0, -15.0],
            [15.0, -15.0],
        ]
    ),
}


@patch.object(file_downloader, "verify_url")
@pytest.mark.parametrize(
    "test_datavibe, expected_tiles, layer_name",
    [
        (MOCK_INPUT_DICT[location], EXPECTED_TILES[location], asset_key)
        for location, asset_key in itertools.product(
            ["northwest", "northeast", "southwest", "southeast"], HansenProduct.asset_keys
        )
    ],
)
def test_hansen_list(
    verify: Mock, test_datavibe: DataVibe, expected_tiles: List[str], layer_name: str
):
    # URLs are of the form:
    # https://storage.googleapis.com/earthenginepartners-hansen/GFC-2022-v1.10/Hansen_GFC-2022-v1.10_treecover2000_20N_090W.tif
    def mock_verify(url: str):
        return (
            url[-12:-4] in MOCK_TILES
            and int(url.split("/")[-2].split("-")[1]) == DEFAULT_DATASET_FINAL_YEAR
        )

    verify.side_effect = mock_verify
    op = OpTester(CONFIG_PATH)
    op.update_parameters({"layer_name": layer_name})

    output_data = op.run(input_item=test_datavibe)
    assert output_data
    assert "hansen_products" in output_data

    tiles = set([product.tile_name for product in output_data["hansen_products"]])  # type: ignore
    assert all(
        [
            product.layer_name == layer_name
            for product in output_data["hansen_products"]  # type: ignore
        ]
    )
    assert tiles == set(expected_tiles), f"Expected {expected_tiles}, got {tiles}"
    assert all(
        [
            product.last_year == DEFAULT_DATASET_FINAL_YEAR
            for product in output_data["hansen_products"]  # type: ignore
        ]
    )
    assert all(
        [
            product.version == DEFAULT_DATASET_VERSION
            for product in output_data["hansen_products"]  # type: ignore
        ]
    )

    for product in output_data["hansen_products"]:  # type: ignore
        expected_url = (
            f"{DEFAULT_DATASET_FOLDER}Hansen_GFC-2022-v1.10_{layer_name}_{product.tile_name}.tif"
        )
        assert set(product.asset_url) == set(expected_url)


def test_hansen_invalid_years():
    op = OpTester(CONFIG_PATH)
    test_datavibe = MOCK_INPUT_DICT["northwest"]
    test_datavibe.time_range = (datetime(1999, 1, 1), datetime(2022, 1, 1))
    with pytest.raises(ValueError):
        op.run(input_item=test_datavibe)

    test_datavibe.time_range = (datetime(2000, 1, 1), datetime(2023, 1, 1))
    with pytest.raises(ValueError):
        op.run(input_item=test_datavibe)
