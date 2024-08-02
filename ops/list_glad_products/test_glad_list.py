# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import itertools
import os
from datetime import datetime
from typing import List, Tuple, cast
from unittest.mock import Mock, patch

import pytest

from vibe_core import file_downloader
from vibe_core.data import DataVibe
from vibe_core.data.products import GLADProduct
from vibe_core.utils import ensure_list
from vibe_dev.testing.op_tester import OpTester

VALID_GLAD_YEARS = [2000, 2020]
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "list_glad_products.yaml")
TILES_MAPPING = {
    "northwest": ["50N_110W", "50N_120W", "60N_110W", "60N_120W"],
    "northeast": ["50N_060E", "50N_070E", "60N_060E", "60N_070E"],
    "southwest": ["10S_060W", "10S_070W", "20S_060W", "20S_070W"],
    "southeast": ["10S_010E", "00N_010E", "10S_020E", "00N_020E"],
}

MOCK_TILES = {
    "50N_110W",
    "50N_120W",
    "60N_110W",
    "60N_120W",
    "50N_060E",
    "50N_070E",
    "60N_060E",
    "60N_070E",
    "10S_060W",
    "10S_070W",
    "20S_060W",
    "20S_070W",
    "10S_010E",
    "00N_010E",
    "10S_020E",
    "00N_020E",
}


def custom_datavibe(
    coordinates: List[List[float]],
    time_range: Tuple[datetime, datetime] = (datetime(2000, 1, 1), datetime(2023, 1, 1)),
) -> DataVibe:
    return DataVibe(
        id=str("test_id"),
        time_range=time_range,
        geometry={
            "type": "Polygon",
            "coordinates": [coordinates],
        },
        assets=[],
    )


TEST_DATAVIBES = {
    "northwest": custom_datavibe(
        [
            [-115.0, 55.0],
            [-105.0, 55.0],
            [-105.0, 45.0],
            [-115.0, 45.0],
        ]
    ),
    "northeast": custom_datavibe(
        [
            [75.0, 55.0],
            [65.0, 55.0],
            [65.0, 45.0],
            [75.0, 45.0],
        ]
    ),
    "southwest": custom_datavibe(
        [
            [-65.0, -15.0],
            [-55.0, -15.0],
            [-55.0, -25.0],
            [-65.0, -25.0],
        ]
    ),
    "southeast": custom_datavibe(
        [
            [15.0, -5.0],
            [25.0, -5.0],
            [25.0, -15.0],
            [15.0, -15.0],
        ]
    ),
}


def mock_verify(url: str):
    # URLs are of the form:
    # https://glad.umd.edu/users/Potapov/GLCLUC2020/Forest_extent_2000/00N_000E.tif
    return url[-12:-4] in MOCK_TILES and int(url[-17:-13]) in VALID_GLAD_YEARS


@patch.object(file_downloader, "verify_url")
@pytest.mark.parametrize(
    "test_datavibe, expected_tiles",
    [
        (TEST_DATAVIBES["northwest"], TILES_MAPPING["northwest"]),
        (TEST_DATAVIBES["northeast"], TILES_MAPPING["northeast"]),
        (TEST_DATAVIBES["southwest"], TILES_MAPPING["southwest"]),
        (TEST_DATAVIBES["southeast"], TILES_MAPPING["southeast"]),
    ],
)
def test_glad_list(verify: Mock, test_datavibe: DataVibe, expected_tiles: List[str]):
    verify.side_effect = mock_verify
    op = OpTester(CONFIG_PATH)
    output_data = op.run(**{"input_item": test_datavibe})
    assert output_data
    assert "glad_products" in output_data

    products: List[GLADProduct] = cast(List[GLADProduct], ensure_list(output_data["glad_products"]))
    expected_combinations = set(itertools.product(expected_tiles, VALID_GLAD_YEARS))

    actual_combinations = set((p.tile_name, p.time_range[0].year) for p in products)

    assert expected_combinations == actual_combinations
    verify.reset_mock()


@patch.object(file_downloader, "verify_url")
def test_glad_list_same_tiles(verify: Mock):
    verify.side_effect = mock_verify

    # Create datavibe_1
    test_data_vibe_1 = custom_datavibe(
        [
            [15.0, -5.0],
            [15.1, -5.0],
            [15.1, -5.1 + 0.1],  # not the same geom
            [15.0, -5.1],
        ],
        time_range=(datetime(2020, 1, 1), datetime(2020, 1, 1)),
    )

    test_data_vibe_2 = custom_datavibe(
        [
            [15.0, -5.0],
            [15.1, -5.0],
            [15.1, -5.1],
            [15.0, -5.1],
        ],
        time_range=(datetime(2020, 1, 1), datetime(2020, 1, 1)),
    )

    op = OpTester(CONFIG_PATH)
    output_1 = op.run(**{"input_item": test_data_vibe_1})
    output_2 = op.run(**{"input_item": test_data_vibe_2})

    products: List[GLADProduct] = []
    for output in [output_1, output_2]:
        assert output
        assert "glad_products" in output
        assert isinstance(output["glad_products"], list)
        assert len(output["glad_products"]) > 0

        products.append(cast(GLADProduct, output["glad_products"][0]))

    assert products[0].id == products[1].id
    assert products[0].time_range == products[1].time_range
    assert products[0].geometry == products[1].geometry
    assert products[0].assets == products[1].assets
    assert products[0].url == products[1].url
