# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from pystac import Item
from shapely import geometry as shpg

from vibe_core.data import DataVibe
from vibe_dev.testing.op_tester import OpTester
from vibe_lib.planetary_computer import Modis16DayVICollection

HERE = os.path.dirname(os.path.abspath(__file__))

FAKE_TIME_RANGE = (datetime(2020, 11, 1), datetime(2020, 11, 2))
FAKE_GEOM = shpg.mapping(shpg.box(0, 0, 2, 2))
INVALID_RESOLUTION = 100


def fake_items(resolution: int):
    return [
        Item(
            id=f"{resolution}m-id",  # type: ignore
            geometry=FAKE_GEOM,
            bbox=None,
            datetime=None,
            properties={
                "start_datetime": FAKE_TIME_RANGE[0].isoformat() + "Z",
                "end_datetime": FAKE_TIME_RANGE[1].isoformat() + "Z",
            },
        )
    ]


@pytest.mark.parametrize("resolution", (250, 500))
@patch("vibe_lib.planetary_computer.get_available_collections")
@patch.object(Modis16DayVICollection, "query")
def test_op(query: MagicMock, get_collections: MagicMock, resolution: int):
    query.return_value = fake_items(resolution)
    get_collections.return_value = list(Modis16DayVICollection.collections.values())

    geom1 = shpg.Point(1, 1).buffer(0.1, cap_style=3)
    geom2 = shpg.Point(2, 2).buffer(0.1, cap_style=3)
    time_range = (datetime(2022, 11, 1), datetime(2022, 11, 16))
    x1 = DataVibe(id="1", time_range=time_range, geometry=shpg.mapping(geom1), assets=[])
    x2 = DataVibe(id="2", time_range=time_range, geometry=shpg.mapping(geom2), assets=[])
    op_tester = OpTester(os.path.join(HERE, "list_modis_vegetation.yaml"))
    op_tester.update_parameters({"resolution": resolution})
    o1 = op_tester.run(input_data=[x1])
    query.assert_called_with(geometry=geom1, time_range=x1.time_range)
    get_collections.assert_called_once()
    o2 = op_tester.run(input_data=[x2])
    query.assert_called_with(geometry=geom2, time_range=x2.time_range)
    assert get_collections.call_count == 2
    o3 = op_tester.run(input_data=[x1, x2])
    assert get_collections.call_count == 3
    assert query.call_count == 4
    products = o1["modis_products"]
    assert isinstance(products, list)
    assert len(products) == 1
    product = products[0]
    assert isinstance(product, DataVibe)
    assert product.id == f"{resolution}m-id"
    assert product.time_range == tuple(t.astimezone() for t in FAKE_TIME_RANGE)
    assert product.geometry == FAKE_GEOM
    assert o1 == o2 == o3


def test_op_fails_invalid_res():
    op_tester = OpTester(os.path.join(HERE, "list_modis_vegetation.yaml"))
    op_tester.update_parameters({"resolution": INVALID_RESOLUTION})
    with pytest.raises(ValueError):
        op_tester.run(input_data=[])
