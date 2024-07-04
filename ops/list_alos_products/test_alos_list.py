import os
from datetime import datetime
from typing import Any, Dict, Tuple
from unittest.mock import Mock, patch

import pytest
from pystac import Item

from vibe_core.data import AlosProduct, DataVibe
from vibe_dev.testing.op_tester import OpTester


@pytest.fixture
def geometry():
    return {
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
    }


@pytest.fixture
def time_range():
    return (datetime(2019, 1, 1), datetime(2020, 1, 1))


@pytest.fixture
def data_vibe(geometry: Dict[str, Any], time_range: Tuple[datetime, datetime]):
    return DataVibe(
        id=str("test_id"),
        time_range=time_range,
        geometry=geometry,
        assets=[],
    )


def expected_items(geometry: Dict[str, Any], time_range: Tuple[datetime, datetime]):
    bbox = [-87.0, 14.0, -86.0, 15.0]
    first_item = Item(
        id="N15W087_20_FNF",
        geometry=geometry,
        bbox=bbox,
        datetime=time_range[0],
        properties={
            "start_datetime": time_range[0].strftime("%Y-%m-%d"),
            "end_datetime": time_range[0].strftime("%Y-%m-%d"),
        },
    )
    second_item = Item(
        id="N15W087_19_FNF",
        geometry=geometry,
        bbox=bbox,
        datetime=time_range[1],
        properties={
            "start_datetime": time_range[1].strftime("%Y-%m-%d"),
            "end_datetime": time_range[1].strftime("%Y-%m-%d"),
        },
    )
    return [first_item, second_item]


@patch("vibe_lib.planetary_computer.AlosForestCollection.query")
def test_alos_list(query: Mock, data_vibe: DataVibe):
    mock_items = expected_items(geometry=data_vibe.geometry, time_range=data_vibe.time_range)
    query.return_value = mock_items
    config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "list_alos_products.yaml"
    )

    op = OpTester(config_path)
    output_data = op.run(**{"input_data": data_vibe})
    assert output_data

    assert "alos_products" in output_data
    products = output_data["alos_products"]

    # Check variable products is a list of AlosProduct
    assert isinstance(products, list)
    assert len(products) == len(mock_items)
    for item, product in zip(mock_items, products):
        assert isinstance(product, AlosProduct)
        assert product.id == item.id
        assert product.geometry == item.geometry
        assert product.time_range == (item.datetime, item.datetime)
        assert product.assets == []
