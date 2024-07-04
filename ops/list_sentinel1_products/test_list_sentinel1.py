import json
import os
from datetime import datetime, timezone
from typing import List
from unittest.mock import Mock, patch

import pytest
from pystac import Item
from shapely import geometry as shpg

from vibe_core.data import DataVibe, Sentinel1Product
from vibe_dev.testing.op_tester import OpTester
from vibe_lib.planetary_computer import Sentinel1GRDCollection, Sentinel1RTCCollection

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH_PC = os.path.join(HERE, "list_sentinel1_products_pc.yaml")


@pytest.fixture
def fake_items_pc():
    filepath = os.path.join(HERE, "sample_pc_output.json")
    with open(filepath) as f:
        out = json.load(f)
    return [Item.from_dict(i) for i in out]


@pytest.fixture
def input_data():
    polygon_coords = [
        (-118.8415739999999943, 46.7963099999999983),
        (-118.6759440000000012, 46.7963099999999983),
        (-118.6759440000000012, 46.9169079999999994),
        (-118.8415739999999943, 46.9169079999999994),
        (-118.8415739999999943, 46.7963099999999983),
    ]

    geom = shpg.Polygon(polygon_coords)
    start_date = datetime(year=2021, month=7, day=10, tzinfo=timezone.utc)
    end_date = datetime(year=2021, month=7, day=28, tzinfo=timezone.utc)
    return DataVibe("input_test_data", (start_date, end_date), shpg.mapping(geom), [])


def compare_product_with_stac(product: Sentinel1Product, stac_item: Item):
    assert product.geometry == stac_item.geometry
    assert product.id == stac_item.id
    assert product.time_range[0] == stac_item.datetime


@patch("vibe_lib.planetary_computer.get_available_collections")
@patch.object(Sentinel1GRDCollection, "query")
def test_list_pc(
    query: Mock, get_collections: Mock, fake_items_pc: List[Item], input_data: DataVibe
):
    query.return_value = fake_items_pc
    get_collections.return_value = [Sentinel1GRDCollection.collection]

    op_tester = OpTester(CONFIG_PATH_PC)
    op_tester.update_parameters({"collection": "grd"})
    output_data = op_tester.run(input_item=input_data)

    # Get op result
    output_name = "sentinel_products"
    assert output_name in output_data
    products = output_data[output_name]
    assert isinstance(products, list)
    assert len(products) == 3
    get_collections.assert_called_once()
    query.assert_called_once_with(
        geometry=shpg.shape(input_data.geometry), time_range=input_data.time_range
    )
    for p, i in zip(products, fake_items_pc):
        assert isinstance(p, Sentinel1Product)
        compare_product_with_stac(p, i)


@patch("vibe_lib.planetary_computer.get_available_collections")
@patch.object(Sentinel1RTCCollection, "query")
def test_list_rtc(
    query: Mock, get_collections: Mock, fake_items_pc: List[Item], input_data: DataVibe
):
    query.return_value = fake_items_pc
    get_collections.return_value = [Sentinel1RTCCollection.collection]

    op_tester = OpTester(CONFIG_PATH_PC)
    output_data = op_tester.run(input_item=input_data)

    # Get op result
    output_name = "sentinel_products"
    assert output_name in output_data
    products = output_data[output_name]
    assert isinstance(products, list)
    assert len(products) == 3
    get_collections.assert_called_once()
    query.assert_called_once_with(
        geometry=shpg.shape(input_data.geometry), time_range=input_data.time_range
    )
    for p, i in zip(products, fake_items_pc):
        assert isinstance(p, Sentinel1Product)
        compare_product_with_stac(p, i)
