# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
from datetime import datetime
from typing import Any, Dict, List, cast
from unittest.mock import Mock, patch

import pytest
from dateutil.parser import parse as parse_date
from shapely import geometry as shpg

from vibe_core.data import DataVibe, GEDIProduct
from vibe_dev.testing.op_tester import OpTester
from vibe_lib.earthdata import EarthDataAPI

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "list_gedi_products.yaml")


@pytest.fixture
def mock_items():
    with open(os.path.join(HERE, "mock_items.json")) as f:
        return json.load(f)


def compare_product_with_json(product: GEDIProduct, item: Dict[str, Any]):
    assert product.product_name == item["producer_granule_id"]
    assert isinstance(shpg.shape(product.geometry), shpg.Polygon)
    assert product.time_range[0] == parse_date(item["time_start"])
    assert product.start_orbit == int(
        item["orbit_calculated_spatial_domains"][0]["start_orbit_number"]
    )


@patch.object(EarthDataAPI, "query")
def test_op(query: Mock, mock_items: List[Dict[str, Any]]):
    query.return_value = mock_items
    now = datetime.now()
    geom = shpg.box(0, 0, 1, 1)
    x = DataVibe(id="1", time_range=(now, now), geometry=shpg.mapping(geom), assets=[])
    out = OpTester(CONFIG_PATH).run(input_data=x)
    assert "gedi_products" in out
    products = cast(List[GEDIProduct], out["gedi_products"])
    assert len(products) == 5
    for p, i in zip(products, mock_items):
        compare_product_with_json(p, i)
