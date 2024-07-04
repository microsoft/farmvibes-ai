import os
from datetime import datetime, timezone
from typing import List, cast
from unittest.mock import MagicMock, patch

import pytest
from pystac import Asset, Item
from shapely.geometry import Point, mapping

from vibe_core.data import DataVibe, GNATSGOProduct
from vibe_dev.testing.op_tester import OpTester
from vibe_lib.planetary_computer import GNATSGOCollection

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "list_gnatsgo_products.yaml")

VALID_GEOMETRY = Point(-92.99900, 42.03580).buffer(0.1, cap_style=3)
INVALID_GEOMETRY = Point(-47.06966, -22.81709).buffer(0.1, cap_style=3)
FAKE_DATE = datetime(year=2020, month=7, day=1, tzinfo=timezone.utc)


def fake_items():
    assets = {f"{var}": Asset(href=f"fake_href_{var}") for var in GNATSGOCollection.asset_keys}
    return [
        Item(
            id="fake_id",  # type: ignore
            geometry=mapping(VALID_GEOMETRY),
            bbox=VALID_GEOMETRY.bounds,  # type: ignore
            datetime=FAKE_DATE,
            properties={},
            assets=assets,
        )
    ]


@patch("vibe_lib.planetary_computer.get_available_collections", return_value=["gnatsgo-rasters"])
@patch.object(GNATSGOCollection, "query")
def test_op(query: MagicMock, _: MagicMock):
    query.return_value = fake_items()

    input_item = DataVibe("input_item", (FAKE_DATE, FAKE_DATE), VALID_GEOMETRY, [])  # type: ignore

    op_tester = OpTester(CONFIG_PATH)
    out = op_tester.run(input_item=input_item)

    assert query.call_args.kwargs["roi"] == VALID_GEOMETRY.bounds

    assert "gnatsgo_products" in out
    products = cast(List[GNATSGOProduct], out["gnatsgo_products"])
    assert isinstance(products, list)
    assert len(products) == 1


@patch("vibe_lib.planetary_computer.get_available_collections", return_value=["gnatsgo-rasters"])
@patch.object(GNATSGOCollection, "query")
def test_op_fails_invalid_geometry(query: MagicMock, _: MagicMock):
    query.return_value = []
    input_item = DataVibe("input_item", (FAKE_DATE, FAKE_DATE), mapping(INVALID_GEOMETRY), [])

    op_tester = OpTester(CONFIG_PATH)
    with pytest.raises(RuntimeError):
        op_tester.run(input_item=input_item)
