import os
from datetime import datetime, timezone
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from azure.cosmos.exceptions import CosmosHttpResponseError
from pystac import Asset, Item
from shapely import geometry as shpg
from shapely.geometry import Polygon, mapping

from vibe_agent.storage.remote_storage import CosmosStorage
from vibe_agent.storage.storage import AssetCopyHandler, ItemDict
from vibe_common.schemas import CacheInfo
from vibe_core.data import DataVibe
from vibe_core.data.utils import StacConverter
from vibe_dev.testing.storage_fixtures import *  # type: ignore # noqa: F403, F401


@pytest.fixture
def item_dict() -> ItemDict:
    num_items = 5
    polygon_coords = [
        (-88.062073563448919, 37.081397673802059),
        (-88.026349330507315, 37.085463858128762),
        (-88.026349330507315, 37.085463858128762),
        (-88.012445388773259, 37.069230099135126),
    ]
    polygon: Dict[str, Any] = mapping(Polygon(polygon_coords))  # type: ignore
    timestamp = datetime.now(timezone.utc)

    def create_item(i: int):
        id = str(i)
        new_item = Item(id=id, geometry=polygon, datetime=timestamp, properties={}, bbox=None)
        asset = Asset(href=os.path.join("/", "fake", id))
        new_item.add_asset(key=id, asset=asset)

        return new_item

    items = [create_item(i) for i in range(num_items)]

    single_item = create_item(num_items)

    output_dict: ItemDict = {"list_input": items, "single_input": single_item}

    return output_dict


@patch("vibe_agent.storage.asset_management.AssetManager")
def test_asset_handler_filename(mock_manager: MagicMock, item_dict: ItemDict):
    expected_href = "changed!"
    mock_manager.store.return_value = expected_href
    asset_handler = AssetCopyHandler(mock_manager)
    new_items = asset_handler.copy_assets(item_dict)

    for items in new_items.values():
        if isinstance(items, list):
            for i in items:
                for a in i.get_assets().values():
                    assert a.href == expected_href
        else:
            for a in items.get_assets().values():
                assert a.href == expected_href


@patch("vibe_agent.storage.CosmosStorage._store_data")
def test_cosmos_storage_split(mock_handle: MagicMock):
    fake_exception = CosmosHttpResponseError(status_code=413)
    mock_handle.side_effect = [fake_exception, fake_exception, None]
    items = {
        "test_data": [
            DataVibe(
                id=f"{i}",
                time_range=(datetime.utcnow(), datetime.utcnow()),
                geometry=shpg.mapping(shpg.box(0, 0, 1, 1)),
                assets=[],
            )
            for i in range(10)
        ]
    }
    converter = StacConverter()
    # `DataVibe` inherits from `BaseVibe` so the below should work fine, but
    # pyright/pylance don't like it.
    test_items: ItemDict = {k: converter.to_stac_item(v) for k, v in items.items()}  # type: ignore
    storage = CosmosStorage(
        key="",
        asset_manager=None,  # type: ignore
        stac_container_name="",
        cosmos_database_name="",
        cosmos_url="",
    )
    cache_info = CacheInfo("test_op", "1.0", {}, {})
    storage.store("test_run", test_items, cache_info)
    assert mock_handle.call_count == 3
    assert len(mock_handle.call_args_list[0].args[2][0]["items"]) == 10
    assert len(mock_handle.call_args_list[1].args[2][0]["items"]) == 5
    assert len(mock_handle.call_args_list[2].args[2][0]["items"]) == 3
