import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, cast

import pytest
from pystac import Asset, Item

from vibe_core.data.utils import serialize_stac
from vibe_core.datamodel import RunConfigInput, RunConfigUser
from vibe_server.href_handler import LocalHrefHandler
from vibe_server.server import TerravibesProvider


@pytest.fixture
def fake_op_name() -> str:
    return "fake.fake"


@pytest.fixture
def fake_asset_name() -> str:
    return "fake_asset"


@pytest.fixture
def one_item_one_asset(fake_asset_name: str) -> Item:
    asset = Asset(href="../../../assets/asdf/test.txt")
    item = Item(
        id="fake_id",
        geometry={},
        bbox=[],
        datetime=datetime.datetime.utcnow(),
        properties={},
    )
    item.add_asset(key=fake_asset_name, asset=asset)
    return item


def test_local_href_handler_parse_item(one_item_one_asset: Item, tmp_path: Path):
    local_href_handler = LocalHrefHandler(tmp_path)
    new_item = local_href_handler._parse_item(one_item_one_asset)
    for _, v in new_item.get_assets().items():
        p = Path(v.href)
        assert p.absolute


def test_local_href_handler_update_asset(tmp_path: Path):
    local_href_handler = LocalHrefHandler(tmp_path)

    asset = Asset(href="../../../assets/asdf/test.txt")
    local_href_handler._update_asset(asset)
    p = tmp_path / "asdf" / "test.txt"
    assert asset.href == str(p)
    assert os.path.isabs(asset.href)

    asset = Asset(href=".././/../assets/asdf/test.txt")
    local_href_handler._update_asset(asset)
    p = tmp_path / "asdf" / "test.txt"
    assert asset.href == str(p)

    asset = Asset(href="../../assets/asdf/blah/../test.txt")
    local_href_handler._update_asset(asset)
    p = tmp_path / "asdf" / "test.txt"
    assert asset.href == str(p)
    assert ".." not in asset.href

    asset = Asset(href="/test.txt")
    local_href_handler._update_asset(asset)
    p = tmp_path / "test.txt"
    assert asset.href == str(p)


@pytest.fixture
def run_config_with_output(
    one_item_one_asset: Item, fake_op_name: str, workflow_run_config: Dict[str, Any]
) -> RunConfigUser:
    provider = TerravibesProvider(LocalHrefHandler("/tmp"))
    _, run_config = provider.create_new_run(RunConfigInput(**workflow_run_config), [])
    run_config.set_output({fake_op_name: [serialize_stac(one_item_one_asset)]})
    return RunConfigUser.from_runconfig(run_config)


def test_href_handler_handle(
    run_config_with_output: RunConfigUser, fake_op_name: str, fake_asset_name: str, tmp_path: Path
):
    local_href_handler = LocalHrefHandler(tmp_path)

    original_item = cast(List[Dict[str, Any]], run_config_with_output.output[fake_op_name])[0]
    original_href = original_item["assets"][fake_asset_name]["href"]
    original_path = str(
        local_href_handler.assets_dir / Path(original_href).parent.name / Path(original_href).name
    )

    local_href_handler.handle(run_config_with_output)

    parsed_item = cast(List[Dict[str, Any]], run_config_with_output.output[fake_op_name])[0]
    parsed_path = parsed_item["assets"][fake_asset_name]["href"]

    assert parsed_path == original_path
