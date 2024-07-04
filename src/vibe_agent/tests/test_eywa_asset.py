import mimetypes
from pathlib import Path

import pytest

from vibe_agent.storage.asset_management import AssetManager
from vibe_core.data import AssetVibe

CONTENT = "FAKE CONTENT FILE"
EXTENSION = ".txt"
ID = "FAKE_FILE"
FNAME = f"{ID}{EXTENSION}"


@pytest.fixture
def local_file(tmp_path: Path) -> str:
    with open(tmp_path / FNAME, "w") as f:
        f.write(CONTENT)

    assert Path.exists(tmp_path / FNAME)
    return (tmp_path / FNAME).as_posix()


@pytest.fixture
def remote_file(local_file: str, blob_asset_manager: AssetManager) -> str:
    blob_asset_manager.store(ID, local_file)
    assert blob_asset_manager.exists(ID)
    return blob_asset_manager.retrieve(ID)


def test_local_asset(local_file: str):
    local_asset = AssetVibe(reference=local_file, type=mimetypes.types_map[EXTENSION], id=ID)

    # file is local, then local path must be equal to passed reference
    assert local_asset.local_path == local_file

    # Local urls are assigned with file:// prefix
    assert local_asset.url == f"file://{local_file}"
