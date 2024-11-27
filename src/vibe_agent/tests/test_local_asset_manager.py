# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, Mock, patch

import pytest
import requests

from vibe_agent.storage.asset_management import LocalFileAssetManager


@pytest.fixture
def manager(tmpdir: str):
    return LocalFileAssetManager(tmpdir)


@patch("os.makedirs")
@patch("shutil.copyfile")
def test_store_add_file(shutil_mock: Mock, makedir_mock: Mock, manager: LocalFileAssetManager):
    guid = "123456"
    file_path = os.path.join("fake", "file", "path")
    manager.exists = MagicMock(return_value=False)

    actual_return = manager.store(guid, file_path)

    target_folder = os.path.join(manager.root_path, guid)
    target_file = os.path.join(target_folder, os.path.basename(file_path))
    makedir_mock.assert_called_once_with(target_folder)
    shutil_mock.assert_called_once_with(file_path, target_file)
    assert actual_return == target_file


@patch("os.makedirs")
@patch("shutil.copyfile")
def test_store_exists(shutil_mock: Mock, makedir_mock: Mock, manager: LocalFileAssetManager):
    guid = "123456"
    file_path = os.path.join("fake", "file", "path")
    manager.exists = MagicMock(return_value=True)
    return_value = "fake_return_path"
    manager.retrieve = MagicMock(return_value=return_value)

    actual_return = manager.store(guid, file_path)

    makedir_mock.assert_not_called()
    shutil_mock.assert_not_called()
    assert actual_return == return_value


def test_remove(manager: LocalFileAssetManager):
    guid = "123456"
    manager.exists = MagicMock(return_value=True)

    with patch("shutil.rmtree") as shutil_mock:
        manager.remove(guid)
        shutil_mock.assert_called_once_with(os.path.join(manager.root_path, guid))


@patch("shutil.rmtree")
def test_remove_not_exists(shutil_mock: Mock, manager: LocalFileAssetManager):
    guid = "123456"
    manager.exists = MagicMock(return_value=False)

    manager.remove(guid)

    shutil_mock.assert_not_called()


@patch("os.path.exists")
@patch("os.listdir")
def test_retrieve(listdir_mock: Mock, exists_mock: Mock):
    with TemporaryDirectory() as tmp_dir:
        guid = "123456"
        file_name = os.path.join("fake_file")
        manager = LocalFileAssetManager(tmp_dir)
        manager.exists = MagicMock(return_value=False)
        listdir_mock.return_value = [file_name]
        exists_mock.return_value = True

        ret = manager.retrieve(guid)

        listdir_mock.assert_called_once_with(os.path.join(tmp_dir, guid))
        assert ret == os.path.join(tmp_dir, guid, file_name)


@patch("os.path.exists")
def test_exists(exists_mock: Mock):
    with TemporaryDirectory() as tmp_dir:
        guid = "123456"
        manager = LocalFileAssetManager(tmp_dir)
        manager.exists(guid)
        exists_mock.assert_called_once_with(os.path.join(tmp_dir, guid))


@pytest.mark.parametrize("local_file_ref", ["path", "uri"], indirect=True)
def test_store_local(manager: LocalFileAssetManager, local_file_ref: str):
    asset_guid = "123456"
    assert not manager.exists(asset_guid)
    manager.store(asset_guid, local_file_ref)
    assert manager.exists(asset_guid)
    assert os.path.exists(manager.retrieve(asset_guid))


@pytest.mark.parametrize("non_existing_file", ["local"], indirect=True)
def test_asset_does_not_exist_on_fail(manager: LocalFileAssetManager, non_existing_file: str):
    asset_guid = "123456"
    assert not manager.exists(asset_guid)
    with pytest.raises((FileNotFoundError, requests.exceptions.HTTPError)):
        manager.store(asset_guid, non_existing_file)
    assert not manager.exists(asset_guid)
    with pytest.raises(ValueError):
        manager.retrieve(asset_guid)
