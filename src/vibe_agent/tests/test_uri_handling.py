import os
from pathlib import Path

import pytest
from azure.storage.blob import ContainerClient

from vibe_agent.storage.file_upload import upload_to_blob
from vibe_core.uri import is_local, local_uri_to_path, uri_to_filename


@pytest.fixture
def filename(local_file: str):
    return os.path.basename(local_file)


@pytest.mark.parametrize("local_file_ref", ["path", "uri"], indirect=True)
def test_filename_from_local_file(filename: str, local_file_ref: str):
    assert is_local(local_file_ref)
    assert uri_to_filename(local_file_ref) == filename


@pytest.fixture(scope="module")
def remote_file(source_container: ContainerClient, local_file: str):
    filename = os.path.basename(local_file)
    blob = source_container.get_blob_client(filename)
    upload_to_blob(local_file, blob, overwrite=True)
    return blob


def test_local_uri_to_path():
    abs_path = "/abs/path/to/file"
    assert is_local(abs_path)
    assert local_uri_to_path(abs_path) == abs_path
    assert local_uri_to_path(Path(abs_path).as_uri()) == abs_path
    rel_path = "rel/path/to/file"
    assert is_local(rel_path)
    assert local_uri_to_path(rel_path) == rel_path
    abs_from_rel = local_uri_to_path(Path(rel_path).absolute().as_uri())
    assert abs_from_rel == os.path.abspath(rel_path)
    assert os.path.relpath(abs_from_rel, ".") == rel_path
