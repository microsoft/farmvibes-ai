# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Any

from azure.storage.blob import BlobClient

from vibe_core.uri import is_local, local_uri_to_path


def upload_to_blob(file_path: str, blob_client: BlobClient, *args: Any, **kwargs: Any):
    if is_local(file_path):
        local_upload(file_path, blob_client, *args, **kwargs)
    else:
        remote_upload(file_path, blob_client, *args, **kwargs)


def local_upload(file_path: str, blob_client: BlobClient, *args: Any, **kwargs: Any):
    # At this point, we expect a valid local path was passed to the file_path
    # which can be something like "file:///path/to/file" or "/path/to/file".
    file_path = local_uri_to_path(file_path)
    with open(file_path, "rb") as data:
        blob_client.upload_blob(data=data, *args, **kwargs)


def remote_upload(file_path: str, blob_client: BlobClient, *args: Any, **kwargs: Any):
    blob_client.upload_blob_from_url(file_path, *args, **kwargs)
