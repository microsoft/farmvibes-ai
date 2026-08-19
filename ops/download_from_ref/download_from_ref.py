# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import hashlib
import ipaddress
import mimetypes
import os
import pathlib
import shutil
import socket
from dataclasses import fields
from tempfile import TemporaryDirectory
from typing import Any, Dict, Type, cast, get_origin
from urllib.parse import urljoin, urlparse

from vibe_core.data import (
    AssetVibe,
    DataVibe,
    ExternalReference,
    data_registry,
    gen_hash_id,
)
from vibe_core.file_downloader import download_file
from vibe_core.uri import is_local, local_uri_to_path, uri_to_filename

CHUNK_SIZE_BYTES = 1024 * 1024

ALLOWED_SCHEMES = ("http", "https")

def check_url(url: str) -> None:
    """Reject references that make the worker read local files or reach internal hosts.

    The url comes straight from user input, so without this the op is an arbitrary file
    read and a full-read SSRF against anything the worker can reach (CWE-918).

    Raises:
        ValueError: If the URL is not an http(s) URL pointing to a public address.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES or not parsed.hostname:
        raise ValueError(
            f"Refusing to fetch reference {url!r}: only {'/'.join(ALLOWED_SCHEMES)} URLs "
            "are supported."
        )

    try:
        addresses = socket.getaddrinfo(parsed.hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise ValueError(f"Refusing to fetch reference {url!r}: could not resolve host") from e

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global or ip.is_multicast:
            raise ValueError(
                f"Refusing to fetch reference {url!r}: host resolves to non-public address {ip}."
            )


def check_redirect(response: Any, *args: Any, **kwargs: Any) -> Any:
    """Validate redirect targets before requests follows them."""
    location = response.headers.get("location")
    if response.is_redirect and location:
        check_url(urljoin(response.url, location))
    return response


def hash_file(filepath: str, chunk_size: int = CHUNK_SIZE_BYTES) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def get_empty_type(t: Any):
    o = get_origin(t)
    if o is not None:
        return o()
    return t()


def get_empty_fields(data_type: Type[DataVibe]) -> Dict[str, Any]:
    base_fields = [f for f in fields(DataVibe) if f.init]
    init_fields = [f for f in fields(data_type) if f.init and f not in base_fields]
    return {f.name: get_empty_type(f.type) for f in init_fields}


def add_mime_type(extension: str):
    if extension == ".geojson":
        mimetypes.add_type("application/json", ".geojson")


class CallbackBuilder:
    def __init__(self, out_type: str):
        self.tmp_dir = TemporaryDirectory()
        self.out_type = cast(Type[DataVibe], data_registry.retrieve(out_type))

    def __call__(self):
        def callback(input_ref: ExternalReference) -> Dict[str, DataVibe]:
            # Download the file
            check_url(input_ref.url)
            out_path = os.path.join(self.tmp_dir.name, uri_to_filename(input_ref.url))
            if is_local(input_ref.url):
                shutil.copy(local_uri_to_path(input_ref.url), out_path)
            else:
                download_file(input_ref.url, out_path, hooks={"response": check_redirect})

            file_extension = pathlib.Path(out_path).suffix
            if file_extension not in mimetypes.types_map.keys():
                add_mime_type(file_extension)

            # Create asset and Raster
            asset_id = hash_file(out_path)
            asset = AssetVibe(
                reference=out_path, type=mimetypes.guess_type(out_path)[0], id=asset_id
            )
            out = self.out_type.clone_from(
                input_ref,
                id=gen_hash_id(asset_id, input_ref.geometry, input_ref.time_range),
                assets=[asset],
                **get_empty_fields(self.out_type),
            )
            return {"downloaded": out}

        return callback

    def __del__(self):
        self.tmp_dir.cleanup()
