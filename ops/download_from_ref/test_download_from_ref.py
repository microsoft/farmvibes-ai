# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Iterator, List

import pytest
import download_from_ref
from download_from_ref import CallbackBuilder, check_redirect

from vibe_core.data import ExternalReference
from vibe_core.file_downloader import download_file

HITS: List[str] = []


def make_ref(url: str) -> ExternalReference:
    now = datetime.now()
    return ExternalReference(
        id="test_id",
        time_range=(now, now),
        geometry={"type": "Point", "coordinates": [0.0, 0.0]},
        assets=[],
        url=url,
    )


def make_handler(redirect_to: str = ""):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            HITS.append(self.path)
            if redirect_to:
                self.send_response(302)
                self.send_header("Location", redirect_to)
            else:
                self.send_response(200)
                self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args: Any):
            pass

    return Handler


@pytest.fixture
def redirect_to_internal() -> Iterator[str]:
    """Serve a url that redirects into another (internal) host, recording every request."""
    internal = HTTPServer(("127.0.0.1", 0), make_handler())
    attacker = HTTPServer(
        ("127.0.0.1", 0), make_handler(f"http://127.0.0.1:{internal.server_port}/x")
    )
    for server in (internal, attacker):
        threading.Thread(target=server.serve_forever, daemon=True).start()
    HITS.clear()
    yield f"http://127.0.0.1:{attacker.server_port}/redirect"
    for server in (internal, attacker):
        server.shutdown()


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",  # arbitrary local file read
        "/etc/passwd",
        "ftp://example.com/raster.tif",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata service
        "http://127.0.0.1:8080/secret",
        "http://10.0.0.5/secret",
        "http://[::1]/secret",
    ],
)
def test_op_refuses_unsafe_refs(url: str):
    with pytest.raises(ValueError):
        CallbackBuilder("Raster")()(make_ref(url))


def test_redirect_into_internal_host_is_not_issued(redirect_to_internal: str, tmp_path: Any):
    with pytest.raises(ValueError):
        download_file(
            redirect_to_internal, str(tmp_path / "out"), hooks={"response": check_redirect}
        )
    assert HITS == ["/redirect"], f"the redirect hop should never be issued, got {HITS}"


def test_callback_pins_host_resolution(monkeypatch: Any):
    hostname = "example.com"
    first_resolution = [(2, 1, 6, "", ("93.184.216.34", 0))]
    second_resolution = [(2, 1, 6, "", ("127.0.0.1", 0))]
    resolution_calls = 0
    download_resolution: List[str] = []

    def fake_getaddrinfo(host: str, *args: Any, **kwargs: Any):
        nonlocal resolution_calls
        if host != hostname:
            raise AssertionError(f"Unexpected host resolution for {host!r}")
        resolution_calls += 1
        return first_resolution if resolution_calls == 1 else second_resolution

    def fake_download_file(url: str, file_path: str, **kwargs: Any):
        download_resolution.append(download_from_ref.socket.getaddrinfo(hostname, None)[0][4][0])
        Path(file_path).write_bytes(b"")
        return file_path

    monkeypatch.setattr(download_from_ref.socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(download_from_ref, "download_file", fake_download_file)

    CallbackBuilder("Raster")()(make_ref(f"http://{hostname}/asset.tif"))

    assert resolution_calls == 1
    assert download_resolution == ["93.184.216.34"]
