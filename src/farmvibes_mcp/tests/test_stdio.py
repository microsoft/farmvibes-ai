# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
import sys
from collections.abc import Generator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import urlparse

import pytest
from mcp import Client, StdioServerParameters
from mcp.types import TextContent

RUN_ID = "00000000-0000-0000-0000-000000000001"


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


class FarmVibesHandler(BaseHTTPRequestHandler):
    submitted: dict[str, Any] = {}

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _write(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path)
        if path.path == "/v0/workflows":
            self._write(["helloworld"])
        elif path.path == "/v0/system-metrics":
            self._write({"disk_free": 100 * 1024**3})
        elif path.path == "/v0/runs":
            self._write(
                [
                    {
                        "id": RUN_ID,
                        "name": "stdio-run",
                        "workflow": "helloworld",
                        "parameters": {"size": 1},
                        "details.status": "done",
                    }
                ]
            )
        elif path.path == f"/v0/runs/{RUN_ID}":
            self._write(
                {
                    "id": RUN_ID,
                    "name": "stdio-run",
                    "workflow": "helloworld",
                    "parameters": {"size": 1},
                    "user_input": FarmVibesHandler.submitted["user_input"],
                    "details": {"status": "done"},
                    "task_details": {},
                    "spatio_temporal_json": None,
                    "output": {"raster": {"type": "Feature"}},
                }
            )
        else:
            self._write({"message": "not found"}, 404)

    def do_POST(self) -> None:
        if self.path != "/v0/runs":
            self._write({"message": "not found"}, 404)
            return
        length = int(self.headers["Content-Length"])
        FarmVibesHandler.submitted = json.loads(self.rfile.read(length))
        self._write({"id": RUN_ID})


@pytest.fixture
def farmvibes_server(tmp_path: Path) -> Generator[dict[str, str], None, None]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), FarmVibesHandler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    (tmp_path / "service_url").write_text(
        f"http://127.0.0.1:{server.server_port}/"
    )
    try:
        yield {**os.environ, "FARMVIBES_AI_CONFIG_DIR": str(tmp_path)}
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.anyio
async def test_real_stdio_server(farmvibes_server: dict[str, str]) -> None:
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "farmvibes_mcp"],
        env=farmvibes_server,
    )
    async with Client(parameters) as client:
        tools = await client.list_tools()
        assert len(tools.tools) == 7
        workflows = await client.call_tool("list_workflows")
        assert workflows.structured_content == {"workflows": ["helloworld"]}

        submitted = await client.call_tool(
            "submit_run",
            {
                "workflow_name": "helloworld",
                "run_name": "stdio-run",
                "geometry": {"type": "Point", "coordinates": [1, 2]},
                "start_time": "2024-01-01T00:00:00+00:00",
                "end_time": "2024-01-02T00:00:00+00:00",
                "parameters": {"size": 1},
            },
        )
        assert submitted.structured_content["id"] == RUN_ID
        assert FarmVibesHandler.submitted["workflow"] == "helloworld"
        assert FarmVibesHandler.submitted["user_input"]["geojson"]["features"][0][
            "geometry"
        ] == {"type": "Point", "coordinates": [1.0, 2.0]}

        output = await client.call_tool("get_run_output", {"run_id": RUN_ID})
        assert output.structured_content["status"] == "done"
        assert output.structured_content["output"] == {
            "raster": {"type": "Feature"}
        }

        invalid = await client.call_tool(
            "submit_run",
            {
                "workflow_name": "helloworld",
                "run_name": "invalid",
                "geometry": {"type": "Point", "coordinates": [1, 2]},
                "start_time": "not-a-date",
                "end_time": "2024-01-02T00:00:00+00:00",
            },
        )
        assert invalid.is_error
        assert isinstance(invalid.content[0], TextContent)
        assert "ISO 8601" in invalid.content[0].text
