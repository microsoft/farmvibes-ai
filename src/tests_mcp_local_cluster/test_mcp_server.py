# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys

import pytest
from mcp import Client, StdioServerParameters

EXPECTED_TOOLS = [
    "list_workflows",
    "describe_workflow",
    "submit_run",
    "list_runs",
    "get_run",
    "get_run_output",
    "cancel_run",
]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_mcp_server_connects_to_local_cluster() -> None:
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "farmvibes_mcp"],
        env=dict(os.environ),
    )
    async with Client(parameters) as client:
        tools = await client.list_tools()
        assert [tool.name for tool in tools.tools] == EXPECTED_TOOLS

        workflows = await client.call_tool("list_workflows")
        assert "helloworld" in workflows.structured_content["workflows"]
