# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import UUID

import pytest
from mcp import Client
from mcp.types import TextContent

from farmvibes_mcp import server
from vibe_core.datamodel import RunStatus, TaskDescription


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@dataclass
class FakeDetails:
    status: RunStatus


@dataclass
class FakeRunDescription:
    id: UUID
    name: str
    workflow: str
    parameters: dict[str, object]
    details: FakeDetails
    task_details: dict[str, object]
    output: dict[str, object]
    history_compacted: bool = False


@pytest.mark.anyio
async def test_tools_reuse_client_and_return_structured_results() -> None:
    client = Mock()
    client.list_workflows.return_value = ["helloworld"]
    client.describe_workflow.return_value = {
        "name": "helloworld",
        "description": TaskDescription(short_description="Hello"),
    }
    client.run.return_value = SimpleNamespace(
        id="run-1",
        name="example",
        workflow="helloworld",
        parameters={"size": 1},
    )
    client.list_runs.return_value = [{"id": "run-1", "details.status": "done"}]
    client.describe_run.return_value = FakeRunDescription(
        id=UUID("00000000-0000-0000-0000-000000000001"),
        name="example",
        workflow="helloworld",
        parameters={"size": 1},
        details=FakeDetails(RunStatus.done),
        task_details={},
        output={"raster": {"type": "Feature"}},
    )
    client.cancel_run.return_value = "Cancellation requested"

    with patch.object(server, "_client", return_value=client):
        async with Client(server.mcp) as mcp_client:
            tools = await mcp_client.list_tools()
            assert [tool.name for tool in tools.tools] == [
                "list_workflows",
                "describe_workflow",
                "submit_run",
                "list_runs",
                "get_run",
                "get_run_output",
                "cancel_run",
            ]

            assert (
                await mcp_client.call_tool("list_workflows")
            ).structured_content == {"workflows": ["helloworld"]}
            workflow = await mcp_client.call_tool(
                "describe_workflow", {"workflow_name": "helloworld"}
            )
            assert workflow.structured_content["description"]["short_description"] == "Hello"
            submitted = await mcp_client.call_tool(
                "submit_run",
                {
                    "workflow_name": "helloworld",
                    "run_name": "example",
                    "geometry": {"type": "Point", "coordinates": [1, 2]},
                    "start_time": "2024-01-01T00:00:00+00:00",
                    "end_time": "2024-01-02T00:00:00+00:00",
                    "parameters": {"size": 1},
                },
            )
            assert submitted.structured_content == {
                "id": "run-1",
                "name": "example",
                "workflow": "helloworld",
                "parameters": {"size": 1},
            }
            runs = await mcp_client.call_tool(
                "list_runs",
                {"ids": ["run-1"], "fields": ["id", "details.status"]},
            )
            assert runs.structured_content == {
                "runs": [{"id": "run-1", "details.status": "done"}]
            }
            run = await mcp_client.call_tool("get_run", {"run_id": "run-1"})
            assert "output" not in run.structured_content
            assert (
                run.structured_content["id"]
                == "00000000-0000-0000-0000-000000000001"
            )
            output = await mcp_client.call_tool(
                "get_run_output", {"run_id": "run-1"}
            )
            assert output.structured_content == {
                "run_id": "run-1",
                "status": "done",
                "history_compacted": False,
                "output": {"raster": {"type": "Feature"}},
            }
            cancelled = await mcp_client.call_tool(
                "cancel_run", {"run_id": "run-1"}
            )
            assert cancelled.structured_content == {
                "run_id": "run-1",
                "message": "Cancellation requested",
            }

    _, kwargs = client.run.call_args
    assert kwargs["geometry"].wkt == "POINT (1 2)"
    assert kwargs["time_range"][0].isoformat() == "2024-01-01T00:00:00+00:00"
    client.list_runs.assert_called_once_with(
        ids=["run-1"], fields=["id", "details.status"]
    )


@pytest.mark.anyio
async def test_submit_run_reports_input_errors() -> None:
    with patch.object(server, "_client", return_value=Mock()):
        async with Client(server.mcp) as client:
            result = await client.call_tool(
                "submit_run",
                {
                    "workflow_name": "helloworld",
                    "run_name": "bad-time",
                    "geometry": {"type": "Point", "coordinates": [1, 2]},
                    "start_time": "not-a-date",
                    "end_time": "2024-01-02T00:00:00+00:00",
                },
            )
            malformed = await client.call_tool(
                "submit_run",
                {
                    "workflow_name": "helloworld",
                    "run_name": "bad-geometry",
                    "geometry": {},
                    "start_time": "2024-01-01T00:00:00+00:00",
                    "end_time": "2024-01-02T00:00:00+00:00",
                },
            )

    assert result.is_error
    assert isinstance(result.content[0], TextContent)
    assert "start_time must be an ISO 8601 timestamp" in result.content[0].text
    assert malformed.is_error
    assert isinstance(malformed.content[0], TextContent)
    assert "geometry must be a nonempty valid GeoJSON geometry" in malformed.content[0].text


def test_client_uses_existing_default_discovery() -> None:
    expected = Mock()
    with patch.object(server, "get_default_vibe_client", return_value=expected) as get:
        assert server._client() is expected
    get.assert_called_once_with()
