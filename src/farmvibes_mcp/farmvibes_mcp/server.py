# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""FarmVibes.AI MCP server."""

import json
from collections.abc import Callable
from datetime import datetime
from typing import Any, TypeVar

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from requests import RequestException
from shapely.errors import ShapelyError
from shapely.geometry import shape

from vibe_core.client import FarmvibesAiClient, get_default_vibe_client
from vibe_core.data.json_converter import dump_to_json

T = TypeVar("T")

mcp = MCPServer("FarmVibes.AI")


def _client() -> FarmvibesAiClient:
    return get_default_vibe_client()


def _execute(action: Callable[[], T]) -> T:
    try:
        return action()
    except (RequestException, RuntimeError, TypeError, ValueError) as error:
        raise ToolError(str(error)) from error


def _jsonable(value: Any) -> Any:
    return json.loads(dump_to_json(value))


def _timestamp(value: str, name: str) -> datetime:
    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as error:
        raise ToolError(f"{name} must be an ISO 8601 timestamp") from error
    if timestamp.utcoffset() is None:
        raise ToolError(f"{name} must include a UTC offset")
    return timestamp


@mcp.tool()
def list_workflows() -> dict[str, Any]:
    """List workflows available on the configured local or remote FarmVibes.AI cluster."""
    return {"workflows": _execute(lambda: _client().list_workflows())}


@mcp.tool()
def describe_workflow(workflow_name: str) -> dict[str, Any]:
    """Describe a workflow's inputs, outputs, parameters, and tasks."""
    return _jsonable(_execute(lambda: _client().describe_workflow(workflow_name)))


@mcp.tool()
def submit_run(
    workflow_name: str,
    run_name: str,
    geometry: dict[str, Any],
    start_time: str,
    end_time: str,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Submit a workflow using GeoJSON geometry and timezone-aware ISO 8601 timestamps."""

    def submit() -> dict[str, Any]:
        try:
            parsed_geometry = shape(geometry)
        except (AttributeError, KeyError, TypeError, ValueError, ShapelyError) as error:
            raise ValueError("geometry must be a nonempty valid GeoJSON geometry") from error
        if parsed_geometry.is_empty or not parsed_geometry.is_valid:
            raise ValueError("geometry must be a nonempty valid GeoJSON geometry")
        start = _timestamp(start_time, "start_time")
        end = _timestamp(end_time, "end_time")
        if start > end:
            raise ValueError("start_time must not be after end_time")
        run = _client().run(
            workflow_name,
            run_name,
            geometry=parsed_geometry,
            time_range=(start, end),
            parameters=parameters,
        )
        return _jsonable(
            {
                "id": run.id,
                "name": run.name,
                "workflow": run.workflow,
                "parameters": run.parameters,
            }
        )

    return _execute(submit)


@mcp.tool()
def list_runs(
    ids: list[str] | None = None,
    fields: list[str] | None = None,
) -> dict[str, Any]:
    """List workflow runs, optionally filtered by IDs and response fields."""
    runs = _execute(lambda: _client().list_runs(ids=ids, fields=fields))
    return {"runs": _jsonable(runs)}


@mcp.tool()
def get_run(run_id: str) -> dict[str, Any]:
    """Get run metadata and task details without its potentially large output."""
    run = _execute(lambda: _client().describe_run(run_id))
    result = _jsonable(run)
    result.pop("output", None)
    return result


@mcp.tool()
def get_run_output(run_id: str) -> dict[str, Any]:
    """Get a run's status and output."""
    run = _execute(lambda: _client().describe_run(run_id))
    return _jsonable(
        {
            "run_id": run_id,
            "status": run.details.status,
            "history_compacted": run.history_compacted,
            "output": run.output,
        }
    )


@mcp.tool()
def cancel_run(run_id: str) -> dict[str, str]:
    """Request cancellation of a pending or running workflow run."""
    return {
        "run_id": run_id,
        "message": _execute(lambda: _client().cancel_run(run_id)),
    }


def main() -> None:
    mcp.run()
