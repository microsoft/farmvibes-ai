# FarmVibes.AI MCP server

The `farmvibes-mcp` package lets Model Context Protocol clients inspect
FarmVibes.AI workflows, submit runs, and retrieve results over stdio. It uses
the existing Python client directly, so local and remote clusters behave the
same way.

## Install

Use a dedicated Python 3.11 environment. Do not install the MCP SDK into the
FarmVibes service environment: the current MCP and service web stacks require
different Starlette, Uvicorn, and OpenTelemetry versions.

```bash
python -m venv .venv-mcp
.venv-mcp/bin/pip install ./src/vibe_core ./src/farmvibes_mcp
```

The server uses the Python client's existing target discovery:

- `remote_service_url` and `private/remote_api_token` for a configured remote
  cluster;
- `service_url` for a configured local cluster;
- `http://127.0.0.1:31108/` otherwise.

Run `farmvibes-ai remote status` to recover a missing remote URL or token.
Tokens are never MCP arguments or results and do not belong in MCP
configuration.

## Configure an MCP client

For VS Code, add `.vscode/mcp.json`:

```json
{
  "servers": {
    "farmvibes": {
      "type": "stdio",
      "command": "${workspaceFolder}/.venv-mcp/bin/farmvibes-mcp"
    }
  }
}
```

Clients that use the `mcpServers` format can launch the same executable:

```json
{
  "mcpServers": {
    "farmvibes": {
      "command": "/absolute/path/to/.venv-mcp/bin/farmvibes-mcp"
    }
  }
}
```

## Tools

| Tool | Purpose |
|---|---|
| `list_workflows` | List workflows available on the selected cluster. |
| `describe_workflow` | Show a workflow's inputs, outputs, parameters, and tasks. |
| `submit_run` | Submit GeoJSON geometry, a time range, and optional parameters. |
| `list_runs` | List runs, optionally filtering IDs and response fields. |
| `get_run` | Get run metadata and task details without the potentially large output. |
| `get_run_output` | Get a run's status and output. |
| `cancel_run` | Request cancellation of a pending or running run. |

`submit_run` requires a GeoJSON geometry and timezone-aware ISO 8601
timestamps. Submission is nonblocking: poll with `get_run`, then call
`get_run_output` when the run is done.

Useful requests include:

- "List the available FarmVibes workflows."
- "Describe the inputs and parameters for `helloworld`."
- "Run `helloworld` over this GeoJSON polygon from
  `2021-02-01T00:00:00+00:00` to `2021-02-11T00:00:00+00:00`."
- "Check that run and show its output when it finishes."

## Try it in Codespaces

The repository devcontainer supports the local k3d cluster in Codespaces. A
32-core Codespace is fastest, but the setup is otherwise the normal local
workflow:

```bash
python -m venv /workspaces/.venv-mcp
uv pip install --python /workspaces/.venv-mcp/bin/python \
  -e ./src/vibe_core -e ./src/farmvibes_mcp

FARMVIBES_AI_CONFIG_DIR=/workspaces/fv-config \
  /workspaces/.venv-mcp/bin/farmvibes-ai local setup \
  --auto-confirm \
  --cluster-name farmvibes-mcp \
  --servers 1 \
  --agents 1 \
  --worker-replicas 1 \
  --disable-telemetry \
  --host 127.0.0.1
```

Point the MCP process at the same configuration directory:

```bash
FARMVIBES_AI_CONFIG_DIR=/workspaces/fv-config \
  /workspaces/.venv-mcp/bin/farmvibes-mcp
```

Delete the test cluster when finished:

```bash
FARMVIBES_AI_CONFIG_DIR=/workspaces/fv-config \
  /workspaces/.venv-mcp/bin/farmvibes-ai local destroy \
  --auto-confirm \
  --cluster-name farmvibes-mcp
```

## Current scope

This initial server supports stdio only. It does not expose prompts,
resources, asset downloads, arbitrary existing-asset submission, or a
blocking wait tool.
