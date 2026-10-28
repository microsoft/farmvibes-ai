# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from importlib.util import find_spec

# MCP tests run in an isolated Python 3.11+ environment, not the legacy service stack.
collect_ignore_glob = (
    []
    if find_spec("mcp") is not None
    else [
        "src/farmvibes_mcp/tests/test_*.py",
        "src/tests_mcp_local_cluster/test_*.py",
    ]
)
