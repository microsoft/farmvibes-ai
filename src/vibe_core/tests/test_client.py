# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import importlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import requests

import vibe_core.client as client_module
from vibe_core.cli.osartifacts import OSArtifacts
from vibe_core.client import (
    FARMVIBES_AI_REMOTE_API_TOKEN_PATH,
    ClusterType,
    FarmvibesAiClient,
    get_default_vibe_client,
    get_vibe_client,
)


def test_remote_token_uses_private_config_directory():
    assert Path(FARMVIBES_AI_REMOTE_API_TOKEN_PATH).parts[-2:] == (
        "private",
        "remote_api_token",
    )


def test_client_and_cli_share_config_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    config_dir = tmp_path / "farmvibes-config"
    with monkeypatch.context() as scoped:
        scoped.setenv("FARMVIBES_AI_CONFIG_DIR", str(config_dir))
        reloaded = importlib.reload(client_module)

        assert OSArtifacts().config_dir == config_dir
        assert Path(reloaded.FARMVIBES_AI_REMOTE_SERVICE_URL_PATH) == (
            config_dir / "remote_service_url"
        )
        assert Path(reloaded.FARMVIBES_AI_REMOTE_API_TOKEN_PATH) == (
            config_dir / "private" / "remote_api_token"
        )
    importlib.reload(client_module)


def test_client_attaches_optional_bearer_token():
    authenticated = FarmvibesAiClient("https://example.test", "secret")
    unauthenticated = FarmvibesAiClient("https://example.test")

    assert authenticated.session.headers["Authorization"] == "Bearer secret"
    assert "Authorization" not in unauthenticated.session.headers
    assert get_vibe_client("https://custom.test", "explicit").session.headers[
        "Authorization"
    ] == "Bearer explicit"


def test_remote_clients_discover_token_and_local_client_stays_unauthenticated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    remote_url = tmp_path / "remote_service_url"
    remote_token = tmp_path / "remote_api_token"
    local_url = tmp_path / "service_url"
    remote_url.write_text("https://remote.test")
    remote_token.write_text("discovered-token\n")
    local_url.write_text("http://local.test")
    monkeypatch.setattr(
        client_module, "FARMVIBES_AI_REMOTE_SERVICE_URL_PATH", str(remote_url)
    )
    monkeypatch.setattr(client_module, "FARMVIBES_AI_REMOTE_API_TOKEN_PATH", str(remote_token))
    monkeypatch.setattr(client_module, "FARMVIBES_AI_SERVICE_URL_PATH", str(local_url))

    default_remote = get_default_vibe_client()
    explicit_remote = get_default_vibe_client(ClusterType.remote)
    local = get_default_vibe_client(ClusterType.local)

    for remote in (default_remote, explicit_remote):
        assert remote.baseurl == "https://remote.test"
        assert remote.session.headers["Authorization"] == "Bearer discovered-token"
    assert local.baseurl == "http://local.test"
    assert "Authorization" not in local.session.headers


def test_missing_token_file_does_not_break_custom_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    remote_url = tmp_path / "remote_service_url"
    remote_url.write_text("https://remote.test")
    monkeypatch.setattr(
        client_module, "FARMVIBES_AI_REMOTE_SERVICE_URL_PATH", str(remote_url)
    )
    monkeypatch.setattr(
        client_module,
        "FARMVIBES_AI_REMOTE_API_TOKEN_PATH",
        str(tmp_path / "missing-remote-api-token"),
    )

    custom = get_vibe_client("https://custom.test")
    default_remote = get_default_vibe_client()

    assert custom.baseurl == "https://custom.test"
    assert "Authorization" not in custom.session.headers
    assert default_remote.baseurl == "https://remote.test"
    assert "Authorization" not in default_remote.session.headers


def test_unauthorized_response_explains_how_to_refresh_remote_access():
    response = requests.Response()
    response.status_code = 401
    response.url = "https://remote.test/v0/workflows"
    response._content = b'{"detail":"Invalid or missing bearer token"}'
    client = FarmvibesAiClient("https://remote.test")
    client.session.request = MagicMock(return_value=response)  # type: ignore

    with pytest.raises(requests.HTTPError, match="farmvibes-ai remote update") as exc:
        client.list_workflows()

    assert "farmvibes-ai remote status" in str(exc.value)
