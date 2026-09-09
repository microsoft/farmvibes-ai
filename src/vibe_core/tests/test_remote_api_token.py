# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import base64
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict
from unittest.mock import Mock

import pytest

import vibe_core.cli.main as cli_main
import vibe_core.cli.remote as remote
import vibe_core.cli.wrappers as wrappers
from vibe_core.cli.osartifacts import OSArtifacts
from vibe_core.cli.parsers import RemoteCliParser
from vibe_core.cli.wrappers import KubectlWrapper


def encoded_secret(token: str) -> Dict[str, Any]:
    return {
        "type": "Opaque",
        "data": {"token": base64.b64encode(token.encode()).decode()},
    }


def test_cli_exits_when_dispatch_reports_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(sys, "argv", ["farmvibes-ai", "remote", "status"])
    monkeypatch.setattr(cli_main, "dispatch_remote", Mock(return_value=False))
    monkeypatch.setattr(cli_main, "setup_logging", Mock(return_value="log"))

    with pytest.raises(SystemExit) as error:
        cli_main.main()

    assert error.value.code == 1


def configured_artifacts(tmp_path: Path) -> Mock:
    artifacts = Mock(spec=OSArtifacts)
    artifacts.private_config_dir = tmp_path / "private"
    artifacts.private_config_dir.mkdir()
    artifacts.kubectl = "kubectl"
    artifacts.config_file.side_effect = lambda name: str(tmp_path / name)
    return artifacts


def test_opaque_secret_upsert_uses_censored_private_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    artifacts = configured_artifacts(tmp_path)
    invocation = {}

    def capture(command: Any, **kwargs: Any) -> str:
        path = Path(command[-1])
        invocation.update(
            command=command,
            kwargs=kwargs,
            mode=path.stat().st_mode & 0o777,
            manifest=json.loads(path.read_text()),
        )
        return ""

    monkeypatch.setattr(wrappers, "execute_cmd", capture)
    KubectlWrapper(artifacts, "cluster").upsert_opaque_secret(
        "farmvibes-api-auth", {"token": "never-log-this"}
    )

    assert invocation["mode"] == 0o600
    assert invocation["manifest"] == {
        "apiVersion": "v1",
        "kind": "Secret",
        "metadata": {"name": "farmvibes-api-auth"},
        "type": "Opaque",
        "data": {
            "token": base64.b64encode(b"never-log-this").decode(),
        },
    }
    assert invocation["command"][:5] == [
        "kubectl",
        "--context",
        "k3d-cluster",
        "apply",
        "-f",
    ]
    assert "never-log-this" not in " ".join(invocation["command"])
    assert invocation["kwargs"]["censor_command"] is True
    assert invocation["kwargs"]["censor_output"] is True
    assert not list(artifacts.private_config_dir.iterdir())


def test_token_create_and_recover(tmp_path: Path):
    artifacts = configured_artifacts(tmp_path)
    kubectl = Mock(spec=KubectlWrapper)
    kubectl.get_secret_or_none.return_value = None

    assert remote.ensure_remote_api_token(kubectl) is True
    created = kubectl.upsert_opaque_secret.call_args.args[1]["token"]
    assert len(base64.urlsafe_b64decode(created + "==")) == 48
    assert not (artifacts.private_config_dir / "remote_api_token").exists()

    kubectl.reset_mock()
    kubectl.get_secret_or_none.return_value = encoded_secret("cluster-token")
    assert remote.ensure_remote_api_token(kubectl) is False
    kubectl.upsert_opaque_secret.assert_not_called()
    assert not (artifacts.private_config_dir / "remote_api_token").exists()


def test_invalid_token_secret_is_replaced(tmp_path: Path):
    artifacts = configured_artifacts(tmp_path)
    kubectl = Mock(spec=KubectlWrapper)
    kubectl.get_secret_or_none.return_value = {"data": {"token": "not-base64!"}}

    assert remote.ensure_remote_api_token(kubectl) is True
    token = kubectl.upsert_opaque_secret.call_args.args[1]["token"]
    assert token
    assert not (artifacts.private_config_dir / "remote_api_token").exists()


def test_redis_migration_backup_is_cluster_scoped(tmp_path: Path):
    artifacts = configured_artifacts(tmp_path)

    first = remote.remote_redis_migration_backup(
        artifacts,
        Mock(cluster_name="cluster", resource_group="group"),
        "subscription",
        "first-uid",
    )
    second = remote.remote_redis_migration_backup(
        artifacts,
        Mock(cluster_name="cluster", resource_group="group"),
        "subscription",
        "second-uid",
    )

    assert first != second
    assert first.parent == artifacts.private_config_dir


def test_remote_kubectl_uses_managed_kubeconfig(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    artifacts = configured_artifacts(tmp_path)
    (tmp_path / "kubeconfig").touch()
    az = Mock(
        os_artifacts=artifacts,
        cluster_name="cluster",
        resource_group="group",
    )
    artifacts.get_kube_context.return_value = "cluster-admin"
    az.get_kubernetes_version.return_value = "1.35.0"

    kubectl = remote._initialize_kubectl(az)

    assert kubectl is not None
    assert kubectl.config_context == "cluster-admin"
    assert os.environ["KUBECONFIG"] == str(tmp_path / "kubeconfig")
    az.refresh_aks_credentials.assert_called_once_with()
    artifacts.ensure_compatible_kubectl.assert_called_once_with("1.35.0")


def test_remote_kubectl_stops_when_requested_cluster_does_not_exist(
    tmp_path: Path,
):
    artifacts = configured_artifacts(tmp_path)
    az = Mock(os_artifacts=artifacts)
    az.refresh_aks_credentials.return_value = False

    assert remote._initialize_kubectl(az) is None
    artifacts.get_kube_context.assert_not_called()


def test_private_file_is_atomically_replaced_with_private_permissions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    artifacts = configured_artifacts(tmp_path)
    path = artifacts.private_config_dir / "remote_api_token"
    path.write_text("old")
    replacements = []
    replace = remote.os.replace

    def capture(source: str, destination: Path):
        assert path.read_text() == "old"
        assert Path(source).stat().st_mode & 0o777 == 0o600
        replacements.append((source, destination))
        replace(source, destination)

    monkeypatch.setattr(remote.os, "replace", capture)
    remote.persist_private_text(path, "new")

    assert len(replacements) == 1
    assert path.read_text() == "new"
    assert path.stat().st_mode & 0o777 == 0o600
    assert list(artifacts.private_config_dir.iterdir()) == [path]


def test_remote_status_recovers_token_from_cluster(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    artifacts = configured_artifacts(tmp_path)
    artifacts.config_file.side_effect = lambda name: str(tmp_path / name)
    az = Mock(cluster_name="cluster", resource_group="group")
    kubectl = Mock(spec=KubectlWrapper)
    kubectl.get_secret_or_none.return_value = encoded_secret("recovered")
    kubectl.url_from_ingress.return_value = "https://example.test"
    monkeypatch.setattr(remote, "TerraformWrapper", Mock())
    monkeypatch.setattr(remote, "_initialize_kubectl", Mock(return_value=kubectl))

    assert remote.status(artifacts, az, "public") is True
    assert (
        artifacts.private_config_dir / "remote_api_token"
    ).read_text() == "recovered"
    url_path = tmp_path / "remote_service_url"
    assert url_path.read_text() == "https://example.test"
    assert url_path.stat().st_mode & 0o777 == 0o600


def test_remote_status_stops_when_requested_cluster_does_not_exist(
    tmp_path: Path,
):
    artifacts = configured_artifacts(tmp_path)
    az = Mock(
        cluster_name="cluster",
        resource_group="group",
        os_artifacts=artifacts,
    )
    az.refresh_aks_credentials.return_value = False

    assert remote.status(artifacts, az, "public") is False
    artifacts.get_kube_context.assert_not_called()


def test_legacy_ingress_service_is_released_before_chart_cutover() -> None:
    kubectl = Mock(spec=KubectlWrapper)
    kubectl.context.return_value = nullcontext()
    kubectl.get_or_none.side_effect = [
        {
            "metadata": {
                "labels": {
                    "app.kubernetes.io/managed-by": "Helm",
                    "app.kubernetes.io/instance": "ingress-nginx",
                }
            }
        },
        None,
    ]

    assert remote.remove_legacy_ingress_service(kubectl)
    kubectl.delete.assert_called_once_with(
        "service",
        remote.LEGACY_INGRESS_SERVICES[0],
        namespace=remote.LEGACY_INGRESS_NAMESPACE,
        wait=False,
    )
    kubectl.wait_for_delete.assert_called_once_with(
        "service",
        remote.LEGACY_INGRESS_SERVICES[0],
        timeout_s=600,
        namespace=remote.LEGACY_INGRESS_NAMESPACE,
    )


def test_legacy_ingress_service_rejects_unknown_owner() -> None:
    kubectl = Mock(spec=KubectlWrapper)
    kubectl.context.return_value = nullcontext()
    kubectl.get_or_none.return_value = {
        "metadata": {"labels": {"app.kubernetes.io/instance": "someone-else"}}
    }

    with pytest.raises(RuntimeError, match="Refusing to replace"):
        remote.remove_legacy_ingress_service(kubectl)

    kubectl.delete.assert_not_called()


def test_legacy_ingress_validates_all_services_before_deletion() -> None:
    kubectl = Mock(spec=KubectlWrapper)
    kubectl.context.return_value = nullcontext()
    kubectl.get_or_none.side_effect = [
        {
            "metadata": {
                "labels": {
                    "app.kubernetes.io/managed-by": "Helm",
                    "app.kubernetes.io/instance": "ingress-nginx",
                }
            }
        },
        {"metadata": {"labels": {}}},
    ]

    with pytest.raises(RuntimeError, match="Refusing to replace"):
        remote.remove_legacy_ingress_service(kubectl)

    kubectl.delete.assert_not_called()


def test_remote_status_keeps_previous_pair_when_url_discovery_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    artifacts = configured_artifacts(tmp_path)
    artifacts.config_file.side_effect = lambda name: str(tmp_path / name)
    token_path = artifacts.private_config_dir / "remote_api_token"
    url_path = tmp_path / "remote_service_url"
    token_path.write_text("old-token")
    url_path.write_text("https://old.test")
    az = Mock(cluster_name="cluster", resource_group="group")
    kubectl = Mock(spec=KubectlWrapper)
    kubectl.get_secret_or_none.return_value = encoded_secret("new-token")
    kubectl.url_from_ingress.return_value = None
    terraform = Mock()
    terraform.get_url_from_terraform_output.return_value = None
    monkeypatch.setattr(remote, "TerraformWrapper", Mock(return_value=terraform))
    monkeypatch.setattr(remote, "_initialize_kubectl", Mock(return_value=kubectl))

    assert remote.status(artifacts, az, "public") is False
    assert token_path.read_text() == "old-token"
    assert url_path.read_text() == "https://old.test"


def test_remote_status_does_not_persist_terraform_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    artifacts = configured_artifacts(tmp_path)
    artifacts.config_file.side_effect = lambda name: str(tmp_path / name)
    token_path = artifacts.private_config_dir / "remote_api_token"
    url_path = tmp_path / "remote_service_url"
    token_path.write_text("old-token")
    url_path.write_text("https://old.test")
    az = Mock(cluster_name="cluster", resource_group="group")
    kubectl = Mock(spec=KubectlWrapper)
    kubectl.get_secret_or_none.return_value = encoded_secret("new-token")
    kubectl.url_from_ingress.return_value = None
    terraform = Mock()
    terraform.get_url_from_terraform_output.return_value = "https://stale.test"
    monkeypatch.setattr(remote, "TerraformWrapper", Mock(return_value=terraform))
    monkeypatch.setattr(remote, "_initialize_kubectl", Mock(return_value=kubectl))

    assert remote.status(artifacts, az, "public") is False
    assert token_path.read_text() == "old-token"
    assert url_path.read_text() == "https://old.test"


def test_remote_config_update_restores_previous_pair_on_write_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    artifacts = configured_artifacts(tmp_path)
    artifacts.config_file.side_effect = lambda name: str(tmp_path / name)
    token_path = artifacts.private_config_dir / "remote_api_token"
    url_path = tmp_path / "remote_service_url"
    token_path.write_text("old-token")
    url_path.write_text("https://old.test")
    persist = remote.persist_private_text
    calls = 0

    def fail_token_write(path: Path, value: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("token write failed")
        persist(path, value)

    monkeypatch.setattr(remote, "persist_private_text", fail_token_write)

    with pytest.raises(OSError, match="token write failed"):
        remote.persist_remote_api_config(
            artifacts, "https://new.test", "new-token"
        )

    assert token_path.read_text() == "old-token"
    assert url_path.read_text() == "https://old.test"


@pytest.mark.parametrize(
    "secret",
    [
        None,
        {},
        {"data": {}},
        {"data": {"token": "not-base64!"}},
        {"data": {"token": base64.b64encode(b"\xff").decode()}},
        encoded_secret(""),
    ],
)
def test_remote_status_rejects_invalid_secret_without_overwriting_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    secret: Any,
):
    artifacts = configured_artifacts(tmp_path)
    token_path = artifacts.private_config_dir / "remote_api_token"
    token_path.write_text("known-good")
    az = Mock(cluster_name="cluster", resource_group="group")
    kubectl = Mock(spec=KubectlWrapper)
    kubectl.get_secret_or_none.return_value = secret
    monkeypatch.setattr(remote, "TerraformWrapper", Mock())
    monkeypatch.setattr(remote, "_initialize_kubectl", Mock(return_value=kubectl))

    assert remote.status(artifacts, az, "public") is False
    assert token_path.read_text() == "known-good"
    kubectl.url_from_ingress.assert_not_called()
    artifacts.config_file.assert_not_called()


def test_remote_update_rotation_flag_is_update_only():
    parser = RemoteCliParser("remote")
    required = ["--region", "eastus", "--cert-email", "user@example.test"]

    assert parser.parse(["update", *required]).rotate_api_token is False
    assert (
        parser.parse(["update", *required, "--rotate-api-token"]).rotate_api_token
        is True
    )
    with pytest.raises(SystemExit):
        parser.parse(["setup", *required, "--rotate-api-token"])


def configured_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    token_changed: Any,
):
    artifacts = configured_artifacts(tmp_path)
    (tmp_path / "kubeconfig").touch()
    az = Mock(
        cluster_name="cluster",
        resource_group="group",
        os_artifacts=artifacts,
    )
    az.get_subscription_and_tenant_id.return_value = ("subscription", "tenant")
    az.check_resource_providers.return_value = True
    az.get_current_user_name.return_value = "admin"
    az.cluster_exists.return_value = True
    az.ensure_azurerm_backend.return_value = ("storage", "container", "key")

    terraform = Mock()
    terraform.get_current_core_count.return_value = (0, 0)
    terraform.workspace.return_value = nullcontext()
    terraform.ensure_infra.return_value = {
        "kubernetes_config_context": {"value": "context"},
        "public_ip_address": {"value": "ip"},
        "public_ip_fqdn": {"value": "fqdn"},
        "public_ip_dns": {"value": "dns"},
        "keyvault_name": {"value": "vault"},
        "application_id": {"value": "app"},
        "storage_connection_key": {"value": "storage-key"},
        "storage_account_name": {"value": "account"},
        "userfile_container_name": {"value": "userfiles"},
        "monitor_instrumentation_key": {"value": "monitor"},
        "worker_node_pool_name": {"value": "workers"},
    }
    terraform.ensure_k8s_cluster.return_value = {
        "shared_resource_pv_claim_name": {"value": "claim"},
        "otel_service_name": {"value": "otel"},
    }
    kubectl = Mock(spec=KubectlWrapper)
    kubectl.cluster_name = "cluster"
    kubectl.os_artifacts = artifacts
    kubectl.context.return_value = nullcontext()
    kubectl.get_cluster_uid.return_value = "cluster-uid"
    dapr = Mock()
    dapr.needs_upgrade.return_value = False

    monkeypatch.setattr(remote, "TerraformWrapper", Mock(return_value=terraform))
    monkeypatch.setattr(remote, "_initialize_kubectl", Mock(return_value=kubectl))
    cert_manager = Mock()
    cert_manager.needs_upgrade.return_value = False
    monkeypatch.setattr(
        remote, "CertManagerWrapper", Mock(return_value=cert_manager)
    )
    monkeypatch.setattr(remote, "DaprWrapper", Mock(return_value=dapr))
    monkeypatch.setattr(remote, "status", Mock(return_value=True))
    monkeypatch.setattr(remote, "needs_service_migration", Mock(return_value=False))
    monkeypatch.setattr(remote, "quiesce_remote_services", Mock(return_value={}))
    monkeypatch.setattr(remote, "restore_remote_services", Mock())
    monkeypatch.setattr(remote, "backup_redis_data", Mock(return_value=True))
    monkeypatch.setattr(remote, "restore_redis_data", Mock(return_value=True))
    token = (
        Mock(side_effect=token_changed)
        if isinstance(token_changed, Exception)
        else Mock(return_value=token_changed)
    )
    monkeypatch.setattr(remote, "prepare_remote_api_token", token)
    monkeypatch.setattr(remote, "activate_remote_api_token", Mock())
    return artifacts, az, terraform, kubectl, token


def test_update_orders_helm_migration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    artifacts, az, terraform, _, _ = configured_update(
        monkeypatch, tmp_path, None
    )
    migration_backup = remote.remote_redis_migration_backup(
        artifacts, az, "subscription", "cluster-uid"
    )
    migration_backup.write_bytes(b"stale-state")
    order = []

    def backup(*args: Any, **kwargs: Any) -> bool:
        order.append("backup")
        migration_backup.write_bytes(b"redis-state")
        return True

    k8s_results = terraform.ensure_k8s_cluster.return_value
    terraform.ensure_k8s_cluster.side_effect = (
        lambda *args, **kwargs: order.append("native") or k8s_results
    )
    monkeypatch.setattr(remote, "needs_service_migration", Mock(return_value=True))
    monkeypatch.setattr(remote, "backup_redis_data", backup)
    monkeypatch.setattr(
        remote,
        "restore_redis_data",
        lambda *args, **kwargs: order.append("restore") or True,
    )

    assert run_update(artifacts, az) is True
    assert order == ["backup", "native", "restore"]
    assert terraform.ensure_k8s_cluster.call_args.kwargs[
        "migrate_legacy_services"
    ] is True
    assert not migration_backup.exists()


def test_update_resumes_pending_redis_restore(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    artifacts, az, _, kubectl, _ = configured_update(monkeypatch, tmp_path, None)
    migration_backup = remote.remote_redis_migration_backup(
        artifacts, az, "subscription", "cluster-uid"
    )
    migration_backup.write_bytes(b"redis-state")
    backup = Mock()
    restore = Mock(return_value=True)
    monkeypatch.setattr(remote, "backup_redis_data", backup)
    monkeypatch.setattr(remote, "restore_redis_data", restore)

    assert run_update(artifacts, az) is True
    backup.assert_not_called()
    restore.assert_called_once()
    assert not migration_backup.exists()


def test_partial_helm_destroy_reuses_existing_redis_backup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    artifacts, az, _, kubectl, _ = configured_update(monkeypatch, tmp_path, None)
    migration_backup = remote.remote_redis_migration_backup(
        artifacts, az, "subscription", "cluster-uid"
    )
    migration_backup.write_bytes(b"redis-state")
    needs_migration = Mock(side_effect=[True, False])
    backup = Mock()
    monkeypatch.setattr(remote, "needs_service_migration", needs_migration)
    monkeypatch.setattr(remote, "backup_redis_data", backup)

    assert run_update(artifacts, az) is True
    backup.assert_not_called()
    assert needs_migration.call_args_list[0].args == (kubectl,)
    assert needs_migration.call_args_list[1].args == (
        kubectl,
        ("redis-master",),
    )
    assert not migration_backup.exists()


def test_update_rejects_empty_redis_migration_backup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    artifacts, az, terraform, _, _ = configured_update(monkeypatch, tmp_path, None)
    migration_backup = remote.remote_redis_migration_backup(
        artifacts, az, "subscription", "cluster-uid"
    )
    migration_backup.touch()

    assert run_update(artifacts, az) is False
    terraform.ensure_k8s_cluster.assert_not_called()


def test_update_provisions_before_services_and_restarts_services(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    artifacts, az, terraform, kubectl, provision = configured_update(
        monkeypatch, tmp_path, None
    )
    order = []
    provision.side_effect = lambda *args, **kwargs: order.append("token")
    terraform.ensure_services.side_effect = lambda *args, **kwargs: order.append(
        "services"
    )

    assert run_update(artifacts, az) is True
    assert order == ["token", "services"]
    getattr(
        remote.DaprWrapper, "return_value"
    ).prepare_for_terraform_reconciliation.assert_called_once()
    assert provision.call_args.args[1] is False
    kubectl.context.assert_called_once_with("cluster")
    assert [(call.args, call.kwargs) for call in kubectl.restart.call_args_list] == [
        (("deployment",), {"selectors": ["backend=terravibes"]}),
        (("deployment",), {"name": remote.CACHE_DEPLOYMENT}),
    ]
    assert [
        call.args for call in kubectl.rollout_status.call_args_list
    ] == [
        *[("deployment", deployment) for deployment in remote.BACKEND_DEPLOYMENTS],
        ("deployment", remote.CACHE_DEPLOYMENT),
    ]


def test_update_token_failure_is_fail_closed_before_services(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    artifacts, az, terraform, kubectl, _ = configured_update(
        monkeypatch, tmp_path, RuntimeError("secret unavailable")
    )

    assert run_update(artifacts, az) is False
    terraform.ensure_services.assert_not_called()
    kubectl.restart.assert_not_called()


def test_migration_backup_failure_restores_quiesced_services(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    artifacts, az, terraform, kubectl, _ = configured_update(
        monkeypatch, tmp_path, None
    )
    replicas = {"terravibes-rest-api": 1, "terravibes-worker": 3}
    monkeypatch.setattr(remote, "needs_service_migration", Mock(return_value=True))
    monkeypatch.setattr(
        remote, "quiesce_remote_services", Mock(return_value=replicas)
    )
    restore = Mock()
    monkeypatch.setattr(remote, "restore_remote_services", restore)
    monkeypatch.setattr(
        remote, "backup_redis_data", Mock(side_effect=RuntimeError("backup failed"))
    )

    assert run_update(artifacts, az) is False
    restore.assert_called_once_with(kubectl, replicas)
    terraform.ensure_k8s_cluster.assert_not_called()


@pytest.mark.parametrize(
    "takeover_started, restore_expected",
    [(False, True), (True, False)],
)
def test_migration_failure_restores_only_before_native_takeover(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    takeover_started: bool,
    restore_expected: bool,
):
    artifacts, az, terraform, kubectl, _ = configured_update(
        monkeypatch, tmp_path, None
    )
    replicas = {"terravibes-rest-api": 1}
    migration_backup = remote.remote_redis_migration_backup(
        artifacts, az, "subscription", "cluster-uid"
    )
    monkeypatch.setattr(remote, "needs_service_migration", Mock(return_value=True))
    monkeypatch.setattr(
        remote, "quiesce_remote_services", Mock(return_value=replicas)
    )

    def backup(*args: Any, **kwargs: Any) -> bool:
        migration_backup.write_bytes(b"redis-state")
        return True

    def fail(*args: Any, **kwargs: Any) -> None:
        if takeover_started:
            kwargs["on_legacy_destroy"]()
        raise RuntimeError("terraform failed")

    restore = Mock()
    monkeypatch.setattr(remote, "backup_redis_data", backup)
    monkeypatch.setattr(remote, "restore_remote_services", restore)
    terraform.ensure_k8s_cluster.side_effect = fail

    assert run_update(artifacts, az) is False
    assert restore.called is restore_expected
    if restore_expected:
        restore.assert_called_once_with(kubectl, replicas)


def test_quiesce_remote_services_preserves_replica_counts() -> None:
    kubectl = Mock(spec=KubectlWrapper)
    kubectl.get_or_none.side_effect = lambda kind, name: (
        {"spec": {"replicas": 3 if name == "terravibes-worker" else 1}}
        if name in {"terravibes-rest-api", "terravibes-worker"}
        else None
    )

    replicas = remote.quiesce_remote_services(kubectl)

    assert replicas == {"terravibes-rest-api": 1, "terravibes-worker": 3}
    assert [call.args for call in kubectl.scale.call_args_list] == [
        ("deployment", "terravibes-rest-api", 0),
        ("deployment", "terravibes-worker", 0),
    ]
    assert [call.args for call in kubectl.rollout_status.call_args_list] == [
        ("deployment", "terravibes-rest-api"),
        ("deployment", "terravibes-worker"),
    ]


def test_rotation_is_applied_after_services(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    artifacts, az, terraform, kubectl, prepare = configured_update(
        monkeypatch, tmp_path, ("old", "new")
    )
    activation = Mock()
    monkeypatch.setattr(remote, "activate_remote_api_token", activation)
    order = []
    terraform.ensure_services.side_effect = lambda *args, **kwargs: order.append(
        "services"
    )
    kubectl.restart.side_effect = lambda *args, **kwargs: order.append("restart")
    activation.side_effect = lambda *args: order.append("rotate")

    assert run_update(artifacts, az, rotate=True) is True
    assert order == ["services", "restart", "restart", "rotate"]
    prepare.assert_called_once_with(kubectl, True)
    activation.assert_called_once_with(kubectl, "old", "new")
    assert [(call.args, call.kwargs) for call in kubectl.restart.call_args_list] == [
        (("deployment",), {"selectors": ["backend=terravibes"]}),
        (("deployment",), {"name": remote.CACHE_DEPLOYMENT}),
    ]
    assert [
        call.args for call in kubectl.rollout_status.call_args_list
    ] == [
        *[("deployment", deployment) for deployment in remote.BACKEND_DEPLOYMENTS],
        ("deployment", remote.CACHE_DEPLOYMENT),
    ]


def test_rotation_does_not_change_token_when_services_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    artifacts, az, terraform, _, _ = configured_update(
        monkeypatch, tmp_path, ("old", "new")
    )
    activation = Mock()
    monkeypatch.setattr(remote, "activate_remote_api_token", activation)
    terraform.ensure_services.side_effect = RuntimeError("services failed")

    assert run_update(artifacts, az, rotate=True) is False
    activation.assert_not_called()


def test_rotation_does_not_activate_before_general_restart_succeeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    artifacts, az, _, kubectl, _ = configured_update(
        monkeypatch, tmp_path, ("old", "new")
    )
    activation = Mock()
    monkeypatch.setattr(remote, "activate_remote_api_token", activation)
    kubectl.restart.side_effect = RuntimeError("restart failed")

    assert run_update(artifacts, az, rotate=True) is False
    activation.assert_not_called()


def test_rotation_rolls_back_when_final_status_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    artifacts, az, _, kubectl, _ = configured_update(
        monkeypatch, tmp_path, ("old", "new")
    )
    activation = Mock()
    monkeypatch.setattr(remote, "activate_remote_api_token", activation)
    remote.status.return_value = False

    assert run_update(artifacts, az, rotate=True) is False
    assert activation.call_args_list[0].args == (kubectl, "old", "new")
    assert activation.call_args_list[1].args == (kubectl, "new", "old")


@pytest.mark.parametrize("failure", ["restart", "rollout"])
def test_rotation_restores_previous_token_on_rollout_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
):
    artifacts = configured_artifacts(tmp_path)
    token_path = artifacts.private_config_dir / "remote_api_token"
    token_path.write_text("old-token")
    kubectl = Mock(spec=KubectlWrapper)
    if failure == "restart":
        kubectl.restart.side_effect = [RuntimeError("restart failed"), None]
    else:
        kubectl.rollout_status.side_effect = [RuntimeError("rollout failed"), None]

    with pytest.raises(RuntimeError, match=f"{failure} failed"):
        remote.activate_remote_api_token(kubectl, "old-token", "new-token")

    assert [
        call.args[1]["token"] for call in kubectl.upsert_opaque_secret.call_args_list
    ] == ["new-token", "old-token"]
    assert token_path.read_text() == "old-token"


def run_update(artifacts: Mock, az: Mock, rotate: bool = False) -> bool:
    return remote.setup_or_upgrade(
        artifacts,
        az,
        "eastus",
        "user@example.test",
        "registry.example/farmvibes",
        "user",
        "password",
        "farmvibes-ai-",
        "latest",
        "info",
        True,
        worker_replicas=3,
        environment="public",
        rotate_api_token=rotate,
    )
