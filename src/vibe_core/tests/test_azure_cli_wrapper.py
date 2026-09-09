# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import base64
import gzip
import json
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import Mock, PropertyMock, patch

import pytest
import requests

from vibe_core.cli import remote
from vibe_core.cli.osartifacts import (
    KubectlInstaller,
    OpenTofuInstaller,
    OSArtifacts,
    github_api_headers,
    secure_path,
)
from vibe_core.cli.wrappers import (
    AzureCliWrapper,
    CertManagerWrapper,
    DaprWrapper,
    KubectlWrapper,
    TerraformWrapper,
)


def test_get_storage_account_key_accepts_wrapped_azure_cli_response() -> None:
    az = AzureCliWrapper(OSArtifacts(), "cluster", "resource-group")

    with patch(
        "vibe_core.cli.wrappers.execute_cmd",
        return_value='{"keys": [{"value": "storage-key"}]}',
    ):
        assert az.get_storage_account_key("storage") == "storage-key"


def test_aks_upgrade_advances_one_minor_at_a_time() -> None:
    artifacts = Mock(spec=OSArtifacts)
    artifacts.az = "az"
    az = AzureCliWrapper(artifacts, "cluster", "group")
    versions = Mock(
        side_effect=["1.30.9", "1.31.8", "1.32.7"]
    )
    az.get_kubernetes_version = versions

    def execute(command: List[str], **kwargs: Any) -> str:
        if "get-upgrades" in command:
            return (
                '["1.31.7", "1.31.8"]'
                if versions.call_count == 1
                else '["1.32.7"]'
            )
        return ""

    with patch("vibe_core.cli.wrappers.execute_cmd", side_effect=execute) as run:
        az.ensure_kubernetes_version()

    upgrades = [
        call.args[0]
        for call in run.call_args_list
        if "upgrade" in call.args[0]
    ]
    assert [
        command[command.index("--kubernetes-version") + 1]
        for command in upgrades
    ] == ["1.31.8", "1.32.7"]
    assert "--control-plane-only" not in upgrades[0]
    assert "--control-plane-only" in upgrades[1]


def test_state_storage_hardening_enables_versioning_and_soft_delete() -> None:
    artifacts = Mock(spec=OSArtifacts)
    artifacts.az = "az"
    az = AzureCliWrapper(artifacts, "cluster", "resource-group")

    with patch("vibe_core.cli.wrappers.execute_cmd") as execute:
        az.harden_storage_account("storage")

    command = execute.call_args.args[0]
    assert command[:6] == [
        "az",
        "storage",
        "account",
        "blob-service-properties",
        "update",
        "--account-name",
    ]
    assert "--enable-versioning" in command
    assert "--enable-delete-retention" in command
    assert "--enable-container-delete-retention" in command
    assert command[command.index("--delete-retention-days") + 1] == "14"
    assert command[command.index("--container-delete-retention-days") + 1] == "14"


def test_infra_replacement_does_not_restart_workloads_mid_upgrade(
    tmp_path: Path,
) -> None:
    artifacts = Mock(spec=OSArtifacts)
    artifacts.aks_directory = "/terraform/aks"
    artifacts.config_dir = tmp_path
    artifacts.get_terraform_file.return_value = "/tmp/infra.tfstate"
    terraform = TerraformWrapper(artifacts)
    events = []
    terraform.init = Mock(side_effect=lambda *args, **kwargs: events.append("init"))

    def plan(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        events.append("plan")
        return {}

    terraform.plan = Mock(side_effect=plan)
    terraform._get_replacements = Mock(return_value=["node-pool"])
    terraform._has_storage_replacement = Mock(return_value=False)
    terraform.apply = Mock()
    terraform.get_output = Mock(return_value={"state": "ready"})
    upgrade = Mock(side_effect=lambda: events.append("upgrade"))

    with (
        patch("vibe_core.cli.wrappers.verify_to_proceed", return_value=True),
        patch("vibe_core.cli.wrappers.KubectlWrapper") as kubectl_class,
    ):
        result = terraform.ensure_infra(
            "tenant",
            "subscription",
            "westus2",
            "cluster",
            "group",
            1,
            "storage",
            "state",
            "key",
            False,
            after_init=upgrade,
        )

    assert result == {"state": "ready"}
    assert events[:3] == ["init", "upgrade", "plan"]
    kubectl_class.assert_not_called()


def test_ingress_cutover_runs_before_kubernetes_apply() -> None:
    artifacts = Mock(spec=OSArtifacts)
    artifacts.aks_directory = "/terraform/aks"
    artifacts.config_file.return_value = "/tmp/kubeconfig"
    artifacts.get_terraform_file.return_value = "/tmp/kubernetes.tfstate"
    terraform = TerraformWrapper(artifacts)
    events = []
    terraform.init = Mock()
    terraform.destroy_legacy_service_charts = Mock(
        side_effect=lambda *args: events.append("legacy")
    )
    terraform.apply = Mock(
        side_effect=lambda *args: events.append("apply")
    )
    terraform.get_output = Mock(return_value={})

    terraform.ensure_k8s_cluster(
        cluster_name="cluster",
        tenant_id="tenant",
        registry_path="registry",
        registry_username="username",
        registry_password="password",
        resource_group="group",
        current_user_name="user",
        certificate_email="user@example.test",
        kubernetes_config_context="context",
        public_ip_address="ip",
        public_ip_fqdn="fqdn",
        public_ip_dns="dns",
        keyvault_name="vault",
        application_id="application",
        storage_connection_key="storage-key",
        storage_account_name="storage",
        userfile_container_name="user-files",
        monitor_instrumentation_key="monitor",
        backend_storage_name="backend",
        backend_container_name="container",
        backend_storage_access_key="backend-key",
        enable_telemetry=False,
        migrate_legacy_services=True,
        before_apply=lambda: events.append("ingress"),
    )

    assert events == ["legacy", "ingress", "apply"]


@pytest.mark.parametrize("resources", [[], [{"type": "test"}]])
def test_services_state_migrates_before_legacy_secret_is_deleted(
    resources: List[Dict[str, str]],
) -> None:
    artifacts = Mock(spec=OSArtifacts)
    artifacts.aks_directory = "/terraform/aks"
    artifacts.get_terraform_file.return_value = "/tmp/services.tfstate"
    az = Mock()
    az.blob_exists.return_value = False
    terraform = TerraformWrapper(artifacts, az)
    terraform._legacy_services_state_secrets = Mock(
        return_value=[
            {"metadata": {"name": terraform.LEGACY_SERVICES_STATE_SECRET}}
        ]
    )
    events = []
    legacy_state = {
        "lineage": "lineage",
        "serial": 2,
        "resources": resources,
    }

    @contextmanager
    def lock(*args: Any):
        events.append("lock")
        yield
        events.append("unlock")

    terraform._lock_legacy_services_state = Mock(side_effect=lock)
    terraform._pull_legacy_services_state = Mock(
        side_effect=lambda *args: events.append("pull-legacy") or legacy_state
    )
    terraform.init = Mock(side_effect=lambda *args, **kwargs: events.append("init-azure"))
    terraform._push_state = Mock(
        side_effect=lambda *args: events.append("push-azure")
    )
    terraform._pull_state = Mock(
        side_effect=lambda *args: events.append("verify-azure") or legacy_state
    )
    terraform._delete_legacy_services_state = Mock(
        side_effect=lambda *args: events.append("delete-legacy")
    )
    terraform._mark_legacy_services_state_migrated = Mock(
        side_effect=lambda *args: events.append("mark-migrated")
    )
    terraform.apply = Mock()
    terraform.get_output = Mock(return_value={})

    terraform.ensure_services(
        cluster_name="cluster",
        resource_group="group",
        registry_path="registry",
        kubernetes_config_path="/tmp/kubeconfig",
        kubernetes_config_context="cluster-admin",
        worker_node_pool_name="worker",
        public_ip_fqdn="cluster.example",
        image_prefix="farmai/",
        image_tag="latest",
        shared_resource_pv_claim_name="claim",
        otel_service_name="",
        worker_replicas=1,
        log_level="info",
        backend_storage_name="storage",
        backend_container_name="terraform-state",
        backend_storage_access_key="key",
        migrate_state=True,
    )

    assert events == [
        "lock",
        "pull-legacy",
        "init-azure",
        "push-azure",
        "verify-azure",
        "mark-migrated",
        "delete-legacy",
        "unlock",
    ]
    backend_config = terraform.init.call_args.kwargs["backend_config"]
    assert backend_config == {
        "storage_account_name": "storage",
        "resource_group_name": "group",
        "container_name": "terraform-state",
        "access_key": "key",
    }


def test_legacy_services_state_reader_reassembles_chunks() -> None:
    artifacts = Mock(spec=OSArtifacts)
    artifacts.kubectl = "kubectl"
    terraform = TerraformWrapper(artifacts)
    state = {"lineage": "lineage", "resources": [{"type": "test"}]}
    compressed = gzip.compress(json.dumps(state).encode())
    midpoint = len(compressed) // 2
    terraform._legacy_services_state_secrets = Mock(
        return_value=[
            {
                "metadata": {
                    "name": f"{terraform.LEGACY_SERVICES_STATE_SECRET}-part-1"
                },
                "data": {
                    "tfstate": base64.b64encode(
                        compressed[midpoint:]
                    ).decode()
                },
            },
            {
                "metadata": {
                    "name": terraform.LEGACY_SERVICES_STATE_SECRET
                },
                "data": {
                    "tfstate": base64.b64encode(
                        compressed[:midpoint]
                    ).decode()
                },
            },
        ]
    )

    assert terraform._pull_legacy_services_state("kubeconfig", "context") == state


@pytest.mark.parametrize(
    "target_state",
    [
        {
            "lineage": "other",
            "serial": 2,
            "resources": [{"type": "test"}],
        },
        {
            "lineage": "legacy",
            "serial": 1,
            "resources": [{"type": "test"}],
        },
    ],
)
def test_services_state_retry_requires_current_snapshot(
    target_state: Dict[str, Any],
) -> None:
    artifacts = Mock(spec=OSArtifacts)
    artifacts.aks_directory = "/terraform/aks"
    az = Mock()
    az.blob_exists.return_value = True
    terraform = TerraformWrapper(artifacts, az)
    terraform._legacy_services_state_secrets = Mock(
        return_value=[
            {"metadata": {"name": terraform.LEGACY_SERVICES_STATE_SECRET}}
        ]
    )
    terraform._pull_legacy_services_state = Mock(
        return_value={
            "lineage": "legacy",
            "serial": 2,
            "resources": [{"type": "test"}],
        }
    )
    terraform._pull_state = Mock(return_value=target_state)
    terraform._lock_legacy_services_state = Mock(
        return_value=nullcontext()
    )
    terraform.init = Mock()
    terraform._push_state = Mock()
    terraform._delete_legacy_services_state = Mock()
    terraform._mark_legacy_services_state_migrated = Mock()

    with patch("vibe_core.cli.wrappers.KubectlWrapper") as kubectl_class:
        kubectl_class.return_value.get_or_none.return_value = {"metadata": {}}
        with pytest.raises(
            RuntimeError, match="Services state migration verification failed"
        ):
            terraform.ensure_services(
                cluster_name="cluster",
                resource_group="group",
                registry_path="registry",
                kubernetes_config_path="/tmp/kubeconfig",
                kubernetes_config_context="cluster-admin",
                worker_node_pool_name="worker",
                public_ip_fqdn="cluster.example",
                image_prefix="farmai/",
                image_tag="latest",
                shared_resource_pv_claim_name="claim",
                otel_service_name="",
                worker_replicas=1,
                log_level="info",
                backend_storage_name="storage",
                backend_container_name="terraform-state",
                backend_storage_access_key="key",
                migrate_state=True,
            )

    terraform._push_state.assert_not_called()
    terraform._delete_legacy_services_state.assert_not_called()


def test_services_state_retry_populates_empty_target() -> None:
    artifacts = Mock(spec=OSArtifacts)
    artifacts.aks_directory = "/terraform/aks"
    az = Mock()
    az.blob_exists.return_value = True
    terraform = TerraformWrapper(artifacts, az)
    terraform._legacy_services_state_secrets = Mock(
        return_value=[
            {"metadata": {"name": terraform.LEGACY_SERVICES_STATE_SECRET}}
        ]
    )
    legacy_state = {"lineage": "legacy", "serial": 1, "resources": []}
    terraform._pull_legacy_services_state = Mock(
        return_value=legacy_state
    )
    terraform._pull_state = Mock(side_effect=[{}, legacy_state])
    terraform._lock_legacy_services_state = Mock(
        return_value=nullcontext()
    )
    terraform.init = Mock()
    terraform._push_state = Mock()
    terraform._delete_legacy_services_state = Mock()
    terraform._mark_legacy_services_state_migrated = Mock()
    terraform.apply = Mock()
    terraform.get_output = Mock(return_value={})

    with patch("vibe_core.cli.wrappers.KubectlWrapper") as kubectl_class:
        kubectl_class.return_value.get_or_none.return_value = {"metadata": {}}
        terraform.ensure_services(
            "cluster",
            "group",
            "registry",
            "kubeconfig",
            "context",
            "worker",
            "cluster.example",
            "farmai/",
            "latest",
            "claim",
            "",
            1,
            "info",
            "storage",
            "terraform-state",
            "key",
            migrate_state=True,
        )

    terraform._push_state.assert_called_once_with(
        "/terraform/aks/../services", legacy_state
    )
    terraform._delete_legacy_services_state.assert_called_once()


def test_services_state_migration_rejects_unmanaged_services() -> None:
    artifacts = Mock(spec=OSArtifacts)
    artifacts.aks_directory = "/terraform/aks"
    az = Mock()
    az.blob_exists.return_value = False
    terraform = TerraformWrapper(artifacts, az)
    terraform._legacy_services_state_secrets = Mock(return_value=[])
    terraform._pull_legacy_services_state = Mock(return_value={})
    terraform._lock_legacy_services_state = Mock(
        return_value=nullcontext()
    )

    with patch("vibe_core.cli.wrappers.KubectlWrapper") as kubectl_class:
        kubectl_class.return_value.context.return_value = nullcontext()
        kubectl_class.return_value.get_or_none.return_value = {
            "metadata": {"name": "terravibes-rest-api"}
        }
        with pytest.raises(
            RuntimeError, match="no recoverable OpenTofu state"
        ):
            terraform.ensure_services(
                "cluster",
                "group",
                "registry",
                "kubeconfig",
                "context",
                "worker",
                "cluster.example",
                "farmai/",
                "latest",
                "claim",
                "",
                1,
                "info",
                "storage",
                "terraform-state",
                "key",
                migrate_state=True,
            )


def test_services_state_retry_cleans_verified_orphaned_chunks() -> None:
    artifacts = Mock(spec=OSArtifacts)
    artifacts.aks_directory = "/terraform/aks"
    az = Mock()
    az.blob_exists.return_value = True
    terraform = TerraformWrapper(artifacts, az)
    terraform._legacy_services_state_secrets = Mock(
        return_value=[
            {
                "metadata": {
                    "name": (
                        f"{terraform.LEGACY_SERVICES_STATE_SECRET}-part-1"
                    )
                }
            }
        ]
    )
    target_state = {"lineage": "lineage", "serial": 3, "resources": []}
    terraform._pull_state = Mock(return_value=target_state)
    terraform._legacy_migration_marker = Mock(
        return_value=("lineage", 3)
    )
    terraform._lock_legacy_services_state = Mock(
        return_value=nullcontext()
    )
    terraform.init = Mock()
    terraform._delete_legacy_services_state = Mock()
    terraform.apply = Mock()
    terraform.get_output = Mock(return_value={})

    terraform.ensure_services(
        "cluster",
        "group",
        "registry",
        "kubeconfig",
        "context",
        "worker",
        "cluster.example",
        "farmai/",
        "latest",
        "claim",
        "",
        1,
        "info",
        "storage",
        "terraform-state",
        "key",
        migrate_state=True,
    )

    terraform._delete_legacy_services_state.assert_called_once()


def test_services_state_retry_rejects_empty_target_and_marker() -> None:
    artifacts = Mock(spec=OSArtifacts)
    artifacts.aks_directory = "/terraform/aks"
    az = Mock()
    az.blob_exists.return_value = True
    terraform = TerraformWrapper(artifacts, az)
    terraform._legacy_services_state_secrets = Mock(
        return_value=[
            {
                "metadata": {
                    "name": f"{terraform.LEGACY_SERVICES_STATE_SECRET}-part-1"
                }
            }
        ]
    )
    terraform._pull_state = Mock(return_value={})
    terraform._legacy_migration_marker = Mock(return_value=(None, None))
    terraform._lock_legacy_services_state = Mock(return_value=nullcontext())
    terraform.init = Mock()
    terraform._delete_legacy_services_state = Mock()

    with pytest.raises(RuntimeError, match="cannot be verified"):
        terraform.ensure_services(
            "cluster",
            "group",
            "registry",
            "kubeconfig",
            "context",
            "worker",
            "cluster.example",
            "farmai/",
            "latest",
            "claim",
            "",
            1,
            "info",
            "storage",
            "terraform-state",
            "key",
            migrate_state=True,
        )

    terraform._delete_legacy_services_state.assert_not_called()


def test_state_push_closes_and_removes_temporary_file(tmp_path: Path) -> None:
    artifacts = Mock(spec=OSArtifacts)
    artifacts.private_config_dir = tmp_path
    artifacts.terraform = "tofu"
    terraform = TerraformWrapper(artifacts)
    state_path = None

    def inspect_state(command: List[str], **kwargs: Any) -> str:
        nonlocal state_path
        assert "-force" not in command
        state_path = Path(command[-1])
        assert state_path.read_text() == '{"lineage": "test"}'
        return ""

    with patch(
        "vibe_core.cli.wrappers.execute_cmd", side_effect=inspect_state
    ):
        terraform._push_state("/terraform/services", {"lineage": "test"})

    assert state_path is not None
    assert not state_path.exists()


def test_legacy_services_state_lock_is_released() -> None:
    artifacts = Mock(spec=OSArtifacts)
    artifacts.kubectl = "kubectl"
    terraform = TerraformWrapper(artifacts)
    commands = []

    def execute(command: List[str], **kwargs: Any) -> str:
        commands.append(command)
        if "create" in command:
            json.loads(Path(command[command.index("--filename") + 1]).read_text())
            raise ValueError("lease exists")
        if "get" in command:
            return json.dumps(
                {
                    "metadata": {"resourceVersion": "1"},
                    "spec": {"holderIdentity": None},
                }
            )
        return ""

    with patch("vibe_core.cli.wrappers.execute_cmd", side_effect=execute):
        with terraform._lock_legacy_services_state(
            "kubeconfig", "context"
        ) as verify:
            verify()
            commands.append(["migration"])

    assert "create" in commands[0]
    assert "get" in commands[1]
    acquire = json.loads(commands[2][commands[2].index("--patch") + 1])
    assert any(
        operation["path"] == "/metadata/annotations"
        for operation in acquire
    )
    assert any(
        operation["path"] == "/spec/renewTime"
        for operation in json.loads(
            commands[3][commands[3].index("--patch") + 1]
        )
    )
    assert commands[4] == ["migration"]
    assert "patch" in commands[5]
    release = json.loads(commands[5][commands[5].index("--patch") + 1])
    assert release[1] == {
        "op": "replace",
        "path": "/spec/holderIdentity",
        "value": None,
    }


def test_legacy_services_state_lock_accepts_ambiguous_create() -> None:
    artifacts = Mock(spec=OSArtifacts)
    artifacts.kubectl = "kubectl"
    terraform = TerraformWrapper(artifacts)
    lock_id = "same-lock"

    def execute(command: List[str], **kwargs: Any) -> str:
        if "create" in command:
            raise ValueError("response lost")
        if "get" in command:
            return json.dumps(
                {
                    "metadata": {
                        "resourceVersion": "1",
                        "annotations": {},
                    },
                    "spec": {"holderIdentity": lock_id},
                }
            )
        return ""

    with (
        patch("vibe_core.cli.wrappers.uuid.uuid4", return_value=lock_id),
        patch("vibe_core.cli.wrappers.execute_cmd", side_effect=execute),
    ):
        with terraform._lock_legacy_services_state("kubeconfig", "context"):
            pass


def test_legacy_services_state_lock_recovers_when_stale() -> None:
    artifacts = Mock(spec=OSArtifacts)
    artifacts.kubectl = "kubectl"
    terraform = TerraformWrapper(artifacts)
    commands = []

    def execute(command: List[str], **kwargs: Any) -> str:
        commands.append(command)
        if "create" in command:
            raise ValueError("lease exists")
        if "get" in command:
            return json.dumps(
                {
                    "metadata": {
                        "resourceVersion": "1",
                        "annotations": {},
                    },
                    "spec": {
                        "holderIdentity": "abandoned",
                        "acquireTime": "2020-01-01T00:00:00Z",
                        "leaseDurationSeconds": 900,
                    },
                }
            )
        return ""

    with patch("vibe_core.cli.wrappers.execute_cmd", side_effect=execute):
        with terraform._lock_legacy_services_state(
            "kubeconfig", "context"
        ):
            pass

    acquire = json.loads(commands[2][commands[2].index("--patch") + 1])
    assert any(
        operation["path"] == "/spec/holderIdentity"
        for operation in acquire
    )
    assert any(
        operation["path"] == "/spec/renewTime"
        for operation in acquire
    )


def test_legacy_services_state_is_deleted_from_default_namespace() -> None:
    artifacts = Mock(spec=OSArtifacts)
    terraform = TerraformWrapper(artifacts)
    terraform._legacy_services_state_secrets = Mock(
        return_value=[
            {
                "metadata": {
                    "name": f"{terraform.LEGACY_SERVICES_STATE_SECRET}-part-1"
                }
            },
            {"metadata": {"name": terraform.LEGACY_SERVICES_STATE_SECRET}},
        ]
    )
    with patch("vibe_core.cli.wrappers.KubectlWrapper") as kubectl_class:
        terraform._delete_legacy_services_state(
            "cluster", "kubeconfig", "cluster-admin"
        )

    assert [
        call.args[1] for call in kubectl_class.return_value.delete.call_args_list
    ] == [
        TerraformWrapper.LEGACY_SERVICES_STATE_SECRET,
        f"{TerraformWrapper.LEGACY_SERVICES_STATE_SECRET}-part-1",
    ]
    assert all(
        call.kwargs == {
            "ignore_not_found": True,
            "namespace": "default",
        }
        for call in kubectl_class.return_value.delete.call_args_list
    )


def test_opentofu_installer_uses_official_release_assets() -> None:
    with patch.object(
        OpenTofuInstaller, "latest_release", new_callable=PropertyMock
    ) as latest:
        latest.return_value = "1.12.6"
        installer = OpenTofuInstaller(Path("/tmp"))
        assert installer.urls.linux.endswith(
            "/v1.12.6/tofu_1.12.6_linux_amd64.zip"
        )
        assert installer.cli_name == "tofu"


def test_opentofu_version_is_pinned_for_state_migration() -> None:
    artifacts = OSArtifacts()
    dependency = artifacts.REQUIRED_TOOLS["tofu"]
    artifacts.get_version = Mock(return_value="1.13.0")

    assert dependency.minimum_version == "1.12.6"
    assert dependency.maximum_version == "1.12.6"
    assert not artifacts.is_supported_version(dependency, Path("/tmp/tofu"))


def test_dapr_upgrade_path_uses_latest_patch_for_each_minor() -> None:
    dapr = DaprWrapper(Mock(), Mock())
    dapr.version = Mock(return_value=["1.9.4"])
    dapr._target_version = Mock(return_value="1.18.3")

    assert dapr.upgrade_path() == [
        "1.9.6",
        "1.10.10",
        "1.11.6",
        "1.12.5",
        "1.13.6",
        "1.14.5",
        "1.15.14",
        "1.16.19",
        "1.17.13",
        "1.18.3",
    ]


def test_cert_manager_rejects_newer_version() -> None:
    cert_manager = CertManagerWrapper(Mock(), Mock())
    cert_manager.version = Mock(return_value="1.22.0")

    with pytest.raises(RuntimeError, match="newer than supported"):
        cert_manager.needs_upgrade()


def test_dapr_version_reads_version_column_from_status() -> None:
    artifacts = Mock(spec=OSArtifacts)
    artifacts.dapr = "dapr"
    kubectl = Mock(cluster_name="cluster")
    kubectl.context.return_value = nullcontext()
    dapr = DaprWrapper(artifacts, kubectl)
    status = (
        "NAME NAMESPACE HEALTH STATUS REPLICAS VERSION AGE CREATED\n"
        "dapr-operator dapr-system Healthy Running 1 1.13.3 1h 2026-01-01\n"
    )

    with patch(
        "vibe_core.cli.wrappers.execute_cmd", return_value=status
    ):
        assert dapr.version() == ["1.13.3"]


def test_dapr_version_is_empty_before_install() -> None:
    artifacts = Mock(spec=OSArtifacts)
    artifacts.dapr = "dapr"
    kubectl = Mock(cluster_name="cluster")
    kubectl.context.return_value = nullcontext()
    kubectl.get_or_none.return_value = None
    dapr = DaprWrapper(artifacts, kubectl)

    with patch(
        "vibe_core.cli.wrappers.execute_cmd", return_value="[]"
    ) as execute:
        assert dapr.version() == []

    assert "list" in execute.call_args.args[0]


def test_dapr_version_rejects_missing_operator_for_existing_release() -> None:
    artifacts = Mock(spec=OSArtifacts)
    artifacts.dapr = "dapr"
    artifacts.helm = "helm"
    kubectl = Mock(cluster_name="cluster")
    kubectl.context.return_value = nullcontext()
    kubectl.get_or_none.return_value = None
    dapr = DaprWrapper(artifacts, kubectl)

    with (
        patch(
            "vibe_core.cli.wrappers.execute_cmd",
            return_value='[{"name":"dapr","namespace":"dapr-system"}]',
        ),
        pytest.raises(RuntimeError, match="operator is missing"),
    ):
        dapr.version()


def test_dapr_crd_upgrade_fails_closed() -> None:
    dapr = DaprWrapper(Mock(), Mock())

    with (
        patch(
            "vibe_core.cli.wrappers.requests.head",
            return_value=Mock(status_code=404),
        ),
        pytest.raises(RuntimeError, match="Required Dapr CRD"),
    ):
        dapr.upgrade_crds("1.18.3")

    with (
        patch(
            "vibe_core.cli.wrappers.requests.head",
            side_effect=requests.ConnectionError("offline"),
        ),
        pytest.raises(requests.ConnectionError),
    ):
        dapr.upgrade_crds("1.18.3")


def test_dapr_reconciliation_recreates_placement_statefulset() -> None:
    kubectl = Mock(cluster_name="cluster")
    kubectl.context.return_value = nullcontext()
    kubectl.get_or_none.return_value = {"spec": {"replicas": 1}}
    dapr = DaprWrapper(Mock(), kubectl)

    assert dapr.prepare_for_terraform_reconciliation()

    kubectl.delete.assert_called_once_with(
        "statefulset",
        dapr.PLACEMENT_STATEFULSET,
        ignore_not_found=True,
        namespace="dapr-system",
        wait=False,
    )
    kubectl.wait_for_delete.assert_called_once_with(
        "statefulset",
        dapr.PLACEMENT_STATEFULSET,
        timeout_s=600,
        namespace="dapr-system",
    )


def test_dapr_reconciliation_keeps_converged_placement() -> None:
    kubectl = Mock(cluster_name="cluster")
    kubectl.context.return_value = nullcontext()
    kubectl.get_or_none.return_value = {"spec": {"replicas": 3}}
    dapr = DaprWrapper(Mock(), kubectl)

    assert not dapr.prepare_for_terraform_reconciliation()
    kubectl.delete.assert_not_called()


def test_dapr_upgrade_applies_crds_before_each_runtime() -> None:
    kubectl = Mock(cluster_name="cluster")
    kubectl.context.return_value = nullcontext()
    kubectl.get_or_none.return_value = {"metadata": {"name": "dapr-scheduler-server"}}
    dapr = DaprWrapper(Mock(), kubectl)
    dapr.upgrade_path = Mock(return_value=["1.17.13", "1.18.3"])
    events = []
    dapr.upgrade_crds = Mock(
        side_effect=lambda version: events.append(("crds", version)) or True
    )
    dapr.upgrade = Mock(
        side_effect=lambda version: events.append(("runtime", version))
    )

    assert dapr.upgrade_sequentially()
    assert events == [
        ("crds", "1.17.13"),
        ("runtime", "1.17.13"),
        ("crds", "1.18.3"),
        ("runtime", "1.18.3"),
    ]
    kubectl.restart.assert_called_once_with(
        "statefulset",
        name=dapr.SCHEDULER_STATEFULSET,
        namespace="dapr-system",
    )
    kubectl.rollout_status.assert_called_once_with(
        "statefulset",
        dapr.SCHEDULER_STATEFULSET,
        namespace="dapr-system",
    )
    kubectl.context.assert_called_once_with("cluster")


def test_dapr_rejects_newer_version() -> None:
    dapr = DaprWrapper(Mock(), Mock())
    dapr.version = Mock(return_value=["1.19.0"])
    dapr._target_version = Mock(return_value="1.18.3")

    with pytest.raises(RuntimeError, match="newer than"):
        dapr.needs_upgrade()
    with pytest.raises(RuntimeError, match="newer than"):
        dapr.upgrade_path()


def test_cert_manager_upgrade_path_uses_latest_patch_for_each_minor() -> None:
    cert_manager = CertManagerWrapper(Mock(), Mock())
    cert_manager.version = Mock(return_value="1.12.2")

    assert cert_manager.upgrade_path() == [
        "1.12.17",
        "1.13.6",
        "1.14.7",
        "1.15.5",
        "1.16.5",
        "1.17.4",
        "1.18.6",
        "1.19.6",
        "1.20.3",
        "1.21.1",
    ]


def test_cert_manager_release_discovery_handles_missing_namespace() -> None:
    artifacts = Mock(spec=OSArtifacts)
    artifacts.helm = "helm"
    kubectl = Mock(cluster_name="cluster")
    kubectl.context.return_value = nullcontext()
    cert_manager = CertManagerWrapper(artifacts, kubectl)

    with patch(
        "vibe_core.cli.wrappers.execute_cmd",
        return_value=(
            '[{"name":"cert-manager","namespace":"kube-system",'
            '"app_version":"v1.21.1"}]'
        ),
    ):
        assert cert_manager._release("cert-manager") is None
        assert cert_manager._release("kube-system") is not None


def test_cert_manager_removes_legacy_namespace_release() -> None:
    artifacts = Mock(spec=OSArtifacts)
    artifacts.helm = "helm"
    artifacts.kubectl = "kubectl"
    kubectl = Mock(cluster_name="cluster")
    kubectl.context.return_value = nullcontext()
    cert_manager = CertManagerWrapper(artifacts, kubectl)
    cert_manager._releases = Mock(
        return_value=[
            {
                "app_version": "v1.21.1",
                "namespace": cert_manager.LEGACY_NAMESPACE,
            }
        ]
    )

    crds = json.dumps(
        {
            "items": [
                {
                    "metadata": {
                        "name": "certificates.cert-manager.io",
                        "annotations": {
                            "meta.helm.sh/release-name": "cert-manager",
                            "meta.helm.sh/release-namespace": "kube-system",
                        },
                    }
                }
            ]
        }
    )
    with patch(
        "vibe_core.cli.wrappers.execute_cmd",
        side_effect=[crds, "", ""],
    ) as execute:
        assert cert_manager.prepare_for_terraform_reconciliation()

    assert execute.call_args_list[1].args[0] == [
        "helm",
        "uninstall",
        "cert-manager",
        "--namespace",
        "kube-system",
    ]
    assert execute.call_args_list[2].args[0] == [
        artifacts.kubectl,
        "annotate",
        "customresourcedefinitions",
        "--selector",
        "app.kubernetes.io/instance=cert-manager",
        "meta.helm.sh/release-name=cert-manager",
        "meta.helm.sh/release-namespace=cert-manager",
        "--overwrite",
    ]


def test_cert_manager_adopts_orphaned_crds() -> None:
    artifacts = Mock(spec=OSArtifacts)
    artifacts.helm = "helm"
    artifacts.kubectl = "kubectl"
    kubectl = Mock(cluster_name="cluster")
    kubectl.context.return_value = nullcontext()
    cert_manager = CertManagerWrapper(artifacts, kubectl)
    cert_manager._releases = Mock(return_value=[])

    with patch(
        "vibe_core.cli.wrappers.execute_cmd",
        side_effect=['{"items":[{"metadata":{"annotations":{}}}]}', ""],
    ) as execute:
        assert not cert_manager.prepare_for_terraform_reconciliation()

    assert len(execute.call_args_list) == 2
    assert "annotate" in execute.call_args_list[1].args[0]


def test_cert_manager_rejects_release_in_unmanaged_namespace() -> None:
    artifacts = Mock(spec=OSArtifacts)
    artifacts.helm = "helm"
    kubectl = Mock(cluster_name="cluster")
    kubectl.context.return_value = nullcontext()
    cert_manager = CertManagerWrapper(artifacts, kubectl)

    with patch(
        "vibe_core.cli.wrappers.execute_cmd",
        return_value='[{"name":"cert-manager","namespace":"other"}]',
    ) as execute:
        with pytest.raises(RuntimeError, match="outside the supported namespaces"):
            cert_manager.prepare_for_terraform_reconciliation()

    execute.assert_called_once()


def test_cert_manager_rejects_releases_in_both_supported_namespaces() -> None:
    artifacts = Mock(spec=OSArtifacts)
    artifacts.helm = "helm"
    kubectl = Mock(cluster_name="cluster")
    kubectl.context.return_value = nullcontext()
    cert_manager = CertManagerWrapper(artifacts, kubectl)

    with patch(
        "vibe_core.cli.wrappers.execute_cmd",
        return_value=json.dumps(
            [
                {"name": "cert-manager", "namespace": "cert-manager"},
                {"name": "cert-manager", "namespace": "kube-system"},
            ]
        ),
    ) as execute:
        with pytest.raises(RuntimeError, match="both supported namespaces"):
            cert_manager.prepare_for_terraform_reconciliation()

    execute.assert_called_once()


def test_cert_manager_rejects_unmanaged_crd_ownership_before_changes() -> None:
    artifacts = Mock(spec=OSArtifacts)
    artifacts.kubectl = "kubectl"
    kubectl = Mock(cluster_name="cluster")
    kubectl.context.return_value = nullcontext()
    cert_manager = CertManagerWrapper(artifacts, kubectl)
    cert_manager._releases = Mock(
        return_value=[{"namespace": cert_manager.LEGACY_NAMESPACE}]
    )
    crds = json.dumps(
        {
            "items": [
                {
                    "metadata": {
                        "name": "certificates.cert-manager.io",
                        "annotations": {
                            "meta.helm.sh/release-name": "cert-manager",
                            "meta.helm.sh/release-namespace": "other",
                        },
                    }
                }
            ]
        }
    )

    with patch(
        "vibe_core.cli.wrappers.execute_cmd",
        return_value=crds,
    ) as execute:
        with pytest.raises(RuntimeError, match="Refusing to take ownership"):
            cert_manager.prepare_for_terraform_reconciliation()

    execute.assert_called_once()


def test_remote_cluster_name_fits_key_vault_limit() -> None:
    assert remote.check_cluster_name_length("a" * 15)
    assert not remote.check_cluster_name_length("a" * 16)


def test_kubectl_uses_baseline_and_target_cluster_skew() -> None:
    assert OSArtifacts.REQUIRED_TOOLS["kubectl"].minimum_version == "1.27.0"
    assert KubectlInstaller.KUBECTL_RELEASE_URL.startswith("https://dl.k8s.io/")
    assert OSArtifacts.kubectl_is_compatible("1.34.9", "1.35.1")
    assert OSArtifacts.kubectl_is_compatible("1.36.0", "1.35.1")
    assert not OSArtifacts.kubectl_is_compatible("1.33.9", "1.35.1")


def test_kubectl_normalizes_linux_arm64_architecture() -> None:
    with patch("vibe_core.cli.osartifacts.platform.machine", return_value="aarch64"):
        assert KubectlInstaller(Path("/tmp")).arch == "arm64"


def test_incompatible_kubectl_installs_target_server_minor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("FARMVIBES_AI_CONFIG_DIR", str(tmp_path))
    artifacts = OSArtifacts()
    artifacts._kubectl_path = "/tmp/old-kubectl"
    artifacts.get_version = Mock(side_effect=["1.30.0", "1.35.9"])

    with patch("vibe_core.cli.osartifacts.KubectlInstaller") as installer:
        installer.return_value.cli_name = "kubectl"
        artifacts.ensure_compatible_kubectl("1.35.1")

    installer.assert_called_once_with(tmp_path, "1.35")
    installer.return_value.install.assert_called_once_with()
    assert artifacts.kubectl == str(tmp_path / "kubectl")


def test_get_kubernetes_version_queries_requested_cluster() -> None:
    artifacts = Mock(spec=OSArtifacts)
    artifacts.az = "az"
    az = AzureCliWrapper(artifacts, "cluster", "resource-group")

    with patch(
        "vibe_core.cli.wrappers.execute_cmd", return_value="1.35.1\n"
    ) as execute:
        assert az.get_kubernetes_version() == "1.35.1"

    assert execute.call_args.args[0] == [
        "az",
        "aks",
        "show",
        "--name",
        "cluster",
        "--resource-group",
        "resource-group",
        "--query",
        "kubernetesVersion",
        "-o",
        "tsv",
    ]


def test_refresh_aks_credentials_uses_private_admin_kubeconfig(
    tmp_path: Path,
) -> None:
    artifacts = Mock(spec=OSArtifacts)
    artifacts.az = "az"
    kubeconfig = tmp_path / "kubeconfig"
    artifacts.config_file.return_value = str(kubeconfig)
    az = AzureCliWrapper(artifacts, "cluster", "resource-group")
    az.refresh_az_creds = Mock()
    az.cluster_exists = Mock(return_value=True)
    commands: List[List[str]] = []

    def execute(command: List[str], *args: Any, **kwargs: Any) -> str:
        commands.append(command)
        kubeconfig.touch()
        return ""

    with patch("vibe_core.cli.wrappers.execute_cmd", side_effect=execute):
        az.refresh_aks_credentials()

    assert commands == [
        [
            "az",
            "aks",
            "get-credentials",
            "--admin",
            "--name",
            "cluster",
            "--resource-group",
            "resource-group",
            "--file",
            str(kubeconfig),
            "--overwrite-existing",
        ]
    ]
    assert kubeconfig.stat().st_mode & 0o777 == 0o600


def test_secure_path_uses_owner_only_windows_acl(tmp_path: Path) -> None:
    path = tmp_path / "token"
    path.touch()

    with (
        patch("vibe_core.cli.osartifacts.platform.system", return_value="Windows"),
        patch("vibe_core.cli.osartifacts.subprocess.run") as run,
    ):
        secure_path(path, 0o600)

    command = run.call_args.args[0]
    assert command[:4] == [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
    ]
    assert command[-1] == str(path)
    assert "SetAccessRuleProtection($true, $false)" in command[-2]
    assert run.call_args.kwargs == {
        "check": True,
        "capture_output": True,
        "text": True,
    }


def test_secure_path_uses_windows_acl_for_wsl_mount() -> None:
    path = Path("/mnt/c/shared/token")

    with (
        patch("vibe_core.cli.osartifacts.platform.system", return_value="Linux"),
        patch("vibe_core.cli.osartifacts.in_wsl", return_value=True),
        patch(
            "vibe_core.cli.osartifacts.subprocess.check_output",
            return_value="C:\\shared\\token\n",
        ) as check_output,
        patch("vibe_core.cli.osartifacts.subprocess.run") as run,
    ):
        secure_path(path, 0o600)

    check_output.assert_called_once_with(
        ["wslpath", "-w", str(path)],
        text=True,
    )
    assert run.call_args.args[0][-1] == "C:\\shared\\token"


def test_github_api_headers_use_actions_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert github_api_headers() == {}

    monkeypatch.setenv("GITHUB_TOKEN", "actions-token")
    assert github_api_headers() == {"Authorization": "Bearer actions-token"}


def test_config_directory_is_owner_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    path = tmp_path / "config"
    path.mkdir(mode=0o777)
    monkeypatch.setenv("FARMVIBES_AI_CONFIG_DIR", str(path))

    assert OSArtifacts().config_dir == path
    assert path.stat().st_mode & 0o777 == 0o700


def test_legacy_service_charts_are_destroyed_before_rabbitmq_pvc_reset() -> None:
    artifacts = Mock(spec=OSArtifacts)
    terraform = TerraformWrapper(artifacts)
    terraform.state_resources = Mock(
        return_value=["helm_release.redis", "helm_release.rabbitmq"]
    )
    terraform.destroy = Mock()
    on_destroy = Mock()

    with patch("vibe_core.cli.wrappers.KubectlWrapper") as kubectl:
        terraform.destroy_legacy_service_charts(
            "/terraform",
            "state.tfstate",
            {"namespace": "default"},
            "cluster",
            "cluster-admin",
            on_destroy,
        )

    on_destroy.assert_called_once_with()
    terraform.destroy.assert_called_once_with(
        "/terraform",
        "state.tfstate",
        {"namespace": "default"},
        targets=["helm_release.redis", "helm_release.rabbitmq"],
    )
    kubectl.return_value.delete.assert_called_once_with(
        "pvc", "data-rabbitmq-0", ignore_not_found=True
    )


def test_unowned_rabbitmq_pvc_is_not_deleted() -> None:
    artifacts = Mock(spec=OSArtifacts)
    terraform = TerraformWrapper(artifacts)
    terraform.state_resources = Mock(return_value=[])

    with patch("vibe_core.cli.wrappers.KubectlWrapper") as kubectl:
        kubectl.return_value.context.return_value = nullcontext()
        kubectl.return_value.get_or_none.return_value = {
            "metadata": {"labels": {"app": "unrelated"}}
        }
        terraform.destroy_legacy_service_charts(
            "/terraform",
            "state.tfstate",
            {"namespace": "default"},
            "cluster",
            "cluster-admin",
        )

    kubectl.return_value.delete.assert_not_called()


def test_residual_helm_rabbitmq_pvc_is_deleted() -> None:
    artifacts = Mock(spec=OSArtifacts)
    terraform = TerraformWrapper(artifacts)
    terraform.state_resources = Mock(return_value=[])

    with patch("vibe_core.cli.wrappers.KubectlWrapper") as kubectl:
        kubectl.return_value.context.return_value = nullcontext()
        kubectl.return_value.get_or_none.return_value = {
            "metadata": {
                "labels": {
                    "app.kubernetes.io/managed-by": "Helm",
                    "app.kubernetes.io/instance": "rabbitmq",
                }
            }
        }
        terraform.destroy_legacy_service_charts(
            "/terraform",
            "state.tfstate",
            {"namespace": "default"},
            "cluster",
            "cluster-admin",
        )

    kubectl.return_value.delete.assert_called_once_with(
        "pvc", "data-rabbitmq-0", ignore_not_found=True
    )


def test_kubectl_ignore_not_found_delete_accepts_empty_output() -> None:
    artifacts = Mock(spec=OSArtifacts)
    artifacts.kubectl = "kubectl"
    kubectl = KubectlWrapper(artifacts, "cluster", config_context="cluster-admin")

    with patch("vibe_core.cli.wrappers.execute_cmd") as execute:
        kubectl.delete("pvc", "missing", ignore_not_found=True)

    assert execute.call_args.args[0] == [
        "kubectl",
        "delete",
        "pvc",
        "missing",
        "--ignore-not-found=true",
    ]
    assert execute.call_args.kwargs["check_empty_result"] is False
