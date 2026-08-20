# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

import vibe_core.cli.local as local
from vibe_core.cli.constants import RABBITMQ_IMAGE, REDIS_IMAGE
from vibe_core.cli.osartifacts import OSArtifacts
from vibe_core.cli.parsers import LocalCliParser
from vibe_core.cli.wrappers import KubectlWrapper, TerraformWrapper


def test_local_parser_service_image_defaults_and_overrides():
    parser = LocalCliParser("local")

    defaults = parser.parse(["setup"])
    assert defaults.redis_image == REDIS_IMAGE
    assert defaults.rabbitmq_image == RABBITMQ_IMAGE

    overrides = parser.parse(
        [
            "update",
            "--redis-image",
            "registry.example/redis:test",
            "--rabbitmq-image",
            "registry.example/rabbitmq:test",
        ]
    )
    assert overrides.redis_image == "registry.example/redis:test"
    assert overrides.rabbitmq_image == "registry.example/rabbitmq:test"


def test_local_dispatch_forwards_service_images(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    class K3d:
        def __init__(self, os_artifacts: OSArtifacts, cluster_name: str):
            self.os_artifacts = os_artifacts
            self.cluster_name = cluster_name

        @staticmethod
        def cluster_exists() -> bool:
            return False

    captured: Dict[str, Any] = {}

    def capture_setup(*args: Any, **kwargs: Any) -> bool:
        captured.update(kwargs)
        return True

    monkeypatch.setenv("FARMVIBES_AI_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setattr(OSArtifacts, "check_dependencies", lambda *args, **kwargs: None)
    monkeypatch.setattr(local, "K3dWrapper", K3d)
    monkeypatch.setattr(local, "setup", capture_setup)

    args = LocalCliParser("local").parse(
        [
            "setup",
            "--storage-path",
            str(tmp_path),
            "--redis-image",
            "registry.example/redis:test",
            "--rabbitmq-image",
            "registry.example/rabbitmq:test",
        ]
    )
    assert local.dispatch(args) is True
    assert captured["redis_image"] == "registry.example/redis:test"
    assert captured["rabbitmq_image"] == "registry.example/rabbitmq:test"


def test_terraform_wrapper_propagates_service_images(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    artifacts = OSArtifacts()
    artifacts._local_terraform_path = str(tmp_path)
    monkeypatch.setattr(
        artifacts,
        "get_terraform_file",
        lambda file_name, *args: str(tmp_path / file_name),
    )
    terraform = TerraformWrapper.__new__(TerraformWrapper)
    terraform.os_artifacts = artifacts
    terraform.environment = "public"
    terraform.az = None
    captured: Dict[str, Any] = {}

    monkeypatch.setattr(terraform, "init", lambda *args, **kwargs: None)
    monkeypatch.setattr(terraform, "getuid", lambda: 1000)
    monkeypatch.setattr(terraform, "getgid", lambda: 1000)
    monkeypatch.setattr(
        terraform,
        "apply",
        lambda directory, state, variables, **kwargs: captured.update(variables),
    )
    monkeypatch.setattr(terraform, "get_output", lambda *args, **kwargs: {})

    terraform.ensure_local_cluster(
        cluster_name="test",
        registry="registry.example",
        log_level="INFO",
        max_log_file_bytes=None,
        log_backup_count=None,
        image_tag="test",
        image_prefix="",
        data_path=str(tmp_path),
        worker_replicas=1,
        config_context="k3d-test",
        enable_telemetry=False,
        redis_image="registry.example/redis:test",
        rabbitmq_image="registry.example/rabbitmq:test",
    )

    assert captured["redis_image"] == "registry.example/redis:test"
    assert captured["rabbitmq_image"] == "registry.example/rabbitmq:test"
    assert "redis_image_tag" not in captured
    assert "rabbitmq_image_tag" not in captured


def test_legacy_chart_services_require_migration():
    class Kubectl(KubectlWrapper):
        def context(self, cluster_name: str = ""):
            return nullcontext()

        def get(self, kind: str, name: str, jsonpath: Optional[str] = None):
            assert kind == "statefulset"
            return {
                "metadata": {
                    "labels": {
                        "app.kubernetes.io/managed-by": (
                            "Helm" if name == "rabbitmq" else "Terraform"
                        )
                    }
                }
            }

    assert local.needs_service_migration(Kubectl.__new__(Kubectl)) is True


def test_native_terraform_preserves_service_contracts():
    terraform_dir = (
        Path(__file__).parents[1] / "vibe_core" / "terraform" / "local" / "modules" / "kubernetes"
    )
    redis = (terraform_dir / "redis.tf").read_text()
    rabbitmq = (terraform_dir / "rabbitmq.tf").read_text()
    dapr = (terraform_dir / "dapr.tf").read_text()
    resources = redis + rabbitmq

    assert "helm_release" not in resources
    assert "bitnami" not in resources.lower()
    assert 'name      = "redis-master"' in redis
    assert 'name      = "redis"' in redis
    assert "redis-password" in redis
    assert 'mount_path = "/data"' in redis
    assert 'name      = "rabbitmq"' in rabbitmq
    assert "rabbitmq-password" in rabbitmq
    assert 'value = "user"' in rabbitmq
    assert 'mount_path = "/var/lib/rabbitmq"' in rabbitmq
    assert "value: redis-master:6379" in dapr
    assert "key: redis-password" in dapr
    assert "key: rabbitmq-password" in dapr
