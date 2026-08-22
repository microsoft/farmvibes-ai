# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import base64
import json
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from vibe_core.cli.constants import RABBITMQ_IMAGE, REDIS_IMAGE
from vibe_core.cli.local import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    LOCAL_CONFIG,
    REDIS_MIGRATION_STATE,
    REGISTRY_PORT,
    migration_complete_state_path,
    migration_state_path,
)
from vibe_core.cli.osartifacts import InstallType, OSArtifacts
from vibe_core.cli.wrappers import K3dWrapper

CHART_PATH = Path(__file__).parent / "legacy_local_services"
MANAGED_BY_PATH = ".metadata.labels.app\\.kubernetes\\.io/managed-by"
IMAGE_PATH = ".spec.template.spec.containers[0].image"
MIGRATION_AGENTS = 1
MIGRATION_PORT = DEFAULT_PORT + 10
MIGRATION_WORKERS = 2
MIGRATION_REGISTRY = "mcr.microsoft.com/farmai"
MIGRATION_IMAGE_PREFIX = "terravibes/"
MIGRATION_IMAGE_TAG = "12072727496"
LEGACY_REDIS_IMAGE = "docker.io/library/redis:7.4.10-alpine"
MIGRATION_REDIS_SOURCE = "redis:7.4.10-bookworm"
MIGRATION_RABBITMQ_SOURCE = "rabbitmq:4.3.5-management"
REGISTRY_USERNAME = "migration"
REGISTRY_PASSWORD = "migration-secret"


def run(command: Sequence[str], capture_output: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        capture_output=capture_output,
        text=True,
    )


def output(command: Sequence[str]) -> str:
    return run(command, capture_output=True).stdout.strip()


def resource_field(
    kubectl_context: Sequence[str], kind: str, name: str, path: str
) -> str:
    return output(
        list(kubectl_context)
        + ["get", kind, name, "-o", f"jsonpath={{{path}}}"]
    )


def secret_password(
    kubectl_context: Sequence[str], name: str, key: str
) -> str:
    encoded = resource_field(kubectl_context, "secret", name, f".data.{key}")
    return base64.b64decode(encoded).decode()


def docker_config(encoded: str) -> Dict[str, Any]:
    return json.loads(base64.b64decode(encoded))


def redis_command(
    kubectl_context: Sequence[str], password: str, command: str, *args: str
) -> str:
    result = subprocess.run(
        list(kubectl_context)
        + ["exec", "-i", "redis-master-0", "--", "redis-cli", "--no-auth-warning"],
        check=True,
        capture_output=True,
        input=f"AUTH {password}\n{' '.join((command,) + args)}\n",
        text=True,
    )
    return result.stdout.strip().splitlines()[-1]


def server_created(k3d: K3dWrapper) -> str:
    return next(
        (
            node["created"]
            for node in k3d.info().get("nodes", [])
            if node.get("role") == "server"
        ),
        "",
    )


def wait_until(
    predicate: Callable[[], bool],
    description: str,
    process: Optional[subprocess.Popen[str]] = None,
    timeout_s: int = 600,
):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        if process is not None and process.poll() is not None:
            raise AssertionError(
                f"Update exited with {process.returncode} before {description}"
            )
        time.sleep(0.1)
    raise AssertionError(f"Timed out waiting for {description}")


def read_migration_state(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def registry_container(
    docker: str, cluster_name: str
) -> str:
    names = output(
        [
            docker,
            "ps",
            "-a",
            "--filter",
            f"label=k3d.cluster={cluster_name}",
            "--format",
            "{{.Names}}",
        ]
    ).splitlines()
    return next(name for name in names if "registry" in name)


def configure_authenticated_registry(
    artifacts: OSArtifacts,
    k3d: K3dWrapper,
):
    docker = artifacts.docker
    name = registry_container(docker, k3d.cluster_name)
    details = json.loads(output([docker, "inspect", name]))[0]
    labels = details["Config"].get("Labels", {})
    network = next(
        name
        for name in details["NetworkSettings"]["Networks"]
        if name == f"k3d-{k3d.cluster_name}"
    )
    registry_port = k3d.get_cluster_config()["registry_port"]
    auth_path = artifacts.private_config_dir / "test-registry-auth"
    data_path = artifacts.private_config_dir / "test-registry-data"
    auth_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    auth_path.chmod(0o700)
    data_path.mkdir(parents=True, exist_ok=True)
    htpasswd = output(
        [
            docker,
            "run",
            "--rm",
            "--entrypoint",
            "htpasswd",
            "httpd:2-alpine",
            "-Bbn",
            REGISTRY_USERNAME,
            REGISTRY_PASSWORD,
        ]
    )
    (auth_path / "htpasswd").write_text(htpasswd + "\n")
    (auth_path / "htpasswd").chmod(0o600)

    run([docker, "rm", "--force", name])
    command = [
        docker,
        "run",
        "--detach",
        "--restart",
        "unless-stopped",
        "--name",
        name,
        "--network",
        network,
        "--network-alias",
        name,
        "--publish",
        f"{DEFAULT_HOST}:{registry_port}:5000",
        "--volume",
        f"{auth_path}:/auth:ro",
        "--volume",
        f"{data_path}:/var/lib/registry",
        "--env",
        "REGISTRY_AUTH=htpasswd",
        "--env",
        "REGISTRY_AUTH_HTPASSWD_REALM=FarmVibes migration",
        "--env",
        "REGISTRY_AUTH_HTPASSWD_PATH=/auth/htpasswd",
    ]
    for key, value in labels.items():
        command.extend(("--label", f"{key}={value}"))
    command.append("registry:2")
    run(command)

    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        ready = subprocess.run(
            [
                "curl",
                "--fail",
                "--silent",
                "--user",
                f"{REGISTRY_USERNAME}:{REGISTRY_PASSWORD}",
                f"http://{DEFAULT_HOST}:{registry_port}/v2/",
            ],
            check=False,
        )
        if ready.returncode == 0:
            return name
        time.sleep(1)
    raise AssertionError("Authenticated registry did not become ready")


def push_private_service_images(
    artifacts: OSArtifacts,
    registry_port: int,
) -> None:
    docker = artifacts.docker
    for host in ("localhost", DEFAULT_HOST):
        login = subprocess.run(
            [
                docker,
                "login",
                f"{host}:{registry_port}",
                "--username",
                REGISTRY_USERNAME,
                "--password-stdin",
            ],
            input=REGISTRY_PASSWORD,
            text=True,
            check=True,
        )
        assert login.returncode == 0
    for source, repository in (
        (MIGRATION_REDIS_SOURCE, "farmvibes/redis:7.4.10-bookworm"),
        (
            MIGRATION_RABBITMQ_SOURCE,
            "farmvibes/rabbitmq:4.3.5-management",
        ),
    ):
        destination = f"{DEFAULT_HOST}:{registry_port}/{repository}"
        run([docker, "pull", source])
        run([docker, "tag", source, destination])
        run([docker, "push", destination])


def replace_pull_secret(
    kubectl_context: Sequence[str],
    registry: str,
    password: str,
) -> str:
    run(
        list(kubectl_context)
        + ["delete", "secret", "acrtoken", "--ignore-not-found"]
    )
    run(
        list(kubectl_context)
        + [
            "create",
            "secret",
            "docker-registry",
            "acrtoken",
            f"--docker-server={registry}",
            f"--docker-username={REGISTRY_USERNAME}",
            f"--docker-password={password}",
            "--docker-email=migration@example.invalid",
        ]
    )
    return resource_field(
        kubectl_context, "secret", "acrtoken", ".data.\\.dockerconfigjson"
    )


def replace_registry_after_cluster_create(
    artifacts: OSArtifacts,
    k3d: K3dWrapper,
    old_target_uid: str,
) -> Tuple[threading.Thread, List[BaseException]]:
    errors: List[BaseException] = []
    state_path = Path(
        migration_state_path(artifacts, k3d.cluster_name)
    )

    def replace():
        try:
            wait_until(
                lambda: (
                    read_migration_state(state_path).get("phase")
                    == "cluster_created"
                    and read_migration_state(state_path).get(
                        "target_cluster_uid"
                    )
                    not in (None, old_target_uid)
                ),
                "replacement k3d registry",
            )
            configure_authenticated_registry(
                artifacts, k3d
            )
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=replace, daemon=True)
    thread.start()
    return thread, errors


def join_registry_replacement(
    replacement: Tuple[threading.Thread, List[BaseException]]
):
    thread, errors = replacement
    thread.join(180)
    assert not thread.is_alive(), "Timed out replacing k3d registry"
    if errors:
        raise errors[0]


def test_chart_to_native_redis_migration_preserves_data():
    cluster_name = os.environ["FARMVIBES_AI_CLUSTER_NAME"]
    storage_path = Path(os.environ["FARMVIBES_AI_STORAGE_PATH"])
    run_id = os.environ.get("GITHUB_RUN_ID", "local")
    run_attempt = os.environ.get("GITHUB_RUN_ATTEMPT", "1")
    commit = os.environ.get("GITHUB_SHA", "local")[:12]
    redis_key = f"farmvibes:migration:{run_id}:{run_attempt}"
    redis_value = f"survived:{commit}:{run_id}:{run_attempt}"

    artifacts = OSArtifacts()
    artifacts.check_dependencies(InstallType.LOCAL)
    k3d = K3dWrapper(artifacts, cluster_name)
    assert not k3d.cluster_exists(), f"Refusing to replace existing cluster {cluster_name}"

    storage_path.mkdir(parents=True, exist_ok=True)
    assert k3d.create(
        servers=1,
        agents=MIGRATION_AGENTS,
        storage_path=str(storage_path),
        registry_port=REGISTRY_PORT,
        farmvibes_port=MIGRATION_PORT,
        host=DEFAULT_HOST,
    )
    legacy_config = k3d.get_cluster_config()
    assert legacy_config == {
        "servers": 1,
        "agents": MIGRATION_AGENTS,
        "port": MIGRATION_PORT,
        "host": DEFAULT_HOST,
        "registry_port": REGISTRY_PORT,
    }
    old_server_created = server_created(k3d)
    assert old_server_created

    context = f"k3d-{cluster_name}"
    kubectl = artifacts.kubectl
    helm = artifacts.helm
    # The released Bitnami tags are gone; this chart keeps their Helm labels,
    # names, Secret keys, data path, and PVC names with maintained images.
    run(
        [
            helm,
            "upgrade",
            "--install",
            "legacy-local-services",
            str(CHART_PATH),
            "--kube-context",
            context,
            "--namespace",
            "default",
            "--wait",
            "--timeout",
            "5m",
            "--set-string",
            f"redis.image={LEGACY_REDIS_IMAGE}",
            "--set-string",
            f"rabbitmq.image={RABBITMQ_IMAGE}",
        ]
    )

    kubectl_context = [kubectl, "--context", context]
    run(kubectl_context + ["rollout", "status", "statefulset/redis-master", "--timeout=5m"])
    assert resource_field(
        kubectl_context, "statefulset", "redis-master", MANAGED_BY_PATH
    ) == "Helm"
    assert resource_field(
        kubectl_context, "statefulset", "rabbitmq", MANAGED_BY_PATH
    ) == "Helm"

    old_pvc_uid = resource_field(
        kubectl_context, "pvc", "redis-data-redis-master-0", ".metadata.uid"
    )
    old_password = secret_password(kubectl_context, "redis", "redis-password")
    assert redis_command(kubectl_context, old_password, "SET", redis_key, redis_value) == "OK"
    registry_name = configure_authenticated_registry(
        artifacts, k3d
    )
    push_private_service_images(artifacts, REGISTRY_PORT)
    private_registry = f"{registry_name}:5000"
    migration_redis_image = (
        f"{private_registry}/farmvibes/redis:7.4.10-bookworm"
    )
    migration_rabbitmq_image = (
        f"{private_registry}/farmvibes/rabbitmq:4.3.5-management"
    )
    run(
        kubectl_context
        + [
            "create",
            "secret",
            "docker-registry",
            "acrtoken",
            f"--docker-server={private_registry}",
            "--docker-username=robot",
            "--docker-password=test-token",
            "--docker-email=robot@invalid",
        ]
    )
    pull_secret = resource_field(
        kubectl_context, "secret", "acrtoken", ".data.\\.dockerconfigjson"
    )

    farmvibes_ai = shutil.which("farmvibes-ai")
    assert farmvibes_ai is not None
    wrong_shared_storage = storage_path.parent / f"{cluster_name}-other-storage"
    (artifacts.config_dir / "storage").write_text(str(wrong_shared_storage))
    assert not (
        artifacts.config_dir / LOCAL_CONFIG.format(cluster_name=cluster_name)
    ).exists()
    update_without_config = [
        farmvibes_ai,
        "local",
        "update",
        "--auto-confirm",
        "--cluster-name",
        cluster_name,
    ]
    first_update_command = update_without_config + [
        "--registry",
        MIGRATION_REGISTRY,
        "--image-prefix",
        MIGRATION_IMAGE_PREFIX,
        "--image-tag",
        MIGRATION_IMAGE_TAG,
        "--worker-replicas",
        str(MIGRATION_WORKERS),
        "--log-level",
        "INFO",
        "--max-log-file-bytes",
        "123456",
        "--log-backup-count",
        "2",
        "--redis-image",
        migration_redis_image,
        "--rabbitmq-image",
        migration_rabbitmq_image,
    ]
    destroy_command = [
        farmvibes_ai,
        "local",
        "destroy",
        "--auto-confirm",
        "--cluster-name",
        cluster_name,
    ]
    private_state_path = Path(
        migration_state_path(artifacts, cluster_name)
    )
    completed_path = Path(
        migration_complete_state_path(artifacts, cluster_name)
    )
    bad_auth_update = subprocess.run(
        first_update_command,
        check=False,
        capture_output=True,
        text=True,
    )
    assert bad_auth_update.returncode != 0
    assert "authenticate" in (
        bad_auth_update.stdout + bad_auth_update.stderr
    ).lower()
    assert server_created(k3d) == old_server_created
    assert (
        resource_field(
            kubectl_context,
            "pvc",
            "redis-data-redis-master-0",
            ".metadata.uid",
        )
        == old_pvc_uid
    )
    assert redis_command(
        kubectl_context, old_password, "GET", redis_key
    ) == redis_value
    assert not private_state_path.exists()
    pull_secret = replace_pull_secret(
        kubectl_context, private_registry, REGISTRY_PASSWORD
    )

    first_update = subprocess.Popen(
        first_update_command,
        text=True,
        start_new_session=True,
    )
    try:
        wait_until(
            lambda: read_migration_state(private_state_path).get("phase")
            == "cluster_created",
            "the post-k3d-create migration checkpoint",
            first_update,
        )
        os.killpg(first_update.pid, signal.SIGTERM)
        assert first_update.wait(timeout=120) != 0
    finally:
        if first_update.poll() is None:
            os.killpg(first_update.pid, signal.SIGTERM)
            first_update.wait(timeout=30)

    migration_state = read_migration_state(private_state_path)
    assert migration_state["phase"] == "cluster_created"
    assert migration_state["config"]["storage_path"] == str(storage_path)
    assert not (wrong_shared_storage / "data").exists()
    assert private_state_path.stat().st_mode & 0o777 == 0o600
    assert docker_config(migration_state["docker_config"]) == docker_config(
        pull_secret
    )
    assert storage_path.resolve() not in private_state_path.resolve().parents
    assert not (
        storage_path / "data" / REDIS_MIGRATION_STATE
    ).exists()
    assert not any(
        path.name.endswith(REDIS_MIGRATION_STATE)
        for path in storage_path.rglob("*")
    )
    assert all(
        pull_secret.encode() not in path.read_bytes()
        for path in (storage_path / "data").rglob("*")
        if path.is_file()
    )
    backup_path = storage_path / "data" / migration_state["backup_file"]
    original_backup = backup_path.read_bytes()
    original_checksum = migration_state["backup_sha256"]
    assert old_server_created != server_created(k3d)

    run(destroy_command)
    assert not k3d.cluster_exists()
    assert backup_path.read_bytes() == original_backup
    assert read_migration_state(private_state_path)["backup_sha256"] == original_checksum

    backup_path.write_bytes(b"not the immutable redis dump\n")
    checksum_failure = subprocess.run(
        update_without_config,
        check=False,
        capture_output=True,
        text=True,
    )
    assert checksum_failure.returncode != 0
    assert "checksum mismatch" in (
        checksum_failure.stdout + checksum_failure.stderr
    ).lower()
    assert not k3d.cluster_exists()

    backup_path.write_bytes(original_backup)
    interrupted_target_uid = read_migration_state(
        private_state_path
    )["target_cluster_uid"]
    first_registry_replacement = replace_registry_after_cluster_create(
        artifacts, k3d, interrupted_target_uid
    )
    restore_update = subprocess.Popen(
        update_without_config,
        text=True,
        start_new_session=True,
    )
    try:
        wait_until(
            lambda: read_migration_state(private_state_path).get("phase")
            == "restoring",
            "the native Redis restore phase",
            restore_update,
        )
        join_registry_replacement(first_registry_replacement)
        completed_path.mkdir()
        assert restore_update.wait(timeout=1200) != 0
    finally:
        if restore_update.poll() is None:
            os.killpg(restore_update.pid, signal.SIGTERM)
            restore_update.wait(timeout=30)

    restored_state = read_migration_state(private_state_path)
    assert restored_state["phase"] == "restored"
    assert restored_state["backup_sha256"] == original_checksum
    assert k3d.get_cluster_config() == legacy_config

    run(kubectl_context + ["rollout", "status", "statefulset/redis-master", "--timeout=5m"])
    first_restored_password = secret_password(
        kubectl_context, "redis", "redis-password"
    )
    assert redis_command(
        kubectl_context, first_restored_password, "GET", redis_key
    ) == redis_value
    first_restored_server = server_created(k3d)
    first_restored_target_uid = restored_state["target_cluster_uid"]

    assert k3d.delete()
    assert not k3d.cluster_exists()
    assert read_migration_state(private_state_path)["phase"] == "restored"
    second_registry_replacement = replace_registry_after_cluster_create(
        artifacts, k3d, first_restored_target_uid
    )
    replaced_restore = subprocess.Popen(
        update_without_config,
        text=True,
        start_new_session=True,
    )
    try:
        wait_until(
            lambda: (
                read_migration_state(private_state_path).get("phase")
                == "restoring"
            ),
            "the replacement-cluster Redis restore phase",
            replaced_restore,
        )
        join_registry_replacement(second_registry_replacement)
        assert replaced_restore.wait(timeout=1200) != 0
    finally:
        if replaced_restore.poll() is None:
            os.killpg(replaced_restore.pid, signal.SIGTERM)
            replaced_restore.wait(timeout=30)

    replaced_state = read_migration_state(private_state_path)
    assert replaced_state["phase"] == "restored"
    assert replaced_state["target_cluster_uid"] != first_restored_target_uid
    assert server_created(k3d) != first_restored_server
    assert k3d.get_cluster_config() == legacy_config
    run(kubectl_context + ["rollout", "status", "statefulset/redis-master", "--timeout=5m"])
    second_restored_password = secret_password(
        kubectl_context, "redis", "redis-password"
    )
    assert redis_command(
        kubectl_context, second_restored_password, "GET", redis_key
    ) == redis_value
    assert resource_field(
        kubectl_context, "statefulset", "redis-master", IMAGE_PATH
    ) == migration_redis_image
    assert resource_field(
        kubectl_context, "statefulset", "rabbitmq", IMAGE_PATH
    ) == migration_rabbitmq_image
    assert docker_config(
        resource_field(
            kubectl_context,
            "secret",
            "acrtoken",
            ".data.\\.dockerconfigjson",
        )
    ) == docker_config(pull_secret)
    newer_key = f"{redis_key}:newer"
    newer_value = f"newer:{commit}:{run_id}:{run_attempt}"
    assert (
        redis_command(
            kubectl_context,
            second_restored_password,
            "SET",
            newer_key,
            newer_value,
        )
        == "OK"
    )

    completed_path.rmdir()
    run(destroy_command)
    normal_backup_path = storage_path / "data" / "redis-dump.rdb"
    assert not k3d.cluster_exists()
    assert not private_state_path.exists()
    assert backup_path.read_bytes() == original_backup
    assert normal_backup_path.stat().st_size > 0

    setup_command = [
        farmvibes_ai,
        "local",
        "setup",
        "--auto-confirm",
        "--cluster-name",
        cluster_name,
        "--servers",
        "1",
        "--agents",
        str(MIGRATION_AGENTS),
        "--storage-path",
        str(storage_path),
        "--registry",
        MIGRATION_REGISTRY,
        "--image-prefix",
        MIGRATION_IMAGE_PREFIX,
        "--image-tag",
        MIGRATION_IMAGE_TAG,
        "--worker-replicas",
        str(MIGRATION_WORKERS),
        "--log-level",
        "INFO",
        "--max-log-file-bytes",
        "123456",
        "--log-backup-count",
        "2",
        "--disable-telemetry",
        "--port",
        str(MIGRATION_PORT),
        "--host",
        DEFAULT_HOST,
        "--registry-port",
        str(REGISTRY_PORT),
        "--redis-image",
        REDIS_IMAGE,
        "--rabbitmq-image",
        RABBITMQ_IMAGE,
    ]
    run(setup_command)
    assert k3d.get_cluster_config() == legacy_config

    run(kubectl_context + ["rollout", "status", "statefulset/redis-master", "--timeout=5m"])
    run(kubectl_context + ["rollout", "status", "statefulset/rabbitmq", "--timeout=5m"])
    assert normal_backup_path.stat().st_size > 0

    new_pvc_uid = resource_field(
        kubectl_context, "pvc", "redis-data-redis-master-0", ".metadata.uid"
    )
    assert new_pvc_uid != old_pvc_uid
    assert resource_field(
        kubectl_context, "statefulset", "redis-master", IMAGE_PATH
    ) == REDIS_IMAGE
    assert resource_field(
        kubectl_context, "statefulset", "rabbitmq", IMAGE_PATH
    ) == RABBITMQ_IMAGE
    assert resource_field(
        kubectl_context, "statefulset", "redis-master", MANAGED_BY_PATH
    ) != "Helm"
    assert (
        output(
            kubectl_context
            + ["get", "pod", "redisvolpod", "--ignore-not-found", "-o", "name"]
        )
        == ""
    )
    assert "legacy-local-services" not in output(
        [helm, "list", "--kube-context", context, "--namespace", "default", "--short"]
    )
    assert resource_field(
        kubectl_context, "deployment", "terravibes-worker", ".spec.replicas"
    ) == str(MIGRATION_WORKERS)
    assert resource_field(
        kubectl_context, "deployment", "terravibes-worker", IMAGE_PATH
    ) == (
        f"{MIGRATION_REGISTRY}/{MIGRATION_IMAGE_PREFIX}"
        f"worker:{MIGRATION_IMAGE_TAG}"
    )
    assert (
        output(
            kubectl_context
            + [
                "get",
                "secret",
                "acrtoken",
                "--ignore-not-found",
                "-o",
                "name",
            ]
        )
        == ""
    )

    new_password = secret_password(
        kubectl_context, "redis", "redis-password"
    )
    assert new_password != old_password
    restored_value = redis_command(kubectl_context, new_password, "GET", redis_key)
    assert restored_value == redis_value
    assert redis_command(kubectl_context, new_password, "GET", newer_key) == newer_value
    saved_config = json.loads(
        (
            artifacts.config_dir
            / LOCAL_CONFIG.format(cluster_name=cluster_name)
        ).read_text()
    )
    assert saved_config == {
        "servers": 1,
        "agents": MIGRATION_AGENTS,
        "storage_path": str(storage_path),
        "registry": MIGRATION_REGISTRY,
        "log_level": "INFO",
        "max_log_file_bytes": 123456,
        "log_backup_count": 2,
        "image_tag": MIGRATION_IMAGE_TAG,
        "image_prefix": MIGRATION_IMAGE_PREFIX,
        "worker_replicas": MIGRATION_WORKERS,
        "enable_telemetry": False,
        "port": MIGRATION_PORT,
        "host": DEFAULT_HOST,
        "registry_port": REGISTRY_PORT,
        "redis_image": REDIS_IMAGE,
        "rabbitmq_image": RABBITMQ_IMAGE,
    }
    print(
        "Redis migration survived bad registry auth, interruption, checksum "
        "rejection, target replacement, and a restored-phase destroy: "
        f"original={redis_key}:{restored_value} newer={newer_key}:{newer_value} "
        f"old_pvc={old_pvc_uid} new_pvc={new_pvc_uid}"
    )
