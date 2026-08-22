# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import base64
import codecs
import errno
import hashlib
import json
import os
import secrets
import shutil
import tempfile
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from vibe_core.cli.constants import (
    AZURE_CR_DOMAIN,
    DEFAULT_IMAGE_PREFIX,
    DEFAULT_IMAGE_TAG,
    DEFAULT_REGISTRY_PATH,
    FARMVIBES_AI_LOG_LEVEL,
    LOCAL_SERVICE_URL_PATH_FILE,
    ONNX_SUBDIR,
    RABBITMQ_IMAGE,
    REDIS_IMAGE,
)
from vibe_core.cli.helper import verify_to_proceed
from vibe_core.cli.logging import log
from vibe_core.cli.osartifacts import InstallType, OSArtifacts
from vibe_core.cli.wrappers import (
    AzureCliWrapper,
    DaprWrapper,
    DockerWrapper,
    ImagePullAuthenticationError,
    K3dWrapper,
    KubectlWrapper,
    TerraformWrapper,
)

DEFAULT_STORAGE_PATH = os.environ.get(
    "FARMVIBES_AI_STORAGE_PATH",
    os.path.join(os.path.expanduser("~"), ".cache", "farmvibes-ai"),
)
DATA_SUFFIX = "data"
REDIS_DUMP = "redis-dump.rdb"
REDIS_MIGRATION_STATE = "redis-migration.json"
REDIS_MIGRATION_COMPLETE_STATE = "redis-migration-complete.json"
REDIS_MIGRATION_STATE_VERSION = 3
PREFLIGHT_PULL_SECRET = "farmvibes-image-preflight-auth"
ACR_ACCESS_TOKEN_USERNAME = "00000000-0000-0000-0000-000000000000"
LOCAL_CONFIG = "local-config-{cluster_name}.json"
MIGRATION_PHASES = (
    "prepared",
    "backed_up",
    "cluster_created",
    "provisioned",
    "restoring",
    "restored",
)
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 31108
REGISTRY_PORT = 5000
OLD_DEFAULT_CLUSTER_NAME = "farmvibes-ai"
MANAGED_BY_LABEL = "app.kubernetes.io/managed-by"
REDIS_BACKUP_COMMAND = (
    'if [ -n "$2" ]; then '
    'REDISCLI_AUTH="$1" redis-cli SET "$2" "$3" >/dev/null || exit 1; '
    "fi; "
    'REDISCLI_AUTH="$1" redis-cli CONFIG SET appendonly no >/dev/null && '
    'REDISCLI_AUTH="$1" redis-cli SAVE >/dev/null'
)
TOPOLOGY_FIELDS = {
    "agents",
    "host",
    "port",
    "registry_port",
    "servers",
    "storage_path",
}


def find_redis_master(kubectl: KubectlWrapper) -> Tuple[str, ...]:
    pods = kubectl.list_pods()
    redis_master_pod = ""
    for pod in pods["items"]:
        if pod["metadata"]["name"].startswith("redis-master"):
            redis_master_pod = pod
            break
    if not redis_master_pod:
        log("Unable to find redis master pod", level="warning")
        with kubectl.context():
            redis_statefulset = kubectl.get_or_none(
                "statefulset", "redis-master"
            )
        if redis_statefulset is None:
            return ("", "", "")
        kind = redis_statefulset["kind"]
        name = redis_statefulset["metadata"]["name"]
    else:
        owner_references = redis_master_pod["metadata"].get("ownerReferences", [])
        redis_master_pod = redis_master_pod["metadata"]["name"]
        kind = owner_references[0]["kind"]
        name = owner_references[0]["name"]
    return (
        redis_master_pod,
        name,
        kind,
    )


def needs_service_migration(kubectl: KubectlWrapper) -> bool:
    with kubectl.context():
        for name in ("redis-master", "rabbitmq"):
            stateful_set = kubectl.get_or_none("statefulset", name)
            if stateful_set is None:
                continue
            labels = stateful_set.get("metadata", {}).get("labels", {})
            if labels.get(MANAGED_BY_LABEL) == "Helm":
                return True
    return False


def inspect_effective_config(
    k3d: K3dWrapper, kubectl: KubectlWrapper
) -> Dict[str, Any]:
    config = k3d.get_cluster_config()
    effective: Dict[str, Any] = {
        key: config[key]
        for key in ("servers", "agents", "port", "host", "registry_port")
    }
    with kubectl.context():
        worker = kubectl.get_or_none("deployment", "terravibes-worker")
        redis = kubectl.get_or_none("statefulset", "redis-master")
        rabbitmq = kubectl.get_or_none("statefulset", "rabbitmq")
        telemetry = kubectl.get_or_none("deployment", "otel-collector")

    if worker:
        spec = worker["spec"]["template"]["spec"]["containers"][0]
        repository, image_tag = spec["image"].rsplit(":", 1)
        registry, image_path = repository.split("/", 1)
        effective.update(
            {
                "registry": registry,
                "image_prefix": (
                    image_path[: -len("worker")]
                    if image_path.endswith("worker")
                    else image_path
                ),
                "image_tag": image_tag,
                "worker_replicas": worker["spec"]["replicas"],
                "enable_telemetry": telemetry is not None,
            }
        )
        arguments = spec.get("args", [])
        for option, key, conversion in (
            ("worker.impl.loglevel=", "log_level", str),
            ("worker.impl.max_log_file_bytes=", "max_log_file_bytes", int),
            ("worker.impl.log_backup_count=", "log_backup_count", int),
        ):
            value = next(
                (argument[len(option) :] for argument in arguments if argument.startswith(option)),
                None,
            )
            if value is not None:
                effective[key] = conversion(value)

    for resource, key in (
        (redis, "redis_image"),
        (rabbitmq, "rabbitmq_image"),
    ):
        if (
            resource
            and resource.get("metadata", {}).get("labels", {}).get(MANAGED_BY_LABEL)
            != "Helm"
        ):
            effective[key] = resource["spec"]["template"]["spec"]["containers"][0][
                "image"
            ]
    return effective


def get_pull_secret(kubectl: KubectlWrapper) -> Optional[str]:
    secret = kubectl.get_secret_or_none("acrtoken")
    if secret is None:
        return None
    if secret.get("type") != "kubernetes.io/dockerconfigjson":
        raise RuntimeError("The acrtoken Secret is not a Docker registry credential")
    docker_config = secret.get("data", {}).get(".dockerconfigjson")
    if not docker_config:
        raise RuntimeError("The acrtoken Secret has no Docker registry credential")
    decode_docker_config(docker_config)
    return docker_config


def normalize_registry(registry: str) -> str:
    registry = registry.strip().lower()
    for prefix in ("https://", "http://"):
        if registry.startswith(prefix):
            registry = registry[len(prefix) :]
    registry = registry.split("/", 1)[0]
    if registry in ("index.docker.io", "registry-1.docker.io"):
        return "docker.io"
    return registry


def image_registry(image: str) -> str:
    if "/" not in image:
        return "docker.io"
    first = image.split("/", 1)[0]
    if "." not in first and ":" not in first and first != "localhost":
        return "docker.io"
    return normalize_registry(first)


def selected_images(config: Dict[str, Any]) -> List[str]:
    registry = config["registry"].rstrip("/")
    prefix = config["image_prefix"]
    tag = config["image_tag"]
    return [
        f"{registry}/{prefix}api-orchestrator:{tag}",
        f"{registry}/{prefix}worker:{tag}",
        f"{registry}/{prefix}cache:{tag}",
        config["redis_image"],
        config["rabbitmq_image"],
    ]


def decode_docker_config(docker_config: str) -> Dict[str, Any]:
    try:
        decoded = base64.b64decode(docker_config, validate=True)
        config = json.loads(decoded)
    except (ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("Invalid Docker registry credential") from error
    if not isinstance(config, dict) or not isinstance(config.get("auths"), dict):
        raise RuntimeError("Invalid Docker registry credential")
    return config


def encode_docker_config(config: Dict[str, Any]) -> str:
    return base64.b64encode(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).decode()


def docker_auth_username(
    docker_config: Optional[str], registry: str
) -> Optional[str]:
    if not docker_config:
        return None
    normalized_registry = normalize_registry(registry)
    for auth_registry, entry in decode_docker_config(
        docker_config
    )["auths"].items():
        if normalize_registry(auth_registry) != normalized_registry:
            continue
        if not isinstance(entry, dict) or not isinstance(
            entry.get("auth"), str
        ):
            raise ValueError(
                f"Invalid Docker credentials for registry {registry}"
            )
        try:
            decoded = base64.b64decode(
                entry["auth"], validate=True
            ).decode()
            username, _ = decoded.split(":", 1)
        except (ValueError, UnicodeDecodeError) as error:
            raise ValueError(
                f"Invalid Docker credentials for registry {registry}"
            ) from error
        return username
    return None


def add_docker_auth(
    docker_config: Optional[str], registry: str, username: str, password: str
) -> str:
    config = (
        decode_docker_config(docker_config)
        if docker_config
        else {"auths": {}}
    )
    host = normalize_registry(registry)
    config["auths"][host] = {
        "auth": base64.b64encode(f"{username}:{password}".encode()).decode()
    }
    return encode_docker_config(config)


def remove_docker_auth(
    docker_config: Optional[str], registries: Set[str]
) -> Optional[str]:
    if not docker_config:
        return None
    config = decode_docker_config(docker_config)
    config["auths"] = {
        key: value
        for key, value in config["auths"].items()
        if normalize_registry(key) not in registries
    }
    return encode_docker_config(config) if config["auths"] else None


def retain_docker_auth(
    docker_config: Optional[str], registries: Set[str]
) -> Optional[str]:
    if not docker_config:
        return None
    config = decode_docker_config(docker_config)
    config["auths"] = {
        key: value
        for key, value in config["auths"].items()
        if normalize_registry(key) in registries
    }
    return encode_docker_config(config) if config["auths"] else None


def merge_docker_auth(
    original: Optional[str], replacement: Optional[str]
) -> Optional[str]:
    if not replacement:
        return original
    config = (
        decode_docker_config(original)
        if original
        else {"auths": {}}
    )
    config["auths"].update(
        decode_docker_config(replacement)["auths"]
    )
    return encode_docker_config(config)


def docker_auth_registries(docker_config: Optional[str]) -> Set[str]:
    if not docker_config:
        return set()
    return {
        normalize_registry(registry)
        for registry in decode_docker_config(docker_config)["auths"]
    }


def refresh_registry_auth(
    os_artifacts: OSArtifacts,
    docker_config: Optional[str],
    acr_registries: List[str],
) -> Tuple[Optional[str], Optional[AzureCliWrapper]]:
    az = None
    combined = docker_config
    for registry in acr_registries:
        if az is None:
            os_artifacts.check_dependencies(InstallType.ALL)
            az = AzureCliWrapper(os_artifacts, "")
        token = az.request_registry_token(registry)
        if not token:
            raise RuntimeError(
                f"Unable to refresh credentials for private registry {registry}"
            )
        combined = add_docker_auth(
            combined,
            registry,
            ACR_ACCESS_TOKEN_USERNAME,
            token,
        )
    return combined, az


def preflight_selected_images(
    kubectl: KubectlWrapper,
    config: Dict[str, Any],
    docker_config: Optional[str],
    acr_registries: List[str],
    pull_secret_name: str = "acrtoken",
):
    authenticated = docker_auth_registries(docker_config) | set(acr_registries)
    images = list(dict.fromkeys(selected_images(config)))
    images.sort(
        key=lambda image: image_registry(image) not in authenticated
    )
    for image in images:
        registry = image_registry(image)
        kubectl.preflight_image_pull(
            image,
            registry in authenticated,
            pull_secret_name,
        )


def prepare_registry_auth(
    kubectl: KubectlWrapper,
    os_artifacts: OSArtifacts,
    config: Dict[str, Any],
    username: Optional[str],
    password: Optional[str],
    docker_config: Optional[str] = None,
    acr_registries: Optional[List[str]] = None,
    preflight: bool = True,
    use_existing: bool = True,
    retain_selected_only: bool = False,
    preserve_existing_secret: bool = False,
) -> Tuple[Optional[str], List[str], Optional[AzureCliWrapper]]:
    if username and not password:
        raise RuntimeError("A registry username requires a registry password")
    if docker_config is None and use_existing:
        docker_config = get_pull_secret(kubectl)

    configured_registry = normalize_registry(config["registry"])
    if password:
        docker_config = add_docker_auth(
            docker_config,
            configured_registry,
            username or ACR_ACCESS_TOKEN_USERNAME,
            password,
        )

    selected_registries = {
        image_registry(image) for image in selected_images(config)
    }
    selected_acr = {
        registry
        for registry in selected_registries
        if registry.endswith(AZURE_CR_DOMAIN)
    }
    dynamic_acr = set(acr_registries or []) & selected_acr
    for registry in selected_acr:
        auth_username = docker_auth_username(docker_config, registry)
        if auth_username == ACR_ACCESS_TOKEN_USERNAME:
            dynamic_acr.add(registry)
        elif auth_username is None:
            dynamic_acr.add(registry)
        else:
            dynamic_acr.discard(registry)

    refresh_acr = {
        registry
        for registry in dynamic_acr
        if registry not in docker_auth_registries(docker_config)
    }
    if not preflight:
        refresh_acr.update(
            registry
            for registry in dynamic_acr
            if not (password and registry == configured_registry)
            and docker_auth_username(docker_config, registry)
            == ACR_ACCESS_TOKEN_USERNAME
        )
    combined, az = refresh_registry_auth(
        os_artifacts, docker_config, sorted(refresh_acr)
    )
    if preflight:
        refreshed_acr: Set[str] = set(refresh_acr)
        while True:
            preflight_secret = (
                f"{PREFLIGHT_PULL_SECRET}-{secrets.token_hex(6)}"
            )
            if combined:
                kubectl.apply_docker_config_secret(
                    preflight_secret, combined
                )
            try:
                preflight_selected_images(
                    kubectl,
                    config,
                    combined,
                    [],
                    preflight_secret,
                )
                break
            except ImagePullAuthenticationError as error:
                registry = image_registry(error.image)
                if (
                    registry not in selected_acr
                    or docker_auth_username(combined, registry)
                    != ACR_ACCESS_TOKEN_USERNAME
                    or registry in refreshed_acr
                ):
                    raise
                combined, refreshed_az = refresh_registry_auth(
                    os_artifacts, combined, [registry]
                )
                az = az or refreshed_az
                refreshed_acr.add(registry)
            finally:
                try:
                    if combined:
                        kubectl.delete_secret(preflight_secret)
                except Exception as error:
                    log(
                        "Unable to remove the registry preflight Secret: "
                        f"{error}",
                        level="warning",
                    )
    docker_config = remove_docker_auth(combined, dynamic_acr)
    if retain_selected_only:
        docker_config = retain_docker_auth(
            docker_config, selected_registries
        )
    if combined and not preserve_existing_secret:
        kubectl.apply_docker_config_secret("acrtoken", combined)
    return docker_config, sorted(dynamic_acr), az


def new_redis_migration_state(
    cluster_name: str,
    context_name: str,
    source_cluster_uid: str,
    config: Dict[str, Any],
    docker_config: Optional[str],
    acr_registries: List[str],
) -> Dict[str, Any]:
    migration_id = secrets.token_hex(12)
    return {
        "version": REDIS_MIGRATION_STATE_VERSION,
        "cluster_name": cluster_name,
        "context_name": context_name,
        "source_cluster_uid": source_cluster_uid,
        "target_cluster_uid": None,
        "phase": "prepared",
        "migration_id": migration_id,
        "backup_file": f"redis-migration-{migration_id}.rdb",
        "marker_key": f"__farmvibes_migration__:{migration_id}",
        "marker_value": secrets.token_hex(16),
        "config": config,
        "docker_config": docker_config,
        "acr_registries": sorted(acr_registries),
    }


def redis_migration_marker_matches(
    kubectl: KubectlWrapper, state: Dict[str, Any]
) -> bool:
    pod, _, _ = find_redis_master(kubectl)
    if not pod:
        return False
    with kubectl.context():
        result = kubectl.exec(
            pod,
            [
                "sh",
                "-c",
                'redis-cli --raw GET "$1"',
                "sh",
                state["marker_key"],
            ],
            censor_command=True,
        )
    return result.strip() == state["marker_value"]


def clear_redis_migration_marker(
    kubectl: KubectlWrapper, state: Dict[str, Any]
):
    pod, _, _ = find_redis_master(kubectl)
    if not pod:
        raise RuntimeError("Unable to find Redis while completing migration")
    with kubectl.context():
        kubectl.exec(
            pod,
            [
                "sh",
                "-c",
                'redis-cli DEL "$1" >/dev/null',
                "sh",
                state["marker_key"],
            ],
            capture_output=False,
            censor_command=True,
        )


def fsync_parent(path: str):
    if not hasattr(os, "O_DIRECTORY"):
        return
    directory = os.open(
        os.path.dirname(path) or ".",
        os.O_RDONLY | os.O_DIRECTORY,
    )
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def atomic_write_json(path: str, contents: Dict[str, Any]):
    parent = os.path.dirname(path) or "."
    descriptor, temporary_path = tempfile.mkstemp(
        dir=parent, prefix=f".{os.path.basename(path)}."
    )
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(contents, output, sort_keys=True)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        os.chmod(path, 0o600)
        fsync_parent(path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def local_config_path(os_artifacts: OSArtifacts, cluster_name: str) -> str:
    return str(
        os_artifacts.config_dir / LOCAL_CONFIG.format(cluster_name=cluster_name)
    )


def load_local_config(
    os_artifacts: OSArtifacts, cluster_name: str
) -> Dict[str, Any]:
    path = local_config_path(os_artifacts, cluster_name)
    if not os.path.exists(path):
        return {}
    with open(path) as config_file:
        config = json.load(config_file)
    return config if isinstance(config, dict) else {}


def save_local_config(
    os_artifacts: OSArtifacts, cluster_name: str, config: Dict[str, Any]
):
    atomic_write_json(local_config_path(os_artifacts, cluster_name), config)


def private_cluster_path(
    os_artifacts: OSArtifacts, cluster_name: str, filename: str
) -> str:
    cluster_id = hashlib.sha256(cluster_name.encode()).hexdigest()[:24]
    return str(os_artifacts.private_config_dir / f"{cluster_id}-{filename}")


def migration_state_path(
    os_artifacts: OSArtifacts, cluster_name: str
) -> str:
    return private_cluster_path(os_artifacts, cluster_name, REDIS_MIGRATION_STATE)


def migration_complete_state_path(
    os_artifacts: OSArtifacts, cluster_name: str
) -> str:
    return private_cluster_path(
        os_artifacts, cluster_name, REDIS_MIGRATION_COMPLETE_STATE
    )


def migrate_shared_redis_state(
    os_artifacts: OSArtifacts,
    data_path: str,
    cluster_name: str,
    context_name: str,
) -> Optional[Dict[str, Any]]:
    shared_path = os.path.join(data_path, REDIS_MIGRATION_STATE)
    if not os.path.exists(shared_path):
        return None
    with open(shared_path) as state_file:
        state = json.load(state_file)
    if state.get("version") != 2:
        raise ValueError("Unsupported Redis migration state version")
    state["version"] = REDIS_MIGRATION_STATE_VERSION
    state["context_name"] = context_name
    state["source_cluster_uid"] = None
    state["target_cluster_uid"] = None
    state["docker_config"] = state.pop("pull_secret", None)
    state["acr_registries"] = []
    atomic_write_json(migration_state_path(os_artifacts, cluster_name), state)
    os.remove(shared_path)
    return state


def load_redis_migration_state(
    os_artifacts: OSArtifacts,
    cluster_name: str,
    context_name: str,
    data_path: str = "",
) -> Optional[Dict[str, Any]]:
    completed_path = migration_complete_state_path(os_artifacts, cluster_name)
    if os.path.isfile(completed_path):
        os.remove(completed_path)
    if data_path:
        shared_completed_path = os.path.join(
            data_path, REDIS_MIGRATION_COMPLETE_STATE
        )
        if os.path.isfile(shared_completed_path):
            os.remove(shared_completed_path)

    state_path = migration_state_path(os_artifacts, cluster_name)
    shared_state_path = (
        os.path.join(data_path, REDIS_MIGRATION_STATE)
        if data_path
        else ""
    )
    if (
        os.path.exists(state_path)
        and shared_state_path
        and os.path.exists(shared_state_path)
    ):
        os.remove(shared_state_path)
    if not os.path.exists(state_path):
        state = (
            migrate_shared_redis_state(
                os_artifacts, data_path, cluster_name, context_name
            )
            if data_path
            else None
        )
        if state is None:
            return None
    else:
        with open(state_path) as state_file:
            state = json.load(state_file)
    if state.get("version") != REDIS_MIGRATION_STATE_VERSION:
        raise ValueError("Unsupported Redis migration state version")
    if state.get("cluster_name") != cluster_name:
        raise ValueError(
            f"Redis migration state belongs to cluster {state.get('cluster_name')}"
        )
    if state.get("context_name") != context_name:
        raise ValueError(
            "Redis migration state belongs to Kubernetes context "
            f"{state.get('context_name')}"
        )
    if state.get("phase") not in MIGRATION_PHASES:
        raise ValueError(f"Invalid Redis migration phase {state.get('phase')}")
    if not isinstance(state.get("config"), dict):
        raise ValueError("Redis migration state has no deployment configuration")
    missing_config = {
        "servers",
        "agents",
        "storage_path",
        "registry",
        "log_level",
        "max_log_file_bytes",
        "log_backup_count",
        "image_tag",
        "image_prefix",
        "worker_replicas",
        "enable_telemetry",
        "port",
        "host",
        "registry_port",
        "redis_image",
        "rabbitmq_image",
    } - state["config"].keys()
    if missing_config:
        raise ValueError(
            "Redis migration configuration is missing "
            + ", ".join(sorted(missing_config))
        )
    for key in (
        "migration_id",
        "marker_key",
        "marker_value",
    ):
        if not isinstance(state.get(key), str) or not state[key]:
            raise ValueError(f"Invalid Redis migration state field {key}")
    if not isinstance(state.get("backup_file"), str):
        raise ValueError("Redis migration state has no backup file")
    if os.path.basename(state["backup_file"]) != state["backup_file"]:
        raise ValueError("Invalid Redis migration backup file")
    if state["phase"] != "prepared":
        checksum = state.get("backup_sha256")
        if (
            not isinstance(checksum, str)
            or len(checksum) != 64
            or any(character not in "0123456789abcdef" for character in checksum)
        ):
            raise ValueError("Invalid Redis migration backup checksum")
    docker_config = state.get("docker_config")
    if docker_config is not None:
        if not isinstance(docker_config, str):
            raise ValueError("Invalid Redis migration pull Secret")
        decode_docker_config(docker_config)
    acr_registries = state.get("acr_registries")
    if not isinstance(acr_registries, list) or not all(
        isinstance(registry, str) and registry
        for registry in acr_registries
    ):
        raise ValueError("Invalid Redis migration ACR registry list")
    for key in ("source_cluster_uid", "target_cluster_uid"):
        if state.get(key) is not None and not isinstance(state[key], str):
            raise ValueError(f"Invalid Redis migration state field {key}")
    return state


def save_redis_migration_state(
    os_artifacts: OSArtifacts,
    state: Dict[str, Any],
    phase: Optional[str] = None,
) -> Dict[str, Any]:
    state = dict(state)
    if phase is not None:
        if phase not in MIGRATION_PHASES:
            raise ValueError(f"Invalid Redis migration phase {phase}")
        state["phase"] = phase
    atomic_write_json(
        migration_state_path(os_artifacts, state["cluster_name"]), state
    )
    return state


def clear_redis_migration_state(
    os_artifacts: OSArtifacts, state: Dict[str, Any]
):
    if state["phase"] != "restored":
        raise ValueError("Cannot complete an unverified Redis migration")
    completed_path = migration_complete_state_path(
        os_artifacts, state["cluster_name"]
    )
    os.replace(
        migration_state_path(os_artifacts, state["cluster_name"]),
        completed_path,
    )
    os.remove(completed_path)
    fsync_parent(completed_path)


def redis_migration_matches_current_cluster(
    kubectl: KubectlWrapper, state: Dict[str, Any]
) -> bool:
    return (
        state["phase"] == "restored"
        and kubectl.get_cluster_uid()
        == state.get("target_cluster_uid")
        and redis_migration_marker_matches(kubectl, state)
    )


def complete_redis_migration(
    kubectl: KubectlWrapper,
    os_artifacts: OSArtifacts,
    state: Dict[str, Any],
):
    if not redis_migration_matches_current_cluster(kubectl, state):
        raise RuntimeError(
            "Cannot complete Redis migration on a changed or unverified cluster"
        )
    clear_redis_migration_state(os_artifacts, state)
    try:
        clear_redis_migration_marker(kubectl, state)
    except Exception as error:
        log(
            f"Unable to remove completed Redis migration marker: {error}",
            level="warning",
        )


def _acquire_windows_lock(descriptor: int, lock_module: Any):
    while True:
        try:
            lock_module.locking(descriptor, lock_module.LK_LOCK, 1)
            return
        except OSError as error:
            if error.errno != errno.EACCES:
                raise


@contextmanager
def local_cluster_lock(
    os_artifacts: OSArtifacts, cluster_name: str
) -> Iterator[None]:
    path = private_cluster_path(os_artifacts, cluster_name, "operation.lock")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    os.chmod(path, 0o600)
    lock_module = __import__("msvcrt" if os.name == "nt" else "fcntl")
    locked = False
    try:
        if os.name == "nt":
            if os.path.getsize(path) == 0:
                os.write(descriptor, b" ")
            os.lseek(descriptor, 0, os.SEEK_SET)
            _acquire_windows_lock(descriptor, lock_module)
        else:
            lock_module.flock(descriptor, lock_module.LOCK_EX)
        locked = True
        os.ftruncate(descriptor, 0)
        os.write(descriptor, str(os.getpid()).encode())
        os.fsync(descriptor)
        yield
    finally:
        if locked:
            if os.name == "nt":
                os.lseek(descriptor, 0, os.SEEK_SET)
                lock_module.locking(descriptor, lock_module.LK_UNLCK, 1)
            else:
                lock_module.flock(descriptor, lock_module.LOCK_UN)
        os.close(descriptor)


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def migration_backup_path(data_path: str, state: Dict[str, Any]) -> str:
    return os.path.join(data_path, state["backup_file"])


def verify_migration_backup(data_path: str, state: Dict[str, Any]) -> str:
    backup_path = migration_backup_path(data_path, state)
    expected = state.get("backup_sha256")
    if not expected or not os.path.isfile(backup_path):
        raise RuntimeError("Redis migration backup is missing")
    actual = file_sha256(backup_path)
    if actual != expected:
        raise RuntimeError(
            f"Redis migration backup checksum mismatch: expected {expected}, got {actual}"
        )
    return backup_path


def backup_redis_data(
    kubectl: KubectlWrapper,
    data_path: str,
    dump_file: str = REDIS_DUMP,
    require_backup: bool = False,
    marker_key: str = "",
    marker_value: str = "",
) -> bool:
    log("Backing up redis data")

    try:
        with kubectl.context():
            result = kubectl.get_secret("redis", ".data.redis-password")
            redis_password = codecs.decode(result.encode(), "base64").decode()

            master_pod, redis_master, kind = find_redis_master(kubectl)
            if not master_pod:
                log("Making sure we have at least one redis master replica")
                kubectl.scale(kind, redis_master, 1)
                master_pod, redis_master, kind = find_redis_master(kubectl)

            log("Requesting redis data dump")
            if not master_pod:
                log(
                    "Unable to find redis master pod, " "unable to backup redis data",
                    level="error",
                )
                return False

            command = [
                "sh",
                "-c",
                REDIS_BACKUP_COMMAND,
                "sh",
                redis_password,
                marker_key,
                marker_value,
            ]
            kubectl.exec(master_pod, command, capture_output=False, censor_command=True)

            log("Saving redis data dump on the host machine")
            final_path = os.path.join(data_path, dump_file)
            kubectl.cp(f"{master_pod}:/data/dump.rdb", final_path)
            log(f"Redis data dump saved to {final_path}")
            return True
    except Exception:
        if require_backup:
            raise
        return False


def ensure_migration_backup(
    kubectl: KubectlWrapper,
    os_artifacts: OSArtifacts,
    data_path: str,
    state: Dict[str, Any],
) -> Dict[str, Any]:
    if state["phase"] != "prepared":
        verify_migration_backup(data_path, state)
        return state

    backup_path = migration_backup_path(data_path, state)
    if not os.path.exists(backup_path):
        temporary_name = f".{state['backup_file']}.tmp"
        temporary_path = os.path.join(data_path, temporary_name)
        try:
            if not backup_redis_data(
                kubectl,
                data_path,
                temporary_name,
                require_backup=True,
                marker_key=state["marker_key"],
                marker_value=state["marker_value"],
            ):
                raise RuntimeError("Unable to create Redis migration backup")
            if not os.path.isfile(temporary_path) or os.path.getsize(temporary_path) == 0:
                raise RuntimeError("Redis migration backup is empty")
            os.chmod(temporary_path, 0o600)
            with open(temporary_path, "rb") as backup:
                os.fsync(backup.fileno())
            os.replace(temporary_path, backup_path)
            fsync_parent(backup_path)
        finally:
            if os.path.exists(temporary_path):
                os.remove(temporary_path)

    state["backup_sha256"] = file_sha256(backup_path)
    return save_redis_migration_state(
        os_artifacts, state, "backed_up"
    )


def restore_redis_data(
    kubectl: KubectlWrapper,
    data_path: str,
    skip_confirmation: bool = False,
    redis_image: str = REDIS_IMAGE,
    dump_file: str = REDIS_DUMP,
) -> bool:
    redis_pod, redis_master, kind = find_redis_master(kubectl)
    backup_path = os.path.join(data_path, dump_file)

    if not redis_master:
        return False
    if not os.path.exists(backup_path):
        return False

    confirmation = skip_confirmation or verify_to_proceed(
        "I've found a state store backup file from a previous installation. "
        "Do you want to restore it?"
    )
    if not confirmation:
        log("Not restoring backup from user instructions.")
        return False

    restore_error: Optional[Exception] = None
    cleanup_error: Optional[Exception] = None
    scale_error: Optional[Exception] = None
    helper_attempted = False
    scaled_down = False
    with kubectl.context():
        try:
            kubectl.scale(kind, redis_master, 0)
            scaled_down = True
            if redis_pod:
                kubectl.wait_for_delete("pod", redis_pod, timeout_s=300)
            helper_attempted = True
            if not kubectl.create_redis_volume_pod(redis_image=redis_image):
                raise RuntimeError("Unable to create redis volume pod")
            kubectl.cp(backup_path, "redisvolpod:/mnt/dump.rdb")
        except Exception as error:
            restore_error = error

        if helper_attempted:
            try:
                kubectl.delete(
                    "pod", "redisvolpod", ignore_not_found=True
                )
            except Exception as error:
                cleanup_error = error

        if scaled_down:
            try:
                kubectl.scale(kind, redis_master, 1)
            except Exception as error:
                scale_error = error

        for message, error in (
            ("Unable to restore Redis data", restore_error),
            ("Unable to remove the Redis volume helper pod", cleanup_error),
            ("Unable to restart Redis after restore", scale_error),
        ):
            if error is not None:
                log(f"{message}: {error}", level="error")
        if restore_error or cleanup_error or scale_error:
            return False

        try:
            kubectl.rollout_status("statefulset", redis_master, timeout_s=600)
        except Exception as error:
            log(
                f"Redis did not become ready after loading the backup: {error}",
                level="error",
            )
            return False
    return True


def destroy_old_registry(
    os_artifacts: OSArtifacts, cluster_name: str = OLD_DEFAULT_CLUSTER_NAME
) -> bool:
    container_name = f"k3d-{cluster_name}-registry.localhost"
    docker = DockerWrapper(os_artifacts)
    try:
        result = docker.get(container_name)
        if not result:
            return True
        docker.rm(container_name)
        return True
    except Exception as e:
        log(f"Unable to remove old registry container: {e}", level="warning")
        return False


def destroy(
    k3d: K3dWrapper,
    data_path: str,
    skip_confirmation: bool = False,
    require_backup: bool = False,
) -> bool:
    log(f"Destroying local cluster with name {k3d.cluster_name}")
    if not k3d.cluster_exists():
        log("Cluster does not exist, nothing to destroy")
        return True
    kubectl = KubectlWrapper(k3d.os_artifacts, k3d.cluster_name)
    migration_state = load_redis_migration_state(
        k3d.os_artifacts,
        k3d.cluster_name,
        kubectl.context_name,
        data_path,
    )
    pending_migration = migration_state is not None
    restored_migration = (
        migration_state is not None
        and migration_state["phase"] == "restored"
    )
    if (
        pending_migration
        and not restored_migration
        and migration_state is not None
    ):
        verify_migration_backup(data_path, migration_state)
    if not skip_confirmation:
        confirmation = verify_to_proceed(
            "Do you want to destroy the local cluster? "
            "This will delete all the data in the cluster."
        )
        if not confirmation:
            log("Aborting destroy due to user confirmation")
            return True
    fresh_backup_required = False
    if restored_migration and migration_state is not None:
        complete_redis_migration(
            kubectl, k3d.os_artifacts, migration_state
        )
        migration_state = None
        pending_migration = False
        fresh_backup_required = True
    confirmation = (
        fresh_backup_required
        or pending_migration
        or skip_confirmation
        or verify_to_proceed(
            "Do you want to backup workflow state data before destroying the cluster?"
        )
    )
    if confirmation and not pending_migration:
        if not backup_redis_data(kubectl, data_path):
            if require_backup or fresh_backup_required:
                log("Unable to migrate without a Redis state backup.", level="error")
                return False
            if not skip_confirmation:
                confirmation = verify_to_proceed(
                    "Unable to backup redis data, do you want to continue?"
                )
                if not confirmation:
                    log("Aborting destroy due to user confirmation")
                    return True
    elif pending_migration:
        log("Keeping the verified Redis migration backup unchanged.")
    if not k3d.delete():
        log("Unable to delete cluster", level="warning")
    if k3d.cluster_exists():
        # So, we just deleted a cluster, right? Yeah. Sometimes k3d doesn't
        # delete the cluster properly. So, we try to delete it again.
        log("Cluster still exists, trying to delete again", level="warning")
        if not k3d.delete():
            log("Unable to delete cluster", level="warning")
            return False
        if k3d.cluster_exists():
            log(
                "Cluster still exists after trying to delete it twice, "
                "please delete it manually with "
                f"`{k3d.os_artifacts.k3d} delete --name {k3d.cluster_name}`",
                level="error",
            )
            return False
    # Do we have an old registry? If we do, delete it...
    if k3d.cluster_name == OLD_DEFAULT_CLUSTER_NAME:
        destroy_old_registry(k3d.os_artifacts)
    terraform = TerraformWrapper(k3d.os_artifacts, AzureCliWrapper(k3d.os_artifacts, ""))
    terraform.set_workspace("default")
    terraform.delete_workspace(f"farmvibes-k3d-{k3d.cluster_name}")
    log("Cluster deleted successfully")
    return True


def check_disk_space(storage_path: str, space_in_gb: int = 30) -> bool:
    log(f"Checking disk space in {storage_path}")
    if not os.path.exists(storage_path):
        log(f"Storage path {storage_path} does not exist", level="error")
        return False
    _, _, free_bytes = shutil.disk_usage(storage_path)
    free_bytes = free_bytes / 1_000_000_000
    if free_bytes < space_in_gb:
        log(
            f"Storage path {storage_path} has {free_bytes:.2f} GB of free space, "
            f"which is less than the recommended {space_in_gb} GB.\n"
            "This may cause the cluster to fail to start.\n"
            "You can free up space by deleting unused Docker images.",
            level="warning",
        )
        confirmation = verify_to_proceed("Would you like to continue with the setup?")
        return confirmation
    return True


def setup(
    k3d: K3dWrapper,
    servers: Optional[int] = None,
    agents: Optional[int] = None,
    storage_path: Optional[str] = None,
    registry: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    log_level: Optional[str] = None,
    max_log_file_bytes: Optional[int] = None,
    log_backup_count: Optional[int] = None,
    image_tag: Optional[str] = None,
    image_prefix: Optional[str] = None,
    data_path: str = "",
    worker_replicas: Optional[int] = None,
    enable_telemetry: Optional[bool] = None,
    port: Optional[int] = None,
    host: Optional[str] = None,
    is_update: bool = False,
    registry_port: Optional[int] = None,
    redis_image: Optional[str] = None,
    rabbitmq_image: Optional[str] = None,
) -> bool:
    log("Updating local cluster" if is_update else "Setting up local cluster")
    os.makedirs(data_path, exist_ok=True)

    kubectl = KubectlWrapper(k3d.os_artifacts, k3d.cluster_name)
    cluster_exists = k3d.cluster_exists()
    migration_state = load_redis_migration_state(
        k3d.os_artifacts,
        k3d.cluster_name,
        kubectl.context_name,
        data_path,
    )
    if migration_state is not None and not is_update:
        log(
            "A Redis migration is pending; resuming it before setup.",
            level="warning",
        )
        is_update = True

    defaults: Dict[str, Any] = {
        "servers": 1,
        "agents": 0,
        "storage_path": DEFAULT_STORAGE_PATH,
        "registry": DEFAULT_REGISTRY_PATH,
        "log_level": FARMVIBES_AI_LOG_LEVEL,
        "max_log_file_bytes": None,
        "log_backup_count": None,
        "image_tag": DEFAULT_IMAGE_TAG,
        "image_prefix": DEFAULT_IMAGE_PREFIX,
        "worker_replicas": 0,
        "enable_telemetry": False,
        "port": DEFAULT_PORT,
        "host": DEFAULT_HOST,
        "registry_port": REGISTRY_PORT,
        "redis_image": REDIS_IMAGE,
        "rabbitmq_image": RABBITMQ_IMAGE,
    }
    requested = {
        "servers": servers,
        "agents": agents,
        "storage_path": storage_path,
        "registry": registry,
        "log_level": log_level,
        "max_log_file_bytes": max_log_file_bytes,
        "log_backup_count": log_backup_count,
        "image_tag": image_tag,
        "image_prefix": image_prefix,
        "worker_replicas": worker_replicas,
        "enable_telemetry": enable_telemetry,
        "port": port,
        "host": host,
        "registry_port": registry_port,
        "redis_image": redis_image,
        "rabbitmq_image": rabbitmq_image,
    }
    current: Dict[str, Any] = {}
    if is_update and cluster_exists:
        current.update(inspect_effective_config(k3d, kubectl))
    if is_update:
        saved = load_local_config(k3d.os_artifacts, k3d.cluster_name)
        for key, value in saved.items():
            if key not in current or key in (
                "storage_path",
                "registry",
                "image_prefix",
            ):
                current[key] = value
    if migration_state is not None:
        current.update(migration_state["config"])
    effective = {**defaults, **current}
    effective.update(
        {key: value for key, value in requested.items() if value is not None}
    )

    storage_path = effective["storage_path"]
    registry = effective["registry"]
    log_level = effective["log_level"]
    max_log_file_bytes = effective["max_log_file_bytes"]
    log_backup_count = effective["log_backup_count"]
    image_tag = effective["image_tag"]
    image_prefix = effective["image_prefix"]
    worker_replicas = effective["worker_replicas"]
    enable_telemetry = effective["enable_telemetry"]
    redis_image = effective["redis_image"]
    rabbitmq_image = effective["rabbitmq_image"]
    if not isinstance(storage_path, str):
        raise ValueError("Invalid local configuration field storage_path")
    if not isinstance(registry, str):
        raise ValueError("Invalid local configuration field registry")
    if not isinstance(log_level, str):
        raise ValueError("Invalid local configuration field log_level")
    if not isinstance(image_tag, str):
        raise ValueError("Invalid local configuration field image_tag")
    if not isinstance(image_prefix, str):
        raise ValueError("Invalid local configuration field image_prefix")
    if type(worker_replicas) is not int:
        raise ValueError("Invalid local configuration field worker_replicas")
    if not isinstance(enable_telemetry, bool):
        raise ValueError("Invalid local configuration field enable_telemetry")
    if not isinstance(redis_image, str):
        raise ValueError("Invalid local configuration field redis_image")
    if not isinstance(rabbitmq_image, str):
        raise ValueError("Invalid local configuration field rabbitmq_image")

    legacy_services = (
        is_update and cluster_exists and needs_service_migration(kubectl)
    )
    if migration_state is not None:
        phase = migration_state["phase"]
        current_uid = kubectl.get_cluster_uid() if cluster_exists else None
        if not cluster_exists:
            if phase == "prepared":
                raise RuntimeError(
                    "Redis migration source disappeared before its backup completed"
                )
            migration_state["target_cluster_uid"] = None
            migration_state = save_redis_migration_state(
                k3d.os_artifacts, migration_state, "backed_up"
            )
        elif phase == "prepared":
            source_uid = migration_state.get("source_cluster_uid")
            if source_uid is not None and source_uid != current_uid:
                raise RuntimeError(
                    "Redis migration source cluster identity changed before backup"
                )
            migration_state["source_cluster_uid"] = current_uid
        elif phase == "backed_up" and legacy_services:
            source_uid = migration_state.get("source_cluster_uid")
            if source_uid is not None and source_uid != current_uid:
                raise RuntimeError(
                    "Redis migration source cluster identity changed after backup"
                )
            migration_state["source_cluster_uid"] = current_uid
        elif phase == "backed_up":
            migration_state["target_cluster_uid"] = current_uid
            migration_state = save_redis_migration_state(
                k3d.os_artifacts, migration_state, "cluster_created"
            )
        else:
            target_uid = migration_state.get("target_cluster_uid")
            if target_uid is not None and target_uid != current_uid:
                log(
                    "The migration target cluster was replaced; provisioning and "
                    "restoring the immutable backup again.",
                    level="warning",
                )
                verify_migration_backup(data_path, migration_state)
                migration_state["target_cluster_uid"] = current_uid
                migration_state = save_redis_migration_state(
                    k3d.os_artifacts, migration_state, "cluster_created"
                )
            elif target_uid is None:
                migration_state["target_cluster_uid"] = current_uid
                migration_state = save_redis_migration_state(
                    k3d.os_artifacts, migration_state
                )

        original_config = migration_state["config"]
        if (
            requested["storage_path"] is not None
            and requested["storage_path"] != original_config["storage_path"]
        ):
            raise RuntimeError(
                "Cannot change storage path while a Redis migration is pending"
            )
        target_exists = (
            cluster_exists
            and not legacy_services
            and MIGRATION_PHASES.index(migration_state["phase"])
            >= MIGRATION_PHASES.index("cluster_created")
        )
        changed_topology = sorted(
            key
            for key in TOPOLOGY_FIELDS
            if requested[key] is not None
            and requested[key] != original_config[key]
        )
        if target_exists and changed_topology:
            raise RuntimeError(
                "Cannot change existing k3d topology during migration: "
                + ", ".join(changed_topology)
            )

        migration_state["config"] = effective
        migration_state = save_redis_migration_state(
            k3d.os_artifacts, migration_state
        )
        if migration_state["phase"] == "restored":
            if not redis_migration_matches_current_cluster(
                kubectl, migration_state
            ):
                migration_state = save_redis_migration_state(
                    k3d.os_artifacts, migration_state, "cluster_created"
                )
            else:
                save_local_config(
                    k3d.os_artifacts, k3d.cluster_name, effective
                )
                complete_redis_migration(
                    kubectl, k3d.os_artifacts, migration_state
                )
                migration_state = None

    storage_checked = False
    if legacy_services or (
        migration_state is not None and migration_state["phase"] == "prepared"
    ):
        os.makedirs(storage_path, exist_ok=True)
        if not check_disk_space(storage_path):
            return False
        storage_checked = True
    if legacy_services:
        log(
            "This cluster uses the previous chart-based Redis and RabbitMQ deployment.",
            level="warning",
        )
        if migration_state is None:
            confirmation = verify_to_proceed(
                "The one-time migration recreates the local cluster and restores Redis "
                "workflow state, but pending RabbitMQ messages and user-added Kubernetes "
                "secrets cannot be migrated. Make sure no workflows are running and save "
                "any secrets you need to re-add. Do you want to continue?"
            )
            if not confirmation:
                log("Aborting update due to user confirmation")
                return False

            docker_config, acr_registries, _ = prepare_registry_auth(
                kubectl,
                k3d.os_artifacts,
                effective,
                username,
                password,
                retain_selected_only=True,
                preserve_existing_secret=True,
            )
            migration_state = new_redis_migration_state(
                k3d.cluster_name,
                kubectl.context_name,
                kubectl.get_cluster_uid(),
                effective,
                docker_config,
                acr_registries,
            )
            migration_state = save_redis_migration_state(
                k3d.os_artifacts, migration_state
            )
        else:
            source_docker_config = merge_docker_auth(
                migration_state.get("docker_config"),
                get_pull_secret(kubectl),
            )
            docker_config, acr_registries, _ = prepare_registry_auth(
                kubectl,
                k3d.os_artifacts,
                effective,
                username,
                password,
                source_docker_config,
                migration_state.get("acr_registries"),
                use_existing=False,
                retain_selected_only=True,
                preserve_existing_secret=True,
            )
            migration_state["docker_config"] = docker_config
            migration_state["acr_registries"] = acr_registries
            migration_state = save_redis_migration_state(
                k3d.os_artifacts, migration_state
            )
        migration_state = ensure_migration_backup(
            kubectl, k3d.os_artifacts, data_path, migration_state
        )
        if not destroy(
            k3d,
            data_path=data_path,
            skip_confirmation=True,
            require_backup=True,
        ):
            return False
        cluster_exists = False
    elif migration_state is not None:
        if migration_state["phase"] == "prepared":
            if not cluster_exists:
                raise RuntimeError(
                    "Redis migration source disappeared before its backup completed"
                )
            migration_state = ensure_migration_backup(
                kubectl, k3d.os_artifacts, data_path, migration_state
            )
            if not destroy(
                k3d,
                data_path=data_path,
                skip_confirmation=True,
                require_backup=True,
            ):
                return False
            cluster_exists = False
        else:
            verify_migration_backup(data_path, migration_state)

    if cluster_exists and not is_update:
        log("Seems like you might have a cluster already created.", level="warning")
        confirmation = verify_to_proceed(
            "Do you want to abort this setup and continue with the existing cluster? "
            "Answering 'no' will destroy the existing cluster and create a new one."
        )
        if confirmation:
            log("Aborting setup. Keeping existing cluster due to user confirmation.")
            return True
        if not destroy(k3d, skip_confirmation=True, data_path=data_path):
            return False
        cluster_exists = False
    elif not cluster_exists and is_update and migration_state is None:
        log("No existing cluster found to update. Aborting update.", level="error")
        return False

    os.makedirs(storage_path, exist_ok=True)
    if not storage_checked and not check_disk_space(storage_path):
        return False

    cluster_created = False
    if not cluster_exists:
        log(f"Creating cluster {k3d.cluster_name}")
        if not k3d.create(
            effective["servers"],
            effective["agents"],
            storage_path,
            effective["registry_port"],
            effective["port"],
            effective["host"],
        ):
            log("Unable to create cluster", level="error")
            return False
        cluster_exists = True
        cluster_created = True
        if migration_state is not None:
            migration_state["target_cluster_uid"] = kubectl.get_cluster_uid()
            migration_state = save_redis_migration_state(
                k3d.os_artifacts, migration_state, "cluster_created"
            )
    elif migration_state is not None and migration_state["phase"] == "backed_up":
        migration_state["target_cluster_uid"] = kubectl.get_cluster_uid()
        migration_state = save_redis_migration_state(
            k3d.os_artifacts, migration_state, "cluster_created"
        )

    fresh_provision = cluster_created or (
        migration_state is not None
        and migration_state["phase"] == "cluster_created"
    )
    terraform_is_update = is_update and not fresh_provision

    if migration_state is not None:
        target_docker_config = merge_docker_auth(
            migration_state.get("docker_config"),
            get_pull_secret(kubectl),
        )
        docker_config, acr_registries, az = prepare_registry_auth(
            kubectl,
            k3d.os_artifacts,
            effective,
            username,
            password,
            target_docker_config,
            migration_state.get("acr_registries"),
            use_existing=False,
            retain_selected_only=True,
        )
        migration_state["config"] = effective
        migration_state["docker_config"] = docker_config
        migration_state["acr_registries"] = acr_registries
        migration_state = save_redis_migration_state(
            k3d.os_artifacts, migration_state
        )
    else:
        _, _, az = prepare_registry_auth(
            kubectl,
            k3d.os_artifacts,
            effective,
            username,
            password,
            preflight=False,
        )

    if not worker_replicas:
        log(
            "No worker replicas specified. "
            "You can change this by re-running with "
            "`farmvibes-ai local setup --worker-replicas <number> ...`",
        )
        return False

    dapr_updated = False
    dapr = DaprWrapper(kubectl.os_artifacts, kubectl)
    if terraform_is_update and migration_state is None and dapr.needs_upgrade():
        log("Upgrading Dapr CRDs")
        if not dapr.upgrade_crds():
            log("Unable to upgrade Dapr CRDs", level="error")
            return False
        dapr_updated = True

    terraform = TerraformWrapper(k3d.os_artifacts, az)
    with terraform.workspace(f"farmvibes-k3d-{k3d.cluster_name}"):
        terraform.ensure_local_cluster(
            k3d.cluster_name,
            registry,
            log_level,
            max_log_file_bytes,
            log_backup_count,
            image_tag,
            image_prefix,
            data_path,
            worker_replicas,
            kubectl.context_name,
            enable_telemetry,
            redis_image,
            rabbitmq_image,
            is_update=terraform_is_update,
        )
    if (
        migration_state is not None
        and MIGRATION_PHASES.index(migration_state["phase"])
        < MIGRATION_PHASES.index("provisioned")
    ):
        migration_state = save_redis_migration_state(
            k3d.os_artifacts, migration_state, "provisioned"
        )
    # We might have downloaded newer images, so we have to fix permissions
    docker = DockerWrapper(k3d.os_artifacts)
    try:
        log("Fixing permissions on containerd image path", level="debug")
        container_name = f"k3d-{k3d.cluster_name}-server-0"
        uid_gid = f"{terraform.getuid()}:{terraform.getgid()}"
        docker.exec(container_name, ["chown", "-R", uid_gid, k3d.CONTAINERD_IMAGE_PATH])

    except Exception:
        log("Unable to fix permissions on containerd image path", level="warning")

    if dapr_updated:
        log("dapr upgraded, restarting services")
        with kubectl.context(kubectl.cluster_name):
            kubectl.restart("deployment", selectors=["backend=terravibes"])

    log(f"Cluster {'update' if is_update else 'setup'} complete!")

    if migration_state is not None:
        if (
            migration_state["phase"] == "restoring"
            and redis_migration_marker_matches(kubectl, migration_state)
        ):
            migration_state = save_redis_migration_state(
                k3d.os_artifacts, migration_state, "restored"
            )
        elif migration_state["phase"] in ("provisioned", "restoring"):
            verify_migration_backup(data_path, migration_state)
            migration_state = save_redis_migration_state(
                k3d.os_artifacts, migration_state, "restoring"
            )
            restored = restore_redis_data(
                kubectl,
                data_path,
                skip_confirmation=True,
                redis_image=redis_image,
                dump_file=migration_state["backup_file"],
            )
            if not restored or not redis_migration_marker_matches(
                kubectl, migration_state
            ):
                message = "Unable to verify Redis workflow state after migration."
                log(message, level="error")
                raise RuntimeError(message)
            migration_state = save_redis_migration_state(
                k3d.os_artifacts, migration_state, "restored"
            )

        save_local_config(k3d.os_artifacts, k3d.cluster_name, effective)
        complete_redis_migration(
            kubectl, k3d.os_artifacts, migration_state
        )
    elif not terraform_is_update:
        restored = restore_redis_data(
            kubectl,
            data_path,
            skip_confirmation=False,
            redis_image=redis_image,
        )
        del restored

    save_local_config(k3d.os_artifacts, k3d.cluster_name, effective)

    status(k3d)
    with open(k3d.os_artifacts.config_dir / "storage", "w") as f:
        f.write(storage_path)
    return True


def get_service_from_docker_network(os_artifacts: OSArtifacts, cluster_name: str):
    docker = DockerWrapper(os_artifacts)
    result = docker.network_inspect(f"k3d-{cluster_name}")
    if not result:
        log("Unable to get service from docker network", level="error")
        return ""
    ip = ""
    for container in result[0]["Containers"].values():
        if container["Name"] == f"k3d-{cluster_name}-server-0":
            ip = container["IPv4Address"]
            break
    if not ip:
        log("Unable to get service from docker network", level="error")
        return ""
    if "/" in ip:
        ip = ip.split("/")[0]
    kubectl = KubectlWrapper(os_artifacts, cluster_name)
    with kubectl.context():
        result = kubectl.get("service", "terravibes-rest-api")
        if not result:
            log("Unable to get service port from kubernetes", level="error")
            return ""
        port_data = result["spec"]["ports"][0]
        port = port_data.get("nodePort", port_data.get("port", ""))
        if not port:
            log("Unable to get service from kubernetes", level="error")
            return ""
    return f"http://{ip}:{port}"


def get_service_from_ingress_loadbalancer(os_artifacts: OSArtifacts, cluster_name: str):
    kubectl = KubectlWrapper(os_artifacts, cluster_name)
    with kubectl.context():
        ip = kubectl.get("ingress", "terravibes-rest-api", ".status.loadBalancer.ingress[0].ip")
        if not ip:
            log("Unable to get service from kubernetes", level="error")
            return ""
        port = kubectl.get(
            "ingress",
            "terravibes-rest-api",
            "{{.spec.rules[0].http.paths[0].backend.service.port.number}}",
        )
        if not port:
            log("Unable to get service from kubernetes", level="error")
            return ""

        return f"http://{ip}" + (f":{port}" if port != "80" else "")


def write_service_url(os_artifacts: OSArtifacts, cluster: Dict[str, Any]):
    service_url = ""
    for node in cluster["nodes"]:
        if node["role"].lower() == "loadbalancer":
            for name, value in node["portMappings"].items():
                if name == "80/tcp":
                    service_url = f"http://{value[0]['HostIp']}:{value[0]['HostPort']}"
                    break
    if not service_url:
        service_url = get_service_from_docker_network(os_artifacts, cluster["name"])
        if not service_url:
            # Old cluster, didn't have port forward, probably has a load balancer after
            # an update. We get the ip of the load balancer and use that.
            service_url = get_service_from_ingress_loadbalancer(os_artifacts, cluster["name"])
            if not service_url:
                log("Unable to get service url", level="error")
                return ""

    service_url_file = os_artifacts.config_file(LOCAL_SERVICE_URL_PATH_FILE)
    log(f"Writing service url {service_url} to {service_url_file}", level="debug")
    with open(service_url_file, "w") as f:
        f.write(service_url)
    return service_url


def status(k3d: K3dWrapper) -> bool:
    cluster = k3d.info()
    if not cluster:
        log(f"Cluster {k3d.cluster_name} not found", level="error")
        return False
    else:
        log(f"Cluster {k3d.cluster_name} found", level="debug")
        if cluster["serversRunning"] > 0:
            log(
                f"Cluster {k3d.cluster_name} is running with {cluster['serversRunning']} "
                f"servers and {cluster['agentsRunning']} agents."
            )
            service_url = write_service_url(k3d.os_artifacts, cluster)
            if service_url:
                log(f"Service url is {service_url}")
        else:
            log(f"Cluster {k3d.cluster_name} is not running", level="warning")
            return False
        return True


def start(k3d: K3dWrapper) -> bool:
    if not k3d.cluster_exists():
        log(f"Cluster {k3d.cluster_name} does not exist, nothing to start", level="error")
        return False
    log(f"Starting cluster '{k3d.cluster_name}'")
    if not k3d.start():
        log("Unable to start cluster", level="error")
        return False
    log(
        "On cluster start, services are not immediately available. "
        "Please wait at least 30 seconds before trying to access the service.",
        level="warning",
    )
    cluster = k3d.info()
    service_url = write_service_url(k3d.os_artifacts, cluster)
    if service_url:
        log(f"When ready, service url is {service_url}")
    return True


def stop(k3d: K3dWrapper) -> bool:
    if not k3d.cluster_exists():
        log(f"Cluster {k3d.cluster_name} does not exist, nothing to stop", level="error")
        return False
    log(f"Stopping cluster '{k3d.cluster_name}'")
    if not k3d.stop():
        log("Unable to stop cluster", level="error")
        return False
    return True


def restart(k3d: K3dWrapper) -> bool:
    return stop(k3d) and start(k3d)


def add_secret(
    os_artifacts: OSArtifacts,
    cluster_name: str,
    secret_name: str,
    secret_value: str,
):
    log(f"Adding secret {secret_name} to cluster {cluster_name}")
    kubectl = KubectlWrapper(os_artifacts, cluster_name)
    with kubectl.context():
        kubectl.add_secret(secret_name, secret_value)
    log(f"Added secret {secret_name} to cluster {cluster_name}")


def delete_secret(
    os_artifacts: OSArtifacts,
    cluster_name: str,
    secret_name: str,
):
    log(f"Deleting secret {secret_name} from cluster {cluster_name}")
    kubectl = KubectlWrapper(os_artifacts, cluster_name)
    with kubectl.context():
        kubectl.delete_secret(secret_name)
    log(f"Deleted secret {secret_name} from cluster {cluster_name}")


def add_onnx(cluster_name: str, storage_path: str, onnx: str):
    log(f"Adding ONNX {onnx} to cluster {cluster_name}")
    if not os.path.exists(onnx):
        log(f"ONNX file {onnx} does not exist", level="error")
        return False
    # Will try to hardlink the file, if not possible will copy it
    destination = os.path.join(storage_path, ONNX_SUBDIR, os.path.basename(onnx))
    if not os.path.exists(os.path.dirname(destination)):
        os.makedirs(os.path.dirname(destination), exist_ok=True)
    if os.path.exists(destination):
        log(f"ONNX file {destination} already exists, skipping", level="warning")
        return False
    try:
        os.link(onnx, destination)
        log(f"Hardlinked {onnx} to {destination}")
    except OSError:
        try:
            copied = shutil.copy(onnx, destination)
            log(f"Copied {onnx} to {copied}")
        except Exception as e:
            log(f"Could not copy {onnx} to {destination}: {e}", level="error")
            return False
    return True


def _dispatch_unlocked(
    args: argparse.Namespace, os_artifacts: OSArtifacts
):
    os_artifacts.check_dependencies(InstallType.LOCAL)

    # We want to prefer our copies of the binaries, especially when the system
    # has an unsupported version, so that we don't slow down every time checking
    # the version.
    os.environ["PATH"] = f"{os_artifacts.config_dir}{os.pathsep}{os.environ['PATH']}"

    k3d = K3dWrapper(os_artifacts, args.cluster_name)

    requested_storage_path = getattr(args, "storage_path", "")
    kubectl = KubectlWrapper(os_artifacts, args.cluster_name)
    pending_state = load_redis_migration_state(
        os_artifacts,
        args.cluster_name,
        kubectl.context_name,
    )
    if pending_state is not None:
        storage_path = pending_state["config"]["storage_path"]
    else:
        storage_path = requested_storage_path
        if not storage_path:
            storage_path = load_local_config(
                os_artifacts, args.cluster_name
            ).get("storage_path", "")
            if not storage_path:
                if args.action in {
                    "setup",
                    "create",
                    "new",
                    "update",
                    "upgrade",
                    "up",
                    "destroy",
                    "delete",
                    "remove",
                    "rm",
                } and k3d.cluster_exists():
                    try:
                        storage_path = k3d.get_storage_path()
                    except ValueError as error:
                        raise RuntimeError(
                            f"Unable to determine the storage path for existing "
                            f"cluster {args.cluster_name}. Pass --storage-path "
                            "explicitly before setup, update, or destroy."
                        ) from error
                else:
                    storage_file = os_artifacts.config_dir / "storage"
                    if storage_file.exists():
                        log(
                            f"Loading storage path from {storage_file}",
                            level="warning",
                        )
                        with open(storage_file) as storage:
                            storage_path = storage.read().strip()
                    else:
                        storage_path = DEFAULT_STORAGE_PATH
        if not isinstance(storage_path, str):
            raise ValueError("Invalid saved storage path")
        data_path = os.path.join(storage_path, DATA_SUFFIX)
        pending_state = load_redis_migration_state(
            os_artifacts,
            args.cluster_name,
            kubectl.context_name,
            data_path,
        )
        if (
            pending_state is not None
            and pending_state["config"]["storage_path"] != storage_path
        ):
            raise ValueError(
                "Redis migration checkpoint storage path does not match "
                "its saved deployment configuration"
            )

    if pending_state is not None:
        if args.action in {"setup", "create", "new"}:
            provided = getattr(args, "_provided_options", set())
            for field in pending_state["config"]:
                if field == "enable_telemetry" or not hasattr(args, field):
                    continue
                option = f"--{field.replace('_', '-')}"
                if option not in provided:
                    setattr(args, field, None)
            if not provided.intersection(
                {"--enable-telemetry", "--disable-telemetry"}
            ):
                args.enable_telemetry = None
            for field in ("registry_username", "registry_password"):
                option = f"--{field.replace('_', '-')}"
                if option not in provided:
                    setattr(args, field, None)
        if not requested_storage_path:
            args.storage_path = None
    if not isinstance(storage_path, str):
        raise ValueError("Invalid saved storage path")
    if hasattr(args, "storage_path") and pending_state is None:
        args.storage_path = storage_path
    data_path = os.path.join(storage_path, DATA_SUFFIX)

    if args.action in {"setup", "create", "new", "update", "upgrade", "up"}:
        is_update = args.action in {"update", "upgrade", "up"}
        old_k3d = K3dWrapper(os_artifacts, OLD_DEFAULT_CLUSTER_NAME)
        if pending_state is None and old_k3d.cluster_exists():
            confirmation = verify_to_proceed(
                "Your have a cluster that uses an old format and needs to be recreated. "
                "Do you want to proceed?"
            )
            if confirmation:
                if not destroy_old_registry(os_artifacts):
                    log("Could not destroy old registry", level="error")
                    return False
                old_k3d.delete()
                is_update = False
            else:
                log("Aborting update due to old cluster being present", level="error")
                return False
        enable_telemetry = args.enable_telemetry if hasattr(args, "enable_telemetry") else False
        return setup(
            k3d,
            args.servers,
            args.agents,
            args.storage_path,
            args.registry,
            args.registry_username,
            args.registry_password,
            args.log_level,
            args.max_log_file_bytes,
            args.log_backup_count,
            args.image_tag,
            args.image_prefix,
            data_path,
            args.worker_replicas,
            enable_telemetry,
            args.port,
            args.host,
            is_update=is_update,
            registry_port=args.registry_port,
            redis_image=args.redis_image,
            rabbitmq_image=args.rabbitmq_image,
        )
    elif args.action in {"destroy", "delete", "remove", "rm"}:
        return destroy(k3d, data_path=data_path)
    elif args.action == "start":
        return start(k3d)
    elif args.action == "stop":
        return stop(k3d)
    elif args.action == "restart":
        return restart(k3d)
    elif args.action in {"status", "url", "show-url"}:
        return status(k3d)
    elif args.action in {"add-secret", "add_secret"}:
        return add_secret(os_artifacts, args.cluster_name, args.secret_name, args.secret_value)
    elif args.action in {"delete-secret", "delete_secret"}:
        return delete_secret(os_artifacts, args.cluster_name, args.secret_name)
    elif args.action == "add-onnx":
        return add_onnx(args.cluster_name, storage_path, args.model_path)
    else:
        raise RuntimeError(f"Unknown action: {args.action}")


def dispatch(args: argparse.Namespace):
    os_artifacts = OSArtifacts()
    if args.action in {
        "setup",
        "create",
        "new",
        "update",
        "upgrade",
        "up",
        "destroy",
        "delete",
        "remove",
        "rm",
    }:
        with local_cluster_lock(os_artifacts, "__all_local_clusters__"):
            return _dispatch_unlocked(args, os_artifacts)
    return _dispatch_unlocked(args, os_artifacts)
