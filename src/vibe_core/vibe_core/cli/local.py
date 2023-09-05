import argparse
import codecs
import os
import shutil
from typing import Any, Dict, Tuple

from vibe_core.cli.constants import (
    AZURE_CR_DOMAIN,
    DEFAULT_IMAGE_PREFIX,
    DEFAULT_IMAGE_TAG,
    DEFAULT_REGISTRY_PATH,
    FARMVIBES_AI_LOG_LEVEL,
    LOCAL_SERVICE_URL_PATH_FILE,
    ONNX_SUBDIR,
)
from vibe_core.cli.helper import log_should_be_logged_in, verify_to_proceed
from vibe_core.cli.logging import log
from vibe_core.cli.osartifacts import InstallType, OSArtifacts
from vibe_core.cli.wrappers import (
    AzureCliWrapper,
    DockerWrapper,
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
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 31108
REGISTRY_PORT = 5000
OLD_DEFAULT_CLUSTER_NAME = "farmvibes-ai"


def find_redis_master(kubectl: KubectlWrapper) -> Tuple[str, ...]:
    pods = kubectl.list_pods()
    redis_master_pod = ""
    for pod in pods["items"]:
        if pod["metadata"]["name"].startswith("redis-master"):
            redis_master_pod = pod
            break
    if not redis_master_pod:
        log("Unable to find redis master pod", level="warning")
        ß = kubectl.get("statefulset", "redis-master")
        kind = ß["kind"]
        name = ß["metadata"]["name"]
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


def backup_redis_data(kubectl: KubectlWrapper, data_path: str) -> bool:
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
                return verify_to_proceed(
                    "Would you like to continue without backing up redis data?"
                )

            save_command = (
                f"echo -e 'AUTH {redis_password}\\nCONFIG SET appendonly no\\nsave' | redis-cli"
            )
            command = ["bash", "-c", save_command]
            kubectl.exec(master_pod, command, capture_output=True)

            log("Saving redis data dump on the host machine")
            final_path = os.path.join(data_path, REDIS_DUMP)
            kubectl.cp(f"{master_pod}:/data/dump.rdb", final_path)
            log(f"Redis data dump saved to {final_path}")
            return True
    except Exception:
        return False


def restore_redis_data(kubectl: KubectlWrapper, data_path: str) -> bool:
    _, redis_master, kind = find_redis_master(kubectl)
    backup_path = os.path.join(data_path, REDIS_DUMP)

    if not redis_master:
        return False
    if not os.path.exists(backup_path):
        return False

    confirmation = verify_to_proceed(
        "I've found a state store backup file from a previous installation. "
        "Do you want to restore it?"
    )
    if not confirmation:
        log("Not restoring backup from user instructions.")

    with kubectl.context():
        try:
            kubectl.scale(kind, redis_master, 0)
            if not kubectl.create_redis_volume_pod():
                log("Unable to create redis volume pod", level="error")
                return False
            kubectl.cp(backup_path, "redisvolpod:/mnt/dump.rdb")
            kubectl.delete("pod", "redisvolpod")
        finally:
            kubectl.scale(kind, redis_master, 1)

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
) -> bool:
    log(f"Destroying local cluster with name {k3d.cluster_name}")
    if not k3d.cluster_exists():
        log("Cluster does not exist, nothing to destroy")
        return True
    if not skip_confirmation:
        confirmation = verify_to_proceed(
            "Do you want to destroy the local cluster? "
            "This will delete all the data in the cluster."
        )
        if not confirmation:
            log("Aborting destroy due to user confirmation")
            return True
    confirmation = skip_confirmation or verify_to_proceed(
        "Do you want to backup workflow state data before destroying the cluster?"
    )
    if confirmation:
        kubectl = KubectlWrapper(k3d.os_artifacts, k3d.cluster_name)
        if not backup_redis_data(kubectl, data_path):
            if not skip_confirmation:
                confirmation = verify_to_proceed(
                    "Unable to backup redis data, do you want to continue?"
                )
                if not confirmation:
                    log("Aborting destroy due to user confirmation")
                    return True
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
    servers: int = 1,
    agents: int = 0,
    storage_path: str = DEFAULT_STORAGE_PATH,
    registry: str = DEFAULT_REGISTRY_PATH,
    username: str = "",
    password: str = "",
    log_level: str = FARMVIBES_AI_LOG_LEVEL,
    image_tag: str = DEFAULT_IMAGE_TAG,
    image_prefix: str = DEFAULT_IMAGE_PREFIX,
    data_path: str = "",
    worker_replicas: int = 0,
    port: int = DEFAULT_PORT,
    host: str = DEFAULT_HOST,
    is_update: bool = False,
) -> bool:
    if not is_update:
        log("Setting up local cluster")
    else:
        log("Updating local cluster")

    if k3d.cluster_exists():
        if not is_update:
            log(
                "Seems like you might have a cluster already created.",
                level="warning",
            )
            confirmation = verify_to_proceed(
                "Do you want to abort this setup and continue with the existing cluster? "
                "Answering 'no' will destroy the existing cluster and create a new one."
            )
            if confirmation:
                log("Aborting setup. Keeping existing cluster due to user confirmation.")
                return True
            else:
                destroy(k3d, skip_confirmation=True, data_path=data_path)
    else:
        if is_update:
            log("No existing cluster found to update. Aborting update.", level="error")
            return False

    if not os.path.exists(storage_path):
        log(f"Creating storage path {storage_path}")
        os.makedirs(storage_path, exist_ok=True)

    if not os.path.exists(data_path):
        log(f"Creating data path {data_path}")
        os.makedirs(data_path, exist_ok=True)

    if not check_disk_space(storage_path):
        return False

    if not is_update:
        log(f"Creating cluster {k3d.cluster_name}")
        if not k3d.create(servers, agents, storage_path, REGISTRY_PORT, port, host):
            log("Unable to create cluster", level="error")
            return False

    az = None
    kubectl = KubectlWrapper(k3d.os_artifacts, k3d.cluster_name)
    if not is_update and registry.endswith(AZURE_CR_DOMAIN) and (not username or not password):
        k3d.os_artifacts.check_dependencies(InstallType.ALL)
        az = AzureCliWrapper(k3d.os_artifacts, "")
        log(
            f"Username and password not provided for {registry}, inferring from Azure CLI",
            level="warning",
        )

        try:
            az.get_subscription_info()  # Needed for confirming subscription
        except Exception as e:
            log_should_be_logged_in(e)
            return False

        username, password = az.infer_registry_credentials(registry)

    if username and password:
        log(f"Creating Docker credentials for registry {registry}")
        kubectl.create_docker_token("acrtoken", registry, username, password)

    if not worker_replicas:
        log(
            "No worker replicas specified. "
            "You can change this by re-running with "
            "`farmvibes-ai local setup --worker-replicas <number> ...`",
        )
        return False

    terraform = TerraformWrapper(k3d.os_artifacts, az)
    with terraform.workspace(f"farmvibes-k3d-{k3d.cluster_name}"):
        terraform.ensure_local_cluster(
            k3d.cluster_name,
            registry,
            log_level,
            image_tag,
            image_prefix,
            data_path,
            worker_replicas,
            kubectl.context_name,
            is_update=is_update,
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

    log(f"Cluster {'update' if is_update else 'setup'} complete!")

    if not is_update:
        restore_redis_data(kubectl, data_path)

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


def dispatch(args: argparse.Namespace):
    os_artifacts = OSArtifacts()
    os_artifacts.check_dependencies(InstallType.LOCAL)

    # We want to prefer our copies of the binaries, especially when the system
    # has an unsupported version, so that we don't slow down every time checking
    # the version.
    os.environ["PATH"] = f"{os_artifacts.config_dir}{os.pathsep}{os.environ['PATH']}"

    k3d = K3dWrapper(os_artifacts, args.cluster_name)

    if hasattr(args, "storage_path"):
        if not args.storage_path:
            storage_file = os_artifacts.config_dir / "storage"
            if storage_file.exists():
                log(f"Loading storage path from {storage_file}", level="warning")
                with open(storage_file) as f:
                    args.storage_path = f.read().strip()
            else:
                args.storage_path = DEFAULT_STORAGE_PATH

    try:
        data_path = os.path.join(args.storage_path, DATA_SUFFIX)
    except AttributeError:
        data_path = os.path.join(DEFAULT_STORAGE_PATH, DATA_SUFFIX)

    if args.action in {"setup", "update", "upgrade", "up"}:
        is_update = args.action.lower().startswith("u")
        old_k3d = K3dWrapper(os_artifacts, OLD_DEFAULT_CLUSTER_NAME)
        if old_k3d.cluster_exists():
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
        return setup(
            k3d,
            args.servers,
            args.agents,
            args.storage_path,
            args.registry,
            args.registry_username,
            args.registry_password,
            args.log_level,
            args.image_tag,
            args.image_prefix,
            data_path,
            args.worker_replicas,
            args.port,
            args.host,
            is_update=is_update,
        )
    elif args.action == "destroy":
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
        return add_onnx(args.cluster_name, args.storage_path, args.model_path)
    else:
        raise RuntimeError(f"Unknown action: {args.action}")
