# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import base64
import gzip
import hashlib
import json
import os
import pkgutil
import platform
import re
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from functools import partialmethod
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from .constants import RABBITMQ_IMAGE, REDIS_IMAGE
from .helper import execute_cmd, is_port_free, log_should_be_logged_in, verify_to_proceed
from .logging import ColorFormatter, log
from .osartifacts import OSArtifacts, secure_path

AZ_CREDS_REFRESH_ATTEMPTS = 2
AZ_LOGIN_PROMPT = "`az login`"
TOTAL_REGIONAL_CPU_NAME = "Total Regional vCPUs"
WORKER_NODE_CPU_NAME = "Standard DSv3 Family vCPUs"
DEFAULT_NODE_CPU_NAME = "Standard BS Family vCPUs"
REGISTERED = "Registered"

AZURE_RESOURCES_REQUIRED = [
    "Microsoft.DocumentDB",
    "Microsoft.KeyVault",
    "Microsoft.ContainerService",
    "Microsoft.Network",
    "Microsoft.Storage",
    "Microsoft.Compute",
]

CPUS_REQUIRED = {
    TOTAL_REGIONAL_CPU_NAME: 8,
    WORKER_NODE_CPU_NAME: 4,
    DEFAULT_NODE_CPU_NAME: 4,
}
MAXIMUM_STORAGE_ACCOUNT_NAME_LENGTH = 24
CONFIG_CONTEXT = "k3d-{cluster_name}"
REDIS_VOL_POD_YAML = """apiVersion: v1
kind: Pod
metadata:
  name: redisvolpod
spec:
  imagePullSecrets:
  - name: acrtoken
  containers:
  - command:
    - tail
    - "-f"
    - "/dev/null"
    image: {redis_image}
    name: delete-this-container
    volumeMounts:
    - mountPath: "/mnt"
      name: redisdata
  restartPolicy: Never
  volumes:
  - name: redisdata
    persistentVolumeClaim:
      claimName: redis-data-redis-master-0
"""


class ImagePullAuthenticationError(RuntimeError):
    def __init__(self, image: str, status: str):
        self.image = image
        super().__init__(
            f"Unable to authenticate while pulling {image}: {status}"
        )


def on_windows() -> bool:
    return platform.system() == "Windows"


class TerraformWrapper:
    STATE_CONTAINER_NAME = "terraform-state"
    INFRA_STATE_FILE = "infra.tfstate"
    SERVICES_STATE_FILE = "services.tfstate"
    LEGACY_SERVICES_STATE_SECRET = "tfstate-default-terraform-state"
    LEGACY_SERVICES_STATE_SELECTOR = (
        "tfstate=true,tfstateSecretSuffix=terraform-state,"
        "tfstateWorkspace=default"
    )
    LEGACY_SERVICES_STATE_LOCK = "lock-tfstate-default-terraform-state"
    LEGACY_MIGRATION_LINEAGE = "farmvibes.ai/migrated-lineage"
    LEGACY_MIGRATION_SERIAL = "farmvibes.ai/migrated-serial"
    LEGACY_LOCK_DURATION_SECONDS = 900
    ANSI_ESCAPE_PAT = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    REPLACEMENT_PAT = re.compile(r"#\s+(.*)\s+must\s+be\s+replaced")
    REPLACEMENT_SUBSTRINGS = [
        "cosmosdb",
        "storageaccount",
    ]
    PLUGIN_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "farmvibes-ai", "opentofu")

    def __init__(
        self,
        os_artifacts: OSArtifacts,
        az: Optional["AzureCliWrapper"] = None,
        environment: str = "public",
    ):
        self.az = az
        self.os_artifacts = os_artifacts
        self.environment = environment
        os.makedirs(self.PLUGIN_CACHE_DIR, exist_ok=True)

    def _get_replacements(self, plan: str) -> List[str]:
        plan = self.ANSI_ESCAPE_PAT.sub("", plan)
        return self.REPLACEMENT_PAT.findall(plan)

    def _has_storage_replacement(self, replacements: List[str]) -> bool:
        return any([s in r for s in self.REPLACEMENT_SUBSTRINGS for r in replacements])

    def _plan_or_apply(
        self,
        working_directory: str,
        state_file: str,
        variables: Dict[str, str],
        refresh_creds: bool = True,
        plan: bool = False,
        destroy: bool = False,
        plan_file: str = "",
        targets: Optional[List[str]] = None,
    ):
        if refresh_creds:
            assert self.az is not None, "AzureCliWrapper must be provided to refresh credentials"
            self.az.refresh_az_creds()
        action = "plan" if plan else "destroy" if destroy else "apply"
        log(
            f"{'Planning' if plan else 'Destroying' if destroy else 'Applying'} "
            f"OpenTofu in {working_directory}"
        )
        command = [
            self.os_artifacts.terraform,
            f"-chdir={working_directory}",
            action,
            f"-state={state_file}",
        ]
        env_vars = {"TF_PLUGIN_CACHE_DIR": self.PLUGIN_CACHE_DIR}
        if not plan:
            command += ["-auto-approve"]
        if plan_file:
            if plan:
                command += [f"-out={plan_file}"]
            else:
                command += ["-input=false", plan_file]
        if plan or not plan_file:
            for k, v in variables.items():
                if "path" in k:
                    v = v.replace("\\", "/")
                command += ["-var", f"{k}={v}"]
            command += ["-var", f"environment={self.environment}"]
        command += [f"-target={target}" for target in targets or []]
        env_vars["ARM_ENVIRONMENT"] = self.environment
        stdout = execute_cmd(
            command,
            check_return_code=True,
            check_empty_result=False,
            error_string=(
                f"Failed to {action} OpenTofu resources "
                f"in {working_directory}"
            ),
            capture_output=True,
            env_vars=env_vars,
        )
        return stdout

    plan = partialmethod(_plan_or_apply, plan=True)
    apply = partialmethod(_plan_or_apply, plan=False)
    destroy = partialmethod(_plan_or_apply, destroy=True)

    def state_resources(self, working_directory: str, state_file: str) -> List[str]:
        output = execute_cmd(
            [
                self.os_artifacts.terraform,
                f"-chdir={working_directory}",
                "state",
                "list",
                f"-state={state_file}",
            ],
            check_return_code=True,
            check_empty_result=False,
            error_string=f"Failed to list OpenTofu state in {working_directory}",
            capture_output=True,
            env_vars={
                "ARM_ENVIRONMENT": self.environment,
                "TF_PLUGIN_CACHE_DIR": self.PLUGIN_CACHE_DIR,
            },
        )
        return output.splitlines()

    def destroy_legacy_service_charts(
        self,
        working_directory: str,
        state_file: str,
        variables: Dict[str, str],
        cluster_name: str,
        kubernetes_config_context: str,
        on_destroy: Optional[Callable[[], None]] = None,
    ) -> None:
        legacy = [
            resource
            for resource in ("helm_release.redis", "helm_release.rabbitmq")
            if resource in self.state_resources(working_directory, state_file)
        ]
        if not legacy:
            log("No legacy Helm service releases found", level="debug")
        else:
            if on_destroy is not None:
                on_destroy()
            self.destroy(working_directory, state_file, variables, targets=legacy)
        kubectl = KubectlWrapper(
            self.os_artifacts,
            cluster_name,
            config_context=kubernetes_config_context,
        )
        delete_rabbitmq_pvc = "helm_release.rabbitmq" in legacy
        with kubectl.context():
            if not delete_rabbitmq_pvc:
                pvc = kubectl.get_or_none("pvc", "data-rabbitmq-0")
                labels = pvc.get("metadata", {}).get("labels", {}) if pvc else {}
                delete_rabbitmq_pvc = (
                    labels.get("app.kubernetes.io/managed-by") == "Helm"
                    and labels.get("app.kubernetes.io/instance") == "rabbitmq"
                )
            if delete_rabbitmq_pvc:
                kubectl.delete(
                    "pvc", "data-rabbitmq-0", ignore_not_found=True
                )

    def get_output(
        self,
        working_directory: str,
        state_file: str,
        refresh_creds: bool = True,
    ):
        if refresh_creds:
            assert self.az is not None, "AzureCliWrapper must be provided to refresh credentials"
            self.az.refresh_az_creds()
        command = [
            self.os_artifacts.terraform,
            f"-chdir={working_directory}",
            "output",
            f"-state={state_file}",
            "-json",
        ]
        env_vars = {
            "ARM_ENVIRONMENT": self.environment,
            "TF_PLUGIN_CACHE_DIR": self.PLUGIN_CACHE_DIR,
        }
        log(f"Trying to get output from {command} with env vars {env_vars}", level="debug")
        output = execute_cmd(
            command,
            True,
            False,
            f"Failed to get OpenTofu results from {working_directory}",
            censor_output=True,
            env_vars=env_vars,
        )
        return json.loads(output)

    def _pull_state(self, working_directory: str) -> Dict[str, Any]:
        output = execute_cmd(
            [
                self.os_artifacts.terraform,
                f"-chdir={working_directory}",
                "state",
                "pull",
            ],
            check_return_code=True,
            check_empty_result=False,
            error_string=f"Failed to pull OpenTofu state from {working_directory}",
            capture_output=True,
            censor_output=True,
            env_vars={
                "ARM_ENVIRONMENT": self.environment,
                "TF_PLUGIN_CACHE_DIR": self.PLUGIN_CACHE_DIR,
            },
        )
        return json.loads(output) if output else {}

    def _legacy_services_state_secrets(
        self,
        kubernetes_config_path: str,
        kubernetes_config_context: str,
    ) -> List[Dict[str, Any]]:
        output = execute_cmd(
            [
                self.os_artifacts.kubectl,
                "--kubeconfig",
                kubernetes_config_path,
                "--context",
                kubernetes_config_context,
                "get",
                "secrets",
                "--namespace",
                "default",
                "--selector",
                self.LEGACY_SERVICES_STATE_SELECTOR,
                "--output",
                "json",
            ],
            check_empty_result=False,
            error_string="Failed to read legacy services state",
            capture_output=True,
            censor_output=True,
            subprocess_log_level="debug",
        )
        return json.loads(output or "{}").get("items", [])

    @contextmanager
    def _lock_legacy_services_state(
        self,
        kubernetes_config_path: str,
        kubernetes_config_context: str,
    ):
        lock_id = str(uuid.uuid4())
        lock_info = json.dumps(
            {
                "ID": lock_id,
                "Operation": "migration",
                "Info": "FarmVibes services state migration",
                "Who": "farmvibes-ai",
                "Version": "OpenTofu",
                "Created": datetime.now(timezone.utc).isoformat(),
                "Path": "default",
            }
        )
        acquired_at = datetime.now(timezone.utc)
        acquired_at_string = acquired_at.isoformat().replace("+00:00", "Z")
        command = [
            self.os_artifacts.kubectl,
            "--kubeconfig",
            kubernetes_config_path,
            "--context",
            kubernetes_config_context,
            "--namespace",
            "default",
        ]
        manifest = {
            "apiVersion": "coordination.k8s.io/v1",
            "kind": "Lease",
            "metadata": {
                "name": self.LEGACY_SERVICES_STATE_LOCK,
                "annotations": {"app.terraform.io/lock-info": lock_info},
                "labels": {
                    "tfstate": "true",
                    "tfstateSecretSuffix": "terraform-state",
                    "tfstateWorkspace": "default",
                    "app.kubernetes.io/managed-by": "terraform",
                },
            },
            "spec": {
                "holderIdentity": lock_id,
                "acquireTime": acquired_at_string,
                "leaseDurationSeconds": self.LEGACY_LOCK_DURATION_SECONDS,
            },
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as manifest_file:
            json.dump(manifest, manifest_file)
            manifest_file.flush()
            manifest_path = manifest_file.name
        try:
            execute_cmd(
                command + ["create", "--filename", manifest_path],
                error_string="Failed to lock legacy services state",
                censor_output=True,
                subprocess_log_level="debug",
            )
        except Exception:
            output = execute_cmd(
                command
                + [
                    "get",
                    "lease",
                    self.LEGACY_SERVICES_STATE_LOCK,
                    "--output",
                    "json",
                ],
                error_string="Failed to inspect legacy services state lock",
                censor_output=True,
                subprocess_log_level="debug",
            )
            lease = json.loads(output)
            spec = lease.get("spec", {})
            holder = spec.get("holderIdentity")
            if holder and holder != lock_id:
                acquire_time = spec.get("renewTime") or spec.get("acquireTime")
                duration = spec.get("leaseDurationSeconds")
                stale = False
                if acquire_time and duration:
                    acquired = datetime.fromisoformat(
                        acquire_time.replace("Z", "+00:00")
                    )
                    stale = (
                        datetime.now(timezone.utc) - acquired
                    ).total_seconds() > duration
                if not stale:
                    raise RuntimeError("Legacy services state is locked")
            if holder != lock_id:
                patch = [
                    {
                        "op": "test",
                        "path": "/metadata/resourceVersion",
                        "value": lease["metadata"]["resourceVersion"],
                    },
                    {
                        "op": "add",
                        "path": "/spec/holderIdentity",
                        "value": lock_id,
                    },
                    {
                        "op": "add",
                        "path": "/spec/acquireTime",
                        "value": acquired_at_string,
                    },
                    {
                        "op": "add",
                        "path": "/spec/renewTime",
                        "value": acquired_at_string,
                    },
                    {
                        "op": "add",
                        "path": "/spec/leaseDurationSeconds",
                        "value": self.LEGACY_LOCK_DURATION_SECONDS,
                    },
                    {
                        "op": "add",
                        "path": (
                            "/metadata/annotations/"
                            "app.terraform.io~1lock-info"
                            if lease["metadata"].get("annotations")
                            else "/metadata/annotations"
                        ),
                        "value": (
                            lock_info
                            if lease["metadata"].get("annotations")
                            else {"app.terraform.io/lock-info": lock_info}
                        ),
                    },
                ]
                execute_cmd(
                    command
                    + [
                        "patch",
                        "lease",
                        self.LEGACY_SERVICES_STATE_LOCK,
                        "--type",
                        "json",
                        "--patch",
                        json.dumps(patch),
                    ],
                    error_string="Failed to lock legacy services state",
                    censor_output=True,
                    subprocess_log_level="debug",
                )
        finally:
            os.remove(manifest_path)
        heartbeat_stop = threading.Event()
        def renew() -> None:
            patch = [
                {
                    "op": "test",
                    "path": "/spec/holderIdentity",
                    "value": lock_id,
                },
                {
                    "op": "add",
                    "path": "/spec/renewTime",
                    "value": datetime.now(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                },
            ]
            execute_cmd(
                command
                + [
                    "patch",
                    "lease",
                    self.LEGACY_SERVICES_STATE_LOCK,
                    "--type",
                    "json",
                    "--patch",
                    json.dumps(patch),
                ],
                error_string="Failed to renew legacy services state lock",
                censor_output=True,
                subprocess_log_level="debug",
            )

        def heartbeat() -> None:
            while not heartbeat_stop.wait(
                self.LEGACY_LOCK_DURATION_SECONDS / 3
            ):
                try:
                    renew()
                except Exception:
                    continue

        heartbeat_thread = threading.Thread(
            target=heartbeat,
            name="farmvibes-state-lock",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            yield renew
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join()
            patch = [
                {
                    "op": "test",
                    "path": "/spec/holderIdentity",
                    "value": lock_id,
                },
                {
                    "op": "replace",
                    "path": "/spec/holderIdentity",
                    "value": None,
                },
                {
                    "op": "remove",
                    "path": "/metadata/annotations/app.terraform.io~1lock-info",
                },
            ]
            execute_cmd(
                command
                + [
                    "patch",
                    "lease",
                    self.LEGACY_SERVICES_STATE_LOCK,
                    "--type",
                    "json",
                    "--patch",
                    json.dumps(patch),
                ],
                error_string="Failed to unlock legacy services state",
                censor_output=True,
                subprocess_log_level="debug",
            )

    def _pull_legacy_services_state(
        self,
        kubernetes_config_path: str,
        kubernetes_config_context: str,
    ) -> Dict[str, Any]:
        secrets = self._legacy_services_state_secrets(
            kubernetes_config_path,
            kubernetes_config_context,
        )
        if not secrets:
            return {}

        def chunk(secret: Dict[str, Any]) -> Tuple[int, bytes]:
            name = secret.get("metadata", {}).get("name", "")
            if name == self.LEGACY_SERVICES_STATE_SECRET:
                index = 0
            else:
                match = re.fullmatch(
                    rf"{re.escape(self.LEGACY_SERVICES_STATE_SECRET)}-part-(\d+)",
                    name,
                )
                if match is None:
                    raise RuntimeError(f"Unexpected services state Secret {name}")
                index = int(match.group(1))
            encoded = secret.get("data", {}).get("tfstate")
            if not encoded:
                raise RuntimeError(f"Services state Secret {name} has no state")
            return index, base64.b64decode(encoded)

        chunks = sorted(map(chunk, secrets))
        if [index for index, _ in chunks] != list(range(len(chunks))):
            raise RuntimeError("Services state Secret chunks are incomplete")
        return json.loads(gzip.decompress(b"".join(data for _, data in chunks)))

    def _legacy_migration_marker(
        self,
        kubernetes_config_path: str,
        kubernetes_config_context: str,
    ) -> Tuple[Optional[str], Optional[int]]:
        output = execute_cmd(
            [
                self.os_artifacts.kubectl,
                "--kubeconfig",
                kubernetes_config_path,
                "--context",
                kubernetes_config_context,
                "--namespace",
                "default",
                "get",
                "lease",
                self.LEGACY_SERVICES_STATE_LOCK,
                "--output",
                "json",
            ],
            error_string="Failed to inspect services state migration",
            censor_output=True,
            subprocess_log_level="debug",
        )
        annotations = json.loads(output).get("metadata", {}).get(
            "annotations", {}
        )
        lineage = annotations.get(self.LEGACY_MIGRATION_LINEAGE)
        serial = annotations.get(self.LEGACY_MIGRATION_SERIAL)
        return lineage, int(serial) if serial is not None else None

    def _mark_legacy_services_state_migrated(
        self,
        kubernetes_config_path: str,
        kubernetes_config_context: str,
        state: Dict[str, Any],
    ) -> None:
        execute_cmd(
            [
                self.os_artifacts.kubectl,
                "--kubeconfig",
                kubernetes_config_path,
                "--context",
                kubernetes_config_context,
                "--namespace",
                "default",
                "annotate",
                "lease",
                self.LEGACY_SERVICES_STATE_LOCK,
                f"{self.LEGACY_MIGRATION_LINEAGE}={state['lineage']}",
                f"{self.LEGACY_MIGRATION_SERIAL}={state.get('serial', 0)}",
                "--overwrite",
            ],
            error_string="Failed to record services state migration",
            censor_output=True,
            subprocess_log_level="debug",
        )

    def _push_state(
        self,
        working_directory: str,
        state: Dict[str, Any],
    ) -> None:
        state_file = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.os_artifacts.private_config_dir,
            suffix=".tfstate",
            delete=False,
        )
        try:
            secure_path(Path(state_file.name), 0o600)
            json.dump(state, state_file)
            state_file.flush()
            os.fsync(state_file.fileno())
            state_file.close()
            execute_cmd(
                [
                    self.os_artifacts.terraform,
                    f"-chdir={working_directory}",
                    "state",
                    "push",
                    state_file.name,
                ],
                check_return_code=True,
                check_empty_result=False,
                error_string="Failed to migrate services state to Azure Blob",
                capture_output=True,
                censor_command=True,
                censor_output=True,
                env_vars={
                    "ARM_ENVIRONMENT": self.environment,
                    "TF_PLUGIN_CACHE_DIR": self.PLUGIN_CACHE_DIR,
                },
            )
        finally:
            state_file.close()
            if os.path.exists(state_file.name):
                os.remove(state_file.name)

    def _delete_legacy_services_state(
        self,
        cluster_name: str,
        kubernetes_config_path: str,
        kubernetes_config_context: str,
    ) -> None:
        secrets = self._legacy_services_state_secrets(
            kubernetes_config_path,
            kubernetes_config_context,
        )
        kubectl = KubectlWrapper(
            self.os_artifacts,
            cluster_name,
            config_context=kubernetes_config_context,
        )
        with kubectl.context():
            for secret in sorted(
                secrets,
                key=lambda item: (
                    item["metadata"]["name"]
                    != self.LEGACY_SERVICES_STATE_SECRET,
                    item["metadata"]["name"],
                ),
            ):
                kubectl.delete(
                    "secret",
                    secret["metadata"]["name"],
                    ignore_not_found=True,
                    namespace="default",
                )

    def init(
        self,
        working_directory: str,
        refresh_creds: bool = True,
        backend_config: Dict[str, str] = {},
        cleanup_state: bool = False,
    ):
        log(f"Initializing OpenTofu in {working_directory}")
        if refresh_creds:
            assert self.az is not None, "AzureCliWrapper must be provided to refresh credentials"
            self.az.refresh_az_creds()

        if cleanup_state:
            log(f"Cleaning up state in {working_directory}", level="debug")
            shutil.rmtree(os.path.join(working_directory, ".terraform"), ignore_errors=True)

        command = [
            self.os_artifacts.terraform,
            f"-chdir={working_directory}",
            "init",
            "-upgrade",
            "-force-copy",
        ]

        env_vars = {
            "ARM_ENVIRONMENT": self.environment,
            "TF_PLUGIN_CACHE_DIR": self.PLUGIN_CACHE_DIR,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            if backend_config:
                f = tempfile.NamedTemporaryFile(mode="w", dir=temp_dir, delete=False)
                contents = "\n".join([f'{k} = "{v}"' for k, v in backend_config.items()])
                if on_windows():
                    log(
                        (
                            "We're on Windows, replacing backslashes in backend file "
                            f"{f.name} with forward slashes"
                        ),
                        "debug",
                    )
                    contents = contents.replace("\\", "/")
                f.write(contents)
                f.close()
                command += [f"-backend-config={f.name}"]

            execute_cmd(
                command,
                True,
                False,
                f"Failed to initialize OpenTofu in {working_directory}",
                env_vars=env_vars,
            )

    def ensure_resource_group(
        self,
        tenant_id: str,
        subscription_id: str,
        region: str,
        cluster_name: str,
        resource_group_name: str,
    ):
        rg_directory = os.path.join(self.os_artifacts.aks_directory, "modules", "rg")
        self.init(rg_directory)
        variables = {
            "tenantId": tenant_id,
            "subscriptionId": subscription_id,
            "location": region,
            "prefix": cluster_name,
            "resource_group_name": resource_group_name,
        }
        state_file = self.os_artifacts.get_terraform_file(
            "rg.tfstate", cluster_name, resource_group_name
        )
        log("Creating resource group if necessary...")
        try:
            with open(os.devnull, "w") as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    self.apply(rg_directory, state_file, variables)
            created_rg = True
        except Exception:
            log("Resource group already exists. Continuing...")
            created_rg = False
        return created_rg

    def ensure_infra(
        self,
        tenant_id: str,
        subscription_id: str,
        region: str,
        cluster_name: str,
        resource_group: str,
        worker_nodes: int,
        storage_name: str,
        container_name: str,
        storage_access_key: str,
        enable_telemetry: bool,
        cleanup_state: bool = False,
        after_init: Optional[Callable[[], None]] = None,
    ):
        infra_directory = os.path.join(self.os_artifacts.aks_directory, "modules", "infra")
        log("Executing OpenTofu to build out infrastructure (this may take up to 30 minutes)...")
        backend_config = {
            "storage_account_name": storage_name,
            "resource_group_name": resource_group,
            "container_name": container_name,
            "access_key": storage_access_key,
        }

        self.init(
            infra_directory,
            backend_config=backend_config,
            cleanup_state=cleanup_state,
            refresh_creds=True,
        )
        if after_init is not None:
            after_init()
        variables = {
            "tenantId": tenant_id,
            "subscriptionId": subscription_id,
            "location": region,
            "prefix": cluster_name,
            "kubeconfig_location": self.os_artifacts.config_dir,
            "max_worker_nodes": worker_nodes,
            "enable_telemetry": f"{'true' if enable_telemetry else 'false'}",
            "resource_group_name": resource_group,
        }

        state_file = self.os_artifacts.get_terraform_file(
            self.INFRA_STATE_FILE, cluster_name, resource_group
        )
        with tempfile.NamedTemporaryFile(delete=False) as plan_file:
            plan = self.plan(infra_directory, state_file, variables, plan_file=plan_file.name)
            replacements = self._get_replacements(plan)
            if replacements:
                log(
                    f"OpenTofu plan requires replacement of resources {', '.join(replacements)}..."
                )
                proceed = True
                if self._has_storage_replacement(replacements):
                    proceed = verify_to_proceed(
                        "\nCluster storage is being replaced. "
                        f"{ColorFormatter.red}This will result in data loss!!!"
                        f"{ColorFormatter.reset} Please backup your data before proceeding. "
                        "Would you like to continue?"
                    )
                else:
                    proceed = verify_to_proceed(
                        f"Some resources ({', '.join(replacements)}) will be replaced, "
                        "but your data should be safe. Would you like to continue?"
                    )
                if not proceed:
                    raise RuntimeError("Cancelation Requested")
                else:
                    log("Continuing with OpenTofu apply...")
            self.apply(infra_directory, state_file, variables, plan_file=plan_file.name)
            return self.get_output(infra_directory, state_file)

    def ensure_k8s_cluster(
        self,
        cluster_name: str,
        tenant_id: str,
        registry_path: str,
        registry_username: str,
        registry_password: str,
        resource_group: str,
        current_user_name: str,
        certificate_email: str,
        kubernetes_config_context: str,
        public_ip_address: str,
        public_ip_fqdn: str,
        public_ip_dns: str,
        keyvault_name: str,
        application_id: str,
        storage_connection_key: str,
        storage_account_name: str,
        userfile_container_name: str,
        monitor_instrumentation_key: str,
        backend_storage_name: str,
        backend_container_name: str,
        backend_storage_access_key: str,
        enable_telemetry: bool,
        cleanup_state: bool = False,
        migrate_legacy_services: bool = False,
        on_legacy_destroy: Optional[Callable[[], None]] = None,
        before_apply: Optional[Callable[[], None]] = None,
    ):
        # Do kubernetes infra now
        kubernetes_directory = os.path.join(
            self.os_artifacts.aks_directory, "modules", "kubernetes"
        )
        backend_config = {
            "storage_account_name": backend_storage_name,
            "resource_group_name": resource_group,
            "container_name": backend_container_name,
            "access_key": backend_storage_access_key,
        }
        self.init(
            kubernetes_directory,
            backend_config=backend_config,
            cleanup_state=cleanup_state,
            refresh_creds=True,
        )
        variables = {
            "tenantId": tenant_id,
            "namespace": "default",
            "acr_registry": registry_path,
            "acr_registry_username": registry_username,
            "acr_registry_password": registry_password,
            "kubernetes_config_path": self.os_artifacts.config_file("kubeconfig"),
            "kubernetes_config_context": kubernetes_config_context,
            "public_ip_address": public_ip_address,
            "public_ip_fqdn": public_ip_fqdn,
            "public_ip_dns": public_ip_dns,
            "keyvault_name": keyvault_name,
            "application_id": application_id,
            "storage_connection_key": storage_connection_key,
            "storage_account_name": storage_account_name,
            "userfile_container_name": userfile_container_name,
            "monitor_instrumentation_key": monitor_instrumentation_key,
            "resource_group_name": resource_group,
            "current_user_name": current_user_name,
            "certificate_email": certificate_email,
            "enable_telemetry": str(enable_telemetry).lower(),
            "redis_image": REDIS_IMAGE,
            "rabbitmq_image": RABBITMQ_IMAGE,
        }

        state_file = self.os_artifacts.get_terraform_file(
            "kubernetes.tfstate", cluster_name, resource_group
        )
        if migrate_legacy_services:
            self.destroy_legacy_service_charts(
                kubernetes_directory,
                state_file,
                variables,
                cluster_name,
                kubernetes_config_context,
                on_legacy_destroy,
            )
        if before_apply is not None:
            before_apply()
        self.apply(kubernetes_directory, state_file, variables)

        return self.get_output(kubernetes_directory, state_file)

    def ensure_services(
        self,
        cluster_name: str,
        resource_group: str,
        registry_path: str,
        kubernetes_config_path: str,
        kubernetes_config_context: str,
        worker_node_pool_name: str,
        public_ip_fqdn: str,
        image_prefix: str,
        image_tag: str,
        shared_resource_pv_claim_name: str,
        otel_service_name: str,
        worker_replicas: int,
        log_level: str,
        backend_storage_name: str,
        backend_container_name: str,
        backend_storage_access_key: str,
        cleanup_state: bool = False,
        migrate_state: bool = False,
    ):
        services_directory = os.path.join(self.os_artifacts.aks_directory, "..", "services")
        backend_config = {
            "storage_account_name": backend_storage_name,
            "resource_group_name": resource_group,
            "container_name": backend_container_name,
            "access_key": backend_storage_access_key,
        }
        migration_lock = (
            self._lock_legacy_services_state(
                kubernetes_config_path,
                kubernetes_config_context,
            )
            if migrate_state
            else nullcontext()
        )
        with migration_lock as verify_migration_lock:
            legacy_state: Dict[str, Any] = {}
            orphaned_legacy_chunks = False
            target_exists = False
            if migrate_state:
                assert self.az is not None
                target_exists = self.az.blob_exists(
                    backend_storage_name,
                    backend_storage_access_key,
                    backend_container_name,
                    self.SERVICES_STATE_FILE,
                )
                legacy_secrets = self._legacy_services_state_secrets(
                    kubernetes_config_path,
                    kubernetes_config_context,
                )
                legacy_names = {
                    secret.get("metadata", {}).get("name")
                    for secret in legacy_secrets
                }
                if self.LEGACY_SERVICES_STATE_SECRET in legacy_names:
                    legacy_state = self._pull_legacy_services_state(
                        kubernetes_config_path,
                        kubernetes_config_context,
                    )
                elif legacy_secrets:
                    orphaned_legacy_chunks = True
                if not target_exists and not legacy_state:
                    kubectl = KubectlWrapper(
                        self.os_artifacts,
                        cluster_name,
                        config_context=kubernetes_config_context,
                    )
                    with kubectl.context():
                        existing_services = kubectl.get_or_none(
                            "deployment",
                            "terravibes-rest-api",
                            namespace="default",
                        )
                    if existing_services is not None:
                        raise RuntimeError(
                            "Existing services have no recoverable OpenTofu state"
                        )
            self.init(
                services_directory,
                backend_config=backend_config,
                cleanup_state=cleanup_state,
                refresh_creds=True,
            )
            if legacy_state:
                migrated_state = (
                    self._pull_state(services_directory)
                    if target_exists
                    else {}
                )
                if not migrated_state:
                    if verify_migration_lock is not None:
                        verify_migration_lock()
                    self._push_state(services_directory, legacy_state)
                    migrated_state = self._pull_state(services_directory)
                if (
                    not migrated_state
                    or migrated_state.get("lineage")
                    != legacy_state.get("lineage")
                    or migrated_state.get("serial", 0)
                    < legacy_state.get("serial", 0)
                ):
                    raise RuntimeError(
                        "Services state migration verification failed"
                    )
                if verify_migration_lock is not None:
                    verify_migration_lock()
                self._mark_legacy_services_state_migrated(
                    kubernetes_config_path,
                    kubernetes_config_context,
                    migrated_state,
                )
                self._delete_legacy_services_state(
                    cluster_name,
                    kubernetes_config_path,
                    kubernetes_config_context,
                )
            elif orphaned_legacy_chunks:
                migrated_state = self._pull_state(services_directory)
                marker = self._legacy_migration_marker(
                    kubernetes_config_path,
                    kubernetes_config_context,
                )
                expected_marker = (
                    migrated_state.get("lineage"),
                    migrated_state.get("serial"),
                )
                if (
                    not expected_marker[0]
                    or expected_marker[1] is None
                    or marker[0] is None
                    or marker[1] is None
                    or marker != expected_marker
                ):
                    raise RuntimeError(
                        "Orphaned services state chunks cannot be verified"
                    )
                if verify_migration_lock is not None:
                    verify_migration_lock()
                self._delete_legacy_services_state(
                    cluster_name,
                    kubernetes_config_path,
                    kubernetes_config_context,
                )
        variables = {
            "namespace": "default",
            "prefix": cluster_name,
            "acr_registry": registry_path,
            "kubernetes_config_path": kubernetes_config_path,
            "kubernetes_config_context": kubernetes_config_context,
            "worker_node_pool_name": worker_node_pool_name,
            "public_ip_fqdn": public_ip_fqdn,
            "dapr_sidecars_deployed": True,
            "startup_type": "aks",
            "image_prefix": image_prefix,
            "image_tag": image_tag,
            "shared_resource_pv_claim_name": shared_resource_pv_claim_name,
            "otel_service_name": otel_service_name,
            "worker_replicas": worker_replicas,
            "farmvibes_log_level": log_level,
        }

        state_file = self.os_artifacts.get_terraform_file(
            self.SERVICES_STATE_FILE, cluster_name, resource_group
        )
        self.apply(services_directory, state_file, variables)

        return self.get_output(services_directory, state_file)

    def ensure_local_cluster(
        self,
        cluster_name: str,
        registry: str,
        log_level: str,
        max_log_file_bytes: Optional[int],
        log_backup_count: Optional[int],
        image_tag: str,
        image_prefix: str,
        data_path: str,
        worker_replicas: int,
        config_context: str,
        enable_telemetry: bool,
        redis_image: str = REDIS_IMAGE,
        rabbitmq_image: str = RABBITMQ_IMAGE,
        is_update: bool = False,
        max_full_history_runs: int = 100,
        max_compact_history_runs: int = 900,
    ):
        self.init(
            self.os_artifacts.local_directory,
            False,
            cleanup_state=not is_update,
        )
        variables: Dict[str, str] = {
            "acr_registry": registry,
            "run_as_user_id": f"{self.getuid()}",
            "run_as_group_id": f"{self.getgid()}",
            "host_assets_dir": os.path.join(data_path, "assets"),
            "kubernetes_config_context": config_context,
            "image_tag": image_tag,
            "node_pool_name": f"{cluster_name}",
            "host_storage_path": "/mnt",
            "worker_replicas": f"{worker_replicas}",
            "image_prefix": image_prefix,
            "redis_image": redis_image,
            "rabbitmq_image": rabbitmq_image,
            "enable_telemetry": f"{'true' if enable_telemetry else 'false'}",
            "farmvibes_log_level": log_level,
            "max_log_file_bytes": f"{max_log_file_bytes}" if max_log_file_bytes else "",
            "log_backup_count": f"{log_backup_count}" if log_backup_count else "",
            "max_full_history_runs": f"{max_full_history_runs}",
            "max_compact_history_runs": f"{max_compact_history_runs}",
        }

        state_file = self.os_artifacts.get_terraform_file("local.tfstate", cluster_name)
        self.apply(
            self.os_artifacts.local_directory,
            state_file,
            variables,
            refresh_creds=False,
        )
        return self.get_output(self.os_artifacts.local_directory, state_file, refresh_creds=False)

    def list_workspaces(self) -> List[str]:
        cmd = [self.os_artifacts.terraform, "workspace", "list"]
        error = "Couldn't list OpenTofu workspaces"
        return (
            execute_cmd(
                cmd,
                check_return_code=True,
                check_empty_result=True,
                error_string=error,
                subprocess_log_level="debug",
            )
            .replace("*", "")
            .split()
        )

    def get_workspace(self) -> str:
        cmd = [self.os_artifacts.terraform, "workspace", "show"]
        error = "Couldn't get OpenTofu workspace"
        return execute_cmd(
            cmd, True, True, error, capture_output=True, subprocess_log_level="debug"
        )

    def set_workspace(self, workspace: str):
        workspaces = self.list_workspaces()
        if workspace not in workspaces:
            log(f"OpenTofu workspace {workspace} does not exist. Creating it...", level="debug")
            cmd = [self.os_artifacts.terraform, "workspace", "new", workspace]
            error = f"Couldn't create OpenTofu workspace {workspace}"
            execute_cmd(
                cmd,
                check_return_code=False,
                check_empty_result=True,
                error_string=error,
                subprocess_log_level="debug",
            )
        else:
            log(f"OpenTofu workspace {workspace} already exists. Selecting it...", level="debug")

        cmd = [self.os_artifacts.terraform, "workspace", "select", workspace]
        error = f"Couldn't select OpenTofu workspace {workspace}"
        execute_cmd(cmd, True, False, error, capture_output=False, subprocess_log_level="debug")

    def delete_workspace(self, workspace: str):
        workspaces = self.list_workspaces()
        if workspace not in workspaces:
            log(
                f"OpenTofu workspace {workspace} does not exist. Nothing to delete...",
                level="debug",
            )
            return
        cmd = [self.os_artifacts.terraform, "workspace", "delete", workspace]
        error = f"Couldn't delete OpenTofu workspace {workspace}"
        try:
            execute_cmd(cmd, True, False, error, capture_output=False, subprocess_log_level="debug")
        except Exception as e:
            log(f"Couldn't delete OpenTofu workspace {workspace}: {e}", level="debug")

    @contextmanager
    def workspace(self, workspace_name: str):
        current_workspace = self.get_workspace()
        log(f"Current OpenTofu workspace is {current_workspace}", level="debug")
        log(f"Setting OpenTofu workspace to {workspace_name}", level="debug")
        if current_workspace != workspace_name:
            self.set_workspace(workspace_name)
        try:
            yield
        finally:
            if current_workspace != workspace_name:
                self.set_workspace(current_workspace)

    @staticmethod
    def getuid(default: int = 1000):
        if hasattr(os, "getuid"):
            return os.getuid()
        else:
            return default

    @staticmethod
    def getgid(default: int = 1000):
        if hasattr(os, "getgid"):
            return os.getgid()
        else:
            return default

    def get_infra_results(self, cluster_name: str, resource_group: str):
        try:
            with self.workspace(f"farmvibes-aks-{cluster_name}-{resource_group}"):
                state_file = self.os_artifacts.get_terraform_file(
                    self.INFRA_STATE_FILE, cluster_name, resource_group
                )
                infra_directory = os.path.join(self.os_artifacts.aks_directory, "modules", "infra")
                results = self.get_output(infra_directory, state_file)
                return results
        except Exception as e:
            log(f"Error getting infra results with OpenTofu: {e}", level="error")
            return {}

    def get_url_from_terraform_output(self, cluster_name: str, resource_group: str) -> str:
        results = self.get_infra_results(cluster_name, resource_group)
        if results:
            return f"https://{results['public_ip_fqdn']['value']}"
        return ""

    def get_kubernetes_config_context(self, cluster_name: str, resource_group: str) -> str:
        results = self.get_infra_results(cluster_name, resource_group)
        if results:
            return results["kubernetes_config_context"]["value"]
        return ""

    def _get_infra_state(self):
        try:
            assert self.az is not None, "Azure client not initialized"
            storage_name, container_name, key = self.az.ensure_azurerm_backend("")
            log(f"Getting OpenTofu state from {storage_name}/{container_name}")
            state = json.loads(
                self.az.download_blob(storage_name, container_name, self.INFRA_STATE_FILE, key=key)
            )
            return state
        except Exception as e:
            log(f"Error getting storage account name from OpenTofu state file: {e}", level="error")
            return {}

    def get_storage_account_name(self):
        state = self._get_infra_state()
        try:
            log("Extracting storage account name from OpenTofu state", level="debug")
            storage_account = state["outputs"]["storage_account_name"]["value"]
            return storage_account
        except Exception as e:
            log(f"Error getting storage account name from OpenTofu state: {e}", level="error")
            return ""

    def get_current_core_count(self) -> Tuple[int, int]:
        state = self._get_infra_state()
        try:
            log("Extracting current core count from OpenTofu state", level="debug")
            max_workers = int(state["outputs"]["max_worker_nodes"]["value"])
            max_default = int(state["outputs"]["max_default_nodes"]["value"])
            return (
                max_workers * CPUS_REQUIRED[WORKER_NODE_CPU_NAME],
                max_default * CPUS_REQUIRED[DEFAULT_NODE_CPU_NAME],
            )
        except Exception as e:
            log(f"Error getting current core count from OpenTofu state: {e}", level="error")
            return 0, 0


class AzureCliWrapper:
    def __init__(self, os_artifacts: OSArtifacts, cluster_name: str, resource_group: str = ""):
        self.os_artifacts = os_artifacts
        self.cluster_name = cluster_name
        self.resource_group = resource_group
        self.subscription_id, self.tenant_id = "", ""

    def cluster_exists(self, cluster_name: Optional[str] = None) -> bool:
        if cluster_name is None:
            cluster_name = self.cluster_name

        if not cluster_name:
            raise ValueError("No cluster name provided")

        cmd = [
            self.os_artifacts.az,
            "aks",
            "show",
            "-n",
            cluster_name,
            "-g",
            self.resource_group,
            "-o",
            "tsv",
        ]
        error = f"Unable to find cluster {cluster_name}"

        try:
            execute_cmd(
                cmd,
                True,
                check_empty_result=False,
                capture_output=True,
                error_string=error,
                subprocess_log_level="debug",
                log_error=False,
            )
            return True
        except Exception:
            return False

    def resource_group_exists(self, resource_group: str = "") -> bool:
        resource_group = resource_group or self.resource_group
        cmd = [self.os_artifacts.az, "group", "exists", "-n", resource_group]
        error = "Couldn't get info of group from azure"
        result = execute_cmd(cmd, True, True, error, subprocess_log_level="debug")
        return result.lower().strip() == "true"

    def list_resources(self, resource_group: str = "") -> List[Dict[str, Any]]:
        resource_group = resource_group or self.resource_group
        cmd = [self.os_artifacts.az, "resource", "list", "--resource-group", resource_group]
        error = "Failed to get group resources. Please try again later"
        existing_resources = execute_cmd(cmd, True, False, error, subprocess_log_level="debug")
        return json.loads(existing_resources)

    def delete_resources(self, resources: List[str], resource_group: str = ""):
        if not resources:
            log("No resources to delete", level="debug")
            return
        resource_group = resource_group or self.resource_group
        cmd = [
            self.os_artifacts.az,
            "resource",
            "delete",
            "--resource-group",
            resource_group,
            "--ids",
        ]
        cmd.extend(resources)
        error = f"Failed to delete resources {resources}. Please try again later"
        execute_cmd(cmd, True, False, error, subprocess_log_level="debug")

    def delete_resource_group(self, resource_group: str = ""):
        resource_group = resource_group or self.resource_group
        cmd = [self.os_artifacts.az, "group", "delete", "-n", resource_group, "-y"]
        error = "Failed to delete group. Please try again later"
        execute_cmd(cmd, True, False, error, subprocess_log_level="debug")

    def expand_azure_region(self, canonical_region: str) -> str:
        cmd = [
            self.os_artifacts.az,
            "account",
            "list-locations",
            "--query",
            f"[?name=='{canonical_region}'].displayName",
            "-o",
            "tsv",
        ]
        error = f"Couldn't get azure region. Maybe it is invalid {canonical_region}"

        return execute_cmd(cmd, True, True, error, subprocess_log_level="debug")

    def get_subscription_and_tenant_id(self) -> Tuple[str, str]:
        if self.subscription_id and self.tenant_id:
            return self.subscription_id, self.tenant_id
        try:
            # Verify Azure CLI is logged in and has a default subscription set
            self.subscription_id, self.tenant_id = self.get_subscription_info()
            return self.subscription_id, self.tenant_id
        except Exception as e:
            log_should_be_logged_in(e)
            raise

    def get_subscription_info(self, max_attempts: int = 2):
        for i in range(max_attempts):
            cmd = [self.os_artifacts.az, "account", "show", "-o", "json"]
            error = "Unable to get default subscription"
            sub_info = json.loads(execute_cmd(cmd, True, True, error, subprocess_log_level="debug"))
            log(f"Found {sub_info['name']} with id {sub_info['id']} as current subscription")

            proceed = verify_to_proceed(
                f"Is this the correct Azure subscription you would like to use? {sub_info['name']}"
            )
            if proceed:
                return sub_info["id"], sub_info["tenantId"]

            if i < max_attempts - 1:
                proceed = verify_to_proceed("Would you like to change now?")
                if proceed:
                    suggested_sub_id = input(
                        "Enter the Azure Subscription ID you would like to use: "
                    )
                    if suggested_sub_id:
                        cmd = [
                            self.os_artifacts.az,
                            "account",
                            "set",
                            "-s",
                            suggested_sub_id,
                        ]
                        execute_cmd(
                            cmd,
                            True,
                            False,
                            "Failed to set subscription",
                        )
                        log(f"Subscription set successfully to {suggested_sub_id}")
                    else:
                        break
                else:
                    break

        raise ValueError("Cancelation Requested")

    def refresh_az_creds(self):
        cmd = [self.os_artifacts.az, "account", "get-access-token"]
        error = "Unable to refresh Azure tokens"

        for _ in range(AZ_CREDS_REFRESH_ATTEMPTS):
            try:
                execute_cmd(cmd, True, True, error, censor_output=True)
                break
            except Exception:
                proceed = verify_to_proceed(
                    "It seems Azure has logged out.\n"
                    f"Please relogin on another prompt using {AZ_LOGIN_PROMPT} and continue here.\n"
                    "Ready to continue?"
                )
                if not proceed:
                    raise ValueError("Unable to get AZ Credentials.")

    def check_resource_providers(self, region: str):
        cmd = (
            f"{self.os_artifacts.az} provider show -n {{provider}} --query registrationState -o tsv"
        )
        status = {
            provider: execute_cmd(
                cmd.format(provider=provider).split(),
                True,
                True,
                f"Couldn't get registration status for {provider}",
                subprocess_log_level="debug",
            )
            for provider in AZURE_RESOURCES_REQUIRED
        }
        not_registered = [provider for provider, state in status.items() if state != REGISTERED]
        if any(not_registered):
            log(f"Resource providers not registered: {', '.join(not_registered)}. ")
            proceed = verify_to_proceed(
                "Would you like me to register them for you? "
                "You can also register them manually using `az provider register -n <provider>`"
            )
            if not proceed:
                log(
                    "User chose not to register the required providers. "
                    "Please register them manually and run the command again.",
                    level="warning",
                )
                return False

        registered = self.register_providers(not_registered)
        if not all(registered):
            not_registered = [
                provider for provider, reg in zip(not_registered, registered) if not reg
            ]
            log(
                f"Some providers ({' '.join(not_registered)}) were not registered. "
                "Please register them manually and try again.",
                level="error",
            )
            return False
        return True

    def register_providers(self, providers: List[str]):
        if not providers:
            return []
        with ThreadPoolExecutor(max_workers=len(providers)) as executor:
            registered = executor.map(self.register_provider, providers)
        return registered

    def register_provider(self, provider: str, max_tries: int = 60, wait_s: int = 10):
        error = f'Unable to register provider "{provider}". You might have to register it manually.'
        cmd = [
            self.os_artifacts.az,
            "provider",
            "register",
            "-n",
            provider,
        ]
        execute_cmd(cmd, True, True, error, subprocess_log_level="debug")
        tries = 0
        registered = False
        cmd = [
            self.os_artifacts.az,
            "provider",
            "show",
            "-n",
            provider,
            "--query",
            "registrationState",
            "-o",
            "tsv",
        ]
        while not registered and tries < max_tries:
            result = execute_cmd(cmd, True, True, error, subprocess_log_level="debug")
            registered = result == REGISTERED
            tries += 1
            if registered:
                break
            log(
                f"Waiting for provider {provider} to register. Try {tries}/{max_tries}",
                level="debug",
            )
            time.sleep(wait_s)
        if tries >= max_tries:
            log(error, "warning")
        return registered

    def verify_enough_cores_available(
        self,
        region: str,
        worker_nodes: int = 1,
        current_worker_cores: int = 0,
        current_default_cores: int = 0,
    ):
        if worker_nodes > 0:
            worker_cpus_per_node = CPUS_REQUIRED[WORKER_NODE_CPU_NAME]
            worker_cpus_needed = max(worker_cpus_per_node * worker_nodes - current_worker_cores, 0)
            CPUS_REQUIRED[WORKER_NODE_CPU_NAME] = worker_cpus_needed
            CPUS_REQUIRED[TOTAL_REGIONAL_CPU_NAME] = (
                CPUS_REQUIRED[TOTAL_REGIONAL_CPU_NAME] - worker_cpus_per_node + worker_cpus_needed
            )

        for cpu_type in CPUS_REQUIRED.keys():
            if cpu_type == DEFAULT_NODE_CPU_NAME:
                required = max(CPUS_REQUIRED[cpu_type] - current_default_cores, 0)
            else:
                required = CPUS_REQUIRED[cpu_type]
            log(f"Validating that {cpu_type} has enough resources in region {region}")

            command = [
                self.os_artifacts.az,
                "vm",
                "list-usage",
                "--location",
                region,
                "--output",
                "json",
                "--query",
                f"[?localName=='{cpu_type}']",
            ]
            error = f"{cpu_type} wasn't available or not parsable"

            result = execute_cmd(command, True, True, error, subprocess_log_level="debug")

            vm_usage = json.loads(result)[0]
            current_usage = int(vm_usage["currentValue"])
            total_allowed = int(vm_usage["limit"])
            available = total_allowed - current_usage

            if required > available:
                raise ValueError(f"{cpu_type} has {available} CPUs. We need {required}.")

    def request_registry_token(self, registry: str) -> str:
        """Requests an access token for a given registry using the az CLI.

        Args:
            registry: the name of the registry under Azure we want to connect to.
        """
        log(f"Getting token credentials for {registry}")
        registry = registry.replace(".azurecr.io", "")  # FIXME: This only works for Azure Public

        self.refresh_az_creds()
        token_command = [
            self.os_artifacts.az,
            "acr",
            "login",
            "-n",
            registry,
            "--expose-token",
        ]
        error = f"Unable to get credentials for {registry}"
        output = json.loads(execute_cmd(token_command, True, True, error, censor_output=True))
        return output["accessToken"] if "accessToken" in output else ""

    def get_storage_account_list(self):
        cmd = [
            self.os_artifacts.az,
            "storage",
            "account",
            "list",
            "--resource-group",
            self.resource_group,
            "-o",
            "json",
        ]

        error = "Couldn't get storage account list. Do you have access to the resource group?"
        results = execute_cmd(cmd, True, False, error, subprocess_log_level="debug")
        accounts = json.loads(results)
        return accounts

    def create_storage_account(self, location: str, storage_name: str):
        cmd = [
            self.os_artifacts.az,
            "storage",
            "account",
            "create",
            "--name",
            storage_name,
            "--location",
            location,
            "--resource-group",
            self.resource_group,
        ]
        error = "Couldn't create storage account. Do you have access to the resource group?"
        try:
            execute_cmd(cmd, True, False, error, subprocess_log_level="debug")
        except Exception:
            return False
        return True

    def harden_storage_account(self, storage_name: str) -> None:
        execute_cmd(
            [
                self.os_artifacts.az,
                "storage",
                "account",
                "blob-service-properties",
                "update",
                "--account-name",
                storage_name,
                "--resource-group",
                self.resource_group,
                "--enable-versioning",
                "true",
                "--enable-delete-retention",
                "true",
                "--delete-retention-days",
                "14",
                "--enable-container-delete-retention",
                "true",
                "--container-delete-retention-days",
                "14",
            ],
            check_return_code=True,
            check_empty_result=False,
            error_string="Couldn't harden the OpenTofu state storage account",
            capture_output=True,
            subprocess_log_level="debug",
        )

    def get_storage_account_key(self, storage_name: str):
        cmd = [
            self.os_artifacts.az,
            "storage",
            "account",
            "keys",
            "list",
            "-g",
            self.resource_group,
            "--account-name",
            storage_name,
            "-o",
            "json",
            "--only-show-errors",
        ]
        error = "Couldn't get storage account keys. Do you have access to the resource group?"
        results = execute_cmd(cmd, True, False, error, censor_output=True)
        keys = json.loads(results)
        if isinstance(keys, dict):
            keys = keys["keys"]
        key = keys[0]["value"]
        return key

    def ensure_container_exists(self, storage_name: str, key: str, container_name: str) -> bool:
        cmd = [
            self.os_artifacts.az,
            "storage",
            "container",
            "exists",
            "--account-name",
            storage_name,
            "--account-key",
            key,
            "--name",
            container_name,
            "-o",
            "json",
        ]
        error = "Couldn't check if container exists. Do you have access to the storage account?"

        try:
            results = json.loads(
                execute_cmd(
                    cmd,
                    True,
                    False,
                    error,
                    subprocess_log_level="debug",
                    censor_output=True,
                    censor_command=True,
                )
            )
        except Exception as e:
            log(f"Error checking if container exists: {e}", level="error")
            return False

        if not results["exists"]:
            cmd = [
                self.os_artifacts.az,
                "storage",
                "container",
                "create",
                "--account-name",
                storage_name,
                "--account-key",
                key,
                "--name",
                container_name,
            ]
            error = "Couldn't create container. Do you have access to the storage account?"
            try:
                execute_cmd(
                    cmd, True, False, error, subprocess_log_level="debug", censor_command=True
                )
            except Exception:
                return False

        return True

    def blob_exists(
        self,
        storage_name: str,
        key: str,
        container_name: str,
        blob_name: str,
    ) -> bool:
        output = execute_cmd(
            [
                self.os_artifacts.az,
                "storage",
                "blob",
                "exists",
                "--account-name",
                storage_name,
                "--account-key",
                key,
                "--container-name",
                container_name,
                "--name",
                blob_name,
                "--output",
                "json",
            ],
            check_return_code=True,
            check_empty_result=True,
            error_string=f"Couldn't check OpenTofu state blob {blob_name}",
            censor_command=True,
            censor_output=True,
        )
        return bool(json.loads(output)["exists"])

    def download_blob(
        self,
        account_name: str,
        container_name: str,
        blob_name: str,
        file_path: str = "",
        key: str = "",
    ):
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = file_path or os.path.join(temp_dir, os.path.basename(blob_name))
            cmd = [
                self.os_artifacts.az,
                "storage",
                "blob",
                "download",
                "--account-name",
                account_name,
                "--container-name",
                container_name,
                "--name",
                blob_name,
                "--file",
                tmp_path,
            ]
            if key:
                cmd.extend(["--account-key", key])
            error = "Couldn't download blob. Do you have access to the storage account?"
            try:
                execute_cmd(
                    cmd,
                    True,
                    False,
                    error,
                    subprocess_log_level="debug",
                    censor_command=True if key else False,
                )
                if file_path:
                    return ""
                with open(tmp_path, "r") as f:
                    return f.read()
            except Exception:
                return ""

    def get_current_user_name(self) -> str:
        cmd = [self.os_artifacts.az, "account", "show", "-o", "json"]
        error = "Unable to get current user name"

        result = execute_cmd(cmd, True, True, error, subprocess_log_level="debug")
        return json.loads(result)["user"]["name"]

    def is_file_in_mount(self) -> bool:
        return "/mnt/" in self.os_artifacts.az

    def refresh_aks_credentials(self):
        self.refresh_az_creds()

        if not self.cluster_exists():
            log("Cluster does not exist. Please create it first.", level="error")
            return False

        cmd = [
            self.os_artifacts.az,
            "aks",
            "get-credentials",
            "--admin",
            "--name",
            self.cluster_name,
            "--resource-group",
            self.resource_group,
            "--file",
            self.os_artifacts.config_file("kubeconfig"),
            "--overwrite-existing",
        ]

        error = "Couldn't get kubernetes credentials. Do you have access to the cluster?"
        execute_cmd(cmd, True, False, error, subprocess_log_level="debug")
        secure_path(Path(self.os_artifacts.config_file("kubeconfig")), 0o600)

    def get_kubernetes_version(self) -> str:
        return execute_cmd(
            [
                self.os_artifacts.az,
                "aks",
                "show",
                "--name",
                self.cluster_name,
                "--resource-group",
                self.resource_group,
                "--query",
                "kubernetesVersion",
                "-o",
                "tsv",
            ],
            check_return_code=True,
            check_empty_result=True,
            error_string="Couldn't get AKS Kubernetes version",
            capture_output=True,
        ).strip()

    def ensure_kubernetes_version(
        self, minimum_version: Tuple[int, int] = (1, 32)
    ) -> None:
        current = self.get_kubernetes_version()
        while tuple(map(int, current.split(".")[:2])) < minimum_version:
            output = execute_cmd(
                [
                    self.os_artifacts.az,
                    "aks",
                    "get-upgrades",
                    "--name",
                    self.cluster_name,
                    "--resource-group",
                    self.resource_group,
                    "--query",
                    "controlPlaneProfile.upgrades[].kubernetesVersion",
                    "-o",
                    "json",
                ],
                error_string="Couldn't get supported AKS upgrades",
                subprocess_log_level="debug",
            )
            candidates = [
                version
                for version in json.loads(output)
                if tuple(map(int, version.split(".")))
                > tuple(map(int, current.split(".")))
            ]
            if not candidates:
                raise RuntimeError(
                    f"AKS {current} cannot be upgraded to support AzureLinux3"
                )
            next_minor = min(
                tuple(map(int, version.split(".")[:2]))
                for version in candidates
            )
            target = max(
                (
                    version
                    for version in candidates
                    if tuple(map(int, version.split(".")[:2]))
                    == next_minor
                ),
                key=lambda version: tuple(map(int, version.split("."))),
            )
            log(f"Upgrading AKS from {current} to {target}")
            command = [
                self.os_artifacts.az,
                "aks",
                "upgrade",
                "--name",
                self.cluster_name,
                "--resource-group",
                self.resource_group,
                "--kubernetes-version",
                target,
                "--yes",
            ]
            if next_minor >= minimum_version:
                command.insert(-1, "--control-plane-only")
            execute_cmd(
                command,
                check_empty_result=False,
                error_string=f"Couldn't upgrade AKS to {target}",
                subprocess_log_level="debug",
            )
            current = self.get_kubernetes_version()

    def ensure_azurerm_backend(
        self,
        location: str,
        container_name: str = "terraform-state",
    ) -> Tuple[str, str, str]:
        accounts = self.get_storage_account_list()
        storage_name = self.storage_name
        if not any(account["name"] == storage_name for account in accounts):
            self.create_storage_account(location, storage_name)
        self.harden_storage_account(storage_name)
        key = self.get_storage_account_key(storage_name)
        if not self.ensure_container_exists(storage_name, key, container_name):
            log("Couldn't create storage container for OpenTofu backend.", level="error")
            return "", "", ""
        return storage_name, container_name, key

    @property
    def storage_name(self) -> str:
        hash = hashlib.sha256((self.cluster_name + self.resource_group).encode("utf-8")).hexdigest()
        base = "azurerm"
        return f"{base}{hash[:MAXIMUM_STORAGE_ACCOUNT_NAME_LENGTH-len(base)]}"

    def get_storage_account_connection_string(self, storage_account: str):
        cmd = [
            self.os_artifacts.az,
            "storage",
            "account",
            "show-connection-string",
            "--name",
            storage_account,
            "--resource-group",
            self.resource_group,
            "--query",
            "connectionString",
            "--output",
            "tsv",
        ]

        connection_string = execute_cmd(
            cmd, True, True, "Couldn't get connection string from storage", censor_output=True
        )
        return connection_string

    def upload_file(self, file_path: str, connection_string: str, file_name: str = ""):
        file_name = file_name or os.path.basename(file_path)
        cmd = [
            self.os_artifacts.az,
            "storage",
            "blob",
            "upload",
            "--connection-string",
            connection_string,
            "--container-name",
            "user-files",
            "--type",
            "block",
            "--overwrite",
            "--name",
            file_name,
            "--file",
            file_path,
        ]

        execute_cmd(
            cmd,
            True,
            False,
            "Failed to upload file",
            subprocess_log_level="debug",
            censor_command=True,
        )
        log(f"Uploaded file {file_name} successfully")


class KubectlWrapper:
    def __init__(self, os_artifacts: OSArtifacts, cluster_name: str = "", config_context: str = ""):
        self.os_artifacts = os_artifacts
        self.cluster_name = cluster_name
        self.config_context = config_context

    def url_from_ingress(self, cluster_name: str):
        with self.context(cluster_name):
            try:
                cmd = [
                    self.os_artifacts.kubectl,
                    "get",
                    "ingress",
                    "terravibes-rest-api-ingress",
                    "-o",
                    'jsonpath="{.spec.rules[0].host}"',
                ]
                error = "Couldn't get ingress hostname from kubernetes"
                url = (
                    f"https://{execute_cmd(cmd, True, False, error, subprocess_log_level='debug')}"
                )
                return url
            except Exception as e:
                log(f"Error getting URL with kubectl: {e}", level="error")
                return ""

    def _actual_cluster_name(self, cluster_name: str = "") -> str:
        cluster_name = cluster_name or self.cluster_name
        if not cluster_name:
            raise ValueError("No cluster name provided")
        return cluster_name

    def list_pods(self, cluster_name: str = "") -> Dict[str, Any]:
        with self.context(cluster_name):
            log("Checking if redis master pod exists")
            cmd = [self.os_artifacts.kubectl, "get", "pods", "-o", "json"]
            result = execute_cmd(
                cmd, error_string="Unable to list pods", subprocess_log_level="debug"
            )

        pods = json.loads(result)
        return pods

    @contextmanager
    def context(self, cluster_name: str = ""):
        cluster_name = self._actual_cluster_name(cluster_name)
        context_name = self.config_context or CONFIG_CONTEXT.format(cluster_name=cluster_name)
        with self.os_artifacts.kube_context(context_name):
            yield

    @property
    def context_name(self, cluster_name: str = "") -> str:
        cluster_name = self._actual_cluster_name(cluster_name)
        return self.config_context or CONFIG_CONTEXT.format(cluster_name=cluster_name)

    def scale(
        self, kind: str, name: str, replicas: int = 0, cluster_name: str = "", timeout_s: int = 30
    ):
        cluster_name = self._actual_cluster_name(cluster_name)
        cmd = [
            self.os_artifacts.kubectl,
            "scale",
            "--timeout",
            f"{timeout_s}s",
            "--replicas",
            str(replicas),
            kind,
            name,
        ]
        execute_cmd(
            cmd,
            error_string=f"Unable to scale {kind} {name} to {replicas}",
            subprocess_log_level="debug",
        )

    def create_redis_volume_pod(
        self, cluster_name: str = "", redis_image: str = REDIS_IMAGE
    ):
        cluster_name = self._actual_cluster_name(cluster_name)
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "redis-vol-pod.yaml"), "w") as fp:
                fp.write(REDIS_VOL_POD_YAML.format(redis_image=redis_image))
            cmd = [self.os_artifacts.kubectl, "apply", "-f", fp.name]
            execute_cmd(
                cmd,
                error_string="Unable to create redis volume pod to backup data",
                subprocess_log_level="debug",
            )

        # Wait for the pod to be ready:
        cmd = [
            self.os_artifacts.kubectl,
            "wait",
            "--for=condition=Ready",
            "--timeout=120s",
            "pod/redisvolpod",
        ]
        execute_cmd(
            cmd,
            error_string="Unable to wait for redis volume pod to be ready",
            subprocess_log_level="debug",
        )

        return True

    def exec(
        self,
        pod: str,
        command: List[str],
        cluster_name: str = "",
        capture_output: bool = True,
        censor_command: bool = False,
    ):
        cluster_name = self._actual_cluster_name(cluster_name)
        cmd = [self.os_artifacts.kubectl, "exec", pod, "--"] + command
        error_string = (
            f"Unable to execute command on pod {pod}"
            if censor_command
            else f"Unable to execute command {command} on pod {pod}"
        )
        result = execute_cmd(
            cmd,
            error_string=error_string,
            capture_output=capture_output,
            censor_command=censor_command,
            subprocess_log_level="debug",
        )
        return result

    def cp(self, source: str, destination: str, cluster_name: str = ""):
        cluster_name = self._actual_cluster_name(cluster_name)
        cmd = [self.os_artifacts.kubectl, "cp", source, destination]
        execute_cmd(
            cmd,
            capture_output=False,
            check_empty_result=False,
            error_string=f"Unable to copy {source} to {destination}",
            subprocess_log_level="debug",
        )

    def delete(
        self,
        kind: str,
        name: str,
        cluster_name: str = "",
        ignore_not_found: bool = False,
        namespace: str = "",
        wait: bool = True,
    ):
        cluster_name = self._actual_cluster_name(cluster_name)
        cmd = [self.os_artifacts.kubectl, "delete", kind, name]
        if namespace:
            cmd.extend(["--namespace", namespace])
        if ignore_not_found:
            cmd.append("--ignore-not-found=true")
        if not wait:
            cmd.append("--wait=false")
        execute_cmd(
            cmd,
            check_empty_result=False,
            error_string=f"Unable to delete {kind} {name}",
            subprocess_log_level="debug",
        )

    def get_secret(self, name: str, key: str, cluster_name: str = ""):
        cluster_name = self._actual_cluster_name(cluster_name)
        context_name = self.config_context or CONFIG_CONTEXT.format(
            cluster_name=cluster_name
        )
        cmd = [
            self.os_artifacts.kubectl,
            "--context",
            context_name,
            "get",
            "secret",
            name,
            "-o",
            f'jsonpath="{{{key}}}"',
        ]
        result = execute_cmd(
            cmd,
            error_string=f"Unable to get secret {name}",
            censor_output=True,
            subprocess_log_level="debug",
        )
        return json.loads(result)

    def get_secret_or_none(self, name: str) -> Optional[Dict[str, Any]]:
        result = execute_cmd(
            [
                self.os_artifacts.kubectl,
                "--context",
                self.context_name,
                "get",
                "secret",
                name,
                "-o",
                "json",
                "--ignore-not-found=true",
            ],
            check_empty_result=False,
            error_string=f"Unable to get secret {name}",
            censor_output=True,
            subprocess_log_level="debug",
        )
        return json.loads(result) if result else None

    def create_docker_token(self, token_name: str, registry: str, username: str, token: str):
        """Add a secret to the kubernetes cluster.

        Args:
            token_name: The name of the token to be added to the cluster
            registry: The (Azure Container) registry this token is for
            username: The user name to use to connect to the registry
            token: The token to use.
        """
        cmd = [
            self.os_artifacts.kubectl,
            "--context",
            self.context_name,
            "create",
            "secret",
            "docker-registry",
            token_name,
            f"--docker-server={registry}",
            f"--docker-username={username}",
            f"--docker-password={token}",
            f"--docker-email={username}",
        ]
        execute_cmd(
            cmd,
            error_string="Unable to create acr token",
            censor_command=True,
            subprocess_log_level="debug",
        )

    def apply_docker_config_secret(self, token_name: str, docker_config: str):
        manifest = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": token_name},
            "type": "kubernetes.io/dockerconfigjson",
            "data": {".dockerconfigjson": docker_config},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as manifest_file:
            json.dump(manifest, manifest_file)
            manifest_file.flush()
            execute_cmd(
                [
                    self.os_artifacts.kubectl,
                    "--context",
                    self.context_name,
                    "apply",
                    "-f",
                    manifest_file.name,
                ],
                error_string="Unable to restore registry credentials",
                censor_command=True,
                censor_output=True,
                subprocess_log_level="debug",
            )

    def upsert_opaque_secret(self, name: str, data: Dict[str, str]) -> None:
        manifest = {
            "apiVersion": "v1",
            "kind": "Secret",
            "metadata": {"name": name},
            "type": "Opaque",
            "data": {
                key: base64.b64encode(value.encode()).decode()
                for key, value in data.items()
            },
        }
        descriptor, manifest_path = tempfile.mkstemp(
            dir=self.os_artifacts.private_config_dir,
            prefix=".secret-",
            suffix=".json",
        )
        try:
            secure_path(Path(manifest_path), 0o600)
            manifest_file = os.fdopen(descriptor, "w")
            descriptor = -1
            with manifest_file:
                json.dump(manifest, manifest_file)
                manifest_file.flush()
                os.fsync(manifest_file.fileno())
            execute_cmd(
                [
                    self.os_artifacts.kubectl,
                    "--context",
                    self.context_name,
                    "apply",
                    "-f",
                    manifest_path,
                ],
                error_string=f"Unable to update secret {name}",
                censor_command=True,
                censor_output=True,
                subprocess_log_level="debug",
            )
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if os.path.exists(manifest_path):
                os.remove(manifest_path)

    def add_secret(self, secret_name: str, secret_value: str):
        cmd = [
            self.os_artifacts.kubectl,
            "--context",
            self.context_name,
            "create",
            "secret",
            "generic",
            secret_name,
            f"--from-literal={secret_name}={secret_value}",
        ]
        execute_cmd(
            cmd,
            check_empty_result=False,
            capture_output=False,
            censor_command=True,
            subprocess_log_level="debug",
        )
        return True

    def delete_secret(self, secret_name: str):
        cmd = [
            self.os_artifacts.kubectl,
            "--context",
            self.context_name,
            "delete",
            "secret",
            secret_name,
        ]
        execute_cmd(
            cmd,
            check_empty_result=False,
            capture_output=False,
            censor_command=True,
            subprocess_log_level="debug",
        )
        return True

    def get_cluster_uid(self) -> str:
        cmd = [
            self.os_artifacts.kubectl,
            "--context",
            self.context_name,
            "get",
            "namespace",
            "kube-system",
            "-o",
            "jsonpath={.metadata.uid}",
        ]
        return execute_cmd(
            cmd,
            error_string="Unable to identify Kubernetes cluster",
            subprocess_log_level="debug",
        )

    def preflight_image_pull(
        self,
        image: str,
        use_pull_secret: bool,
        pull_secret_name: str = "acrtoken",
        timeout_s: int = 300,
    ):
        preflight_id = hashlib.sha256(
            f"{image}\0{pull_secret_name}".encode()
        ).hexdigest()[:12]
        name = f"farmvibes-image-preflight-{preflight_id}"
        spec: Dict[str, Any] = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": name},
            "spec": {
                "automountServiceAccountToken": False,
                "containers": [
                    {
                        "name": "image",
                        "image": image,
                        "imagePullPolicy": "Always",
                    }
                ],
                "restartPolicy": "Never",
                "terminationGracePeriodSeconds": 0,
            },
        }
        if use_pull_secret:
            spec["spec"]["imagePullSecrets"] = [
                {"name": pull_secret_name}
            ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as manifest_file:
            json.dump(spec, manifest_file)
            manifest_file.flush()
            execute_cmd(
                [
                    self.os_artifacts.kubectl,
                    "--context",
                    self.context_name,
                    "apply",
                    "-f",
                    manifest_file.name,
                ],
                error_string=f"Unable to preflight image {image}",
                subprocess_log_level="debug",
            )

        deadline = time.monotonic() + timeout_s
        # Kubelet can report auth failure while its node-authorizer cache learns
        # that the new pod may read the new pull Secret.
        auth_failure_at: Optional[float] = None
        last_status = ""
        try:
            while time.monotonic() < deadline:
                result = execute_cmd(
                    [
                        self.os_artifacts.kubectl,
                        "--context",
                        self.context_name,
                        "get",
                        "pod",
                        name,
                        "-o",
                        "json",
                    ],
                    error_string=f"Unable to inspect image preflight for {image}",
                    subprocess_log_level="debug",
                )
                pod = json.loads(result)
                statuses = pod.get("status", {}).get("containerStatuses", [])
                if statuses:
                    if statuses[0].get("imageID"):
                        return
                    waiting = statuses[0].get("state", {}).get("waiting", {})
                    last_status = ": ".join(
                        value
                        for value in (waiting.get("reason"), waiting.get("message"))
                        if value
                    )
                    authentication_failed = any(
                        marker in last_status.lower()
                        for marker in (
                            "401 unauthorized",
                            "authentication required",
                            "authorization failed",
                            "denied: requested access",
                            "403 forbidden",
                            "insufficient_scope",
                            "no basic auth credentials",
                            "pull access denied",
                        )
                    )
                    if authentication_failed:
                        if auth_failure_at is None:
                            auth_failure_at = time.monotonic()
                        elif time.monotonic() - auth_failure_at >= 10:
                            raise ImagePullAuthenticationError(
                                image, last_status
                            )
                    else:
                        auth_failure_at = None
                time.sleep(1)
            raise RuntimeError(
                f"Timed out pulling {image}"
                + (f": {last_status}" if last_status else "")
            )
        finally:
            try:
                execute_cmd(
                    [
                        self.os_artifacts.kubectl,
                        "--context",
                        self.context_name,
                        "delete",
                        "pod",
                        name,
                        "--ignore-not-found=true",
                    ],
                    check_empty_result=False,
                    capture_output=False,
                    subprocess_log_level="debug",
                )
            except Exception as error:
                log(
                    f"Unable to remove image preflight pod {name}: {error}",
                    level="warning",
                )

    def get(self, kind: str, name: str, jsonpath: Optional[str] = None):
        cmd = [
            self.os_artifacts.kubectl,
            "get",
            kind,
            name,
            "-o",
            "json" if not jsonpath else f'jsonpath="{jsonpath}"',
        ]
        return json.loads(
            execute_cmd(
                cmd,
                error_string=f"Unable to get {kind} {name}",
                check_empty_result=False,
                subprocess_log_level="debug",
            )
        )

    def get_or_none(self, kind: str, name: str, namespace: str = ""):
        cmd = [
            self.os_artifacts.kubectl,
            "get",
            kind,
            name,
            "-o",
            "json",
            "--ignore-not-found",
        ]
        if namespace:
            cmd.extend(["--namespace", namespace])
        result = execute_cmd(
            cmd,
            error_string=f"Unable to get {kind} {name}",
            check_empty_result=False,
            subprocess_log_level="debug",
        )
        return json.loads(result) if result else None

    def wait_for_delete(
        self,
        kind: str,
        name: str,
        timeout_s: int = 120,
        namespace: str = "",
    ):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.get_or_none(kind, name, namespace) is None:
                return
            time.sleep(1)
        raise ValueError(f"Timed out waiting for {kind} {name} to be deleted")

    def rollout_status(
        self, kind: str, name: str, timeout_s: int = 600, namespace: str = ""
    ):
        cmd = [
            self.os_artifacts.kubectl,
            "rollout",
            "status",
            f"{kind}/{name}",
            f"--timeout={timeout_s}s",
        ]
        if namespace:
            cmd.extend(["--namespace", namespace])
        execute_cmd(
            cmd,
            error_string=f"Unable to roll out {kind} {name}",
            check_empty_result=False,
        )

    def restart(
        self,
        kind: str,
        selectors: List[str] = [],
        name: str = "",
        cluster_name: str = "",
        namespace: str = "",
    ):
        if not name and not selectors:
            raise ValueError("Either name or selectors must be provided")
        if name and selectors:
            raise ValueError("Either name or selectors must be provided, but not both")
        cluster_name = self._actual_cluster_name(cluster_name)
        cmd = [self.os_artifacts.kubectl, "rollout", "restart", kind]
        if name:
            cmd += [name]
        else:
            cmd += ["-l", ",".join(selectors)]
        if namespace:
            cmd.extend(["--namespace", namespace])
        execute_cmd(
            cmd,
            error_string=f"Unable to restart {kind} with selectors {selectors}",
            subprocess_log_level="debug",
        )
        return True

    def apply_or_replace(self, file_path: str, cluster_name: str = ""):
        cluster_name = self._actual_cluster_name(cluster_name)
        with self.context(cluster_name):
            for kind in "apply replace".split():
                try:
                    log(f"Applying {kind} {file_path}", level="debug")
                    cmd = [self.os_artifacts.kubectl, kind, "-f", file_path]
                    execute_cmd(
                        cmd,
                        error_string=f"Unable to {kind} {file_path}",
                        subprocess_log_level="debug",
                    )
                    log(f"Successfully {kind} {file_path}", level="debug")
                    return True
                except Exception as e:
                    if kind == "apply":
                        log(f"Failed to apply {file_path}: {e} (will try again)", level="warning")
                        continue
        log(f"Failed to apply updates to CRD {file_path}", level="error")
        return False  # Should never reach here


class K3dWrapper:
    CONTAINERD_IMAGE_PATH = "/var/lib/rancher/k3s/agent/containerd/io.containerd.content.v1.content"
    K3D_SIMPLE_CONFIG_TEMPLATE = """
        apiVersion: k3d.io/v1alpha5
        kind: Simple
        metadata:
            name: {cluster_name}
        servers: {servers}
        agents: {agents}
        ports:
            - port: {host}:{farmvibes_ai_port}:80
              nodeFilters:
                - loadbalancer
        volumes:
            - volume: {storage_path}:/mnt
              nodeFilters:
                - server:*
                - agent:*
            - volume: {storage_path}%sregistry:%s
              nodeFilters:
                - server:0
        registries:
            create:
                name: {cluster_name}-registry
                host: "{host}"
                hostPort: "{registry_port}"
        options:
            k3s:
                nodeLabels:
                    - label: agentpool={cluster_name}
                      nodeFilters:
                        - server:*
                        - agent:*
    """ % (
        os.path.sep,
        CONTAINERD_IMAGE_PATH,
    )

    def __init__(self, os_artifacts: OSArtifacts, cluster_name: str):
        self.os_artifacts = os_artifacts
        self.cluster_name = cluster_name

    def cluster_exists(self, cluster_name: Optional[str] = None) -> bool:
        cluster_name = cluster_name or self.cluster_name
        cmd = [self.os_artifacts.k3d, "cluster", "list", "-o", "json"]
        result = execute_cmd(
            cmd, error_string="Unable to list clusters", subprocess_log_level="debug"
        )
        clusters = json.loads(result)
        return any(cluster["name"] == cluster_name for cluster in clusters)

    def delete(self, cluster_name: Optional[str] = None) -> bool:
        cluster_name = cluster_name or self.cluster_name
        cmd = [self.os_artifacts.k3d, "cluster", "delete", cluster_name]
        try:
            execute_cmd(
                cmd,
                error_string="Unable to delete cluster",
                check_empty_result=False,
                capture_output=False,
                subprocess_log_level="debug",
            )
            return True
        except Exception:
            return False

    def start(self, cluster_name: Optional[str] = None) -> bool:
        cluster_name = cluster_name or self.cluster_name
        cmd = [self.os_artifacts.k3d, "cluster", "start", cluster_name]
        try:
            execute_cmd(
                cmd,
                error_string="Unable to start cluster",
                check_empty_result=False,
                capture_output=False,
                subprocess_log_level="debug",
            )
            return True
        except Exception:
            return False

    def stop(self, cluster_name: Optional[str] = None) -> bool:
        cluster_name = cluster_name or self.cluster_name
        cmd = [self.os_artifacts.k3d, "cluster", "stop", cluster_name]
        try:
            execute_cmd(
                cmd,
                error_string="Unable to stop cluster",
                check_empty_result=False,
                capture_output=False,
                subprocess_log_level="debug",
            )
            return True
        except Exception:
            return False

    def info(self, cluster_name: Optional[str] = None) -> Dict[str, Any]:
        cluster_name = cluster_name or self.cluster_name
        cmd = [self.os_artifacts.k3d, "cluster", "list", "-o", "json"]
        result = execute_cmd(
            cmd, check_empty_result=False, capture_output=True, subprocess_log_level="debug"
        )
        clusters = json.loads(result)
        if not clusters:
            log("No clusters found")
            return {}
        for cluster in clusters:
            if cluster["name"] == cluster_name:
                return cluster
        return {}

    def get_cluster_config(self, cluster_name: Optional[str] = None) -> Dict[str, Any]:
        cluster_name = cluster_name or self.cluster_name
        cluster = self.info(cluster_name)
        if not cluster:
            raise ValueError(f"Unable to inspect cluster {cluster_name}")

        load_balancer = next(
            (
                node
                for node in cluster.get("nodes", [])
                if node.get("role", "").lower() == "loadbalancer"
            ),
            None,
        )
        if load_balancer is None:
            raise ValueError(
                f"Unable to inspect FarmVibes.AI port binding for cluster {cluster_name}"
            )
        try:
            api_binding = load_balancer["portMappings"]["80/tcp"][0]
        except (KeyError, IndexError, TypeError):
            raise ValueError(
                f"Unable to inspect FarmVibes.AI port binding for cluster {cluster_name}"
            )

        cmd = [self.os_artifacts.k3d, "registry", "list", "-o", "json"]
        result = execute_cmd(
            cmd,
            error_string="Unable to list registries",
            subprocess_log_level="debug",
        )
        registry = next(
            (
                item
                for item in json.loads(result)
                if item.get("runtimeLabels", {}).get("k3d.cluster") == cluster_name
            ),
            None,
        )
        if registry is None:
            raise ValueError(
                f"Unable to inspect registry port binding for cluster {cluster_name}"
            )
        try:
            registry_binding = registry["portMappings"]["5000/tcp"][0]
        except (KeyError, IndexError, TypeError):
            raise ValueError(
                f"Unable to inspect registry port binding for cluster {cluster_name}"
            )

        host = api_binding["HostIp"]
        if registry_binding["HostIp"] != host:
            raise ValueError(
                f"Cluster {cluster_name} uses different API and registry bind hosts"
            )
        return {
            "servers": int(cluster["serversCount"]),
            "agents": int(cluster["agentsCount"]),
            "port": int(api_binding["HostPort"]),
            "host": host,
            "registry_port": int(registry_binding["HostPort"]),
        }

    def get_storage_path(self, cluster_name: Optional[str] = None) -> str:
        cluster_name = cluster_name or self.cluster_name
        cluster = self.info(cluster_name)
        if not cluster:
            raise ValueError(f"Unable to inspect cluster {cluster_name}")

        raw_nodes = cluster.get("nodes", [])
        if not isinstance(raw_nodes, list):
            raise ValueError(f"Unable to inspect cluster {cluster_name} nodes")
        nodes = [
            node
            for node in raw_nodes
            if isinstance(node, dict)
            and str(node.get("role", "")).lower() in ("server", "agent")
        ]
        try:
            expected_nodes = int(cluster["serversCount"]) + int(cluster["agentsCount"])
        except (KeyError, TypeError, ValueError):
            expected_nodes = 0
        if not nodes or len(nodes) != expected_nodes:
            raise ValueError(
                f"Unable to inspect every server and agent storage bind for cluster "
                f"{cluster_name}"
            )

        storage_paths = []
        mount_marker = ":/mnt"
        for node in nodes:
            node_paths = []
            volumes = node.get("volumes", [])
            if not isinstance(volumes, list):
                volumes = []
            for volume in volumes:
                if not isinstance(volume, str):
                    continue
                marker_index = volume.rfind(mount_marker)
                if marker_index <= 0:
                    continue
                mount_options = volume[marker_index + len(mount_marker) :]
                if mount_options and not mount_options.startswith(":"):
                    continue
                source = volume[:marker_index]
                if os.path.isabs(source) or PureWindowsPath(source).is_absolute():
                    node_paths.append(source)
            if len(node_paths) != 1:
                raise ValueError(
                    f"Unable to identify one /mnt host bind for node "
                    f"{node.get('name', '<unknown>')} in cluster {cluster_name}"
                )
            storage_paths.append(node_paths[0])

        normalized_paths = {
            os.path.normcase(os.path.normpath(storage_path))
            for storage_path in storage_paths
        }
        if len(normalized_paths) != 1:
            raise ValueError(
                f"Cluster {cluster_name} uses inconsistent /mnt host binds"
            )
        return storage_paths[0]

    def create(
        self,
        servers: int,
        agents: int,
        storage_path: str,
        registry_port: int,
        farmvibes_port: int,
        host: str,
        cluster_name: Optional[str] = None,
    ) -> bool:
        cluster_name = cluster_name or self.cluster_name
        for p in (registry_port, farmvibes_port):
            if not is_port_free(p):
                log(
                    f"Port {p} is not free. Please free the port and retry.",
                    level="error",
                )
                if p == registry_port:
                    log(
                        "This is the port of the registry. You probably have a "
                        "registry running already. Stop it and delete it before "
                        "retrying.",
                        level="error",
                    )
                return False

        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(storage_path, "registry"), exist_ok=True)
            f = tempfile.NamedTemporaryFile(dir=d, delete=False, mode="w")
            f.write(
                self.K3D_SIMPLE_CONFIG_TEMPLATE.format(
                    cluster_name=cluster_name,
                    servers=servers,
                    agents=agents,
                    storage_path=storage_path,
                    registry_port=registry_port,
                    farmvibes_ai_port=farmvibes_port,
                    host=host,
                )
            )
            f.close()
            cmd = [self.os_artifacts.k3d, "cluster", "create", "--config", f.name]
            error = "Failed to create local cluster with k3d"
            execute_cmd(
                cmd,
                check_empty_result=False,
                capture_output=False,
                error_string=error,
                env_vars={"K3D_FIX_DNS": "1"},
            )
            log("Cluster created successfully")
        return True


class DockerWrapper:
    def __init__(self, os_artifacts: OSArtifacts):
        self.os_artifacts = os_artifacts

    def rm(self, container_name: str):
        cmd = [self.os_artifacts.docker, "rm", "-f", container_name]
        execute_cmd(
            cmd,
            error_string=f"Unable to remove container {container_name}",
            subprocess_log_level="debug",
        )

    def get(self, container_name: str):
        cmd = [self.os_artifacts.docker, "ps", "-a", "-q", "-f", f"name={container_name}"]
        result = execute_cmd(
            cmd,
            error_string=f"Unable to get container {container_name}",
            subprocess_log_level="debug",
        )
        return result

    def network_inspect(self, network_name: str):
        cmd = [self.os_artifacts.docker, "network", "inspect", network_name]
        result = execute_cmd(
            cmd,
            error_string=f"Unable to inspect network {network_name}",
            subprocess_log_level="debug",
        )
        return json.loads(result)

    def exec(self, container_name: str, command: List[str]):
        cmd = [self.os_artifacts.docker, "exec", "-it", container_name] + command
        result = execute_cmd(
            cmd,
            error_string=f"Unable to execute command {command} on container {container_name}",
            subprocess_log_level="debug",
            check_empty_result=False,
        )
        return result


class CertManagerWrapper:
    TARGET_VERSION = "1.21.1"
    NAMESPACE = "cert-manager"
    LEGACY_NAMESPACE = "kube-system"
    STABLE_MINOR_VERSIONS = (
        "1.12.17",
        "1.13.6",
        "1.14.7",
        "1.15.5",
        "1.16.5",
        "1.17.4",
        "1.18.6",
        "1.19.6",
        "1.20.3",
        TARGET_VERSION,
    )

    def __init__(
        self,
        os_artifacts: OSArtifacts,
        kubectl: KubectlWrapper,
    ) -> None:
        self.os_artifacts = os_artifacts
        self.kubectl = kubectl

    @staticmethod
    def _version_tuple(version: str) -> Tuple[int, ...]:
        return tuple(map(int, version.lstrip("v").split(".")))

    def _releases(self) -> List[Dict[str, Any]]:
        with self.kubectl.context(self.kubectl.cluster_name):
            output = execute_cmd(
                [
                    self.os_artifacts.helm,
                    "list",
                    "--all-namespaces",
                    "--filter",
                    "^cert-manager$",
                    "--output",
                    "json",
                ],
                check_empty_result=False,
                error_string="Unable to get cert-manager version",
                subprocess_log_level="debug",
            )
        releases = json.loads(output or "[]")
        allowed_namespaces = {self.NAMESPACE, self.LEGACY_NAMESPACE}
        unexpected_namespaces = sorted(
            {
                str(release.get("namespace"))
                for release in releases
                if release.get("namespace") not in allowed_namespaces
            }
        )
        if unexpected_namespaces:
            raise RuntimeError(
                "Refusing to modify cert-manager owned outside the supported "
                f"namespaces: {', '.join(unexpected_namespaces)}"
            )
        if len({release.get("namespace") for release in releases}) > 1:
            raise RuntimeError(
                "Refusing to modify cert-manager releases found in both "
                "supported namespaces"
            )
        return releases

    def _release(self, namespace: str) -> Optional[Dict[str, Any]]:
        return next(
            (
                release
                for release in self._releases()
                if release.get("namespace") == namespace
            ),
            None,
        )

    def _validate_crd_ownership(self) -> None:
        with self.kubectl.context(self.kubectl.cluster_name):
            output = execute_cmd(
                [
                    self.os_artifacts.kubectl,
                    "get",
                    "customresourcedefinitions",
                    "--selector",
                    "app.kubernetes.io/instance=cert-manager",
                    "--output",
                    "json",
                ],
                check_empty_result=False,
                error_string="Unable to inspect cert-manager CRD ownership",
                subprocess_log_level="debug",
            )
        for crd in json.loads(output)["items"]:
            metadata = crd.get("metadata", {})
            annotations = metadata.get("annotations") or {}
            release_name = annotations.get("meta.helm.sh/release-name")
            release_namespace = annotations.get("meta.helm.sh/release-namespace")
            if release_name is None and release_namespace is None:
                continue
            if release_name != "cert-manager" or release_namespace not in {
                self.NAMESPACE,
                self.LEGACY_NAMESPACE,
            }:
                raise RuntimeError(
                    "Refusing to take ownership of cert-manager CRD "
                    f"{metadata.get('name', '<unknown>')}"
                )

    def version(self) -> Optional[str]:
        release = self._release(self.NAMESPACE) or self._release(
            self.LEGACY_NAMESPACE
        )
        return (
            release["app_version"].lstrip("v")
            if release is not None
            else None
        )

    def upgrade_path(self) -> List[str]:
        current = self.version()
        if current is None:
            return []
        current_tuple = self._version_tuple(current)
        target_tuple = self._version_tuple(self.TARGET_VERSION)
        if current_tuple > target_tuple:
            raise RuntimeError(
                f"Installed cert-manager {current} is newer than supported "
                f"{self.TARGET_VERSION}"
            )
        return [
            version
            for version in self.STABLE_MINOR_VERSIONS
            if current_tuple < self._version_tuple(version) <= target_tuple
        ]

    def needs_upgrade(self) -> bool:
        return bool(self.upgrade_path())

    def upgrade_sequentially(self) -> None:
        namespace = (
            self.LEGACY_NAMESPACE
            if self._release(self.LEGACY_NAMESPACE)
            else self.NAMESPACE
        )
        for version in self.upgrade_path():
            crd_flag = "installCRDs" if self._version_tuple(version) < (1, 15) else "crds.enabled"
            log(f"Upgrading cert-manager to {version}")
            with self.kubectl.context(self.kubectl.cluster_name):
                execute_cmd(
                    [
                        self.os_artifacts.helm,
                        "upgrade",
                        "cert-manager",
                        "cert-manager",
                        "--repo",
                        "https://charts.jetstack.io",
                        "--namespace",
                        namespace,
                        "--version",
                        f"v{version}",
                        "--set",
                        f"{crd_flag}=true",
                        "--set",
                        r"nodeSelector.kubernetes\.io/os=linux",
                        *(
                            ["--set", "startupapicheck.enabled=false"]
                            if namespace == self.LEGACY_NAMESPACE
                            else []
                        ),
                        "--atomic",
                        "--timeout",
                        "10m",
                    ],
                    check_empty_result=False,
                    error_string=f"Unable to upgrade cert-manager to {version}",
                    subprocess_log_level="debug",
                )

    def prepare_for_terraform_reconciliation(self) -> bool:
        releases = self._releases()
        legacy_release = any(
            release.get("namespace") == self.LEGACY_NAMESPACE
            for release in releases
        )
        self._validate_crd_ownership()
        with self.kubectl.context(self.kubectl.cluster_name):
            if legacy_release:
                execute_cmd(
                    [
                        self.os_artifacts.helm,
                        "uninstall",
                        "cert-manager",
                        "--namespace",
                        self.LEGACY_NAMESPACE,
                    ],
                    check_empty_result=False,
                    error_string="Unable to remove legacy cert-manager release",
                    subprocess_log_level="debug",
                )
            execute_cmd(
                [
                    self.os_artifacts.kubectl,
                    "annotate",
                    "customresourcedefinitions",
                    "--selector",
                    "app.kubernetes.io/instance=cert-manager",
                    "meta.helm.sh/release-name=cert-manager",
                    f"meta.helm.sh/release-namespace={self.NAMESPACE}",
                    "--overwrite",
                ],
                check_empty_result=False,
                error_string="Unable to transfer cert-manager CRD ownership",
                subprocess_log_level="debug",
            )
        return legacy_release


class DaprWrapper:  # DaprWrapr 🫠
    VERSION_STRING = "VERSION"
    PLACEMENT_STATEFULSET = "dapr-placement-server"
    SCHEDULER_STATEFULSET = "dapr-scheduler-server"
    CRD_BASE = "https://raw.githubusercontent.com/dapr/dapr/v{}/charts/dapr/crds/"
    STABLE_MINOR_VERSIONS = (
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
    )
    CRD_FILES = [
        "components.yaml",
        "configuration.yaml",
        "subscription.yaml",
        "resiliency.yaml",
        "httpendpoints.yaml",
        "mcpservers.yaml",
        "workflowaccesspolicy.yaml",
    ]
    REQUIRED_CRD_FILES = {
        "components.yaml",
        "configuration.yaml",
        "subscription.yaml",
    }

    def __init__(
        self,
        os_artifacts: OSArtifacts,
        kubectl: KubectlWrapper,
        cluster_kind: str = "local",
        namespace: str = "dapr-system",
    ):
        self.cluster_kind = cluster_kind
        self.os_artifacts = os_artifacts
        self.namespace = namespace
        self.kubectl = kubectl

    def _version_column(self, header: str) -> int:
        return header.split().index(self.VERSION_STRING)

    def _target_version(self) -> str:
        # use pkg_resources to find dapr.tf:
        dapr_tf = pkgutil.get_data(
            "vibe_core.terraform", f"{self.cluster_kind}/modules/kubernetes/dapr.tf"
        )
        if not dapr_tf:
            raise ValueError("Unable to find dapr.tf")
        target = re.findall('version\\s+=\\s+"(.*)"', dapr_tf.decode("utf-8"))[0]
        assert len(target) > 0, "Unable to find Dapr version in dapr.tf"
        return target

    def version(self):
        cmd = [self.os_artifacts.dapr, "status", "-k"]
        with self.kubectl.context(self.kubectl.cluster_name):
            if (
                self.kubectl.get_or_none(
                    "deployment",
                    "dapr-operator",
                    namespace=self.namespace,
                )
                is None
            ):
                output = execute_cmd(
                    [
                        self.os_artifacts.helm,
                        "list",
                        "--all-namespaces",
                        "--filter",
                        "^dapr$",
                        "--output",
                        "json",
                    ],
                    check_empty_result=False,
                    error_string="Unable to inspect Dapr release",
                    subprocess_log_level="debug",
                )
                releases = json.loads(output or "[]")
                if any(
                    release.get("namespace") == self.namespace
                    for release in releases
                ):
                    raise RuntimeError(
                        "Dapr release exists but its operator is missing"
                    )
                return []
            result = execute_cmd(
                cmd, error_string="Unable to get Dapr version", subprocess_log_level="debug"
            )
            lines = result.split("\n")
            version_column = self._version_column(lines[0])
            all_versions = set([line.split()[version_column] for line in lines[1:] if line])
        return [v for v in all_versions]

    def needs_upgrade(self):
        version_tuple = tuple(map(int, self._target_version().split(".")))
        current_versions_tuples = [tuple(map(int, v.split("."))) for v in self.version()]
        if any(version > version_tuple for version in current_versions_tuples):
            raise RuntimeError(
                "Installed Dapr version is newer than the supported target"
            )
        return len(current_versions_tuples) == 0 or any(
            [v < version_tuple for v in current_versions_tuples if v > (1, 0, 0)]
        )

    def upgrade_path(self) -> List[str]:
        current_versions = [
            tuple(map(int, version.split(".")))
            for version in self.version()
            if tuple(map(int, version.split("."))) > (1, 0, 0)
        ]
        if not current_versions:
            return []
        current = min(current_versions)
        target = tuple(map(int, self._target_version().split(".")))
        if any(version > target for version in current_versions):
            raise RuntimeError(
                "Installed Dapr version is newer than the supported target"
            )
        return [
            version
            for version in self.STABLE_MINOR_VERSIONS
            if current < tuple(map(int, version.split("."))) <= target
        ]

    def upgrade_crds(self, version: Optional[str] = None):
        # Upgrading dapr is a two-stage process.
        # First, we upgrade the CRDs, then OpenTofu converges the Dapr chart.
        status = []
        version = version or self._target_version()
        for crd in self.CRD_FILES:
            url = self.CRD_BASE.format(version) + crd
            response = requests.head(url, timeout=30)
            if response.status_code == 404:
                if crd in self.REQUIRED_CRD_FILES:
                    raise RuntimeError(
                        f"Required Dapr CRD {crd} is missing for {version}"
                    )
                log(f"CRD {crd} not found at {url}, ignoring it", level="warning")
                continue
            response.raise_for_status()
            status.append(self.kubectl.apply_or_replace(url))
        return all(status)

    def upgrade(self, version: Optional[str] = None):
        version = version or self._target_version()
        cmd = [
            self.os_artifacts.dapr,
            "upgrade",
            "-k",
            f"--runtime-version={version}",
        ]
        log(f"Upgrading Dapr to version {version}")
        with self.kubectl.context(self.kubectl.cluster_name):
            execute_cmd(
                cmd,
                error_string="Unable to upgrade Dapr",
                subprocess_log_level="debug",
            )

    def upgrade_sequentially(self) -> bool:
        with self.kubectl.context(self.kubectl.cluster_name):
            upgrade_path = self.upgrade_path()
            for version in upgrade_path:
                if not self.upgrade_crds(version):
                    return False
                self.upgrade(version)
            if upgrade_path and self.kubectl.get_or_none(
                "statefulset", self.SCHEDULER_STATEFULSET, namespace=self.namespace
            ):
                self.kubectl.restart(
                    "statefulset",
                    name=self.SCHEDULER_STATEFULSET,
                    namespace=self.namespace,
                )
                self.kubectl.rollout_status(
                    "statefulset",
                    self.SCHEDULER_STATEFULSET,
                    namespace=self.namespace,
                )
        return True

    def prepare_for_terraform_reconciliation(self) -> bool:
        with self.kubectl.context(self.kubectl.cluster_name):
            statefulset = self.kubectl.get_or_none(
                "statefulset",
                self.PLACEMENT_STATEFULSET,
                namespace=self.namespace,
            )
            if statefulset is None or statefulset.get("spec", {}).get(
                "replicas"
            ) == 3:
                return False
            self.kubectl.delete(
                "statefulset",
                self.PLACEMENT_STATEFULSET,
                ignore_not_found=True,
                namespace=self.namespace,
                wait=False,
            )
            self.kubectl.wait_for_delete(
                "statefulset",
                self.PLACEMENT_STATEFULSET,
                timeout_s=600,
                namespace=self.namespace,
            )
        return True
