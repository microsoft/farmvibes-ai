import hashlib
import json
import os
import platform
import re
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from functools import partialmethod
from typing import Any, Dict, List, Optional, Tuple

from .constants import RABBITMQ_IMAGE_TAG, REDIS_IMAGE_TAG
from .helper import execute_cmd, is_port_free, log_should_be_logged_in, verify_to_proceed
from .logging import ColorFormatter, log
from .osartifacts import OSArtifacts

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
  containers:
  - command:
    - tail
    - "-f"
    - "/dev/null"
    image: bitnami/minideb
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


def on_windows() -> bool:
    return platform.system() == "Windows"


class TerraformWrapper:
    STATE_CONTAINER_NAME = "terraform-state"
    INFRA_STATE_FILE = "infra.tfstate"
    ANSI_ESCAPE_PAT = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    REPLACEMENT_PAT = re.compile(r"#\s+(.*)\s+must\s+be\s+replaced")
    REPLACEMENT_SUBSTRINGS = [
        "cosmosdb",
        "storageaccount",
    ]
    PLUGIN_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "farmvibes-ai", "terraform")

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
        plan_file: str = "",
    ):
        if refresh_creds:
            assert self.az is not None, "AzureCliWrapper must be provided to refresh credentials"
            self.az.refresh_az_creds()
        log(f"{'Planning' if plan else 'Applying'} terraform in {working_directory}")
        command = [
            self.os_artifacts.terraform,
            f"-chdir={working_directory}",
            "plan" if plan else "apply",
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
        env_vars["ARM_ENVIRONMENT"] = self.environment
        stdout = execute_cmd(
            command,
            check_return_code=True,
            check_empty_result=False,
            error_string=(
                f"Failed to {'plan' if plan else 'apply'} terraform resources "
                f"in {working_directory}"
            ),
            capture_output=True,
            env_vars=env_vars,
        )
        return stdout

    plan = partialmethod(_plan_or_apply, plan=True)
    apply = partialmethod(_plan_or_apply, plan=False)

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
            f"Failed to get terraform results from {working_directory}",
            censor_output=True,
            env_vars=env_vars,
        )
        return json.loads(output)

    def init(
        self,
        working_directory: str,
        refresh_creds: bool = True,
        backend_config: Dict[str, str] = {},
        cleanup_state: bool = False,
    ):
        log(f"Initializing terraform in {working_directory}")
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
                if on_windows:
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
                f"Failed to initialize terraform in {working_directory}",
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
        cleanup_state: bool = False,
        is_update: bool = False,
    ):
        infra_directory = os.path.join(self.os_artifacts.aks_directory, "modules", "infra")
        log("Executing terraform to build out infrastructure (this may take up to 30 minutes)...")
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
        variables = {
            "tenantId": tenant_id,
            "subscriptionId": subscription_id,
            "location": region,
            "prefix": cluster_name,
            "kubeconfig_location": self.os_artifacts.config_dir,
            "max_worker_nodes": worker_nodes,
            "resource_group_name": resource_group,
        }

        state_file = self.os_artifacts.get_terraform_file(
            self.INFRA_STATE_FILE, cluster_name, resource_group
        )
        with tempfile.NamedTemporaryFile(delete=False) as plan_file:
            plan = self.plan(infra_directory, state_file, variables, plan_file=plan_file.name)
            replacements = self._get_replacements(plan)
            needs_restart = False
            if replacements:
                log(
                    f"Terraform plan requires replacement of resources {', '.join(replacements)}..."
                )
                proceed = True
                needs_restart = True
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
                    log("Continuing with terraform apply...")
            apply = self.apply(infra_directory, state_file, variables, plan_file=plan_file.name)
            if is_update and (needs_restart or "azurerm_key_vault_secret" in apply):
                kubectl = KubectlWrapper(self.os_artifacts, cluster_name)
                kubectl.restart("deployment", selectors=["backend=terravibes"])
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
        backend_storage_name: str,
        backend_container_name: str,
        backend_storage_access_key: str,
        cleanup_state: bool = False,
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
            "resource_group_name": resource_group,
            "current_user_name": current_user_name,
            "certificate_email": certificate_email,
        }

        state_file = self.os_artifacts.get_terraform_file(
            "kubernetes.tfstate", cluster_name, resource_group
        )
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
        worker_replicas: int,
        log_level: str,
        cleanup_state: bool = False,
    ):
        services_directory = os.path.join(self.os_artifacts.aks_directory, "..", "services")
        backend_config = {
            "config_path": kubernetes_config_path,
            "config_context": kubernetes_config_context,
        }
        self.init(
            services_directory,
            backend_config=backend_config,
            cleanup_state=cleanup_state,
            refresh_creds=True,
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
            "worker_replicas": worker_replicas,
            "farmvibes_log_level": log_level,
        }

        state_file = self.os_artifacts.get_terraform_file(
            "services.tfstate", cluster_name, resource_group
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
        redis_image_tag: str = REDIS_IMAGE_TAG,
        rabbitmq_image_tag: str = RABBITMQ_IMAGE_TAG,
        is_update: bool = False,
    ):
        if not is_update:
            self.init(self.os_artifacts.local_directory, False, cleanup_state=True)
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
            "redis_image_tag": redis_image_tag,
            "rabbitmq_image_tag": rabbitmq_image_tag,
            "farmvibes_log_level": log_level,
            "max_log_file_bytes": f"{max_log_file_bytes}" if max_log_file_bytes else "",
            "log_backup_count": f"{log_backup_count}" if log_backup_count else "",
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
        error = "Couldn't list terraform workspaces"
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
        error = "Couldn't get terraform workspace"
        return execute_cmd(
            cmd, True, True, error, capture_output=True, subprocess_log_level="debug"
        )

    def set_workspace(self, workspace: str):
        workspaces = self.list_workspaces()
        if workspace not in workspaces:
            log(f"Terraform workspace {workspace} does not exist. Creating it...", level="debug")
            cmd = [self.os_artifacts.terraform, "workspace", "new", workspace]
            error = f"Couldn't create terraform workspace {workspace}"
            execute_cmd(
                cmd,
                check_return_code=False,
                check_empty_result=True,
                error_string=error,
                subprocess_log_level="debug",
            )
        else:
            log(f"Terraform workspace {workspace} already exists. Selecting it...", level="debug")

        cmd = [self.os_artifacts.terraform, "workspace", "select", workspace]
        error = f"Couldn't select terraform workspace {workspace}"
        execute_cmd(cmd, True, False, error, capture_output=False, subprocess_log_level="debug")

    def delete_workspace(self, workspace: str):
        workspaces = self.list_workspaces()
        if workspace not in workspaces:
            log(
                f"Terraform workspace {workspace} does not exist. Nothing to delete...",
                level="debug",
            )
            return
        cmd = [self.os_artifacts.terraform, "workspace", "delete", workspace]
        error = f"Couldn't delete terraform workspace {workspace}"
        try:
            execute_cmd(cmd, True, False, error, capture_output=False, subprocess_log_level="debug")
        except Exception as e:
            log(f"Couldn't delete terraform workspace {workspace}: {e}", level="debug")

    @contextmanager
    def workspace(self, workspace_name: str):
        current_workspace = self.get_workspace()
        log(f"Current terraform workspace is {current_workspace}", level="debug")
        log(f"Setting terraform workspace to {workspace_name}", level="debug")
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
            log(f"Error getting infra results with terraform: {e}", level="error")
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
            log(f"Getting terraform state from {storage_name}/{container_name}")
            state = json.loads(
                self.az.download_blob(storage_name, container_name, self.INFRA_STATE_FILE, key=key)
            )
            return state
        except Exception as e:
            log(f"Error getting storage account name from terraform state file: {e}", level="error")
            return {}

    def get_storage_account_name(self):
        state = self._get_infra_state()
        try:
            log("Extracting storage account name from terraform state", level="debug")
            storage_account = state["outputs"]["storage_account_name"]["value"]
            return storage_account
        except Exception as e:
            log(f"Error getting storage account name from terraform state: {e}", level="error")
            return ""

    def get_current_core_count(self) -> Tuple[int, int]:
        state = self._get_infra_state()
        try:
            log("Extracting current core count from terraform state", level="debug")
            max_workers = int(state["outputs"]["max_worker_nodes"]["value"])
            max_default = int(state["outputs"]["max_default_nodes"]["value"])
            return (
                max_workers * CPUS_REQUIRED[WORKER_NODE_CPU_NAME],
                max_default * CPUS_REQUIRED[DEFAULT_NODE_CPU_NAME],
            )
        except Exception as e:
            log(f"Error getting current core count from terraform state: {e}", level="error")
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

    def infer_registry_credentials(self, registry: str) -> Tuple[str, str]:
        log(f"Inferring credentials for {registry}")
        registry = registry.replace(".azurecr.io", "")  # FIXME: This only works for Azure Public

        self.refresh_az_creds()
        username_command = [
            self.os_artifacts.az,
            "acr",
            "credential",
            "show",
            "-n",
            registry,
            "--query",
            "username",
        ]
        password_command = [
            self.os_artifacts.az,
            "acr",
            "credential",
            "show",
            "-n",
            registry,
            "--query",
            "passwords[0].value",
        ]
        error = f"Unable to infer credentials for {registry}"
        username = json.loads(execute_cmd(username_command, True, True, error, censor_output=True))
        password = json.loads(execute_cmd(password_command, True, True, error, censor_output=True))
        return username, password

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
        ]
        error = "Couldn't get storage account keys. Do you have access to the resource group?"
        results = execute_cmd(cmd, True, False, error, censor_output=True)
        keys = json.loads(results)
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
            results = json.loads(execute_cmd(cmd, True, False, error, subprocess_log_level="debug"))
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
                execute_cmd(cmd, True, False, error, subprocess_log_level="debug")
            except Exception:
                return False

        return True

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
            "--name",
            self.cluster_name,
            "--resource-group",
            self.resource_group,
            "--overwrite-existing",
        ]

        error = "Couldn't get kubernetes credentials. Do you have access to the cluster?"
        execute_cmd(cmd, True, False, error, subprocess_log_level="debug")

        # Now we have to use kubelogin to get/convert the credentials
        cmd = [self.os_artifacts.kubelogin, "convert-kubeconfig", "-l", "azurecli"]
        error = "Couldn't convert kubernetes credentials using kubelogin. Sorry."
        execute_cmd(cmd, True, False, error_string=error, subprocess_log_level="debug")

    def ensure_azurerm_backend(
        self,
        location: str,
        container_name: str = "terraform-state",
    ) -> Tuple[str, str, str]:
        accounts = self.get_storage_account_list()
        storage_name = self.storage_name
        if not any(account["name"] == storage_name for account in accounts):
            self.create_storage_account(location, storage_name)
        key = self.get_storage_account_key(storage_name)
        if not self.ensure_container_exists(storage_name, key, container_name):
            log("Couldn't create storage container for Terraform backend.", level="error")
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

    def create_redis_volume_pod(self, cluster_name: str = ""):
        cluster_name = self._actual_cluster_name(cluster_name)
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "redis-vol-pod.yaml"), "w") as fp:
                fp.write(REDIS_VOL_POD_YAML)
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
        self, pod: str, command: List[str], cluster_name: str = "", capture_output: bool = True
    ):
        cluster_name = self._actual_cluster_name(cluster_name)
        cmd = [self.os_artifacts.kubectl, "exec", pod, "--"] + command
        result = execute_cmd(
            cmd,
            error_string=f"Unable to execute command {command} on pod {pod}",
            capture_output=capture_output,
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

    def delete(self, kind: str, name: str, cluster_name: str = ""):
        cluster_name = self._actual_cluster_name(cluster_name)
        cmd = [self.os_artifacts.kubectl, "delete", kind, name]
        execute_cmd(
            cmd, error_string=f"Unable to delete {kind} {name}", subprocess_log_level="debug"
        )

    def get_secret(self, name: str, key: str, cluster_name: str = ""):
        cluster_name = self._actual_cluster_name(cluster_name)
        cmd = [self.os_artifacts.kubectl, "get", "secret", name, "-o", f'jsonpath="{{{key}}}"']
        result = execute_cmd(
            cmd, error_string=f"Unable to get secret {name}", subprocess_log_level="debug"
        )
        return json.loads(result)

    def create_docker_token(self, token: str, registry: str, username: str, password: str):
        cmd = [
            self.os_artifacts.kubectl,
            "create",
            "secret",
            "docker-registry",
            token,
            f"--docker-server={registry}",
            f"--docker-username={username}",
            f"--docker-password={password}",
            f"--docker-email={username}",
        ]
        execute_cmd(
            cmd,
            error_string="Unable to create acr token",
            censor_command=True,
            subprocess_log_level="debug",
        )

    def add_secret(self, secret_name: str, secret_value: str):
        cmd = [
            self.os_artifacts.kubectl,
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
        cmd = [self.os_artifacts.kubectl, "delete", "secret", secret_name]
        execute_cmd(
            cmd,
            check_empty_result=False,
            capture_output=False,
            censor_command=True,
            subprocess_log_level="debug",
        )
        return True

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

    def restart(self, kind: str, selectors: List[str] = [], name: str = "", cluster_name: str = ""):
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
        execute_cmd(
            cmd,
            error_string=f"Unable to restart {kind} with selectors {selectors}",
            subprocess_log_level="debug",
        )
        return True


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
                - server:*
                - agent:*
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
