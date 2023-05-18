import json
import os
from typing import Dict
from .osartifacts import OSArtifacts
from .helper import execute_cmd, verify_to_proceed
from .logging import log
from .constants import (
    AZ_LOGIN_PROMPT,
    AZ_CREDS_REFRESH_ATTEMPTS,
    CACHE_NODE_CPU_NAME,
    PREFIX_FILE_NAME,
    TOTAL_REGIONAL_CPU_NAME,
    WORKER_NODE_CPU_NAME,
)


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
    CACHE_NODE_CPU_NAME: 4,
}


def expand_azure_region(os_artifacts: OSArtifacts, canonical_region: str) -> str:
    cmd = (
        f"{os_artifacts.get_az_cmd()} account list-locations "
        f"--query \"[?name=='{canonical_region}'].displayName\" "
        "-o tsv"
    )
    error = f"Couldn't get azure region. Maybe it is invalid {canonical_region}"

    return execute_cmd(cmd, True, True, error)


def get_subscription_info(os_artifacts: OSArtifacts):
    MAX_SUBSCRIPTION_ATTEMPTS = 2
    cmd = f"{os_artifacts.get_az_cmd()} account show"
    error = "Unable to get default subscription"

    for i in range(MAX_SUBSCRIPTION_ATTEMPTS):
        sub_info = json.loads(execute_cmd(cmd, True, True, error))

        proceed = verify_to_proceed(f"Is this the correct subscription? {sub_info['name']}")
        if proceed:
            return sub_info["id"], sub_info["tenantId"]

        if i < MAX_SUBSCRIPTION_ATTEMPTS - 1:
            proceed = verify_to_proceed("Would you like to change now?")
            if proceed:
                suggested_sub_id = input(
                    "Enter the GUID of the subscription you would like to use: "
                )
                if suggested_sub_id:
                    execute_cmd(
                        f"{os_artifacts.get_az_cmd()} account set -s {suggested_sub_id}",
                        True,
                        False,
                        "Failed to set subscription",
                    )
                else:
                    break
            else:
                break

    raise ValueError("Cancelation Requested")


def refresh_az_creds(os_artifacts: OSArtifacts):
    cmd = f"{os_artifacts.get_az_cmd()} account get-access-token"
    error = "Unable to refresh Azure tokens"

    for _ in range(AZ_CREDS_REFRESH_ATTEMPTS):
        try:
            execute_cmd(cmd, True, True, error)
            break
        except Exception:
            proceed = verify_to_proceed(
                "It seems Azure has logged out.\n"
                f"Please relogin on another prompt using {AZ_LOGIN_PROMPT} and continue here.\n"
                "Ready to continue?"
            )
            if not proceed:
                raise ValueError("Unable to get AZ Credentials.")


def check_resource_providers(os_artifacts: OSArtifacts, region: str):
    expanded_region = expand_azure_region(os_artifacts, region)
    for provider in AZURE_RESOURCES_REQUIRED:
        log(f"Validating that {provider} is available in the subscription selected")
        cmd = (
            f"{os_artifacts.get_az_cmd()} provider show -n {provider} "
            '--query "registrationState" -o tsv'
        )
        error = f"{provider} resource provider not registered"
        result = execute_cmd(cmd, True, True, error)
        if result != "Registered":
            raise ValueError(error)

        cmd = (
            f"{os_artifacts.get_az_cmd()} provider show -n {provider} "
            '--query "resourceTypes[].locations" -o tsv'
        )

        result = execute_cmd(cmd, True, True, error)
        if expanded_region not in result:
            raise ValueError(error)


def verify_enough_cores_available(os_artifacts: OSArtifacts, region: str, worker_nodes: int = 1):
    if worker_nodes > 0:
        worker_cpus_per_node = CPUS_REQUIRED[WORKER_NODE_CPU_NAME]
        worker_cpus_needed = worker_cpus_per_node * worker_nodes
        CPUS_REQUIRED[WORKER_NODE_CPU_NAME] = worker_cpus_needed
        CPUS_REQUIRED[TOTAL_REGIONAL_CPU_NAME] = (
            CPUS_REQUIRED[TOTAL_REGIONAL_CPU_NAME] - worker_cpus_per_node + worker_cpus_needed
        )

    for cpu_type in CPUS_REQUIRED.keys():
        required = CPUS_REQUIRED[cpu_type]
        log(f"Validating that {cpu_type} has enough resources in region {region}")

        command = (
            f"{os_artifacts.get_az_cmd()} vm list-usage --location {region} "
            f"--output json --query \"[?localName=='{cpu_type}']\""
        )
        error = f"{cpu_type} wasn't available or not parsable"

        result = execute_cmd(command, True, True, error)

        vm_usage = json.loads(result)[0]
        current_usage = int(vm_usage["currentValue"])
        total_allowed = int(vm_usage["limit"])
        available = total_allowed - current_usage

        if required > available:
            raise ValueError(f"{cpu_type} has {available} CPUs. We need {required}.")


def get_saved_prefix(os_artifacts: OSArtifacts) -> str:
    prefix_file = os_artifacts.get_config_file(PREFIX_FILE_NAME)
    if prefix_file and os.path.isfile(prefix_file) and os.path.getsize(prefix_file) > 0:
        with open(prefix_file) as f:
            return f.read().strip()
    else:
        raise ValueError("No saved data found. Have you created the cluster with a setup command?")


def does_prefix_exist(os_artifacts: OSArtifacts) -> bool:
    prefix_file = os_artifacts.get_config_file(PREFIX_FILE_NAME)
    if prefix_file and os.path.isfile(prefix_file) and os.path.getsize(prefix_file) > 0:
        return True
    return False


def save_prefix(os_artifacts: OSArtifacts, prefix: str):
    prefix_file = os_artifacts.get_config_file(PREFIX_FILE_NAME)
    with open(prefix_file, "w") as f:
        f.write(prefix)


def apply_terraform(
    os_artifacts: OSArtifacts, working_directory: str, state_file: str, variables: Dict[str, str]
):
    refresh_az_creds(os_artifacts)
    log(f"Applying terraform in {working_directory}")
    command = (
        f"{os_artifacts.get_terraform_cmd()} -chdir={working_directory} apply "
        f"-state={state_file} -auto-approve "
    )
    for v in variables.keys():
        command += f"-var {v}={variables[v]} "
    execute_cmd(command, True, False, f"Failed to apply terraform resources in {working_directory}")


def get_terraform_output(os_artifacts: OSArtifacts, working_directory: str, state_file: str):
    refresh_az_creds(os_artifacts)
    command = (
        f"{os_artifacts.get_terraform_cmd()} -chdir={working_directory} "
        f"output -state={state_file} -json"
    )
    output = execute_cmd(
        command, True, False, f"Failed to get terraform results from {working_directory}"
    )
    return json.loads(output)


def initialize_terraform(os_artifacts: OSArtifacts, working_directory: str):
    log(f"Initializing terraform in {working_directory}")
    refresh_az_creds(os_artifacts)
    command = (
        f"{os_artifacts.get_terraform_cmd()} -chdir={working_directory} init -upgrade -force-copy"
    )
    execute_cmd(command, True, False, f"Failed to initialize terraform in {working_directory}")
