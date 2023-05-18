import argparse
import json
import platform
import os
from vibe_core.cli.constants import (
    DEFAULT_IMAGE_PREFIX,
    DEFAULT_IMAGE_TAG,
    DEFAULT_REGISTRY_PATH,
    LOGGING_LEVEL_VERBOSE,
    MAX_WORKER_NODES,
    PREFIX_FILE_NAME,
    REMOTE_SERVICE_URL_PATH_FILE,
)
from vibe_core.cli.helper import (
    execute_cmd,
    generate_random_string,
    verify_to_proceed,
    in_wsl,
    is_file_in_mount,
    set_auto_confirm,
)
from vibe_core.cli.azure_helper import (
    apply_terraform,
    check_resource_providers,
    does_prefix_exist,
    expand_azure_region,
    get_saved_prefix,
    get_subscription_info,
    get_terraform_output,
    initialize_terraform,
    save_prefix,
    verify_enough_cores_available,
)
from vibe_core.cli.linuxosartifacts import LinuxOSArtifacts
from vibe_core.cli.osartifacts import OSArtifacts
from vibe_core.cli.windowsosartifacts import WindowsOSArtifacts
from vibe_core.cli.logging import log, set_log_level


def show_url(os_artifacts: OSArtifacts, existing_rg: str):
    prefix = get_saved_prefix(os_artifacts)
    rg_name = existing_rg if existing_rg else prefix + "-tvrg"

    cmd = (
        f"{os_artifacts.get_az_cmd()} aks get-credentials "
        f"--name {prefix}-kbstv "
        f"--resource-group {rg_name} "
        "--overwrite-existing"
    )

    # Detect if we're running in WSL
    if in_wsl() and is_file_in_mount(os_artifacts.get_az_cmd()):
        log("Show URL command does not run correctly in run within WSL due AZ context issues")
        log("Execute this script with the 'show_url' option on your Windows prompt to get the URL")
        return

    error = "Couldn't get kubernetes credentials"
    execute_cmd(cmd, True, False, error)

    cmd = (
        f"{os_artifacts.get_kubectl_cmd()} get ingress terravibes-rest-api-ingress "
        f'-o jsonpath="{{.spec.rules[0].host}}"'
    )
    error = "Couldn't get ingress hostname from kubernetes"
    url = execute_cmd(cmd, True, True, error)

    with open(os_artifacts.get_config_file(REMOTE_SERVICE_URL_PATH_FILE), "w") as f:
        f.write(url)

    log(f"URL for your AKS Cluster is: {url}")


def setup_or_upgrade(
    os_artifacts: OSArtifacts,
    region: str,
    subscription_id: str,
    tenant_id: str,
    certificate_email: str,
    registry_path: str,
    registry_username: str,
    registry_password: str,
    image_prefix: str,
    image_tag: str,
    existing_rg: str,
    force_new: bool,
):
    check_resource_providers(os_artifacts, region)
    verify_enough_cores_available(os_artifacts, region, MAX_WORKER_NODES)

    if force_new:
        log("Ensuring the prefix doesn't exist")
        if does_prefix_exist(os_artifacts):
            log("Seems like you might have a cluster already created.")
            confirmation = verify_to_proceed("Do you want to delete it?")
            if confirmation:
                destroy(os_artifacts, existing_rg)
            else:
                log("Canceling installation...")
                raise Exception("Previous cluster exists. Canceled")

    try:
        if force_new:
            log("Generating a unique prefix...")
            prefix = "farmvb" + generate_random_string()
            save_prefix(os_artifacts, prefix)
            log(f"Prefix generated {prefix}")
        else:
            prefix = get_saved_prefix(os_artifacts)

        aks_root_directory = os_artifacts.get_aks_directory()
        infra_directory = os.path.join(aks_root_directory, "modules", "infra")
        kubernetes_directory = os.path.join(aks_root_directory, "modules", "kubernetes")
        services_directory = os.path.join(aks_root_directory, "..", "services")

        log("Updating helm repos...")
        cmd = f"{os_artifacts.get_helm_cmd()} repo update"
        execute_cmd(cmd, False, False, "Couldn't update helm repos")

        # Do Infra First
        log("Executing terraform to build out infrastructure (this may take up to 30 minutes)...")
        initialize_terraform(os_artifacts, infra_directory)
        variables = {
            "tenantId": tenant_id,
            "subscriptionId": subscription_id,
            "location": region,
            "prefix": prefix,
            "kubeconfig_location": os_artifacts.get_config_directory(),
            "max_worker_nodes": MAX_WORKER_NODES,
        }

        if existing_rg:
            variables["existing_resource_group"] = existing_rg

        state_file = os_artifacts.get_terraform_file("infra.tfstate")
        apply_terraform(os_artifacts, infra_directory, state_file, variables)
        results = get_terraform_output(os_artifacts, infra_directory, state_file)

        # Do kubernetes infra now
        initialize_terraform(os_artifacts, kubernetes_directory)
        variables = {
            "tenantId": tenant_id,
            "namespace": "default",
            "acr_registry": registry_path,
            "acr_registry_username": registry_username,
            "acr_registry_password": registry_password,
            "kubernetes_config_path": os_artifacts.get_config_file("kubeconfig"),
            "kubernetes_config_context": results["kubernetes_config_context"]["value"],
            "public_ip_address": results["public_ip_address"]["value"],
            "public_ip_fqdn": results["public_ip_fqdn"]["value"],
            "public_ip_dns": results["public_ip_dns"]["value"],
            "keyvault_name": results["keyvault_name"]["value"],
            "application_id": results["application_id"]["value"],
            "storage_connection_key": results["storage_connection_key"]["value"],
            "storage_account_name": results["storage_account_name"]["value"],
            "userfile_container_name": results["userfile_container_name"]["value"],
            "resource_group_name": results["resource_group_name"]["value"],
            "certificate_email": certificate_email,
        }

        state_file = os_artifacts.get_terraform_file("kubernetes.tfstate")
        apply_terraform(os_artifacts, kubernetes_directory, state_file, variables)
        results_kb = get_terraform_output(os_artifacts, kubernetes_directory, state_file)

        # Do services now
        initialize_terraform(os_artifacts, services_directory)
        variables = {
            "namespace": "default",
            "prefix": prefix,
            "acr_registry": registry_path,
            "kubernetes_config_path": os_artifacts.get_config_file("kubeconfig"),
            "kubernetes_config_context": results["kubernetes_config_context"]["value"],
            "cache_node_pool_name": results["cache_node_pool_name"]["value"],
            "worker_node_pool_name": results["worker_node_pool_name"]["value"],
            "public_ip_fqdn": results["public_ip_fqdn"]["value"],
            "dapr_sidecars_deployed": True,
            "startup_type": "aks",
            "image_prefix": image_prefix,
            "image_tag": image_tag,
            "shared_resource_pv_claim_name": results_kb["shared_resource_pv_claim_name"]["value"],
            "worker_replicas": MAX_WORKER_NODES,
        }

        state_file = os_artifacts.get_terraform_file("services.tfstate")
        apply_terraform(os_artifacts, services_directory, state_file, variables)
    except Exception as e:
        log(f"{e.__class__.__name__}: {e}")
        log("Failed to create AKS cluster.")
        confirmation = verify_to_proceed(
            "Do you wish the destroy the cluster (Answering 'n' will leave the cluster as is)?"
        )
        if confirmation:
            destroy(os_artifacts, existing_rg)
        return

    show_url(os_artifacts, existing_rg)


def upload_file(os_artifacts: OSArtifacts, file_to_upload: str, existing_rg: str):
    prefix = get_saved_prefix(os_artifacts)
    rg_name = existing_rg if existing_rg else prefix + "-tvrg"

    log("Getting storage connection string...")
    cmd = (
        f"{os_artifacts.get_az_cmd()} storage account show-connection-string "
        f"--name {prefix}storage "
        f"--resource-group {rg_name} "
        "--query connectionString "
        "--output tsv"
    )

    connection_string = execute_cmd(cmd, True, True, "Couldn't get connection string from storage")

    log("Uploading files...")
    file_name = os.path.basename(file_to_upload)
    cmd = (
        f"{os_artifacts.get_az_cmd()} storage blob upload "
        f"--connection-string {connection_string} "
        "--container-name user-files "
        "--type block "
        "--overwrite "
        f"--name {file_name} "
        f"--file {file_to_upload}"
    )

    execute_cmd(cmd, True, False, "Failed to upload file")


def destroy(os_artifacts: OSArtifacts, existing_rg: str):
    prefix = get_saved_prefix(os_artifacts)
    rg_name = existing_rg if existing_rg else prefix + "-tvrg"
    log("Destroying cluster...")

    log("Verifying if group still exists ...")
    cmd = f'{os_artifacts.get_az_cmd()} group exists -n "{rg_name}"'
    error = "Couldn't get info of group from azure"
    result = execute_cmd(cmd, True, True, error)

    if result == "true":
        log("Group exists. Requesting destruction (this may take some time)...")
        if existing_rg:
            cmd = f'{os_artifacts.get_az_cmd()} resource list --resource-group "{rg_name}"'
            error = "Failed to get group resources. Please try again later"
            existing_resources = execute_cmd(cmd, True, False, error)
            if existing_resources:
                existing_resources_json = json.loads(existing_resources)
                id_list = ""
                for res in existing_resources_json:
                    id_list = id_list + " " + res["id"]

                cmd = (
                    f"{os_artifacts.get_az_cmd()} resource delete "
                    f'--resource-group "{rg_name}" --ids {id_list}'
                )
                error = "Failed to delete resource. Please try again later"
                execute_cmd(cmd, True, False, error)

        else:
            cmd = f'{os_artifacts.get_az_cmd()} group delete -n "{rg_name}" -y'
            error = "Failed to delete group. Please try again later"
            execute_cmd(cmd, True, False, error)

    kubeconfig_file = os_artifacts.get_config_file("kubeconfig")
    if os.path.isfile(kubeconfig_file):
        os.remove(kubeconfig_file)

    terraform_directory = os_artifacts.get_terraform_directory()
    for file in os.listdir(terraform_directory):
        os.remove(os.path.join(terraform_directory, file))

    prefix_file = os_artifacts.get_config_file(PREFIX_FILE_NAME)
    if os.path.isfile(prefix_file):
        os.remove(prefix_file)

    log("Cluster destroyed.")


def main():
    parser = argparse.ArgumentParser(description="FarmVibes.AI cluster deployment tool")
    subparsers = parser.add_subparsers(dest="cluster", help="Cluster type to manage")
    remote = subparsers.add_parser("remote", help="Remote AKS cluster management")
    subparsers.add_parser("local", help="Local cluster management")

    remote.add_argument(
        "command",
        choices=["setup", "update", "destroy", "upload_file", "show_url"],
        help="Command to execute",
    )
    remote.add_argument("--region", required=True, help="Azure region")
    remote.add_argument(
        "--file-path",
        required=False,
        help="File path to upload. Only needed for upload_file",
    )
    remote.add_argument(
        "--registry",
        required=False,
        help="Registry to overwrite where to pull images from",
    )
    remote.add_argument("--registry-username", required=False, help="Username for the registry")
    remote.add_argument("--registry-password", required=False, help="Password for the registry")
    remote.add_argument(
        "--image-prefix",
        required=False,
        help="Prefix for the image names in the registry",
    )
    remote.add_argument(
        "--image-tag", required=False, help="Image tags for the images in the registry"
    )
    remote.add_argument(
        "--cert-email", required=False, help="Email for the certificate issuing authority"
    )
    remote.add_argument("--existing-rg", required=False, help="An exisiting Azure RG to use")
    remote.add_argument("--verbose", required=False, help="Verbose logging", action="store_true")
    remote.add_argument(
        "--auto-confirm", required=False, help="Answer every question as yes", action="store_true"
    )

    args = parser.parse_args()
    if args.cluster == "local":
        raise NotImplementedError("Local cluster management not implemented yet")

    registry = DEFAULT_REGISTRY_PATH
    registry_username = ""
    registry_password = ""
    image_prefix = DEFAULT_IMAGE_PREFIX
    image_tag = DEFAULT_IMAGE_TAG

    if args.registry:
        registry = args.registry

    if args.registry_username:
        registry_username = args.registry_username

    if args.registry_password:
        registry_password = args.registry_password

    if args.image_prefix:
        image_prefix = args.image_prefix

    if args.image_tag:
        image_tag = args.image_tag

    if args.verbose:
        set_log_level(LOGGING_LEVEL_VERBOSE)

    if args.auto_confirm:
        set_auto_confirm()

    if args.command == "setup" or args.command == "update":
        if not args.cert_email:
            log("You need to specify a certificate email for setup/update with --cert-email")
            raise SystemExit

    os_artifacts = None
    if platform.system() == "Windows":
        os_artifacts = WindowsOSArtifacts()
    else:
        os_artifacts = LinuxOSArtifacts()

    os_artifacts.ensure_dependencies_installed()
    expand_azure_region(os_artifacts, args.region.strip())

    try:
        # Verify Azure CLI is logged in and has a default subscription set
        subscription_id, tenant_id = get_subscription_info(os_artifacts)
    except Exception as e:
        log(f"Error: {e}")
        log(
            "Ensure you are logged into Azure via `az login "
            "--scope https://graph.microsoft.com/.default`"
        )
        log("And set a default subscription via `az account set -s <subscription guid>`")
        raise SystemExit

    # Execute command
    try:
        process_command(
            args,
            os_artifacts,
            registry,
            registry_username,
            registry_password,
            image_prefix,
            image_tag,
            subscription_id,
            tenant_id,
        )
    except Exception as e:
        log(f"Error: {e}")
        raise SystemExit


def process_command(
    args: argparse.Namespace,
    os_artifacts: OSArtifacts,
    registry: str,
    registry_username: str,
    registry_password: str,
    image_prefix: str,
    image_tag: str,
    subscription_id: str,
    tenant_id: str,
):
    if args.command in ["setup", "update"]:
        setup_or_upgrade(
            os_artifacts,
            args.region,
            subscription_id,
            tenant_id,
            args.cert_email,
            registry,
            registry_username,
            registry_password,
            image_prefix,
            image_tag,
            args.existing_rg,
            True if args.command == "setup" else False,
        )
    elif args.command == "destroy":
        destroy(os_artifacts, args.existing_rg)
    elif args.command == "upload_file":
        upload_file(os_artifacts, args.file_path, args.existing_rg)
    elif args.command == "show_url":
        show_url(os_artifacts, args.existing_rg)
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
