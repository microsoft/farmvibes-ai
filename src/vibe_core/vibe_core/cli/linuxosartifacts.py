import os
import pathlib
import platform
import tarfile
import zipfile
from typing import Dict, List, Union

from .constants import (
    AUTO_INSTALL_AVAILABLE_NAME,
    INSTALL_ACTION_NAME,
    KUBECTL_BASE_PATH,
    UPGRADE_ACTION_NAME,
    VERSION_SELECTOR_CMD_NAME,
)
from .osartifacts import OSArtifacts
from .helper import (
    execute_cmd,
    get_latest_helm_version,
    get_latest_kubectl_version,
    get_latest_terraform_version,
)
from .logging import log

ANCILLARY_TOOLS_REQUIRED_NAME = "ancillary_tools"

REQUIRED_TOOLS_LINUX = {
    "terraform": {
        AUTO_INSTALL_AVAILABLE_NAME: True,
        INSTALL_ACTION_NAME: (
            "https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli"
        ),
        UPGRADE_ACTION_NAME: "",
        VERSION_SELECTOR_CMD_NAME: "version | awk '/^Terraform v/{print $2}'",
        ANCILLARY_TOOLS_REQUIRED_NAME: ["unzip"],
    },
    "az": {
        AUTO_INSTALL_AVAILABLE_NAME: False,
        INSTALL_ACTION_NAME: "curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash",
        UPGRADE_ACTION_NAME: "az upgrade",
        VERSION_SELECTOR_CMD_NAME: "--version | awk '/^azure-cli/{print $2}'",
        ANCILLARY_TOOLS_REQUIRED_NAME: [],
    },
    "helm": {
        AUTO_INSTALL_AVAILABLE_NAME: True,
        INSTALL_ACTION_NAME: "https://helm.sh/docs/intro/install/",
        UPGRADE_ACTION_NAME: "",
        VERSION_SELECTOR_CMD_NAME: "version --short",
        ANCILLARY_TOOLS_REQUIRED_NAME: [],
    },
    "kubectl": {
        AUTO_INSTALL_AVAILABLE_NAME: True,
        INSTALL_ACTION_NAME: "https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/",
        UPGRADE_ACTION_NAME: "",
        VERSION_SELECTOR_CMD_NAME: (
            "version --short --client=true | awk '/^Client Version: /{print $3}'"
        ),
        ANCILLARY_TOOLS_REQUIRED_NAME: [],
    },
}

HOME = os.path.expanduser("~")
FARMVIBES_CONFIG_DIR = os.path.join(HOME, ".config", "farmvibes-ai")


class LinuxOSArtifacts(OSArtifacts):
    def __init__(self):
        pathlib.Path(FARMVIBES_CONFIG_DIR).mkdir(parents=True, exist_ok=True)

    def get_config_directory(self) -> str:
        return FARMVIBES_CONFIG_DIR

    def get_aks_directory(self) -> str:
        return "resources/terraform/aks"

    def get_os_tool_definition(self) -> Dict[str, Dict[str, Union[bool, str, List[str]]]]:
        return REQUIRED_TOOLS_LINUX

    def install_dependency(self, tool: str):
        arch = platform.machine()
        if arch == "AMD64":
            arch_to_use = "amd64"
        else:
            arch_to_use = "386"
        if tool == "terraform":
            self.install_terraform(arch_to_use)
        elif tool == "kubectl":
            self.install_kubectl(arch_to_use)
        elif tool == "helm":
            self.install_helm(arch_to_use)

    def install_terraform(self, arch_to_use: str):
        terraform_version = get_latest_terraform_version()
        terraform_zip = self.get_config_file("terraform.zip")

        try:
            cmd = (
                f"curl https://releases.hashicorp.com/terraform/{terraform_version}/"
                f"terraform_{terraform_version}_linux_{arch_to_use}.zip > {terraform_zip}"
            )
            execute_cmd(cmd, True, False, "Failed to download terraform")
            with zipfile.ZipFile(terraform_zip, "r") as zip_ref:
                zip_ref.extractall(self.get_config_directory())
                os.chmod(self.get_config_file("terraform"), 0o777)
        except Exception:
            raise ValueError("Failed to acquire terraform")
        finally:
            if os.path.isfile(terraform_zip):
                os.remove(terraform_zip)

    def install_helm(self, arch_to_use: str):
        helm_version = get_latest_helm_version()
        helm_zip = self.get_config_file("helm.tar.gz")
        tar_file = None

        try:
            cmd = (
                f"curl -L https://get.helm.sh/helm-v{helm_version}-linux-{arch_to_use}.tar.gz "
                f"> {helm_zip}"
            )
            execute_cmd(cmd, True, False, "Failed to download helm")
            tar_file = tarfile.open(helm_zip)
            tar_file.extractall(self.get_config_directory())
            helm_path = os.path.join(self.get_config_directory(), f"linux-{arch_to_use}", "helm")
            execute_cmd(
                f'mv {helm_path} {self.get_config_file("helm")}', True, False, "Failed to move helm"
            )
        except Exception:
            raise ValueError("Failed to acquire helm")
        finally:
            if tar_file:
                tar_file.close()

            if os.path.isfile(helm_zip):
                os.remove(helm_zip)

    def install_kubectl(self, arch_to_use: str):
        kubectl_version = get_latest_kubectl_version()
        kubectl = self.get_config_file("kubectl")

        try:
            cmd = (
                f'curl -L "{KUBECTL_BASE_PATH}/{kubectl_version}/bin/linux/{arch_to_use}/kubectl" '
                f"> {kubectl}"
            )
            execute_cmd(cmd, True, False, "Failed to download kubectl")
            os.chmod(kubectl, 0o777)
        except Exception:
            raise ValueError("Failed to acquire kubectl")

    def get_version(self, tool: str, path: str, version_cmd: str) -> str:
        try:
            cmd = f'"{path.strip()}" {version_cmd.strip()}'
            error = f"Failed to execute command to get current tool version for {tool} at {path}"
            return execute_cmd(cmd, True, True, error)
        except Exception:
            # Had trouble parsing. Stop
            log(f"We couldn't parse the version information for {tool}")
            return "0.0"
