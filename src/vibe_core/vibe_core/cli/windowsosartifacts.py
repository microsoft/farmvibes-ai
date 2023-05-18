import os
import pathlib
import platform
import zipfile
from typing import Dict, List, Union

from .constants import (
    ALTERNATE_NAMES,
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

REQUIRED_TOOLS_WINDOWS = {
    "terraform": {
        AUTO_INSTALL_AVAILABLE_NAME: True,
        INSTALL_ACTION_NAME: (
            "https://developer.hashicorp.com/terraform/tutorials/aws-get-started/install-cli"
        ),
        UPGRADE_ACTION_NAME: "",
        VERSION_SELECTOR_CMD_NAME: (
            "version -json | ConvertFrom-Json | Select-Object -ExpandProperty 'terraform_version'"
        ),
        ALTERNATE_NAMES: "terraform.exe",
    },
    "az": {
        AUTO_INSTALL_AVAILABLE_NAME: False,
        INSTALL_ACTION_NAME: "https://aka.ms/installazurecliwindows",
        UPGRADE_ACTION_NAME: "az --upgrade",
        VERSION_SELECTOR_CMD_NAME: (
            "version | ConvertFrom-Json | Select-Object -ExpandProperty 'azure-cli'"
        ),
    },
    "helm": {
        AUTO_INSTALL_AVAILABLE_NAME: True,
        INSTALL_ACTION_NAME: "https://helm.sh/docs/intro/install/",
        UPGRADE_ACTION_NAME: "",
        VERSION_SELECTOR_CMD_NAME: "version --short",
        ALTERNATE_NAMES: "helm.exe",
    },
    "kubectl": {
        AUTO_INSTALL_AVAILABLE_NAME: True,
        INSTALL_ACTION_NAME: "https://kubernetes.io/docs/tasks/tools/install-kubectl-windows/",
        UPGRADE_ACTION_NAME: "",
        VERSION_SELECTOR_CMD_NAME: (
            "version --client=true -o json | ConvertFrom-Json | "
            "Select-Object -ExpandProperty 'clientVersion' | ConvertTo-Json | "
            "ConvertFrom-Json | Select-Object -ExpandProperty 'gitVersion'"
        ),
        ALTERNATE_NAMES: "kubectl.exe",
    },
}

HOME = os.path.expanduser("~")
FARMVIBES_CONFIG_DIR = os.path.join(HOME, ".config", "farmvibes-ai")


class WindowsOSArtifacts(OSArtifacts):
    def __init__(self):
        pathlib.Path(FARMVIBES_CONFIG_DIR).mkdir(parents=True, exist_ok=True)

    def get_config_directory(self) -> str:
        return FARMVIBES_CONFIG_DIR

    def get_aks_directory(self) -> str:
        return "resources\\terraform\\aks"

    def get_os_tool_definition(self) -> Dict[str, Dict[str, Union[bool, str, List[str]]]]:
        return REQUIRED_TOOLS_WINDOWS

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
                "curl https://releases.hashicorp.com/terraform/"
                f"{terraform_version}/terraform_{terraform_version}_windows_{arch_to_use}.zip "
                f"> {terraform_zip}"
            )
            execute_cmd(cmd, True, False, "Failed to download terraform")
            with zipfile.ZipFile(terraform_zip, "r") as zip_ref:
                zip_ref.extractall(self.get_config_directory())
        except Exception:
            raise ValueError("Failed to acquire terraform")
        finally:
            if os.path.isfile(terraform_zip):
                os.remove(terraform_zip)

    def install_helm(self, arch_to_use: str):
        helm_version = get_latest_helm_version()
        helm_zip = self.get_config_file("helm.zip")

        try:
            cmd = (
                "curl -L https://get.helm.sh/"
                f"helm-v{helm_version}-windows-{arch_to_use}.zip > {helm_zip}"
            )
            execute_cmd(cmd, True, False, "Failed to download helm")
            with zipfile.ZipFile(helm_zip, "r") as zip_ref:
                zip_ref.extractall(self.get_config_directory())
            helm_path = os.path.join(
                self.get_config_directory(), f"windows-{arch_to_use}", "helm.exe"
            )
            execute_cmd(
                f'move {helm_path} {self.get_config_file("helm.exe")}',
                True,
                False,
                "Failed to move helm",
            )
        except Exception:
            raise ValueError("Failed to acquire helm")
        finally:
            if os.path.isfile(helm_zip):
                os.remove(helm_zip)

    def install_kubectl(self, arch_to_use: str):
        kubectl_version = get_latest_kubectl_version()
        kubectl = self.get_config_file("kubectl.exe")

        try:
            cmd = (
                f'curl -L "{KUBECTL_BASE_PATH}/{kubectl_version}/bin/windows/'
                f'{arch_to_use}/kubectl.exe" > {kubectl}'
            )
            execute_cmd(cmd, True, False, "Failed to download kubectl")
        except Exception:
            raise ValueError("Failed to acquire kubectl")

    def get_version(self, tool: str, path: str, version_cmd: str) -> str:
        try:
            cmd = f"powershell.exe -Command \"&'{path.strip()}' {version_cmd.strip()}\""
            error = f"Failed to execute command to get current tool version for {tool} at {path}"
            return execute_cmd(cmd, True, True, error)
        except Exception:
            # Had trouble parsing. Stop
            log(f"We couldn't parse the version information for {tool}")
            return "0.0"
