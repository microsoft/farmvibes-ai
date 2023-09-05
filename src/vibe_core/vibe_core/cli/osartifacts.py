import os
import pathlib
import platform
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, NamedTuple, Optional

import pkg_resources
import requests

from .helper import execute_cmd
from .logging import log

MAJOR_MINOR_PATCH_REGEX = r"\b(?:v)?((?:\d+)(?:\.\d+)?(?:\.\d+)?)(?:(?:\+[a-zA-Z0-9]+)?)\b"


def download_file(url: str, local_path: str) -> None:
    log(f"Downloading {url} to {local_path}", "debug")
    with requests.get(url, stream=True, allow_redirects=True) as r:
        r.raise_for_status()
        with open(local_path, "wb") as f:
            shutil.copyfileobj(r.raw, f)


class DependencyError(Exception):
    pass


class InstallType(Enum):
    ALL = "all"
    LOCAL = "local"
    REMOTE = "remote"


@dataclass
class Dependency:
    name: str
    install_instructions: str
    type: InstallType
    version_argument: str = "--version"
    version_regex: Optional[str] = MAJOR_MINOR_PATCH_REGEX
    minimum_version: Optional[str] = None
    maximum_version: Optional[str] = None
    upgrade_command: Optional[str] = None


class Urls(NamedTuple):
    windows: str
    linux: str
    macos: str


class OSArtifacts:
    TERRAFORM_FOLDER_NAME = "terraform_state"
    REQUIRED_TOOLS: Dict[str, Dependency] = {
        "az": Dependency(
            "az",
            "https://learn.microsoft.com/en-us/cli/azure/install-azure-cli",
            InstallType.REMOTE,
            minimum_version="2.46.0",
            version_regex=rf"azure-cli\s+{MAJOR_MINOR_PATCH_REGEX}\s?.*",
            upgrade_command="az upgrade",
        ),
        "dapr": Dependency(
            "dapr",
            "https://docs.dapr.io/getting-started/install-dapr-cli/",
            InstallType.ALL,
            minimum_version="1.9.0",
            version_regex=rf".*:\s{MAJOR_MINOR_PATCH_REGEX}",
        ),
        "docker": Dependency(
            "docker",
            "https://docs.docker.com/get-docker/",
            InstallType.ALL,
            minimum_version="20.10.0",
        ),
        "helm": Dependency(
            "helm",
            "https://helm.sh/docs/intro/install/",
            InstallType.ALL,
            version_argument="version",
            minimum_version="3.0.0",
        ),
        "k3d": Dependency(
            "k3d", "https://k3d.io/#installation", InstallType.LOCAL, minimum_version="5.5.0"
        ),
        "kubectl": Dependency(
            "kubectl",
            "https://kubernetes.io/docs/tasks/tools/install-kubectl/",
            InstallType.ALL,
            version_argument="version --client --output=yaml",
            minimum_version="1.27.0",
            version_regex=rf"gitVersion:\s+{MAJOR_MINOR_PATCH_REGEX}",
        ),
        "kubelogin": Dependency(
            "kubelogin",
            "https://azure.github.io/kubelogin/",
            InstallType.REMOTE,
            minimum_version="0.0.30",
        ),
        "terraform": Dependency(
            "terraform",
            "https://www.terraform.io/downloads.html",
            InstallType.ALL,
            version_argument="version",
            minimum_version="1.0.2",
        ),
    }

    def __init__(self):
        self._local_terraform_path = ""
        self._aks_terraform_path = ""

    def check_dependencies(self, type: InstallType = InstallType.ALL) -> None:
        for dependency in self.REQUIRED_TOOLS.values():
            if type == InstallType.ALL or dependency.type in {type, InstallType.ALL}:
                self.check_dependency(dependency)

    def get_version(self, dependency: Dependency, full_path: pathlib.Path) -> str:
        version = subprocess.check_output(
            [str(full_path)] + shlex.split(dependency.version_argument), universal_newlines=True
        )
        if dependency.version_regex is not None:
            versions: List[str] = re.findall(dependency.version_regex, version)
            if not versions:
                msg = f"Could not find version of {dependency.name}"
                log(msg, level="error")
                raise DependencyError(msg)
            version = versions[0]
        return version

    def is_supported_version(self, dependency: Dependency, full_path: pathlib.Path) -> bool:
        try:
            if dependency.version_regex is not None and dependency.minimum_version is not None:
                version = self.get_version(dependency, full_path)
                if self.verify_min_version(version, dependency.minimum_version):
                    return True
                else:
                    return False
            else:
                # No version check, assume it's good
                return True
        except Exception:
            log(f"Could not check version of {full_path}", level="debug")
            return False

    def check_dependency(self, dependency: Dependency) -> None:
        log(f"Checking dependency {dependency.name}", level="debug")
        system_found: bool = False
        reason: str
        if hasattr(self, dependency.name):
            # Dependency will either be a full path, or a binary name, have to use an absolute path
            full_path = pathlib.Path(getattr(self, dependency.name))
            if full_path.is_absolute():
                system_found = True
            else:
                full_path = self.config_dir / getattr(self, dependency.name)
            if (full_path).exists() and self.is_supported_version(dependency, full_path):
                version_modifier: str = ""
                if dependency.minimum_version is not None:
                    version_modifier = f">= {dependency.minimum_version}"
                log(
                    f"Dependency {dependency.name} found at {full_path} with supported version "
                    f"{version_modifier}",
                    level="debug",
                )
                return

            if system_found and dependency.upgrade_command is not None:
                log(
                    f"Trying to use native upgrade command for {dependency.name}: "
                    f'"{dependency.upgrade_command}"'
                )
                process = subprocess.run(shlex.split(dependency.upgrade_command))
                if process.returncode != 0:
                    log(
                        f"Unable to upgrade {dependency.name} with native command "
                        f'"{dependency.upgrade_command}"',
                        level="warning",
                    )
                else:
                    log(
                        f"Successfully upgraded {dependency.name} with native command "
                        f'"{dependency.upgrade_command}"',
                        level="debug",
                    )
                    return

            if hasattr(self, f"install_{dependency.name}"):
                if system_found:
                    reason = "found at system level, but is unsupported"
                else:
                    reason = "not found"
                log(f"Dependency {dependency.name} {reason}, trying to install", level="debug")
                try:
                    getattr(self, f"install_{dependency.name}")()
                    return
                except Exception:
                    error_msg = (
                        f"Dependency {dependency.name} not found "
                        "and I failed to install it. "
                        f"Please install {dependency.name} manually. "
                        f"({dependency.install_instructions})"
                    )
                    log(error_msg, level="error")
                    raise DependencyError(error_msg)
            else:
                error_msg = (
                    f"Dependency {dependency.name} not found "
                    "and I don't know how to install it. "
                    f"Please install {dependency.name} manually. "
                    f"({dependency.install_instructions})"
                )
                log(error_msg, level="error")
                raise DependencyError(error_msg)
        else:
            if shutil.which(dependency.name) is not None:
                log(f"Dependency {dependency.name} found", level="debug")
                return
        error_msg = (
            f"Internal error: Dependency {dependency.name} not found and no install method found"
        )
        log(error_msg, level="error")
        raise RuntimeError(error_msg)

    @property
    def config_dir(self):
        if "FARMVIBES_AI_CONFIG_DIR" in os.environ:
            ret = pathlib.Path(os.environ["FARMVIBES_AI_CONFIG_DIR"]).expanduser()
        elif "XDG_HOME" in os.environ:
            ret = pathlib.Path(os.environ["XDG_HOME"]).expanduser() / ".config" / "farmvibes-ai"
        else:
            ret = (pathlib.Path("~") / ".config" / "farmvibes-ai").expanduser()
        if not ret.exists():
            log(f"Creating config directory {ret}")
            ret.mkdir(exist_ok=True, parents=True)
        return ret

    @property
    def az(self) -> str:
        az = shutil.which("az")
        if az is None:
            log("az not found in PATH", level="debug")
        return az or "not found"

    def _binary(self, name: str) -> str:
        base = str((pathlib.Path(self.config_dir) / name))
        fallback = base if platform.system() != "Windows" else f"{base}.exe"
        candidate = shutil.which(name)
        if not candidate:
            return fallback
        # There's a system-installed version, do we support it?
        # Before checking, make sure to escape any spaces (I'm looking at you, Windows)
        if self.is_supported_version(self.REQUIRED_TOOLS[name], pathlib.Path(candidate)):
            return candidate
        else:
            return fallback

    @property
    def dapr(self) -> str:
        return self._binary("dapr")

    @property
    def docker(self) -> str:
        return self._binary("docker")

    @property
    def helm(self) -> str:
        return self._binary("helm")

    @property
    def k3d(self) -> str:
        return self._binary("k3d")

    @property
    def kubectl(self) -> str:
        return self._binary("kubectl")

    @property
    def kubelogin(self) -> str:
        return self._binary("kubelogin")

    @property
    def terraform(self) -> str:
        return self._binary("terraform")

    def config_file(self, file_name: str) -> str:
        return str((pathlib.Path(self.config_dir) / file_name))

    @property
    def terraform_directory(self) -> str:
        path = pathlib.Path(self.config_dir) / self.TERRAFORM_FOLDER_NAME
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def get_terraform_file(self, file_name: str) -> str:
        return os.path.join(self.terraform_directory, file_name)

    def verify_min_version(self, tool_version: str, expected_version: str) -> bool:
        # We should ideally use a package that does this, but we don't want to have package
        # dependencies that we can avoid
        if not tool_version or not expected_version:
            return False

        if tool_version[0] == "v":
            tool_version = tool_version[1:]

        try:
            tool_version = tool_version.split("+")[0]

            current_versions = tool_version.split(".")
            wanted_versions = expected_version.split(".")

            zipped_versions = zip(current_versions, wanted_versions)
            for zv in zipped_versions:
                cv = int(zv[0])
                wv = int(zv[1])

                if wv > cv:
                    return False
                elif cv > wv:
                    return True

            if len(current_versions) >= len(wanted_versions):
                return True
        except Exception:
            log(f"Couldn't parse version {tool_version}")

        return False

    @property
    def terraform_base(self) -> str:
        terraform_dir = os.path.abspath(
            pkg_resources.resource_filename(__name__, os.path.join("..", "terraform"))
        )
        return terraform_dir

    def _resolve_terraform_directory(self, directory: str) -> str:
        if not os.access(directory, os.W_OK) or "site-packages" in directory:
            log(
                "Terrafom directory not writable or in site-packages, copying to local directory",
                level="debug",
            )
            local_dir = self.config_dir / "terraform-user"
            shutil.copytree(os.path.dirname(directory), local_dir, dirs_exist_ok=True)
            return str(local_dir / os.path.basename(directory))
        return directory

    @property
    def aks_directory(self) -> str:
        if not self._aks_terraform_path:
            self._aks_terraform_path = self._resolve_terraform_directory(
                os.path.join(self.terraform_base, "aks")
            )
        return self._aks_terraform_path

    @property
    def local_directory(self) -> str:
        if not self._local_terraform_path:
            self._local_terraform_path = self._resolve_terraform_directory(
                os.path.join(self.terraform_base, "local")
            )
        return self._local_terraform_path

    def install_docker(self) -> None:
        installer = DockerCliInstaller(self.config_dir)
        installer.install()

    def install_k3d(self) -> None:
        installer = K3dInstaller(self.config_dir)
        installer.install()

    def install_dapr(self) -> None:
        installer = DaprInstaller(self.config_dir)
        installer.install()

    def install_kubectl(self) -> None:
        installer = KubectlInstaller(self.config_dir)
        installer.install()

    def install_kubelogin(self) -> None:
        installer = KubeloginInstaller(self.config_dir)
        installer.install()

    def install_helm(self) -> None:
        installer = HelmInstaller(self.config_dir)
        installer.install()

    def install_terraform(self) -> None:
        installer = TerraformInstaller(self.config_dir)
        installer.install()

    def install_az(self) -> None:
        installer = AzCliInstaller(self.config_dir)
        installer.install()

    @contextmanager
    def kube_context(self, context_name: str):
        current_context = self.get_kube_context()
        log(f"Current kubectl context is {current_context}", level="debug")
        log(f"Setting kubectl context to {context_name}", level="debug")
        self.set_kube_context(context_name)
        try:
            yield
        finally:
            self.set_kube_context(current_context)

    def set_kube_context(self, context: str) -> None:
        log("Setting kubectl context to {}".format(context), level="debug")
        cmd = [self.kubectl, "config", "use-context", context]
        error = f"Couldn't set kubectl context {context}"
        execute_cmd(cmd, True, False, error, capture_output=False, subprocess_log_level="debug")

    def get_kube_context(self) -> str:
        cmd = [self.kubectl, "config", "current-context"]
        error = "Couldn't get kubectl current context"
        return execute_cmd(
            cmd, True, True, error, capture_output=True, subprocess_log_level="debug"
        )


class Installer(ABC):
    def __init__(self, config_dir: Optional[pathlib.Path]) -> None:
        self.config_dir = config_dir

    def install(self) -> None:
        if platform.system() == "Windows":
            self.install_windows()
        elif platform.system() == "Linux":
            self.install_linux()
        elif platform.system() == "Darwin":
            self.install_macos()
        else:
            raise NotImplementedError(f"Unsupported platform {platform.system()}")

    @abstractmethod
    def install_windows(self) -> None:
        raise NotImplementedError("install_windows is not implemented")

    @abstractmethod
    def install_linux(self) -> None:
        raise NotImplementedError("install_linux is not implemented")

    @abstractmethod
    def install_macos(self) -> None:
        raise NotImplementedError("install_macos is not implemented")


class AzCliInstaller(Installer):
    def install_windows(self) -> None:
        if not shutil.which("winget"):
            raise DependencyError(
                "winget not found in PATH. Please follow the instructions at "
                "https://docs.microsoft.com/en-us/windows/package-manager/winget/ to install it."
            )
        log("Installing Azure CLI using winget")
        try:
            subprocess.check_output(["winget", "install", "Microsoft.AzureCLI"], shell=True)
        except subprocess.CalledProcessError as e:
            log(f"Failed to install Azure CLI using winget: {e}", level="error")
            raise

    def install_linux(self) -> None:
        def is_ubuntu_or_debian():
            if shutil.which("lsb_release"):
                distribution = subprocess.check_output(["lsb_release", "-a"]).decode()
                if "ubuntu" in distribution.lower() or "debian" in distribution.lower():
                    return True
            return False

        if not shutil.which("curl") or not shutil.which("bash"):
            raise DependencyError(
                "curl or bash not found in PATH. Please install them using your package manager "
                "or install Azure CLI manually using "
                "https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-linux"
            )
        log("Installing Azure CLI using curl")
        try:
            with tempfile.NamedTemporaryFile() as f:
                if not is_ubuntu_or_debian():
                    raise DependencyError(
                        "Automatic Installation of Azure CLI is only supported "
                        "on Ubuntu or Debian. "
                        "Please install the Azure CLI manually using "
                        "https://learn.microsoft.com/en-us/cli/azure/install-azure-cli-linux"
                    )

                download_file(
                    "https://aka.ms/InstallAzureCLIDeb",
                    f.name,
                )
                subprocess.check_output(["bash", f.name, "-y"], shell=True)
        except subprocess.CalledProcessError as e:
            log(f"Failed to install Azure CLI: {e}", level="error")
            raise

    def install_macos(self) -> None:
        if not shutil.which("brew"):
            raise DependencyError(
                "brew not found in PATH. Please follow the instructions at "
                "https://brew.sh/ to install it. "
                "Then run `brew install azure-cli` to install Azure CLI."
            )
        log("Installing Azure CLI using brew")
        try:
            subprocess.check_output(["brew", "install", "azure-cli"], shell=True)
        except subprocess.CalledProcessError as e:
            log(f"Failed to install Azure CLI using brew: {e}", level="error")
            raise


class PrivateCliToolInstaller(Installer, ABC):
    @property
    @abstractmethod
    def urls(self) -> Urls:
        raise NotImplementedError("urls is not implemented")

    @property
    @abstractmethod
    def cli_name(self) -> str:
        raise NotImplementedError("cli_name is not implemented")

    @property
    def arch(self) -> str:
        return "amd64" if platform.machine().lower() in {"x86_64", "amd64"} else platform.machine()

    def install_helper(self, url: str, file_name: str) -> None:
        log(f"Downloading {file_name} from {url}")

        if self.config_dir is None:
            raise RuntimeError(f"config_dir is None, unable to install {file_name}")
        final_path = self.config_dir / file_name

        try:
            # Download to temp dir
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file = os.path.join(temp_dir, os.path.basename(url))
                download_file(url, temp_file)
                # If this is a zip, we need to unzip it and find the binary
                if not url.endswith(".zip") and not temp_file.endswith("gz"):
                    shutil.copy(temp_file, final_path)
                    return
                elif url.endswith(".zip"):
                    log(f"Extracting {file_name} from zip", level="debug")
                    with zipfile.ZipFile(temp_file, "r") as zip_ref:
                        zip_ref.extractall(temp_dir)
                # Might be a tar.gz
                elif url.endswith("gz"):
                    log(f"Extracting {file_name} from tar.gz", level="debug")
                    with tarfile.open(temp_file, "r:gz") as tar_ref:
                        tar_ref.extractall(temp_dir)
                else:
                    raise RuntimeError(f"Unknown file type for {file_name}")

                # Find the binary and copy it to the final path
                log(f"Searching for {file_name} in {temp_dir}", level="debug")
                for root, _, files in os.walk(temp_dir):
                    log(
                        f"Walking {root} with files {files} looking for {self.cli_name}",
                        level="debug",
                    )
                    for file in files:
                        if file == self.cli_name:
                            log(f"Found {file}. Copying it to {final_path}", level="debug")
                            shutil.copy(os.path.join(root, file), final_path)
                            break
        finally:
            if final_path.exists():
                log(f"Successfully installed {file_name} to {final_path}")
                if platform.system() != "Windows":
                    final_path.chmod(0o755)
            else:
                log(f"Failed to install {self.cli_name} to {final_path}", level="error")

    def install_windows(self) -> None:
        self.install_helper(self.urls.windows, self.cli_name)

    def install_linux(self) -> None:
        self.install_helper(self.urls.linux, self.cli_name)

    def install_macos(self) -> None:
        self.install_helper(self.urls.macos, self.cli_name)


class TerraformInstaller(PrivateCliToolInstaller):
    TERRAFORM_RELEASE_URL = "https://api.github.com/repos/hashicorp/terraform/releases/latest"
    TERRAFORM_BASE_URL = "https://releases.hashicorp.com/terraform"

    @property
    def latest_release(self) -> str:
        try:
            response = requests.get(self.TERRAFORM_RELEASE_URL)
            response.raise_for_status()
            return response.json()["tag_name"].replace("v", "")
        except Exception:
            log("Failed to get latest Terraform release", level="error")
            raise

    @property
    def urls(self) -> Urls:
        latest_release = self.latest_release
        arch = self.arch.replace("i386", "386")
        base = f"{self.TERRAFORM_BASE_URL}/{latest_release}/terraform_{latest_release}"

        return Urls(
            windows=f"{base}_windows_{arch}.zip",
            linux=f"{base}_linux_{arch}.zip",
            macos=f"{base}_darwin_{arch}.zip",
        )

    @property
    def cli_name(self) -> str:
        return "terraform" if platform.system() != "Windows" else "terraform.exe"


class KubectlInstaller(PrivateCliToolInstaller):
    KUBECTL_RELEASE_URL = "https://storage.googleapis.com/kubernetes-release/release/stable.txt"
    KUBECTL_BASE_URL = "https://storage.googleapis.com/kubernetes-release/release"

    @property
    def latest_release(self) -> str:
        try:
            response = requests.get(self.KUBECTL_RELEASE_URL)
            response.raise_for_status()
            return response.text.strip()
        except Exception:
            log("Failed to get latest kubectl release", level="error")
            raise

    @property
    def urls(self) -> Urls:
        latest_release = self.latest_release
        arch = self.arch.replace("i386", "386")
        base = f"{self.KUBECTL_BASE_URL}/{latest_release}/bin"
        cli_name = self.cli_name

        return Urls(
            windows=f"{base}/windows/{arch}/{cli_name}",
            linux=f"{base}/linux/{arch}/{cli_name}",
            macos=f"{base}/darwin/{arch}/{cli_name}",
        )

    @property
    def cli_name(self) -> str:
        return "kubectl" if platform.system() != "Windows" else "kubectl.exe"


class HelmInstaller(PrivateCliToolInstaller):
    HELM_RELEASE_URL = "https://api.github.com/repos/helm/helm/releases/latest"
    HELM_BASE_URL = "https://get.helm.sh"

    @property
    def latest_release(self) -> str:
        try:
            response = requests.get(self.HELM_RELEASE_URL)
            response.raise_for_status()
            return response.json()["tag_name"]
        except Exception:
            log("Failed to get latest Helm release", level="error")
            raise

    @property
    def urls(self) -> Urls:
        latest_release = self.latest_release
        arch = self.arch.replace("i386", "386")
        base = f"{self.HELM_BASE_URL}/helm-{latest_release}"

        return Urls(
            windows=f"{base}-windows-{arch}.zip",
            linux=f"{base}-linux-{arch}.tar.gz",
            macos=f"{base}-darwin-{arch}.tar.gz",
        )

    @property
    def cli_name(self) -> str:
        return "helm" if platform.system() != "Windows" else "helm.exe"


class K3dInstaller(PrivateCliToolInstaller):
    K3D_RELEASE_URL = "https://api.github.com/repos/k3d-io/k3d/releases/latest"
    K3D_BASE_URL = "https://github.com/k3d-io/k3d/releases/download"

    @property
    def latest_release(self) -> str:
        try:
            response = requests.get(self.K3D_RELEASE_URL)
            response.raise_for_status()
            return response.json()["tag_name"]
        except Exception:
            log("Failed to get latest k3d release", level="error")
            raise

    @property
    def cli_name(self) -> str:
        return "k3d" if platform.system() != "Windows" else "k3d.exe"

    @property
    def urls(self) -> Urls:
        latest_release = self.latest_release
        arch = self.arch.replace("i386", "386")
        base = f"{self.K3D_BASE_URL}/{latest_release}"

        return Urls(
            windows=f"{base}/k3d-windows-{arch}.exe",
            linux=f"{base}/k3d-linux-{arch}",
            macos=f"{base}/k3d-darwin-{arch}",
        )


class KubeloginInstaller(PrivateCliToolInstaller):
    KUBELOGIN_RELEASE_URL = "https://api.github.com/repos/Azure/kubelogin/releases/latest"
    KUBELOGIN_BASE_URL = "https://github.com/Azure/kubelogin/releases/download"

    @property
    def latest_release(self) -> str:
        try:
            response = requests.get(self.KUBELOGIN_RELEASE_URL)
            response.raise_for_status()
            return response.json()["tag_name"]
        except Exception:
            log("Failed to get latest kubelogin release", level="error")
            raise

    @property
    def cli_name(self) -> str:
        return "kubelogin" if platform.system() != "Windows" else "kubelogin.exe"

    @property
    def urls(self) -> Urls:
        latest_release = self.latest_release
        arch = self.arch.replace("i386", "386")
        base = f"{self.KUBELOGIN_BASE_URL}/{latest_release}"

        return Urls(
            windows=f"{base}/kubelogin-win-{arch}.zip",
            linux=f"{base}/kubelogin-linux-{arch}.zip",
            macos=f"{base}/kubelogin-darwin-{arch}.zip",
        )


class DockerCliInstaller(PrivateCliToolInstaller):
    DOCKER_CLI_URL = "https://download.docker.com/{os}/static/stable/{arch}/docker-24.0.4.{ext}"

    @property
    def arch(self) -> str:
        arch = platform.machine().lower().replace("arm64", "aarch64").replace("amd64", "x86_64")
        return arch

    @property
    def cli_name(self) -> str:
        return "docker" if platform.system() != "Windows" else "docker.exe"

    @property
    def urls(self) -> Urls:
        arch = self.arch.replace("i386", "386")

        return Urls(
            windows=self.DOCKER_CLI_URL.format(os="win", arch=arch, ext="zip"),
            linux=self.DOCKER_CLI_URL.format(os="linux", arch=arch, ext="tgz"),
            macos=self.DOCKER_CLI_URL.format(os="mac", arch=arch, ext="tgz"),
        )


class DaprInstaller(PrivateCliToolInstaller):
    DAPR_RELEASE_URL = "https://api.github.com/repos/dapr/cli/releases/latest"
    DAPR_BASE_URL = "https://github.com/dapr/cli/releases/download"

    @property
    def latest_release(self) -> str:
        try:
            response = requests.get(self.DAPR_RELEASE_URL)
            response.raise_for_status()
            return response.json()["tag_name"]
        except Exception:
            log("Failed to get latest dapr release", level="error")
            raise

    @property
    def cli_name(self) -> str:
        return "dapr" if platform.system() != "Windows" else "dapr.exe"

    @property
    def urls(self) -> Urls:
        latest_release = self.latest_release
        arch = self.arch.replace("i386", "386")
        base = f"{self.DAPR_BASE_URL}/{latest_release}"

        return Urls(
            windows=f"{base}/dapr_windows_{arch}.zip",
            linux=f"{base}/dapr_linux_{arch}.tar.gz",
            macos=f"{base}/dapr_darwin_{arch}.tar.gz",
        )
