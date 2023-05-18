import os
import pathlib
import shutil
from abc import ABC, abstractmethod
from typing import Dict, List, Union, cast

from .helper import verify_to_proceed
from .logging import log
from .constants import (
    ALTERNATE_NAMES,
    AUTO_INSTALL_AVAILABLE_NAME,
    FRIENDLY_NAME,
    INSTALL_ACTION_NAME,
    MIN_VERSION_REQUIRED_NAME,
    REQUIRED_TOOLS,
    TERRAFORM_FOLDER_NAME,
    UPGRADE_ACTION_NAME,
    VERSION_SELECTOR_CMD_NAME,
)


class OSArtifacts(ABC):
    paths_selected = {}

    def get_az_cmd(self) -> str:
        return self.paths_selected["az"]

    def get_kubectl_cmd(self) -> str:
        return self.paths_selected["kubectl"]

    def get_helm_cmd(self) -> str:
        return self.paths_selected["helm"]

    def get_terraform_cmd(self) -> str:
        return self.paths_selected["terraform"]

    def get_config_file(self, file_name: str) -> str:
        return os.path.join(self.get_config_directory(), file_name)

    def get_terraform_directory(self) -> str:
        terraform_dir = os.path.join(self.get_config_directory(), TERRAFORM_FOLDER_NAME)
        pathlib.Path(terraform_dir).mkdir(parents=True, exist_ok=True)
        return terraform_dir

    def get_terraform_file(self, file_name: str) -> str:
        terraform_dir = self.get_terraform_directory()
        return os.path.join(terraform_dir, file_name)

    def register_tool_path(self, tool: str, path: str):
        self.paths_selected[tool] = f'"{path}"'

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

    def ensure_dependencies_installed(self):
        log("Verifying all required tools are present")
        os_tool_definitions = self.get_os_tool_definition()
        MAX_ATTEMPTS = 3
        for tool in REQUIRED_TOOLS.keys():
            friendly_name = REQUIRED_TOOLS[tool][FRIENDLY_NAME]
            min_version_required = REQUIRED_TOOLS[tool][MIN_VERSION_REQUIRED_NAME]
            path_suggested = ""
            for i in range(0, MAX_ATTEMPTS):
                log(f"Verifying {friendly_name} ...")
                if tool not in os_tool_definitions:
                    raise ValueError(
                        f"{tool} is not defined in OS Specific Artifacts. Unable to verify. "
                        "This is a bug in the FarmVibes.AI code."
                    )

                os_tool = os_tool_definitions[tool]
                version = cast(str, os_tool[VERSION_SELECTOR_CMD_NAME])
                found_path = False

                if path_suggested:
                    path = path_suggested
                    if path and self.verify_min_version(
                        self.get_version(tool, path, version),
                        min_version_required,
                    ):
                        self.register_tool_path(tool, path)
                        break

                # Check if present in system
                path = shutil.which(tool)
                found_path = found_path or path

                if path and self.verify_min_version(
                    self.get_version(tool, path, version),
                    min_version_required,
                ):
                    self.register_tool_path(tool, path)
                    break

                # Not found in system. Check if config dir has it
                path = self.get_config_file(tool)
                found_path = found_path or os.path.isfile(path)

                if (
                    path
                    and os.path.isfile(path)
                    and self.verify_min_version(
                        self.get_version(tool, path, version),
                        min_version_required,
                    )
                ):
                    self.register_tool_path(tool, path)
                    break

                # Check if tool has alternative name
                if ALTERNATE_NAMES in os_tool.keys() and os_tool[ALTERNATE_NAMES]:
                    path = self.get_config_file(cast(str, os_tool[ALTERNATE_NAMES]))
                    found_path = found_path or os.path.isfile(path)

                    if (
                        path
                        and os.path.isfile(path)
                        and self.verify_min_version(
                            self.get_version(tool, path, version),
                            min_version_required,
                        )
                    ):
                        self.register_tool_path(tool, path)
                        break

                # Nope, neither system, nor config dir have it (or it is not the correct version)
                if not os_tool[AUTO_INSTALL_AVAILABLE_NAME] or found_path:
                    if found_path:
                        log(
                            f"{friendly_name} is installed on the system but doesn't meet the "
                            f"minimum version required of {min_version_required}."
                        )
                        log("Please upgrade it using the following command:")
                        log(cast(str, os_tool[UPGRADE_ACTION_NAME]))
                    else:
                        log(f"{friendly_name} cannot be found.")
                        log("Please install it using the following command:")
                        log(cast(str, os_tool[INSTALL_ACTION_NAME]))

                    if i < MAX_ATTEMPTS - 1:
                        log("Waiting until you install/upgrade via another terminal...")
                        confirmation = verify_to_proceed("Should I try again now?")
                        if confirmation:
                            confirmation = verify_to_proceed(
                                "Would you like to give me the path of the exe?"
                            )
                            if confirmation:
                                path_suggested = input(
                                    "Please paste the full path to the exe here: "
                                )
                            continue

                    raise ValueError(f"{friendly_name} could not be found")

                # We can attempt installing ourselves
                confirmation = verify_to_proceed(
                    f"{friendly_name} is not installed anywhere. Should I try to install it?"
                )
                if not confirmation:
                    raise ValueError(
                        f"{friendly_name} could not be found. "
                        "Please install it yourself and run again."
                    )

                self.install_dependency(tool)

    @abstractmethod
    def get_config_directory(self) -> str:
        pass

    @abstractmethod
    def get_aks_directory(self) -> str:
        pass

    @abstractmethod
    def get_os_tool_definition(self) -> Dict[str, Dict[str, Union[bool, str, List[str]]]]:
        pass

    @abstractmethod
    def install_dependency(self, tool: str):
        pass

    @abstractmethod
    def get_version(self, tool: str, path: str, version_cmd: str) -> str:
        pass
