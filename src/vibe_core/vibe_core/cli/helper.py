import json
import re
import string
import random
import subprocess
from platform import uname
from .constants import (
    HELM_RELEASE_PATH,
    HELM_VERSION_SEARCH_STRING,
    KUBECTL_BASE_PATH,
    LOGGING_LEVEL_VERBOSE,
    PREFIX_LENGTH,
    TERRAFORM_RELEASE_PATH,
)
from .logging import log

AUTO_CONFIRMATION = False


def execute_cmd(
    cmd: str, check_return_code: bool, check_empty_result: bool, error_string: str
) -> str:
    log(f"Executing command:\n{cmd}", LOGGING_LEVEL_VERBOSE)
    result = subprocess.run(cmd, shell=True, check=check_return_code, stdout=subprocess.PIPE)
    if check_return_code and result.returncode != 0:
        raise ValueError(error_string)

    result = str(result.stdout, "utf-8").strip()
    if check_empty_result and not result:
        raise ValueError(error_string)

    return result


def verify_to_proceed(message: str) -> bool:
    if AUTO_CONFIRMATION:
        return True

    confirmation = input(f"{message} (y/n): ")
    if confirmation and confirmation.lower() == "y":
        return True
    return False


def set_auto_confirm():
    global AUTO_CONFIRMATION
    AUTO_CONFIRMATION = True


def generate_random_string() -> str:
    # Define the possible characters to use in the string
    chars = string.ascii_lowercase + string.digits
    random_string = random.choice(string.ascii_lowercase) + "".join(
        random.choice(chars) for _ in range(PREFIX_LENGTH - 1)
    )
    return random_string


def in_wsl() -> bool:
    return "microsoft-standard" in uname().release


def is_file_in_mount(filename: str) -> bool:
    return "/mnt/" in filename


def get_latest_helm_version() -> str:
    cmd = f'curl -Ls "{HELM_RELEASE_PATH}"'
    helm_version_dump = execute_cmd(cmd, True, True, "Failed to get helm stable version")
    matched_version = re.search(HELM_VERSION_SEARCH_STRING + '3.[0-9]*.[0-9]*"', helm_version_dump)
    assert matched_version is not None, "Failed to get helm stable version"
    return matched_version.group(0).replace(HELM_VERSION_SEARCH_STRING, "").replace('"', "")


def get_latest_kubectl_version() -> str:
    cmd = f'curl -s "{KUBECTL_BASE_PATH}/stable.txt"'
    return execute_cmd(cmd, True, True, "Failed to get kubectl stable version")


def get_latest_terraform_version() -> str:
    cmd = f'curl -s "{TERRAFORM_RELEASE_PATH}"'
    version_dump = execute_cmd(cmd, True, True, "Failed to get terraform stable version")
    version_json = json.loads(version_dump)
    return version_json["tag_name"].replace("v", "")
