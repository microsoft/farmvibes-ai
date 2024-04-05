import locale
import os
import socket
import subprocess
from functools import lru_cache
from platform import uname
from typing import Dict, List, Optional

from .logging import log, log_subprocess

AUTO_CONFIRMATION = False
DEFAULT_ERROR_STRING = "Unable to execute command"
WARNING_STRINGS = ("[warning]", "[Warning]", "[WARNING]", "WARNING:", "Warning:", "warning:")


@lru_cache
def get_subprocess_encoding():
    return locale.getpreferredencoding()


def execute_cmd(
    cmd: List[str],
    check_return_code: bool = True,
    check_empty_result: bool = True,
    error_string: str = DEFAULT_ERROR_STRING,
    capture_output: bool = True,
    censor_command: bool = False,
    censor_output: bool = False,
    subprocess_log_level: str = "info",
    env_vars: Dict[str, str] = {},
    log_error: bool = False,
) -> str:
    command_in_logs = (cmd[:3] + ["******"]) if censor_command else cmd
    log(f"Executing command: {' '.join(command_in_logs)}", "debug")

    process = subprocess.Popen(
        cmd,
        shell=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, **env_vars},
    )
    stdout_capture: List[str] = []
    with process.stdout:  # type: ignore
        binary = os.path.basename(cmd[0])
        is_running_az = binary.split(".")[0].lower() == "az"
        for line in iter(process.stdout.readline, b""):  # type: ignore
            if line:
                decoded = line.decode(get_subprocess_encoding()).rstrip()
                if not is_running_az or (is_running_az and not decoded.startswith(WARNING_STRINGS)):
                    stdout_capture.append(decoded)
                if not censor_output:
                    log_subprocess(binary, decoded, subprocess_log_level)
    retcode = process.wait()
    if retcode:
        log_message = f"Unable to run command {command_in_logs}.\n"
        log_message = (
            f"{error_string}. " + log_message
            if error_string != DEFAULT_ERROR_STRING
            else log_message
        )
        if log_error:
            log(log_message, level="error")
        raise ValueError(error_string)

    if check_return_code and retcode != 0:
        raise ValueError(f"{error_string}. (Return code: {retcode})")

    if capture_output:
        if check_empty_result and not stdout_capture:
            raise ValueError(error_string)

    return "\n".join(stdout_capture) if capture_output else ""  # type: ignore


def verify_to_proceed(message: str) -> bool:
    if AUTO_CONFIRMATION:
        return True

    answered = False
    confirmation = False
    while not answered:
        confirmation = input(f"{message} (y/n): ").lower()
        if confirmation not in ["y", "n", "yes", "no"]:
            print("Invalid input. Please enter 'y' or 'n'")
            continue
        answered = True
        confirmation = confirmation[0]
    if confirmation == "y":
        return True
    return False


def set_auto_confirm():
    global AUTO_CONFIRMATION
    AUTO_CONFIRMATION = True


def in_wsl() -> bool:
    return "microsoft-standard" in uname().release


def is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) != 0


def log_should_be_logged_in(error: Optional[Exception] = None):
    if error:
        log(f"Error: {error}", level="error")
    log(
        "Ensure you are logged into Azure via `az login "
        "--scope https://graph.microsoft.com/.default`"
    )
    log("And set a default subscription via `az account set -s <subscription guid>`")
