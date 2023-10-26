import os
import socket
import subprocess
from platform import uname
from typing import Dict, List, Optional

from .logging import log, log_subprocess

AUTO_CONFIRMATION = False


def execute_cmd(
    cmd: List[str],
    check_return_code: bool = True,
    check_empty_result: bool = True,
    error_string: str = "Unable to execute command",
    capture_output: bool = True,
    censor_command: bool = False,
    censor_output: bool = False,
    subprocess_log_level: str = "info",
    env_vars: Dict[str, str] = {},
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
        for line in iter(process.stdout.readline, b""):  # type: ignore
            if line:
                decoded = line.decode("utf-8").rstrip()
                stdout_capture.append(decoded)
                if not censor_output:
                    log_subprocess(binary, decoded, subprocess_log_level)
    retcode = process.wait()
    if retcode:
        log(
            f"Unable to run command {command_in_logs}.\n",
            level="error",
        )
        raise ValueError(error_string)

    if check_return_code and retcode != 0:
        raise ValueError(f"{error_string} (return code: {retcode})")

    if capture_output:
        if check_empty_result and not stdout_capture:
            raise ValueError(error_string)

    return "\n".join(stdout_capture) if capture_output else ""  # type: ignore


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
