from datetime import datetime
from .constants import LOGGING_LEVEL_INFO

CURRENT_LOGGING_LEVEL = LOGGING_LEVEL_INFO


def set_log_level(log_level: int):
    global CURRENT_LOGGING_LEVEL
    CURRENT_LOGGING_LEVEL = log_level


def log(to_print: str, log_level: int = LOGGING_LEVEL_INFO):
    if log_level <= CURRENT_LOGGING_LEVEL:
        now = datetime.now()
        time = now.strftime("[%H:%M:%S]")
        print(f"{time} {to_print}")
