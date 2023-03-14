import json
import logging
import logging.handlers
import os
from logging import Filter, LogRecord, getLogger
from platform import node
from typing import Dict, List, Optional

LOG_FORMAT = "[%(asctime)s] [%(hostname)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s"
JSON_FORMAT = (
    '{"app_id": "%(app)s", "instance": "%(hostname)s", "level": "%(levelname)s", '
    '"msg": %(json_message)s, "scope": "%(name)s", "time": "%(asctime)s", "type": "log", '
    '"ver": "dev"}'
)

DEFAULT_LOGGER_LEVELS: Dict[str, str] = {
    "gdal": "INFO",
    "rasterio": "INFO",
    "urllib3": "INFO",
    "urllib3.connectionpool": "DEBUG",
    "fiona": "INFO",
    "werkzeug": "INFO",
    "azure": "WARNING",
    "matplotlib": "INFO",
    "uvicorn": "WARNING",
}


class HostnameFilter(Filter):
    "Log filter that adds support for host names."

    hostname = node()

    def filter(self, record: LogRecord):
        record.hostname = self.hostname
        return True


class AppFilter(Filter):
    "Adds an app field to log record."

    def __init__(self, app: str):
        super().__init__()
        self.app = app

    def filter(self, record: LogRecord):
        record.app = self.app
        return True


class JsonMessageFilter(Filter):
    "Log filter to convert messages to JSON."

    def filter(self, record: LogRecord):
        record.json_message = json.dumps(record.getMessage())
        return True


def change_logger_level(loggername: str, level: str):
    "Sets the default log level for a logger."

    logger = getLogger(loggername)
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def configure_logging(
    default_level: str = "DEBUG",
    logdir: Optional[str] = None,
    logfile: str = f"{node()}.log",
    json: bool = True,
    appname: str = "",
):
    "Configures logging for the calling process"

    handlers: List[logging.Handler] = [logging.StreamHandler()]

    if logdir:
        os.makedirs(logdir, exist_ok=True)
        logfile = os.path.join(logdir, logfile)
        handlers.append(logging.FileHandler(logfile))

    logger = logging.getLogger()
    for handler in handlers:
        handler.addFilter(AppFilter(appname))
        handler.addFilter(HostnameFilter())
        handler.addFilter(JsonMessageFilter())
        handler.setFormatter(logging.Formatter(JSON_FORMAT if json else LOG_FORMAT))
        logger.addHandler(handler)

    logger.setLevel(default_level)
    for logger_name, level in DEFAULT_LOGGER_LEVELS.items():
        change_logger_level(logger_name, level)
