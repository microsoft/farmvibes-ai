import json
import logging
import logging.handlers
import os
from logging import Filter, LogRecord, getLogger
from platform import node
from typing import Dict, List, Optional

LOG_FORMAT = "[%(asctime)s] [%(hostname)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s"
"""The default log format."""

JSON_FORMAT = (
    '{"app_id": "%(app)s", "instance": "%(hostname)s", "level": "%(levelname)s", '
    '"msg": %(json_message)s, "scope": "%(name)s", "time": "%(asctime)s", "type": "log", '
    '"ver": "dev"}'
)
"""JSON log format."""

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
    "aiohttp_retry": "INFO",
}
"""The default log levels for the different loggers."""


class HostnameFilter(Filter):
    """Filter class to add hostname field to the log record."""

    hostname = node()

    def filter(self, record: LogRecord):
        """Adds a hostname field to the log record with the value of
        the node() function from the platform module.

        :param record: The log record to be filtered.

        :return: True
        """
        record.hostname = self.hostname
        return True


class AppFilter(Filter):
    """Filter class to add app field to the log record.

    :param app: The name of the application.
    """

    def __init__(self, app: str):
        super().__init__()
        self.app = app

    def filter(self, record: LogRecord):
        """Adds an app field to the log record with the value of the app attribute.

        :param record: The log record to be filtered.

        :return: True
        """
        record.app = self.app
        return True


class JsonMessageFilter(Filter):
    """Log filter to convert messages to JSON."""

    def filter(self, record: LogRecord):
        """Converts the message of the log record to JSON.

        :param record: The log record to be filtered.

        :return: True
        """
        if record.exc_info:
            # Convert message to the message + traceback as json
            record.msg = record.msg + "\n" + logging.Formatter().formatException(record.exc_info)
            record.exc_info = None
            record.exc_text = None

        record.json_message = json.dumps(record.getMessage())
        return True


def change_logger_level(loggername: str, level: str):
    """Sets the default log level for a logger.

    :param loggername: The name of the logger for which to set the log level.

    :param level: The desired log level (e.g. INFO, DEBUG, WARNING).

    """

    logger = getLogger(loggername)
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def configure_logging(
    default_level: Optional[str] = None,
    logdir: Optional[str] = None,
    logfile: str = f"{node()}.log",
    json: bool = True,
    appname: str = "",
):
    """Configures logging for the calling process.

    This method will create a logger and set its level to the given default_level argument.
    It will also create a StreamHandler and FileHandler if the logdir argument is provided,
    with the respective logfile name. It will add filters for the application name, hostname
    and json message, and set the formatter to JSON_FORMAT if json is True,
    or LOG_FORMAT otherwise.

    :param default_level: Default log level (defaults to 'DEBUG').

    :param logdir: Path to the directory where the log file will be stored.
        If not provided, no FileHandler will be added.

    :param logfile: Name of the log file (defaults to '{node()}.log').

    :param json: Flag to enable or disable JSON format (defaults to True).

    :param appname: Application name to be filtered (defaults to "").

    """

    handlers: List[logging.Handler] = [logging.StreamHandler()]
    default_level = "INFO" if default_level is None else default_level

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
