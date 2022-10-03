from logging import basicConfig, getLogger
from typing import Dict

LOG_FORMAT = "[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)s] %(message)s"

DEFAULT_LOGGER_LEVELS: Dict[str, str] = {
    "gdal": "INFO",
    "rasterio": "INFO",
    "urllib3": "INFO",
    "urllib3.connectionpool": "DEBUG",
    "fiona": "INFO",
    "werkzeug": "INFO",
    "azure": "WARNING",
    "matplotlib": "INFO",
}


def change_logger_level(loggername: str, level: str):
    logger = getLogger(loggername)
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)


def configure_logging(default_level: str = "DEBUG"):
    basicConfig(level=default_level, format=LOG_FORMAT)
    for logger, level in DEFAULT_LOGGER_LEVELS.items():
        change_logger_level(logger, level)
