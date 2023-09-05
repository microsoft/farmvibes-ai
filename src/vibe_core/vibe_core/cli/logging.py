import logging
import logging.handlers
import pathlib
import sys

FORMAT = "%(asctime)s - %(levelname)-7s - %(message)s"
LOGGER = logging.getLogger("farmvibes-ai")
LOG_PATH = pathlib.Path.home() / ".cache" / "farmvibes-ai"


# Custom logging formatter with colors:
class ColorFormatter(logging.Formatter):
    blue = "\x1b[34;1m"
    green = "\x1b[32;1m"
    grey = "\x1b[38;1m"
    yellow = "\x1b[33;1m"
    red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: blue + FORMAT + reset,
        logging.INFO: green + FORMAT + reset,
        logging.WARNING: yellow + FORMAT + reset,
        logging.ERROR: red + FORMAT + reset,
        logging.CRITICAL: red + FORMAT + reset,
    }

    def format(self, record: logging.LogRecord):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logging(name: str):
    LOGGER.setLevel(logging.DEBUG)

    # console handler:
    console_handler = logging.StreamHandler(sys.stderr)
    if sys.stderr.isatty():
        console_handler.setFormatter(ColorFormatter())
    else:
        console_handler.setFormatter(logging.Formatter(FORMAT))

    console_handler.setLevel(logging.INFO)
    LOGGER.addHandler(console_handler)

    logfile = LOG_PATH / f"farmvibes-ai-{name}.log"
    if not logfile.parent.exists():
        logfile.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        logfile, maxBytes=1024 * 1024 * 10, backupCount=5
    )
    file_handler.setFormatter(logging.Formatter(FORMAT))
    file_handler.setLevel(logging.DEBUG)
    LOGGER.addHandler(file_handler)
    return logfile


def set_log_level(log_level: str):
    LOGGER.handlers[0].setLevel(logging._nameToLevel[log_level.upper()])


def log(to_print: str, level: str = "info"):
    level = level.lower()
    if level == "info":
        LOGGER.info(f"\x1b[32;1m{to_print}\x1b[0m")
    elif level == "warning":
        LOGGER.warning(f"\x1b[33;1m{to_print}\x1b[0m")
    elif level == "error":
        LOGGER.error(f"\x1b[31;1m{to_print}\x1b[0m")
    elif level == "debug":
        LOGGER.debug(f"\x1b[34;1m{to_print}\x1b[0m")
    else:
        raise ValueError("Unknown log level")


def log_subprocess(binary: str, to_print: str, level: str = "debug"):
    logger = LOGGER.getChild(f"subprocess.{binary}")
    logger.setLevel(logging.DEBUG)
    for handler in logger.handlers:
        handler.formatter = logging.Formatter(FORMAT)
    logger.log(logging._nameToLevel[level.upper()], f"{binary}: {to_print}")
