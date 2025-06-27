import json
import logging
import logging.config
import os
import sys
import time
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Optional, ParamSpec, TypeVar

if sys.version_info >= (3, 10):
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

# Used for callable types
T = TypeVar("T")
P = ParamSpec("P")


def setup_logging(
    *,
    logger_name: str = "opera_tropo",
    debug: bool = False,
    filename: Optional[str] = None,
):
    """Set up logging configuration for the specified logger.

    Parameters
    ----------
    logger_name : str, optional
        The name of the logger to configure. Default is "opera_tropo".
    debug : bool, optional
        Whether to set the logger and handlers to DEBUG level.
        If True, logging will include debug messages. Default is False.
    filename : Optional[str], optional
        The file path where logs should be written.
        If provided, logs will be saved to this file.

    The function reads logging configuration from 'log-config.json'
    located in the same directory as this script.
    It updates the configuration based on the provided parameters
    and applies it.

    """
    config_file = Path(__file__).parent / "log-config.json"

    with open(config_file) as f_in:
        config = json.load(f_in)

    if logger_name not in config["loggers"]:
        config["loggers"][logger_name] = {"level": "INFO", "handlers": ["stderr"]}

    if debug:
        config["loggers"][logger_name]["level"] = "DEBUG"
        config["handlers"]["stderr"]["level"] = "DEBUG"
        config["handlers"]["file"]["level"] = "DEBUG"

    if filename:
        if "file" not in config["loggers"][logger_name]["handlers"]:
            config["loggers"][logger_name]["handlers"].append("file")
        config["handlers"]["file"]["filename"] = os.fspath(filename)
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

    if "filename" not in config["handlers"]["file"]:
        config["handlers"].pop("file", None)

    logging.config.dictConfig(config)


def log_runtime(f: Callable[P, T]) -> Callable[P, T]:
    # f: Callable[P, T]) -> Callable[P, T]:
    """Decorate a function to time how long it takes to run.

    Usage
    -----
    @log_runtime
    def test_func():
        return 2 + 4
    """
    logger = logging.getLogger(__name__)

    @wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs):
        t1 = time.time()

        result = f(*args, **kwargs)

        t2 = time.time()
        elapsed_seconds = t2 - t1
        elapsed_minutes = elapsed_seconds / 60.0

        time_string = (
            f"Total elapsed time for {f.__module__}.{f.__name__}: "
            f"{elapsed_minutes:.2f} minutes ({elapsed_seconds:.2f} seconds)"
        )

        logger.debug(time_string)

        return result

    return wrapper


def remove_raider_logs():
    """Remove RAiDER's internal file handlers writing."""
    for logger_name in logging.root.manager.loggerDict:
        if "RAiDER" in logger_name:
            logger = logging.getLogger(logger_name)
            handlers_to_remove = []

            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    filename = getattr(handler, "baseFilename", "")
                    if filename.endswith("debug.log") or filename.endswith("error.log"):
                        handlers_to_remove.append(handler)

            for handler in handlers_to_remove:
                logger.removeHandler(handler)
                try:
                    handler.close()
                    Path(handler.baseFilename).unlink(missing_ok=True)
                except Exception:
                    pass  # Optionally log the error
