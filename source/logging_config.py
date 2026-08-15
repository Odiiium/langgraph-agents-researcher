import logging
from pathlib import Path
from datetime import datetime

from .config import LOG_DIR

_LOGGER_NAME = "pipeline"
_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"


def _set_file(path: Path) -> logging.Logger:
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(handler)
    return logger


def configure_query_logging() -> logging.Logger:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return _set_file(Path(LOG_DIR) / f"{stamp}.log")


def configure_scenario_logging(name: str) -> logging.Logger:
    return _set_file(Path(LOG_DIR) / "_scenarios" / f"{name}.log")


def get_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        _set_file(Path(LOG_DIR) / "pipeline.log")
    return logger
