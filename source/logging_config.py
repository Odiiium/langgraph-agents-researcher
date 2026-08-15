import logging
from pathlib import Path

from .config import LOG_DIR

_LOGGER_NAME = "guardrails"


def get_logger(name: str = _LOGGER_NAME) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(Path(LOG_DIR) / "guardrails.log", encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        )
        logger.addHandler(handler)
        logger.propagate = False
    return logger
