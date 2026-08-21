"""
Logging utilities for MAL-ViT.
"""

import logging
from pathlib import Path


def get_logger(
    name: str = "MAL-ViT",
    log_file: str = "logs/training.log",
):
    """
    Create and return a configured logger.
    """

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)

    # File handler
    log_path = Path(log_file)
    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_handler = logging.FileHandler(
        log_path,
        encoding="utf-8",
    )

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":

    print("=" * 60)
    print("MAL-ViT Logger Test")
    print("=" * 60)

    logger = get_logger(
        name="MAL-ViT-Test",
        log_file="logs/test.log",
    )

    logger.info("Logger initialized successfully.")
    logger.info("MAL-ViT logging test passed.")

    print("\nLogger validation: PASSED")