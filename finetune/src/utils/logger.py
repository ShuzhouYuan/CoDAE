import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logger(name: str, log_dir: str) -> logging.Logger:
    """
    Set up a logger with file and console handlers.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if not logger.handlers:
        # File handler with rotation
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, f"{name}.log"), maxBytes=10**6, backupCount=5
        )
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(console_handler)

    return logger