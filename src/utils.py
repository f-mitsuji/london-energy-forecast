import logging
from datetime import datetime
from logging import FileHandler, Formatter, StreamHandler
from pathlib import Path


def setup_logger(name: str, log_dir: Path | str = "logs") -> logging.Logger:
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = FileHandler(log_dir / f"{name}_{timestamp}.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    console_handler = StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
