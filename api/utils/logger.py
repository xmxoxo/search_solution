import sys
from pathlib import Path
from loguru import logger
from config.app_config import LOG_DIR, LOG_LEVEL, LOG_RETENTION_DAYS

def setup_logger():
    logger.remove()
    
    format_str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    logger.add(
        sys.stdout,
        format=format_str,
        level=LOG_LEVEL,
        colorize=True
    )
    
    log_file = LOG_DIR / "ime_{time:YYYY-MM-DD}.log"
    logger.add(
        str(log_file),
        format=format_str,
        level=LOG_LEVEL,
        rotation="00:00",
        retention=f"{LOG_RETENTION_DAYS} days",
        compression="zip",
        encoding="utf-8"
    )
    
    return logger

g_logger = setup_logger()

__all__ = ["g_logger", "logger"]
