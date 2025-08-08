import sys

from loguru import logger


LOG_FORMAT = '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <4}</level> - <level>{message}</level>'
VERBOSE_LOG_FORMAT = '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level> | <level>{extra}</level>'


def setup_logger(level: str = 'INFO', verbose: bool = False):
    logger.remove()
    logger.add(
        sink=sys.stderr,
        format=VERBOSE_LOG_FORMAT if verbose else LOG_FORMAT,
        level=level,
    )
