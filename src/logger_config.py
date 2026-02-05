"""
Конфигурация логгера для проекта бенчмарка LLM
"""

import logging
import sys
from pathlib import Path


def setup_logger(name: str = None, level: int = logging.INFO, log_file: Path = None) -> logging.Logger:
    """
    Настраивает и возвращает логгер

    Args:
        name: имя логгера (None для корневого)
        level: уровень логирования
        log_file: путь к файлу для записи логов (опционально)

    Returns:
        настроенный логгер
    """
    logger = logging.getLogger(name)

    # Не добавляем обработчики повторно
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Формат сообщений
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Обработчик для вывода в консоль
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Обработчик для записи в файл (если указан)
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Возвращает логгер по имени

    Args:
        name: имя логгера

    Returns:
        логгер
    """
    return logging.getLogger(name)


def configure_root_logger(level: int = logging.INFO, log_file: Path = None):
    """
    Настраивает корневой логгер для всего приложения

    Args:
        level: уровень логирования
        log_file: путь к файлу для записи логов (опционально)
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            *([] if not log_file else [logging.FileHandler(log_file, encoding='utf-8')])
        ]
    )
