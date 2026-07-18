# logging_config.py
import logging
import os
import sys

from config import (
    LOG_FILE,
    LOG_LEVEL,
    CONSOLE_LOG_LEVEL,
    FRAME_LOG_FILE,
    FRAME_LOG_LEVEL,
    SPEECH_LOG_FILE,
    SPEECH_LOG_LEVEL,
)

_QUIET_THIRD_PARTY_LOGGERS = (
    "google",
    "google.generativeai",
    "google_genai",
    "google_genai.models",
    "httpx",
    "httpcore",
    "urllib3",
    "requests",
    "grpc",
    "proto",
    "h11",
)


def _resolve_log_level(level: object) -> int:
    if isinstance(level, int):
        return int(level)
    if isinstance(level, str):
        return int(getattr(logging, level.strip().upper(), logging.INFO))
    return int(logging.INFO)


def _has_file_handler(logger: logging.Logger, filename: str) -> bool:
    filename = os.path.abspath(filename)
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            if os.path.abspath(getattr(handler, "baseFilename", "")) == filename:
                return True
    return False


def setup_logging():
    """
    Configures the logging for the application.
    - File handler: full app logs
    - Console handler: warnings/errors only
    - Frame handler: per-frame human-readable reports
    - Speech handler: narration/tts timing lines
    """
    if getattr(setup_logging, "_configured", False):
        return
    setup_logging._configured = True

    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)

    # Clear any existing handlers to avoid duplicated output
    for handler in list(root.handlers):
        root.removeHandler(handler)

    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(CONSOLE_LOG_LEVEL)
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )

    root.addHandler(file_handler)
    root.addHandler(console_handler)

    for logger_name in _QUIET_THIRD_PARTY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    frame_logger = logging.getLogger("statescribe.frame")
    frame_level = _resolve_log_level(FRAME_LOG_LEVEL)
    frame_logger.setLevel(frame_level)
    frame_logger.propagate = False

    if frame_level < logging.WARNING and not _has_file_handler(frame_logger, FRAME_LOG_FILE):
        frame_handler = logging.FileHandler(FRAME_LOG_FILE)
        frame_handler.setLevel(frame_level)
        frame_handler.setFormatter(logging.Formatter("%(message)s"))
        frame_logger.addHandler(frame_handler)

    speech_logger = logging.getLogger("statescribe.speech")
    speech_logger.setLevel(SPEECH_LOG_LEVEL)
    speech_logger.propagate = False

    if not _has_file_handler(speech_logger, SPEECH_LOG_FILE):
        speech_handler = logging.FileHandler(SPEECH_LOG_FILE)
        speech_handler.setLevel(SPEECH_LOG_LEVEL)
        speech_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        speech_logger.addHandler(speech_handler)
