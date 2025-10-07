from __future__ import annotations
import logging
import logging.handlers
import socket
import sys
import traceback
from typing import Optional, Callable

from flask import request
from taranis_base_bot.config import get_settings


class TaranisBotLogger:
    def __init__(self, module: str, debug: bool, colored: bool,
                 syslog_address: Optional[tuple[str, int]]):
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        if colored:
            stream_handler.setFormatter(TaranisLogFormatter(module))
        else:
            stream_handler.setFormatter(logging.Formatter(f"[{module}] [%(levelname)s] - %(message)s"))

        sys_log_handler = None
        if syslog_address:
            try:
                sys_log_handler = logging.handlers.SysLogHandler(
                    address=syslog_address, socktype=socket.SOCK_STREAM
                )
            except Exception:
                print("Unable to connect to syslog server!")

        self.logger = logging.getLogger(module)
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG if debug else logging.INFO)

        if sys_log_handler:
            self.logger.addHandler(sys_log_handler)
        self.logger.addHandler(stream_handler)

    def debug(self, message): self.logger.debug(message)
    def exception(self, message=None):
        if message: self.logger.debug(message)
        self.logger.debug(traceback.format_exc())
    def info(self, message): self.logger.info(message)
    def critical(self, message): self.logger.critical(message)
    def warning(self, message): self.logger.warning(message)
    def error(self, message): self.logger.error(message)


class TaranisLogFormatter(logging.Formatter):
    def __init__(self, module):
        grey = "\x1b[38;20m"
        blue = "\x1b[1;36m"
        yellow = "\x1b[33;20m"
        red = "\x1b[31;20m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"
        self.format_string = f"[{module}] [%(levelname)s] - %(message)s"
        self.FORMATS = {
            logging.DEBUG: grey + self.format_string + reset,
            logging.INFO: blue + self.format_string + reset,
            logging.WARNING: yellow + self.format_string + reset,
            logging.ERROR: red + self.format_string + reset,
            logging.CRITICAL: bold_red + self.format_string + reset,
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class Logger(TaranisBotLogger):
    def resolve_ip_address(self):
        headers_list = request.headers.getlist("X-Forwarded-For")
        return headers_list[0] if headers_list else request.remote_addr

    def resolve_method(self):
        return request.method

    def resolve_resource(self):
        fp_len = len(request.full_path)
        return request.full_path[: fp_len - 1] if request.full_path and request.full_path.endswith("?") else request.full_path

    def resolve_data(self):
        if "application/json" not in request.headers.get("Content-Type", ""):
            return ""
        if not request.data:
            return ""
        return str(request.data)[:4096].replace("\\r", "").replace("\\n", "").replace(" ", "")[2:-1]


def configure_logger(settings_provider: Callable = get_settings, syslog_address: tuple[str, int] | None = None) -> None:
    """
    Build the global logger using the provided settings.
    """
    global _logger
    config = settings_provider()
    _logger = Logger(module=config.MODULE_ID, colored=config.COLORED_LOGS, debug=config.DEBUG, syslog_address=syslog_address)


def get_logger() -> Logger:
    """
    Return the configured logger. If not configured yet, build it from the base settings.
    """
    global _logger
    if _logger is None:
        configure_logger()
    assert _logger is not None, "Logger was not configured correctly"
    return _logger

_logger: Logger | None = None