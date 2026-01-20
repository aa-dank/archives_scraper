# logging.py

"""
Centralized logging configuration for the extraction worker.

This module provides a single, idempotent logging setup shared across
worker, CLI, and extractors. Supports console, file (rotating), and
JSON logging modes.
"""

import logging
import logging.handlers
import json
from typing import Optional


class JSONFormatter(logging.Formatter):
    """
    Formats log records as JSON for structured logging.
    
    Each log record is serialized as a JSON object with timestamp,
    level, logger name, message, and any extra fields.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': self.formatTime(record, self.datefmt),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        # Add extra fields if present
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'created', 'filename', 'funcName',
                          'levelname', 'levelno', 'lineno', 'module', 'msecs',
                          'message', 'pathname', 'process', 'processName',
                          'relativeCreated', 'thread', 'threadName', 'exc_info',
                          'exc_text', 'stack_info'):
                log_data[key] = value
        
        if record.exc_info:
            log_data['exc_info'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


def configure_logging(
    *,
    level: str | int = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
    json_format: bool = False,
    max_bytes: int = 5_000_000,
    backups: int = 3,
    fmt: Optional[str] = None,
) -> logging.Logger:
    """
    Configure centralized logging for the application.
    
    This function is idempotent and safe to call multiple times. It removes
    any existing handlers to prevent duplication before configuring new ones.
    
    Parameters
    ----------
    level : str | int, default="INFO"
        Logging level (e.g., "DEBUG", "INFO", "WARNING", "ERROR").
    log_file : str | None, default=None
        Path to log file. If None, file logging is disabled.
    console : bool, default=True
        Whether to enable console (stdout) logging.
    json_format : bool, default=False
        Whether to use JSON formatting for logs.
    max_bytes : int, default=5_000_000
        Maximum size of log file before rotation (in bytes).
    backups : int, default=3
        Number of backup log files to keep.
    fmt : str | None, default=None
        Custom log format string. If None, uses default format.
    
    Returns
    -------
    logging.Logger
        The configured root logger.
    """
    # Get root logger
    logger = logging.getLogger()
    
    # Convert level string to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    logger.setLevel(level)
    
    # Remove existing handlers to prevent duplication
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    
    # Determine formatters
    if json_format:
        formatter = JSONFormatter()
    else:
        if fmt is None:
            if console:
                # Concise format for console
                fmt = "%(levelname)s: %(message)s"
            else:
                # Detailed format for file
                fmt = "%(asctime)s %(name)s %(levelname)s %(message)s"
        formatter = logging.Formatter(fmt)
    
    # Add console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        if json_format:
            console_handler.setFormatter(JSONFormatter())
        else:
            # Always use concise format for console
            console_formatter = logging.Formatter("%(levelname)s: %(message)s")
            console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Add file handler with rotation
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backups,
        )
        file_handler.setLevel(level)
        # File handler always uses detailed format
        if json_format:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_formatter = logging.Formatter(
                "%(asctime)s %(name)s %(levelname)s %(message)s"
            )
            file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for the given name.
    
    This is a thin wrapper around logging.getLogger() for consistency
    with the module's API.
    
    Parameters
    ----------
    name : str
        Name of the logger, typically __name__ from the calling module.
    
    Returns
    -------
    logging.Logger
        Logger instance for the given name.
    """
    return logging.getLogger(name)
