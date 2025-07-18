"""
Logging configuration for Audio2DTX application.
"""

import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", format_string: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        stream=sys.stdout
    )
    
    # Create application logger
    logger = logging.getLogger('audio2dtx')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Suppress verbose third-party logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    logging.getLogger('librosa').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('spleeter').setLevel(logging.WARNING)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f'audio2dtx.{name}')


class ProgressLogger:
    """Helper class for logging processing progress."""
    
    def __init__(self, logger: logging.Logger, total_steps: int, description: str = "Processing"):
        self.logger = logger
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        
    def step(self, message: str = ""):
        """Log progress for current step."""
        self.current_step += 1
        progress = (self.current_step / self.total_steps) * 100
        log_message = f"{self.description}: {progress:.1f}% ({self.current_step}/{self.total_steps})"
        if message:
            log_message += f" - {message}"
        self.logger.info(log_message)
        
    def complete(self, message: str = "Completed"):
        """Log completion."""
        self.logger.info(f"{self.description}: {message}")