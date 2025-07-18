"""
External services integration (Magenta, ML models, etc.).
"""

from .magenta_client import MagentaClient
from .ml_service import MLService

__all__ = [
    "MagentaClient",
    "MLService",
]