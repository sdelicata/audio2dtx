"""
Audio2DTX - Audio to DTXMania Chart Converter

A Python-based tool that converts audio files into DTXMania drumming game chart files
using machine learning, audio processing, and signal analysis.
"""

__version__ = "2.0.0"
__author__ = "Audio2DTX Team"

from .core.audio_processor import AudioProcessor
from .config.settings import Settings

__all__ = [
    "AudioProcessor",
    "Settings",
]