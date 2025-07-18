"""
Custom exceptions for Audio2DTX application.
"""


class Audio2DTXError(Exception):
    """Base exception for all Audio2DTX errors."""
    pass


class ProcessingError(Audio2DTXError):
    """Error during audio processing or chart generation."""
    pass


class ValidationError(Audio2DTXError):
    """Error during input validation."""
    pass


class ConfigurationError(Audio2DTXError):
    """Error in configuration or settings."""
    pass


class ServiceError(Audio2DTXError):
    """Error communicating with external services."""
    pass


class ClassificationError(ProcessingError):
    """Error during drum classification."""
    pass


class OnsetDetectionError(ProcessingError):
    """Error during onset detection."""
    pass


class AudioLoadError(ProcessingError):
    """Error loading or processing audio files."""
    pass


class DTXGenerationError(ProcessingError):
    """Error generating DTX files."""
    pass