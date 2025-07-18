"""
Validation utilities for Audio2DTX application.
"""

import os
from pathlib import Path
from typing import Dict, Any, List

from .exceptions import ValidationError


def validate_audio_file(file_path: str) -> str:
    """
    Validate that an audio file exists and has a supported format.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Absolute path to the validated file
        
    Raises:
        ValidationError: If file doesn't exist or has unsupported format
    """
    if not os.path.exists(file_path):
        raise ValidationError(f"Audio file not found: {file_path}")
    
    valid_formats = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
    file_ext = Path(file_path).suffix.lower()
    
    if file_ext not in valid_formats:
        raise ValidationError(
            f"Unsupported audio format: {file_ext}. "
            f"Supported formats: {', '.join(valid_formats)}"
        )
    
    return os.path.abspath(file_path)


def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate metadata dictionary.
    
    Args:
        metadata: Metadata dictionary to validate
        
    Returns:
        Validated and normalized metadata
        
    Raises:
        ValidationError: If metadata is invalid
    """
    required_fields = ['title', 'artist', 'author']
    for field in required_fields:
        if field not in metadata or not metadata[field]:
            raise ValidationError(f"Missing required metadata field: {field}")
    
    # Validate difficulty
    if 'difficulty' in metadata:
        try:
            difficulty = int(metadata['difficulty'])
            if not 1 <= difficulty <= 100:
                raise ValueError()
            metadata['difficulty'] = difficulty
        except (ValueError, TypeError):
            raise ValidationError("Difficulty must be an integer between 1 and 100")
    
    # Validate time signature
    if 'time_signature' in metadata:
        valid_signatures = ['4/4', '3/4', '6/8', '2/4', '5/4']
        if metadata['time_signature'] not in valid_signatures:
            raise ValidationError(
                f"Invalid time signature: {metadata['time_signature']}. "
                f"Valid options: {', '.join(valid_signatures)}"
            )
    
    # Validate boolean fields
    bool_fields = ['use_original_bgm']
    for field in bool_fields:
        if field in metadata and not isinstance(metadata[field], bool):
            raise ValidationError(f"Field '{field}' must be a boolean value")
    
    return metadata


def validate_track_selection(track_flags: Dict[str, bool]) -> str:
    """
    Validate track selection flags and return the selected track.
    
    Args:
        track_flags: Dictionary of track flags
        
    Returns:
        Selected track name or 'default' if none selected
        
    Raises:
        ValidationError: If multiple tracks are selected
    """
    selected_tracks = [track for track, enabled in track_flags.items() if enabled]
    
    if len(selected_tracks) > 1:
        raise ValidationError(
            f"Multiple tracks selected: {', '.join(selected_tracks)}. "
            "Please select only one track."
        )
    
    return selected_tracks[0] if selected_tracks else 'default'


def validate_output_directory(output_path: str) -> str:
    """
    Validate and create output directory if needed.
    
    Args:
        output_path: Path to output directory
        
    Returns:
        Absolute path to the validated directory
        
    Raises:
        ValidationError: If directory cannot be created or accessed
    """
    abs_path = os.path.abspath(output_path)
    
    try:
        os.makedirs(abs_path, exist_ok=True)
    except OSError as e:
        raise ValidationError(f"Cannot create output directory {abs_path}: {e}")
    
    if not os.access(abs_path, os.W_OK):
        raise ValidationError(f"Output directory is not writable: {abs_path}")
    
    return abs_path


def validate_processing_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate processing parameters.
    
    Args:
        params: Processing parameters to validate
        
    Returns:
        Validated parameters
        
    Raises:
        ValidationError: If parameters are invalid
    """
    validated = params.copy()
    
    # Validate sample rate
    if 'sample_rate' in params:
        if not isinstance(params['sample_rate'], int) or params['sample_rate'] <= 0:
            raise ValidationError("Sample rate must be a positive integer")
    
    # Validate hop length
    if 'hop_length' in params:
        if not isinstance(params['hop_length'], int) or params['hop_length'] <= 0:
            raise ValidationError("Hop length must be a positive integer")
    
    # Validate confidence threshold
    if 'confidence_threshold' in params:
        threshold = params['confidence_threshold']
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValidationError("Confidence threshold must be a number between 0 and 1")
    
    return validated