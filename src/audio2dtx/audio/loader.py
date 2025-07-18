"""
Audio file loading and validation.
"""

import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from ..config.settings import Settings
from ..config.constants import SUPPORTED_AUDIO_FORMATS, MAX_AUDIO_DURATION, MIN_AUDIO_DURATION
from ..utils.exceptions import AudioLoadError, ValidationError
from ..utils.validators import validate_audio_file
from ..utils.logging import get_logger

logger = get_logger(__name__)


class AudioLoader:
    """
    Audio file loader with validation and preprocessing.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize audio loader.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.target_sr = settings.audio.sample_rate
        
    def load_audio(self, file_path: str, validate: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load audio file with validation and preprocessing.
        
        Args:
            file_path: Path to audio file
            validate: Whether to validate the file before loading
            
        Returns:
            Tuple of (audio_data, metadata)
            
        Raises:
            AudioLoadError: If loading fails
            ValidationError: If validation fails
        """
        if validate:
            file_path = validate_audio_file(file_path)
        
        try:
            logger.info(f"Loading audio file: {file_path}")
            
            # Load audio with librosa
            audio, original_sr = librosa.load(file_path, sr=None, mono=False)
            
            # Get file info
            file_info = sf.info(file_path)
            
            # Create metadata
            metadata = {
                'file_path': file_path,
                'file_name': Path(file_path).name,
                'original_sample_rate': original_sr,
                'target_sample_rate': self.target_sr,
                'original_channels': file_info.channels,
                'original_duration': file_info.duration,
                'file_format': file_info.format,
                'subtype': file_info.subtype
            }
            
            # Convert to mono if needed
            if audio.ndim > 1:
                audio = librosa.to_mono(audio)
                logger.info("Converted stereo audio to mono")
            
            # Resample if needed
            if original_sr != self.target_sr:
                audio = librosa.resample(audio, orig_sr=original_sr, target_sr=self.target_sr)
                logger.info(f"Resampled from {original_sr}Hz to {self.target_sr}Hz")
            
            # Validate duration
            duration = len(audio) / self.target_sr
            metadata['final_duration'] = duration
            
            if duration < MIN_AUDIO_DURATION:
                raise ValidationError(
                    f"Audio too short: {duration:.1f}s (minimum: {MIN_AUDIO_DURATION}s)"
                )
            
            if duration > MAX_AUDIO_DURATION:
                logger.warning(
                    f"Audio is long: {duration:.1f}s (maximum recommended: {MAX_AUDIO_DURATION}s)"
                )
            
            # Normalize audio
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio))
                logger.info("Normalized audio to [-1, 1] range")
            
            metadata['final_length'] = len(audio)
            metadata['peak_amplitude'] = float(np.max(np.abs(audio)))
            metadata['rms_level'] = float(np.sqrt(np.mean(audio**2)))
            
            logger.info(f"Successfully loaded audio: {duration:.1f}s, {self.target_sr}Hz")
            
            return audio, metadata
            
        except (librosa.LibrosaError, sf.SoundFileError) as e:
            raise AudioLoadError(f"Failed to load audio file {file_path}: {e}")
        except Exception as e:
            raise AudioLoadError(f"Unexpected error loading {file_path}: {e}")
    
    def get_audio_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get audio file information without loading the full file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with audio file information
            
        Raises:
            AudioLoadError: If reading file info fails
        """
        try:
            file_path = validate_audio_file(file_path)
            file_info = sf.info(file_path)
            
            return {
                'file_path': file_path,
                'file_name': Path(file_path).name,
                'sample_rate': file_info.samplerate,
                'channels': file_info.channels,
                'duration': file_info.duration,
                'frames': file_info.frames,
                'format': file_info.format,
                'subtype': file_info.subtype,
                'file_size': os.path.getsize(file_path)
            }
            
        except Exception as e:
            raise AudioLoadError(f"Failed to read audio file info: {e}")
    
    def validate_audio_file(self, file_path: str) -> bool:
        """
        Validate audio file without loading it.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            True if valid, raises exception otherwise
            
        Raises:
            ValidationError: If file is invalid
        """
        try:
            # Check file exists and format
            validate_audio_file(file_path)
            
            # Check file can be read
            info = self.get_audio_info(file_path)
            
            # Check duration
            if info['duration'] < MIN_AUDIO_DURATION:
                raise ValidationError(
                    f"Audio too short: {info['duration']:.1f}s (minimum: {MIN_AUDIO_DURATION}s)"
                )
            
            # Check for corruption by trying to read a small sample
            try:
                librosa.load(file_path, sr=self.target_sr, duration=1.0)
            except Exception as e:
                raise ValidationError(f"Audio file appears corrupted: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Audio validation failed for {file_path}: {e}")
            raise
    
    def batch_validate(self, file_paths: list) -> Dict[str, bool]:
        """
        Validate multiple audio files.
        
        Args:
            file_paths: List of file paths to validate
            
        Returns:
            Dictionary mapping file paths to validation results
        """
        results = {}
        
        for file_path in file_paths:
            try:
                self.validate_audio_file(file_path)
                results[file_path] = True
            except Exception as e:
                logger.error(f"Validation failed for {file_path}: {e}")
                results[file_path] = False
        
        return results