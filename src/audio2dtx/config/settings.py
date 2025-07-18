"""
Configuration management for Audio2DTX application.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from .constants import *
from ..utils.exceptions import ConfigurationError


@dataclass
class AudioSettings:
    """Audio processing settings."""
    sample_rate: int = SAMPLE_RATE
    hop_length: int = HOP_LENGTH
    n_fft: int = N_FFT
    n_mels: int = N_MELS


@dataclass
class ClassificationSettings:
    """Classification settings."""
    confidence_threshold: float = DEFAULT_CONFIDENCE
    min_confidence: float = MIN_CONFIDENCE
    high_confidence: float = HIGH_CONFIDENCE
    n_mfcc: int = N_MFCC
    n_chroma: int = N_CHROMA
    n_contrast: int = N_CONTRAST


@dataclass
class DTXSettings:
    """DTX generation settings."""
    resolution: int = DTX_RESOLUTION
    bars_before_song: int = 2
    max_bars: int = 1000


@dataclass
class ServiceSettings:
    """External service settings."""
    magenta_url: str = DEFAULT_MAGENTA_URL
    magenta_timeout: float = 30.0
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class ProcessingSettings:
    """Processing pipeline settings."""
    max_audio_duration: int = MAX_AUDIO_DURATION
    min_audio_duration: int = MIN_AUDIO_DURATION
    max_onset_count: int = MAX_ONSET_COUNT
    temporal_scales: list = field(default_factory=lambda: TEMPORAL_SCALES)


@dataclass 
class Settings:
    """Main settings container."""
    audio: AudioSettings = field(default_factory=AudioSettings)
    classification: ClassificationSettings = field(default_factory=ClassificationSettings)
    dtx: DTXSettings = field(default_factory=DTXSettings)
    services: ServiceSettings = field(default_factory=ServiceSettings)
    processing: ProcessingSettings = field(default_factory=ProcessingSettings)
    
    @classmethod
    def load_from_file(cls, config_path: str) -> 'Settings':
        """
        Load settings from a YAML configuration file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Settings instance
            
        Raises:
            ConfigurationError: If file cannot be loaded or parsed
        """
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            return cls.from_dict(config_data or {})
            
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {e}")
    
    @classmethod
    def from_dict(cls, config_data: Dict[str, Any]) -> 'Settings':
        """
        Create settings from a dictionary.
        
        Args:
            config_data: Configuration data dictionary
            
        Returns:
            Settings instance
        """
        settings = cls()
        
        # Update audio settings
        if 'audio' in config_data:
            audio_data = config_data['audio']
            settings.audio = AudioSettings(**audio_data)
        
        # Update classification settings
        if 'classification' in config_data:
            classification_data = config_data['classification']
            settings.classification = ClassificationSettings(**classification_data)
        
        # Update DTX settings
        if 'dtx' in config_data:
            dtx_data = config_data['dtx']
            settings.dtx = DTXSettings(**dtx_data)
        
        # Update service settings
        if 'services' in config_data:
            service_data = config_data['services']
            settings.services = ServiceSettings(**service_data)
        
        # Update processing settings
        if 'processing' in config_data:
            processing_data = config_data['processing']
            settings.processing = ProcessingSettings(**processing_data)
        
        return settings
    
    @classmethod
    def from_environment(cls) -> 'Settings':
        """
        Create settings from environment variables.
        
        Returns:
            Settings instance with environment overrides
        """
        settings = cls()
        
        # Override from environment variables
        if 'MAGENTA_SERVICE_URL' in os.environ:
            settings.services.magenta_url = os.environ['MAGENTA_SERVICE_URL']
        
        if 'AUDIO2DTX_SAMPLE_RATE' in os.environ:
            try:
                settings.audio.sample_rate = int(os.environ['AUDIO2DTX_SAMPLE_RATE'])
            except ValueError:
                pass
        
        if 'AUDIO2DTX_CONFIDENCE_THRESHOLD' in os.environ:
            try:
                settings.classification.confidence_threshold = float(os.environ['AUDIO2DTX_CONFIDENCE_THRESHOLD'])
            except ValueError:
                pass
        
        return settings
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert settings to dictionary.
        
        Returns:
            Settings as dictionary
        """
        return {
            'audio': {
                'sample_rate': self.audio.sample_rate,
                'hop_length': self.audio.hop_length,
                'n_fft': self.audio.n_fft,
                'n_mels': self.audio.n_mels
            },
            'classification': {
                'confidence_threshold': self.classification.confidence_threshold,
                'min_confidence': self.classification.min_confidence,
                'high_confidence': self.classification.high_confidence,
                'n_mfcc': self.classification.n_mfcc,
                'n_chroma': self.classification.n_chroma,
                'n_contrast': self.classification.n_contrast
            },
            'dtx': {
                'resolution': self.dtx.resolution,
                'bars_before_song': self.dtx.bars_before_song,
                'max_bars': self.dtx.max_bars
            },
            'services': {
                'magenta_url': self.services.magenta_url,
                'magenta_timeout': self.services.magenta_timeout,
                'retry_attempts': self.services.retry_attempts,
                'retry_delay': self.services.retry_delay
            },
            'processing': {
                'max_audio_duration': self.processing.max_audio_duration,
                'min_audio_duration': self.processing.min_audio_duration,
                'max_onset_count': self.processing.max_onset_count,
                'temporal_scales': self.processing.temporal_scales
            }
        }
    
    def save_to_file(self, config_path: str):
        """
        Save settings to a YAML configuration file.
        
        Args:
            config_path: Path to save the configuration file
            
        Raises:
            ConfigurationError: If file cannot be saved
        """
        try:
            config_dir = Path(config_path).parent
            config_dir.mkdir(parents=True, exist_ok=True)
            
            with open(config_path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=True)
                
        except Exception as e:
            raise ConfigurationError(f"Error saving configuration: {e}")


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the global settings instance.
    
    Returns:
        Settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings.from_environment()
    return _settings


def load_settings(config_path: Optional[str] = None) -> Settings:
    """
    Load settings from file or environment.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Settings instance
    """
    global _settings
    
    if config_path and os.path.exists(config_path):
        _settings = Settings.load_from_file(config_path)
    else:
        _settings = Settings.from_environment()
    
    return _settings