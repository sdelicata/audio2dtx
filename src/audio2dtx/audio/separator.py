"""
Audio source separation using Spleeter.
"""

import os
import tempfile
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from ..config.settings import Settings
from ..utils.exceptions import ProcessingError
from ..utils.logging import get_logger

logger = get_logger(__name__)


class AudioSeparator:
    """
    Audio source separation for isolating drum tracks.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize audio separator.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.sr = settings.audio.sample_rate
        self.separator = None
        self.adapter = None
        
    def _initialize_spleeter(self):
        """Initialize Spleeter components."""
        try:
            from spleeter.separator import Separator
            from spleeter.audio.adapter import AudioAdapter
            
            # Initialize separator with 4stems model (vocals, drums, bass, other)
            self.separator = Separator('spleeter:4stems-16kHz')
            self.adapter = AudioAdapter.default()
            
            logger.info("Spleeter initialized successfully")
            
        except ImportError:
            logger.error("Spleeter not available. Audio separation will be disabled.")
            self.separator = None
            self.adapter = None
        except Exception as e:
            logger.error(f"Failed to initialize Spleeter: {e}")
            self.separator = None
            self.adapter = None
    
    def separate_audio(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Separate audio into stems using Spleeter.
        
        Args:
            audio: Input audio signal (mono)
            
        Returns:
            Dictionary with separated audio stems
            
        Raises:
            ProcessingError: If separation fails
        """
        if self.separator is None:
            self._initialize_spleeter()
        
        if self.separator is None:
            # Fallback: return original audio as drums
            logger.warning("Spleeter not available, using original audio as drums")
            return {
                'drums': audio,
                'vocals': np.zeros_like(audio),
                'bass': np.zeros_like(audio),
                'other': np.zeros_like(audio)
            }
        
        try:
            # Convert mono to stereo for Spleeter
            if audio.ndim == 1:
                stereo_audio = np.stack([audio, audio], axis=-1)
            else:
                stereo_audio = audio
            
            # Spleeter expects specific format
            waveform = stereo_audio.astype(np.float32)
            
            # Perform separation
            logger.info("Separating audio into stems...")
            separated = self.separator.separate(waveform)
            
            # Convert back to mono and resample if needed
            stems = {}
            for stem_name, stem_audio in separated.items():
                # Convert to mono
                if stem_audio.ndim > 1:
                    mono_stem = librosa.to_mono(stem_audio.T)
                else:
                    mono_stem = stem_audio
                
                # Resample to target sample rate if needed
                if len(mono_stem) != len(audio):
                    mono_stem = librosa.resample(
                        mono_stem, 
                        orig_sr=16000,  # Spleeter default
                        target_sr=self.sr
                    )
                
                stems[stem_name] = mono_stem
            
            logger.info(f"Successfully separated audio into {len(stems)} stems")
            return stems
            
        except Exception as e:
            logger.error(f"Audio separation failed: {e}")
            # Return fallback
            return {
                'drums': audio,
                'vocals': np.zeros_like(audio),
                'bass': np.zeros_like(audio),
                'other': np.zeros_like(audio)
            }
    
    def extract_drums(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract drum track from audio.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Isolated drum track
        """
        stems = self.separate_audio(audio)
        return stems.get('drums', audio)
    
    def create_backing_track(self, 
                           audio: np.ndarray,
                           include_drums: bool = False) -> np.ndarray:
        """
        Create backing track (everything except drums, or original audio).
        
        Args:
            audio: Input audio signal
            include_drums: Whether to include drums in backing track
            
        Returns:
            Backing track audio
        """
        if include_drums:
            # Return original audio
            return audio
        
        stems = self.separate_audio(audio)
        
        # Combine non-drum stems
        backing_track = np.zeros_like(audio)
        for stem_name, stem_audio in stems.items():
            if stem_name != 'drums':
                backing_track += stem_audio
        
        # Normalize
        if np.max(np.abs(backing_track)) > 0:
            backing_track = backing_track / np.max(np.abs(backing_track))
        
        return backing_track
    
    def analyze_separation_quality(self, 
                                 original: np.ndarray,
                                 stems: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Analyze the quality of audio separation.
        
        Args:
            original: Original audio signal
            stems: Separated audio stems
            
        Returns:
            Dictionary with quality metrics
        """
        metrics = {}
        
        # Calculate reconstruction error
        reconstructed = np.zeros_like(original)
        for stem_audio in stems.values():
            if len(stem_audio) == len(original):
                reconstructed += stem_audio
        
        if len(reconstructed) == len(original):
            reconstruction_error = np.mean((original - reconstructed) ** 2)
            metrics['reconstruction_mse'] = float(reconstruction_error)
            
            # Signal-to-noise ratio
            signal_power = np.mean(original ** 2)
            noise_power = reconstruction_error
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
                metrics['reconstruction_snr'] = float(snr)
            else:
                metrics['reconstruction_snr'] = float('inf')
        
        # Analyze individual stems
        for stem_name, stem_audio in stems.items():
            if len(stem_audio) == len(original):
                # RMS level
                rms = np.sqrt(np.mean(stem_audio ** 2))
                metrics[f'{stem_name}_rms'] = float(rms)
                
                # Dynamic range
                dynamic_range = np.max(stem_audio) - np.min(stem_audio)
                metrics[f'{stem_name}_dynamic_range'] = float(dynamic_range)
        
        return metrics
    
    def save_stems(self, 
                  stems: Dict[str, np.ndarray],
                  output_dir: str,
                  base_filename: str) -> Dict[str, str]:
        """
        Save separated stems to files.
        
        Args:
            stems: Dictionary of separated audio stems
            output_dir: Output directory
            base_filename: Base filename (without extension)
            
        Returns:
            Dictionary mapping stem names to file paths
        """
        import soundfile as sf
        
        os.makedirs(output_dir, exist_ok=True)
        saved_files = {}
        
        for stem_name, stem_audio in stems.items():
            output_path = os.path.join(output_dir, f"{base_filename}_{stem_name}.wav")
            
            try:
                sf.write(output_path, stem_audio, self.sr)
                saved_files[stem_name] = output_path
                logger.debug(f"Saved {stem_name} stem to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save {stem_name} stem: {e}")
        
        return saved_files
    
    def is_available(self) -> bool:
        """
        Check if Spleeter is available.
        
        Returns:
            True if Spleeter can be used
        """
        if self.separator is None:
            self._initialize_spleeter()
        
        return self.separator is not None
    
    def get_info(self) -> Dict[str, any]:
        """
        Get information about the separator.
        
        Returns:
            Dictionary with separator information
        """
        return {
            'available': self.is_available(),
            'model': 'spleeter:4stems-16kHz' if self.is_available() else None,
            'stems': ['vocals', 'drums', 'bass', 'other'] if self.is_available() else []
        }