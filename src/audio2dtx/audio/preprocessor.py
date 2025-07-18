"""
Audio preprocessing utilities.
"""

import numpy as np
import librosa
from scipy.signal import savgol_filter
from typing import Dict, Any, Optional, Tuple

from ..config.settings import Settings
from ..utils.exceptions import ProcessingError
from ..utils.logging import get_logger

logger = get_logger(__name__)


class AudioPreprocessor:
    """
    Audio preprocessing for improved analysis and classification.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize audio preprocessor.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.sr = settings.audio.sample_rate
        self.hop_length = settings.audio.hop_length
        self.n_fft = settings.audio.n_fft
        
    def normalize_audio(self, audio: np.ndarray, method: str = "peak") -> np.ndarray:
        """
        Normalize audio signal.
        
        Args:
            audio: Input audio signal
            method: Normalization method ("peak", "rms", "lufs")
            
        Returns:
            Normalized audio signal
        """
        if len(audio) == 0:
            return audio
        
        if method == "peak":
            # Peak normalization
            peak = np.max(np.abs(audio))
            if peak > 0:
                return audio / peak
            else:
                return audio
                
        elif method == "rms":
            # RMS normalization
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                target_rms = 0.1  # Target RMS level
                return audio * (target_rms / rms)
            else:
                return audio
                
        elif method == "lufs":
            # Simple LUFS-like normalization
            # This is a simplified version
            rms = np.sqrt(np.mean(audio**2))
            if rms > 0:
                # Target around -23 LUFS equivalent
                target_level = 0.1
                return audio * (target_level / rms)
            else:
                return audio
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    def apply_noise_reduction(self, audio: np.ndarray, noise_factor: float = 0.1) -> np.ndarray:
        """
        Apply basic noise reduction using spectral subtraction.
        
        Args:
            audio: Input audio signal
            noise_factor: Noise reduction factor (0.0 to 1.0)
            
        Returns:
            Noise-reduced audio signal
        """
        try:
            # Compute STFT
            stft = librosa.stft(audio, hop_length=self.hop_length, n_fft=self.n_fft)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise floor from quiet segments
            # Use first and last 10% of the signal as noise estimate
            noise_frames = int(magnitude.shape[1] * 0.1)
            noise_magnitude = np.mean(
                np.concatenate([
                    magnitude[:, :noise_frames],
                    magnitude[:, -noise_frames:]
                ], axis=1),
                axis=1,
                keepdims=True
            )
            
            # Apply spectral subtraction
            clean_magnitude = magnitude - (noise_factor * noise_magnitude)
            
            # Ensure non-negative values
            clean_magnitude = np.maximum(clean_magnitude, 0.1 * magnitude)
            
            # Reconstruct signal
            clean_stft = clean_magnitude * np.exp(1j * phase)
            clean_audio = librosa.istft(clean_stft, hop_length=self.hop_length)
            
            return clean_audio
            
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
            return audio
    
    def apply_dynamic_range_compression(self, 
                                      audio: np.ndarray,
                                      threshold: float = 0.5,
                                      ratio: float = 4.0,
                                      attack: float = 0.003,
                                      release: float = 0.1) -> np.ndarray:
        """
        Apply dynamic range compression.
        
        Args:
            audio: Input audio signal
            threshold: Compression threshold (0.0 to 1.0)
            ratio: Compression ratio
            attack: Attack time in seconds
            release: Release time in seconds
            
        Returns:
            Compressed audio signal
        """
        try:
            # Calculate envelope
            envelope = np.abs(audio)
            
            # Smooth envelope
            window_length = int(self.sr * 0.01)  # 10ms window
            if window_length % 2 == 0:
                window_length += 1
            if window_length >= len(envelope):
                return audio
                
            envelope = savgol_filter(envelope, window_length, 3)
            
            # Calculate gain reduction
            gain_reduction = np.ones_like(envelope)
            over_threshold = envelope > threshold
            
            if np.any(over_threshold):
                # Apply compression to signals over threshold
                excess = envelope[over_threshold] - threshold
                compressed_excess = excess / ratio
                gain_reduction[over_threshold] = (threshold + compressed_excess) / envelope[over_threshold]
            
            # Apply gain reduction
            compressed_audio = audio * gain_reduction
            
            return compressed_audio
            
        except Exception as e:
            logger.warning(f"Dynamic range compression failed: {e}")
            return audio
    
    def enhance_transients(self, audio: np.ndarray, factor: float = 1.5) -> np.ndarray:
        """
        Enhance transients in the audio signal.
        
        Args:
            audio: Input audio signal
            factor: Enhancement factor
            
        Returns:
            Enhanced audio signal
        """
        try:
            # Separate harmonic and percussive components
            harmonic, percussive = librosa.effects.hpss(audio)
            
            # Enhance percussive component
            enhanced_percussive = percussive * factor
            
            # Combine back
            enhanced_audio = harmonic + enhanced_percussive
            
            # Normalize to prevent clipping
            peak = np.max(np.abs(enhanced_audio))
            if peak > 1.0:
                enhanced_audio = enhanced_audio / peak
            
            return enhanced_audio
            
        except Exception as e:
            logger.warning(f"Transient enhancement failed: {e}")
            return audio
    
    def apply_high_pass_filter(self, audio: np.ndarray, cutoff_freq: float = 40.0) -> np.ndarray:
        """
        Apply high-pass filter to remove low-frequency noise.
        
        Args:
            audio: Input audio signal
            cutoff_freq: Cutoff frequency in Hz
            
        Returns:
            Filtered audio signal
        """
        try:
            # Design and apply butterworth high-pass filter
            from scipy.signal import butter, filtfilt
            
            nyquist = self.sr / 2
            normal_cutoff = cutoff_freq / nyquist
            
            if normal_cutoff >= 1.0:
                logger.warning(f"Cutoff frequency {cutoff_freq}Hz too high for sample rate {self.sr}Hz")
                return audio
            
            b, a = butter(4, normal_cutoff, btype='high', analog=False)
            filtered_audio = filtfilt(b, a, audio)
            
            return filtered_audio
            
        except Exception as e:
            logger.warning(f"High-pass filtering failed: {e}")
            return audio
    
    def preprocess_for_classification(self, 
                                    audio: np.ndarray,
                                    enable_noise_reduction: bool = True,
                                    enable_compression: bool = True,
                                    enable_transient_enhancement: bool = True) -> np.ndarray:
        """
        Apply comprehensive preprocessing for drum classification.
        
        Args:
            audio: Input audio signal
            enable_noise_reduction: Whether to apply noise reduction
            enable_compression: Whether to apply compression
            enable_transient_enhancement: Whether to enhance transients
            
        Returns:
            Preprocessed audio signal
        """
        processed_audio = audio.copy()
        
        # Step 1: High-pass filter to remove low-frequency noise
        processed_audio = self.apply_high_pass_filter(processed_audio)
        logger.debug("Applied high-pass filter")
        
        # Step 2: Noise reduction
        if enable_noise_reduction:
            processed_audio = self.apply_noise_reduction(processed_audio)
            logger.debug("Applied noise reduction")
        
        # Step 3: Dynamic range compression
        if enable_compression:
            processed_audio = self.apply_dynamic_range_compression(processed_audio)
            logger.debug("Applied dynamic range compression")
        
        # Step 4: Transient enhancement
        if enable_transient_enhancement:
            processed_audio = self.enhance_transients(processed_audio)
            logger.debug("Applied transient enhancement")
        
        # Step 5: Final normalization
        processed_audio = self.normalize_audio(processed_audio, method="peak")
        logger.debug("Applied final normalization")
        
        return processed_audio
    
    def preprocess_for_onset_detection(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing optimized for onset detection.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Preprocessed audio signal
        """
        # For onset detection, we want to preserve transients
        processed_audio = audio.copy()
        
        # Light high-pass filtering
        processed_audio = self.apply_high_pass_filter(processed_audio, cutoff_freq=20.0)
        
        # Enhance transients more aggressively
        processed_audio = self.enhance_transients(processed_audio, factor=2.0)
        
        # Normalize
        processed_audio = self.normalize_audio(processed_audio, method="peak")
        
        return processed_audio
    
    def analyze_audio_characteristics(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Analyze audio characteristics for preprocessing decisions.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary with audio characteristics
        """
        characteristics = {}
        
        # Basic statistics
        characteristics['rms_level'] = float(np.sqrt(np.mean(audio**2)))
        characteristics['peak_level'] = float(np.max(np.abs(audio)))
        characteristics['dynamic_range'] = float(np.max(audio) - np.min(audio))
        characteristics['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
        
        # Spectral characteristics
        stft = librosa.stft(audio, hop_length=self.hop_length, n_fft=self.n_fft)
        magnitude = np.abs(stft)
        
        spectral_centroid = librosa.feature.spectral_centroid(
            S=magnitude, sr=self.sr, hop_length=self.hop_length
        )
        characteristics['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
        characteristics['spectral_centroid_std'] = float(np.std(spectral_centroid))
        
        spectral_rolloff = librosa.feature.spectral_rolloff(
            S=magnitude, sr=self.sr, hop_length=self.hop_length
        )
        characteristics['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        
        # Estimate noise level
        noise_frames = int(magnitude.shape[1] * 0.1)
        if noise_frames > 0:
            noise_magnitude = np.mean(magnitude[:, :noise_frames])
            signal_magnitude = np.mean(magnitude)
            characteristics['estimated_snr'] = float(20 * np.log10(signal_magnitude / (noise_magnitude + 1e-8)))
        else:
            characteristics['estimated_snr'] = float('inf')
        
        return characteristics