"""
Feature extraction for drum classification.
"""

import numpy as np
import librosa
from typing import Dict, List, Optional, Any
from scipy.stats import skew, kurtosis

from ..config.settings import Settings
from ..config.constants import FREQ_BANDS
from ..utils.exceptions import ProcessingError


class FeatureExtractor:
    """
    Unified feature extractor for drum classification.
    
    Extracts various audio features used by different classification tracks.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize feature extractor.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.sr = settings.audio.sample_rate
        self.hop_length = settings.audio.hop_length
        self.n_fft = settings.audio.n_fft
        self.n_mels = settings.audio.n_mels
        self.n_mfcc = settings.classification.n_mfcc
        self.n_chroma = settings.classification.n_chroma
        self.n_contrast = settings.classification.n_contrast
        
        # Cache for computed features
        self.feature_cache = {}
        
    def extract_basic_features(self, audio_window: np.ndarray) -> Dict[str, float]:
        """
        Extract basic spectral and temporal features.
        
        Args:
            audio_window: Audio data
            
        Returns:
            Dictionary of basic features
        """
        try:
            features = {}
            
            # Temporal features
            features['rms_energy'] = float(np.sqrt(np.mean(audio_window**2)))
            features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(audio_window)))
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio_window, sr=self.sr, hop_length=self.hop_length
            )
            features['spectral_centroid'] = float(np.mean(spectral_centroid))
            
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_window, sr=self.sr, hop_length=self.hop_length
            )
            features['spectral_rolloff'] = float(np.mean(spectral_rolloff))
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio_window, sr=self.sr, hop_length=self.hop_length
            )
            features['spectral_bandwidth'] = float(np.mean(spectral_bandwidth))
            
            return features
            
        except Exception as e:
            raise ProcessingError(f"Basic feature extraction failed: {e}")
    
    def extract_mfcc_features(self, audio_window: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract MFCC features and their derivatives.
        
        Args:
            audio_window: Audio data
            
        Returns:
            Dictionary of MFCC features
        """
        try:
            # Extract MFCCs
            mfcc = librosa.feature.mfcc(
                y=audio_window,
                sr=self.sr,
                n_mfcc=self.n_mfcc,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            )
            
            # Calculate delta (first derivative)
            delta_mfcc = librosa.feature.delta(mfcc)
            
            # Calculate delta-delta (second derivative)
            delta2_mfcc = librosa.feature.delta(mfcc, order=2)
            
            return {
                'mfcc': np.mean(mfcc, axis=1),
                'delta_mfcc': np.mean(delta_mfcc, axis=1),
                'delta2_mfcc': np.mean(delta2_mfcc, axis=1)
            }
            
        except Exception as e:
            raise ProcessingError(f"MFCC feature extraction failed: {e}")
    
    def extract_spectral_features(self, audio_window: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract spectral features (chroma, contrast, tonnetz).
        
        Args:
            audio_window: Audio data
            
        Returns:
            Dictionary of spectral features
        """
        try:
            features = {}
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(
                y=audio_window,
                sr=self.sr,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            )
            features['chroma'] = np.mean(chroma, axis=1)
            
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(
                y=audio_window,
                sr=self.sr,
                n_bands=self.n_contrast-1,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            )
            features['spectral_contrast'] = np.mean(contrast, axis=1)
            
            # Tonnetz features
            tonnetz = librosa.feature.tonnetz(
                y=audio_window,
                sr=self.sr,
                hop_length=self.hop_length
            )
            features['tonnetz'] = np.mean(tonnetz, axis=1)
            
            return features
            
        except Exception as e:
            raise ProcessingError(f"Spectral feature extraction failed: {e}")
    
    def extract_frequency_band_features(self, audio_window: np.ndarray) -> Dict[str, float]:
        """
        Extract energy features for different frequency bands.
        
        Args:
            audio_window: Audio data
            
        Returns:
            Dictionary of frequency band features
        """
        try:
            # Compute STFT
            stft = librosa.stft(audio_window, hop_length=self.hop_length, n_fft=self.n_fft)
            magnitude = np.abs(stft)
            
            # Get frequency bins
            freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
            
            features = {}
            total_energy = np.sum(magnitude)
            
            for band_name, (low_freq, high_freq) in FREQ_BANDS.items():
                # Find frequency bin indices
                low_bin = np.argmax(freqs >= low_freq)
                high_bin = np.argmax(freqs >= high_freq)
                if high_bin == 0:  # If no frequency is >= high_freq
                    high_bin = len(freqs)
                
                # Calculate band energy
                band_energy = np.sum(magnitude[low_bin:high_bin])
                
                # Normalize by total energy
                if total_energy > 0:
                    features[f'{band_name}_energy_ratio'] = float(band_energy / total_energy)
                else:
                    features[f'{band_name}_energy_ratio'] = 0.0
                
                # Additional band statistics
                features[f'{band_name}_energy_mean'] = float(np.mean(magnitude[low_bin:high_bin]))
                features[f'{band_name}_energy_std'] = float(np.std(magnitude[low_bin:high_bin]))
            
            return features
            
        except Exception as e:
            raise ProcessingError(f"Frequency band feature extraction failed: {e}")
    
    def extract_statistical_features(self, audio_window: np.ndarray) -> Dict[str, float]:
        """
        Extract statistical features from the audio signal.
        
        Args:
            audio_window: Audio data
            
        Returns:
            Dictionary of statistical features
        """
        try:
            features = {}
            
            # Time domain statistics
            features['mean'] = float(np.mean(audio_window))
            features['std'] = float(np.std(audio_window))
            features['skewness'] = float(skew(audio_window))
            features['kurtosis'] = float(kurtosis(audio_window))
            features['min'] = float(np.min(audio_window))
            features['max'] = float(np.max(audio_window))
            
            # Spectral statistics
            stft = librosa.stft(audio_window, hop_length=self.hop_length, n_fft=self.n_fft)
            magnitude_spectrum = np.abs(stft)
            power_spectrum = magnitude_spectrum ** 2
            
            # Flatten for statistics
            flat_spectrum = power_spectrum.flatten()
            features['spectral_mean'] = float(np.mean(flat_spectrum))
            features['spectral_std'] = float(np.std(flat_spectrum))
            features['spectral_skewness'] = float(skew(flat_spectrum))
            features['spectral_kurtosis'] = float(kurtosis(flat_spectrum))
            
            return features
            
        except Exception as e:
            raise ProcessingError(f"Statistical feature extraction failed: {e}")
    
    def extract_temporal_features(self, audio_window: np.ndarray) -> Dict[str, float]:
        """
        Extract temporal characteristics of the audio.
        
        Args:
            audio_window: Audio data
            
        Returns:
            Dictionary of temporal features
        """
        try:
            features = {}
            
            # Attack time (time to reach peak)
            peak_idx = np.argmax(np.abs(audio_window))
            features['attack_time'] = float(peak_idx / self.sr)
            
            # Decay characteristics
            if peak_idx < len(audio_window) - 1:
                decay_signal = audio_window[peak_idx:]
                envelope = np.abs(decay_signal)
                if len(envelope) > 1 and np.max(envelope) > 0:
                    # Find 60% decay point
                    peak_level = np.max(envelope)
                    decay_target = peak_level * 0.4  # 60% decay
                    decay_indices = np.where(envelope <= decay_target)[0]
                    if len(decay_indices) > 0:
                        decay_time = float(decay_indices[0] / self.sr)
                        features['decay_time'] = decay_time
                    else:
                        features['decay_time'] = float(len(decay_signal) / self.sr)
                else:
                    features['decay_time'] = 0.0
            else:
                features['decay_time'] = 0.0
            
            # Spectral flux (measure of spectral change)
            stft = librosa.stft(audio_window, hop_length=self.hop_length, n_fft=self.n_fft)
            spectral_flux = np.sum(np.diff(np.abs(stft), axis=1)**2, axis=0)
            features['spectral_flux_mean'] = float(np.mean(spectral_flux))
            features['spectral_flux_std'] = float(np.std(spectral_flux))
            
            return features
            
        except Exception as e:
            raise ProcessingError(f"Temporal feature extraction failed: {e}")
    
    def extract_all_features(self, 
                           audio_window: np.ndarray,
                           feature_set: str = 'basic') -> Dict[str, Any]:
        """
        Extract a comprehensive set of features.
        
        Args:
            audio_window: Audio data
            feature_set: Feature set to extract ('basic', 'advanced', 'comprehensive')
            
        Returns:
            Dictionary of all extracted features
        """
        features = {}
        
        # Always include basic features
        features.update(self.extract_basic_features(audio_window))
        features.update(self.extract_frequency_band_features(audio_window))
        
        if feature_set in ['advanced', 'comprehensive']:
            features.update(self.extract_statistical_features(audio_window))
            features.update(self.extract_temporal_features(audio_window))
            
            # Add MFCC features as flattened arrays
            mfcc_features = self.extract_mfcc_features(audio_window)
            for key, values in mfcc_features.items():
                for i, value in enumerate(values):
                    features[f'{key}_{i}'] = float(value)
        
        if feature_set == 'comprehensive':
            # Add spectral features as flattened arrays
            spectral_features = self.extract_spectral_features(audio_window)
            for key, values in spectral_features.items():
                for i, value in enumerate(values):
                    features[f'{key}_{i}'] = float(value)
        
        return features
    
    def get_feature_vector(self, 
                          audio_window: np.ndarray,
                          feature_set: str = 'basic') -> np.ndarray:
        """
        Extract features and return as a numerical vector.
        
        Args:
            audio_window: Audio data
            feature_set: Feature set to extract
            
        Returns:
            Feature vector as numpy array
        """
        features = self.extract_all_features(audio_window, feature_set)
        
        # Convert to numerical vector
        feature_values = []
        for key in sorted(features.keys()):
            value = features[key]
            if isinstance(value, (int, float)):
                feature_values.append(float(value))
            elif isinstance(value, np.ndarray):
                feature_values.extend(value.flatten().astype(float))
        
        return np.array(feature_values)
    
    def clear_cache(self):
        """Clear the feature cache."""
        self.feature_cache.clear()