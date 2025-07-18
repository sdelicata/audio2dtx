"""
Spectral analysis utilities for audio processing.
"""

import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional, Any
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks

from ..config.settings import Settings
from ..config.constants import FREQ_BANDS
from ..utils.exceptions import ProcessingError
from ..utils.logging import get_logger

logger = get_logger(__name__)


class SpectralAnalyzer:
    """
    Spectral analysis for audio signals.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize spectral analyzer.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.sr = settings.audio.sample_rate
        self.hop_length = settings.audio.hop_length
        self.n_fft = settings.audio.n_fft
        self.n_mels = settings.audio.n_mels
        
    def compute_stft(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Short-Time Fourier Transform.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Tuple of (magnitude, phase)
        """
        stft = librosa.stft(audio, hop_length=self.hop_length, n_fft=self.n_fft)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        return magnitude, phase
    
    def compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute mel-scale spectrogram.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Mel-scale spectrogram
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            hop_length=self.hop_length,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=20,
            fmax=self.sr // 2
        )
        
        return mel_spec
    
    def compute_onset_strength(self, audio: np.ndarray, method: str = 'energy') -> np.ndarray:
        """
        Compute onset strength function.
        
        Args:
            audio: Input audio signal
            method: Onset detection method
            
        Returns:
            Onset strength envelope
        """
        try:
            if method == 'energy':
                onset_envelope = librosa.onset.onset_strength(
                    y=audio, sr=self.sr, hop_length=self.hop_length
                )
            elif method == 'spectral_flux':
                # Compute spectral flux manually
                magnitude, _ = self.compute_stft(audio)
                # Calculate spectral difference
                flux = np.sum(np.diff(magnitude, axis=1) ** 2, axis=0)
                # Pad to match expected length
                onset_envelope = np.concatenate([[0], flux])
            else:
                # Use librosa method
                onset_envelope = getattr(librosa.onset, f'onset_{method}')(
                    y=audio, sr=self.sr, hop_length=self.hop_length
                )
            
            return onset_envelope
            
        except Exception as e:
            logger.error(f"Failed to compute onset strength using {method}: {e}")
            # Fallback to simple energy
            rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
            return rms
    
    def analyze_frequency_content(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Analyze frequency content of audio signal.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary with frequency analysis results
        """
        magnitude, _ = self.compute_stft(audio)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        
        # Calculate frequency band energies
        total_energy = np.sum(magnitude)
        band_energies = {}
        
        for band_name, (low_freq, high_freq) in FREQ_BANDS.items():
            # Find frequency bin indices
            low_bin = np.argmax(freqs >= low_freq)
            high_bin = np.argmax(freqs >= high_freq)
            if high_bin == 0:
                high_bin = len(freqs)
            
            # Calculate band energy
            band_energy = np.sum(magnitude[low_bin:high_bin])
            
            if total_energy > 0:
                band_energies[f'{band_name}_energy_ratio'] = float(band_energy / total_energy)
            else:
                band_energies[f'{band_name}_energy_ratio'] = 0.0
            
            band_energies[f'{band_name}_energy_absolute'] = float(band_energy)
        
        # Calculate spectral features
        spectral_features = {}
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(
            S=magnitude, sr=self.sr, hop_length=self.hop_length
        )
        spectral_features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
        spectral_features['spectral_centroid_std'] = float(np.std(spectral_centroid))
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(
            S=magnitude, sr=self.sr, hop_length=self.hop_length
        )
        spectral_features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        spectral_features['spectral_rolloff_std'] = float(np.std(spectral_rolloff))
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            S=magnitude, sr=self.sr, hop_length=self.hop_length
        )
        spectral_features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
        spectral_features['spectral_bandwidth_std'] = float(np.std(spectral_bandwidth))
        
        # Combine results
        result = {**band_energies, **spectral_features}
        
        return result
    
    def detect_harmonic_percussive(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate harmonic and percussive components.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Tuple of (harmonic, percussive) components
        """
        try:
            harmonic, percussive = librosa.effects.hpss(audio)
            return harmonic, percussive
        except Exception as e:
            logger.error(f"Harmonic-percussive separation failed: {e}")
            # Return original as percussive
            return np.zeros_like(audio), audio
    
    def analyze_temporal_envelope(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Analyze temporal envelope characteristics.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary with temporal analysis results
        """
        # Calculate RMS envelope
        rms = librosa.feature.rms(y=audio, hop_length=self.hop_length)[0]
        
        # Calculate envelope statistics
        envelope_stats = {
            'rms_mean': float(np.mean(rms)),
            'rms_std': float(np.std(rms)),
            'rms_max': float(np.max(rms)),
            'rms_min': float(np.min(rms))
        }
        
        # Attack and decay analysis
        peak_idx = np.argmax(rms)
        if peak_idx > 0 and peak_idx < len(rms) - 1:
            # Attack time (time to reach peak)
            attack_samples = peak_idx
            attack_time = attack_samples * self.hop_length / self.sr
            envelope_stats['attack_time'] = float(attack_time)
            
            # Decay analysis
            decay_envelope = rms[peak_idx:]
            if len(decay_envelope) > 1:
                peak_level = rms[peak_idx]
                # Find 60% decay point
                decay_target = peak_level * 0.4
                decay_indices = np.where(decay_envelope <= decay_target)[0]
                if len(decay_indices) > 0:
                    decay_samples = decay_indices[0]
                    decay_time = decay_samples * self.hop_length / self.sr
                    envelope_stats['decay_time'] = float(decay_time)
                else:
                    envelope_stats['decay_time'] = float(len(decay_envelope) * self.hop_length / self.sr)
            else:
                envelope_stats['decay_time'] = 0.0
        else:
            envelope_stats['attack_time'] = 0.0
            envelope_stats['decay_time'] = 0.0
        
        return envelope_stats
    
    def compute_spectral_features_comprehensive(self, audio: np.ndarray) -> Dict[str, Any]:
        """
        Compute comprehensive spectral features.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary with all spectral features
        """
        features = {}
        
        # Basic spectral analysis
        features.update(self.analyze_frequency_content(audio))
        
        # Temporal envelope
        features.update(self.analyze_temporal_envelope(audio))
        
        # MFCC features
        try:
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sr,
                n_mfcc=13,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            )
            
            mfcc_stats = {}
            for i in range(mfcc.shape[0]):
                mfcc_stats[f'mfcc_{i}_mean'] = float(np.mean(mfcc[i]))
                mfcc_stats[f'mfcc_{i}_std'] = float(np.std(mfcc[i]))
            
            features.update(mfcc_stats)
            
        except Exception as e:
            logger.warning(f"MFCC computation failed: {e}")
        
        # Chroma features
        try:
            chroma = librosa.feature.chroma_stft(
                y=audio,
                sr=self.sr,
                hop_length=self.hop_length,
                n_fft=self.n_fft
            )
            
            chroma_stats = {
                'chroma_mean': float(np.mean(chroma)),
                'chroma_std': float(np.std(chroma)),
                'chroma_energy': float(np.sum(chroma))
            }
            
            features.update(chroma_stats)
            
        except Exception as e:
            logger.warning(f"Chroma computation failed: {e}")
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features.update({
            'zcr_mean': float(np.mean(zcr)),
            'zcr_std': float(np.std(zcr))
        })
        
        return features
    
    def find_spectral_peaks(self, audio: np.ndarray, n_peaks: int = 5) -> List[Tuple[float, float]]:
        """
        Find dominant spectral peaks.
        
        Args:
            audio: Input audio signal
            n_peaks: Number of peaks to return
            
        Returns:
            List of (frequency, magnitude) tuples
        """
        magnitude, _ = self.compute_stft(audio)
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        
        # Average magnitude over time
        avg_magnitude = np.mean(magnitude, axis=1)
        
        # Find peaks
        peaks, properties = find_peaks(
            avg_magnitude,
            height=np.max(avg_magnitude) * 0.1,  # At least 10% of max
            distance=10  # Minimum distance between peaks
        )
        
        # Sort by magnitude
        peak_magnitudes = avg_magnitude[peaks]
        sorted_indices = np.argsort(peak_magnitudes)[::-1]
        
        # Return top n_peaks
        result_peaks = []
        for i in sorted_indices[:n_peaks]:
            peak_idx = peaks[i]
            freq = freqs[peak_idx]
            mag = peak_magnitudes[i]
            result_peaks.append((float(freq), float(mag)))
        
        return result_peaks
    
    def compute_spectral_similarity(self, audio1: np.ndarray, audio2: np.ndarray) -> float:
        """
        Compute spectral similarity between two audio signals.
        
        Args:
            audio1: First audio signal
            audio2: Second audio signal
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        try:
            # Compute mel spectrograms
            mel1 = self.compute_mel_spectrogram(audio1)
            mel2 = self.compute_mel_spectrogram(audio2)
            
            # Make same length
            min_frames = min(mel1.shape[1], mel2.shape[1])
            mel1 = mel1[:, :min_frames]
            mel2 = mel2[:, :min_frames]
            
            # Convert to dB
            mel1_db = librosa.power_to_db(mel1)
            mel2_db = librosa.power_to_db(mel2)
            
            # Compute cosine similarity
            mel1_flat = mel1_db.flatten()
            mel2_flat = mel2_db.flatten()
            
            dot_product = np.dot(mel1_flat, mel2_flat)
            norm1 = np.linalg.norm(mel1_flat)
            norm2 = np.linalg.norm(mel2_flat)
            
            if norm1 > 0 and norm2 > 0:
                similarity = dot_product / (norm1 * norm2)
                # Convert from [-1, 1] to [0, 1]
                similarity = (similarity + 1) / 2
            else:
                similarity = 0.0
            
            return float(np.clip(similarity, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Spectral similarity computation failed: {e}")
            return 0.0