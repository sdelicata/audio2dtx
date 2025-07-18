"""
Onset detection interface and implementation.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import librosa
from dataclasses import dataclass

from ..config.settings import Settings
from ..config.constants import ONSET_DETECTION_METHODS
from ..utils.exceptions import OnsetDetectionError
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OnsetDetectionResult:
    """Result of onset detection."""
    method: str
    onset_times: np.ndarray
    onset_strengths: Optional[np.ndarray] = None
    success: bool = True
    error_message: Optional[str] = None


class BaseOnsetDetector(ABC):
    """Abstract base class for onset detection algorithms."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.sr = settings.audio.sample_rate
        self.hop_length = settings.audio.hop_length
        self.name = self.__class__.__name__
        
    @abstractmethod
    def detect_onsets(self, audio: np.ndarray) -> OnsetDetectionResult:
        """
        Detect onsets in audio signal.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Onset detection result
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this detector."""
        return {
            'name': self.name,
            'sample_rate': self.sr,
            'hop_length': self.hop_length
        }


class LibrosaOnsetDetector(BaseOnsetDetector):
    """Librosa-based onset detection."""
    
    def __init__(self, settings: Settings, method: str = 'energy'):
        super().__init__(settings)
        if method not in ONSET_DETECTION_METHODS:
            raise ValueError(f"Unsupported onset detection method: {method}")
        self.method = method
        self.name = f"Librosa_{method}"
        
    def detect_onsets(self, audio: np.ndarray) -> OnsetDetectionResult:
        """Detect onsets using librosa method."""
        try:
            # Get onset envelope using the correct librosa 0.8.1 API
            if self.method == 'energy':
                # For energy method, use onset_strength with specific parameters
                onset_envelope = librosa.onset.onset_strength(
                    y=audio, sr=self.sr, hop_length=self.hop_length,
                    aggregate=np.median, fmax=8000, n_mels=256
                )
            else:
                # For other methods, use onset_strength with method parameter
                onset_envelope = librosa.onset.onset_strength(
                    y=audio, sr=self.sr, hop_length=self.hop_length,
                    feature=self.method if self.method in ['spectral_centroid', 'chroma', 'mfcc', 'melspectrogram'] else None
                )
            
            # Detect onset frames
            onset_frames = librosa.onset.onset_detect(
                y=audio,
                sr=self.sr,
                hop_length=self.hop_length,
                onset_envelope=onset_envelope
            )
            
            # Convert frames to time
            onset_times = librosa.frames_to_time(
                onset_frames, sr=self.sr, hop_length=self.hop_length
            )
            
            # Get onset strengths
            onset_strengths = onset_envelope[onset_frames] if len(onset_frames) > 0 else np.array([])
            
            return OnsetDetectionResult(
                method=self.method,
                onset_times=onset_times,
                onset_strengths=onset_strengths,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Onset detection failed for method {self.method}: {e}")
            return OnsetDetectionResult(
                method=self.method,
                onset_times=np.array([]),
                success=False,
                error_message=str(e)
            )


class EnergyBasedOnsetDetector(BaseOnsetDetector):
    """Energy-based onset detection."""
    
    def __init__(self, settings: Settings):
        super().__init__(settings)
        self.name = "EnergyBased"
        
    def detect_onsets(self, audio: np.ndarray) -> OnsetDetectionResult:
        """Detect onsets using energy analysis."""
        try:
            # Calculate RMS energy
            frame_length = self.hop_length * 2
            rms = librosa.feature.rms(
                y=audio, 
                frame_length=frame_length, 
                hop_length=self.hop_length
            )[0]
            
            # Calculate energy differences
            energy_diff = np.diff(rms)
            energy_diff = np.concatenate([[0], energy_diff])
            
            # Find peaks in energy differences
            from scipy.signal import find_peaks
            peak_indices, properties = find_peaks(
                energy_diff,
                height=np.std(energy_diff) * 0.5,
                distance=int(self.sr * 0.05 / self.hop_length)  # Min 50ms between onsets
            )
            
            # Convert to time
            onset_times = librosa.frames_to_time(
                peak_indices, sr=self.sr, hop_length=self.hop_length
            )
            
            onset_strengths = energy_diff[peak_indices] if len(peak_indices) > 0 else np.array([])
            
            return OnsetDetectionResult(
                method="energy_based",
                onset_times=onset_times,
                onset_strengths=onset_strengths,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Energy-based onset detection failed: {e}")
            return OnsetDetectionResult(
                method="energy_based",
                onset_times=np.array([]),
                success=False,
                error_message=str(e)
            )


class MLOnsetDetector(BaseOnsetDetector):
    """Machine learning based onset detection."""
    
    def __init__(self, settings: Settings, model_path: str = "PredictOnset.h5"):
        super().__init__(settings)
        self.model_path = model_path
        self.model = None
        self.name = "MLBased"
        
    def _load_model(self):
        """Load the ML model for onset detection."""
        try:
            import tensorflow as tf
            self.model = tf.keras.models.load_model(self.model_path)
            logger.info(f"Loaded ML onset detection model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            self.model = None
            
    def detect_onsets(self, audio: np.ndarray) -> OnsetDetectionResult:
        """Detect onsets using ML model."""
        try:
            if self.model is None:
                self._load_model()
                
            if self.model is None:
                raise OnsetDetectionError("ML model not available")
            
            # Prepare audio for model (mel-spectrogram)
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.sr,
                hop_length=self.hop_length,
                n_mels=128
            )
            
            # Convert to dB and normalize
            mel_spec_db = librosa.power_to_db(mel_spec)
            mel_spec_norm = (mel_spec_db - np.mean(mel_spec_db)) / (np.std(mel_spec_db) + 1e-8)
            
            # Reshape to match expected input format: (None, 128, 4)
            # Split the spectrogram into chunks of 4 frames each
            n_frames = mel_spec_norm.shape[1]
            n_chunks = n_frames // 4
            
            if n_chunks == 0:
                # If audio is too short, pad to create at least one chunk
                mel_spec_padded = np.pad(mel_spec_norm, ((0, 0), (0, 4)), mode='constant')
                n_chunks = 1
            else:
                # Truncate to fit complete chunks
                mel_spec_padded = mel_spec_norm[:, :n_chunks*4]
            
            # Reshape to (n_chunks, 128, 4) - model expects this format
            # mel_spec_padded shape: (128, n_chunks*4)
            # We need to reshape to: (n_chunks, 128, 4)
            model_input = mel_spec_padded.reshape(128, n_chunks, 4).transpose(1, 0, 2)
            
            
            # Get predictions - process in smaller batches if needed
            if n_chunks > 500:  # Process in batches to avoid memory issues
                predictions = []
                batch_size = 100
                for i in range(0, n_chunks, batch_size):
                    batch_end = min(i + batch_size, n_chunks)
                    batch_input = model_input[i:batch_end]
                    batch_pred = self.model.predict(batch_input, verbose=0)
                    predictions.append(batch_pred)
                predictions = np.concatenate(predictions, axis=0)
            else:
                predictions = self.model.predict(model_input, verbose=0)
            
            # Handle predictions based on output shape
            if len(predictions.shape) == 3:
                # Model outputs (time, batch, features) or (batch, time, features)
                if predictions.shape[0] == n_chunks:
                    # (time, batch, features) format - take first feature
                    onset_probs = predictions[:, 0, 0] if predictions.shape[2] > 0 else predictions[:, 0]
                else:
                    # (batch, time, features) format - take first feature  
                    onset_probs = predictions[0, :, 0] if predictions.shape[2] > 0 else predictions[0, :]
            elif len(predictions.shape) == 2:
                # Model outputs (time, features) or (batch, time)
                if predictions.shape[0] == n_chunks:
                    # (time, features) format - take first feature
                    onset_probs = predictions[:, 0] if predictions.shape[1] > 0 else predictions[:, 0]
                else:
                    # (batch, time) format - take first batch
                    onset_probs = predictions[0, :]
            else:
                # Fallback: flatten predictions
                onset_probs = predictions.flatten()
            
            # Find peaks in probabilities
            from scipy.signal import find_peaks
            threshold = np.mean(onset_probs) + 2 * np.std(onset_probs)
            peak_indices, _ = find_peaks(
                onset_probs,
                height=threshold,
                distance=max(1, int(self.sr * 0.05 / self.hop_length))  # Min 50ms between onsets
            )
            
            # Convert chunk indices back to time frames
            # Each chunk represents 4 frames, so multiply by 4
            frame_indices = peak_indices * 4
            
            # Convert to time
            onset_times = librosa.frames_to_time(
                frame_indices, sr=self.sr, hop_length=self.hop_length
            )
            
            onset_strengths = onset_probs[peak_indices] if len(peak_indices) > 0 else np.array([])
            
            return OnsetDetectionResult(
                method="ml_based",
                onset_times=onset_times,
                onset_strengths=onset_strengths,
                success=True
            )
            
        except Exception as e:
            logger.error(f"ML onset detection failed: {e}")
            return OnsetDetectionResult(
                method="ml_based",
                onset_times=np.array([]),
                success=False,
                error_message=str(e)
            )


class OnsetDetector:
    """
    Main onset detector that combines multiple detection methods.
    """
    
    def __init__(self, settings: Settings, model_path: str = "PredictOnset.h5"):
        """
        Initialize the onset detector.
        
        Args:
            settings: Application settings
            model_path: Path to ML model for onset detection
        """
        self.settings = settings
        self.detectors = []
        
        # Initialize all available detectors
        for method in ONSET_DETECTION_METHODS:
            self.detectors.append(LibrosaOnsetDetector(settings, method))
        
        # Add energy-based detector
        self.detectors.append(EnergyBasedOnsetDetector(settings))
        
        # Add ML-based detector
        self.detectors.append(MLOnsetDetector(settings, model_path))
        
        logger.info(f"Initialized {len(self.detectors)} onset detection methods")
        
    def detect_onsets(self, audio: np.ndarray) -> Dict[str, OnsetDetectionResult]:
        """
        Detect onsets using all available methods.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Dictionary mapping method names to results
        """
        results = {}
        
        for detector in self.detectors:
            logger.info(f"Running onset detection: {detector.name}")
            result = detector.detect_onsets(audio)
            results[detector.name] = result
            
            if result.success:
                logger.info(f"{detector.name}: {len(result.onset_times)} onsets detected")
            else:
                logger.warning(f"{detector.name}: failed - {result.error_message}")
        
        return results
    
    def fuse_onsets(self, 
                   detection_results: Dict[str, OnsetDetectionResult],
                   method: str = 'union') -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fuse onset detection results from multiple methods.
        
        Args:
            detection_results: Results from multiple detection methods
            method: Fusion method ('union', 'intersection', 'majority')
            
        Returns:
            Tuple of (fused_onset_times, fusion_info)
        """
        successful_results = {
            name: result for name, result in detection_results.items() 
            if result.success and len(result.onset_times) > 0
        }
        
        if not successful_results:
            logger.warning("No successful onset detection results to fuse")
            return np.array([]), {'method': method, 'sources': []}
        
        if method == 'union':
            # Combine all onsets and remove duplicates
            all_onsets = []
            for result in successful_results.values():
                all_onsets.extend(result.onset_times)
            
            if all_onsets:
                # Sort and remove duplicates (within 25ms tolerance)
                all_onsets = np.array(sorted(all_onsets))
                fused_onsets = []
                for onset in all_onsets:
                    if not fused_onsets or (onset - fused_onsets[-1]) > 0.025:
                        fused_onsets.append(onset)
                fused_onsets = np.array(fused_onsets)
            else:
                fused_onsets = np.array([])
                
        elif method == 'intersection':
            # Only keep onsets detected by multiple methods
            # This is more complex and requires clustering nearby onsets
            fused_onsets = self._find_consensus_onsets(successful_results)
            
        elif method == 'majority':
            # Keep onsets detected by majority of methods
            fused_onsets = self._find_majority_onsets(successful_results)
            
        else:
            raise ValueError(f"Unknown fusion method: {method}")
        
        fusion_info = {
            'method': method,
            'sources': list(successful_results.keys()),
            'total_onsets': len(fused_onsets),
            'source_counts': {name: len(result.onset_times) for name, result in successful_results.items()}
        }
        
        logger.info(f"Fused {len(fused_onsets)} onsets using {method} method")
        return fused_onsets, fusion_info
    
    def _find_consensus_onsets(self, results: Dict[str, OnsetDetectionResult], 
                              tolerance: float = 0.025) -> np.ndarray:
        """Find onsets that are detected by multiple methods."""
        if len(results) < 2:
            return list(results.values())[0].onset_times if results else np.array([])
        
        # Collect all onsets with their source
        onset_sources = []
        for method_name, result in results.items():
            for onset_time in result.onset_times:
                onset_sources.append((onset_time, method_name))
        
        # Sort by time
        onset_sources.sort(key=lambda x: x[0])
        
        # Group nearby onsets
        consensus_onsets = []
        i = 0
        while i < len(onset_sources):
            current_time = onset_sources[i][0]
            group = [onset_sources[i]]
            
            # Find all onsets within tolerance
            j = i + 1
            while j < len(onset_sources) and onset_sources[j][0] - current_time <= tolerance:
                group.append(onset_sources[j])
                j += 1
            
            # If detected by at least 2 methods, include it
            unique_methods = set(item[1] for item in group)
            if len(unique_methods) >= 2:
                # Use median time of the group
                group_times = [item[0] for item in group]
                consensus_onsets.append(np.median(group_times))
            
            i = j
        
        return np.array(consensus_onsets)
    
    def _find_majority_onsets(self, results: Dict[str, OnsetDetectionResult],
                             tolerance: float = 0.025) -> np.ndarray:
        """Find onsets detected by majority of methods."""
        if len(results) < 3:
            # For less than 3 methods, use consensus
            return self._find_consensus_onsets(results, tolerance)
        
        # Similar to consensus but require majority
        onset_sources = []
        for method_name, result in results.items():
            for onset_time in result.onset_times:
                onset_sources.append((onset_time, method_name))
        
        onset_sources.sort(key=lambda x: x[0])
        
        majority_onsets = []
        majority_threshold = len(results) // 2 + 1
        
        i = 0
        while i < len(onset_sources):
            current_time = onset_sources[i][0]
            group = [onset_sources[i]]
            
            j = i + 1
            while j < len(onset_sources) and onset_sources[j][0] - current_time <= tolerance:
                group.append(onset_sources[j])
                j += 1
            
            unique_methods = set(item[1] for item in group)
            if len(unique_methods) >= majority_threshold:
                group_times = [item[0] for item in group]
                majority_onsets.append(np.median(group_times))
            
            i = j
        
        return np.array(majority_onsets)