"""
Base mixin for classification tracks to share common functionality.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

from .base_classifier import ClassificationResult
from ..config.constants import DRUM_CLASSES
from ..utils.logging import get_logger

logger = get_logger(__name__)


class BaseTrackMixin:
    """
    Mixin class providing common functionality for classification tracks.
    
    This mixin reduces code duplication by providing shared methods for:
    - Basic feature vector conversion
    - Common classification patterns
    - Error handling and fallback mechanisms
    - Velocity calculations
    - Instrument mapping
    """
    
    def _features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """
        Convert features dictionary to numerical vector.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Feature vector as numpy array
        """
        feature_vector = []
        
        # Standard feature ordering for consistency
        standard_features = [
            'rms_energy', 'spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth',
            'zero_crossing_rate', 'low_energy_ratio', 'mid_energy_ratio', 'high_energy_ratio'
        ]
        
        # Add standard features first
        for key in standard_features:
            if key in features:
                value = features[key]
                if isinstance(value, (int, float)):
                    feature_vector.append(float(value))
                elif isinstance(value, np.ndarray):
                    feature_vector.extend(value.flatten().astype(float))
                else:
                    feature_vector.append(0.0)
            else:
                feature_vector.append(0.0)
        
        # Add remaining features in sorted order
        remaining_keys = sorted([k for k in features.keys() if k not in standard_features])
        for key in remaining_keys:
            value = features[key]
            if isinstance(value, (int, float)):
                feature_vector.append(float(value))
            elif isinstance(value, np.ndarray):
                feature_vector.extend(value.flatten().astype(float))
            else:
                feature_vector.append(0.0)
        
        return np.array(feature_vector)
    
    def _calculate_velocity(self, 
                          audio_window: np.ndarray, 
                          features: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate velocity based on audio energy and features.
        
        Args:
            audio_window: Audio signal
            features: Optional feature dictionary
            
        Returns:
            Velocity value between 0.1 and 1.0
        """
        try:
            if features:
                # Use RMS energy from features if available
                rms_energy = features.get('rms_energy', np.sqrt(np.mean(audio_window**2)))
                spectral_centroid = features.get('spectral_centroid', 1000)
            else:
                # Calculate directly from audio
                rms_energy = np.sqrt(np.mean(audio_window**2))
                spectral_centroid = 1000
            
            # Base velocity from RMS energy
            base_velocity = min(rms_energy * 5, 1.0)
            
            # Adjust based on spectral characteristics
            if spectral_centroid > 3000:  # High frequency instruments (cymbals)
                velocity = base_velocity * 1.1
            elif spectral_centroid < 200:  # Low frequency instruments (kick)
                velocity = base_velocity * 0.9
            else:
                velocity = base_velocity
            
            return float(np.clip(velocity, 0.1, 1.0))
            
        except Exception as e:
            logger.warning(f"Velocity calculation failed: {e}")
            return 0.5
    
    def _class_id_to_instrument(self, class_id: int) -> str:
        """
        Convert class ID to instrument name.
        
        Args:
            class_id: Instrument class ID
            
        Returns:
            Instrument name string
        """
        return DRUM_CLASSES.get(class_id, 'kick')
    
    def _instrument_to_class_id(self, instrument: str) -> int:
        """
        Convert instrument name to class ID.
        
        Args:
            instrument: Instrument name
            
        Returns:
            Class ID integer
        """
        for class_id, instr_name in DRUM_CLASSES.items():
            if instr_name == instrument:
                return class_id
        return 2  # Default to kick
    
    def _rule_based_fallback(self, 
                           audio_window: np.ndarray, 
                           features: Optional[Dict[str, Any]] = None) -> ClassificationResult:
        """
        Rule-based fallback classification.
        
        Args:
            audio_window: Audio signal
            features: Optional feature dictionary
            
        Returns:
            Classification result
        """
        try:
            if features:
                spectral_centroid = features.get('spectral_centroid', 1000)
                rms_energy = features.get('rms_energy', 0.1)
                low_ratio = features.get('low_energy_ratio', 0.33)
                high_ratio = features.get('high_energy_ratio', 0.33)
            else:
                # Simple FFT-based analysis
                fft = np.fft.rfft(audio_window)
                freqs = np.fft.rfftfreq(len(audio_window), 1/22050)  # Assume 22050 Hz
                magnitude = np.abs(fft)
                
                if len(magnitude) > 0:
                    spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
                else:
                    spectral_centroid = 1000
                
                rms_energy = np.sqrt(np.mean(audio_window**2))
                
                # Simple frequency band analysis
                total_energy = np.sum(magnitude)
                if total_energy > 0:
                    low_energy = np.sum(magnitude[freqs <= 500])
                    high_energy = np.sum(magnitude[freqs >= 2000])
                    low_ratio = low_energy / total_energy
                    high_ratio = high_energy / total_energy
                else:
                    low_ratio = 0.33
                    high_ratio = 0.33
            
            # Enhanced rule-based classification
            if low_ratio > 0.6 and spectral_centroid < 150:
                instrument = 'kick'
                confidence = 0.7
            elif 200 < spectral_centroid < 1000 and rms_energy > 0.2:
                instrument = 'snare'
                confidence = 0.65
            elif high_ratio > 0.5 and spectral_centroid > 3000:
                if rms_energy > 0.25:
                    instrument = 'crash'
                    confidence = 0.6
                else:
                    instrument = 'hi-hat-close'
                    confidence = 0.6
            elif 2000 < spectral_centroid < 3000 and high_ratio > 0.3:
                instrument = 'ride-cymbal'
                confidence = 0.55
            elif 300 < spectral_centroid < 600:
                instrument = 'tom-high'
                confidence = 0.5
            elif 200 < spectral_centroid < 400:
                instrument = 'tom-low'
                confidence = 0.5
            else:
                instrument = 'kick'
                confidence = 0.4
            
            velocity = self._calculate_velocity(audio_window, features)
            
            return ClassificationResult(
                instrument=instrument,
                confidence=confidence,
                velocity=velocity,
                features={'source': 'rule_based_fallback'}
            )
            
        except Exception as e:
            logger.error(f"Rule-based fallback failed: {e}")
            return ClassificationResult(
                instrument='kick',
                confidence=0.1,
                velocity=0.5,
                features={'source': 'error_fallback'}
            )
    
    def _validate_audio_window(self, audio_window: np.ndarray, min_length: int = 100) -> bool:
        """
        Validate audio window for processing.
        
        Args:
            audio_window: Audio signal
            min_length: Minimum required length
            
        Returns:
            True if valid, False otherwise
        """
        try:
            if not isinstance(audio_window, np.ndarray):
                return False
            
            if len(audio_window) < min_length:
                return False
            
            if np.all(audio_window == 0):
                return False
            
            if not np.isfinite(audio_window).all():
                return False
            
            return True
            
        except Exception:
            return False
    
    def _safe_feature_extraction(self, 
                                feature_extractor, 
                                audio_window: np.ndarray, 
                                feature_set: str = 'advanced') -> Dict[str, Any]:
        """
        Safe feature extraction with error handling.
        
        Args:
            feature_extractor: FeatureExtractor instance
            audio_window: Audio signal
            feature_set: Feature set to extract
            
        Returns:
            Feature dictionary
        """
        try:
            if not self._validate_audio_window(audio_window):
                logger.warning("Invalid audio window for feature extraction")
                return self._get_default_features()
            
            features = feature_extractor.extract_all_features(audio_window, feature_set)
            
            # Validate extracted features
            if not features:
                logger.warning("No features extracted")
                return self._get_default_features()
            
            # Check for NaN or infinite values
            cleaned_features = {}
            for key, value in features.items():
                if isinstance(value, (int, float)):
                    if np.isfinite(value):
                        cleaned_features[key] = float(value)
                    else:
                        cleaned_features[key] = 0.0
                else:
                    cleaned_features[key] = value
            
            return cleaned_features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return self._get_default_features()
    
    def _get_default_features(self) -> Dict[str, Any]:
        """
        Get default feature values for fallback.
        
        Returns:
            Default feature dictionary
        """
        return {
            'rms_energy': 0.1,
            'spectral_centroid': 1000.0,
            'spectral_rolloff': 2000.0,
            'spectral_bandwidth': 1000.0,
            'zero_crossing_rate': 0.1,
            'low_energy_ratio': 0.33,
            'mid_energy_ratio': 0.33,
            'high_energy_ratio': 0.33,
            'source': 'default_features'
        }
    
    def _log_classification_stats(self, 
                                 results: list, 
                                 track_name: str = "unknown") -> None:
        """
        Log classification statistics.
        
        Args:
            results: List of classification results
            track_name: Name of the track
        """
        try:
            if not results:
                logger.warning(f"{track_name}: No classification results")
                return
            
            # Count instruments
            instrument_counts = {}
            confidence_sum = 0
            
            for result in results:
                instrument = result.instrument
                instrument_counts[instrument] = instrument_counts.get(instrument, 0) + 1
                confidence_sum += result.confidence
            
            avg_confidence = confidence_sum / len(results)
            
            logger.info(f"{track_name}: {len(results)} classifications, "
                       f"avg confidence: {avg_confidence:.3f}")
            logger.debug(f"{track_name}: instrument distribution: {instrument_counts}")
            
        except Exception as e:
            logger.warning(f"Failed to log classification stats: {e}")
    
    def _create_classification_result(self, 
                                    instrument: str, 
                                    confidence: float, 
                                    velocity: float, 
                                    features: Dict[str, Any]) -> ClassificationResult:
        """
        Create a standardized classification result.
        
        Args:
            instrument: Instrument name
            confidence: Classification confidence
            velocity: Velocity value
            features: Additional features
            
        Returns:
            ClassificationResult instance
        """
        # Validate and clamp values
        confidence = float(np.clip(confidence, 0.0, 1.0))
        velocity = float(np.clip(velocity, 0.0, 1.0))
        
        # Ensure minimum velocity
        velocity = max(velocity, 0.1)
        
        return ClassificationResult(
            instrument=instrument,
            confidence=confidence,
            velocity=velocity,
            features=features
        )