"""
Track 5: Multi-Scale Temporal Analysis

Analyzes audio at multiple temporal scales (25ms, 50ms, 100ms, 200ms) to capture
both fast transients and longer decay characteristics.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.ensemble import RandomForestClassifier

from ..base_classifier import BaseClassifier, ClassificationResult
from ..feature_extractor import FeatureExtractor
from ..base_track_mixin import BaseTrackMixin
from ...config.settings import Settings
from ...utils.exceptions import ClassificationError
from ...utils.logging import get_logger

logger = get_logger(__name__)


class MultiScaleTrack(BaseClassifier, BaseTrackMixin):
    """
    Track 5: Multi-Scale Temporal Analysis
    
    Analyzes audio at multiple temporal scales to capture instrument-specific
    temporal characteristics from fast transients to long decay patterns.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize multi-scale track classifier.
        
        Args:
            settings: Application settings
        """
        super().__init__(settings)
        self.feature_extractor = FeatureExtractor(settings)
        
        # Define temporal scales in milliseconds
        self.scales = [25, 50, 100, 200]
        self.scale_samples = [int(scale * settings.audio.sample_rate / 1000) for scale in self.scales]
        
        # Scale-specific classifiers
        self.scale_classifiers = {}
        
        # Scale weights for each instrument class [25ms, 50ms, 100ms, 200ms]
        self.instrument_scale_weights = {
            0: [0.4, 0.3, 0.2, 0.1],  # Hi-hat Close - favors short scales
            1: [0.2, 0.4, 0.3, 0.1],  # Snare - balanced towards medium scales
            2: [0.1, 0.2, 0.3, 0.4],  # Kick - favors longer scales
            3: [0.2, 0.3, 0.3, 0.2],  # Tom High - balanced
            4: [0.1, 0.2, 0.4, 0.3],  # Tom Low - favors longer scales
            5: [0.1, 0.1, 0.3, 0.5],  # Ride - favors longest scales
            6: [0.1, 0.2, 0.3, 0.4],  # Tom Floor - favors longer scales
            7: [0.3, 0.3, 0.2, 0.2],  # Hi-hat Open - favors shorter scales
            8: [0.1, 0.2, 0.3, 0.4],  # Ride Bell - favors longer scales
            9: [0.1, 0.1, 0.2, 0.6],  # Crash - favors longest scales
        }
        
        # Initialize scale-specific classifiers
        for scale in self.scales:
            self.scale_classifiers[scale] = RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
            
    def initialize(self) -> None:
        """Initialize the multi-scale classifier."""
        try:
            logger.info("⏰ Initializing Track 5: Multi-Scale Temporal Analysis")
            
            # Initialize feature extractor
            self.feature_extractor.clear_cache()
            
            self.is_initialized = True
            logger.info("✅ Multi-scale track initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize multi-scale track: {e}")
            raise ClassificationError(f"Multi-scale track initialization failed: {e}")
    
    def extract_multi_scale_windows(self, audio: np.ndarray, onset_time: float) -> Dict[int, np.ndarray]:
        """
        Extract audio windows at multiple temporal scales.
        
        Args:
            audio: Full audio signal
            onset_time: Time of the onset in seconds
            
        Returns:
            Dictionary mapping scales to audio windows
        """
        windows = {}
        onset_sample = int(onset_time * self.settings.audio.sample_rate)
        
        for i, scale in enumerate(self.scales):
            window_samples = self.scale_samples[i]
            
            # Extract window centered on onset
            start_sample = onset_sample - window_samples // 2
            end_sample = start_sample + window_samples
            
            # Ensure we don't go out of bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            if end_sample - start_sample >= window_samples // 2:
                window = audio[start_sample:end_sample]
                
                # Pad if necessary
                if len(window) < window_samples:
                    padding = window_samples - len(window)
                    window = np.pad(window, (0, padding), mode='constant')
                
                windows[scale] = window
            else:
                # Create zero-padded window if too short
                windows[scale] = np.zeros(window_samples)
        
        return windows
    
    def extract_scale_specific_features(self, windows: Dict[int, np.ndarray]) -> Dict[int, np.ndarray]:
        """
        Extract features for each temporal scale.
        
        Args:
            windows: Dictionary mapping scales to audio windows
            
        Returns:
            Dictionary mapping scales to feature vectors
        """
        scale_features = {}
        
        for scale, window in windows.items():
            try:
                # Extract basic features
                basic_features = self.feature_extractor.extract_basic_features(window)
                
                # Extract frequency band features
                band_features = self.feature_extractor.extract_frequency_band_features(window)
                
                # Extract scale-specific features
                scale_specific = self._extract_scale_specific_features(window, scale)
                
                # Combine all features
                combined_features = {**basic_features, **band_features, **scale_specific}
                
                # Convert to vector
                feature_vector = self.feature_extractor.get_feature_vector(window, 'basic')
                
                scale_features[scale] = feature_vector
                
            except Exception as e:
                logger.warning(f"Error extracting features for scale {scale}ms: {e}")
                scale_features[scale] = np.zeros(20)  # Default feature vector size
        
        return scale_features
    
    def _extract_scale_specific_features(self, window: np.ndarray, scale: int) -> Dict[str, float]:
        """
        Extract features specific to this temporal scale.
        
        Args:
            window: Audio window
            scale: Temporal scale in milliseconds
            
        Returns:
            Dictionary of scale-specific features
        """
        try:
            features = {}
            
            # Scale-specific energy features
            energy = np.sum(window ** 2) / len(window)
            features['scale_energy'] = float(energy)
            
            # Scale-specific onset characteristics
            peak_position = np.argmax(np.abs(window)) / len(window)
            features['scale_peak_position'] = float(peak_position)
            
            # Scale-specific spectral features
            fft = np.fft.rfft(window)
            magnitude = np.abs(fft)
            freqs = np.fft.rfftfreq(len(window), 1/self.settings.audio.sample_rate)
            
            # Frequency centroid weighted by scale
            if np.sum(magnitude) > 0:
                freq_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
                features['scale_freq_centroid'] = float(freq_centroid)
            else:
                features['scale_freq_centroid'] = 0.0
            
            # Scale-specific temporal decay
            decay_time = self._calculate_decay_time(window, scale)
            features['scale_decay_time'] = float(decay_time)
            
            # Scale identifier
            features['scale'] = float(scale)
            
            return features
            
        except Exception as e:
            logger.warning(f"Error extracting scale-specific features: {e}")
            return {'scale_energy': 0.0, 'scale_peak_position': 0.5, 
                   'scale_freq_centroid': 1000.0, 'scale_decay_time': 0.0, 'scale': float(scale)}
    
    def _calculate_decay_time(self, window: np.ndarray, scale: int) -> float:
        """Calculate decay time normalized by scale."""
        try:
            envelope = np.abs(window)
            peak_idx = np.argmax(envelope)
            
            if peak_idx < len(envelope) - 1:
                post_peak = envelope[peak_idx:]
                peak_val = envelope[peak_idx]
                
                # Find 10% decay point
                threshold = peak_val * 0.1
                decay_indices = np.where(post_peak < threshold)[0]
                
                if len(decay_indices) > 0:
                    decay_samples = decay_indices[0]
                    decay_time = decay_samples / self.settings.audio.sample_rate
                    # Normalize by scale
                    return decay_time / (scale / 1000.0)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def classify_at_scale(self, features: np.ndarray, scale: int) -> tuple:
        """
        Classify instrument at specific temporal scale.
        
        Args:
            features: Feature vector
            scale: Temporal scale in milliseconds
            
        Returns:
            Tuple of (prediction, confidence)
        """
        try:
            # Scale-specific classification logic
            if scale <= 25:
                # Very short scale - good for transients
                prediction, confidence = self._classify_transient_focused(features)
            elif scale <= 50:
                # Short scale - balanced
                prediction, confidence = self._classify_balanced(features)
            elif scale <= 100:
                # Medium scale - good for body
                prediction, confidence = self._classify_body_focused(features)
            else:
                # Long scale - good for decay
                prediction, confidence = self._classify_decay_focused(features)
            
            return prediction, confidence
            
        except Exception as e:
            logger.warning(f"Classification at scale {scale}ms failed: {e}")
            return 2, 0.3  # Default to kick
    
    def _classify_transient_focused(self, features: np.ndarray) -> tuple:
        """Classification focused on transient characteristics."""
        try:
            # Extract key features for transients
            if len(features) > 5:
                spectral_centroid = features[1]  # From basic features
                rms_energy = features[0]
                zero_crossing_rate = features[2] if len(features) > 2 else 0.05
            else:
                spectral_centroid = 2000
                rms_energy = 0.3
                zero_crossing_rate = 0.05
            
            # Normalize features
            centroid_norm = spectral_centroid / 10000.0
            
            # Transient-based classification
            if centroid_norm > 0.6 and zero_crossing_rate > 0.08:
                return 0, 0.75  # Hi-hat close
            elif centroid_norm > 0.4 and zero_crossing_rate > 0.06:
                return 7, 0.7   # Hi-hat open
            elif centroid_norm > 0.15 and rms_energy > 0.2:
                return 1, 0.72  # Snare
            elif centroid_norm < 0.1:
                return 2, 0.8   # Kick
            elif centroid_norm > 0.3:
                return 9, 0.65  # Crash
            else:
                return 3, 0.6   # Tom high
                
        except Exception:
            return 2, 0.3
    
    def _classify_balanced(self, features: np.ndarray) -> tuple:
        """Balanced classification for medium-short scales."""
        try:
            if len(features) > 3:
                spectral_centroid = features[1]
                spectral_rolloff = features[2] if len(features) > 2 else spectral_centroid * 2
                rms_energy = features[0]
            else:
                spectral_centroid = 2000
                spectral_rolloff = 4000
                rms_energy = 0.3
            
            # Balanced classification
            if spectral_centroid > 5000:
                return (0, 0.7) if spectral_rolloff > 7000 else (7, 0.68)
            elif spectral_centroid > 2500:
                return 1, 0.75  # Snare
            elif spectral_centroid > 1200:
                return 3, 0.65  # Tom high
            elif spectral_centroid > 600:
                return 4, 0.6   # Tom low
            elif spectral_centroid > 300:
                return 6, 0.62  # Tom floor
            else:
                return 2, 0.8   # Kick
                
        except Exception:
            return 2, 0.3
    
    def _classify_body_focused(self, features: np.ndarray) -> tuple:
        """Classification focused on instrument body characteristics."""
        try:
            if len(features) > 3:
                spectral_centroid = features[1]
                spectral_bandwidth = features[2] if len(features) > 2 else 1000
                rms_energy = features[0]
            else:
                spectral_centroid = 2000
                spectral_bandwidth = 1000
                rms_energy = 0.3
            
            # Body-focused classification
            if spectral_bandwidth > 1500 and spectral_centroid > 3000:
                return 5, 0.7   # Ride
            elif spectral_bandwidth > 800 and spectral_centroid > 2000:
                return 1, 0.75  # Snare
            elif spectral_bandwidth > 500 and spectral_centroid > 1000:
                return 3, 0.65  # Tom high
            elif spectral_bandwidth > 300 and spectral_centroid > 500:
                return 4, 0.6   # Tom low
            elif spectral_bandwidth < 200:
                return 2, 0.8   # Kick
            elif spectral_centroid > 4000:
                return 9, 0.68  # Crash
            else:
                return 6, 0.62  # Tom floor
                
        except Exception:
            return 2, 0.3
    
    def _classify_decay_focused(self, features: np.ndarray) -> tuple:
        """Classification focused on decay characteristics."""
        try:
            if len(features) > 3:
                spectral_centroid = features[1]
                spectral_rolloff = features[2] if len(features) > 2 else spectral_centroid * 2
                rms_energy = features[0]
            else:
                spectral_centroid = 2000
                spectral_rolloff = 4000
                rms_energy = 0.3
            
            # Decay-focused classification
            if spectral_rolloff > 8000 and spectral_centroid > 4000:
                return 9, 0.8   # Crash (long decay)
            elif spectral_rolloff > 6000 and spectral_centroid > 3000:
                return 5, 0.75  # Ride (sustained)
            elif spectral_rolloff > 4000 and spectral_centroid > 2000:
                return 8, 0.7   # Ride bell
            elif spectral_rolloff < 1000 and spectral_centroid < 300:
                return 2, 0.85  # Kick (short decay)
            elif spectral_rolloff < 2000 and spectral_centroid < 800:
                return 6, 0.7   # Tom floor
            elif spectral_rolloff < 3000 and spectral_centroid < 1200:
                return 4, 0.65  # Tom low
            else:
                return 1, 0.6   # Snare
                
        except Exception:
            return 2, 0.3
    
    def classify_onset(self, 
                      audio_window: np.ndarray, 
                      onset_time: float,
                      context: Optional[Dict[str, Any]] = None) -> ClassificationResult:
        """
        Classify a drum onset using multi-scale analysis.
        
        Args:
            audio_window: Audio data around the onset
            onset_time: Time of the onset in seconds
            context: Additional context information (should contain full audio)
            
        Returns:
            Classification result
        """
        try:
            # We need the full audio for multi-scale analysis
            if context and 'full_audio' in context:
                full_audio = context['full_audio']
            else:
                # Fallback to using just the window
                full_audio = audio_window
                onset_time = len(audio_window) / (2 * self.settings.audio.sample_rate)
            
            # Extract multi-scale windows
            windows = self.extract_multi_scale_windows(full_audio, onset_time)
            
            # Extract features for each scale
            scale_features = self.extract_scale_specific_features(windows)
            
            # Classify at each scale
            scale_predictions = {}
            scale_confidences = {}
            
            for scale in self.scales:
                if scale in scale_features:
                    prediction, confidence = self.classify_at_scale(scale_features[scale], scale)
                    scale_predictions[scale] = prediction
                    scale_confidences[scale] = confidence
                else:
                    scale_predictions[scale] = 2  # Default to kick
                    scale_confidences[scale] = 0.3
            
            # Combine predictions using learned weights
            final_prediction, final_confidence = self._combine_scale_predictions(
                scale_predictions, scale_confidences
            )
            
            # Map to instrument name
            instrument = self._class_id_to_instrument(final_prediction)
            
            # Calculate velocity
            velocity = self._calculate_velocity(audio_window)
            
            return ClassificationResult(
                instrument=instrument,
                confidence=final_confidence,
                velocity=velocity,
                features={
                    'source': 'multi_scale',
                    'scale_predictions': scale_predictions,
                    'scale_confidences': scale_confidences,
                    'scales_used': self.scales
                }
            )
            
        except Exception as e:
            logger.error(f"Multi-scale classification failed at {onset_time:.3f}s: {e}")
            return self._fallback_classification(audio_window)
    
    def _combine_scale_predictions(self, 
                                 scale_predictions: Dict[int, int], 
                                 scale_confidences: Dict[int, float]) -> tuple:
        """
        Combine predictions from multiple scales using learned weights.
        
        Args:
            scale_predictions: Predictions for each scale
            scale_confidences: Confidences for each scale
            
        Returns:
            Tuple of (final_prediction, final_confidence)
        """
        # Count votes for each instrument, weighted by scale and confidence
        instrument_votes = {i: 0.0 for i in range(10)}
        
        for scale, prediction in scale_predictions.items():
            confidence = scale_confidences[scale]
            
            # Get scale weight for this instrument
            scale_idx = self.scales.index(scale)
            scale_weight = self.instrument_scale_weights[prediction][scale_idx]
            
            # Add weighted vote
            vote_strength = confidence * scale_weight
            instrument_votes[prediction] += vote_strength
        
        # Find instrument with highest vote
        final_prediction = max(instrument_votes, key=instrument_votes.get)
        final_confidence = instrument_votes[final_prediction]
        
        # Normalize confidence
        total_votes = sum(instrument_votes.values())
        if total_votes > 0:
            final_confidence = final_confidence / total_votes
        else:
            final_confidence = 0.3
        
        return final_prediction, min(final_confidence, 1.0)
    
    def _calculate_velocity(self, audio_window: np.ndarray) -> float:
        """Calculate velocity based on audio energy."""
        try:
            rms = np.sqrt(np.mean(audio_window**2))
            return float(np.clip(rms * 10, 0.1, 1.0))
        except Exception:
            return 0.5
    
    def _class_id_to_instrument(self, class_id: int) -> str:
        """Convert class ID to instrument name."""
        from ...config.constants import DRUM_CLASSES
        return DRUM_CLASSES.get(class_id, 'kick')
    
    def _fallback_classification(self, audio_window: np.ndarray) -> ClassificationResult:
        """Simple fallback classification."""
        try:
            # Simple frequency analysis
            fft = np.fft.rfft(audio_window)
            freqs = np.fft.rfftfreq(len(audio_window), 1/self.settings.audio.sample_rate)
            magnitude = np.abs(fft)
            
            peak_idx = np.argmax(magnitude)
            peak_freq = freqs[peak_idx]
            
            # Simple classification
            if peak_freq < 100:
                instrument = 'kick'
                confidence = 0.6
            elif peak_freq < 300:
                instrument = 'snare'
                confidence = 0.5
            elif peak_freq > 3000:
                instrument = 'hi-hat-close'
                confidence = 0.5
            else:
                instrument = 'tom-high'
                confidence = 0.4
            
            velocity = self._calculate_velocity(audio_window)
            
            return ClassificationResult(
                instrument=instrument,
                confidence=confidence,
                velocity=velocity,
                features={'source': 'fallback'}
            )
            
        except Exception:
            return ClassificationResult(
                instrument='kick',
                confidence=0.1,
                velocity=0.5,
                features={'source': 'error_fallback'}
            )
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this track."""
        info = super().get_info()
        info.update({
            'track_number': 5,
            'track_name': 'Multi-Scale Temporal Analysis',
            'description': 'Analyzes audio at multiple temporal scales for comprehensive classification',
            'scales': self.scales,
            'scale_samples': self.scale_samples,
            'instrument_scale_weights': self.instrument_scale_weights
        })
        return info
    
    def cleanup(self):
        """Clean up resources."""
        super().cleanup()
        if hasattr(self.feature_extractor, 'clear_cache'):
            self.feature_extractor.clear_cache()
        
        # Clear classifiers
        for classifier in self.scale_classifiers.values():
            del classifier
        self.scale_classifiers.clear()