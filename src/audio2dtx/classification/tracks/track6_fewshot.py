"""
Track 6: Real-Time Few-Shot Learning

Adapts the classification model to specific song characteristics during processing
using few-shot learning techniques.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict

from ..base_classifier import BaseClassifier, ClassificationResult
from ..feature_extractor import FeatureExtractor
from ...config.settings import Settings
from ...utils.exceptions import ClassificationError
from ...utils.logging import get_logger

logger = get_logger(__name__)


class FewShotTrack(BaseClassifier):
    """
    Track 6: Real-Time Few-Shot Learning
    
    Adapts classification model to specific song characteristics during processing
    using few-shot learning techniques and progressive model improvement.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize few-shot learning track classifier.
        
        Args:
            settings: Application settings
        """
        super().__init__(settings)
        self.feature_extractor = FeatureExtractor(settings)
        
        # Few-shot learning parameters
        self.confidence_threshold = 0.6
        self.adaptation_rate = 0.1
        self.min_samples_for_adaptation = 3
        
        # Instrument-specific adaptation parameters
        self.instrument_profiles = {}
        self.adaptation_history = []
        self.song_characteristics = {}
        
        # Song-specific learned patterns
        self.song_tempo = None
        self.global_characteristics = {}
        self.instrument_signatures = {}
        
        # Phase tracking
        self.phase = 'initialization'  # 'initialization', 'learning', 'adaptation'
        self.processed_onsets = 0
        
    def initialize(self) -> None:
        """Initialize the few-shot learning classifier."""
        try:
            logger.info("🚀 Initializing Track 6: Real-Time Few-Shot Learning")
            
            # Initialize feature extractor
            self.feature_extractor.clear_cache()
            
            # Initialize instrument profiles
            self._initialize_instrument_profiles()
            
            self.is_initialized = True
            logger.info("✅ Few-shot learning track initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize few-shot learning track: {e}")
            raise ClassificationError(f"Few-shot learning track initialization failed: {e}")
    
    def set_song_context(self, 
                        full_audio: np.ndarray,
                        beat_times: List[float],
                        tempo_bpm: float):
        """
        Set song context for few-shot learning.
        
        Args:
            full_audio: Full audio signal
            beat_times: Beat times in seconds
            tempo_bpm: Song tempo in BPM
        """
        try:
            logger.info("🎵 Initializing song-specific profile for few-shot learning")
            
            self.song_tempo = tempo_bpm
            
            # Analyze global song characteristics
            self._analyze_global_characteristics(full_audio)
            
            # Initialize instrument profiles
            self._initialize_instrument_profiles()
            
            logger.info(f"✅ Song profile initialized: Tempo={tempo_bpm:.1f} BPM")
            
        except Exception as e:
            logger.error(f"Failed to set song context: {e}")
    
    def _analyze_global_characteristics(self, audio: np.ndarray):
        """Analyze global song characteristics."""
        try:
            # Convert to mono if stereo
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            
            # Extract global spectral characteristics
            import librosa
            
            # Global spectral centroid
            global_centroid = np.mean(librosa.feature.spectral_centroid(
                y=audio, sr=self.settings.audio.sample_rate
            ))
            
            # Global RMS energy
            global_rms = np.mean(librosa.feature.rms(y=audio))
            
            # Global spectral rolloff
            global_rolloff = np.mean(librosa.feature.spectral_rolloff(
                y=audio, sr=self.settings.audio.sample_rate
            ))
            
            # Store global characteristics
            self.global_characteristics = {
                'global_centroid': float(global_centroid),
                'global_rms': float(global_rms),
                'global_rolloff': float(global_rolloff),
                'global_tempo': self.song_tempo or 120.0
            }
            
            logger.debug(f"Global characteristics: {self.global_characteristics}")
            
        except Exception as e:
            logger.warning(f"Error in global analysis: {e}")
            self.global_characteristics = {
                'global_centroid': 2000.0,
                'global_rms': 0.1,
                'global_rolloff': 4000.0,
                'global_tempo': 120.0
            }
    
    def _initialize_instrument_profiles(self):
        """Initialize instrument-specific profiles for adaptation."""
        for instrument_id in range(10):
            self.instrument_profiles[instrument_id] = {
                'feature_means': None,
                'feature_stds': None,
                'confidence_scores': [],
                'adaptation_count': 0,
                'characteristic_features': {},
                'samples': []
            }
    
    def classify_onsets(self, 
                       onsets: List[tuple],
                       context: Optional[Dict[str, Any]] = None) -> List[ClassificationResult]:
        """
        Classify multiple onsets with progressive few-shot learning.
        
        Args:
            onsets: List of (audio_window, onset_time) tuples
            context: Additional context information
            
        Returns:
            List of classification results
        """
        if not self.is_initialized:
            self.initialize()
        
        # Set song context if provided
        if context:
            if 'full_audio' in context:
                self.set_song_context(
                    context['full_audio'],
                    context.get('beat_times', []),
                    context.get('tempo_bpm', 120.0)
                )
        
        results = []
        
        # Phase 1: Initial profiling (first 20% of onsets)
        phase1_count = max(5, len(onsets) // 5)
        
        for i, (audio_window, onset_time) in enumerate(onsets):
            self.processed_onsets = i + 1
            
            # Determine current phase
            if i < phase1_count:
                self.phase = 'initialization'
            elif i < phase1_count * 2:
                self.phase = 'learning'
            else:
                self.phase = 'adaptation'
            
            # Classify onset
            result = self.classify_onset(audio_window, onset_time, context)
            results.append(result)
            
            # Learn from confident predictions
            if result.confidence > self.confidence_threshold:
                self._learn_from_prediction(audio_window, result, onset_time)
        
        # Log learning statistics
        self._log_learning_statistics()
        
        return results
    
    def classify_onset(self, 
                      audio_window: np.ndarray, 
                      onset_time: float,
                      context: Optional[Dict[str, Any]] = None) -> ClassificationResult:
        """
        Classify a drum onset using few-shot learning.
        
        Args:
            audio_window: Audio data around the onset
            onset_time: Time of the onset in seconds
            context: Additional context information
            
        Returns:
            Classification result
        """
        try:
            # Validate audio window
            if len(audio_window) < 100:
                logger.warning(f"Audio window too short at {onset_time:.3f}s")
                return self._fallback_classification(audio_window)
            
            # Extract features
            features = self.feature_extractor.extract_all_features(
                audio_window, 
                feature_set='advanced'
            )
            
            # Convert to feature vector
            feature_vector = self._features_to_vector(features)
            
            # Get initial prediction using rule-based classification
            initial_prediction, initial_confidence = self._initial_classification(
                feature_vector, features
            )
            
            # Apply few-shot learning adaptation
            if self.phase in ['learning', 'adaptation']:
                adapted_prediction, adapted_confidence = self._adapt_classification(
                    feature_vector, features, initial_prediction, initial_confidence
                )
            else:
                adapted_prediction = initial_prediction
                adapted_confidence = initial_confidence
            
            # Map to instrument name
            instrument = self._class_id_to_instrument(adapted_prediction)
            
            # Calculate velocity
            velocity = self._calculate_velocity(audio_window, features)
            
            return ClassificationResult(
                instrument=instrument,
                confidence=adapted_confidence,
                velocity=velocity,
                features={
                    'source': 'few_shot',
                    'phase': self.phase,
                    'initial_prediction': initial_prediction,
                    'initial_confidence': initial_confidence,
                    'adaptation_applied': adapted_prediction != initial_prediction,
                    'processed_onsets': self.processed_onsets
                }
            )
            
        except Exception as e:
            logger.error(f"Few-shot classification failed at {onset_time:.3f}s: {e}")
            return self._fallback_classification(audio_window)
    
    def _features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert features dictionary to vector."""
        feature_vector = []
        
        # Get key features in consistent order
        key_features = [
            'rms_energy', 'spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth',
            'zero_crossing_rate', 'low_energy_ratio', 'mid_energy_ratio', 'high_energy_ratio'
        ]
        
        for key in key_features:
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
        
        # Add statistical features
        for stat_key in ['mean', 'std', 'skewness', 'kurtosis']:
            if stat_key in features:
                feature_vector.append(float(features[stat_key]))
            else:
                feature_vector.append(0.0)
        
        return np.array(feature_vector)
    
    def _initial_classification(self, 
                              feature_vector: np.ndarray,
                              features: Dict[str, Any]) -> Tuple[int, float]:
        """Get initial classification using rule-based approach."""
        try:
            # Extract key features
            spectral_centroid = features.get('spectral_centroid', 1000)
            rms_energy = features.get('rms_energy', 0.1)
            zero_crossing_rate = features.get('zero_crossing_rate', 0.1)
            
            # Frequency band ratios
            low_ratio = features.get('low_energy_ratio', 0.33)
            mid_ratio = features.get('mid_energy_ratio', 0.33)
            high_ratio = features.get('high_energy_ratio', 0.33)
            
            # Rule-based classification
            if low_ratio > 0.6 and spectral_centroid < 150:
                return 2, 0.7  # Kick
            elif mid_ratio > 0.4 and 200 < spectral_centroid < 1000:
                return 1, 0.65  # Snare
            elif high_ratio > 0.5 and spectral_centroid > 2000:
                return 0, 0.6  # Hi-hat close
            elif spectral_centroid > 4000 and high_ratio > 0.4:
                return 9, 0.6  # Crash
            elif 300 < spectral_centroid < 800:
                return 3, 0.5  # Tom high
            elif spectral_centroid > 1500:
                return 5, 0.55  # Ride
            else:
                return 2, 0.4  # Default to kick
                
        except Exception as e:
            logger.warning(f"Initial classification failed: {e}")
            return 2, 0.3
    
    def _adapt_classification(self, 
                            feature_vector: np.ndarray,
                            features: Dict[str, Any],
                            initial_prediction: int,
                            initial_confidence: float) -> Tuple[int, float]:
        """Adapt classification based on learned patterns."""
        try:
            # If confidence is already high, minimal adaptation
            if initial_confidence > 0.85:
                return initial_prediction, initial_confidence
            
            # Calculate similarities to learned instrument profiles
            instrument_similarities = {}
            
            for instrument_id, profile in self.instrument_profiles.items():
                if profile['adaptation_count'] >= self.min_samples_for_adaptation:
                    similarity = self._calculate_similarity(feature_vector, profile)
                    instrument_similarities[instrument_id] = similarity
            
            # Find best matching instrument
            if instrument_similarities:
                best_instrument = max(instrument_similarities.keys(), 
                                    key=lambda x: instrument_similarities[x])
                best_similarity = instrument_similarities[best_instrument]
                
                # Adaptive threshold based on song characteristics
                similarity_threshold = self._calculate_adaptive_threshold(features)
                
                if best_similarity > similarity_threshold:
                    # Boost confidence based on similarity
                    adapted_confidence = min(
                        initial_confidence + best_similarity * 0.3, 
                        0.9
                    )
                    
                    # Only change prediction if similarity is significantly higher
                    if best_similarity > similarity_threshold * 1.5:
                        return best_instrument, adapted_confidence
                    else:
                        return initial_prediction, adapted_confidence
            
            return initial_prediction, initial_confidence
            
        except Exception as e:
            logger.warning(f"Adaptation failed: {e}")
            return initial_prediction, initial_confidence
    
    def _calculate_similarity(self, feature_vector: np.ndarray, profile: Dict[str, Any]) -> float:
        """Calculate similarity between feature vector and instrument profile."""
        try:
            if profile['feature_means'] is None:
                return 0.0
            
            mean_features = profile['feature_means']
            std_features = profile['feature_stds']
            
            # Handle size mismatch
            min_size = min(len(feature_vector), len(mean_features))
            if min_size == 0:
                return 0.0
            
            feature_vector = feature_vector[:min_size]
            mean_features = mean_features[:min_size]
            std_features = std_features[:min_size] if std_features is not None else np.ones(min_size)
            
            # Normalized distance
            normalized_diff = (feature_vector - mean_features) / (std_features + 1e-6)
            similarity = np.exp(-np.mean(normalized_diff ** 2) / 2)
            
            # Weight by adaptation confidence
            if profile['confidence_scores']:
                avg_confidence = np.mean(profile['confidence_scores'][-5:])  # Last 5 predictions
                weighted_similarity = similarity * avg_confidence
            else:
                weighted_similarity = similarity
            
            return float(weighted_similarity)
            
        except Exception as e:
            logger.warning(f"Similarity calculation failed: {e}")
            return 0.0
    
    def _calculate_adaptive_threshold(self, features: Dict[str, Any]) -> float:
        """Calculate adaptive threshold based on song characteristics."""
        try:
            base_threshold = 0.3
            
            # Adjust based on global characteristics
            if self.global_characteristics:
                global_centroid = self.global_characteristics.get('global_centroid', 2000)
                global_rms = self.global_characteristics.get('global_rms', 0.1)
                
                # Adjust threshold based on song complexity
                if global_centroid > 3000:  # High frequency content
                    base_threshold *= 0.9
                elif global_centroid < 1000:  # Low frequency content
                    base_threshold *= 1.1
                
                if global_rms > 0.3:  # High energy song
                    base_threshold *= 0.95
                elif global_rms < 0.05:  # Low energy song
                    base_threshold *= 1.05
            
            # Adjust based on processing phase
            if self.phase == 'learning':
                base_threshold *= 0.8  # More aggressive in learning phase
            elif self.phase == 'adaptation':
                base_threshold *= 0.9  # Moderate in adaptation phase
            
            return base_threshold
            
        except Exception:
            return 0.3
    
    def _learn_from_prediction(self, 
                             audio_window: np.ndarray,
                             result: ClassificationResult,
                             onset_time: float):
        """Learn from confident prediction to improve future classifications."""
        try:
            if result.confidence < self.confidence_threshold:
                return
            
            # Get instrument ID from result
            instrument_id = self._instrument_to_class_id(result.instrument)
            
            # Extract features
            features = self.feature_extractor.extract_all_features(
                audio_window, 
                feature_set='advanced'
            )
            feature_vector = self._features_to_vector(features)
            
            # Update instrument profile
            profile = self.instrument_profiles[instrument_id]
            
            # Initialize or update feature means and stds
            if profile['feature_means'] is None:
                profile['feature_means'] = feature_vector.copy()
                profile['feature_stds'] = np.ones_like(feature_vector)
            else:
                # Exponential moving average
                alpha = self.adaptation_rate * result.confidence
                
                # Handle size mismatch
                min_size = min(len(feature_vector), len(profile['feature_means']))
                if min_size > 0:
                    feature_vector = feature_vector[:min_size]
                    old_means = profile['feature_means'][:min_size]
                    old_stds = profile['feature_stds'][:min_size]
                    
                    # Update means
                    new_means = (1 - alpha) * old_means + alpha * feature_vector
                    
                    # Update standard deviations
                    feature_diff = feature_vector - old_means
                    new_stds = (1 - alpha) * old_stds + alpha * np.abs(feature_diff)
                    
                    profile['feature_means'] = new_means
                    profile['feature_stds'] = new_stds
            
            # Track confidence and adaptation
            profile['confidence_scores'].append(result.confidence)
            profile['adaptation_count'] += 1
            
            # Keep only recent samples
            if len(profile['confidence_scores']) > 20:
                profile['confidence_scores'] = profile['confidence_scores'][-20:]
            
            # Store in adaptation history
            self.adaptation_history.append({
                'instrument_id': instrument_id,
                'confidence': result.confidence,
                'onset_time': onset_time,
                'phase': self.phase,
                'processed_onsets': self.processed_onsets
            })
            
        except Exception as e:
            logger.warning(f"Learning from prediction failed: {e}")
    
    def _calculate_velocity(self, audio_window: np.ndarray, features: Dict[str, Any]) -> float:
        """Calculate velocity based on audio energy."""
        try:
            rms_energy = features.get('rms_energy', np.sqrt(np.mean(audio_window**2)))
            peak_amplitude = np.max(np.abs(audio_window))
            
            # Combine RMS and peak for velocity estimation
            velocity = (rms_energy * 8 + peak_amplitude * 2) / 10
            return float(np.clip(velocity, 0.1, 1.0))
            
        except Exception:
            return 0.5
    
    def _class_id_to_instrument(self, class_id: int) -> str:
        """Convert class ID to instrument name."""
        from ...config.constants import DRUM_CLASSES
        return DRUM_CLASSES.get(class_id, 'kick')
    
    def _instrument_to_class_id(self, instrument: str) -> int:
        """Convert instrument name to class ID."""
        from ...config.constants import DRUM_CLASSES
        for class_id, instr_name in DRUM_CLASSES.items():
            if instr_name == instrument:
                return class_id
        return 2  # Default to kick
    
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
                confidence = 0.5
            elif peak_freq < 300:
                instrument = 'snare'
                confidence = 0.4
            elif peak_freq > 3000:
                instrument = 'hi-hat-close'
                confidence = 0.4
            else:
                instrument = 'tom-high'
                confidence = 0.3
            
            velocity = np.sqrt(np.mean(audio_window**2)) * 10
            velocity = float(np.clip(velocity, 0.1, 1.0))
            
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
    
    def _log_learning_statistics(self):
        """Log learning statistics."""
        try:
            total_adaptations = sum(
                profile['adaptation_count'] 
                for profile in self.instrument_profiles.values()
            )
            
            if total_adaptations > 0:
                logger.info(f"🧠 Few-shot learning statistics:")
                logger.info(f"  Total adaptations: {total_adaptations}")
                logger.info(f"  Adaptation phases: {len(set(h['phase'] for h in self.adaptation_history))}")
                
                # Log per-instrument statistics
                for instrument_id, profile in self.instrument_profiles.items():
                    if profile['adaptation_count'] > 0:
                        instrument_name = self._class_id_to_instrument(instrument_id)
                        avg_confidence = np.mean(profile['confidence_scores'])
                        logger.info(f"  {instrument_name}: {profile['adaptation_count']} adaptations, "
                                   f"{avg_confidence:.3f} avg confidence")
            else:
                logger.info("🧠 No confident predictions found for adaptation")
                
        except Exception as e:
            logger.warning(f"Failed to log learning statistics: {e}")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get detailed learning statistics."""
        stats = {}
        
        for instrument_id, profile in self.instrument_profiles.items():
            if profile['adaptation_count'] > 0:
                instrument_name = self._class_id_to_instrument(instrument_id)
                stats[instrument_id] = {
                    'instrument_name': instrument_name,
                    'adaptations': profile['adaptation_count'],
                    'avg_confidence': float(np.mean(profile['confidence_scores'])),
                    'stability': float(1.0 - np.std(profile['confidence_scores']) / np.mean(profile['confidence_scores']))
                        if len(profile['confidence_scores']) > 1 else 1.0
                }
        
        return stats
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this track."""
        info = super().get_info()
        info.update({
            'track_number': 6,
            'track_name': 'Real-Time Few-Shot Learning',
            'description': 'Adapts classification to song-specific characteristics during processing',
            'confidence_threshold': self.confidence_threshold,
            'adaptation_rate': self.adaptation_rate,
            'min_samples_for_adaptation': self.min_samples_for_adaptation,
            'current_phase': self.phase,
            'processed_onsets': self.processed_onsets,
            'total_adaptations': sum(p['adaptation_count'] for p in self.instrument_profiles.values()),
            'song_tempo': self.song_tempo
        })
        return info
    
    def cleanup(self):
        """Clean up resources."""
        super().cleanup()
        if hasattr(self.feature_extractor, 'clear_cache'):
            self.feature_extractor.clear_cache()
        
        # Clear learning data
        self.instrument_profiles.clear()
        self.adaptation_history.clear()
        self.global_characteristics.clear()