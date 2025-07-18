"""
Track 4: Advanced Spectral Features + Context

Enhanced frequency-based classification using 139 advanced spectral features
with contextual analysis and Random Forest classifier.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from ..base_classifier import BaseClassifier, ClassificationResult
from ..feature_extractor import FeatureExtractor
from ..base_track_mixin import BaseTrackMixin
from ...config.settings import Settings
from ...utils.exceptions import ClassificationError
from ...utils.logging import get_logger

logger = get_logger(__name__)


class AdvancedFeaturesTrack(BaseClassifier, BaseTrackMixin):
    """
    Track 4: Advanced Spectral Features + Context Classification
    
    Uses 139 advanced spectral features with Random Forest classifier
    and contextual pattern analysis for improved accuracy.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize advanced features track classifier.
        
        Args:
            settings: Application settings
        """
        super().__init__(settings)
        self.feature_extractor = FeatureExtractor(settings)
        self.rf_classifier = None
        self.is_trained = False
        self.beat_times = []
        self.min_training_samples = 10
        self.confidence_threshold = 0.6
        
    def initialize(self) -> None:
        """Initialize the advanced features classifier."""
        try:
            logger.info("🔬 Initializing Track 4: Advanced Spectral Features + Context")
            
            # Initialize feature extractor
            self.feature_extractor.clear_cache()
            
            # Initialize Random Forest classifier
            self.rf_classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            self.is_initialized = True
            logger.info("✅ Advanced features track initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize advanced features track: {e}")
            raise ClassificationError(f"Advanced features track initialization failed: {e}")
    
    def set_beat_times(self, beat_times: List[float]):
        """
        Set beat times for contextual analysis.
        
        Args:
            beat_times: List of beat times in seconds
        """
        self.beat_times = beat_times
        logger.debug(f"Set {len(beat_times)} beat times for contextual analysis")
    
    def train_classifier(self, 
                        training_data: List[tuple],
                        validation_split: float = 0.2) -> float:
        """
        Train the Random Forest classifier on training data.
        
        Args:
            training_data: List of (audio_window, label) tuples
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training accuracy score
        """
        try:
            logger.info(f"🌲 Training Random Forest with {len(training_data)} samples")
            
            # Extract features and labels
            features_list = []
            labels_list = []
            
            for audio_window, label in training_data:
                features = self.feature_extractor.extract_all_features(
                    audio_window, 
                    feature_set='comprehensive'
                )
                
                # Convert to feature vector
                feature_vector = []
                for key in sorted(features.keys()):
                    value = features[key]
                    if isinstance(value, (int, float)):
                        feature_vector.append(float(value))
                    elif isinstance(value, np.ndarray):
                        feature_vector.extend(value.flatten().astype(float))
                
                features_list.append(feature_vector)
                labels_list.append(label)
            
            # Convert to numpy arrays
            X = np.array(features_list)
            y = np.array(labels_list)
            
            # Handle NaN values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Split data
            if len(X) > self.min_training_samples:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=validation_split, random_state=42, stratify=y
                )
            else:
                X_train, X_test, y_train, y_test = X, X, y, y
            
            # Train classifier
            self.rf_classifier.fit(X_train, y_train)
            
            # Evaluate
            if len(X_test) > 0:
                y_pred = self.rf_classifier.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                logger.info(f"🎯 Random Forest training accuracy: {accuracy:.3f}")
            else:
                accuracy = 0.0
            
            self.is_trained = True
            return accuracy
            
        except Exception as e:
            logger.error(f"Random Forest training failed: {e}")
            self.is_trained = False
            return 0.0
    
    def classify_onsets(self, 
                       onsets: List[tuple],
                       context: Optional[Dict[str, Any]] = None) -> List[ClassificationResult]:
        """
        Classify multiple onsets with advanced features and optional training.
        
        Args:
            onsets: List of (audio_window, onset_time) tuples
            context: Additional context information
            
        Returns:
            List of classification results
        """
        if not self.is_initialized:
            self.initialize()
        
        results = []
        
        # If we have enough onsets, train the classifier first
        if len(onsets) >= self.min_training_samples and not self.is_trained:
            logger.info("🧠 Preparing training data from onsets...")
            
            # Create training data using initial frequency-based classification
            training_data = []
            for audio_window, onset_time in onsets:
                # Get initial label using simple frequency classification
                initial_label = self._simple_frequency_classify(audio_window)
                training_data.append((audio_window, initial_label))
            
            # Train the classifier
            self.train_classifier(training_data)
        
        # Classify each onset
        for audio_window, onset_time in onsets:
            result = self.classify_onset(audio_window, onset_time, context)
            results.append(result)
        
        return results
    
    def classify_onset(self, 
                      audio_window: np.ndarray, 
                      onset_time: float,
                      context: Optional[Dict[str, Any]] = None) -> ClassificationResult:
        """
        Classify a drum onset using advanced features.
        
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
            
            # Extract comprehensive features
            features = self.feature_extractor.extract_all_features(
                audio_window, 
                feature_set='comprehensive'
            )
            
            # Add contextual features
            if self.beat_times:
                contextual_features = self._extract_contextual_features(onset_time)
                features.update(contextual_features)
            
            # If classifier is trained, use it
            if self.is_trained and self.rf_classifier is not None:
                prediction, confidence = self._classify_with_rf(features)
                
                if confidence > self.confidence_threshold:
                    instrument = self._class_id_to_instrument(prediction)
                    
                    # Calculate velocity based on audio energy
                    velocity = self._calculate_velocity(audio_window, features)
                    
                    return ClassificationResult(
                        instrument=instrument,
                        confidence=confidence,
                        velocity=velocity,
                        features={
                            'source': 'random_forest',
                            'feature_count': len(features),
                            'rf_prediction': int(prediction)
                        }
                    )
            
            # Fallback to rule-based classification
            return self._rule_based_classification(audio_window, features, onset_time)
            
        except Exception as e:
            logger.error(f"Advanced features classification failed at {onset_time:.3f}s: {e}")
            return self._fallback_classification(audio_window)
    
    def _extract_contextual_features(self, onset_time: float) -> Dict[str, float]:
        """Extract contextual features based on beat timing."""
        contextual_features = {}
        
        if not self.beat_times:
            return contextual_features
        
        try:
            # Find closest beat
            beat_diffs = np.abs(np.array(self.beat_times) - onset_time)
            closest_beat_idx = np.argmin(beat_diffs)
            closest_beat_time = self.beat_times[closest_beat_idx]
            
            # Beat-relative position
            beat_position = onset_time - closest_beat_time
            contextual_features['beat_relative_position'] = float(beat_position)
            contextual_features['beat_distance'] = float(beat_diffs[closest_beat_idx])
            
            # Beat index within bar (assuming 4/4 time)
            beat_in_bar = closest_beat_idx % 4
            contextual_features['beat_in_bar'] = float(beat_in_bar)
            
            # Is this on a strong beat? (beat 0 or 2 in 4/4)
            contextual_features['is_strong_beat'] = float(beat_in_bar in [0, 2])
            
            # Pattern analysis - look for alternating patterns
            if closest_beat_idx > 0 and closest_beat_idx < len(self.beat_times) - 1:
                prev_interval = self.beat_times[closest_beat_idx] - self.beat_times[closest_beat_idx - 1]
                next_interval = self.beat_times[closest_beat_idx + 1] - self.beat_times[closest_beat_idx]
                
                # Tempo stability
                tempo_stability = 1.0 - abs(prev_interval - next_interval) / max(prev_interval, next_interval)
                contextual_features['tempo_stability'] = float(tempo_stability)
            
        except Exception as e:
            logger.warning(f"Failed to extract contextual features: {e}")
        
        return contextual_features
    
    def _classify_with_rf(self, features: Dict[str, Any]) -> tuple:
        """
        Classify using Random Forest classifier.
        
        Args:
            features: Feature dictionary
            
        Returns:
            Tuple of (prediction, confidence)
        """
        try:
            # Convert features to vector
            feature_vector = []
            for key in sorted(features.keys()):
                value = features[key]
                if isinstance(value, (int, float)):
                    feature_vector.append(float(value))
                elif isinstance(value, np.ndarray):
                    feature_vector.extend(value.flatten().astype(float))
            
            # Convert to numpy array and handle NaN
            X = np.array(feature_vector).reshape(1, -1)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Make prediction
            prediction = self.rf_classifier.predict(X)[0]
            
            # Get confidence from probability
            probabilities = self.rf_classifier.predict_proba(X)[0]
            confidence = float(np.max(probabilities))
            
            return int(prediction), confidence
            
        except Exception as e:
            logger.error(f"Random Forest prediction failed: {e}")
            return 2, 0.1  # Default to kick with low confidence
    
    def _rule_based_classification(self, 
                                 audio_window: np.ndarray, 
                                 features: Dict[str, Any],
                                 onset_time: float) -> ClassificationResult:
        """
        Rule-based classification using advanced features.
        
        Args:
            audio_window: Audio window
            features: Extracted features
            onset_time: Onset time
            
        Returns:
            Classification result
        """
        try:
            # Get key features
            spectral_centroid = features.get('spectral_centroid', 1000)
            rms_energy = features.get('rms_energy', 0.1)
            zero_crossing_rate = features.get('zero_crossing_rate', 0.1)
            
            # Frequency band ratios
            low_ratio = features.get('low_energy_ratio', 0.33)
            mid_ratio = features.get('mid_energy_ratio', 0.33)
            high_ratio = features.get('high_energy_ratio', 0.33)
            
            # Enhanced classification rules
            confidence = 0.5
            
            # Kick drum classification
            if low_ratio > 0.65 and spectral_centroid < 150 and rms_energy > 0.2:
                instrument = 'kick'
                confidence = min(0.85, 0.6 + low_ratio * 0.3)
            
            # Snare drum classification
            elif mid_ratio > 0.4 and 200 < spectral_centroid < 1000 and zero_crossing_rate > 0.1:
                instrument = 'snare'
                confidence = min(0.8, 0.6 + mid_ratio * 0.25)
            
            # Hi-hat classification
            elif high_ratio > 0.5 and spectral_centroid > 2000:
                spectral_spread = features.get('spectral_bandwidth', 1000)
                if spectral_spread < 1000:
                    instrument = 'hi-hat-close'
                    confidence = min(0.75, 0.5 + high_ratio * 0.3)
                else:
                    instrument = 'hi-hat-open'
                    confidence = min(0.7, 0.5 + high_ratio * 0.25)
            
            # Tom classification
            elif low_ratio > 0.3 and mid_ratio > 0.3 and spectral_centroid < 500:
                if spectral_centroid < 200:
                    instrument = 'tom-floor'
                    confidence = min(0.7, 0.5 + low_ratio * 0.2)
                elif spectral_centroid < 350:
                    instrument = 'tom-low'
                    confidence = min(0.68, 0.5 + mid_ratio * 0.2)
                else:
                    instrument = 'tom-high'
                    confidence = min(0.65, 0.5 + mid_ratio * 0.18)
            
            # Crash cymbal classification
            elif high_ratio > 0.6 and spectral_centroid > 1500 and zero_crossing_rate > 0.15:
                instrument = 'crash'
                confidence = min(0.72, 0.5 + high_ratio * 0.25)
            
            # Ride cymbal classification
            elif high_ratio > 0.4 and mid_ratio > 0.25 and 1000 < spectral_centroid < 3000:
                if spectral_centroid > 2000:
                    instrument = 'ride-bell'
                    confidence = min(0.68, 0.5 + high_ratio * 0.2)
                else:
                    instrument = 'ride'
                    confidence = min(0.7, 0.5 + (high_ratio + mid_ratio) * 0.15)
            
            else:
                # Default classification
                instrument = 'snare'
                confidence = 0.4
            
            # Calculate velocity
            velocity = self._calculate_velocity(audio_window, features)
            
            return ClassificationResult(
                instrument=instrument,
                confidence=confidence,
                velocity=velocity,
                features={
                    'source': 'rule_based',
                    'spectral_centroid': spectral_centroid,
                    'low_ratio': low_ratio,
                    'mid_ratio': mid_ratio,
                    'high_ratio': high_ratio
                }
            )
            
        except Exception as e:
            logger.error(f"Rule-based classification failed: {e}")
            return self._fallback_classification(audio_window)
    
    
    def _simple_frequency_classify(self, audio_window: np.ndarray) -> int:
        """Simple frequency classification for training labels."""
        try:
            # Basic frequency analysis
            fft = np.fft.rfft(audio_window)
            freqs = np.fft.rfftfreq(len(audio_window), 1/self.settings.audio.sample_rate)
            magnitude = np.abs(fft)
            
            # Find peak frequency
            peak_idx = np.argmax(magnitude)
            peak_freq = freqs[peak_idx]
            
            # Simple classification
            if peak_freq < 100:
                return 2  # kick
            elif peak_freq < 300:
                return 1  # snare
            elif peak_freq < 800:
                return 3  # tom-high
            elif peak_freq < 1500:
                return 4  # tom-low
            elif peak_freq < 3000:
                return 5  # ride
            elif peak_freq < 6000:
                return 0  # hi-hat-close
            else:
                return 9  # crash
                
        except Exception:
            return 2  # Default to kick
    
    
    def _fallback_classification(self, audio_window: np.ndarray) -> ClassificationResult:
        """Simple fallback classification using mixin."""
        return self._rule_based_fallback(audio_window)
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this track."""
        info = super().get_info()
        info.update({
            'track_number': 4,
            'track_name': 'Advanced Spectral Features + Context',
            'description': 'Enhanced frequency-based classification with 139 features and Random Forest',
            'is_trained': self.is_trained,
            'confidence_threshold': self.confidence_threshold,
            'min_training_samples': self.min_training_samples,
            'beat_times_set': len(self.beat_times) > 0
        })
        return info
    
    def cleanup(self):
        """Clean up resources."""
        super().cleanup()
        if hasattr(self.feature_extractor, 'clear_cache'):
            self.feature_extractor.clear_cache()
        self.rf_classifier = None
        self.is_trained = False