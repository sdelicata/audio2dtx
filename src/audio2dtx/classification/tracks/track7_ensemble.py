"""
Track 7: Ensemble of Specialized Models

Hierarchical classification using specialized models for different instrument groups 
with confidence-based voting.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from ..base_classifier import BaseClassifier, ClassificationResult
from ..feature_extractor import FeatureExtractor
from ..base_track_mixin import BaseTrackMixin
from ...config.settings import Settings
from ...utils.exceptions import ClassificationError
from ...utils.logging import get_logger

logger = get_logger(__name__)


class EnsembleTrack(BaseClassifier, BaseTrackMixin):
    """
    Track 7: Ensemble of Specialized Models
    
    Hierarchical classification using specialized models for different instrument groups
    with confidence-based voting and expert models for each category.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize ensemble classification track.
        
        Args:
            settings: Application settings
        """
        super().__init__(settings)
        self.feature_extractor = FeatureExtractor(settings)
        
        # Specialized classifiers for different instrument groups
        self.kick_snare_classifier = None
        self.cymbal_classifier = None
        self.tom_classifier = None
        self.general_classifier = None
        
        # Hierarchical decision structure
        self.category_classifier = None  # First tier: kick/snare vs cymbals vs toms
        
        # Beat-synchronized training data
        self.training_data = {
            'kick_snare': {'features': [], 'labels': []},
            'cymbals': {'features': [], 'labels': []},
            'toms': {'features': [], 'labels': []},
            'general': {'features': [], 'labels': []}
        }
        
        # Confidence thresholds for each specialist
        self.confidence_thresholds = {
            'kick_snare': 0.7,
            'cymbals': 0.6,
            'toms': 0.6,
            'general': 0.5
        }
        
        # Instrument category mapping
        self.instrument_categories = {
            'kick': 'kick_snare',
            'snare': 'kick_snare',
            'hi-hat-close': 'cymbals',
            'hi-hat-open': 'cymbals',
            'tom-high': 'toms',
            'tom-low': 'toms',
            'tom-floor': 'toms',
            'ride-cymbal': 'cymbals',
            'ride-bell': 'cymbals',
            'crash': 'cymbals'
        }
        
        # Category to class mapping
        self.category_to_classes = {
            'kick_snare': [0, 1],  # kick, snare
            'cymbals': [2, 3, 7, 8, 9],  # hi-hat-close, hi-hat-open, ride-cymbal, ride-bell, crash
            'toms': [4, 5, 6]  # tom-high, tom-low, tom-floor
        }
        
        self.trained_models = False
        
    def initialize(self) -> None:
        """Initialize the ensemble classifier."""
        try:
            logger.info("🎯 Initializing Track 7: Ensemble of Specialized Models")
            
            # Initialize feature extractor
            self.feature_extractor.clear_cache()
            
            # Initialize specialized classifiers
            self._initialize_specialized_classifiers()
            
            # Generate synthetic training data
            self._generate_synthetic_training_data()
            
            # Train the ensemble
            self._train_ensemble()
            
            self.is_initialized = True
            logger.info("✅ Ensemble track initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ensemble track: {e}")
            raise ClassificationError(f"Ensemble track initialization failed: {e}")
    
    def _initialize_specialized_classifiers(self):
        """Initialize specialized classifiers for each instrument group."""
        # Kick/Snare specialist - focused on energy and spectral features
        self.kick_snare_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Cymbal specialist - focused on high-frequency analysis
        self.cymbal_classifier = RandomForestClassifier(
            n_estimators=80,
            max_depth=8,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42
        )
        
        # Tom specialist - focused on mid-frequency analysis
        self.tom_classifier = RandomForestClassifier(
            n_estimators=60,
            max_depth=6,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42
        )
        
        # General classifier - backup for all categories
        self.general_classifier = RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        # Category classifier - first tier decision
        self.category_classifier = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42
        )
    
    def _generate_synthetic_training_data(self):
        """Generate synthetic training data for each specialist."""
        try:
            logger.info("🔧 Generating synthetic training data for ensemble")
            
            # Generate features for each instrument class
            n_samples_per_class = 50
            
            for instrument_id in range(10):
                instrument_name = self._class_id_to_instrument(instrument_id)
                category = self.instrument_categories.get(instrument_name, 'general')
                
                # Generate synthetic features based on instrument characteristics
                features = self._generate_instrument_features(instrument_id, n_samples_per_class)
                labels = [instrument_id] * n_samples_per_class
                
                # Add to appropriate category
                self.training_data[category]['features'].extend(features)
                self.training_data[category]['labels'].extend(labels)
                
                # Add to general training data
                self.training_data['general']['features'].extend(features)
                self.training_data['general']['labels'].extend(labels)
            
            logger.info(f"✅ Generated training data for {len(self.training_data)} categories")
            
        except Exception as e:
            logger.error(f"Failed to generate synthetic training data: {e}")
            raise
    
    def _generate_instrument_features(self, instrument_id: int, n_samples: int) -> List[np.ndarray]:
        """Generate synthetic features for a specific instrument."""
        features = []
        
        # Base characteristics for each instrument
        base_characteristics = {
            0: {'centroid': 200, 'rms': 0.15, 'low_ratio': 0.8, 'high_ratio': 0.1},  # kick
            1: {'centroid': 800, 'rms': 0.25, 'low_ratio': 0.3, 'high_ratio': 0.4},  # snare
            2: {'centroid': 4000, 'rms': 0.1, 'low_ratio': 0.1, 'high_ratio': 0.7},  # hi-hat-close
            3: {'centroid': 3500, 'rms': 0.15, 'low_ratio': 0.1, 'high_ratio': 0.6},  # hi-hat-open
            4: {'centroid': 600, 'rms': 0.2, 'low_ratio': 0.4, 'high_ratio': 0.2},  # tom-high
            5: {'centroid': 400, 'rms': 0.22, 'low_ratio': 0.5, 'high_ratio': 0.15},  # tom-low
            6: {'centroid': 300, 'rms': 0.25, 'low_ratio': 0.6, 'high_ratio': 0.1},  # tom-floor
            7: {'centroid': 2500, 'rms': 0.18, 'low_ratio': 0.2, 'high_ratio': 0.5},  # ride-cymbal
            8: {'centroid': 3000, 'rms': 0.2, 'low_ratio': 0.15, 'high_ratio': 0.55},  # ride-bell
            9: {'centroid': 4500, 'rms': 0.3, 'low_ratio': 0.05, 'high_ratio': 0.8}   # crash
        }
        
        characteristics = base_characteristics.get(instrument_id, base_characteristics[0])
        
        for _ in range(n_samples):
            # Add variation to base characteristics
            centroid = characteristics['centroid'] * (1 + np.random.normal(0, 0.3))
            rms = characteristics['rms'] * (1 + np.random.normal(0, 0.2))
            low_ratio = np.clip(characteristics['low_ratio'] + np.random.normal(0, 0.1), 0, 1)
            high_ratio = np.clip(characteristics['high_ratio'] + np.random.normal(0, 0.1), 0, 1)
            mid_ratio = np.clip(1 - low_ratio - high_ratio, 0, 1)
            
            # Create feature vector
            feature_vector = np.array([
                rms,
                centroid,
                centroid * 0.8,  # spectral_rolloff
                centroid * 0.3,  # spectral_bandwidth
                np.random.uniform(0.01, 0.2),  # zero_crossing_rate
                low_ratio,
                mid_ratio,
                high_ratio,
                np.random.uniform(0.1, 0.9),  # spectral_flatness
                np.random.uniform(-1, 1),  # spectral_skewness
                np.random.uniform(1, 5),  # spectral_kurtosis
                np.random.uniform(0.5, 2.0)  # spectral_spread
            ])
            
            features.append(feature_vector)
        
        return features
    
    def _train_ensemble(self):
        """Train all specialized classifiers."""
        try:
            logger.info("🎓 Training ensemble classifiers")
            
            # Train category classifier (first tier)
            self._train_category_classifier()
            
            # Train specialized classifiers
            self._train_kick_snare_classifier()
            self._train_cymbal_classifier()
            self._train_tom_classifier()
            self._train_general_classifier()
            
            self.trained_models = True
            logger.info("✅ All ensemble classifiers trained successfully")
            
        except Exception as e:
            logger.error(f"Failed to train ensemble: {e}")
            raise
    
    def _train_category_classifier(self):
        """Train the category classifier for first-tier decisions."""
        try:
            # Prepare category training data
            category_features = []
            category_labels = []
            
            for category, data in self.training_data.items():
                if category == 'general':
                    continue
                    
                for feature in data['features']:
                    category_features.append(feature)
                    category_labels.append(category)
            
            if category_features:
                X = np.array(category_features)
                y = np.array(category_labels)
                
                self.category_classifier.fit(X, y)
                
                # Evaluate category classifier
                scores = cross_val_score(self.category_classifier, X, y, cv=3)
                logger.info(f"Category classifier accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
                
        except Exception as e:
            logger.warning(f"Category classifier training failed: {e}")
    
    def _train_kick_snare_classifier(self):
        """Train the kick/snare specialist."""
        try:
            data = self.training_data['kick_snare']
            if data['features']:
                X = np.array(data['features'])
                y = np.array(data['labels'])
                
                self.kick_snare_classifier.fit(X, y)
                
                # Evaluate
                scores = cross_val_score(self.kick_snare_classifier, X, y, cv=3)
                logger.info(f"Kick/Snare classifier accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
                
        except Exception as e:
            logger.warning(f"Kick/Snare classifier training failed: {e}")
    
    def _train_cymbal_classifier(self):
        """Train the cymbal specialist."""
        try:
            data = self.training_data['cymbals']
            if data['features']:
                X = np.array(data['features'])
                y = np.array(data['labels'])
                
                self.cymbal_classifier.fit(X, y)
                
                # Evaluate
                scores = cross_val_score(self.cymbal_classifier, X, y, cv=3)
                logger.info(f"Cymbal classifier accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
                
        except Exception as e:
            logger.warning(f"Cymbal classifier training failed: {e}")
    
    def _train_tom_classifier(self):
        """Train the tom specialist."""
        try:
            data = self.training_data['toms']
            if data['features']:
                X = np.array(data['features'])
                y = np.array(data['labels'])
                
                self.tom_classifier.fit(X, y)
                
                # Evaluate
                scores = cross_val_score(self.tom_classifier, X, y, cv=3)
                logger.info(f"Tom classifier accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
                
        except Exception as e:
            logger.warning(f"Tom classifier training failed: {e}")
    
    def _train_general_classifier(self):
        """Train the general classifier."""
        try:
            data = self.training_data['general']
            if data['features']:
                X = np.array(data['features'])
                y = np.array(data['labels'])
                
                self.general_classifier.fit(X, y)
                
                # Evaluate
                scores = cross_val_score(self.general_classifier, X, y, cv=3)
                logger.info(f"General classifier accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
                
        except Exception as e:
            logger.warning(f"General classifier training failed: {e}")
    
    def classify_onsets(self, 
                       onsets: List[tuple],
                       context: Optional[Dict[str, Any]] = None) -> List[ClassificationResult]:
        """
        Classify multiple onsets using ensemble of specialized models.
        
        Args:
            onsets: List of (audio_window, onset_time) tuples
            context: Additional context information
            
        Returns:
            List of classification results
        """
        if not self.is_initialized:
            self.initialize()
        
        results = []
        
        for audio_window, onset_time in onsets:
            result = self.classify_onset(audio_window, onset_time, context)
            results.append(result)
        
        logger.info(f"🎯 Ensemble track classified {len(results)} onsets")
        return results
    
    def classify_onset(self, 
                      audio_window: np.ndarray, 
                      onset_time: float,
                      context: Optional[Dict[str, Any]] = None) -> ClassificationResult:
        """
        Classify a drum onset using ensemble of specialized models.
        
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
            
            # Hierarchical classification
            if self.trained_models:
                # Tier 1: Category prediction
                category_prediction = self._predict_category(feature_vector)
                
                # Tier 2: Specialized classification
                specialized_result = self._classify_with_specialist(
                    feature_vector, category_prediction, features
                )
                
                # Tier 3: Confidence-based voting
                final_result = self._ensemble_voting(
                    feature_vector, specialized_result, features
                )
                
                return final_result
            else:
                # Fallback to rule-based classification
                return self._rule_based_classification(feature_vector, features)
                
        except Exception as e:
            logger.error(f"Ensemble classification failed at {onset_time:.3f}s: {e}")
            return self._fallback_classification(audio_window)
    
    def _features_to_vector(self, features: Dict[str, Any]) -> np.ndarray:
        """Convert features dictionary to vector for ensemble classifiers."""
        feature_vector = []
        
        # Core features for ensemble classification
        key_features = [
            'rms_energy', 'spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth',
            'zero_crossing_rate', 'low_energy_ratio', 'mid_energy_ratio', 'high_energy_ratio',
            'spectral_flatness', 'spectral_skewness', 'spectral_kurtosis', 'spectral_spread'
        ]
        
        for key in key_features:
            if key in features:
                value = features[key]
                if isinstance(value, (int, float)):
                    feature_vector.append(float(value))
                elif isinstance(value, np.ndarray):
                    feature_vector.append(float(np.mean(value)))
                else:
                    feature_vector.append(0.0)
            else:
                feature_vector.append(0.0)
        
        return np.array(feature_vector).reshape(1, -1)
    
    def _predict_category(self, feature_vector: np.ndarray) -> str:
        """Predict instrument category using first-tier classifier."""
        try:
            if self.category_classifier is not None:
                category_proba = self.category_classifier.predict_proba(feature_vector)[0]
                categories = self.category_classifier.classes_
                
                # Get category with highest probability
                best_category_idx = np.argmax(category_proba)
                best_category = categories[best_category_idx]
                confidence = category_proba[best_category_idx]
                
                # Use prediction if confidence is high enough
                if confidence > 0.4:
                    return best_category
            
            # Fallback to rule-based category prediction
            return self._rule_based_category_prediction(feature_vector)
            
        except Exception as e:
            logger.warning(f"Category prediction failed: {e}")
            return 'general'
    
    def _rule_based_category_prediction(self, feature_vector: np.ndarray) -> str:
        """Rule-based category prediction as fallback."""
        try:
            # Extract key features (assuming standard order)
            if len(feature_vector[0]) >= 8:
                spectral_centroid = feature_vector[0][1]
                low_ratio = feature_vector[0][5]
                high_ratio = feature_vector[0][7]
                
                # Simple rules
                if high_ratio > 0.5 and spectral_centroid > 2000:
                    return 'cymbals'
                elif low_ratio > 0.5 and spectral_centroid < 500:
                    return 'kick_snare'
                elif 300 < spectral_centroid < 1000:
                    return 'toms'
                else:
                    return 'kick_snare'
            
            return 'general'
            
        except Exception:
            return 'general'
    
    def _classify_with_specialist(self, 
                                feature_vector: np.ndarray,
                                category: str,
                                features: Dict[str, Any]) -> Tuple[int, float]:
        """Classify using appropriate specialist model."""
        try:
            specialist_map = {
                'kick_snare': self.kick_snare_classifier,
                'cymbals': self.cymbal_classifier,
                'toms': self.tom_classifier,
                'general': self.general_classifier
            }
            
            specialist = specialist_map.get(category, self.general_classifier)
            
            if specialist is not None:
                # Get prediction and confidence
                prediction = specialist.predict(feature_vector)[0]
                probabilities = specialist.predict_proba(feature_vector)[0]
                confidence = np.max(probabilities)
                
                return int(prediction), float(confidence)
            else:
                # Fallback to rule-based
                return self._rule_based_prediction(feature_vector, features)
                
        except Exception as e:
            logger.warning(f"Specialist classification failed: {e}")
            return self._rule_based_prediction(feature_vector, features)
    
    def _ensemble_voting(self, 
                        feature_vector: np.ndarray,
                        specialized_result: Tuple[int, float],
                        features: Dict[str, Any]) -> ClassificationResult:
        """Perform ensemble voting for final classification."""
        try:
            specialist_prediction, specialist_confidence = specialized_result
            
            # Get predictions from all available classifiers
            predictions = []
            confidences = []
            
            # Specialist prediction
            predictions.append(specialist_prediction)
            confidences.append(specialist_confidence)
            
            # General classifier prediction
            if self.general_classifier is not None:
                try:
                    general_pred = self.general_classifier.predict(feature_vector)[0]
                    general_proba = self.general_classifier.predict_proba(feature_vector)[0]
                    general_conf = np.max(general_proba)
                    
                    predictions.append(int(general_pred))
                    confidences.append(float(general_conf))
                except:
                    pass
            
            # Rule-based prediction
            rule_pred, rule_conf = self._rule_based_prediction(feature_vector, features)
            predictions.append(rule_pred)
            confidences.append(rule_conf)
            
            # Weighted voting
            if len(predictions) > 1:
                # Weight by confidence
                weights = np.array(confidences)
                weights = weights / np.sum(weights)
                
                # Count votes with weights
                vote_counts = defaultdict(float)
                for pred, weight in zip(predictions, weights):
                    vote_counts[pred] += weight
                
                # Find winner
                winner = max(vote_counts.keys(), key=lambda x: vote_counts[x])
                winner_confidence = vote_counts[winner]
                
                # Boost confidence if multiple classifiers agree
                if len([p for p in predictions if p == winner]) > 1:
                    winner_confidence *= 1.2
                
                final_prediction = winner
                final_confidence = min(winner_confidence, 1.0)
            else:
                final_prediction = specialist_prediction
                final_confidence = specialist_confidence
            
            # Convert to instrument name
            instrument = self._class_id_to_instrument(final_prediction)
            
            # Calculate velocity
            velocity = self._calculate_velocity(features)
            
            return ClassificationResult(
                instrument=instrument,
                confidence=final_confidence,
                velocity=velocity,
                features={
                    'source': 'ensemble',
                    'specialist_prediction': specialist_prediction,
                    'specialist_confidence': specialist_confidence,
                    'ensemble_votes': len(predictions),
                    'voting_weights': confidences
                }
            )
            
        except Exception as e:
            logger.error(f"Ensemble voting failed: {e}")
            return self._fallback_classification_result(features)
    
    def _rule_based_prediction(self, 
                             feature_vector: np.ndarray,
                             features: Dict[str, Any]) -> Tuple[int, float]:
        """Rule-based classification as fallback."""
        try:
            spectral_centroid = features.get('spectral_centroid', 1000)
            rms_energy = features.get('rms_energy', 0.1)
            low_ratio = features.get('low_energy_ratio', 0.33)
            high_ratio = features.get('high_energy_ratio', 0.33)
            
            # Enhanced rule-based classification
            if low_ratio > 0.6 and spectral_centroid < 150:
                return 0, 0.7  # kick
            elif 200 < spectral_centroid < 1000 and rms_energy > 0.2:
                return 1, 0.65  # snare
            elif high_ratio > 0.5 and spectral_centroid > 3000:
                if rms_energy > 0.25:
                    return 9, 0.6  # crash
                else:
                    return 2, 0.6  # hi-hat-close
            elif 2000 < spectral_centroid < 3000 and high_ratio > 0.3:
                return 7, 0.55  # ride-cymbal
            elif 300 < spectral_centroid < 600:
                return 4, 0.5  # tom-high
            elif 200 < spectral_centroid < 400:
                return 5, 0.5  # tom-low
            else:
                return 0, 0.4  # default to kick
                
        except Exception:
            return 0, 0.3
    
    def _calculate_velocity(self, features: Dict[str, Any]) -> float:
        """Calculate velocity based on audio features."""
        try:
            rms_energy = features.get('rms_energy', 0.1)
            spectral_centroid = features.get('spectral_centroid', 1000)
            
            # Base velocity from RMS energy
            base_velocity = min(rms_energy * 5, 1.0)
            
            # Adjust based on spectral characteristics
            if spectral_centroid > 3000:  # High frequency instruments
                velocity = base_velocity * 1.2
            elif spectral_centroid < 200:  # Low frequency instruments
                velocity = base_velocity * 0.8
            else:
                velocity = base_velocity
            
            return float(np.clip(velocity, 0.1, 1.0))
            
        except Exception:
            return 0.5
    
    def _class_id_to_instrument(self, class_id: int) -> str:
        """Convert class ID to instrument name."""
        from ...config.constants import DRUM_CLASSES
        return DRUM_CLASSES.get(class_id, 'kick')
    
    def _fallback_classification(self, audio_window: np.ndarray) -> ClassificationResult:
        """Simple fallback classification."""
        try:
            # Simple spectral analysis
            fft = np.fft.rfft(audio_window)
            freqs = np.fft.rfftfreq(len(audio_window), 1/self.settings.audio.sample_rate)
            magnitude = np.abs(fft)
            
            if len(magnitude) > 0:
                peak_idx = np.argmax(magnitude)
                peak_freq = freqs[peak_idx]
                
                # Simple frequency-based classification
                if peak_freq < 100:
                    instrument = 'kick'
                    confidence = 0.5
                elif peak_freq < 300:
                    instrument = 'snare'
                    confidence = 0.4
                elif peak_freq > 3000:
                    instrument = 'crash'
                    confidence = 0.4
                else:
                    instrument = 'tom-high'
                    confidence = 0.3
            else:
                instrument = 'kick'
                confidence = 0.2
            
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
    
    def _fallback_classification_result(self, features: Dict[str, Any]) -> ClassificationResult:
        """Fallback classification result from features."""
        try:
            spectral_centroid = features.get('spectral_centroid', 1000)
            
            if spectral_centroid < 200:
                instrument = 'kick'
                confidence = 0.4
            elif spectral_centroid > 3000:
                instrument = 'crash'
                confidence = 0.4
            else:
                instrument = 'snare'
                confidence = 0.3
            
            velocity = self._calculate_velocity(features)
            
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
            'track_number': 7,
            'track_name': 'Ensemble of Specialized Models',
            'description': 'Hierarchical classification using specialized models for instrument groups',
            'trained_models': self.trained_models,
            'confidence_thresholds': self.confidence_thresholds,
            'specialist_models': {
                'kick_snare': self.kick_snare_classifier is not None,
                'cymbals': self.cymbal_classifier is not None,
                'toms': self.tom_classifier is not None,
                'general': self.general_classifier is not None,
                'category': self.category_classifier is not None
            }
        })
        return info
    
    def cleanup(self):
        """Clean up resources."""
        super().cleanup()
        if hasattr(self.feature_extractor, 'clear_cache'):
            self.feature_extractor.clear_cache()
        
        # Clear training data
        for category in self.training_data:
            self.training_data[category]['features'].clear()
            self.training_data[category]['labels'].clear()