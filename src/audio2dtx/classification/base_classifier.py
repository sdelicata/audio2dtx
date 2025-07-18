"""
Base classifier interface for drum classification tracks.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

from ..utils.exceptions import ClassificationError
from ..config.settings import Settings


class ClassificationResult:
    """Result of drum classification for a single onset."""
    
    def __init__(self, 
                 instrument: str, 
                 confidence: float, 
                 velocity: float = 0.5,
                 features: Optional[Dict[str, Any]] = None):
        self.instrument = instrument
        self.confidence = confidence
        self.velocity = velocity
        self.features = features or {}
        self.timestamp = None
        
    def __repr__(self) -> str:
        return f"ClassificationResult(instrument='{self.instrument}', confidence={self.confidence:.3f})"


class BaseClassifier(ABC):
    """
    Abstract base class for drum classification algorithms.
    
    All track implementations should inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize the classifier.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.is_initialized = False
        self.name = self.__class__.__name__
        
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the classifier (load models, prepare resources, etc.).
        
        Raises:
            ClassificationError: If initialization fails
        """
        pass
    
    @abstractmethod
    def classify_onset(self, 
                      audio_window: np.ndarray, 
                      onset_time: float,
                      context: Optional[Dict[str, Any]] = None) -> ClassificationResult:
        """
        Classify a single drum onset.
        
        Args:
            audio_window: Audio data around the onset
            onset_time: Time of the onset in seconds
            context: Additional context information
            
        Returns:
            Classification result
            
        Raises:
            ClassificationError: If classification fails
        """
        pass
    
    def classify_onsets(self, 
                       onsets: List[Tuple[np.ndarray, float]],
                       context: Optional[Dict[str, Any]] = None) -> List[ClassificationResult]:
        """
        Classify multiple drum onsets.
        
        Args:
            onsets: List of (audio_window, onset_time) tuples
            context: Additional context information
            
        Returns:
            List of classification results
            
        Raises:
            ClassificationError: If classification fails
        """
        if not self.is_initialized:
            self.initialize()
            
        results = []
        for audio_window, onset_time in onsets:
            try:
                result = self.classify_onset(audio_window, onset_time, context)
                result.timestamp = onset_time
                results.append(result)
            except Exception as e:
                # Log error but continue with other onsets
                self._log_classification_error(onset_time, e)
                # Add fallback result
                fallback = ClassificationResult('snare', 0.1, 0.5)
                fallback.timestamp = onset_time
                results.append(fallback)
                
        return results
    
    def get_confidence_threshold(self) -> float:
        """Get the confidence threshold for this classifier."""
        return self.settings.classification.confidence_threshold
    
    def get_supported_instruments(self) -> List[str]:
        """Get list of instruments this classifier can detect."""
        from ..config.constants import DRUM_CLASSES
        return list(DRUM_CLASSES.values())
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this classifier.
        
        Returns:
            Dictionary with classifier information
        """
        return {
            'name': self.name,
            'initialized': self.is_initialized,
            'supported_instruments': self.get_supported_instruments(),
            'confidence_threshold': self.get_confidence_threshold()
        }
    
    def _log_classification_error(self, onset_time: float, error: Exception):
        """Log classification error (to be implemented with proper logging)."""
        print(f"Classification error at {onset_time:.3f}s: {error}")
    
    def cleanup(self):
        """Clean up resources (override if needed)."""
        pass
    
    def __enter__(self):
        """Context manager entry."""
        if not self.is_initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


class EnsembleClassifier(BaseClassifier):
    """
    Ensemble classifier that combines multiple classifiers.
    """
    
    def __init__(self, settings: Settings, classifiers: List[BaseClassifier]):
        """
        Initialize ensemble classifier.
        
        Args:
            settings: Application settings
            classifiers: List of classifiers to combine
        """
        super().__init__(settings)
        self.classifiers = classifiers
        self.weights = [1.0] * len(classifiers)  # Equal weights by default
        
    def set_weights(self, weights: List[float]):
        """
        Set weights for combining classifier results.
        
        Args:
            weights: List of weights (should sum to 1.0)
        """
        if len(weights) != len(self.classifiers):
            raise ValueError("Number of weights must match number of classifiers")
        self.weights = weights
    
    def initialize(self):
        """Initialize all classifiers in the ensemble."""
        for classifier in self.classifiers:
            if not classifier.is_initialized:
                classifier.initialize()
        self.is_initialized = True
    
    def classify_onset(self, 
                      audio_window: np.ndarray, 
                      onset_time: float,
                      context: Optional[Dict[str, Any]] = None) -> ClassificationResult:
        """
        Classify onset using ensemble voting.
        
        Args:
            audio_window: Audio data around the onset
            onset_time: Time of the onset in seconds
            context: Additional context information
            
        Returns:
            Combined classification result
        """
        # Get results from all classifiers
        results = []
        for classifier in self.classifiers:
            try:
                result = classifier.classify_onset(audio_window, onset_time, context)
                results.append(result)
            except Exception as e:
                # Add low-confidence fallback
                fallback = ClassificationResult('snare', 0.1, 0.5)
                results.append(fallback)
        
        # Combine results using weighted voting
        return self._combine_results(results)
    
    def _combine_results(self, results: List[ClassificationResult]) -> ClassificationResult:
        """
        Combine multiple classification results using weighted voting.
        
        Args:
            results: List of classification results
            
        Returns:
            Combined result
        """
        if not results:
            return ClassificationResult('snare', 0.1, 0.5)
        
        # Count weighted votes for each instrument
        instrument_scores = {}
        total_weight = 0
        
        for result, weight in zip(results, self.weights):
            score = result.confidence * weight
            if result.instrument not in instrument_scores:
                instrument_scores[result.instrument] = 0
            instrument_scores[result.instrument] += score
            total_weight += weight
        
        # Normalize scores
        if total_weight > 0:
            for instrument in instrument_scores:
                instrument_scores[instrument] /= total_weight
        
        # Find best instrument
        best_instrument = max(instrument_scores, key=instrument_scores.get)
        best_confidence = instrument_scores[best_instrument]
        
        # Calculate average velocity
        avg_velocity = np.mean([r.velocity for r in results])
        
        return ClassificationResult(best_instrument, best_confidence, avg_velocity)
    
    def cleanup(self):
        """Clean up all classifiers."""
        for classifier in self.classifiers:
            classifier.cleanup()