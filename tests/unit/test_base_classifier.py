"""
Unit tests for base classifier components.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from audio2dtx.classification.base_classifier import BaseClassifier, ClassificationResult
from audio2dtx.config.settings import Settings
from audio2dtx.utils.exceptions import ClassificationError


class TestBaseClassifier:
    """Test cases for BaseClassifier."""
    
    def test_classification_result_creation(self):
        """Test ClassificationResult creation."""
        result = ClassificationResult(
            instrument='kick',
            confidence=0.8,
            velocity=0.7,
            features={'source': 'test'}
        )
        
        assert result.instrument == 'kick'
        assert result.confidence == 0.8
        assert result.velocity == 0.7
        assert result.features == {'source': 'test'}
    
    def test_classification_result_validation(self):
        """Test ClassificationResult validation."""
        # Valid result
        result = ClassificationResult(
            instrument='snare',
            confidence=0.5,
            velocity=0.6,
            features={}
        )
        assert result.instrument == 'snare'
        
        # Test confidence bounds
        result_low = ClassificationResult(
            instrument='kick',
            confidence=0.0,
            velocity=0.5,
            features={}
        )
        assert result_low.confidence == 0.0
        
        result_high = ClassificationResult(
            instrument='kick',
            confidence=1.0,
            velocity=0.5,
            features={}
        )
        assert result_high.confidence == 1.0
    
    def test_base_classifier_initialization(self, settings):
        """Test BaseClassifier initialization."""
        
        class TestClassifier(BaseClassifier):
            def initialize(self):
                self.is_initialized = True
            
            def classify_onset(self, audio_window, onset_time, context=None):
                return ClassificationResult(
                    instrument='test',
                    confidence=0.5,
                    velocity=0.5,
                    features={}
                )
        
        classifier = TestClassifier(settings)
        assert classifier.settings == settings
        assert not classifier.is_initialized
        
        classifier.initialize()
        assert classifier.is_initialized
    
    def test_base_classifier_classify_onset_not_implemented(self, settings):
        """Test that classify_onset raises NotImplementedError in base class."""
        
        class IncompleteClassifier(BaseClassifier):
            def initialize(self):
                self.is_initialized = True
        
        classifier = IncompleteClassifier(settings)
        classifier.initialize()
        
        with pytest.raises(NotImplementedError):
            classifier.classify_onset(np.zeros(1024), 0.0)
    
    def test_base_classifier_classify_onsets_batch(self, settings):
        """Test batch classification with classify_onsets."""
        
        class TestClassifier(BaseClassifier):
            def initialize(self):
                self.is_initialized = True
            
            def classify_onset(self, audio_window, onset_time, context=None):
                return ClassificationResult(
                    instrument='test',
                    confidence=0.5,
                    velocity=0.5,
                    features={'onset_time': onset_time}
                )
        
        classifier = TestClassifier(settings)
        classifier.initialize()
        
        # Test batch processing
        onsets = [
            (np.zeros(1024), 0.1),
            (np.zeros(1024), 0.3),
            (np.zeros(1024), 0.5)
        ]
        
        results = classifier.classify_onsets(onsets)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.instrument == 'test'
            assert result.features['onset_time'] == onsets[i][1]
    
    def test_base_classifier_get_info(self, settings):
        """Test get_info method."""
        
        class TestClassifier(BaseClassifier):
            def initialize(self):
                self.is_initialized = True
            
            def classify_onset(self, audio_window, onset_time, context=None):
                return ClassificationResult(
                    instrument='test',
                    confidence=0.5,
                    velocity=0.5,
                    features={}
                )
        
        classifier = TestClassifier(settings)
        classifier.initialize()
        
        info = classifier.get_info()
        
        assert isinstance(info, dict)
        assert 'initialized' in info
        assert info['initialized'] is True
    
    def test_base_classifier_cleanup(self, settings):
        """Test cleanup method."""
        
        class TestClassifier(BaseClassifier):
            def initialize(self):
                self.is_initialized = True
                self.test_resource = "allocated"
            
            def classify_onset(self, audio_window, onset_time, context=None):
                return ClassificationResult(
                    instrument='test',
                    confidence=0.5,
                    velocity=0.5,
                    features={}
                )
            
            def cleanup(self):
                super().cleanup()
                self.test_resource = None
        
        classifier = TestClassifier(settings)
        classifier.initialize()
        
        assert classifier.is_initialized
        assert classifier.test_resource == "allocated"
        
        classifier.cleanup()
        
        assert not classifier.is_initialized
        assert classifier.test_resource is None
    
    def test_classification_result_repr(self):
        """Test ClassificationResult string representation."""
        result = ClassificationResult(
            instrument='kick',
            confidence=0.85,
            velocity=0.7,
            features={'source': 'test'}
        )
        
        repr_str = repr(result)
        assert 'kick' in repr_str
        assert '0.85' in repr_str
        assert '0.7' in repr_str
    
    def test_classification_result_equality(self):
        """Test ClassificationResult equality comparison."""
        result1 = ClassificationResult(
            instrument='kick',
            confidence=0.8,
            velocity=0.7,
            features={'source': 'test'}
        )
        
        result2 = ClassificationResult(
            instrument='kick',
            confidence=0.8,
            velocity=0.7,
            features={'source': 'test'}
        )
        
        result3 = ClassificationResult(
            instrument='snare',
            confidence=0.8,
            velocity=0.7,
            features={'source': 'test'}
        )
        
        assert result1 == result2
        assert result1 != result3
    
    def test_classification_result_to_dict(self):
        """Test ClassificationResult to dictionary conversion."""
        result = ClassificationResult(
            instrument='kick',
            confidence=0.8,
            velocity=0.7,
            features={'source': 'test', 'extra': 'data'}
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['instrument'] == 'kick'
        assert result_dict['confidence'] == 0.8
        assert result_dict['velocity'] == 0.7
        assert result_dict['features'] == {'source': 'test', 'extra': 'data'}
    
    def test_classification_result_from_dict(self):
        """Test ClassificationResult creation from dictionary."""
        result_dict = {
            'instrument': 'snare',
            'confidence': 0.9,
            'velocity': 0.8,
            'features': {'source': 'test', 'method': 'advanced'}
        }
        
        result = ClassificationResult.from_dict(result_dict)
        
        assert result.instrument == 'snare'
        assert result.confidence == 0.9
        assert result.velocity == 0.8
        assert result.features == {'source': 'test', 'method': 'advanced'}


class TestEnsembleClassifier:
    """Test cases for EnsembleClassifier."""
    
    def test_ensemble_classifier_initialization(self, settings):
        """Test EnsembleClassifier initialization."""
        from audio2dtx.classification.base_classifier import EnsembleClassifier
        
        # Create mock classifiers
        mock_classifier1 = Mock()
        mock_classifier1.classify_onset.return_value = ClassificationResult(
            instrument='kick', confidence=0.8, velocity=0.7, features={}
        )
        
        mock_classifier2 = Mock()
        mock_classifier2.classify_onset.return_value = ClassificationResult(
            instrument='snare', confidence=0.6, velocity=0.5, features={}
        )
        
        classifiers = [mock_classifier1, mock_classifier2]
        weights = [0.6, 0.4]
        
        ensemble = EnsembleClassifier(settings, classifiers, weights)
        
        assert ensemble.classifiers == classifiers
        assert ensemble.weights == weights
        assert ensemble.settings == settings
    
    def test_ensemble_classifier_classify_onset(self, settings):
        """Test EnsembleClassifier onset classification."""
        from audio2dtx.classification.base_classifier import EnsembleClassifier
        
        # Create mock classifiers with different predictions
        mock_classifier1 = Mock()
        mock_classifier1.classify_onset.return_value = ClassificationResult(
            instrument='kick', confidence=0.9, velocity=0.8, features={}
        )
        
        mock_classifier2 = Mock()
        mock_classifier2.classify_onset.return_value = ClassificationResult(
            instrument='kick', confidence=0.7, velocity=0.6, features={}
        )
        
        mock_classifier3 = Mock()
        mock_classifier3.classify_onset.return_value = ClassificationResult(
            instrument='snare', confidence=0.6, velocity=0.5, features={}
        )
        
        classifiers = [mock_classifier1, mock_classifier2, mock_classifier3]
        weights = [0.5, 0.3, 0.2]
        
        ensemble = EnsembleClassifier(settings, classifiers, weights)
        ensemble.initialize()
        
        audio_window = np.zeros(1024)
        result = ensemble.classify_onset(audio_window, 0.5)
        
        # Should predict 'kick' since it has higher weighted vote
        assert result.instrument == 'kick'
        assert result.confidence > 0.7  # Should be high due to consensus
        assert 0.0 <= result.velocity <= 1.0
    
    def test_ensemble_classifier_no_consensus(self, settings):
        """Test EnsembleClassifier when there's no clear consensus."""
        from audio2dtx.classification.base_classifier import EnsembleClassifier
        
        # Create mock classifiers with different predictions
        mock_classifier1 = Mock()
        mock_classifier1.classify_onset.return_value = ClassificationResult(
            instrument='kick', confidence=0.5, velocity=0.6, features={}
        )
        
        mock_classifier2 = Mock()
        mock_classifier2.classify_onset.return_value = ClassificationResult(
            instrument='snare', confidence=0.5, velocity=0.7, features={}
        )
        
        mock_classifier3 = Mock()
        mock_classifier3.classify_onset.return_value = ClassificationResult(
            instrument='hi-hat-close', confidence=0.5, velocity=0.4, features={}
        )
        
        classifiers = [mock_classifier1, mock_classifier2, mock_classifier3]
        weights = [0.33, 0.33, 0.34]
        
        ensemble = EnsembleClassifier(settings, classifiers, weights)
        ensemble.initialize()
        
        audio_window = np.zeros(1024)
        result = ensemble.classify_onset(audio_window, 0.5)
        
        # Should still return a valid result
        assert result.instrument in ['kick', 'snare', 'hi-hat-close']
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.velocity <= 1.0