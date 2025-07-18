"""
Machine learning service for onset detection and classification.
"""

import os
import numpy as np
from typing import Dict, Any, Optional, List
import warnings
warnings.filterwarnings('ignore')

from ..config.settings import Settings
from ..utils.exceptions import ServiceError
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MLService:
    """
    Service for machine learning model loading and inference.
    """
    
    def __init__(self, settings: Settings, model_path: str = "PredictOnset.h5"):
        """
        Initialize ML service.
        
        Args:
            settings: Application settings
            model_path: Path to the ML model file
        """
        self.settings = settings
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
        
        # Try to load model on initialization
        self._load_model()
    
    def _load_model(self):
        """Load the machine learning model."""
        try:
            import tensorflow as tf
            
            if not os.path.exists(self.model_path):
                logger.warning(f"Model file not found: {self.model_path}")
                return
            
            # Load model with custom objects if needed
            self.model = tf.keras.models.load_model(
                self.model_path,
                compile=False  # Skip compilation for faster loading
            )
            
            self.is_loaded = True
            logger.info(f"Successfully loaded ML model from {self.model_path}")
            
            # Log model info
            if hasattr(self.model, 'summary'):
                total_params = self.model.count_params()
                logger.info(f"Model loaded with {total_params:,} parameters")
            
        except ImportError:
            logger.error("TensorFlow not available. ML-based onset detection will be disabled.")
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            self.model = None
            self.is_loaded = False
    
    def predict_onsets(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Predict onset probabilities for audio signal.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Array of onset probabilities or None if prediction fails
        """
        if not self.is_loaded:
            logger.warning("ML model not loaded, cannot predict onsets")
            return None
        
        try:
            # Prepare audio for model input
            model_input = self._prepare_audio_for_model(audio)
            
            if model_input is None:
                return None
            
            # Make prediction
            predictions = self.model.predict(model_input, verbose=0)
            
            # Extract onset probabilities (assuming model outputs onset probabilities)
            if predictions.ndim == 3:
                # Shape: (batch, time, features)
                onset_probs = predictions[0, :, 0]  # Take first batch, first feature
            elif predictions.ndim == 2:
                # Shape: (batch, time) or (time, features)
                if predictions.shape[0] == 1:
                    onset_probs = predictions[0, :]  # Take first batch
                else:
                    onset_probs = predictions[:, 0]  # Take first feature
            else:
                onset_probs = predictions.flatten()
            
            return onset_probs
            
        except Exception as e:
            logger.error(f"ML onset prediction failed: {e}")
            return None
    
    def _prepare_audio_for_model(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """
        Prepare audio for ML model input.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Preprocessed audio ready for model or None if preparation fails
        """
        try:
            import librosa
            
            # Parameters for mel-spectrogram (adjust based on your model)
            sr = self.settings.audio.sample_rate
            hop_length = self.settings.audio.hop_length
            n_mels = 128
            n_fft = 2048
            
            # Compute mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                hop_length=hop_length,
                n_fft=n_fft,
                n_mels=n_mels,
                fmin=20,
                fmax=sr // 2
            )
            
            # Convert to dB scale
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize
            mel_spec_norm = (mel_spec_db - np.mean(mel_spec_db)) / (np.std(mel_spec_db) + 1e-8)
            
            # Transpose to get time x frequency format
            mel_spec_transposed = mel_spec_norm.T
            
            # Add batch and channel dimensions if needed
            # Shape: (1, time, frequency, 1) for most TensorFlow models
            model_input = mel_spec_transposed[np.newaxis, :, :, np.newaxis]
            
            return model_input.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Audio preparation for ML model failed: {e}")
            return None
    
    def classify_drum_window(self, audio_window: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Classify a single drum audio window.
        
        Args:
            audio_window: Audio window to classify
            
        Returns:
            Classification result or None if classification fails
        """
        if not self.is_loaded:
            return None
        
        try:
            # For classification, we might need a different model or different processing
            # This is a placeholder implementation
            
            # Prepare window for model
            model_input = self._prepare_audio_for_model(audio_window)
            
            if model_input is None:
                return None
            
            # Make prediction
            predictions = self.model.predict(model_input, verbose=0)
            
            # Interpret predictions as instrument probabilities
            # This depends on your specific model architecture
            if predictions.ndim >= 2:
                # Take mean over time dimension if present
                class_probs = np.mean(predictions, axis=1).flatten()
            else:
                class_probs = predictions.flatten()
            
            # Map to instrument classes (adjust based on your model)
            from ..config.constants import DRUM_CLASSES
            
            if len(class_probs) >= len(DRUM_CLASSES):
                # Find most likely class
                best_class_idx = np.argmax(class_probs[:len(DRUM_CLASSES)])
                confidence = float(class_probs[best_class_idx])
                instrument = DRUM_CLASSES[best_class_idx]
                
                # Estimate velocity based on audio energy
                rms_energy = np.sqrt(np.mean(audio_window**2))
                velocity = min(1.0, max(0.1, rms_energy * 10))
                
                return {
                    'instrument': instrument,
                    'confidence': confidence,
                    'velocity': velocity,
                    'class_id': best_class_idx
                }
            else:
                logger.warning("Model output dimension mismatch with drum classes")
                return None
            
        except Exception as e:
            logger.error(f"ML drum classification failed: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        info = {
            'model_path': self.model_path,
            'is_loaded': self.is_loaded,
            'model_exists': os.path.exists(self.model_path)
        }
        
        if self.is_loaded and self.model is not None:
            try:
                info.update({
                    'input_shape': self.model.input_shape,
                    'output_shape': self.model.output_shape,
                    'total_params': self.model.count_params(),
                    'trainable_params': sum([np.prod(layer.get_weights()[0].shape) 
                                           for layer in self.model.layers 
                                           if layer.get_weights()]),
                })
            except Exception as e:
                logger.warning(f"Could not get detailed model info: {e}")
        
        return info
    
    def is_available(self) -> bool:
        """
        Check if ML service is available for use.
        
        Returns:
            True if service is available
        """
        return self.is_loaded and self.model is not None
    
    def reload_model(self) -> bool:
        """
        Reload the ML model.
        
        Returns:
            True if reload was successful
        """
        self.model = None
        self.is_loaded = False
        self._load_model()
        return self.is_loaded
    
    def test_prediction(self) -> bool:
        """
        Test the model with dummy data.
        
        Returns:
            True if test prediction succeeds
        """
        if not self.is_available():
            return False
        
        try:
            # Create dummy audio data
            dummy_audio = np.random.randn(44100).astype(np.float32)  # 1 second
            
            # Test onset prediction
            onset_probs = self.predict_onsets(dummy_audio)
            if onset_probs is None:
                return False
            
            # Test classification
            dummy_window = np.random.randn(4410).astype(np.float32)  # 0.1 second
            classification = self.classify_drum_window(dummy_window)
            if classification is None:
                return False
            
            logger.info("ML service test prediction successful")
            return True
            
        except Exception as e:
            logger.error(f"ML service test prediction failed: {e}")
            return False