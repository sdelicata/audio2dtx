"""
Track 3: Magenta-Only Classification

Simplified approach using only Google's Magenta OaF (Onsets and Frames) 
Drums model for classification.
"""

import numpy as np
from typing import Dict, Any, Optional

from ..base_classifier import BaseClassifier, ClassificationResult
from ...config.settings import Settings
from ...services.magenta_client import MagentaClient
from ...utils.exceptions import ClassificationError
from ...utils.logging import get_logger

logger = get_logger(__name__)


class MagentaTrack(BaseClassifier):
    """
    Track 3: Magenta-Only Classification
    
    Uses only the Magenta service for drum classification, eliminating
    the complexity of hybrid voting systems.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize Magenta track classifier.
        
        Args:
            settings: Application settings
        """
        super().__init__(settings)
        self.magenta_client = MagentaClient(settings)
        self.confidence_threshold = settings.classification.confidence_threshold
        self.fallback_enabled = True
        
        # Instrument mapping for Magenta results
        self.instrument_map = {
            'kick': 2, 'bass': 2, 'bass_drum': 2,
            'snare': 1, 'snare_drum': 1,
            'hi-hat': 0, 'hihat': 0, 'hi_hat': 0, 'hi-hat-close': 0,
            'hi-hat-open': 7, 'hihat_open': 7, 'open_hihat': 7,
            'tom': 3, 'tom_high': 3, 'high_tom': 3,
            'tom_low': 4, 'low_tom': 4,
            'tom_floor': 6, 'floor_tom': 6,
            'ride': 5, 'ride_cymbal': 5,
            'ride_bell': 8, 'bell': 8,
            'crash': 9, 'crash_cymbal': 9
        }
        
    def initialize(self) -> None:
        """Initialize the Magenta classifier."""
        try:
            logger.info("🔮 Initializing Track 3: Magenta-Only Classification")
            
            # Test Magenta service availability
            if self.magenta_client.check_health():
                logger.info("✅ Magenta service is available and ready")
                self.is_initialized = True
            else:
                logger.warning("⚠️ Magenta service not available, will use fallback")
                if self.fallback_enabled:
                    self.is_initialized = True
                else:
                    raise ClassificationError("Magenta service required but not available")
                    
        except Exception as e:
            logger.error(f"Failed to initialize Magenta track: {e}")
            if self.fallback_enabled:
                logger.info("Using fallback classification")
                self.is_initialized = True
            else:
                raise ClassificationError(f"Magenta track initialization failed: {e}")
    
    def classify_onset(self, 
                      audio_window: np.ndarray, 
                      onset_time: float,
                      context: Optional[Dict[str, Any]] = None) -> ClassificationResult:
        """
        Classify a drum onset using Magenta service.
        
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
            
            # Try Magenta classification first
            if self.magenta_client.is_available():
                magenta_result = self.magenta_client.classify_drums(audio_window)
                
                if magenta_result and magenta_result.get('confidence', 0) > self.confidence_threshold:
                    # Map Magenta result to our class system
                    instrument = magenta_result.get('instrument', 'kick')
                    confidence = magenta_result.get('confidence', 0.5)
                    velocity = magenta_result.get('velocity', 0.5)
                    
                    # Map to standardized instrument name
                    mapped_instrument = self._map_magenta_instrument(instrument)
                    
                    return ClassificationResult(
                        instrument=mapped_instrument,
                        confidence=confidence,
                        velocity=velocity,
                        features={'source': 'magenta', 'original_instrument': instrument}
                    )
                else:
                    logger.debug(f"Low confidence Magenta result at {onset_time:.3f}s")
            
            # Fallback to frequency-based classification
            return self._fallback_classification(audio_window)
            
        except Exception as e:
            logger.error(f"Magenta classification failed at {onset_time:.3f}s: {e}")
            return self._fallback_classification(audio_window)
    
    def _map_magenta_instrument(self, magenta_instrument: str) -> str:
        """
        Map Magenta instrument name to standardized instrument name.
        
        Args:
            magenta_instrument: Instrument name from Magenta
            
        Returns:
            Standardized instrument name
        """
        # Direct mapping to our standard names
        standard_mapping = {
            'kick': 'kick',
            'bass': 'kick',
            'bass_drum': 'kick',
            'snare': 'snare',
            'snare_drum': 'snare',
            'hi-hat': 'hi-hat-close',
            'hihat': 'hi-hat-close',
            'hi_hat': 'hi-hat-close',
            'hi-hat-close': 'hi-hat-close',
            'hi-hat-open': 'hi-hat-open',
            'hihat_open': 'hi-hat-open',
            'open_hihat': 'hi-hat-open',
            'tom': 'tom-high',
            'tom_high': 'tom-high',
            'high_tom': 'tom-high',
            'tom_low': 'tom-low',
            'low_tom': 'tom-low',
            'tom_floor': 'tom-floor',
            'floor_tom': 'tom-floor',
            'ride': 'ride',
            'ride_cymbal': 'ride',
            'ride_bell': 'ride-bell',
            'bell': 'ride-bell',
            'crash': 'crash',
            'crash_cymbal': 'crash'
        }
        
        return standard_mapping.get(magenta_instrument.lower(), 'kick')
    
    def _fallback_classification(self, audio_window: np.ndarray) -> ClassificationResult:
        """
        Fallback frequency-based classification when Magenta is not available.
        
        Args:
            audio_window: Audio data to classify
            
        Returns:
            Classification result
        """
        try:
            # Simple FFT-based frequency analysis
            fft = np.fft.rfft(audio_window)
            freqs = np.fft.rfftfreq(len(audio_window), 1/self.settings.audio.sample_rate)
            magnitude = np.abs(fft)
            
            # Find peak frequency
            peak_idx = np.argmax(magnitude)
            peak_freq = freqs[peak_idx]
            
            # Calculate energy distribution
            total_energy = np.sum(magnitude)
            low_energy = np.sum(magnitude[freqs <= 200])
            mid_energy = np.sum(magnitude[(freqs > 200) & (freqs <= 2000)])
            high_energy = np.sum(magnitude[freqs > 2000])
            
            # Energy ratios
            low_ratio = low_energy / total_energy if total_energy > 0 else 0
            mid_ratio = mid_energy / total_energy if total_energy > 0 else 0
            high_ratio = high_energy / total_energy if total_energy > 0 else 0
            
            # Classification based on frequency characteristics
            if peak_freq < 100 and low_ratio > 0.6:
                instrument = 'kick'
                confidence = 0.7
            elif 200 < peak_freq < 800 and mid_ratio > 0.4:
                instrument = 'snare'
                confidence = 0.6
            elif peak_freq > 3000 and high_ratio > 0.5:
                if high_ratio > 0.7:
                    instrument = 'hi-hat-close'
                    confidence = 0.65
                else:
                    instrument = 'crash'
                    confidence = 0.6
            elif 100 < peak_freq < 300 and low_ratio > 0.3:
                instrument = 'tom-low'
                confidence = 0.5
            elif 300 < peak_freq < 800 and mid_ratio > 0.3:
                instrument = 'tom-high'
                confidence = 0.5
            elif 800 < peak_freq < 3000 and mid_ratio > 0.3:
                instrument = 'ride'
                confidence = 0.55
            else:
                # Default to kick
                instrument = 'kick'
                confidence = 0.4
            
            # Calculate velocity based on RMS energy
            rms = np.sqrt(np.mean(audio_window ** 2))
            velocity = min(1.0, max(0.1, rms * 10))
            
            return ClassificationResult(
                instrument=instrument,
                confidence=confidence,
                velocity=velocity,
                features={
                    'source': 'fallback',
                    'peak_frequency': float(peak_freq),
                    'low_ratio': float(low_ratio),
                    'mid_ratio': float(mid_ratio),
                    'high_ratio': float(high_ratio)
                }
            )
            
        except Exception as e:
            logger.error(f"Fallback classification failed: {e}")
            # Ultimate fallback
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
            'track_number': 3,
            'track_name': 'Magenta-Only Classification',
            'description': 'Simplified approach using only Magenta OaF Drums model',
            'magenta_available': self.magenta_client.is_available(),
            'fallback_enabled': self.fallback_enabled,
            'confidence_threshold': self.confidence_threshold
        })
        return info
    
    def cleanup(self):
        """Clean up resources."""
        super().cleanup()
        # Magenta client doesn't need explicit cleanup