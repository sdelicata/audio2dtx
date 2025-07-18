"""
Track 8: Data Augmentation and Preprocessing

Advanced preprocessing and data augmentation for improved robustness and consistency 
across different recording conditions.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from scipy import signal
import librosa
import warnings
warnings.filterwarnings('ignore')

from ..base_classifier import BaseClassifier, ClassificationResult
from ..feature_extractor import FeatureExtractor
from ..base_track_mixin import BaseTrackMixin
from ...config.settings import Settings
from ...utils.exceptions import ClassificationError
from ...utils.logging import get_logger

logger = get_logger(__name__)


class AdvancedAudioPreprocessor:
    """Advanced audio preprocessing pipeline."""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.noise_profile = None
        
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Three-stage preprocessing pipeline.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Preprocessed audio signal
        """
        try:
            # Stage 1: Noise reduction
            denoised = self._spectral_subtraction(audio)
            
            # Stage 2: Dynamic range compression
            compressed = self._dynamic_range_compression(denoised)
            
            # Stage 3: Adaptive normalization
            normalized = self._adaptive_normalization(compressed)
            
            return normalized
            
        except Exception as e:
            logger.warning(f"Preprocessing failed: {e}")
            return audio
    
    def _spectral_subtraction(self, audio: np.ndarray) -> np.ndarray:
        """Spectral subtraction for noise reduction."""
        try:
            # Compute STFT
            stft = librosa.stft(audio, n_fft=2048, hop_length=512)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Estimate noise profile from quiet segments
            if self.noise_profile is None:
                self.noise_profile = self._estimate_noise_profile(magnitude)
            
            # Apply spectral subtraction
            alpha = 2.0  # Over-subtraction factor
            beta = 0.01  # Spectral floor
            
            enhanced_magnitude = magnitude - alpha * self.noise_profile[:, np.newaxis]
            enhanced_magnitude = np.maximum(enhanced_magnitude, beta * magnitude)
            
            # Reconstruct signal
            enhanced_stft = enhanced_magnitude * np.exp(1j * phase)
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=512)
            
            return enhanced_audio
            
        except Exception as e:
            logger.warning(f"Spectral subtraction failed: {e}")
            return audio
    
    def _estimate_noise_profile(self, magnitude: np.ndarray) -> np.ndarray:
        """Estimate noise profile from magnitude spectrogram."""
        try:
            # Use first 10% of frames as noise estimate
            noise_frames = int(0.1 * magnitude.shape[1])
            noise_profile = np.mean(magnitude[:, :noise_frames], axis=1)
            
            # Smooth the noise profile
            noise_profile = signal.medfilt(noise_profile, kernel_size=5)
            
            return noise_profile
            
        except Exception:
            return np.ones(magnitude.shape[0]) * 0.01
    
    def _dynamic_range_compression(self, audio: np.ndarray) -> np.ndarray:
        """Dynamic range compression with adaptive threshold."""
        try:
            # Calculate crest factor to determine compression parameters
            rms = np.sqrt(np.mean(audio**2))
            peak = np.max(np.abs(audio))
            crest_factor = peak / (rms + 1e-8)
            
            # Adaptive threshold based on crest factor
            if crest_factor > 10:
                threshold = 0.3
                ratio = 4.0
            elif crest_factor > 5:
                threshold = 0.4
                ratio = 3.0
            else:
                threshold = 0.5
                ratio = 2.0
            
            # Apply compression
            compressed = np.copy(audio)
            above_threshold = np.abs(audio) > threshold
            
            # Compress signals above threshold
            compressed[above_threshold] = np.sign(audio[above_threshold]) * (
                threshold + (np.abs(audio[above_threshold]) - threshold) / ratio
            )
            
            return compressed
            
        except Exception as e:
            logger.warning(f"Dynamic range compression failed: {e}")
            return audio
    
    def _adaptive_normalization(self, audio: np.ndarray) -> np.ndarray:
        """Adaptive normalization based on signal characteristics."""
        try:
            # Calculate signal statistics
            rms = np.sqrt(np.mean(audio**2))
            peak = np.max(np.abs(audio))
            
            # Determine target level based on signal characteristics
            if rms > 0.5:
                target_rms = 0.3
            elif rms > 0.2:
                target_rms = 0.2
            else:
                target_rms = 0.15
            
            # Normalize while preserving dynamics
            if rms > 0:
                gain = target_rms / rms
                normalized = audio * gain
                
                # Prevent clipping
                if np.max(np.abs(normalized)) > 0.95:
                    normalized = normalized * 0.95 / np.max(np.abs(normalized))
                
                return normalized
            else:
                return audio
                
        except Exception as e:
            logger.warning(f"Adaptive normalization failed: {e}")
            return audio


class AugmentationTrack(BaseClassifier, BaseTrackMixin):
    """
    Track 8: Data Augmentation and Preprocessing
    
    Advanced preprocessing and data augmentation for improved robustness and consistency
    across different recording conditions.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize augmentation track classifier.
        
        Args:
            settings: Application settings
        """
        super().__init__(settings)
        self.feature_extractor = FeatureExtractor(settings)
        self.preprocessor = AdvancedAudioPreprocessor(settings.audio.sample_rate)
        
        # Augmentation parameters
        self.pitch_shift_semitones = [-2, -1, 0, 1, 2]
        self.time_stretch_rates = [0.8, 0.9, 1.0, 1.1, 1.2]
        self.noise_levels = [0.0, 0.005, 0.01, 0.015, 0.02]
        
        # Ensemble weights (main processed audio gets higher weight)
        self.main_weight = 0.6
        self.augmentation_weight = 0.4
        
    def initialize(self) -> None:
        """Initialize the augmentation classifier."""
        try:
            logger.info("🔄 Initializing Track 8: Data Augmentation and Preprocessing")
            
            # Initialize feature extractor
            self.feature_extractor.clear_cache()
            
            self.is_initialized = True
            logger.info("✅ Augmentation track initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize augmentation track: {e}")
            raise ClassificationError(f"Augmentation track initialization failed: {e}")
    
    def classify_onsets(self, 
                       onsets: List[tuple],
                       context: Optional[Dict[str, Any]] = None) -> List[ClassificationResult]:
        """
        Classify multiple onsets using augmentation and preprocessing.
        
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
        
        logger.info(f"🔄 Augmentation track classified {len(results)} onsets")
        return results
    
    def classify_onset(self, 
                      audio_window: np.ndarray, 
                      onset_time: float,
                      context: Optional[Dict[str, Any]] = None) -> ClassificationResult:
        """
        Classify a drum onset using augmentation and preprocessing.
        
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
            
            # Stage 1: Advanced preprocessing
            preprocessed_audio = self.preprocessor.preprocess_audio(audio_window)
            
            # Stage 2: Generate augmented variants
            augmented_variants = self._generate_augmented_variants(preprocessed_audio)
            
            # Stage 3: Classify each variant
            variant_results = []
            
            # Classify main preprocessed audio
            main_result = self._classify_single_variant(preprocessed_audio, 'main')
            variant_results.append(('main', main_result))
            
            # Classify augmented variants
            for variant_name, variant_audio in augmented_variants:
                variant_result = self._classify_single_variant(variant_audio, variant_name)
                variant_results.append((variant_name, variant_result))
            
            # Stage 4: Ensemble voting across variants
            final_result = self._ensemble_voting_across_variants(variant_results)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Augmentation classification failed at {onset_time:.3f}s: {e}")
            return self._fallback_classification(audio_window)
    
    def _generate_augmented_variants(self, audio: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """Generate augmented variants of the audio."""
        variants = []
        
        try:
            # Pitch shifting variants
            for semitones in [-1, 1]:  # Limited set for performance
                try:
                    pitched = librosa.effects.pitch_shift(
                        audio, 
                        sr=self.settings.audio.sample_rate, 
                        n_steps=semitones
                    )
                    variants.append((f'pitch_{semitones:+d}', pitched))
                except Exception as e:
                    logger.warning(f"Pitch shift {semitones} failed: {e}")
            
            # Time stretching variants
            for rate in [0.9, 1.1]:  # Limited set for performance
                try:
                    stretched = librosa.effects.time_stretch(audio, rate=rate)
                    # Ensure same length as original
                    if len(stretched) > len(audio):
                        stretched = stretched[:len(audio)]
                    elif len(stretched) < len(audio):
                        stretched = np.pad(stretched, (0, len(audio) - len(stretched)), 'constant')
                    variants.append((f'time_{rate:.1f}', stretched))
                except Exception as e:
                    logger.warning(f"Time stretch {rate} failed: {e}")
            
            # Noise addition variants
            for noise_level in [0.005, 0.01]:  # Limited set for performance
                try:
                    noisy = self._add_noise(audio, noise_level)
                    variants.append((f'noise_{noise_level:.3f}', noisy))
                except Exception as e:
                    logger.warning(f"Noise addition {noise_level} failed: {e}")
            
            return variants
            
        except Exception as e:
            logger.warning(f"Augmentation generation failed: {e}")
            return []
    
    def _add_noise(self, audio: np.ndarray, noise_level: float) -> np.ndarray:
        """Add controlled noise to audio."""
        try:
            # Generate noise with same characteristics as signal
            noise = np.random.normal(0, noise_level, len(audio))
            
            # Apply noise shaping (reduce high frequency noise)
            if len(audio) > 100:
                b, a = signal.butter(4, 0.3, btype='low')
                noise = signal.filtfilt(b, a, noise)
            
            # Add noise to signal
            noisy_audio = audio + noise
            
            # Ensure no clipping
            if np.max(np.abs(noisy_audio)) > 1.0:
                noisy_audio = noisy_audio / np.max(np.abs(noisy_audio))
            
            return noisy_audio
            
        except Exception as e:
            logger.warning(f"Noise addition failed: {e}")
            return audio
    
    def _classify_single_variant(self, audio: np.ndarray, variant_name: str) -> ClassificationResult:
        """Classify a single audio variant."""
        try:
            # Extract features
            features = self.feature_extractor.extract_all_features(
                audio, 
                feature_set='advanced'
            )
            
            # Rule-based classification with augmentation-specific adaptations
            prediction, confidence = self._augmentation_aware_classification(features, variant_name)
            
            # Map to instrument name
            instrument = self._class_id_to_instrument(prediction)
            
            # Calculate velocity
            velocity = self._calculate_velocity(audio, features)
            
            return ClassificationResult(
                instrument=instrument,
                confidence=confidence,
                velocity=velocity,
                features={
                    'source': 'augmentation',
                    'variant': variant_name,
                    'preprocessing_applied': variant_name != 'original'
                }
            )
            
        except Exception as e:
            logger.warning(f"Single variant classification failed for {variant_name}: {e}")
            return self._fallback_classification_result(audio)
    
    def _augmentation_aware_classification(self, 
                                         features: Dict[str, Any],
                                         variant_name: str) -> Tuple[int, float]:
        """
        Augmentation-aware classification with adaptive thresholds.
        
        Args:
            features: Extracted features
            variant_name: Name of the variant being classified
            
        Returns:
            Tuple of (prediction, confidence)
        """
        try:
            # Extract key features
            spectral_centroid = features.get('spectral_centroid', 1000)
            rms_energy = features.get('rms_energy', 0.1)
            zero_crossing_rate = features.get('zero_crossing_rate', 0.1)
            
            # Frequency band ratios
            low_ratio = features.get('low_energy_ratio', 0.33)
            mid_ratio = features.get('mid_energy_ratio', 0.33)
            high_ratio = features.get('high_energy_ratio', 0.33)
            
            # Adjust thresholds based on variant type
            threshold_adjustment = self._get_threshold_adjustment(variant_name)
            
            # Enhanced rule-based classification with adaptive thresholds
            if low_ratio > (0.6 + threshold_adjustment['low']) and spectral_centroid < (150 + threshold_adjustment['centroid']):
                return 0, 0.7 + threshold_adjustment['confidence']  # kick
            elif mid_ratio > (0.4 + threshold_adjustment['mid']) and 200 < spectral_centroid < 1000:
                return 1, 0.65 + threshold_adjustment['confidence']  # snare
            elif high_ratio > (0.5 + threshold_adjustment['high']) and spectral_centroid > (2000 + threshold_adjustment['centroid']):
                if rms_energy > 0.25:
                    return 9, 0.6 + threshold_adjustment['confidence']  # crash
                else:
                    return 2, 0.6 + threshold_adjustment['confidence']  # hi-hat-close
            elif (2000 + threshold_adjustment['centroid']) < spectral_centroid < 3000 and high_ratio > 0.3:
                return 7, 0.55 + threshold_adjustment['confidence']  # ride-cymbal
            elif 1500 < spectral_centroid < 2500 and high_ratio > 0.4:
                return 8, 0.55 + threshold_adjustment['confidence']  # ride-bell
            elif 300 < spectral_centroid < 600:
                return 4, 0.5 + threshold_adjustment['confidence']  # tom-high
            elif 200 < spectral_centroid < 400:
                return 5, 0.5 + threshold_adjustment['confidence']  # tom-low
            elif 100 < spectral_centroid < 300:
                return 6, 0.5 + threshold_adjustment['confidence']  # tom-floor
            elif spectral_centroid > 3000 and high_ratio > 0.4:
                return 3, 0.45 + threshold_adjustment['confidence']  # hi-hat-open
            else:
                return 0, 0.4 + threshold_adjustment['confidence']  # default to kick
                
        except Exception as e:
            logger.warning(f"Augmentation-aware classification failed: {e}")
            return 0, 0.3
    
    def _get_threshold_adjustment(self, variant_name: str) -> Dict[str, float]:
        """Get threshold adjustments based on variant type."""
        adjustments = {
            'main': {
                'low': 0.0, 'mid': 0.0, 'high': 0.0,
                'centroid': 0.0, 'confidence': 0.0
            },
            'original': {
                'low': 0.0, 'mid': 0.0, 'high': 0.0,
                'centroid': 0.0, 'confidence': 0.0
            }
        }
        
        # Pitch-shifted variants
        if 'pitch_' in variant_name:
            semitones = int(variant_name.split('_')[1])
            adjustments[variant_name] = {
                'low': 0.0, 'mid': 0.0, 'high': 0.0,
                'centroid': semitones * 50,  # Adjust centroid thresholds
                'confidence': -0.05  # Slightly lower confidence for pitch-shifted
            }
        
        # Time-stretched variants
        elif 'time_' in variant_name:
            rate = float(variant_name.split('_')[1])
            adjustments[variant_name] = {
                'low': 0.0, 'mid': 0.0, 'high': 0.0,
                'centroid': 0.0,
                'confidence': -0.03  # Slightly lower confidence for time-stretched
            }
        
        # Noisy variants
        elif 'noise_' in variant_name:
            noise_level = float(variant_name.split('_')[1])
            adjustments[variant_name] = {
                'low': -0.05, 'mid': -0.05, 'high': -0.05,  # Relaxed thresholds
                'centroid': 0.0,
                'confidence': -0.1 * noise_level * 100  # Lower confidence for noisy
            }
        
        return adjustments.get(variant_name, adjustments['main'])
    
    def _ensemble_voting_across_variants(self, 
                                       variant_results: List[Tuple[str, ClassificationResult]]) -> ClassificationResult:
        """Ensemble voting across augmented variants."""
        try:
            if not variant_results:
                return self._fallback_classification_result(np.zeros(1024))
            
            # Separate main result from augmented variants
            main_result = None
            augmented_results = []
            
            for variant_name, result in variant_results:
                if variant_name == 'main':
                    main_result = result
                else:
                    augmented_results.append(result)
            
            # If no main result, use first available
            if main_result is None:
                main_result = variant_results[0][1]
            
            # Collect predictions and confidences
            predictions = [main_result.instrument]
            confidences = [main_result.confidence * self.main_weight]
            velocities = [main_result.velocity]
            
            # Add augmented results with lower weights
            if augmented_results:
                aug_weight = self.augmentation_weight / len(augmented_results)
                for result in augmented_results:
                    predictions.append(result.instrument)
                    confidences.append(result.confidence * aug_weight)
                    velocities.append(result.velocity)
            
            # Weighted voting
            from collections import Counter
            vote_counts = Counter()
            confidence_sums = {}
            velocity_sums = {}
            
            for pred, conf, vel in zip(predictions, confidences, velocities):
                vote_counts[pred] += conf
                if pred not in confidence_sums:
                    confidence_sums[pred] = []
                    velocity_sums[pred] = []
                confidence_sums[pred].append(conf)
                velocity_sums[pred].append(vel)
            
            # Find winner
            winner = vote_counts.most_common(1)[0][0]
            
            # Calculate final confidence and velocity
            final_confidence = sum(confidence_sums[winner]) / len(confidence_sums[winner])
            final_velocity = sum(velocity_sums[winner]) / len(velocity_sums[winner])
            
            # Boost confidence if multiple variants agree
            consensus_count = sum(1 for pred in predictions if pred == winner)
            if consensus_count > 1:
                consensus_boost = min(0.1 * (consensus_count - 1), 0.3)
                final_confidence = min(final_confidence + consensus_boost, 1.0)
            
            return ClassificationResult(
                instrument=winner,
                confidence=final_confidence,
                velocity=final_velocity,
                features={
                    'source': 'augmentation_ensemble',
                    'variant_count': len(variant_results),
                    'consensus_count': consensus_count,
                    'main_prediction': main_result.instrument,
                    'main_confidence': main_result.confidence,
                    'augmented_count': len(augmented_results)
                }
            )
            
        except Exception as e:
            logger.error(f"Ensemble voting failed: {e}")
            return main_result if main_result else self._fallback_classification_result(np.zeros(1024))
    
    def _calculate_velocity(self, audio: np.ndarray, features: Dict[str, Any]) -> float:
        """Calculate velocity based on audio energy and features."""
        try:
            rms_energy = features.get('rms_energy', np.sqrt(np.mean(audio**2)))
            peak_amplitude = np.max(np.abs(audio))
            spectral_centroid = features.get('spectral_centroid', 1000)
            
            # Base velocity from RMS energy
            base_velocity = min(rms_energy * 5, 1.0)
            
            # Adjust based on spectral characteristics
            if spectral_centroid > 3000:  # High frequency instruments (cymbals)
                velocity = base_velocity * 1.1
            elif spectral_centroid < 200:  # Low frequency instruments (kick)
                velocity = base_velocity * 0.9
            else:
                velocity = base_velocity
            
            # Combine with peak amplitude
            combined_velocity = (velocity * 0.7 + peak_amplitude * 0.3)
            
            return float(np.clip(combined_velocity, 0.1, 1.0))
            
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
    
    def _fallback_classification_result(self, audio: np.ndarray) -> ClassificationResult:
        """Fallback classification result."""
        return ClassificationResult(
            instrument='kick',
            confidence=0.2,
            velocity=0.5,
            features={'source': 'fallback'}
        )
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this track."""
        info = super().get_info()
        info.update({
            'track_number': 8,
            'track_name': 'Data Augmentation and Preprocessing',
            'description': 'Advanced preprocessing and augmentation for improved robustness',
            'augmentation_methods': {
                'pitch_shift': self.pitch_shift_semitones,
                'time_stretch': self.time_stretch_rates,
                'noise_addition': self.noise_levels
            },
            'ensemble_weights': {
                'main': self.main_weight,
                'augmentation': self.augmentation_weight
            },
            'preprocessing_stages': [
                'spectral_subtraction',
                'dynamic_range_compression',
                'adaptive_normalization'
            ]
        })
        return info
    
    def cleanup(self):
        """Clean up resources."""
        super().cleanup()
        if hasattr(self.feature_extractor, 'clear_cache'):
            self.feature_extractor.clear_cache()