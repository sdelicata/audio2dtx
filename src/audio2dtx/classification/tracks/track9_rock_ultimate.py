"""
Track 9: Ultimate Rock/Metal Hybrid

Ultimate optimization combining all tracks for maximum accuracy specifically tailored 
for rock and metal music genres.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

from ..base_classifier import BaseClassifier, ClassificationResult
from ..feature_extractor import FeatureExtractor
from ..base_track_mixin import BaseTrackMixin
from ...config.settings import Settings
from ...utils.exceptions import ClassificationError
from ...utils.logging import get_logger

logger = get_logger(__name__)


class RockPatternDetector:
    """Detector for rock/metal-specific drum patterns."""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.detected_patterns = {}
        
    def detect_patterns(self, 
                       onset_times: List[float], 
                       classified_onsets: List[ClassificationResult],
                       tempo_bpm: float) -> Dict[str, Any]:
        """
        Detect rock/metal-specific patterns.
        
        Args:
            onset_times: List of onset times
            classified_onsets: List of classification results
            tempo_bpm: Song tempo in BPM
            
        Returns:
            Dictionary of detected patterns
        """
        patterns = {
            'kick_snare_alternation': self._detect_kick_snare_alternation(onset_times, classified_onsets, tempo_bpm),
            'double_bass': self._detect_double_bass(onset_times, classified_onsets, tempo_bpm),
            'blast_beats': self._detect_blast_beats(onset_times, classified_onsets, tempo_bpm),
            'fill_patterns': self._detect_fill_patterns(onset_times, classified_onsets, tempo_bpm),
            'crash_emphasis': self._detect_crash_emphasis(onset_times, classified_onsets, tempo_bpm)
        }
        
        self.detected_patterns = patterns
        return patterns
    
    def _detect_kick_snare_alternation(self, 
                                     onset_times: List[float],
                                     classified_onsets: List[ClassificationResult],
                                     tempo_bpm: float) -> Dict[str, Any]:
        """Detect kick-snare alternation patterns."""
        try:
            beat_duration = 60.0 / tempo_bpm
            
            # Find kick and snare onsets
            kick_times = []
            snare_times = []
            
            for onset_time, result in zip(onset_times, classified_onsets):
                if result.instrument == 'kick':
                    kick_times.append(onset_time)
                elif result.instrument == 'snare':
                    snare_times.append(onset_time)
            
            # Check for alternation patterns
            alternation_score = 0
            pattern_count = 0
            
            # Look for kick-snare-kick-snare patterns
            for i in range(len(kick_times) - 1):
                kick_time = kick_times[i]
                next_kick_time = kick_times[i + 1]
                
                # Look for snare between these kicks
                snares_between = [s for s in snare_times if kick_time < s < next_kick_time]
                
                if len(snares_between) == 1:
                    # Check timing
                    snare_time = snares_between[0]
                    if abs((snare_time - kick_time) - beat_duration) < beat_duration * 0.2:
                        alternation_score += 1
                        pattern_count += 1
            
            alternation_ratio = alternation_score / max(len(kick_times), 1)
            
            return {
                'detected': alternation_ratio > 0.3,
                'confidence': min(alternation_ratio, 1.0),
                'pattern_count': pattern_count,
                'kick_count': len(kick_times),
                'snare_count': len(snare_times)
            }
            
        except Exception as e:
            logger.warning(f"Kick-snare alternation detection failed: {e}")
            return {'detected': False, 'confidence': 0.0, 'pattern_count': 0}
    
    def _detect_double_bass(self, 
                           onset_times: List[float],
                           classified_onsets: List[ClassificationResult],
                           tempo_bpm: float) -> Dict[str, Any]:
        """Detect double bass patterns."""
        try:
            # Find kick drum onsets
            kick_times = [onset_time for onset_time, result in zip(onset_times, classified_onsets) 
                         if result.instrument == 'kick']
            
            if len(kick_times) < 2:
                return {'detected': False, 'confidence': 0.0, 'pattern_count': 0}
            
            # Calculate inter-kick intervals
            intervals = [kick_times[i+1] - kick_times[i] for i in range(len(kick_times)-1)]
            
            # Look for rapid kick patterns (double bass)
            rapid_threshold = 0.25  # 250ms or faster
            rapid_intervals = [interval for interval in intervals if interval < rapid_threshold]
            
            double_bass_score = len(rapid_intervals) / len(intervals)
            
            # Count consecutive rapid patterns
            consecutive_patterns = 0
            current_pattern_length = 0
            
            for interval in intervals:
                if interval < rapid_threshold:
                    current_pattern_length += 1
                else:
                    if current_pattern_length >= 2:
                        consecutive_patterns += 1
                    current_pattern_length = 0
            
            # Final check
            if current_pattern_length >= 2:
                consecutive_patterns += 1
            
            return {
                'detected': double_bass_score > 0.2 and consecutive_patterns > 0,
                'confidence': min(double_bass_score * 2, 1.0),
                'pattern_count': consecutive_patterns,
                'rapid_kicks': len(rapid_intervals),
                'total_kicks': len(kick_times)
            }
            
        except Exception as e:
            logger.warning(f"Double bass detection failed: {e}")
            return {'detected': False, 'confidence': 0.0, 'pattern_count': 0}
    
    def _detect_blast_beats(self, 
                           onset_times: List[float],
                           classified_onsets: List[ClassificationResult],
                           tempo_bpm: float) -> Dict[str, Any]:
        """Detect blast beat patterns."""
        try:
            # Find kick and snare onsets
            kick_times = []
            snare_times = []
            
            for onset_time, result in zip(onset_times, classified_onsets):
                if result.instrument == 'kick':
                    kick_times.append(onset_time)
                elif result.instrument == 'snare':
                    snare_times.append(onset_time)
            
            blast_beat_score = 0
            pattern_count = 0
            
            # Look for simultaneous or very close kick+snare hits
            simultaneous_threshold = 0.05  # 50ms
            
            for kick_time in kick_times:
                close_snares = [s for s in snare_times if abs(s - kick_time) < simultaneous_threshold]
                if close_snares:
                    blast_beat_score += 1
                    pattern_count += 1
            
            blast_ratio = blast_beat_score / max(len(kick_times), 1)
            
            return {
                'detected': blast_ratio > 0.1 and pattern_count > 2,
                'confidence': min(blast_ratio * 5, 1.0),
                'pattern_count': pattern_count,
                'simultaneous_hits': blast_beat_score
            }
            
        except Exception as e:
            logger.warning(f"Blast beat detection failed: {e}")
            return {'detected': False, 'confidence': 0.0, 'pattern_count': 0}
    
    def _detect_fill_patterns(self, 
                             onset_times: List[float],
                             classified_onsets: List[ClassificationResult],
                             tempo_bpm: float) -> Dict[str, Any]:
        """Detect drum fill patterns."""
        try:
            # Find tom onsets
            tom_times = [onset_time for onset_time, result in zip(onset_times, classified_onsets) 
                        if 'tom' in result.instrument]
            
            if len(tom_times) < 3:
                return {'detected': False, 'confidence': 0.0, 'pattern_count': 0}
            
            # Look for clusters of tom hits (fills)
            fill_patterns = 0
            i = 0
            while i < len(tom_times) - 2:
                # Check if next 3 toms are close together
                if tom_times[i+2] - tom_times[i] < 2.0:  # Within 2 seconds
                    fill_patterns += 1
                    i += 3  # Skip ahead
                else:
                    i += 1
            
            fill_ratio = fill_patterns / max(len(tom_times) // 3, 1)
            
            return {
                'detected': fill_patterns > 0,
                'confidence': min(fill_ratio, 1.0),
                'pattern_count': fill_patterns,
                'tom_count': len(tom_times)
            }
            
        except Exception as e:
            logger.warning(f"Fill pattern detection failed: {e}")
            return {'detected': False, 'confidence': 0.0, 'pattern_count': 0}
    
    def _detect_crash_emphasis(self, 
                              onset_times: List[float],
                              classified_onsets: List[ClassificationResult],
                              tempo_bpm: float) -> Dict[str, Any]:
        """Detect crash emphasis patterns."""
        try:
            # Find crash onsets
            crash_times = [onset_time for onset_time, result in zip(onset_times, classified_onsets) 
                          if result.instrument == 'crash']
            
            if len(crash_times) < 2:
                return {'detected': False, 'confidence': 0.0, 'pattern_count': 0}
            
            beat_duration = 60.0 / tempo_bpm
            
            # Look for crashes on strong beats
            strong_beat_crashes = 0
            for crash_time in crash_times:
                # Check if crash aligns with strong beats (simplified)
                beat_position = (crash_time % (beat_duration * 4)) / beat_duration
                if beat_position < 0.2 or beat_position > 3.8:  # On beat 1 or 4
                    strong_beat_crashes += 1
            
            emphasis_ratio = strong_beat_crashes / len(crash_times)
            
            return {
                'detected': emphasis_ratio > 0.5,
                'confidence': emphasis_ratio,
                'pattern_count': strong_beat_crashes,
                'crash_count': len(crash_times)
            }
            
        except Exception as e:
            logger.warning(f"Crash emphasis detection failed: {e}")
            return {'detected': False, 'confidence': 0.0, 'pattern_count': 0}


class MetalFeatureEnhancer:
    """Feature enhancer for metal-specific characteristics."""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        
    def enhance_features(self, 
                        features: Dict[str, Any],
                        audio: np.ndarray) -> Dict[str, Any]:
        """
        Enhance features for metal characteristics.
        
        Args:
            features: Original features
            audio: Audio signal
            
        Returns:
            Enhanced features
        """
        enhanced = features.copy()
        
        try:
            # High-gain compensation
            enhanced.update(self._high_gain_compensation(features, audio))
            
            # Drop-tuning detection
            enhanced.update(self._drop_tuning_detection(features, audio))
            
            # Triggered drum analysis
            enhanced.update(self._triggered_drum_analysis(features, audio))
            
            # Frequency masking compensation
            enhanced.update(self._frequency_masking_compensation(features, audio))
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Feature enhancement failed: {e}")
            return features
    
    def _high_gain_compensation(self, features: Dict[str, Any], audio: np.ndarray) -> Dict[str, Any]:
        """Compensate for high-gain distortion effects."""
        try:
            # Calculate harmonic distortion indicator
            fft = np.fft.rfft(audio)
            freqs = np.fft.rfftfreq(len(audio), 1/self.sample_rate)
            magnitude = np.abs(fft)
            
            # Look for harmonic peaks
            fundamental_peaks = []
            for i in range(1, len(magnitude) // 4):
                if magnitude[i] > np.mean(magnitude) * 2:
                    fundamental_peaks.append(freqs[i])
            
            # Calculate distortion metric
            distortion_metric = len(fundamental_peaks) / max(len(magnitude) // 100, 1)
            
            return {
                'high_gain_distortion': min(distortion_metric, 1.0),
                'harmonic_peaks': len(fundamental_peaks),
                'distortion_compensated_centroid': features.get('spectral_centroid', 1000) * (1 - distortion_metric * 0.1)
            }
            
        except Exception:
            return {'high_gain_distortion': 0.0}
    
    def _drop_tuning_detection(self, features: Dict[str, Any], audio: np.ndarray) -> Dict[str, Any]:
        """Detect drop-tuning characteristics."""
        try:
            spectral_centroid = features.get('spectral_centroid', 1000)
            low_energy_ratio = features.get('low_energy_ratio', 0.33)
            
            # Drop tuning tends to emphasize low frequencies
            drop_tuning_indicator = low_energy_ratio * (1 - spectral_centroid / 2000)
            drop_tuning_indicator = max(0, min(drop_tuning_indicator, 1.0))
            
            return {
                'drop_tuning_indicator': drop_tuning_indicator,
                'drop_compensated_low_ratio': low_energy_ratio * (1 - drop_tuning_indicator * 0.2)
            }
            
        except Exception:
            return {'drop_tuning_indicator': 0.0}
    
    def _triggered_drum_analysis(self, features: Dict[str, Any], audio: np.ndarray) -> Dict[str, Any]:
        """Analyze triggered drum characteristics."""
        try:
            # Triggered drums have very sharp attacks and consistent levels
            attack_sharpness = self._calculate_attack_sharpness(audio)
            level_consistency = self._calculate_level_consistency(audio)
            
            triggered_probability = (attack_sharpness + level_consistency) / 2
            
            return {
                'triggered_probability': triggered_probability,
                'attack_sharpness': attack_sharpness,
                'level_consistency': level_consistency
            }
            
        except Exception:
            return {'triggered_probability': 0.0}
    
    def _calculate_attack_sharpness(self, audio: np.ndarray) -> float:
        """Calculate attack sharpness."""
        try:
            # Find the peak and measure rise time
            peak_idx = np.argmax(np.abs(audio))
            
            # Look at the 10ms before peak
            lookback_samples = int(0.01 * self.sample_rate)
            start_idx = max(0, peak_idx - lookback_samples)
            
            attack_portion = audio[start_idx:peak_idx+1]
            
            if len(attack_portion) > 1:
                # Calculate rise rate
                rise_rate = (np.max(np.abs(attack_portion)) - np.min(np.abs(attack_portion))) / len(attack_portion)
                return min(rise_rate * 1000, 1.0)  # Normalize
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_level_consistency(self, audio: np.ndarray) -> float:
        """Calculate level consistency."""
        try:
            # Calculate RMS in small windows
            window_size = int(0.005 * self.sample_rate)  # 5ms windows
            rms_values = []
            
            for i in range(0, len(audio) - window_size, window_size):
                window = audio[i:i+window_size]
                rms = np.sqrt(np.mean(window**2))
                rms_values.append(rms)
            
            if len(rms_values) > 1:
                # Calculate consistency (inverse of coefficient of variation)
                consistency = 1.0 - (np.std(rms_values) / (np.mean(rms_values) + 1e-8))
                return max(0, min(consistency, 1.0))
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _frequency_masking_compensation(self, features: Dict[str, Any], audio: np.ndarray) -> Dict[str, Any]:
        """Compensate for frequency masking effects."""
        try:
            # In metal, heavy guitar can mask drum frequencies
            # Adjust frequency-based features accordingly
            
            spectral_centroid = features.get('spectral_centroid', 1000)
            high_ratio = features.get('high_energy_ratio', 0.33)
            
            # If high frequencies are suppressed, boost sensitivity
            masking_compensation = 1.0 - (high_ratio * 0.3)
            
            return {
                'frequency_masking_compensation': masking_compensation,
                'masked_adjusted_centroid': spectral_centroid * masking_compensation
            }
            
        except Exception:
            return {'frequency_masking_compensation': 1.0}


class RockUltimateTrack(BaseClassifier, BaseTrackMixin):
    """
    Track 9: Ultimate Rock/Metal Hybrid
    
    Ultimate optimization combining all tracks for maximum accuracy specifically 
    tailored for rock and metal music genres.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize ultimate rock/metal track classifier.
        
        Args:
            settings: Application settings
        """
        super().__init__(settings)
        self.feature_extractor = FeatureExtractor(settings)
        self.pattern_detector = RockPatternDetector(settings.audio.sample_rate)
        self.metal_enhancer = MetalFeatureEnhancer(settings.audio.sample_rate)
        
        # Import track managers dynamically
        self.track_managers = {}
        self.available_tracks = ['magenta_only', 'advanced_features', 'multi_scale', 'few_shot', 'ensemble']
        
        # Ultimate voting weights (rock/metal optimized)
        self.ultimate_weights = {
            'ensemble': 0.35,        # Hierarchical expertise
            'advanced_features': 0.25,  # Rich feature analysis
            'multi_scale': 0.20,     # Temporal precision
            'few_shot': 0.15,        # Adaptation
            'magenta_only': 0.05     # Consistency baseline
        }
        
        # Rock pattern bonuses
        self.pattern_bonuses = {
            'kick_snare_alternation': 0.1,
            'double_bass': 0.15,
            'blast_beats': 0.2,
            'fill_patterns': 0.05,
            'crash_emphasis': 0.1
        }
        
        # Processing statistics
        self.processing_stats = {
            'total_onsets': 0,
            'pattern_detections': {},
            'track_votes': defaultdict(int),
            'ultimate_confidence': 0.0
        }
    
    def initialize(self) -> None:
        """Initialize the ultimate rock/metal classifier."""
        try:
            logger.info("🎸 Initializing Track 9: Ultimate Rock/Metal Hybrid")
            
            # Initialize feature extractor
            self.feature_extractor.clear_cache()
            
            # Initialize track managers
            self._initialize_track_managers()
            
            self.is_initialized = True
            logger.info("✅ Ultimate rock/metal track initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ultimate rock/metal track: {e}")
            raise ClassificationError(f"Ultimate rock/metal track initialization failed: {e}")
    
    def _initialize_track_managers(self):
        """Initialize individual track managers."""
        try:
            # Import track classes dynamically
            from .track3_magenta import MagentaTrack
            from .track4_advanced import AdvancedFeaturesTrack
            from .track5_multiscale import MultiScaleTrack
            from .track6_fewshot import FewShotTrack
            from .track7_ensemble import EnsembleTrack
            
            # Initialize each track
            track_classes = {
                'magenta_only': MagentaTrack,
                'advanced_features': AdvancedFeaturesTrack,
                'multi_scale': MultiScaleTrack,
                'few_shot': FewShotTrack,
                'ensemble': EnsembleTrack
            }
            
            for track_name, track_class in track_classes.items():
                try:
                    track_instance = track_class(self.settings)
                    track_instance.initialize()
                    self.track_managers[track_name] = track_instance
                    logger.info(f"✅ Initialized {track_name} for ultimate hybrid")
                except Exception as e:
                    logger.warning(f"Failed to initialize {track_name}: {e}")
            
            logger.info(f"🎸 Ultimate hybrid initialized with {len(self.track_managers)} tracks")
            
        except Exception as e:
            logger.error(f"Track manager initialization failed: {e}")
            raise
    
    def classify_onsets(self, 
                       onsets: List[tuple],
                       context: Optional[Dict[str, Any]] = None) -> List[ClassificationResult]:
        """
        Classify multiple onsets using ultimate rock/metal optimization.
        
        Args:
            onsets: List of (audio_window, onset_time) tuples
            context: Additional context information
            
        Returns:
            List of classification results
        """
        if not self.is_initialized:
            self.initialize()
        
        logger.info(f"🎸 Ultimate rock/metal hybrid processing {len(onsets)} onsets")
        
        # Phase 1: Multi-Track Classification
        logger.info("🎯 Phase 1: Running all tracks in parallel")
        track_results = self._run_all_tracks(onsets, context)
        
        # Phase 2: Pattern Detection
        logger.info("🔍 Phase 2: Detecting rock/metal patterns")
        initial_results = self._get_initial_classification(track_results)
        patterns = self._detect_rock_patterns(onsets, initial_results, context)
        
        # Phase 3: Ultimate Voting with Rock Bonuses
        logger.info("🗳️  Phase 3: Ultimate voting with rock/metal bonuses")
        final_results = self._ultimate_voting_with_bonuses(track_results, patterns, onsets)
        
        # Phase 4: Final Validation and Statistics
        logger.info("📊 Phase 4: Final validation and statistics")
        validated_results = self._validate_and_finalize(final_results, patterns)
        
        self.processing_stats['total_onsets'] = len(onsets)
        self.processing_stats['pattern_detections'] = patterns
        
        logger.info(f"🎸 Ultimate rock/metal hybrid completed with {len(validated_results)} classifications")
        return validated_results
    
    def classify_onset(self, 
                      audio_window: np.ndarray, 
                      onset_time: float,
                      context: Optional[Dict[str, Any]] = None) -> ClassificationResult:
        """
        Classify a single drum onset using ultimate rock/metal optimization.
        
        Args:
            audio_window: Audio data around the onset
            onset_time: Time of the onset in seconds
            context: Additional context information
            
        Returns:
            Classification result
        """
        # For single onset, use simplified approach
        return self.classify_onsets([(audio_window, onset_time)], context)[0]
    
    def _run_all_tracks(self, 
                       onsets: List[tuple],
                       context: Optional[Dict[str, Any]] = None) -> Dict[str, List[ClassificationResult]]:
        """Run all available tracks in parallel."""
        track_results = {}
        
        for track_name, track_instance in self.track_managers.items():
            try:
                logger.info(f"🎵 Running {track_name}...")
                results = track_instance.classify_onsets(onsets, context)
                track_results[track_name] = results
                logger.info(f"✅ {track_name} completed with {len(results)} results")
                
                # Update statistics
                self.processing_stats['track_votes'][track_name] = len(results)
                
            except Exception as e:
                logger.error(f"❌ {track_name} failed: {e}")
                # Continue with other tracks
                continue
        
        return track_results
    
    def _get_initial_classification(self, track_results: Dict[str, List[ClassificationResult]]) -> List[ClassificationResult]:
        """Get initial classification for pattern detection."""
        if not track_results:
            return []
        
        # Use the first available track's results for pattern detection
        first_track = list(track_results.keys())[0]
        return track_results[first_track]
    
    def _detect_rock_patterns(self, 
                             onsets: List[tuple],
                             initial_results: List[ClassificationResult],
                             context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Detect rock/metal-specific patterns."""
        try:
            # Extract onset times
            onset_times = [onset_time for _, onset_time in onsets]
            
            # Get tempo from context
            tempo_bpm = 120.0
            if context:
                tempo_bpm = context.get('tempo_bpm', 120.0)
            
            # Detect patterns
            patterns = self.pattern_detector.detect_patterns(onset_times, initial_results, tempo_bpm)
            
            # Log detected patterns
            detected_patterns = [name for name, info in patterns.items() if info.get('detected', False)]
            if detected_patterns:
                logger.info(f"🎸 Detected rock/metal patterns: {', '.join(detected_patterns)}")
            else:
                logger.info("🎸 No specific rock/metal patterns detected")
            
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
            return {}
    
    def _ultimate_voting_with_bonuses(self, 
                                    track_results: Dict[str, List[ClassificationResult]],
                                    patterns: Dict[str, Any],
                                    onsets: List[tuple]) -> List[ClassificationResult]:
        """Ultimate voting system with rock/metal-specific bonuses."""
        try:
            if not track_results:
                logger.warning("No track results available for voting")
                return []
            
            # Get onset count
            onset_count = len(onsets)
            if onset_count == 0:
                return []
            
            # Ensure all tracks have the same number of results
            min_results = min(len(results) for results in track_results.values())
            onset_count = min(onset_count, min_results)
            
            ultimate_results = []
            
            for onset_idx in range(onset_count):
                # Collect votes for this onset
                votes = defaultdict(float)
                total_weight = 0
                
                for track_name, results in track_results.items():
                    if onset_idx < len(results):
                        result = results[onset_idx]
                        track_weight = self.ultimate_weights.get(track_name, 0.1)
                        
                        # Base vote weight
                        vote_weight = track_weight * result.confidence
                        
                        # Apply rock/metal pattern bonuses
                        pattern_bonus = self._calculate_pattern_bonus(result.instrument, patterns)
                        vote_weight += pattern_bonus
                        
                        votes[result.instrument] += vote_weight
                        total_weight += vote_weight
                
                # Find winner
                if votes:
                    winner = max(votes, key=votes.get)
                    winner_score = votes[winner]
                    
                    # Calculate ultimate confidence
                    if total_weight > 0:
                        ultimate_confidence = winner_score / total_weight
                    else:
                        ultimate_confidence = 0.5
                    
                    # Calculate average velocity from participating tracks
                    velocities = []
                    for track_name, results in track_results.items():
                        if onset_idx < len(results) and results[onset_idx].instrument == winner:
                            velocities.append(results[onset_idx].velocity)
                    
                    avg_velocity = np.mean(velocities) if velocities else 0.5
                    
                    # Create ultimate result
                    ultimate_result = ClassificationResult(
                        instrument=winner,
                        confidence=float(ultimate_confidence),
                        velocity=float(avg_velocity),
                        features={
                            'source': 'rock_ultimate',
                            'participating_tracks': list(track_results.keys()),
                            'vote_distribution': dict(votes),
                            'total_weight': total_weight,
                            'pattern_bonuses': self._get_pattern_bonus_summary(winner, patterns),
                            'ultimate_confidence': ultimate_confidence
                        }
                    )
                    
                    ultimate_results.append(ultimate_result)
                else:
                    # Fallback result
                    ultimate_results.append(ClassificationResult(
                        instrument='kick',
                        confidence=0.1,
                        velocity=0.5,
                        features={'source': 'ultimate_fallback'}
                    ))
            
            return ultimate_results
            
        except Exception as e:
            logger.error(f"Ultimate voting failed: {e}")
            # Return first available track's results as fallback
            if track_results:
                first_track = list(track_results.keys())[0]
                return track_results[first_track]
            return []
    
    def _calculate_pattern_bonus(self, instrument: str, patterns: Dict[str, Any]) -> float:
        """Calculate pattern bonus for an instrument."""
        total_bonus = 0.0
        
        try:
            # Kick-snare alternation bonus
            if patterns.get('kick_snare_alternation', {}).get('detected', False):
                if instrument in ['kick', 'snare']:
                    confidence = patterns['kick_snare_alternation'].get('confidence', 0.0)
                    total_bonus += self.pattern_bonuses['kick_snare_alternation'] * confidence
            
            # Double bass bonus
            if patterns.get('double_bass', {}).get('detected', False):
                if instrument == 'kick':
                    confidence = patterns['double_bass'].get('confidence', 0.0)
                    total_bonus += self.pattern_bonuses['double_bass'] * confidence
            
            # Blast beats bonus
            if patterns.get('blast_beats', {}).get('detected', False):
                if instrument in ['kick', 'snare']:
                    confidence = patterns['blast_beats'].get('confidence', 0.0)
                    total_bonus += self.pattern_bonuses['blast_beats'] * confidence
            
            # Fill patterns bonus
            if patterns.get('fill_patterns', {}).get('detected', False):
                if 'tom' in instrument:
                    confidence = patterns['fill_patterns'].get('confidence', 0.0)
                    total_bonus += self.pattern_bonuses['fill_patterns'] * confidence
            
            # Crash emphasis bonus
            if patterns.get('crash_emphasis', {}).get('detected', False):
                if instrument == 'crash':
                    confidence = patterns['crash_emphasis'].get('confidence', 0.0)
                    total_bonus += self.pattern_bonuses['crash_emphasis'] * confidence
            
            return total_bonus
            
        except Exception as e:
            logger.warning(f"Pattern bonus calculation failed: {e}")
            return 0.0
    
    def _get_pattern_bonus_summary(self, instrument: str, patterns: Dict[str, Any]) -> Dict[str, float]:
        """Get summary of pattern bonuses applied."""
        bonuses = {}
        
        for pattern_name, pattern_info in patterns.items():
            if pattern_info.get('detected', False):
                bonus = self._calculate_pattern_bonus(instrument, {pattern_name: pattern_info})
                if bonus > 0:
                    bonuses[pattern_name] = bonus
        
        return bonuses
    
    def _validate_and_finalize(self, 
                             results: List[ClassificationResult],
                             patterns: Dict[str, Any]) -> List[ClassificationResult]:
        """Validate and finalize results."""
        try:
            # Calculate overall statistics
            if results:
                confidences = [r.confidence for r in results]
                avg_confidence = np.mean(confidences)
                self.processing_stats['ultimate_confidence'] = avg_confidence
                
                # Log final statistics
                logger.info(f"🎸 Ultimate rock/metal statistics:")
                logger.info(f"  Average confidence: {avg_confidence:.3f}")
                logger.info(f"  Total classifications: {len(results)}")
                
                # Log pattern statistics
                pattern_summary = {name: info.get('detected', False) for name, info in patterns.items()}
                detected_count = sum(pattern_summary.values())
                logger.info(f"  Detected patterns: {detected_count}/5")
                
                # Log instrument distribution
                instrument_counts = Counter(r.instrument for r in results)
                logger.info(f"  Instrument distribution: {dict(instrument_counts)}")
            
            return results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return results
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get detailed processing statistics."""
        return self.processing_stats.copy()
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about this track."""
        info = super().get_info()
        info.update({
            'track_number': 9,
            'track_name': 'Ultimate Rock/Metal Hybrid',
            'description': 'Ultimate optimization combining all tracks for maximum rock/metal accuracy',
            'available_tracks': self.available_tracks,
            'initialized_tracks': list(self.track_managers.keys()),
            'ultimate_weights': self.ultimate_weights,
            'pattern_bonuses': self.pattern_bonuses,
            'processing_stats': self.processing_stats,
            'rock_patterns': [
                'kick_snare_alternation',
                'double_bass',
                'blast_beats',
                'fill_patterns',
                'crash_emphasis'
            ]
        })
        return info
    
    def cleanup(self):
        """Clean up resources."""
        super().cleanup()
        if hasattr(self.feature_extractor, 'clear_cache'):
            self.feature_extractor.clear_cache()
        
        # Clean up track managers
        for track_name, track_instance in self.track_managers.items():
            try:
                track_instance.cleanup()
                logger.debug(f"Cleaned up {track_name}")
            except Exception as e:
                logger.warning(f"Cleanup failed for {track_name}: {e}")
        
        self.track_managers.clear()
        
        # Clear statistics
        self.processing_stats.clear()