"""
Beat tracking and tempo analysis interface.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Optional, Any
import numpy as np
import librosa
from dataclasses import dataclass

from ..config.settings import Settings
from ..utils.exceptions import ProcessingError
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BeatTrackingResult:
    """Result of beat tracking analysis."""
    tempo_bpm: float
    beat_times: np.ndarray
    bar_times: Optional[np.ndarray] = None
    confidence: float = 0.0
    method: str = "librosa"


class BaseBeatTracker(ABC):
    """Abstract base class for beat tracking algorithms."""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.sr = settings.audio.sample_rate
        self.hop_length = settings.audio.hop_length
        
    @abstractmethod
    def track_beats(self, audio: np.ndarray) -> BeatTrackingResult:
        """
        Track beats in audio signal.
        
        Args:
            audio: Input audio signal
            
        Returns:
            Beat tracking result
        """
        pass


class LibrosaBeatTracker(BaseBeatTracker):
    """Librosa-based beat tracking."""
    
    def track_beats(self, audio: np.ndarray) -> BeatTrackingResult:
        """Track beats using librosa."""
        try:
            # Estimate tempo and beat positions
            tempo, beat_frames = librosa.beat.beat_track(
                y=audio,
                sr=self.sr,
                hop_length=self.hop_length,
                units='frames'
            )
            
            # Convert beat frames to time
            beat_times = librosa.frames_to_time(
                beat_frames, sr=self.sr, hop_length=self.hop_length
            )
            
            # Calculate bar positions (assuming 4/4 time)
            if len(beat_times) >= 4:
                bar_times = beat_times[::4]  # Every 4th beat
            else:
                bar_times = np.array([0.0])
            
            # Estimate confidence based on beat consistency
            if len(beat_times) > 1:
                beat_intervals = np.diff(beat_times)
                consistency = 1.0 - (np.std(beat_intervals) / np.mean(beat_intervals))
                confidence = max(0.0, min(1.0, consistency))
            else:
                confidence = 0.0
            
            return BeatTrackingResult(
                tempo_bpm=float(tempo),
                beat_times=beat_times,
                bar_times=bar_times,
                confidence=confidence,
                method="librosa"
            )
            
        except Exception as e:
            logger.error(f"Librosa beat tracking failed: {e}")
            # Return default result
            return BeatTrackingResult(
                tempo_bpm=120.0,
                beat_times=np.array([]),
                bar_times=np.array([0.0]),
                confidence=0.0,
                method="librosa_fallback"
            )


class AdaptiveBeatTracker(BaseBeatTracker):
    """Adaptive beat tracking with tempo curve analysis."""
    
    def track_beats(self, audio: np.ndarray) -> BeatTrackingResult:
        """Track beats with adaptive tempo analysis."""
        try:
            # Get onset strength
            onset_envelope = librosa.onset.onset_strength(
                y=audio, sr=self.sr, hop_length=self.hop_length
            )
            
            # Estimate dynamic tempo
            tempo_curve = librosa.beat.tempo(
                onset_envelope=onset_envelope,
                sr=self.sr,
                hop_length=self.hop_length,
                aggregate=None  # Get tempo curve
            )
            
            # Get static tempo estimate
            static_tempo = librosa.beat.tempo(
                onset_envelope=onset_envelope,
                sr=self.sr,
                hop_length=self.hop_length
            )[0]
            
            # Track beats with dynamic programming
            beat_frames = librosa.beat.beat_track(
                onset_envelope=onset_envelope,
                sr=self.sr,
                hop_length=self.hop_length,
                start_bpm=static_tempo,
                units='frames'
            )[1]
            
            # Convert to time
            beat_times = librosa.frames_to_time(
                beat_frames, sr=self.sr, hop_length=self.hop_length
            )
            
            # Calculate adaptive bar positions
            bar_times = self._calculate_adaptive_bars(beat_times, static_tempo)
            
            # Calculate confidence based on tempo stability
            if len(tempo_curve) > 1:
                tempo_stability = 1.0 - (np.std(tempo_curve) / np.mean(tempo_curve))
                confidence = max(0.0, min(1.0, tempo_stability))
            else:
                confidence = 0.5
            
            return BeatTrackingResult(
                tempo_bpm=float(static_tempo),
                beat_times=beat_times,
                bar_times=bar_times,
                confidence=confidence,
                method="adaptive"
            )
            
        except Exception as e:
            logger.error(f"Adaptive beat tracking failed: {e}")
            # Fallback to librosa
            fallback_tracker = LibrosaBeatTracker(self.settings)
            return fallback_tracker.track_beats(audio)
    
    def _calculate_adaptive_bars(self, beat_times: np.ndarray, tempo_bpm: float) -> np.ndarray:
        """Calculate bar positions with adaptive timing."""
        if len(beat_times) < 4:
            return np.array([0.0])
        
        # Estimate beats per bar based on tempo and common time signatures
        if tempo_bpm > 140:
            # Likely 4/4 at high tempo or 2/4
            beats_per_bar = 4
        elif tempo_bpm < 80:
            # Likely 4/4 at slow tempo
            beats_per_bar = 4
        else:
            # Could be 3/4 or 4/4
            # Analyze beat patterns to decide
            beat_intervals = np.diff(beat_times)
            avg_interval = np.mean(beat_intervals)
            
            # Look for patterns that suggest 3/4 vs 4/4
            # This is a simplified heuristic
            beats_per_bar = 4
        
        # Extract bar positions
        bar_indices = np.arange(0, len(beat_times), beats_per_bar)
        bar_times = beat_times[bar_indices]
        
        return bar_times


class BeatTracker:
    """
    Main beat tracker that combines multiple tracking methods.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize the beat tracker.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.trackers = [
            LibrosaBeatTracker(settings),
            AdaptiveBeatTracker(settings)
        ]
        self.quantizer = GridQuantizer(settings)
        
    def track_beats(self, audio: np.ndarray, method: str = "best") -> BeatTrackingResult:
        """
        Track beats using specified method.
        
        Args:
            audio: Input audio signal
            method: Tracking method ("librosa", "adaptive", "best")
            
        Returns:
            Beat tracking result
        """
        if method == "librosa":
            return self.trackers[0].track_beats(audio)
        elif method == "adaptive":
            return self.trackers[1].track_beats(audio)
        elif method == "best":
            return self._find_best_result(audio)
        else:
            raise ValueError(f"Unknown beat tracking method: {method}")
    
    def _find_best_result(self, audio: np.ndarray) -> BeatTrackingResult:
        """Find the best beat tracking result."""
        results = []
        
        for tracker in self.trackers:
            try:
                result = tracker.track_beats(audio)
                results.append(result)
            except Exception as e:
                logger.warning(f"Beat tracker {tracker.__class__.__name__} failed: {e}")
        
        if not results:
            logger.error("All beat trackers failed")
            # Return default result
            return BeatTrackingResult(
                tempo_bpm=120.0,
                beat_times=np.array([]),
                bar_times=np.array([0.0]),
                confidence=0.0,
                method="fallback"
            )
        
        # Choose result with highest confidence
        best_result = max(results, key=lambda r: r.confidence)
        
        logger.info(f"Selected beat tracking result: {best_result.method} "
                   f"(tempo: {best_result.tempo_bpm:.1f} BPM, "
                   f"confidence: {best_result.confidence:.3f})")
        
        return best_result
    
    def align_onsets_to_beats(self, 
                             onset_times: np.ndarray, 
                             beat_times: np.ndarray,
                             tolerance: float = 0.1) -> Dict[int, List[float]]:
        """
        Align detected onsets to beat grid.
        
        Args:
            onset_times: Array of onset times
            beat_times: Array of beat times
            tolerance: Maximum time difference for alignment (seconds)
            
        Returns:
            Dictionary mapping beat indices to aligned onset times
        """
        aligned_onsets = {}
        
        for onset_time in onset_times:
            # Find closest beat
            if len(beat_times) == 0:
                continue
                
            beat_diffs = np.abs(beat_times - onset_time)
            closest_beat_idx = np.argmin(beat_diffs)
            closest_beat_time = beat_times[closest_beat_idx]
            
            # Check if within tolerance
            if beat_diffs[closest_beat_idx] <= tolerance:
                if closest_beat_idx not in aligned_onsets:
                    aligned_onsets[closest_beat_idx] = []
                aligned_onsets[closest_beat_idx].append(onset_time)
        
        return aligned_onsets
    
    def quantize_onsets_to_grid(self, 
                               onset_times: np.ndarray,
                               beat_times: np.ndarray,
                               resolution: int = 4) -> np.ndarray:
        """
        Quantize onsets to a rhythmic grid.
        
        Args:
            onset_times: Array of onset times
            beat_times: Array of beat times
            resolution: Grid resolution (subdivisions per beat)
            
        Returns:
            Array of quantized onset times
        """
        if len(beat_times) < 2:
            return onset_times
        
        # Calculate average beat interval
        beat_interval = np.mean(np.diff(beat_times))
        grid_interval = beat_interval / resolution
        
        quantized_onsets = []
        
        for onset_time in onset_times:
            # Find the grid position closest to this onset
            # Calculate relative position within the beat structure
            if len(beat_times) > 0:
                # Find beat before this onset
                beat_before_indices = np.where(beat_times <= onset_time)[0]
                if len(beat_before_indices) > 0:
                    beat_before_idx = beat_before_indices[-1]
                    beat_before_time = beat_times[beat_before_idx]
                    
                    # Calculate position within beat
                    position_in_beat = onset_time - beat_before_time
                    
                    # Quantize to grid
                    grid_position = round(position_in_beat / grid_interval)
                    quantized_time = beat_before_time + (grid_position * grid_interval)
                    
                    quantized_onsets.append(quantized_time)
                else:
                    # Before first beat
                    quantized_onsets.append(onset_time)
            else:
                quantized_onsets.append(onset_time)
        
        return np.array(quantized_onsets)
    
    def calculate_bar_timing(self, 
                           beat_times: np.ndarray,
                           time_signature: str = "4/4") -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate bar positions and timing.
        
        Args:
            beat_times: Array of beat times
            time_signature: Time signature (e.g., "4/4", "3/4")
            
        Returns:
            Tuple of (bar_positions, bar_times)
        """
        # Parse time signature
        if "/" in time_signature:
            beats_per_bar = int(time_signature.split("/")[0])
        else:
            beats_per_bar = 4  # Default to 4/4
        
        # Calculate bar positions
        num_bars = len(beat_times) // beats_per_bar
        bar_positions = np.arange(0, len(beat_times), beats_per_bar)[:num_bars]
        bar_times = beat_times[bar_positions]
        
        return bar_positions, bar_times
    
    def apply_magnetic_quantization(self,
                                   onset_times: np.ndarray,
                                   beat_times: np.ndarray,
                                   tempo_bpm: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Apply magnetic quantization to onsets and calculate BGM offset.
        
        Args:
            onset_times: Original onset times
            beat_times: Beat times
            tempo_bpm: Tempo in BPM
            
        Returns:
            Tuple of (quantized_onsets, adjusted_beat_times, quantization_info)
        """
        # Apply magnetic quantization to onsets
        quantized_onsets, quantization_info = self.quantizer.quantize_onsets_magnetic(
            onset_times, beat_times, tempo_bpm
        )
        
        # Calculate BGM offset
        bgm_offset, offset_info = self.quantizer.calculate_bgm_offset(
            onset_times, quantized_onsets, beat_times
        )
        
        # Apply BGM offset to beat times
        adjusted_beat_times = self.quantizer.apply_quantization_to_beat_times(
            beat_times, bgm_offset
        )
        
        # Combine quantization info
        combined_info = {
            **quantization_info,
            'bgm_offset': bgm_offset,
            'offset_info': offset_info
        }
        
        return quantized_onsets, adjusted_beat_times, combined_info


class GridQuantizer:
    """
    Advanced rhythmic quantization with magnetic attraction to grid positions.
    
    This class implements intelligent quantization that can:
    - Snap onsets to musical grid positions with configurable precision
    - Preserve intentional timing variations (groove)
    - Calculate optimal BGM timing adjustments
    - Support multiple quantization resolutions
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize the GridQuantizer.
        
        Args:
            settings: Application settings containing quantization parameters
        """
        self.settings = settings
        self.quantization = settings.quantization
        
        # Resolution mapping to subdivisions per beat
        self.resolution_map = {
            'quarter': 1,      # Quarter note (1/4)
            'eighth': 2,       # Eighth note (1/8)
            'sixteenth': 4,    # Sixteenth note (1/16) - recommended for drums
            'thirty_second': 8 # Thirty-second note (1/32)
        }
        
        self.subdivision = self.resolution_map.get(self.quantization.resolution, 4)
        
    def quantize_onsets_magnetic(self, 
                                onset_times: np.ndarray,
                                beat_times: np.ndarray,
                                tempo_bpm: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply magnetic quantization to onset times.
        
        Args:
            onset_times: Original onset times in seconds
            beat_times: Beat positions in seconds
            tempo_bpm: Tempo in BPM
            
        Returns:
            Tuple of (quantized_onset_times, quantization_info)
        """
        if not self.quantization.enabled or len(beat_times) < 2:
            return onset_times, {'method': 'disabled', 'adjustments': 0}
        
        # Calculate grid timing
        beat_interval = np.mean(np.diff(beat_times))
        grid_interval = beat_interval / self.subdivision
        
        quantized_onsets = []
        adjustments = []
        preserved_count = 0
        
        for onset_time in onset_times:
            # Find closest grid position
            grid_pos, distance = self._find_closest_grid_position(
                onset_time, beat_times, grid_interval
            )
            
            # Decide whether to quantize based on distance and settings
            if self._should_quantize_onset(distance):
                # Apply magnetic attraction
                quantized_time = self._apply_magnetic_attraction(
                    onset_time, grid_pos, distance
                )
                quantized_onsets.append(quantized_time)
                adjustments.append(quantized_time - onset_time)
            else:
                # Preserve original timing
                quantized_onsets.append(onset_time)
                adjustments.append(0.0)
                preserved_count += 1
        
        quantized_onsets = np.array(quantized_onsets)
        
        # Calculate quantization statistics
        abs_adjustments = np.abs(adjustments)
        quantization_info = {
            'method': 'magnetic',
            'resolution': self.quantization.resolution,
            'subdivision': self.subdivision,
            'adjustments': len([a for a in adjustments if abs(a) > 0.001]),
            'preserved': preserved_count,
            'total_onsets': len(onset_times),
            'avg_adjustment': np.mean(abs_adjustments) if len(abs_adjustments) > 0 else 0.0,
            'max_adjustment': np.max(abs_adjustments) if len(abs_adjustments) > 0 else 0.0
        }
        
        logger.info(f"Magnetic quantization: {quantization_info['adjustments']}/{quantization_info['total_onsets']} onsets quantized, "
                   f"{preserved_count} preserved")
        
        return quantized_onsets, quantization_info
    
    def _find_closest_grid_position(self, 
                                   onset_time: float,
                                   beat_times: np.ndarray,
                                   grid_interval: float) -> Tuple[float, float]:
        """
        Find the closest grid position to an onset time.
        
        Args:
            onset_time: Time of the onset
            beat_times: Array of beat times
            grid_interval: Time between grid positions
            
        Returns:
            Tuple of (grid_position_time, distance_to_grid)
        """
        # Find the beat before this onset
        beat_before_indices = np.where(beat_times <= onset_time)[0]
        if len(beat_before_indices) == 0:
            # Before first beat, use first beat as reference
            beat_before_time = beat_times[0]
            beat_before_idx = 0
        else:
            beat_before_idx = beat_before_indices[-1]
            beat_before_time = beat_times[beat_before_idx]
        
        # Calculate position within the beat
        position_in_beat = onset_time - beat_before_time
        
        # Find closest grid position
        grid_positions = np.arange(0, beat_times[1] - beat_times[0] if len(beat_times) > 1 else 1.0, grid_interval)
        
        # Include the next beat position as well
        if len(beat_times) > beat_before_idx + 1:
            next_beat_time = beat_times[beat_before_idx + 1]
            next_beat_position = next_beat_time - beat_before_time
            grid_positions = np.append(grid_positions, next_beat_position)
        
        # Find closest grid position
        distances = np.abs(grid_positions - position_in_beat)
        closest_idx = np.argmin(distances)
        
        closest_grid_position = beat_before_time + grid_positions[closest_idx]
        distance = distances[closest_idx]
        
        return closest_grid_position, distance
    
    def _should_quantize_onset(self, distance: float) -> bool:
        """
        Determine whether an onset should be quantized based on distance to grid.
        
        Args:
            distance: Distance to closest grid position in seconds
            
        Returns:
            True if onset should be quantized
        """
        # Always quantize if within magnetic radius
        if distance <= self.quantization.magnetic_radius:
            return True
        
        # Don't quantize if preserve_groove is enabled and distance is large
        if self.quantization.preserve_groove and distance > self.quantization.preserve_threshold:
            return False
        
        return True
    
    def _apply_magnetic_attraction(self, 
                                  onset_time: float,
                                  grid_position: float,
                                  distance: float) -> float:
        """
        Apply magnetic attraction to pull onset toward grid position.
        
        Args:
            onset_time: Original onset time
            grid_position: Target grid position
            distance: Distance to grid position
            
        Returns:
            Quantized onset time
        """
        # Calculate attraction strength based on distance
        if distance <= self.quantization.magnetic_radius:
            # Strong attraction within magnetic radius
            strength = self.quantization.magnetic_strength
        else:
            # Weaker attraction outside radius
            strength = self.quantization.magnetic_strength * 0.5
        
        # Apply attraction
        attraction = (grid_position - onset_time) * strength
        quantized_time = onset_time + attraction
        
        return quantized_time
    
    def calculate_bgm_offset(self, 
                           original_onsets: np.ndarray,
                           quantized_onsets: np.ndarray,
                           beat_times: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate optimal BGM timing offset to maintain synchronization.
        
        Args:
            original_onsets: Original onset times
            quantized_onsets: Quantized onset times
            beat_times: Beat times
            
        Returns:
            Tuple of (bgm_offset_seconds, offset_info)
        """
        if not self.quantization.adjust_bgm_timing:
            return 0.0, {'method': 'disabled', 'offset': 0.0}
        
        # Calculate the adjustments made to onsets
        adjustments = quantized_onsets - original_onsets
        
        # Filter out very small adjustments
        significant_adjustments = adjustments[np.abs(adjustments) > 0.001]
        
        if len(significant_adjustments) == 0:
            return 0.0, {'method': 'no_adjustments', 'offset': 0.0}
        
        # Calculate average adjustment
        avg_adjustment = np.mean(significant_adjustments)
        
        # Only apply BGM offset if the average adjustment is significant
        if abs(avg_adjustment) < self.quantization.bgm_adjustment_threshold:
            return 0.0, {'method': 'below_threshold', 'offset': 0.0, 'avg_adjustment': avg_adjustment}
        
        # The BGM offset should be the negative of the average adjustment
        # to maintain synchronization
        bgm_offset = -avg_adjustment
        
        offset_info = {
            'method': 'calculated',
            'offset': bgm_offset,
            'avg_adjustment': avg_adjustment,
            'significant_adjustments': len(significant_adjustments),
            'total_onsets': len(original_onsets)
        }
        
        logger.info(f"BGM offset calculated: {bgm_offset:.3f}s (avg adjustment: {avg_adjustment:.3f}s)")
        
        return bgm_offset, offset_info
    
    def apply_quantization_to_beat_times(self, 
                                       beat_times: np.ndarray,
                                       bgm_offset: float) -> np.ndarray:
        """
        Apply BGM offset to beat times.
        
        Args:
            beat_times: Original beat times
            bgm_offset: BGM timing offset in seconds
            
        Returns:
            Adjusted beat times
        """
        if abs(bgm_offset) < 0.001:
            return beat_times
        
        # Apply offset to all beat times
        adjusted_beat_times = beat_times + bgm_offset
        
        # Ensure no negative beat times
        adjusted_beat_times = np.maximum(adjusted_beat_times, 0.0)
        
        return adjusted_beat_times
    
    def get_quantization_stats(self) -> Dict[str, Any]:
        """
        Get current quantization settings and statistics.
        
        Returns:
            Dictionary with quantization information
        """
        return {
            'enabled': self.quantization.enabled,
            'resolution': self.quantization.resolution,
            'subdivision': self.subdivision,
            'magnetic_strength': self.quantization.magnetic_strength,
            'magnetic_radius': self.quantization.magnetic_radius,
            'preserve_groove': self.quantization.preserve_groove,
            'preserve_threshold': self.quantization.preserve_threshold,
            'adjust_bgm_timing': self.quantization.adjust_bgm_timing,
            'bgm_adjustment_threshold': self.quantization.bgm_adjustment_threshold
        }