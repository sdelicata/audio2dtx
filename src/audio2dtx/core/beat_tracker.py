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