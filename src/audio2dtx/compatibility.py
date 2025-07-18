"""
Compatibility layer for maintaining backward compatibility with existing API.

This module provides a bridge between the old monolithic AudioToChart class
and the new modular architecture.
"""

import os
import numpy as np
from typing import Dict, Any, Optional, List

from .config.settings import Settings, get_settings
from .core.audio_processor import AudioProcessor
from .utils.exceptions import ProcessingError
from .utils.logging import get_logger

logger = get_logger(__name__)


class AudioToChart:
    """
    Backward-compatible wrapper for the original AudioToChart class.
    
    This class maintains the same interface as the original implementation
    while using the new modular architecture underneath.
    """
    
    def __init__(self, 
                 input_audio: str,
                 metadata: Dict[str, Any],
                 use_magenta_only: bool = False,
                 use_advanced_features: bool = False,
                 use_multi_scale: bool = False,
                 use_few_shot: bool = False,
                 use_ensemble: bool = False,
                 use_augmentation: bool = False,
                 use_rock_ultimate: bool = False):
        """
        Initialize AudioToChart with backward compatibility.
        
        Args:
            input_audio: Path to input audio file
            metadata: Song metadata
            use_magenta_only: Use Track 3 (Magenta-only)
            use_advanced_features: Use Track 4 (Advanced features)
            use_multi_scale: Use Track 5 (Multi-scale)
            use_few_shot: Use Track 6 (Few-shot learning)
            use_ensemble: Use Track 7 (Ensemble)
            use_augmentation: Use Track 8 (Augmentation)
            use_rock_ultimate: Use Track 9 (Rock/Metal ultimate)
        """
        self.input_audio = input_audio
        self.metadata = metadata
        
        # Initialize settings and processor
        self.settings = get_settings()
        self.processor = AudioProcessor(self.settings)
        
        # Determine track type from flags
        self.track_type = self._determine_track_type(
            use_magenta_only, use_advanced_features, use_multi_scale,
            use_few_shot, use_ensemble, use_augmentation, use_rock_ultimate
        )
        
        # Legacy attributes for compatibility
        self.song_name = metadata.get('title', 'song')
        self.tempo_bpm = 120.0  # Will be updated during processing
        self.beat_times = []
        self.beat_aligned_onsets = {i: [] for i in range(10)}
        self.num_class = 10
        
        # DTX constants for compatibility
        self.INT_TO_CHANNEL = {
            0: '11', 1: '12', 2: '13', 3: '14', 4: '15',
            5: '16', 6: '17', 7: '18', 8: '19', 9: '1A'
        }
        
        logger.info(f"AudioToChart initialized with track type: {self.track_type}")
    
    def _determine_track_type(self, *track_flags) -> str:
        """Determine which track to use based on flags."""
        track_names = [
            'magenta_only', 'advanced_features', 'multi_scale',
            'few_shot', 'ensemble', 'augmentation', 'rock_ultimate'
        ]
        
        selected_tracks = [name for name, flag in zip(track_names, track_flags) if flag]
        
        if len(selected_tracks) > 1:
            logger.warning(f"Multiple tracks selected: {selected_tracks}. Using first one.")
            return selected_tracks[0]
        elif len(selected_tracks) == 1:
            return selected_tracks[0]
        else:
            return 'default'
    
    def extract_beats(self):
        """
        Extract beats from audio (legacy method).
        
        This method processes the audio and extracts beats, maintaining
        compatibility with the original interface.
        """
        try:
            logger.info("Starting beat extraction (legacy method)")
            
            # Load and process audio using new architecture
            audio, audio_metadata = self.processor.audio_loader.load_audio(self.input_audio)
            
            # Update metadata
            self.metadata.update(audio_metadata)
            
            # Preprocess audio
            processed_audio = self.processor.preprocessor.preprocess_for_classification(audio)
            
            # Separate audio
            stems = self.processor.separator.separate_audio(processed_audio)
            drum_audio = stems.get('drums', processed_audio)
            
            # Track beats
            beat_result = self.processor.beat_tracker.track_beats(drum_audio)
            self.tempo_bpm = beat_result.tempo_bpm
            self.beat_times = beat_result.beat_times.tolist()
            
            # Detect onsets
            onset_results = self.processor.onset_detector.detect_onsets(drum_audio)
            fused_onsets, _ = self.processor.onset_detector.fuse_onsets(onset_results)
            
            # Classify onsets using selected track
            classified_onsets = self._classify_onsets_legacy(drum_audio, fused_onsets)
            
            # Update beat-aligned onsets for compatibility
            self.beat_aligned_onsets = classified_onsets
            
            logger.info(f"Beat extraction completed: {len(fused_onsets)} onsets, "
                       f"tempo: {self.tempo_bpm:.1f} BPM")
            
        except Exception as e:
            logger.error(f"Beat extraction failed: {e}")
            raise ProcessingError(f"Failed to extract beats: {e}")
    
    def _classify_onsets_legacy(self, 
                               audio: np.ndarray,
                               onset_times: np.ndarray) -> Dict[int, List[float]]:
        """Legacy onset classification for compatibility."""
        # Use the processor's classification method
        return self.processor._classify_onsets(audio, onset_times, self.track_type)
    
    def create_chart(self):
        """
        Create DTX chart (legacy method).
        
        This method is called after extract_beats() to generate the chart data.
        """
        logger.info("Creating DTX chart (legacy method)")
        
        # This method doesn't need to do much since extract_beats() does most work
        # Just validate that we have the necessary data
        if not self.beat_times:
            raise ProcessingError("No beat times available. Call extract_beats() first.")
        
        if not any(self.beat_aligned_onsets.values()):
            logger.warning("No classified onsets available")
    
    def export(self, output_dir: str) -> str:
        """
        Export DTX chart to files (legacy method).
        
        Args:
            output_dir: Output directory
            
        Returns:
            Path to exported DTX file
        """
        try:
            logger.info(f"Exporting DTX chart to {output_dir}")
            
            # Create DTX chart using new architecture
            from .dtx.writer import DTXChart
            
            chart = DTXChart(metadata=self.metadata, notes=[])
            
            # Generate notes from beat-aligned onsets
            notes = self.processor.dtx_writer.generate_notes_from_onsets(
                self.beat_aligned_onsets,
                self.beat_times,
                self.tempo_bpm
            )
            
            chart.notes = notes
            chart.bar_count = max(note.bar for note in notes) if notes else 0
            
            # Create DTX package
            song_name = self.metadata.get('title', 'song')
            safe_song_name = "".join(c for c in song_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            
            dtx_path = self.processor.dtx_writer.create_complete_dtx_package(
                chart=chart,
                bgm_audio_path=None,  # Will be implemented later
                drum_sounds_dir=None,
                output_dir=output_dir,
                song_name=safe_song_name
            )
            
            logger.info(f"DTX chart exported successfully: {dtx_path}")
            return dtx_path
            
        except Exception as e:
            logger.error(f"DTX export failed: {e}")
            raise ProcessingError(f"Failed to export DTX chart: {e}")
    
    def calculate_adaptive_bar_timing(self, beat_times: List[float]) -> tuple:
        """
        Calculate adaptive bar timing (legacy method).
        
        Args:
            beat_times: List of beat times
            
        Returns:
            Tuple of (bar_positions, bar_times)
        """
        return self.processor.beat_tracker.calculate_bar_timing(
            np.array(beat_times), 
            self.metadata.get('time_signature', '4/4')
        )
    
    def generate_beat_based_notes(self, dtx_file, bar_offset: int, wav_mappings: Dict):
        """Legacy method for generating notes (placeholder)."""
        logger.warning("generate_beat_based_notes is deprecated. Use export() instead.")
    
    def get_processing_info(self) -> Dict[str, Any]:
        """
        Get information about the processing pipeline.
        
        Returns:
            Dictionary with processing information
        """
        return {
            'track_type': self.track_type,
            'tempo_bpm': self.tempo_bpm,
            'beat_count': len(self.beat_times),
            'onset_counts': {str(k): len(v) for k, v in self.beat_aligned_onsets.items()},
            'total_onsets': sum(len(v) for v in self.beat_aligned_onsets.values()),
            'metadata': self.metadata,
            'processor_info': self.processor.get_component_info()
        }


# Legacy function for direct compatibility
def create_audio_to_chart(input_audio: str, 
                         metadata: Dict[str, Any],
                         **kwargs) -> AudioToChart:
    """
    Create AudioToChart instance (legacy function).
    
    Args:
        input_audio: Path to input audio file
        metadata: Song metadata
        **kwargs: Track selection flags
        
    Returns:
        AudioToChart instance
    """
    return AudioToChart(input_audio, metadata, **kwargs)