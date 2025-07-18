"""
Main audio processor that orchestrates the entire conversion pipeline.
"""

import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from ..config.settings import Settings, get_settings
from ..audio.loader import AudioLoader
from ..audio.preprocessor import AudioPreprocessor
from ..audio.separator import AudioSeparator
from ..audio.analyzer import SpectralAnalyzer
from .onset_detector import OnsetDetector
from .beat_tracker import BeatTracker
from ..classification.feature_extractor import FeatureExtractor
from ..classification.track_manager import TrackManager
from ..dtx.writer import DTXWriter, DTXChart
from ..services.magenta_client import MagentaClient
from ..utils.exceptions import ProcessingError, ValidationError
from ..utils.validators import validate_audio_file, validate_metadata
from ..utils.logging import get_logger, ProgressLogger

logger = get_logger(__name__)


class AudioProcessor:
    """
    Main audio processor that coordinates the entire audio-to-DTX conversion pipeline.
    
    This is the primary interface for converting audio files to DTX charts.
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize the audio processor.
        
        Args:
            settings: Application settings (uses global settings if None)
        """
        self.settings = settings or get_settings()
        
        # Initialize components
        self.audio_loader = AudioLoader(self.settings)
        self.preprocessor = AudioPreprocessor(self.settings)
        self.separator = AudioSeparator(self.settings)
        self.analyzer = SpectralAnalyzer(self.settings)
        self.onset_detector = OnsetDetector(self.settings)
        self.beat_tracker = BeatTracker(self.settings)
        self.feature_extractor = FeatureExtractor(self.settings)
        self.track_manager = TrackManager(self.settings)
        self.dtx_writer = DTXWriter(self.settings)
        self.magenta_client = MagentaClient(self.settings)
        
        # Processing state
        self.current_audio = None
        self.current_metadata = None
        self.processing_results = {}
        
    def process_audio_file(self, 
                          input_file: str,
                          output_dir: str,
                          metadata: Dict[str, Any],
                          track_type: str = 'default') -> str:
        """
        Process an audio file and generate DTX chart.
        
        Args:
            input_file: Path to input audio file
            output_dir: Output directory for generated files
            metadata: Song metadata
            track_type: Classification track to use
            
        Returns:
            Path to generated DTX file
            
        Raises:
            ProcessingError: If processing fails
            ValidationError: If inputs are invalid
        """
        # Validate inputs
        input_file = validate_audio_file(input_file)
        metadata = validate_metadata(metadata)
        
        # Initialize progress tracking
        total_steps = 8
        progress = ProgressLogger(logger, total_steps, "Audio Processing")
        
        try:
            # Step 1: Load audio
            progress.step("Loading audio file")
            audio, audio_metadata = self.audio_loader.load_audio(input_file)
            self.current_audio = audio
            self.current_metadata = {**metadata, **audio_metadata}
            
            # Step 2: Preprocess audio
            progress.step("Preprocessing audio")
            processed_audio = self.preprocessor.preprocess_for_classification(audio)
            
            # Step 3: Audio separation
            progress.step("Separating audio sources")
            separated_stems = self.separator.separate_audio(processed_audio)
            drum_audio = separated_stems.get('drums', processed_audio)
            
            # Step 4: Beat tracking
            progress.step("Analyzing tempo and beats")
            beat_result = self.beat_tracker.track_beats(drum_audio)
            
            # Step 5: Onset detection
            progress.step("Detecting drum onsets")
            onset_results = self.onset_detector.detect_onsets(drum_audio)
            
            # Step 6: Fuse onset detection results
            progress.step("Fusing onset detection results")
            fused_onsets, fusion_info = self.onset_detector.fuse_onsets(
                onset_results, method='union'
            )
            
            # Step 7: Classify onsets
            progress.step("Classifying drum instruments")
            classified_onsets = self._classify_onsets(
                drum_audio, fused_onsets, track_type
            )
            
            # Step 8: Generate DTX
            progress.step("Generating DTX chart")
            dtx_path = self._generate_dtx_chart(
                classified_onsets, beat_result, output_dir, metadata
            )
            
            progress.complete("Audio processing completed successfully")
            
            # Store processing results
            self.processing_results = {
                'audio_metadata': audio_metadata,
                'beat_result': beat_result,
                'onset_count': len(fused_onsets),
                'fusion_info': fusion_info,
                'classified_count': len(classified_onsets),
                'output_file': dtx_path
            }
            
            return dtx_path
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            raise ProcessingError(f"Failed to process audio file: {e}")
    
    def _classify_onsets(self, 
                        audio: np.ndarray,
                        onset_times: np.ndarray,
                        track_type: str) -> Dict[int, List[float]]:
        """
        Classify detected onsets using specified track.
        
        Args:
            audio: Drum audio signal
            onset_times: Array of onset times
            track_type: Classification track to use
            
        Returns:
            Dictionary mapping instrument IDs to onset times
        """
        try:
            # Extract audio windows around onsets
            window_size = int(0.1 * self.settings.audio.sample_rate)  # 100ms windows
            onset_windows = []
            
            for onset_time in onset_times:
                onset_sample = int(onset_time * self.settings.audio.sample_rate)
                start_sample = max(0, onset_sample - window_size // 2)
                end_sample = min(len(audio), onset_sample + window_size // 2)
                
                window = audio[start_sample:end_sample]
                # Pad if necessary
                if len(window) < window_size:
                    padding = window_size - len(window)
                    window = np.pad(window, (0, padding), mode='constant')
                
                onset_windows.append((window, onset_time))
            
            # Prepare context for classification
            context = {
                'full_audio': audio,
                'metadata': self.current_metadata,
                'beat_times': getattr(self, 'beat_times', []),
                'tempo_bpm': getattr(self, 'tempo_bpm', 120.0)
            }
            
            # Use track manager for classification
            if track_type == 'default':
                # Use best track for genre
                genre = self.current_metadata.get('genre', 'unknown') if self.current_metadata else 'unknown'
                results = self.track_manager.classify_with_best_track(onset_windows, context, genre)
            else:
                # Use specific track
                results = self.track_manager.classify_with_track(track_type, onset_windows, context)
            
            # Convert results to the expected format
            classified_onsets = self._convert_results_to_legacy_format(results, onset_times)
            
            logger.info(f"Classified {len(onset_times)} onsets using {track_type} method")
            return classified_onsets
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            # Fall back to simple classification
            return self._fallback_classification([(audio[int(ot * self.settings.audio.sample_rate):int(ot * self.settings.audio.sample_rate) + window_size], ot) for ot in onset_times])
    
    def _fallback_classification(self, 
                               onset_windows: List[Tuple[np.ndarray, float]]) -> Dict[int, List[float]]:
        """
        Fallback classification using simple frequency analysis.
        
        Args:
            onset_windows: List of (audio_window, onset_time) tuples
            
        Returns:
            Dictionary mapping instrument IDs to onset times
        """
        classified = {i: [] for i in range(10)}  # 10 instrument classes
        
        for audio_window, onset_time in onset_windows:
            # Simple frequency-based classification
            features = self.feature_extractor.extract_basic_features(audio_window)
            
            spectral_centroid = features.get('spectral_centroid', 1000)
            rms_energy = features.get('rms_energy', 0.1)
            
            # Basic classification rules
            if spectral_centroid < 150 and rms_energy > 0.3:
                # Kick drum
                instrument_id = 2
            elif 200 < spectral_centroid < 1000 and rms_energy > 0.2:
                # Snare drum
                instrument_id = 1
            elif spectral_centroid > 2000:
                # Hi-hat or cymbal
                instrument_id = 0 if rms_energy < 0.4 else 9
            else:
                # Default to snare
                instrument_id = 1
            
            classified[instrument_id].append(onset_time)
        
        return classified
    
    def _convert_results_to_legacy_format(self, 
                                        results: List,
                                        onset_times: np.ndarray) -> Dict[int, List[float]]:
        """
        Convert classification results to legacy format.
        
        Args:
            results: List of ClassificationResult objects
            onset_times: Array of onset times
            
        Returns:
            Dictionary mapping instrument IDs to onset times
        """
        from ..config.constants import DRUM_CLASSES
        
        # Create reverse mapping
        instrument_to_id = {v: k for k, v in DRUM_CLASSES.items()}
        
        # Initialize result dictionary
        classified_onsets = {i: [] for i in range(10)}
        
        # Convert results
        for i, result in enumerate(results):
            if i < len(onset_times):
                onset_time = onset_times[i]
                instrument_id = instrument_to_id.get(result.instrument, 2)  # Default to kick
                classified_onsets[instrument_id].append(onset_time)
        
        return classified_onsets
    
    def _generate_dtx_chart(self, 
                           classified_onsets: Dict[int, List[float]],
                           beat_result,
                           output_dir: str,
                           metadata: Dict[str, Any]) -> str:
        """
        Generate DTX chart from classified onsets.
        
        Args:
            classified_onsets: Dictionary mapping instruments to onset times
            beat_result: Beat tracking result
            output_dir: Output directory
            metadata: Song metadata
            
        Returns:
            Path to generated DTX file
        """
        # Create DTX chart
        chart = DTXChart(metadata=metadata, notes=[])
        
        # Generate notes from onsets
        notes = self.dtx_writer.generate_notes_from_onsets(
            classified_onsets,
            beat_result.beat_times.tolist(),
            beat_result.tempo_bpm
        )
        
        chart.notes = notes
        chart.bar_count = max(note.bar for note in notes) if notes else 0
        
        # Validate chart
        issues = self.dtx_writer.validate_dtx_chart(chart)
        if issues:
            logger.warning(f"DTX chart validation issues: {issues}")
        
        # Generate output files
        song_name = metadata.get('title', 'song')
        safe_song_name = "".join(c for c in song_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        
        # Create BGM file (for now, just a placeholder)
        bgm_path = None  # Will be implemented in Phase 2
        
        # Create DTX package
        dtx_package_path = self.dtx_writer.create_complete_dtx_package(
            chart=chart,
            bgm_audio_path=bgm_path,
            drum_sounds_dir=None,  # Will use template sounds
            output_dir=output_dir,
            song_name=safe_song_name
        )
        
        return dtx_package_path
    
    def get_processing_results(self) -> Dict[str, Any]:
        """
        Get results from the last processing operation.
        
        Returns:
            Dictionary with processing results
        """
        return self.processing_results.copy()
    
    def get_component_info(self) -> Dict[str, Any]:
        """
        Get information about all processing components.
        
        Returns:
            Dictionary with component information
        """
        return {
            'audio_loader': {
                'target_sample_rate': self.audio_loader.target_sr
            },
            'separator': self.separator.get_info(),
            'magenta_client': self.magenta_client.get_connection_info(),
            'settings': self.settings.to_dict()
        }
    
    def test_components(self) -> Dict[str, bool]:
        """
        Test all components to ensure they're working correctly.
        
        Returns:
            Dictionary mapping component names to test results
        """
        test_results = {}
        
        # Test Magenta service
        try:
            test_results['magenta_service'] = self.magenta_client.is_available()
        except Exception as e:
            logger.error(f"Magenta service test failed: {e}")
            test_results['magenta_service'] = False
        
        # Test audio separation
        try:
            test_results['audio_separation'] = self.separator.is_available()
        except Exception as e:
            logger.error(f"Audio separation test failed: {e}")
            test_results['audio_separation'] = False
        
        # Test onset detection
        try:
            # Create dummy audio for testing
            dummy_audio = np.random.randn(44100).astype(np.float32)  # 1 second
            onset_results = self.onset_detector.detect_onsets(dummy_audio)
            test_results['onset_detection'] = any(result.success for result in onset_results.values())
        except Exception as e:
            logger.error(f"Onset detection test failed: {e}")
            test_results['onset_detection'] = False
        
        # Test beat tracking
        try:
            dummy_audio = np.random.randn(44100).astype(np.float32)
            beat_result = self.beat_tracker.track_beats(dummy_audio)
            test_results['beat_tracking'] = beat_result.tempo_bpm > 0
        except Exception as e:
            logger.error(f"Beat tracking test failed: {e}")
            test_results['beat_tracking'] = False
        
        return test_results
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self.feature_extractor, 'clear_cache'):
            self.feature_extractor.clear_cache()
        
        self.current_audio = None
        self.current_metadata = None
        self.processing_results.clear()