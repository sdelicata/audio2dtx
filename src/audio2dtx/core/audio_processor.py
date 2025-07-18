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
        self.quantization_results = {}
        
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
        total_steps = 9
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
            
            # Step 6.5: Apply magnetic quantization
            progress.step("Applying magnetic quantization")
            quantized_onsets, adjusted_beat_times, quantization_info = self.beat_tracker.apply_magnetic_quantization(
                fused_onsets, beat_result.beat_times, beat_result.tempo_bpm
            )
            
            # Update beat result with adjusted times
            beat_result.beat_times = adjusted_beat_times
            
            # Step 7: Classify onsets
            progress.step("Classifying drum instruments")
            classified_onsets = self._classify_onsets(
                drum_audio, quantized_onsets, track_type
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
            
            # Store quantization results
            self.quantization_results = {
                'original_onsets': len(fused_onsets),
                'quantized_onsets': len(quantized_onsets),
                'quantization_info': quantization_info
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
                # Handle different result formats
                if hasattr(result, 'instrument'):
                    instrument_name = result.instrument
                elif hasattr(result, 'prediction'):
                    instrument_name = result.prediction
                elif isinstance(result, str):
                    instrument_name = result
                else:
                    # Fallback to kick drum
                    instrument_name = 'kick'
                    logger.warning(f"Unknown result format: {type(result)}, using kick drum")
                
                instrument_id = instrument_to_id.get(instrument_name, 2)  # Default to kick
                classified_onsets[instrument_id].append(onset_time)
        
        # Log statistics for debugging
        total_onsets = sum(len(times) for times in classified_onsets.values())
        logger.info(f"Converted {len(results)} results to {total_onsets} onsets across {len([k for k, v in classified_onsets.items() if v])} instruments")
        
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
        # Add detected BPM to metadata
        metadata_with_bpm = metadata.copy()
        metadata_with_bpm['bpm'] = beat_result.tempo_bpm
        
        # Create DTX chart
        chart = DTXChart(metadata=metadata_with_bpm, notes=[])
        
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
        
        # Create BGM file with quantization offset
        bgm_offset = self.quantization_results.get('quantization_info', {}).get('bgm_offset', 0.0)
        bgm_path = self._create_bgm_file(metadata_with_bpm, output_dir, safe_song_name, bgm_offset)
        
        # Create DTX package
        dtx_package_path = self.dtx_writer.create_complete_dtx_package(
            chart=chart,
            bgm_audio_path=bgm_path,
            drum_sounds_dir=None,  # Will use template sounds
            output_dir=output_dir,
            song_name=safe_song_name
        )
        
        return dtx_package_path
    
    def _create_bgm_file(self, 
                        metadata: Dict[str, Any], 
                        output_dir: str,
                        safe_song_name: str,
                        bgm_offset: float = 0.0) -> Optional[str]:
        """
        Create BGM file for DTX package.
        
        Args:
            metadata: Song metadata
            output_dir: Output directory
            safe_song_name: Safe filename for the song
            bgm_offset: BGM timing offset in seconds
            
        Returns:
            Path to created BGM file or None if creation failed
        """
        if self.current_audio is None:
            logger.warning("No audio loaded, cannot create BGM file")
            return None
        
        try:
            import soundfile as sf
            
            # Determine BGM type based on metadata
            use_original_bgm = metadata.get('use_original_bgm', True)
            
            if use_original_bgm:
                # Use original audio as BGM
                bgm_audio = self.current_audio
                bgm_filename = "bgm.wav"
                logger.info("Creating BGM from original audio")
            else:
                # Create separated BGM (no drums)
                bgm_audio = self.separator.create_backing_track(
                    self.current_audio, 
                    include_drums=False
                )
                bgm_filename = "bgm_separated.wav"
                logger.info("Creating separated BGM (no drums)")
            
            # Apply BGM offset if needed
            if abs(bgm_offset) > 0.001:
                bgm_audio = self._apply_bgm_offset(bgm_audio, bgm_offset)
                logger.info(f"Applied BGM offset: {bgm_offset:.3f}s")
            
            # Create temporary BGM file
            temp_dir = os.path.join(output_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Convert to WAV format (DTXMania works better with WAV)
            bgm_path = self._convert_audio_to_wav(
                bgm_audio, 
                temp_dir, 
                bgm_filename, 
                self.settings.audio.sample_rate
            )
            
            if bgm_path:
                logger.info(f"Created BGM file: {bgm_path}")
            else:
                logger.error("Failed to create BGM file")
            
            return bgm_path
            
        except Exception as e:
            logger.error(f"Failed to create BGM file: {e}")
            return None
    
    def _convert_audio_to_mp3(self, 
                             audio: np.ndarray, 
                             output_dir: str, 
                             filename: str, 
                             sample_rate: int) -> Optional[str]:
        """
        Convert audio to MP3 format using pydub.
        
        Args:
            audio: Audio data
            output_dir: Output directory
            filename: Output filename
            sample_rate: Sample rate
            
        Returns:
            Path to created MP3 file or None if conversion failed
        """
        try:
            # Try using pydub for MP3 conversion
            from pydub import AudioSegment
            import soundfile as sf
            
            # First save as temporary WAV file
            temp_wav_path = os.path.join(output_dir, "temp_bgm.wav")
            sf.write(temp_wav_path, audio, sample_rate)
            
            # Convert WAV to MP3 using pydub
            wav_audio = AudioSegment.from_wav(temp_wav_path)
            mp3_path = os.path.join(output_dir, filename)
            
            # Export as MP3 with good quality settings
            wav_audio.export(
                mp3_path, 
                format="mp3", 
                bitrate="192k",  # Good quality, reasonable file size
                tags={}
            )
            
            # Clean up temporary WAV file
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
            
            logger.info(f"Successfully converted audio to MP3: {mp3_path}")
            return mp3_path
            
        except ImportError:
            logger.warning("pydub not available, trying alternative conversion method")
            return self._convert_audio_to_mp3_fallback(audio, output_dir, filename, sample_rate)
        except Exception as e:
            logger.error(f"MP3 conversion failed: {e}")
            return self._convert_audio_to_mp3_fallback(audio, output_dir, filename, sample_rate)
    
    def _convert_audio_to_mp3_fallback(self, 
                                      audio: np.ndarray, 
                                      output_dir: str, 
                                      filename: str, 
                                      sample_rate: int) -> Optional[str]:
        """
        Fallback method for MP3 conversion using system ffmpeg.
        
        Args:
            audio: Audio data
            output_dir: Output directory
            filename: Output filename
            sample_rate: Sample rate
            
        Returns:
            Path to created file or None if conversion failed
        """
        try:
            import soundfile as sf
            import subprocess
            
            # First save as temporary WAV file
            temp_wav_path = os.path.join(output_dir, "temp_bgm.wav")
            sf.write(temp_wav_path, audio, sample_rate)
            
            # Try to convert using ffmpeg
            mp3_path = os.path.join(output_dir, filename)
            
            # Use ffmpeg to convert WAV to MP3
            cmd = [
                'ffmpeg', '-i', temp_wav_path,
                '-codec:a', 'mp3',
                '-b:a', '192k',
                '-y',  # Overwrite output file
                mp3_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Clean up temporary WAV file
                if os.path.exists(temp_wav_path):
                    os.remove(temp_wav_path)
                logger.info(f"Successfully converted audio to MP3 using ffmpeg: {mp3_path}")
                return mp3_path
            else:
                logger.warning(f"ffmpeg conversion failed: {result.stderr}")
                # Fall back to WAV format
                return self._convert_audio_to_wav_fallback(audio, output_dir, filename, sample_rate)
                
        except Exception as e:
            logger.warning(f"ffmpeg conversion failed: {e}")
            # Fall back to WAV format
            return self._convert_audio_to_wav_fallback(audio, output_dir, filename, sample_rate)
    
    def _convert_audio_to_wav(self, 
                             audio: np.ndarray, 
                             output_dir: str, 
                             filename: str, 
                             sample_rate: int) -> Optional[str]:
        """
        Convert audio to WAV format.
        
        Args:
            audio: Audio data
            output_dir: Output directory
            filename: Output filename
            sample_rate: Sample rate
            
        Returns:
            Path to created WAV file
        """
        try:
            import soundfile as sf
            
            # Ensure filename has .wav extension
            if not filename.endswith('.wav'):
                filename = filename.replace('.mp3', '.wav')
            
            wav_path = os.path.join(output_dir, filename)
            
            # Write WAV file with high quality settings
            sf.write(wav_path, audio, sample_rate, subtype='PCM_16')
            logger.info(f"Successfully created WAV file: {wav_path}")
            
            return wav_path
            
        except Exception as e:
            logger.error(f"Failed to save WAV file: {e}")
            return None
    
    def _convert_audio_to_wav_fallback(self, 
                                      audio: np.ndarray, 
                                      output_dir: str, 
                                      filename: str, 
                                      sample_rate: int) -> Optional[str]:
        """
        Final fallback: save as WAV file.
        
        Args:
            audio: Audio data
            output_dir: Output directory
            filename: Output filename
            sample_rate: Sample rate
            
        Returns:
            Path to created WAV file
        """
        try:
            import soundfile as sf
            
            # Change extension to WAV
            wav_filename = filename.replace('.mp3', '.wav')
            wav_path = os.path.join(output_dir, wav_filename)
            
            sf.write(wav_path, audio, sample_rate)
            logger.warning(f"MP3 conversion not available, saved as WAV: {wav_path}")
            
            return wav_path
            
        except Exception as e:
            logger.error(f"Failed to save audio file: {e}")
            return None
    
    def get_processing_results(self) -> Dict[str, Any]:
        """
        Get results from the last processing operation.
        
        Returns:
            Dictionary with processing results
        """
        results = self.processing_results.copy()
        results['quantization'] = self.quantization_results.copy()
        return results
    
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
        self.quantization_results.clear()
    
    def _apply_bgm_offset(self, audio: np.ndarray, offset: float) -> np.ndarray:
        """
        Apply timing offset to audio signal.
        
        Args:
            audio: Input audio signal
            offset: Offset in seconds (positive = add silence at start, negative = trim start)
            
        Returns:
            Audio with applied offset
        """
        if abs(offset) < 0.001:
            return audio
        
        sample_rate = self.settings.audio.sample_rate
        offset_samples = int(offset * sample_rate)
        
        if offset_samples > 0:
            # Add silence at the beginning
            silence = np.zeros(offset_samples, dtype=audio.dtype)
            adjusted_audio = np.concatenate([silence, audio])
        else:
            # Trim from the beginning
            trim_samples = abs(offset_samples)
            if trim_samples < len(audio):
                adjusted_audio = audio[trim_samples:]
            else:
                # If trim is longer than audio, return minimal audio
                adjusted_audio = audio[-1000:] if len(audio) > 1000 else audio
        
        return adjusted_audio