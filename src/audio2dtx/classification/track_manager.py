"""
Track manager for handling different classification tracks.
"""

from typing import Dict, List, Any, Optional, Type
import importlib

from .base_classifier import BaseClassifier, ClassificationResult
from .voting_system import VotingSystem
from ..config.settings import Settings
from ..utils.exceptions import ClassificationError
from ..utils.logging import get_logger

logger = get_logger(__name__)


class TrackManager:
    """
    Manager for classification tracks that handles track selection,
    initialization, and coordination.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize track manager.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.voting_system = VotingSystem(settings)
        self.available_tracks = {}
        self.active_tracks = {}
        
        # Register available tracks
        self._register_available_tracks()
    
    def _register_available_tracks(self):
        """Register all available classification tracks."""
        track_configs = {
            'magenta_only': {
                'module': 'audio2dtx.classification.tracks.track3_magenta',
                'class': 'MagentaTrack',
                'description': 'Magenta-Only Classification (Track 3)'
            },
            'advanced_features': {
                'module': 'audio2dtx.classification.tracks.track4_advanced',
                'class': 'AdvancedFeaturesTrack',
                'description': 'Advanced Spectral Features + Context (Track 4)'
            },
            'multi_scale': {
                'module': 'audio2dtx.classification.tracks.track5_multiscale',
                'class': 'MultiScaleTrack',
                'description': 'Multi-Scale Temporal Analysis (Track 5)'
            },
            'few_shot': {
                'module': 'audio2dtx.classification.tracks.track6_fewshot',
                'class': 'FewShotTrack',
                'description': 'Real-Time Few-Shot Learning (Track 6)'
            },
            # Tracks 7-9 now implemented
            'ensemble': {
                'module': 'audio2dtx.classification.tracks.track7_ensemble',
                'class': 'EnsembleTrack',
                'description': 'Ensemble of Specialized Models (Track 7)',
                'available': True
            },
            'augmentation': {
                'module': 'audio2dtx.classification.tracks.track8_augmentation',
                'class': 'AugmentationTrack',
                'description': 'Data Augmentation and Preprocessing (Track 8)',
                'available': True
            },
            'rock_ultimate': {
                'module': 'audio2dtx.classification.tracks.track9_rock_ultimate',
                'class': 'RockUltimateTrack',
                'description': 'Ultimate Rock/Metal Hybrid (Track 9)',
                'available': True
            }
        }
        
        self.available_tracks = track_configs
        logger.info(f"Registered {len(track_configs)} track configurations")
    
    def get_available_tracks(self) -> Dict[str, str]:
        """Get list of available tracks with descriptions."""
        return {
            name: config['description'] 
            for name, config in self.available_tracks.items()
            if config.get('available', True)
        }
    
    def initialize_track(self, track_name: str) -> BaseClassifier:
        """
        Initialize a specific track.
        
        Args:
            track_name: Name of the track to initialize
            
        Returns:
            Initialized track classifier
            
        Raises:
            ClassificationError: If track cannot be initialized
        """
        try:
            if track_name not in self.available_tracks:
                raise ClassificationError(f"Unknown track: {track_name}")
            
            track_config = self.available_tracks[track_name]
            
            # Check if track is available
            if not track_config.get('available', True):
                raise ClassificationError(f"Track {track_name} is not yet implemented")
            
            # Import and instantiate the track
            module_name = track_config['module']
            class_name = track_config['class']
            
            module = importlib.import_module(module_name)
            track_class = getattr(module, class_name)
            
            # Create instance
            track_instance = track_class(self.settings)
            
            # Initialize the track
            track_instance.initialize()
            
            # Store in active tracks
            self.active_tracks[track_name] = track_instance
            
            logger.info(f"✅ Initialized track: {track_name}")
            return track_instance
            
        except Exception as e:
            logger.error(f"Failed to initialize track {track_name}: {e}")
            raise ClassificationError(f"Track initialization failed: {e}")
    
    def classify_with_track(self, 
                          track_name: str,
                          onsets: List[tuple],
                          context: Optional[Dict[str, Any]] = None) -> List[ClassificationResult]:
        """
        Classify onsets using a specific track.
        
        Args:
            track_name: Name of the track to use
            onsets: List of (audio_window, onset_time) tuples
            context: Additional context information
            
        Returns:
            List of classification results
        """
        try:
            # Initialize track if not already active
            if track_name not in self.active_tracks:
                self.initialize_track(track_name)
            
            track = self.active_tracks[track_name]
            
            # Classify onsets
            if hasattr(track, 'classify_onsets'):
                # Use batch classification if available
                results = track.classify_onsets(onsets, context)
            else:
                # Use individual classification
                results = []
                for audio_window, onset_time in onsets:
                    result = track.classify_onset(audio_window, onset_time, context)
                    results.append(result)
            
            logger.info(f"🎯 Track {track_name} classified {len(results)} onsets")
            return results
            
        except Exception as e:
            logger.error(f"Classification failed for track {track_name}: {e}")
            raise ClassificationError(f"Track classification failed: {e}")
    
    def classify_with_multiple_tracks(self, 
                                    track_names: List[str],
                                    onsets: List[tuple],
                                    context: Optional[Dict[str, Any]] = None,
                                    voting_method: str = 'weighted_confidence') -> List[ClassificationResult]:
        """
        Classify onsets using multiple tracks and combine results.
        
        Args:
            track_names: List of track names to use
            onsets: List of (audio_window, onset_time) tuples
            context: Additional context information
            voting_method: Method for combining results
            
        Returns:
            List of combined classification results
        """
        try:
            logger.info(f"🎼 Running multi-track classification with {len(track_names)} tracks")
            
            # Optimize voting weights based on genre if available
            if context and 'metadata' in context:
                genre = context['metadata'].get('genre', 'unknown')
                self.voting_system.optimize_weights_for_genre(genre)
            
            # Run each track
            track_results = {}
            
            for track_name in track_names:
                try:
                    logger.info(f"🎸 Running {track_name}...")
                    results = self.classify_with_track(track_name, onsets, context)
                    track_results[track_name] = results
                    logger.info(f"✅ {track_name} completed with {len(results)} results")
                    
                except Exception as e:
                    logger.error(f"❌ {track_name} failed: {e}")
                    # Continue with other tracks
                    continue
            
            # Combine results using voting system
            if track_results:
                combined_results = self.voting_system.combine_track_results(
                    track_results, voting_method
                )
                
                # Log voting statistics
                stats = self.voting_system.get_voting_statistics(combined_results)
                logger.info(f"🗳️  Voting statistics: {stats['confidence_distribution']}")
                
                return combined_results
            else:
                logger.error("No tracks produced valid results")
                raise ClassificationError("All tracks failed to produce results")
                
        except Exception as e:
            logger.error(f"Multi-track classification failed: {e}")
            raise ClassificationError(f"Multi-track classification failed: {e}")
    
    def classify_with_best_track(self, 
                               onsets: List[tuple],
                               context: Optional[Dict[str, Any]] = None,
                               genre: Optional[str] = None) -> List[ClassificationResult]:
        """
        Classify onsets using the best track for the given context.
        
        Args:
            onsets: List of (audio_window, onset_time) tuples
            context: Additional context information
            genre: Music genre for track selection
            
        Returns:
            List of classification results
        """
        try:
            # Select best track based on context
            if genre:
                best_track = self._select_best_track_for_genre(genre)
            else:
                # Use a good default track
                best_track = 'advanced_features'
            
            logger.info(f"🎯 Using best track: {best_track}")
            
            # Classify with selected track
            return self.classify_with_track(best_track, onsets, context)
            
        except Exception as e:
            logger.error(f"Best track classification failed: {e}")
            # Fall back to simple track
            return self.classify_with_track('magenta_only', onsets, context)
    
    def _select_best_track_for_genre(self, genre: str) -> str:
        """
        Select the best track for a given genre.
        
        Args:
            genre: Music genre
            
        Returns:
            Track name
        """
        genre_lower = genre.lower()
        
        if genre_lower in ['rock', 'metal', 'hard rock', 'heavy metal']:
            # Rock/Metal - use ultimate rock/metal hybrid
            if 'rock_ultimate' in self.available_tracks and self.available_tracks['rock_ultimate'].get('available', False):
                return 'rock_ultimate'
            elif 'ensemble' in self.available_tracks and self.available_tracks['ensemble'].get('available', False):
                return 'ensemble'
            else:
                return 'advanced_features'
        elif genre_lower in ['electronic', 'edm', 'techno', 'house']:
            # Electronic - use multi-scale
            return 'multi_scale'
        elif genre_lower in ['jazz', 'fusion', 'progressive']:
            # Complex genres - use few-shot learning
            return 'few_shot'
        else:
            # Default - use advanced features
            return 'advanced_features'
    
    def get_track_info(self, track_name: str) -> Dict[str, Any]:
        """
        Get information about a specific track.
        
        Args:
            track_name: Name of the track
            
        Returns:
            Track information dictionary
        """
        if track_name in self.active_tracks:
            return self.active_tracks[track_name].get_info()
        elif track_name in self.available_tracks:
            return {
                'name': track_name,
                'description': self.available_tracks[track_name]['description'],
                'available': self.available_tracks[track_name].get('available', True),
                'initialized': False
            }
        else:
            return {'error': f'Unknown track: {track_name}'}
    
    def get_manager_info(self) -> Dict[str, Any]:
        """Get information about the track manager."""
        return {
            'available_tracks': self.get_available_tracks(),
            'active_tracks': list(self.active_tracks.keys()),
            'voting_system': self.voting_system.get_info()
        }
    
    def cleanup(self):
        """Clean up all active tracks."""
        for track_name, track in self.active_tracks.items():
            try:
                track.cleanup()
                logger.debug(f"Cleaned up track: {track_name}")
            except Exception as e:
                logger.warning(f"Cleanup failed for track {track_name}: {e}")
        
        self.active_tracks.clear()
        logger.info("✅ All tracks cleaned up")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()