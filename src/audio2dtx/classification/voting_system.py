"""
Voting system for combining results from multiple classification tracks.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass

from .base_classifier import ClassificationResult
from ..config.settings import Settings
from ..utils.exceptions import ClassificationError
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrackResult:
    """Result from a single classification track."""
    track_name: str
    results: List[ClassificationResult]
    track_weight: float = 1.0
    track_confidence: float = 0.0


class VotingSystem:
    """
    Voting system for combining classification results from multiple tracks.
    
    Implements weighted voting with confidence-based adjustment and
    track-specific optimization.
    """
    
    def __init__(self, settings: Settings):
        """
        Initialize voting system.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        
        # Default track weights (can be adjusted based on performance)
        self.track_weights = {
            'magenta_only': 0.15,      # Simple but consistent
            'advanced_features': 0.20,  # Rich feature analysis
            'multi_scale': 0.15,       # Temporal precision
            'few_shot': 0.10,          # Song-specific adaptation
            'ensemble': 0.25,          # Hierarchical expertise
            'augmentation': 0.10,      # Robustness
            'rock_ultimate': 0.05      # Genre-specific (when not rock/metal)
        }
        
        # Confidence thresholds for different voting strategies
        self.high_confidence_threshold = 0.8
        self.medium_confidence_threshold = 0.6
        self.low_confidence_threshold = 0.4
        
        # Track performance history for adaptive weighting
        self.track_performance = defaultdict(list)
        
    def combine_track_results(self, 
                            track_results: Dict[str, List[ClassificationResult]],
                            voting_method: str = 'weighted_confidence') -> List[ClassificationResult]:
        """
        Combine results from multiple tracks using specified voting method.
        
        Args:
            track_results: Dictionary mapping track names to their results
            voting_method: Voting method ('weighted_confidence', 'majority', 'confidence_threshold')
            
        Returns:
            List of combined classification results
        """
        try:
            logger.info(f"🗳️  Combining results from {len(track_results)} tracks using {voting_method}")
            
            # Validate input
            if not track_results:
                raise ClassificationError("No track results provided for voting")
            
            # Get the number of onsets (should be consistent across tracks)
            onset_count = self._validate_and_get_onset_count(track_results)
            
            # Choose voting method
            if voting_method == 'weighted_confidence':
                combined_results = self._weighted_confidence_voting(track_results, onset_count)
            elif voting_method == 'majority':
                combined_results = self._majority_voting(track_results, onset_count)
            elif voting_method == 'confidence_threshold':
                combined_results = self._confidence_threshold_voting(track_results, onset_count)
            elif voting_method == 'adaptive_weighted':
                combined_results = self._adaptive_weighted_voting(track_results, onset_count)
            else:
                logger.warning(f"Unknown voting method: {voting_method}, using weighted_confidence")
                combined_results = self._weighted_confidence_voting(track_results, onset_count)
            
            logger.info(f"✅ Combined {len(combined_results)} classifications")
            return combined_results
            
        except Exception as e:
            logger.error(f"Voting system failed: {e}")
            # Return first available track's results as fallback
            if track_results:
                fallback_track = list(track_results.keys())[0]
                return track_results[fallback_track]
            else:
                return []
    
    def _validate_and_get_onset_count(self, track_results: Dict[str, List[ClassificationResult]]) -> int:
        """Validate track results and return consistent onset count."""
        onset_counts = [len(results) for results in track_results.values()]
        
        if not onset_counts:
            raise ClassificationError("No classification results found")
        
        # Check if all tracks have the same number of results
        if len(set(onset_counts)) > 1:
            logger.warning(f"Inconsistent onset counts across tracks: {onset_counts}")
            # Use the minimum count to avoid index errors
            onset_count = min(onset_counts)
        else:
            onset_count = onset_counts[0]
        
        return onset_count
    
    def _weighted_confidence_voting(self, 
                                  track_results: Dict[str, List[ClassificationResult]],
                                  onset_count: int) -> List[ClassificationResult]:
        """
        Weighted voting based on track weights and individual confidences.
        
        Args:
            track_results: Track results dictionary
            onset_count: Number of onsets to process
            
        Returns:
            List of combined results
        """
        combined_results = []
        
        for onset_idx in range(onset_count):
            # Collect votes for this onset
            votes = defaultdict(float)
            total_weight = 0
            
            for track_name, results in track_results.items():
                if onset_idx < len(results):
                    result = results[onset_idx]
                    track_weight = self.track_weights.get(track_name, 1.0)
                    
                    # Weight by track importance and individual confidence
                    vote_weight = track_weight * result.confidence
                    votes[result.instrument] += vote_weight
                    total_weight += vote_weight
            
            # Find winning instrument
            if votes:
                winner = max(votes, key=votes.get)
                winner_score = votes[winner]
                
                # Calculate combined confidence
                if total_weight > 0:
                    combined_confidence = winner_score / total_weight
                else:
                    combined_confidence = 0.5
                
                # Calculate average velocity from participating tracks
                velocities = []
                for track_name, results in track_results.items():
                    if onset_idx < len(results) and results[onset_idx].instrument == winner:
                        velocities.append(results[onset_idx].velocity)
                
                avg_velocity = np.mean(velocities) if velocities else 0.5
                
                # Create combined result
                combined_result = ClassificationResult(
                    instrument=winner,
                    confidence=float(combined_confidence),
                    velocity=float(avg_velocity),
                    features={
                        'source': 'weighted_voting',
                        'participating_tracks': list(track_results.keys()),
                        'vote_distribution': dict(votes),
                        'total_weight': total_weight
                    }
                )
                
                combined_results.append(combined_result)
            else:
                # Fallback result
                combined_results.append(ClassificationResult(
                    instrument='kick',
                    confidence=0.1,
                    velocity=0.5,
                    features={'source': 'voting_fallback'}
                ))
        
        return combined_results
    
    def _majority_voting(self, 
                        track_results: Dict[str, List[ClassificationResult]],
                        onset_count: int) -> List[ClassificationResult]:
        """
        Simple majority voting - most common prediction wins.
        
        Args:
            track_results: Track results dictionary
            onset_count: Number of onsets to process
            
        Returns:
            List of combined results
        """
        combined_results = []
        
        for onset_idx in range(onset_count):
            # Collect predictions for this onset
            predictions = []
            confidences = []
            velocities = []
            
            for track_name, results in track_results.items():
                if onset_idx < len(results):
                    result = results[onset_idx]
                    predictions.append(result.instrument)
                    confidences.append(result.confidence)
                    velocities.append(result.velocity)
            
            if predictions:
                # Count votes
                vote_counts = Counter(predictions)
                winner = vote_counts.most_common(1)[0][0]
                winner_count = vote_counts[winner]
                
                # Calculate confidence based on consensus
                consensus_ratio = winner_count / len(predictions)
                
                # Get average confidence and velocity for winner
                winner_confidences = [conf for pred, conf in zip(predictions, confidences) if pred == winner]
                winner_velocities = [vel for pred, vel in zip(predictions, velocities) if pred == winner]
                
                avg_confidence = np.mean(winner_confidences) if winner_confidences else 0.5
                avg_velocity = np.mean(winner_velocities) if winner_velocities else 0.5
                
                # Boost confidence based on consensus
                final_confidence = avg_confidence * consensus_ratio
                
                combined_result = ClassificationResult(
                    instrument=winner,
                    confidence=float(final_confidence),
                    velocity=float(avg_velocity),
                    features={
                        'source': 'majority_voting',
                        'consensus_ratio': consensus_ratio,
                        'vote_counts': dict(vote_counts),
                        'participating_tracks': list(track_results.keys())
                    }
                )
                
                combined_results.append(combined_result)
            else:
                # Fallback
                combined_results.append(ClassificationResult(
                    instrument='kick',
                    confidence=0.1,
                    velocity=0.5,
                    features={'source': 'majority_fallback'}
                ))
        
        return combined_results
    
    def _confidence_threshold_voting(self, 
                                   track_results: Dict[str, List[ClassificationResult]],
                                   onset_count: int) -> List[ClassificationResult]:
        """
        Confidence-based voting - high confidence predictions get priority.
        
        Args:
            track_results: Track results dictionary
            onset_count: Number of onsets to process
            
        Returns:
            List of combined results
        """
        combined_results = []
        
        for onset_idx in range(onset_count):
            # Collect all predictions for this onset
            all_predictions = []
            
            for track_name, results in track_results.items():
                if onset_idx < len(results):
                    result = results[onset_idx]
                    all_predictions.append((result, track_name))
            
            if all_predictions:
                # Sort by confidence (highest first)
                all_predictions.sort(key=lambda x: x[0].confidence, reverse=True)
                
                # Use highest confidence prediction if above threshold
                best_result, best_track = all_predictions[0]
                
                if best_result.confidence >= self.high_confidence_threshold:
                    # Use best prediction as-is
                    combined_result = ClassificationResult(
                        instrument=best_result.instrument,
                        confidence=best_result.confidence,
                        velocity=best_result.velocity,
                        features={
                            'source': 'confidence_threshold',
                            'selected_track': best_track,
                            'selection_reason': 'high_confidence'
                        }
                    )
                elif best_result.confidence >= self.medium_confidence_threshold:
                    # Look for supporting predictions
                    supporting_predictions = [
                        pred for pred, _ in all_predictions 
                        if pred.instrument == best_result.instrument and pred.confidence >= self.medium_confidence_threshold
                    ]
                    
                    if len(supporting_predictions) >= 2:
                        # Multiple tracks agree with medium confidence
                        avg_confidence = np.mean([p.confidence for p in supporting_predictions])
                        avg_velocity = np.mean([p.velocity for p in supporting_predictions])
                        
                        combined_result = ClassificationResult(
                            instrument=best_result.instrument,
                            confidence=float(avg_confidence),
                            velocity=float(avg_velocity),
                            features={
                                'source': 'confidence_threshold',
                                'selection_reason': 'medium_confidence_consensus',
                                'supporting_tracks': len(supporting_predictions)
                            }
                        )
                    else:
                        # Fall back to weighted voting
                        combined_result = self._weighted_confidence_voting(
                            {k: [v[onset_idx]] for k, v in track_results.items() if onset_idx < len(v)}, 1
                        )[0]
                else:
                    # All predictions have low confidence, use weighted voting
                    combined_result = self._weighted_confidence_voting(
                        {k: [v[onset_idx]] for k, v in track_results.items() if onset_idx < len(v)}, 1
                    )[0]
                
                combined_results.append(combined_result)
            else:
                # Fallback
                combined_results.append(ClassificationResult(
                    instrument='kick',
                    confidence=0.1,
                    velocity=0.5,
                    features={'source': 'confidence_fallback'}
                ))
        
        return combined_results
    
    def _adaptive_weighted_voting(self, 
                                track_results: Dict[str, List[ClassificationResult]],
                                onset_count: int) -> List[ClassificationResult]:
        """
        Adaptive weighted voting that adjusts track weights based on performance.
        
        Args:
            track_results: Track results dictionary
            onset_count: Number of onsets to process
            
        Returns:
            List of combined results
        """
        # For now, use standard weighted voting
        # In the future, this could adapt weights based on track performance
        return self._weighted_confidence_voting(track_results, onset_count)
    
    def update_track_performance(self, 
                               track_name: str,
                               performance_metrics: Dict[str, float]):
        """
        Update performance metrics for a track.
        
        Args:
            track_name: Name of the track
            performance_metrics: Dictionary of performance metrics
        """
        self.track_performance[track_name].append(performance_metrics)
        
        # Keep only recent performance data
        if len(self.track_performance[track_name]) > 100:
            self.track_performance[track_name] = self.track_performance[track_name][-100:]
    
    def get_track_weights(self) -> Dict[str, float]:
        """Get current track weights."""
        return self.track_weights.copy()
    
    def set_track_weights(self, weights: Dict[str, float]):
        """
        Set track weights.
        
        Args:
            weights: Dictionary of track weights
        """
        self.track_weights.update(weights)
        logger.info(f"Updated track weights: {self.track_weights}")
    
    def optimize_weights_for_genre(self, genre: str):
        """
        Optimize track weights for specific genre.
        
        Args:
            genre: Music genre
        """
        if genre.lower() in ['rock', 'metal', 'hard rock', 'heavy metal']:
            # Boost rock-specific tracks
            self.track_weights['rock_ultimate'] = 0.3
            self.track_weights['ensemble'] = 0.2
            self.track_weights['advanced_features'] = 0.15
            self.track_weights['multi_scale'] = 0.15
            self.track_weights['magenta_only'] = 0.1
            self.track_weights['augmentation'] = 0.05
            self.track_weights['few_shot'] = 0.05
            
            logger.info(f"Optimized weights for {genre} genre")
        elif genre.lower() in ['electronic', 'edm', 'techno', 'house']:
            # Boost tracks good for electronic music
            self.track_weights['multi_scale'] = 0.25
            self.track_weights['augmentation'] = 0.2
            self.track_weights['advanced_features'] = 0.2
            self.track_weights['ensemble'] = 0.15
            self.track_weights['magenta_only'] = 0.1
            self.track_weights['few_shot'] = 0.05
            self.track_weights['rock_ultimate'] = 0.05
            
            logger.info(f"Optimized weights for {genre} genre")
        else:
            # Use default balanced weights
            logger.info(f"Using default balanced weights for {genre} genre")
    
    def get_voting_statistics(self, results: List[ClassificationResult]) -> Dict[str, Any]:
        """
        Get statistics about the voting process.
        
        Args:
            results: Combined classification results
            
        Returns:
            Dictionary with voting statistics
        """
        stats = {
            'total_classifications': len(results),
            'source_distribution': Counter(),
            'confidence_distribution': {
                'high': 0,
                'medium': 0,
                'low': 0
            },
            'instrument_distribution': Counter()
        }
        
        for result in results:
            # Count sources
            source = result.features.get('source', 'unknown')
            stats['source_distribution'][source] += 1
            
            # Count confidence levels
            if result.confidence >= self.high_confidence_threshold:
                stats['confidence_distribution']['high'] += 1
            elif result.confidence >= self.medium_confidence_threshold:
                stats['confidence_distribution']['medium'] += 1
            else:
                stats['confidence_distribution']['low'] += 1
            
            # Count instruments
            stats['instrument_distribution'][result.instrument] += 1
        
        return stats
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the voting system."""
        return {
            'track_weights': self.track_weights,
            'confidence_thresholds': {
                'high': self.high_confidence_threshold,
                'medium': self.medium_confidence_threshold,
                'low': self.low_confidence_threshold
            },
            'performance_history_size': {
                track: len(history) for track, history in self.track_performance.items()
            }
        }