"""
Track implementations for different classification strategies.
"""

from .track3_magenta import MagentaTrack
from .track4_advanced import AdvancedFeaturesTrack
from .track5_multiscale import MultiScaleTrack
from .track6_fewshot import FewShotTrack
from .track7_ensemble import EnsembleTrack
from .track8_augmentation import AugmentationTrack
from .track9_rock_ultimate import RockUltimateTrack

__all__ = [
    "MagentaTrack",
    "AdvancedFeaturesTrack",
    "MultiScaleTrack", 
    "FewShotTrack",
    "EnsembleTrack",
    "AugmentationTrack",
    "RockUltimateTrack",
]