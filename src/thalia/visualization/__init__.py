"""
Thalia Visualization Package.

Provides beautiful visualizations of brain activity using Manim.
"""

from .manim_brain import (
    BrainActivityVisualization,
    BrainArchitectureScene,
    SpikeActivityScene,
    LearningScene,
    GrowthScene,
    MANIM_AVAILABLE,
)

__all__ = [
    'BrainActivityVisualization',
    'BrainArchitectureScene',
    'SpikeActivityScene',
    'LearningScene',
    'GrowthScene',
    'MANIM_AVAILABLE',
]
