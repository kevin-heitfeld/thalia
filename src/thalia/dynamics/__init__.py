"""
Attractor dynamics and manifold representations.

This module provides tools for analyzing and creating attractor-based
neural dynamics, enabling pattern storage, recall, and spontaneous
thought-like transitions.
"""

from .attractor import AttractorNetwork, AttractorConfig
from .manifold import ActivityTracker, ThoughtTrajectory

__all__ = [
    "AttractorNetwork",
    "AttractorConfig",
    "ActivityTracker",
    "ThoughtTrajectory",
]
