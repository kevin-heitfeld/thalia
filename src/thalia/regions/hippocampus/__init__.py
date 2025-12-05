"""
Hippocampus package - Trisynaptic Circuit for Episodic Memory.

This package provides the trisynaptic hippocampus (DG→CA3→CA1) with:
- Pattern separation (DG sparse coding)
- Pattern completion (CA3 recurrent attractors)  
- Match/mismatch detection (CA1 coincidence)
- Episodic memory storage and replay

Usage:
    from thalia.regions.hippocampus import (
        TrisynapticHippocampus, 
        TrisynapticConfig,
        Episode,
    )
"""

from .config import Episode, TrisynapticConfig, TrisynapticState

# Import main class from package
from .trisynaptic import TrisynapticHippocampus

__all__ = [
    "TrisynapticHippocampus",
    "TrisynapticConfig", 
    "TrisynapticState",
    "Episode",
]
