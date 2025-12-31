"""
Hippocampus package - Trisynaptic Circuit for Episodic Memory.

This package provides the trisynaptic hippocampus (DG→CA3→CA2→CA1) with:
- Pattern separation (DG sparse coding)
- Pattern completion (CA3 recurrent attractors)
- Match/mismatch detection (CA1 coincidence)
- Episodic memory storage and replay

Usage:
    from thalia.regions.hippocampus import (
        TrisynapticHippocampus,  # or: Hippocampus
        Episode,
    )
"""

from .config import (
    Episode,
    HippocampusConfig,
    HippocampusState,
)
from .replay_engine import ReplayEngine, ReplayConfig, ReplayMode

# Import main class from package
from .trisynaptic import TrisynapticHippocampus

# Preferred alias
Hippocampus = TrisynapticHippocampus

__all__ = [
    # Primary names (preferred)
    "Hippocampus",
    "HippocampusConfig",
    "HippocampusState",
    "Episode",
    "ReplayEngine",
    "ReplayConfig",
    "ReplayMode",
]
