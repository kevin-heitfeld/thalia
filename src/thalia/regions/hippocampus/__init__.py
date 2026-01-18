"""
Hippocampus package - Trisynaptic Circuit for Episodic Memory.

This package provides the trisynaptic hippocampus (DG→CA3→CA2→CA1) with:
- Pattern separation (DG sparse coding)
- Pattern completion (CA3 recurrent attractors)
- Match/mismatch detection (CA1 coincidence)
- Episodic memory storage and replay

Usage:
    from thalia.regions.hippocampus import (
        TrisynapticHippocampus,
        Episode,
    )
"""

from __future__ import annotations

from .memory_component import Episode
from .replay_engine import ReplayConfig, ReplayEngine, ReplayMode
from .trisynaptic import HippocampusState, TrisynapticHippocampus

__all__ = [
    "TrisynapticHippocampus",
    "HippocampusState",
    "Episode",
    "ReplayEngine",
    "ReplayConfig",
    "ReplayMode",
]
