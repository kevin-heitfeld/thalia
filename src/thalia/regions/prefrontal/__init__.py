"""
Prefrontal Cortex Module - Working Memory and Goal Management.

This module provides the prefrontal cortex implementation with:
- Working memory maintenance via recurrent gating
- Goal hierarchy management
- Context-dependent processing
- Checkpoint management for state persistence

Author: Thalia Project
Date: January 16, 2026
"""

from __future__ import annotations

from thalia.regions.prefrontal.checkpoint_manager import PrefrontalCheckpointManager
from thalia.regions.prefrontal.hierarchy import (
    Goal,
    GoalStatus,
)
from thalia.regions.prefrontal.prefrontal import (
    Prefrontal,
    PrefrontalConfig,
    PrefrontalState,
    sample_heterogeneous_wm_neurons,
)

__all__ = [
    "Prefrontal",
    "PrefrontalConfig",
    "PrefrontalState",
    "sample_heterogeneous_wm_neurons",
    "Goal",
    "GoalStatus",
    "PrefrontalCheckpointManager",
]
