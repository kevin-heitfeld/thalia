"""
Prefrontal Cortex Module - Working Memory and Goal Management.

This module provides the prefrontal cortex implementation with:
- Working memory maintenance via recurrent gating
- Emergent goal representations (NEW - biologically plausible)
- Goal hierarchy management (DEPRECATED - use emergent_goals)
- Context-dependent processing
- Checkpoint management for state persistence

Author: Thalia Project
Date: January 19, 2026
"""

from __future__ import annotations

from .checkpoint_manager import PrefrontalCheckpointManager
from .goal_emergence import EmergentGoalSystem
from .prefrontal import (
    Prefrontal,
    PrefrontalState,
    sample_heterogeneous_wm_neurons,
)

__all__ = [
    "Prefrontal",
    "PrefrontalState",
    "PrefrontalCheckpointManager",
    "EmergentGoalSystem",
    "sample_heterogeneous_wm_neurons",
]
