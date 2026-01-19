"""Unified Replay System for Thalia.

This module provides a unified architecture for all replay types in the brain,
replacing separate ConsolidationManager, MentalSimulationCoordinator, and
DynaPlanner systems with a single biologically-grounded coordinator.

**Key Insight**: In the brain, all replay uses the same hippocampal CA3→CA1
machinery, differing only in:
- Triggering context (sleep, reward, choice, idle)
- Neuromodulatory state (ACh, NE, DA levels)
- Replay direction (forward, reverse, mixed)
- Compression speed (5-20x)

Components:
- ReplayContext: Enum defining 4 biological replay contexts
- UnifiedReplayCoordinator: Main coordinator for all replay operations

Author: Thalia Project
Date: January 2026
"""

from thalia.replay.contexts import ReplayContext
from thalia.replay.unified_replay import UnifiedReplayCoordinator

__all__ = [
    "ReplayContext",
    "UnifiedReplayCoordinator",
]
