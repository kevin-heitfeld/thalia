"""
Prefrontal Cortex Module - Working Memory and Goal Management.

This module provides the prefrontal cortex implementation with:
- Working memory maintenance via recurrent gating
- Emergent goal representations
- Context-dependent processing
"""

from __future__ import annotations

from .goal_emergence import EmergentGoalSystem
from .prefrontal import Prefrontal

__all__ = [
    "Prefrontal",
    "EmergentGoalSystem",
]
