"""
Exploration Constants - Reinforcement learning exploration strategies.

Author: Thalia Project
Date: January 16, 2026
"""

from __future__ import annotations

# =============================================================================
# Epsilon-Greedy Exploration
# =============================================================================

DEFAULT_EPSILON_EXPLORATION = 0.1
"""Default epsilon for epsilon-greedy exploration (10% random actions)."""

# =============================================================================
# Softmax (Boltzmann) Exploration
# =============================================================================

SOFTMAX_TEMPERATURE_DEFAULT = 1.0
"""Default temperature for softmax action selection."""


__all__ = [
    "DEFAULT_EPSILON_EXPLORATION",
    "SOFTMAX_TEMPERATURE_DEFAULT",
]
