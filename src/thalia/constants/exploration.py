"""
Exploration Constants - Reinforcement learning exploration strategies.

Author: Thalia Project
Date: January 16, 2026
"""

from __future__ import annotations

# =============================================================================
# Softmax (Boltzmann) Exploration
# =============================================================================

SOFTMAX_TEMPERATURE_DEFAULT = 1.0
"""Default temperature for softmax action selection."""


__all__ = [
    "SOFTMAX_TEMPERATURE_DEFAULT",
]
