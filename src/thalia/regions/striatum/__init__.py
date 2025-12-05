"""
Striatum package - Reinforcement Learning with Three-Factor Rule.

This package provides the Striatum region with:
- Three-factor learning (eligibility × dopamine)
- D1/D2 opponent pathways (Go/No-Go)
- Population coding for robust action selection
- Adaptive exploration (UCB + uncertainty-driven)

Usage:
    from thalia.regions.striatum import Striatum, StriatumConfig
"""

from .config import StriatumConfig
from .dopamine import DopamineSystem, EligibilityTraces
from .action_selection import ActionSelectionMixin

# Import Striatum from legacy file until full migration is complete
from thalia.regions._legacy_striatum import Striatum

__all__ = [
    "Striatum",
    "StriatumConfig",
    "DopamineSystem",
    "EligibilityTraces",
    "ActionSelectionMixin",
]
