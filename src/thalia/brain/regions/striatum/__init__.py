"""
Striatum package - Reinforcement Learning with Three-Factor Rule.

This package provides the Striatum region with:
- Three-factor learning (eligibility × dopamine)
- D1/D2 opponent pathways (Go/No-Go)
- Population coding for robust action selection
- Adaptive exploration (UCB + uncertainty-driven)
"""

from __future__ import annotations

from .striatum import Striatum

__all__ = [
    "Striatum",
]
