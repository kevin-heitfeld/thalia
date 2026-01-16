"""
Striatum package - Reinforcement Learning with Three-Factor Rule.

This package provides the Striatum region with:
- Three-factor learning (eligibility × dopamine)
- D1/D2 opponent pathways (Go/No-Go)
- Population coding for robust action selection
- Adaptive exploration (UCB + uncertainty-driven)
- TD(λ) for multi-step credit assignment (Phase 1 Enhancement)

Usage:
    from thalia.regions.striatum import Striatum, StriatumConfig
    
    # Enable TD(λ) for extended temporal credit assignment:
    from thalia.regions.striatum import TDLambdaLearner, TDLambdaConfig
    config = StriatumConfig(use_td_lambda=True, td_lambda=0.9)
    striatum = Striatum(config)
"""

from __future__ import annotations


from .config import StriatumConfig
from .action_selection import ActionSelectionMixin
from .striatum import Striatum
from .td_lambda import (
    TDLambdaConfig,
    TDLambdaTraces,
    TDLambdaLearner,
    compute_n_step_return,
    compute_lambda_return,
)

__all__ = [
    "Striatum",
    "StriatumConfig",
    "ActionSelectionMixin",
    # TD(λ) components
    "TDLambdaConfig",
    "TDLambdaTraces",
    "TDLambdaLearner",
    "compute_n_step_return",
    "compute_lambda_return",
]
