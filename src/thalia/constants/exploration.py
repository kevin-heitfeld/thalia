"""
Exploration Constants - Reinforcement learning exploration strategies.

Consolidated from regulation/exploration_constants.py.

Author: Thalia Project
Date: January 16, 2026 (Architecture Review Tier 1.2 - Complete Migration)
"""

# =============================================================================
# Epsilon-Greedy Exploration
# =============================================================================

DEFAULT_EPSILON_EXPLORATION = 0.1
"""Default epsilon for epsilon-greedy exploration (10% random actions)."""

EPSILON_MIN = 0.01
"""Minimum epsilon after decay (1% random actions)."""

EPSILON_DECAY = 0.995
"""Multiplicative decay per episode."""

EPSILON_LINEAR_DECAY = 0.001
"""Linear decay subtracted per episode."""

EPSILON_INVERSE_DECAY_K = 0.01
"""Decay rate parameter for inverse decay schedule."""

# =============================================================================
# Upper Confidence Bound (UCB) Exploration
# =============================================================================

UCB_CONFIDENCE_MULTIPLIER = 2.0
"""C parameter in UCB formula: sqrt(C * log(N) / n_i)."""

UCB_MIN_VISITS = 1
"""Minimum visits before UCB calculation."""

# =============================================================================
# Softmax (Boltzmann) Exploration
# =============================================================================

SOFTMAX_TEMPERATURE_DEFAULT = 1.0
"""Default temperature for softmax action selection."""

SOFTMAX_TEMPERATURE_MIN = 0.1
"""Minimum temperature (more deterministic)."""

SOFTMAX_TEMPERATURE_MAX = 10.0
"""Maximum temperature (more random)."""

# =============================================================================
# Thompson Sampling
# =============================================================================

THOMPSON_ALPHA_PRIOR = 1.0
"""Beta distribution alpha parameter."""

THOMPSON_BETA_PRIOR = 1.0
"""Beta distribution beta parameter."""


__all__ = [
    "DEFAULT_EPSILON_EXPLORATION",
    "EPSILON_MIN",
    "EPSILON_DECAY",
    "EPSILON_LINEAR_DECAY",
    "EPSILON_INVERSE_DECAY_K",
    "UCB_CONFIDENCE_MULTIPLIER",
    "UCB_MIN_VISITS",
    "SOFTMAX_TEMPERATURE_DEFAULT",
    "SOFTMAX_TEMPERATURE_MIN",
    "SOFTMAX_TEMPERATURE_MAX",
    "THOMPSON_ALPHA_PRIOR",
    "THOMPSON_BETA_PRIOR",
]
