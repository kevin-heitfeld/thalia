"""
Exploration Constants for Reinforcement Learning.

This module centralizes exploration-related hyperparameters used across
striatum, prefrontal cortex, and other RL-capable regions.

Design Philosophy:
==================
- Named constants improve discoverability and maintainability
- Easy to adjust hyperparameters across entire codebase
- Clear documentation of biological/theoretical motivation

Usage:
======
    from thalia.regulation.exploration_constants import (
        DEFAULT_EPSILON_EXPLORATION,
        UCB_CONFIDENCE_MULTIPLIER,
    )

    # Use in region configuration
    if random.random() < DEFAULT_EPSILON_EXPLORATION:
        action = explore()

Author: Thalia Project
Date: December 2025
"""

# =============================================================================
# Exploration Strategy Parameters
# =============================================================================

# Epsilon-greedy exploration
DEFAULT_EPSILON_EXPLORATION = 0.1  # 10% random exploration
EPSILON_MIN = 0.01  # Minimum epsilon after decay
EPSILON_DECAY = 0.995  # Multiplicative decay per episode

# Upper Confidence Bound (UCB) exploration
UCB_CONFIDENCE_MULTIPLIER = 2.0  # C parameter in UCB formula: sqrt(C * log(N) / n_i)
UCB_MIN_VISITS = 1  # Minimum visits before UCB calculation

# Softmax (Boltzmann) exploration
SOFTMAX_TEMPERATURE_DEFAULT = 1.0  # Default temperature for softmax action selection
SOFTMAX_TEMPERATURE_MIN = 0.1  # Minimum temperature (more deterministic)
SOFTMAX_TEMPERATURE_MAX = 10.0  # Maximum temperature (more random)

# Thompson Sampling
THOMPSON_ALPHA_PRIOR = 1.0  # Beta distribution alpha parameter
THOMPSON_BETA_PRIOR = 1.0  # Beta distribution beta parameter

# =============================================================================
# Learning Rate Presets
# =============================================================================

# Standard learning rates for different learning speeds
LR_VERY_SLOW = 0.0001  # For stable, incremental learning (e.g., late training)
LR_SLOW = 0.001  # Standard slow learning (default for most regions)
LR_MODERATE = 0.01  # Moderate learning (early training, rapid adaptation)
LR_FAST = 0.1  # Fast learning (one-shot learning, critical periods)

# Region-specific defaults (can be overridden in config)
LR_CORTEX_DEFAULT = LR_SLOW  # Cortex: slow, stable learning
LR_HIPPOCAMPUS_DEFAULT = LR_MODERATE  # Hippocampus: faster for episodic memory
LR_STRIATUM_DEFAULT = LR_SLOW  # Striatum: slow RL policy improvement
LR_CEREBELLUM_DEFAULT = LR_MODERATE  # Cerebellum: error-corrective learning
LR_PFC_DEFAULT = LR_SLOW  # PFC: stable working memory maintenance

# =============================================================================
# Exploration Decay Schedules
# =============================================================================

# Linear decay: epsilon = max(EPSILON_MIN, epsilon - EPSILON_LINEAR_DECAY)
EPSILON_LINEAR_DECAY = 0.001  # Subtracted per episode

# Exponential decay: epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
# Already defined above

# Inverse decay: epsilon = EPSILON_MIN + (1 - EPSILON_MIN) / (1 + k * episode)
EPSILON_INVERSE_DECAY_K = 0.01  # Decay rate parameter

# =============================================================================
# Biological Justification
# =============================================================================

# Exploration-exploitation tradeoff is fundamental to biological learning:
#
# 1. **Epsilon-greedy (10% default)**:
#    - Simple, effective baseline
#    - Biological analog: occasional "curiosity-driven" behavior
#    - Used in striatum for action selection
#
# 2. **UCB (C=2.0 default)**:
#    - Optimistic exploration: "try what you're uncertain about"
#    - Biological analog: novelty-seeking behavior, norepinephrine modulation
#    - Used in striatum with action value uncertainty
#
# 3. **Softmax (T=1.0 default)**:
#    - Temperature-controlled probabilistic selection
#    - Biological analog: neural noise, stochastic spiking
#    - Used for smooth policy gradients
#
# References:
# - Sutton & Barto (2018): "Reinforcement Learning: An Introduction"
# - Daw et al. (2006): "Cortical substrates for exploratory decisions in humans"
# - Cohen et al. (2007): "Should I stay or should I go? How the human brain manages the trade-off between exploitation and exploration"
