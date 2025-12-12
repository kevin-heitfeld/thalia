"""
Task Constants - Extracted magic numbers for behavioral tasks.

This module centralizes numeric constants used across task generation,
replacing scattered magic numbers with named, documented values.

Author: Thalia Project
Date: December 12, 2025
"""

# =============================================================================
# Spike Probabilities (for motor output generation)
# =============================================================================

SPIKE_PROBABILITY_LOW = 0.05
"""Low spike probability for minimal motor activity (5%)."""

SPIKE_PROBABILITY_MEDIUM = 0.2
"""Medium spike probability for moderate motor activity (20%)."""

SPIKE_PROBABILITY_HIGH = 0.4
"""High spike probability for strong motor activity (40%)."""

# =============================================================================
# Stimulus Strengths
# =============================================================================

STIMULUS_STRENGTH_HIGH = 1.0
"""High stimulus strength for salient/attended stimuli."""

STIMULUS_STRENGTH_MEDIUM = 0.5
"""Medium stimulus strength for neutral stimuli."""

STIMULUS_STRENGTH_LOW = 0.2
"""Low stimulus strength for weak/background stimuli."""

# =============================================================================
# Noise Scales
# =============================================================================

PROPRIOCEPTION_NOISE_SCALE = 0.1
"""Noise scale for proprioceptive feedback (10% of signal)."""

WEIGHT_INIT_SCALE_SMALL = 0.01
"""Small weight initialization scale for fine-grained variation."""

WEIGHT_INIT_SCALE_MEDIUM = 0.05
"""Medium weight initialization scale for moderate variation."""

STIMULUS_NOISE_SCALE = 0.05
"""General stimulus noise scale for task variation."""

# =============================================================================
# Feature Variation (for visual/cognitive tasks)
# =============================================================================

FEATURE_INCREMENT_BASE = 0.2
"""Base increment for progressive feature changes."""

FEATURE_INCREMENT_COLUMN = 0.15
"""Column-wise feature increment for spatial patterns."""

FEATURE_INCREMENT_INTERACTION = 0.1
"""Interaction term increment for combinatorial features."""

FEATURE_NOISE_MATCH = 0.3
"""Noise scale for creating near-match stimuli."""

# =============================================================================
# Spatial Parameters
# =============================================================================

TARGET_POSITION_MARGIN = 0.1
"""Margin for target position validation (10% of workspace)."""

REACHING_SUCCESS_THRESHOLD = 0.05
"""Distance threshold for successful reaching (5% of workspace)."""
