"""
Visualization Constants - UI positioning and threshold values.

This module centralizes constants used in training visualization,
replacing magic numbers with named, documented values.

Author: Thalia Project
Date: December 12, 2025
"""

# =============================================================================
# Text Positioning
# =============================================================================

TEXT_POSITION_CENTER = 0.5
"""Center position for text placement in plots."""

TEXT_POSITION_BOTTOM_RIGHT_X = 0.98
"""X position for bottom-right text (right-aligned)."""

TEXT_POSITION_BOTTOM_RIGHT_Y = 0.98
"""Y position for bottom-right text (top-aligned)."""

TEXT_POSITION_TOP_LEFT = 0.1
"""Position for top-left text placement."""

# =============================================================================
# UI Element Dimensions
# =============================================================================

PROGRESS_BAR_HEIGHT = 0.5
"""Height of horizontal progress bars."""

AXIS_MARGIN_POSITIVE = 0.5
"""Positive axis margin for plot boundaries."""

AXIS_MARGIN_NEGATIVE = -0.5
"""Negative axis margin for plot boundaries."""

ALPHA_SEMI_TRANSPARENT = 0.5
"""Alpha value for semi-transparent overlays."""

ALPHA_HIGHLIGHT = 0.2
"""Alpha value for subtle highlight regions."""

# =============================================================================
# Biological Firing Rate Thresholds
# =============================================================================

TARGET_SPIKE_RATE_LOWER = 0.05
"""Lower bound for target spike rate (5% - sparse coding)."""

TARGET_SPIKE_RATE_UPPER = 0.15
"""Upper bound for target spike rate (15% - active coding)."""

FIRING_RATE_SILENCE_THRESHOLD = 0.01
"""Below this rate, region is considered silent (1%)."""

FIRING_RATE_RUNAWAY_THRESHOLD = 0.9
"""Above this rate, region shows runaway excitation (90%)."""

# =============================================================================
# Performance Thresholds
# =============================================================================

PERFORMANCE_EXCELLENT = 0.95
"""Excellent performance threshold (95% accuracy)."""

PERFORMANCE_GOOD = 0.90
"""Good performance threshold (90% accuracy)."""

PERFORMANCE_ACCEPTABLE = 0.85
"""Acceptable performance threshold (85% accuracy)."""

PERFORMANCE_POOR = 0.70
"""Poor performance threshold (70% accuracy)."""

# =============================================================================
# Calibration Thresholds (Metacognition)
# =============================================================================

CALIBRATION_EXCELLENT_ECE = 0.10
"""Expected Calibration Error threshold for excellent calibration."""

CALIBRATION_GOOD_ECE = 0.15
"""Expected Calibration Error threshold for good calibration."""

DIFFICULTY_RANGE_MIN = 0.3
"""Minimum difficulty for task generation (30%)."""

DIFFICULTY_RANGE_MAX = 0.9
"""Maximum difficulty for task generation (90%)."""
