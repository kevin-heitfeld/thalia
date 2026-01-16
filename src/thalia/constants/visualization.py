"""
Visualization Constants - UI positioning, plot alphas, thresholds.

Consolidated from training/visualization/constants.py.

Author: Thalia Project
Date: January 16, 2026 (Architecture Review Tier 1.2)
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

# =============================================================================
# Alpha (Transparency) Values
# =============================================================================

ALPHA_SEMI_TRANSPARENT = 0.5
"""Alpha value for semi-transparent overlays."""

ALPHA_HIGHLIGHT = 0.2
"""Alpha value for subtle highlight regions."""

ALPHA_GRID = 0.3
"""Alpha value for grid lines."""

ALPHA_TRANSPARENT = 0.3
"""General transparency value for backgrounds."""

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

# =============================================================================
# Difficulty Range
# =============================================================================

DIFFICULTY_RANGE_MIN = 0.3
"""Minimum difficulty for task generation (30%)."""

DIFFICULTY_RANGE_MAX = 0.9
"""Maximum difficulty for task generation (90%)."""

# =============================================================================
# Plot Configuration
# =============================================================================

LINE_WIDTH_STANDARD = 2.0
"""Standard line width for plots."""

LINE_WIDTH_THIN = 1.0
"""Thin line width for secondary elements."""

LINE_WIDTH_HIGHLIGHT = 0.5
"""Very thin line width for background elements."""


__all__ = [
    "TEXT_POSITION_CENTER",
    "TEXT_POSITION_BOTTOM_RIGHT_X",
    "TEXT_POSITION_BOTTOM_RIGHT_Y",
    "TEXT_POSITION_TOP_LEFT",
    "PROGRESS_BAR_HEIGHT",
    "AXIS_MARGIN_POSITIVE",
    "AXIS_MARGIN_NEGATIVE",
    "ALPHA_SEMI_TRANSPARENT",
    "ALPHA_HIGHLIGHT",
    "ALPHA_GRID",
    "ALPHA_TRANSPARENT",
    "TARGET_SPIKE_RATE_LOWER",
    "TARGET_SPIKE_RATE_UPPER",
    "FIRING_RATE_SILENCE_THRESHOLD",
    "FIRING_RATE_RUNAWAY_THRESHOLD",
    "PERFORMANCE_EXCELLENT",
    "PERFORMANCE_GOOD",
    "PERFORMANCE_ACCEPTABLE",
    "PERFORMANCE_POOR",
    "CALIBRATION_EXCELLENT_ECE",
    "CALIBRATION_GOOD_ECE",
    "DIFFICULTY_RANGE_MIN",
    "DIFFICULTY_RANGE_MAX",
    "LINE_WIDTH_STANDARD",
    "LINE_WIDTH_THIN",
    "LINE_WIDTH_HIGHLIGHT",
]
