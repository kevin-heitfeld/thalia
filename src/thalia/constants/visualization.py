"""
Visualization Constants - UI positioning, plot alphas, thresholds.

Author: Thalia Project
Date: January 16, 2026
"""

from __future__ import annotations

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

ALPHA_TRANSPARENT = 0.3
"""General transparency value for backgrounds."""

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
# NETWORK GRAPH VISUALIZATION
# =============================================================================

DEFAULT_NODE_SIZE_SCALE = 1000.0
"""Default scaling factor for node sizes in network graphs.

Multiplied by normalized node importance/activity to determine visual size.
"""

NODE_ALPHA_DEFAULT = 0.8
"""Default alpha (transparency) for network nodes.

Range: [0.0, 1.0] where 1.0 is fully opaque.
"""

EDGE_WIDTH_SCALE = 5.0
"""Scaling factor for edge widths based on connection strength.

Edge width = (weight / max_weight) * EDGE_WIDTH_SCALE
"""

EDGE_ALPHA_DEFAULT = 0.6
"""Default alpha (transparency) for network edges.

Range: [0.0, 1.0] where 1.0 is fully opaque.
Lower than nodes for better visibility of node-edge structure.
"""

LAYOUT_K_FACTOR = 2.0
"""Spring layout optimal distance parameter.

Higher values = more spread out nodes.
Used in networkx spring_layout(k=...).
"""

LAYOUT_ITERATIONS = 50
"""Number of iterations for spring layout algorithm.

More iterations = better layout but slower computation.
"""

ARC_RADIUS = 0.1
"""Radius for curved edges in directed graphs.

Used in connectionstyle parameter: 'arc3,rad=0.1'
Prevents overlapping edges between same node pairs.
"""

LEGEND_FRAMEALPHA = 0.9
"""Alpha (transparency) for legend background.

Nearly opaque to ensure legend readability.
"""

HIERARCHICAL_Y_SPACING = 2.0
"""Vertical spacing between layers in hierarchical layouts."""

HIERARCHICAL_X_SPACING = 3.0
"""Horizontal spacing between nodes in same layer (hierarchical layouts)."""


__all__ = [
    # Text positioning
    "TEXT_POSITION_CENTER",
    "TEXT_POSITION_BOTTOM_RIGHT_X",
    "TEXT_POSITION_BOTTOM_RIGHT_Y",
    "TEXT_POSITION_TOP_LEFT",
    # UI dimensions
    "PROGRESS_BAR_HEIGHT",
    "AXIS_MARGIN_POSITIVE",
    "AXIS_MARGIN_NEGATIVE",
    # Alpha values
    "ALPHA_SEMI_TRANSPARENT",
    "ALPHA_TRANSPARENT",
    # Performance thresholds
    "PERFORMANCE_EXCELLENT",
    "PERFORMANCE_GOOD",
    "PERFORMANCE_ACCEPTABLE",
    "PERFORMANCE_POOR",
    # Calibration thresholds
    "CALIBRATION_EXCELLENT_ECE",
    "CALIBRATION_GOOD_ECE",
    # Network graph - nodes
    "DEFAULT_NODE_SIZE_SCALE",
    "NODE_ALPHA_DEFAULT",
    # Network graph - edges
    "EDGE_WIDTH_SCALE",
    "EDGE_ALPHA_DEFAULT",
    # Network graph - layout
    "LAYOUT_K_FACTOR",
    "LAYOUT_ITERATIONS",
    "ARC_RADIUS",
    "LEGEND_FRAMEALPHA",
    "HIERARCHICAL_Y_SPACING",
    "HIERARCHICAL_X_SPACING",
]
