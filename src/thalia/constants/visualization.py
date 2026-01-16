"""
Visualization Constants - UI positioning, plot alphas, thresholds.

Consolidated from training/visualization/constants.py.

Author: Thalia Project
Date: January 16, 2026 (Architecture Review Tier 1.2)
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

# =============================================================================
# NETWORK GRAPH VISUALIZATION
# =============================================================================

# Node visualization
DEFAULT_NODE_SIZE_SCALE = 1000.0
"""Default scaling factor for node sizes in network graphs.

Multiplied by normalized node importance/activity to determine visual size.
"""

NODE_ALPHA_DEFAULT = 0.8
"""Default alpha (transparency) for network nodes.

Range: [0.0, 1.0] where 1.0 is fully opaque.
"""

# Edge visualization
EDGE_WIDTH_SCALE = 5.0
"""Scaling factor for edge widths based on connection strength.

Edge width = (weight / max_weight) * EDGE_WIDTH_SCALE
"""

EDGE_ALPHA_DEFAULT = 0.6
"""Default alpha (transparency) for network edges.

Range: [0.0, 1.0] where 1.0 is fully opaque.
Lower than nodes for better visibility of node-edge structure.
"""

# Layout parameters
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

# Legend and labels
LEGEND_FRAMEALPHA = 0.9
"""Alpha (transparency) for legend background.

Nearly opaque to ensure legend readability.
"""

# Default spacing for hierarchical layouts
HIERARCHICAL_Y_SPACING = 2.0
"""Vertical spacing between layers in hierarchical layouts."""

HIERARCHICAL_X_SPACING = 3.0
"""Horizontal spacing between nodes in same layer (hierarchical layouts)."""

# =============================================================================
# DIAGNOSTIC PLOTS
# =============================================================================

# Time series plots
DEFAULT_LINE_WIDTH = 1.5
"""Default line width for time series plots."""

SPIKE_RASTER_MARKER_SIZE = 1.0
"""Marker size for spike raster plots."""

# Heatmaps
HEATMAP_CMAP_DIVERGING = "RdBu_r"
"""Colormap for diverging data (positive and negative values)."""

HEATMAP_CMAP_SEQUENTIAL = "viridis"
"""Colormap for sequential data (0 to max)."""

# =============================================================================
# FIGURE SIZING
# =============================================================================

FIGURE_SIZE_SMALL = (6, 4)
"""Small figure size (width, height) in inches."""

FIGURE_SIZE_MEDIUM = (10, 6)
"""Medium figure size (width, height) in inches."""

FIGURE_SIZE_LARGE = (14, 8)
"""Large figure size (width, height) in inches."""

DPI_DEFAULT = 100
"""Default DPI (dots per inch) for saved figures."""

DPI_MEDIUM = 150
"""Medium resolution DPI for screen display and quick saves."""

DPI_HIGH_RES = 300
"""High resolution DPI for publication-quality figures."""


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
    "ALPHA_HIGHLIGHT",
    "ALPHA_GRID",
    "ALPHA_TRANSPARENT",
    # Firing rate thresholds
    "TARGET_SPIKE_RATE_LOWER",
    "TARGET_SPIKE_RATE_UPPER",
    "FIRING_RATE_SILENCE_THRESHOLD",
    "FIRING_RATE_RUNAWAY_THRESHOLD",
    # Performance thresholds
    "PERFORMANCE_EXCELLENT",
    "PERFORMANCE_GOOD",
    "PERFORMANCE_ACCEPTABLE",
    "PERFORMANCE_POOR",
    # Calibration thresholds
    "CALIBRATION_EXCELLENT_ECE",
    "CALIBRATION_GOOD_ECE",
    # Difficulty range
    "DIFFICULTY_RANGE_MIN",
    "DIFFICULTY_RANGE_MAX",
    # Plot line widths
    "LINE_WIDTH_STANDARD",
    "LINE_WIDTH_THIN",
    "LINE_WIDTH_HIGHLIGHT",
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
    # Diagnostic plots
    "DEFAULT_LINE_WIDTH",
    "SPIKE_RASTER_MARKER_SIZE",
    "HEATMAP_CMAP_DIVERGING",
    "HEATMAP_CMAP_SEQUENTIAL",
    # Figure sizing
    "FIGURE_SIZE_SMALL",
    "FIGURE_SIZE_MEDIUM",
    "FIGURE_SIZE_LARGE",
    "DPI_DEFAULT",
    "DPI_MEDIUM",
    "DPI_HIGH_RES",
]
