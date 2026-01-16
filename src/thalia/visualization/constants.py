"""
Visualization Constants for Network Graphs and Plots.

This module centralizes magic numbers used in visualization code,
making it easier to adjust visual parameters and maintain consistency.

Author: Thalia Project
Date: December 23, 2025
"""

from __future__ import annotations

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
