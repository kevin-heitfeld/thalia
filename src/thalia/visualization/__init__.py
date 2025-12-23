"""
Visualization utilities for Thalia brain networks.

Provides tools for visualizing:
- Network topology (regions and pathways)
- Connectivity matrices
- Spike raster plots
"""

from .network_graph import (
    visualize_brain_topology,
    export_topology_to_graphviz,
    plot_connectivity_matrix,
)
from .constants import (
    DEFAULT_NODE_SIZE_SCALE,
    NODE_ALPHA_DEFAULT,
    EDGE_ALPHA_DEFAULT,
    EDGE_WIDTH_SCALE,
    LAYOUT_K_FACTOR,
    LAYOUT_ITERATIONS,
    ARC_RADIUS,
    LEGEND_FRAMEALPHA,
    HIERARCHICAL_Y_SPACING,
    HIERARCHICAL_X_SPACING,
    DPI_DEFAULT,
    DPI_MEDIUM,
    DPI_HIGH_RES,
    FIGURE_SIZE_SMALL,
    FIGURE_SIZE_MEDIUM,
    FIGURE_SIZE_LARGE,
)

__all__ = [
    'visualize_brain_topology',
    'export_topology_to_graphviz',
    'plot_connectivity_matrix',
    # Constants
    'DEFAULT_NODE_SIZE_SCALE',
    'NODE_ALPHA_DEFAULT',
    'EDGE_ALPHA_DEFAULT',
    'EDGE_WIDTH_SCALE',
    'LAYOUT_K_FACTOR',
    'LAYOUT_ITERATIONS',
    'ARC_RADIUS',
    'LEGEND_FRAMEALPHA',
    'HIERARCHICAL_Y_SPACING',
    'HIERARCHICAL_X_SPACING',
    'DPI_DEFAULT',
    'DPI_MEDIUM',
    'DPI_HIGH_RES',
    'FIGURE_SIZE_SMALL',
    'FIGURE_SIZE_MEDIUM',
    'FIGURE_SIZE_LARGE',
]
