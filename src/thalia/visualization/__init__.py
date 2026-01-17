"""
Visualization utilities for Thalia brain networks.

Provides tools for visualizing:
- Network topology (regions and pathways)
- Connectivity matrices
- Spike raster plots
"""

from __future__ import annotations

from .network_graph import (
    export_topology_to_graphviz,
    plot_connectivity_matrix,
    visualize_brain_topology,
)

__all__ = [
    "visualize_brain_topology",
    "export_topology_to_graphviz",
    "plot_connectivity_matrix",
]
