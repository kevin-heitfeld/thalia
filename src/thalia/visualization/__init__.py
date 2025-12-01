"""
Visualization utilities for SNN activity.
"""

from thalia.visualization.raster import plot_raster
from thalia.visualization.traces import plot_membrane_traces
from thalia.visualization.weights import (
    plot_weight_matrix,
    plot_recurrent_weights,
    plot_learning_curve,
    plot_weight_evolution,
    plot_learned_mapping,
    create_training_summary_figure,
)

__all__ = [
    "plot_raster",
    "plot_membrane_traces",
    "plot_weight_matrix",
    "plot_recurrent_weights",
    "plot_learning_curve",
    "plot_weight_evolution",
    "plot_learned_mapping",
    "create_training_summary_figure",
]
