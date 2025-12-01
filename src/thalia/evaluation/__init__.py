"""Evaluation and diagnostics for trained networks.

This module provides functions for analyzing trained weight matrices
and evaluating network performance.
"""

from thalia.evaluation.weights import (
    compute_diagonal_score,
    compute_paired_diagonal_score,
    analyze_recurrent_structure,
    compute_weight_statistics,
    print_recurrent_analysis,
    RecurrentAnalysis,
    WeightStatistics,
)

__all__ = [
    "compute_diagonal_score",
    "compute_paired_diagonal_score",
    "analyze_recurrent_structure",
    "compute_weight_statistics",
    "print_recurrent_analysis",
    "RecurrentAnalysis",
    "WeightStatistics",
]
