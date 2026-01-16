"""Standard diagnostic key names for consistent metrics reporting.

This module defines canonical key names for diagnostic dictionaries returned
by get_diagnostics() methods across all components. Using these constants
ensures consistency in logging, visualization, and analysis.

Usage:
======
    from thalia.core.diagnostics_keys import DiagnosticKeys as DK

    def get_diagnostics(self):
        return {
            DK.FIRING_RATE: self.firing_rate.mean().item(),
            DK.WEIGHT_MEAN: self.weights.data.mean().item(),
            DK.SPARSITY: (self.spikes.sum() / self.spikes.numel()).item(),
        }

Rationale:
==========
Before standardization, we had:
- "firing_rate" vs "avg_firing_rate" vs "mean_firing_rate"
- "weight_mean" vs "mean_weight" vs "avg_weight"
- Inconsistent naming made cross-component analysis difficult

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations


class DiagnosticKeys:
    """Standard diagnostic key names."""

    # =========================================================================
    # Activity Metrics
    # =========================================================================
    FIRING_RATE = "firing_rate"
    """Mean firing rate (spikes/timestep or Hz)."""

    SPARSITY = "sparsity"
    """Fraction of neurons active (0-1)."""

    BURST_RATE = "burst_rate"
    """Fraction of bursting neurons."""

    # =========================================================================
    # Weight Metrics
    # =========================================================================
    WEIGHT_MEAN = "weight_mean"
    """Mean synaptic weight value."""

    WEIGHT_STD = "weight_std"
    """Standard deviation of weights."""

    WEIGHT_MIN = "weight_min"
    """Minimum weight value."""

    WEIGHT_MAX = "weight_max"
    """Maximum weight value."""

    WEIGHT_SPARSITY = "weight_sparsity"
    """Fraction of near-zero weights."""

    # =========================================================================
    # Learning Metrics
    # =========================================================================
    LEARNING_RATE = "learning_rate"
    """Current learning rate (may be adaptive)."""

    WEIGHT_CHANGE = "weight_change"
    """Magnitude of recent weight updates."""

    ELIGIBILITY_MEAN = "eligibility_mean"
    """Mean eligibility trace value."""

    # =========================================================================
    # Membrane Dynamics
    # =========================================================================
    MEMBRANE_MEAN = "membrane_mean"
    """Mean membrane potential."""

    MEMBRANE_STD = "membrane_std"
    """Membrane potential standard deviation."""

    ADAPTATION_MEAN = "adaptation_mean"
    """Mean adaptation current."""

    # =========================================================================
    # Neuromodulation
    # =========================================================================
    DOPAMINE_LEVEL = "dopamine_level"
    """Current dopamine concentration."""

    ACETYLCHOLINE_LEVEL = "acetylcholine_level"
    """Current acetylcholine concentration."""

    NOREPINEPHRINE_LEVEL = "norepinephrine_level"
    """Current norepinephrine concentration."""

    # =========================================================================
    # Region-Specific Metrics
    # =========================================================================
    # Striatum
    D1_ACTIVITY = "d1_activity"
    """D1 pathway activity (Go signal)."""

    D2_ACTIVITY = "d2_activity"
    """D2 pathway activity (NoGo signal)."""

    EXPLORATION_BONUS = "exploration_bonus"
    """UCB exploration bonus magnitude."""

    # Hippocampus
    MEMORY_SIZE = "memory_size"
    """Number of stored episodes."""

    PATTERN_SEPARATION = "pattern_separation"
    """DG pattern separation strength."""

    # Cortex
    PREDICTION_ERROR = "prediction_error"
    """Predictive coding error magnitude."""

    # Cerebellum
    ERROR_SIGNAL = "error_signal"
    """Teaching signal for error-corrective learning."""

    # =========================================================================
    # Growth Metrics
    # =========================================================================
    NEURON_COUNT = "neuron_count"
    """Current number of neurons."""

    GROWTH_POTENTIAL = "growth_potential"
    """Readiness for structural growth (0-1)."""

    CAPACITY = "capacity"
    """Fraction of capacity used (0-1)."""


__all__ = ["DiagnosticKeys"]
