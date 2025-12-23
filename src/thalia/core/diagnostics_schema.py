"""
Diagnostics Schema - Standardized Structure for Region Diagnostics.

This module provides TypedDict schemas for consistent diagnostic reporting
across all brain regions, enabling better monitoring and dashboard integration.

Author: Thalia Project
Date: December 22, 2025 (Tier 2.2 - Diagnostics standardization)
"""

from __future__ import annotations

from typing import TypedDict, Any

import torch

from thalia.components.coding import compute_firing_rate


class ActivityMetrics(TypedDict, total=False):
    """Standard activity metrics for all regions.

    These metrics describe the overall activity level and spiking behavior.
    """

    firing_rate: float
    """Mean firing rate across all neurons (0.0-1.0)"""

    spike_count: int
    """Total number of spikes in current timestep"""

    sparsity: float
    """Fraction of silent neurons (1.0 - firing_rate)"""

    active_neurons: int
    """Number of neurons that spiked"""

    total_neurons: int
    """Total number of neurons in region"""


class PlasticityMetrics(TypedDict, total=False):
    """Standard plasticity/learning metrics.

    These metrics describe synaptic changes and learning dynamics.
    """

    weight_mean: float
    """Mean synaptic weight across all connections"""

    weight_std: float
    """Standard deviation of synaptic weights"""

    weight_min: float
    """Minimum synaptic weight"""

    weight_max: float
    """Maximum synaptic weight"""

    learning_rate_effective: float
    """Current effective learning rate (after modulation)"""

    weight_change_magnitude: float
    """Magnitude of weight changes in last update"""

    num_potentiated: int
    """Number of synapses that strengthened"""

    num_depressed: int
    """Number of synapses that weakened"""


class HealthMetrics(TypedDict, total=False):
    """Standard health/stability metrics.

    These metrics identify potential issues or abnormal states.
    """

    is_silent: bool
    """True if region has no activity (firing rate < threshold)"""

    is_saturated: bool
    """True if region has excessive activity (firing rate > threshold)"""

    has_nan: bool
    """True if NaN values detected in state"""

    has_inf: bool
    """True if Inf values detected in state"""

    stability_score: float
    """Overall stability metric (0.0-1.0, higher = more stable)"""

    issues: list[str]
    """List of detected issues (empty if healthy)"""


class NeuromodulatorMetrics(TypedDict, total=False):
    """Neuromodulator levels and effects.

    Optional metrics for regions with neuromodulation.
    """

    dopamine: float
    """Current dopamine level (0.0-1.0)"""

    acetylcholine: float
    """Current acetylcholine level (0.0-1.0)"""

    norepinephrine: float
    """Current norepinephrine level (0.0-1.0)"""

    modulator_gate: float
    """Effective gating from neuromodulators (0.0-1.0)"""


class DiagnosticsDict(TypedDict):
    """Complete diagnostics dictionary with standardized sections.

    All regions should return diagnostics in this format for consistency.
    Optional sections can be omitted if not applicable.

    Usage:
        def get_diagnostics(self) -> DiagnosticsDict:
            return {
                "activity": {
                    "firing_rate": self.output_spikes.float().mean().item(),
                    "spike_count": self.output_spikes.sum().item(),
                    "sparsity": 1.0 - self.output_spikes.float().mean().item(),
                },
                "plasticity": {
                    "weight_mean": self.weights.mean().item(),
                    "weight_std": self.weights.std().item(),
                } if self.learning_enabled else None,
                "health": self._check_health(),
                "neuromodulators": {
                    "dopamine": self.dopamine_level,
                } if hasattr(self, "dopamine_level") else None,
                "region_specific": self._get_custom_metrics(),
            }
    """

    activity: ActivityMetrics
    """Activity metrics (required for all regions)"""

    plasticity: PlasticityMetrics | None
    """Plasticity metrics (None if learning disabled)"""

    health: HealthMetrics
    """Health metrics (required for all regions)"""

    neuromodulators: NeuromodulatorMetrics | None
    """Neuromodulator metrics (None if not applicable)"""

    region_specific: dict[str, Any]
    """Region-specific custom metrics (e.g., D1/D2 votes, WM slots, etc.)"""


# =============================================================================
# HELPER FUNCTIONS FOR COMMON DIAGNOSTICS
# =============================================================================


def compute_activity_metrics(
    output_spikes: Any,  # torch.Tensor
    total_neurons: int | None = None,
) -> ActivityMetrics:
    """Compute standard activity metrics from output spikes.

    Args:
        output_spikes: Binary spike tensor [n_neurons] or [batch, n_neurons]
        total_neurons: Total neuron count (defaults to output_spikes.shape[-1])

    Returns:
        ActivityMetrics dict with firing_rate, spike_count, sparsity, etc.
    """
    if not isinstance(output_spikes, torch.Tensor):
        output_spikes = torch.tensor(output_spikes)

    # Handle batch dimension
    if output_spikes.ndim > 1:
        output_spikes = output_spikes[-1]  # Use last timestep

    n_neurons = output_spikes.shape[0] if total_neurons is None else total_neurons
    firing_rate = compute_firing_rate(output_spikes)

    return ActivityMetrics(
        firing_rate=firing_rate,
        spike_count=int(output_spikes.sum().item()),
        sparsity=1.0 - firing_rate,
        active_neurons=int((output_spikes > 0).sum().item()),
        total_neurons=n_neurons,
    )


def compute_plasticity_metrics(
    weights: Any,  # torch.Tensor
    learning_rate: float,
    weight_changes: Any | None = None,  # torch.Tensor
) -> PlasticityMetrics:
    """Compute standard plasticity metrics from weight matrix.

    Args:
        weights: Weight matrix [n_output, n_input]
        learning_rate: Current learning rate
        weight_changes: Optional weight change matrix (for potentiation/depression counts)

    Returns:
        PlasticityMetrics dict with weight statistics
    """
    if not isinstance(weights, torch.Tensor):
        weights = torch.tensor(weights)

    metrics = PlasticityMetrics(
        weight_mean=float(weights.mean().item()),
        weight_std=float(weights.std().item()),
        weight_min=float(weights.min().item()),
        weight_max=float(weights.max().item()),
        learning_rate_effective=learning_rate,
    )

    if weight_changes is not None:
        if not isinstance(weight_changes, torch.Tensor):
            weight_changes = torch.tensor(weight_changes)

        metrics["weight_change_magnitude"] = float(weight_changes.abs().mean().item())
        metrics["num_potentiated"] = int((weight_changes > 0).sum().item())
        metrics["num_depressed"] = int((weight_changes < 0).sum().item())

    return metrics


def compute_health_metrics(
    state_tensors: dict[str, Any],  # dict[str, torch.Tensor]
    firing_rate: float,
    silence_threshold: float = 0.001,
    saturation_threshold: float = 0.9,
) -> HealthMetrics:
    """Compute standard health metrics from region state.

    Args:
        state_tensors: Dict of state tensors to check for NaN/Inf
        firing_rate: Current firing rate
        silence_threshold: Threshold below which region is considered silent
        saturation_threshold: Threshold above which region is considered saturated

    Returns:
        HealthMetrics dict with health indicators
    """
    issues = []
    has_nan = False
    has_inf = False

    # Check for NaN/Inf in state tensors
    for name, tensor in state_tensors.items():
        if not isinstance(tensor, torch.Tensor):
            continue

        if torch.isnan(tensor).any():
            has_nan = True
            issues.append(f"{name} contains NaN")

        if torch.isinf(tensor).any():
            has_inf = True
            issues.append(f"{name} contains Inf")

    # Check activity levels
    is_silent = firing_rate < silence_threshold
    is_saturated = firing_rate > saturation_threshold

    if is_silent:
        issues.append(f"Region is silent (firing_rate={firing_rate:.4f})")
    if is_saturated:
        issues.append(f"Region is saturated (firing_rate={firing_rate:.4f})")

    # Compute stability score (1.0 = perfectly healthy)
    stability_score = 1.0
    if has_nan or has_inf:
        stability_score = 0.0
    elif is_silent or is_saturated:
        stability_score = 0.5
    elif abs(firing_rate - 0.2) > 0.4:  # Deviation from typical 20% sparsity
        stability_score = 0.8

    return HealthMetrics(
        is_silent=is_silent,
        is_saturated=is_saturated,
        has_nan=has_nan,
        has_inf=has_inf,
        stability_score=stability_score,
        issues=issues,
    )


__all__ = [
    "ActivityMetrics",
    "PlasticityMetrics",
    "HealthMetrics",
    "NeuromodulatorMetrics",
    "DiagnosticsDict",
    "compute_activity_metrics",
    "compute_plasticity_metrics",
    "compute_health_metrics",
]
