"""
Diagnostics and Logging System for Thalia Brain Simulation

This module provides a centralized, configurable diagnostics system for
monitoring and debugging brain region activity, learning, and decision-making.

Features:
=========
1. DIAGNOSTIC LEVELS: Control verbosity (OFF, SUMMARY, DETAILED, TRACE)
2. STRUCTURED DATA: All diagnostics return typed dictionaries
3. PER-COMPONENT CONFIG: Enable/disable diagnostics for specific regions
4. AGGREGATION: BrainSystem can collect diagnostics from all regions
5. HISTORY: Optional rolling history for time-series analysis
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, TypedDict

import torch

from thalia.constants import DEFAULT_DT_MS, MS_PER_SECOND
from thalia.utils import compute_firing_rate


# =========================================================================
# STANDARDIZED DIAGNOSTIC SCHEMAS
# =========================================================================


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


# =============================================================================
# UTILITY FUNCTIONS FOR DIAGNOSTICS
# =============================================================================


class DiagnosticsUtils:
    """Utility functions for computing diagnostics on tensors."""

    @staticmethod
    def weight_diagnostics(
        weights: torch.Tensor,
        prefix: str = "",
        include_histogram: bool = False,
    ) -> Dict[str, float]:
        """Compute standard weight statistics.

        Args:
            weights: Weight tensor (any shape)
            prefix: Prefix for metric names (e.g., "d1" → "d1_weight_mean")
            include_histogram: Include histogram bins (more expensive)

        Returns:
            Dict with weight statistics
        """
        prefix = f"{prefix}_" if prefix else ""

        w = weights.detach()

        # Handle empty tensors
        if w.numel() == 0:
            return {
                f"{prefix}weight_mean": 0.0,
                f"{prefix}weight_std": 0.0,
                f"{prefix}weight_min": 0.0,
                f"{prefix}weight_max": 0.0,
                f"{prefix}weight_sparsity": 1.0,
            }

        stats = {
            f"{prefix}weight_mean": w.mean().item(),
            f"{prefix}weight_std": w.std().item(),
            f"{prefix}weight_min": w.min().item(),
            f"{prefix}weight_max": w.max().item(),
            f"{prefix}weight_sparsity": (w.abs() < 1e-6).float().mean().item(),
        }

        # Non-zero weights (for sparse analysis)
        nonzero_mask = w.abs() >= 1e-6
        if nonzero_mask.any():
            stats[f"{prefix}weight_nonzero_mean"] = w[nonzero_mask].mean().item()

        if include_histogram:
            # 10-bin histogram
            hist = torch.histc(w.float(), bins=10)
            for i, count in enumerate(hist.tolist()):
                stats[f"{prefix}weight_hist_{i}"] = count

        return stats

    @staticmethod
    def spike_diagnostics(
        spikes: torch.Tensor,
        prefix: str = "",
        dt_ms: float = DEFAULT_DT_MS,
    ) -> Dict[str, float]:
        """Compute spike/activity statistics.

        Args:
            spikes: Spike tensor (binary or rate)
            prefix: Prefix for metric names
            dt_ms: Timestep in milliseconds (for rate calculation)

        Returns:
            Dict with spike statistics
        """
        prefix = f"{prefix}_" if prefix else ""

        s = spikes.detach().float()

        # Sparsity: fraction of neurons NOT spiking
        sparsity = 1.0 - s.mean().item()

        # Active count
        active_count = (s > 0.5).sum().item()
        total_neurons = s.numel()

        # Firing rate (if binary spikes)
        rate_hz = s.mean().item() * (MS_PER_SECOND / dt_ms)

        return {
            f"{prefix}sparsity": sparsity,
            f"{prefix}active_count": active_count,
            f"{prefix}total_neurons": total_neurons,
            f"{prefix}firing_rate_hz": rate_hz,
            f"{prefix}mean_activity": s.mean().item(),
        }

    @staticmethod
    def trace_diagnostics(
        trace: torch.Tensor,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Compute eligibility/activity trace statistics.

        Args:
            trace: Trace tensor (typically exponentially decaying)
            prefix: Prefix for metric names

        Returns:
            Dict with trace statistics
        """
        prefix = f"{prefix}_" if prefix else ""

        t = trace.detach()

        return {
            f"{prefix}trace_mean": t.mean().item(),
            f"{prefix}trace_max": t.max().item(),
            f"{prefix}trace_norm": t.norm().item(),
            f"{prefix}trace_nonzero": (t.abs() > 1e-6).sum().item(),
        }


def compute_activity_metrics(
    output_spikes: torch.Tensor,
    total_neurons: Optional[int] = None,
) -> ActivityMetrics:
    """Compute standard activity metrics from output spikes.

    Args:
        output_spikes: Binary spike tensor [n_neurons] or [batch, n_neurons]
        total_neurons: Total neuron count (defaults to output_spikes.shape[-1])

    Returns:
        ActivityMetrics dict with firing_rate, spike_count, sparsity, etc.
    """
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
    weights: torch.Tensor,
    learning_rate: float,
    weight_changes: Optional[torch.Tensor] = None,
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
    state_tensors: dict[str, torch.Tensor],
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
    issues: list[str] = []
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


# =========================================================================
# DIAGNOSTICS MANAGER
# =========================================================================


class DiagnosticLevel(Enum):
    """Verbosity levels for diagnostics."""

    OFF = auto()  # No diagnostics
    SUMMARY = auto()  # Epoch-level summaries only
    DETAILED = auto()  # Per-trial key metrics
    TRACE = auto()  # Full per-timestep traces (expensive!)


@dataclass
class ComponentDiagnosticsConfig:
    """Configuration for a single component's diagnostics."""

    enabled: bool = True
    level: Optional[DiagnosticLevel] = None  # None = use global level
    history_size: int = 100  # Rolling history buffer size


@dataclass
class MainDiagnosticsConfig:
    """Global diagnostics configuration."""

    level: DiagnosticLevel = DiagnosticLevel.SUMMARY
    print_to_console: bool = True
    collect_history: bool = False
    history_size: int = 1000
    timestamp_entries: bool = False
    components: Dict[str, ComponentDiagnosticsConfig] = field(default_factory=dict)


class DiagnosticsManager:
    """
    Centralized diagnostics manager for the brain simulation.

    Collects, stores, and formats diagnostic information from all brain
    components (regions, pathways, learning systems).
    """

    def __init__(
        self,
        level: DiagnosticLevel = DiagnosticLevel.SUMMARY,
        config: Optional[MainDiagnosticsConfig] = None,
    ):
        self.config = config or MainDiagnosticsConfig(level=level)
        self._current: Dict[str, Dict[str, Any]] = {}  # Current trial data
        self._history: Dict[str, deque] = {}  # Rolling history per component
        self._epoch_data: List[Dict[str, Any]] = []  # Epoch summaries
        self._trial_count = 0
        self._epoch_count = 0
        self._start_time = time.time()

    def configure_component(
        self,
        name: str,
        enabled: bool = True,
        level: Optional[DiagnosticLevel] = None,
        history_size: int = 100,
    ) -> None:
        """Configure diagnostics for a specific component."""
        self.config.components[name] = ComponentDiagnosticsConfig(
            enabled=enabled,
            level=level,
            history_size=history_size,
        )
        if name not in self._history:
            self._history[name] = deque(maxlen=history_size)

    def get_level(self, component: Optional[str] = None) -> DiagnosticLevel:
        """Get effective diagnostic level for a component."""
        if component and component in self.config.components:
            comp_config = self.config.components[component]
            if not comp_config.enabled:
                return DiagnosticLevel.OFF
            if comp_config.level is not None:
                return comp_config.level
        return self.config.level

    def is_enabled(
        self, component: str, min_level: DiagnosticLevel = DiagnosticLevel.SUMMARY
    ) -> bool:
        """Check if diagnostics are enabled for a component at given level."""
        level = self.get_level(component)
        return level.value >= min_level.value

    def record(self, component: str, data: Dict[str, Any]) -> None:
        """Record diagnostic data for a component."""
        if not self.is_enabled(component):
            return

        if self.config.timestamp_entries:
            data["_timestamp"] = time.time() - self._start_time

        self._current[component] = data

        if self.config.collect_history and component in self._history:
            self._history[component].append(data.copy())

    def get_current(self, component: Optional[str] = None) -> Dict[str, Any]:
        """Get current diagnostic data."""
        if component:
            return self._current.get(component, {})
        return self._current.copy()

    def get_history(self, component: str, n: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get diagnostic history for a component."""
        if component not in self._history:
            return []
        hist = list(self._history[component])
        if n is not None:
            return hist[-n:]
        return hist

    def clear_current(self) -> None:
        """Clear current trial data (call at trial end)."""
        self._current.clear()

    def new_trial(self) -> None:
        """Start a new trial."""
        self._trial_count += 1
        self.clear_current()

    def new_epoch(self) -> None:
        """Start a new epoch."""
        self._epoch_count += 1
        self._trial_count = 0

    @property
    def trial_count(self) -> int:
        """Get current trial count."""
        return self._trial_count

    @property
    def epoch_count(self) -> int:
        """Get current epoch count."""
        return self._epoch_count

    # =========================================================================
    # FORMATTED OUTPUT
    # =========================================================================

    def format_trial_line(
        self,
        trial: int,
        gt: str,
        action: str,
        correct: bool,
        extras: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format a single trial result line."""
        rwd = "+" if correct else "-"
        line = f"  Trial {trial:3d}: GT={gt}, Act={action}, {rwd}"

        if extras and self.get_level().value >= DiagnosticLevel.DETAILED.value:
            extra_strs = []
            for k, v in extras.items():
                if isinstance(v, float):
                    extra_strs.append(f"{k}={v:.3f}")
                else:
                    extra_strs.append(f"{k}={v}")
            if extra_strs:
                line += f" ({', '.join(extra_strs)})"

        return line

    def print_trial(
        self,
        trial: int,
        gt: str,
        action: str,
        correct: bool,
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Print a trial result if enabled."""
        if not self.config.print_to_console:
            return
        if self.get_level().value < DiagnosticLevel.DETAILED.value:
            return
        print(self.format_trial_line(trial, gt, action, correct, extras))

    def format_epoch_summary(
        self,
        epoch: int,
        accuracy: float,
        match_acc: float,
        nomatch_acc: float,
        extras: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Format epoch summary line."""
        line = f"  Epoch {epoch:3d}: Acc={accuracy*100:.1f}% (M:{match_acc*100:.1f}%, NM:{nomatch_acc*100:.1f}%)"

        if extras:
            extra_strs = []
            for k, v in extras.items():
                if isinstance(v, float):
                    extra_strs.append(f"{k}={v:.2f}")
                elif isinstance(v, (list, tuple)):
                    extra_strs.append(f"{k}=[{', '.join(f'{x:.2f}' for x in v)}]")
                else:
                    extra_strs.append(f"{k}={v}")
            if extra_strs:
                line += f" [{', '.join(extra_strs)}]"

        return line

    def print_epoch(
        self,
        epoch: int,
        accuracy: float,
        match_acc: float,
        nomatch_acc: float,
        extras: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Print epoch summary if enabled."""
        if not self.config.print_to_console:
            return
        if self.get_level().value < DiagnosticLevel.SUMMARY.value:
            return
        print(self.format_epoch_summary(epoch, accuracy, match_acc, nomatch_acc, extras))

    def format_weights_summary(
        self,
        d1_per_action: List[float],
        d2_per_action: List[float],
        net_per_action: Optional[List[float]] = None,
        val_per_action: Optional[List[float]] = None,
    ) -> str:
        """Format D1/D2/NET weight summary."""
        if net_per_action is None:
            net_per_action = [d1 - d2 for d1, d2 in zip(d1_per_action, d2_per_action)]

        d1_str = ", ".join(f"{x:.4f}" for x in d1_per_action)
        d2_str = ", ".join(f"{x:.4f}" for x in d2_per_action)
        net_str = ", ".join(f"{x:.4f}" for x in net_per_action)

        line = f"D1: [{d1_str}] | D2: [{d2_str}] | NET: [{net_str}]"

        if val_per_action:
            val_str = ", ".join(f"{x:.3f}" for x in val_per_action)
            line += f" | VAL: [{val_str}]"

        return line

    def print_weights(
        self,
        d1_per_action: List[float],
        d2_per_action: List[float],
        net_per_action: Optional[List[float]] = None,
        val_per_action: Optional[List[float]] = None,
        prefix: str = "",
    ) -> None:
        """Print weight summary if enabled."""
        if not self.config.print_to_console:
            return
        if self.get_level().value < DiagnosticLevel.SUMMARY.value:
            return
        line = self.format_weights_summary(
            d1_per_action, d2_per_action, net_per_action, val_per_action
        )
        print(f"{prefix}{line}")


# =========================================================================
# GLOBAL DIAGNOSTICS INSTANCE (optional convenience)
# =========================================================================

_global_diagnostics: Optional[DiagnosticsManager] = None


def get_diagnostics() -> DiagnosticsManager:
    """Get the global diagnostics manager (creates one if needed)."""
    global _global_diagnostics
    if _global_diagnostics is None:
        _global_diagnostics = DiagnosticsManager()
    return _global_diagnostics


def set_diagnostics(manager: DiagnosticsManager) -> None:
    """Set the global diagnostics manager."""
    global _global_diagnostics
    _global_diagnostics = manager


def configure_diagnostics(
    level: DiagnosticLevel = DiagnosticLevel.SUMMARY,
    print_to_console: bool = True,
    collect_history: bool = False,
) -> DiagnosticsManager:
    """Configure and return the global diagnostics manager."""
    config = MainDiagnosticsConfig(
        level=level,
        print_to_console=print_to_console,
        collect_history=collect_history,
    )
    manager = DiagnosticsManager(config=config)
    set_diagnostics(manager)
    return manager
