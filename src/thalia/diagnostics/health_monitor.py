"""
Health Monitor: Real-time network health diagnostics.

This module provides automated health monitoring for neural networks,
detecting pathological states like:
- Activity collapse (too few spikes)
- Seizure-like activity (too many spikes)
- Weight explosion/collapse
- E/I imbalance
- Criticality drift

The monitor provides:
1. Binary health checks (healthy/unhealthy)
2. Severity scoring (0-100, higher = worse)
3. Actionable recommendations for fixing issues

Biological Motivation:
======================
Real brains have multiple overlapping homeostatic mechanisms that detect
and correct deviations from healthy operating regimes. This monitor
provides similar oversight for artificial neural networks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .oscillator_health import OscillatorHealthMonitor


class HealthIssue(Enum):
    """Types of network health issues."""

    ACTIVITY_COLLAPSE = "activity_collapse"
    SEIZURE_RISK = "seizure_risk"
    WEIGHT_EXPLOSION = "weight_explosion"
    WEIGHT_COLLAPSE = "weight_collapse"
    EI_IMBALANCE = "ei_imbalance"
    CRITICALITY_DRIFT = "criticality_drift"
    DOPAMINE_SATURATION = "dopamine_saturation"
    LEARNING_STALL = "learning_stall"
    OSCILLATOR_PATHOLOGY = "oscillator_pathology"


class IssueSeverity(Enum):
    """Severity levels for health issues.

    Values represent severity scores (0-100, higher = worse).
    """

    LOW = 20.0  # Minor issues, informational
    MEDIUM = 50.0  # Moderate issues, should be addressed
    HIGH = 80.0  # Critical issues, need immediate attention
    CRITICAL = 100.0  # Catastrophic issues, system failure imminent


@dataclass
class HealthConfig:
    """Configuration for health monitoring.

    Attributes:
        spike_rate_min: Minimum healthy average spike rate
        spike_rate_max: Maximum healthy average spike rate
        weight_min: Minimum healthy weight magnitude
        weight_max: Maximum healthy weight magnitude
        ei_ratio_min: Minimum healthy E/I ratio
        ei_ratio_max: Maximum healthy E/I ratio
        criticality_min: Minimum healthy branching ratio
        criticality_max: Maximum healthy branching ratio
        dopamine_max: Maximum healthy dopamine level
        learning_rate_min: Minimum learning rate to consider "learning"
        severity_threshold: Minimum severity to report (0-100)
    """

    # Spike rate bounds (fraction of neurons active per timestep)
    spike_rate_min: float = 0.001  # Below this (0.1%) = activity collapse
    spike_rate_max: float = 0.8  # Above this (80%) = seizure risk

    # Weight magnitude bounds
    weight_min: float = 0.001  # Below this = weight collapse
    weight_max: float = 5.0  # Above this = weight explosion

    # E/I ratio bounds
    ei_ratio_min: float = 1.0  # Below this = over-inhibited
    ei_ratio_max: float = 10.0  # Above this = under-inhibited

    # Criticality bounds (branching ratio)
    criticality_min: float = 0.8  # Below this = subcritical
    criticality_max: float = 1.2  # Above this = supercritical

    # Dopamine saturation
    dopamine_max: float = 2.0  # Above this = saturated

    # Learning stall detection
    learning_rate_min: float = 1e-6  # Below this = no learning

    # Reporting threshold
    severity_threshold: float = 10.0  # Only report issues >= this severity


@dataclass
class IssueReport:
    """Report for a single health issue.

    Attributes:
        issue_type: Type of health issue
        severity: Severity score (0-100, higher = worse)
        description: Human-readable description
        recommendation: Suggested fix
        metrics: Raw metrics that triggered this issue
    """

    issue_type: HealthIssue
    severity: float
    description: str
    recommendation: str
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class HealthReport:
    """Complete health report for a brain/network.

    Attributes:
        is_healthy: Whether the network is in a healthy state
        overall_severity: Maximum severity across all issues
        issues: List of detected issues
        summary: One-line summary
        metrics: Raw diagnostic metrics
    """

    is_healthy: bool
    overall_severity: float
    issues: List[IssueReport]
    summary: str
    metrics: Dict[str, Any] = field(default_factory=dict)


class HealthMonitor:
    """Monitor network health and detect pathological states.

    This class provides automated health monitoring that can:
    1. Detect common pathological states
    2. Provide severity scoring
    3. Suggest corrective actions
    4. Monitor oscillator health (optional)

    It's designed to be called during training/testing to catch
    issues early before they cause mysterious failures.
    """

    def __init__(self, config: Optional[HealthConfig] = None):
        """Initialize health monitor.

        Args:
            config: Health configuration (uses defaults if None)
        """
        self.config = config or HealthConfig()

        # History for tracking trends
        self._spike_rate_history: List[float] = []
        self._weight_mean_history: List[float] = []
        self._max_history_len = 100

        # Oscillator health monitor
        self.oscillator_monitor = OscillatorHealthMonitor()

    def check_health(self, diagnostics: Dict[str, Any]) -> HealthReport:
        """Check network health from diagnostic data.

        Args:
            diagnostics: Dictionary of diagnostic metrics

        Returns:
            HealthReport with detected issues
        """
        cfg = self.config
        issues: List[IssueReport] = []

        # Extract metrics
        spike_counts = diagnostics.get("spike_counts", {})
        total_spikes = sum(spike_counts.values())

        # Calculate total neurons from region diagnostics
        # Each region reports its architecture in region_specific or activity sections
        regions_diag = diagnostics.get("regions", {})
        total_neurons = 0

        for region_name, region_diag in regions_diag.items():
            # Try multiple sources for neuron count (regions report differently)
            neuron_count = 0

            # Method 1: Direct n_neurons (simple regions)
            if "n_neurons" in region_diag:
                neuron_count = region_diag["n_neurons"]

            # Method 2: Sum of layer/population sizes (cortex, striatum)
            elif "region_specific" in region_diag:
                region_spec = region_diag["region_specific"]

                # Cortex: sum all layer sizes
                if "architecture" in region_spec:
                    arch = region_spec["architecture"]
                    neuron_count = sum(
                        arch.get(f"{layer}_size", 0)
                        for layer in ["l4", "l23", "l5", "l6a", "l6b"]
                    )

                # Striatum: neurons_per_action * n_actions + FSI
                elif "n_actions" in region_spec and "neurons_per_action" in region_spec:
                    n_actions = region_spec["n_actions"]
                    neurons_per_action = region_spec["neurons_per_action"]
                    neuron_count = n_actions * neurons_per_action
                    # Note: FSI neurons are separate but typically small (~4)

            # Method 3: From activity metrics total_neurons field
            elif "activity" in region_diag and "total_neurons" in region_diag["activity"]:
                neuron_count = region_diag["activity"]["total_neurons"]

            # Fallback: estimate from n_sources (crude approximation)
            if neuron_count == 0 and "n_sources" in region_diag:
                neuron_count = region_diag["n_sources"] * 100

            total_neurons += neuron_count

        # Absolute fallback if no region info found
        if total_neurons == 0:
            total_neurons = len(spike_counts) * 100

        # Calculate average spike rate (fraction of neurons spiking per timestep)
        avg_spike_rate = total_spikes / max(1, total_neurons)

        # Track history
        self._spike_rate_history.append(avg_spike_rate)
        if len(self._spike_rate_history) > self._max_history_len:
            self._spike_rate_history.pop(0)

        # =====================================================================
        # CHECK 1: Activity Level
        # =====================================================================

        if avg_spike_rate < cfg.spike_rate_min:
            severity = 100 * (cfg.spike_rate_min - avg_spike_rate) / cfg.spike_rate_min
            issues.append(
                IssueReport(
                    issue_type=HealthIssue.ACTIVITY_COLLAPSE,
                    severity=min(100, severity),
                    description=f"Activity collapse: spike rate {avg_spike_rate:.4f} < {cfg.spike_rate_min}",
                    recommendation="Increase input strength, reduce inhibition, or check for weight collapse",
                    metrics={"spike_rate": avg_spike_rate, "threshold": cfg.spike_rate_min},
                )
            )

        if avg_spike_rate > cfg.spike_rate_max:
            severity = 100 * (avg_spike_rate - cfg.spike_rate_max) / cfg.spike_rate_max
            issues.append(
                IssueReport(
                    issue_type=HealthIssue.SEIZURE_RISK,
                    severity=min(100, severity),
                    description=f"Seizure-like activity: spike rate {avg_spike_rate:.4f} > {cfg.spike_rate_max}",
                    recommendation="Enable E/I balance, reduce excitation, or increase inhibition",
                    metrics={"spike_rate": avg_spike_rate, "threshold": cfg.spike_rate_max},
                )
            )

        # =====================================================================
        # CHECK 2: Weight Magnitudes
        # =====================================================================

        # Check cortex weights (if available)
        cortex_diag = diagnostics.get("cortex", {})
        if cortex_diag:
            # Look for weight statistics
            for key in cortex_diag:
                if "_w_mean" in key:
                    w_mean = abs(cortex_diag[key])

                    if w_mean < cfg.weight_min:
                        severity = 100 * (cfg.weight_min - w_mean) / cfg.weight_min
                        issues.append(
                            IssueReport(
                                issue_type=HealthIssue.WEIGHT_COLLAPSE,
                                severity=min(100, severity),
                                description=f"Weight collapse in {key}: {w_mean:.4f} < {cfg.weight_min}",
                                recommendation="Check learning rates, may need weight initialization",
                                metrics={"weight_mean": w_mean, "threshold": cfg.weight_min},
                            )
                        )

                    if w_mean > cfg.weight_max:
                        severity = 100 * (w_mean - cfg.weight_max) / cfg.weight_max
                        issues.append(
                            IssueReport(
                                issue_type=HealthIssue.WEIGHT_EXPLOSION,
                                severity=min(100, severity),
                                description=f"Weight explosion in {key}: {w_mean:.4f} > {cfg.weight_max}",
                                recommendation="Enable homeostasis, reduce learning rate, or add weight decay",
                                metrics={"weight_mean": w_mean, "threshold": cfg.weight_max},
                            )
                        )

        # =====================================================================
        # CHECK 3: E/I Balance (if available)
        # =====================================================================

        ei_ratio = diagnostics.get("robustness_ei_ratio")
        if ei_ratio is not None:
            if ei_ratio < cfg.ei_ratio_min:
                severity = 50 * (cfg.ei_ratio_min - ei_ratio) / cfg.ei_ratio_min
                issues.append(
                    IssueReport(
                        issue_type=HealthIssue.EI_IMBALANCE,
                        severity=min(100, severity),
                        description=f"Over-inhibited: E/I ratio {ei_ratio:.2f} < {cfg.ei_ratio_min}",
                        recommendation="Reduce inhibitory strength or increase excitation",
                        metrics={"ei_ratio": ei_ratio, "threshold": cfg.ei_ratio_min},
                    )
                )

            if ei_ratio > cfg.ei_ratio_max:
                severity = 50 * (ei_ratio - cfg.ei_ratio_max) / cfg.ei_ratio_max
                issues.append(
                    IssueReport(
                        issue_type=HealthIssue.EI_IMBALANCE,
                        severity=min(100, severity),
                        description=f"Under-inhibited: E/I ratio {ei_ratio:.2f} > {cfg.ei_ratio_max}",
                        recommendation="Increase inhibitory strength or enable E/I balance mechanism",
                        metrics={"ei_ratio": ei_ratio, "threshold": cfg.ei_ratio_max},
                    )
                )

        # =====================================================================
        # CHECK 4: Criticality (if available)
        # =====================================================================

        criticality = diagnostics.get("criticality", {})
        if criticality.get("enabled"):
            branching = criticality.get("branching_ratio", 1.0)

            if branching < cfg.criticality_min:
                severity = 30 * (cfg.criticality_min - branching) / cfg.criticality_min
                issues.append(
                    IssueReport(
                        issue_type=HealthIssue.CRITICALITY_DRIFT,
                        severity=min(100, severity),
                        description=f"Subcritical: branching ratio {branching:.3f} < {cfg.criticality_min}",
                        recommendation="Increase connection strength or reduce inhibition",
                        metrics={"branching_ratio": branching, "threshold": cfg.criticality_min},
                    )
                )

            if branching > cfg.criticality_max:
                severity = 30 * (branching - cfg.criticality_max) / (2.0 - cfg.criticality_max)
                issues.append(
                    IssueReport(
                        issue_type=HealthIssue.CRITICALITY_DRIFT,
                        severity=min(100, severity),
                        description=f"Supercritical: branching ratio {branching:.3f} > {cfg.criticality_max}",
                        recommendation="Reduce connection strength or increase inhibition",
                        metrics={"branching_ratio": branching, "threshold": cfg.criticality_max},
                    )
                )

        # =====================================================================
        # CHECK 5: Dopamine Saturation
        # =====================================================================

        dopamine = diagnostics.get("dopamine", {})
        global_da = abs(dopamine.get("global", 0.0))

        if global_da > cfg.dopamine_max:
            severity = 40 * (global_da - cfg.dopamine_max) / cfg.dopamine_max
            issues.append(
                IssueReport(
                    issue_type=HealthIssue.DOPAMINE_SATURATION,
                    severity=min(100, severity),
                    description=f"Dopamine saturated: {global_da:.2f} > {cfg.dopamine_max}",
                    recommendation="Reduce reward scaling or enable reward normalization",
                    metrics={"dopamine": global_da, "threshold": cfg.dopamine_max},
                )
            )

        # =====================================================================
        # CHECK 6: Oscillator Health (if enabled)
        # =====================================================================

        if self.oscillator_monitor:
            oscillator_data = diagnostics.get("oscillators", {})
            if oscillator_data:
                try:
                    osc_report = self.oscillator_monitor.check_health(
                        phases=oscillator_data.get("phases", {}),
                        frequencies=oscillator_data.get("frequencies", {}),
                        amplitudes=oscillator_data.get("amplitudes", {}),
                        signals=oscillator_data.get("signals"),
                        couplings=oscillator_data.get("couplings"),
                    )

                    # Add oscillator issues to main report
                    for osc_issue in osc_report.issues:
                        issues.append(
                            IssueReport(
                                issue_type=HealthIssue.OSCILLATOR_PATHOLOGY,
                                severity=osc_issue.severity,
                                description=f"[{osc_issue.oscillator_name}] {osc_issue.description}",
                                recommendation=osc_issue.recommendation,
                                metrics=osc_issue.metrics,
                            )
                        )
                except Exception:
                    # Graceful degradation on oscillator check failure
                    pass

        # =====================================================================
        # Filter by severity threshold
        # =====================================================================

        issues = [i for i in issues if i.severity >= cfg.severity_threshold]

        # Compute overall health
        is_healthy = len(issues) == 0
        overall_severity = max([i.severity for i in issues], default=0.0)

        # Generate summary
        if is_healthy:
            summary = "✓ Network is healthy"
        else:
            issue_types = [i.issue_type.value for i in issues]
            summary = f"⚠ {len(issues)} issue(s): {', '.join(issue_types)}"

        return HealthReport(
            is_healthy=is_healthy,
            overall_severity=overall_severity,
            issues=issues,
            summary=summary,
            metrics={
                "avg_spike_rate": avg_spike_rate,
                "ei_ratio": ei_ratio,
                "dopamine": global_da,
            },
        )

    def get_trend_summary(self) -> Dict[str, str]:
        """Get summary of recent trends.

        Returns:
            Dictionary with trend descriptions
        """
        if len(self._spike_rate_history) < 10:
            return {"status": "insufficient_data"}

        recent = self._spike_rate_history[-10:]
        early = self._spike_rate_history[-20:-10] if len(self._spike_rate_history) >= 20 else recent

        recent_mean = sum(recent) / len(recent)
        early_mean = sum(early) / len(early)

        trends = {}

        # Spike rate trend
        if recent_mean > early_mean * 1.2:
            trends["spike_rate"] = "increasing"
        elif recent_mean < early_mean * 0.8:
            trends["spike_rate"] = "decreasing"
        else:
            trends["spike_rate"] = "stable"

        return trends

    def reset_history(self):
        """Reset trend history."""
        self._spike_rate_history.clear()
        self._weight_mean_history.clear()
