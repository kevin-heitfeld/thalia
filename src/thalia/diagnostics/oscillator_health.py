"""
Oscillator Health Monitor: Detect oscillatory pathology patterns.

This module provides comprehensive monitoring of neural oscillations,
detecting pathological patterns like:
- Abnormal frequency drift
- Phase locking (stuck oscillators)
- Loss of cross-frequency coupling
- Pathological phase-amplitude coupling
- Abnormal oscillator synchrony

The monitor complements the general HealthMonitor with oscillator-specific
diagnostics critical for temporal dynamics and coordination.

Biological Motivation:
======================
Brain oscillations are fundamental to cognition, and their pathology indicates
serious dysfunction:
- **Abnormal theta**: Memory encoding/retrieval deficits (Alzheimer's)
- **Alpha suppression**: Attention and consciousness impairments
- **Gamma disruption**: Feature binding failures (schizophrenia)
- **Loss of coupling**: Coordination breakdown across regions

Usage:
======
    from thalia.diagnostics.oscillator_health import OscillatorHealthMonitor

    monitor = OscillatorHealthMonitor()

    # Check oscillator health
    report = monitor.check_health(
        phases=brain.oscillators.get_phases(),
        frequencies=brain.oscillators.get_frequencies(),
        amplitudes=brain.oscillators.get_effective_amplitudes(),
        couplings=brain.oscillators.couplings,
    )

    if not report.is_healthy:
        print(f"Oscillator issue: {report.summary}")
        for issue in report.issues:
            print(f"  - {issue.description}")

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class OscillatorIssue(Enum):
    """Types of oscillator health issues."""

    FREQUENCY_DRIFT = "frequency_drift"
    PHASE_LOCKING = "phase_locking"
    ABNORMAL_AMPLITUDE = "abnormal_amplitude"
    COUPLING_FAILURE = "coupling_failure"
    SYNCHRONY_LOSS = "synchrony_loss"
    PATHOLOGICAL_COUPLING = "pathological_coupling"
    OSCILLATOR_DEAD = "oscillator_dead"
    CROSS_REGION_DESYNCHRONY = "cross_region_desynchrony"


@dataclass
class OscillatorHealthConfig:
    """Configuration for oscillator health monitoring.

    Attributes:
        # Frequency drift thresholds (Hz deviation from baseline)
        delta_freq_range: Valid range for delta (0.5-4 Hz)
        theta_freq_range: Valid range for theta (4-10 Hz)
        alpha_freq_range: Valid range for alpha (8-13 Hz)
        beta_freq_range: Valid range for beta (13-30 Hz)
        gamma_freq_range: Valid range for gamma (30-100 Hz)
        ripple_freq_range: Valid range for ripple (100-200 Hz)

        # Phase locking detection
        phase_change_min: Minimum phase change per step (radians)
        phase_lock_window: Window size for checking phase stagnation

        # Amplitude thresholds
        amplitude_min: Minimum healthy amplitude
        amplitude_max: Maximum healthy amplitude

        # Coupling health
        coupling_strength_min: Minimum healthy coupling strength
        coupling_strength_max: Maximum healthy coupling strength

        # History tracking
        history_length: Number of timesteps to track
        severity_threshold: Minimum severity to report (0-100)
    """

    # Frequency ranges (Hz) - based on biological literature
    delta_freq_range: tuple = (0.5, 4.0)
    theta_freq_range: tuple = (4.0, 10.0)
    alpha_freq_range: tuple = (8.0, 13.0)
    beta_freq_range: tuple = (13.0, 30.0)
    gamma_freq_range: tuple = (30.0, 100.0)
    ripple_freq_range: tuple = (100.0, 200.0)

    # Phase dynamics
    phase_change_min: float = 0.001  # Min phase change (radians) to avoid locking
    phase_lock_window: int = 50  # Timesteps to check for stagnation

    # Amplitude bounds
    amplitude_min: float = 0.05  # Below this = dead oscillator
    amplitude_max: float = 1.5  # Above this = pathological amplitude

    # Coupling bounds
    coupling_strength_min: float = 0.1  # Below this = ineffective coupling
    coupling_strength_max: float = 1.0  # Above this = pathological coupling

    # Cross-region synchrony thresholds
    phase_coherence_min: float = 0.3  # Below this = poor synchrony
    phase_coherence_window: int = 50  # Window for coherence computation

    # History and reporting
    history_length: int = 100
    severity_threshold: float = 10.0


@dataclass
class OscillatorIssueReport:
    """Report for a single oscillator health issue.

    Attributes:
        issue_type: Type of oscillator issue
        oscillator_name: Name of affected oscillator (or "cross-oscillator")
        severity: Severity score (0-100, higher = worse)
        description: Human-readable description
        recommendation: Suggested fix
        metrics: Raw metrics that triggered this issue
    """

    issue_type: OscillatorIssue
    oscillator_name: str
    severity: float
    description: str
    recommendation: str
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class OscillatorHealthReport:
    """Complete health report for oscillator system.

    Attributes:
        is_healthy: Whether oscillators are in a healthy state
        overall_severity: Maximum severity across all issues
        issues: List of detected issues
        summary: One-line summary
        metrics: Raw oscillator metrics
    """

    is_healthy: bool
    overall_severity: float
    issues: List[OscillatorIssueReport]
    summary: str
    metrics: Dict[str, Any] = field(default_factory=dict)


class OscillatorHealthMonitor:
    """Monitor oscillator health and detect pathological patterns.

    This class provides comprehensive oscillator monitoring:
    1. Frequency drift detection
    2. Phase locking detection
    3. Amplitude pathology
    4. Coupling health
    5. Cross-oscillator synchrony
    """

    def __init__(self, config: Optional[OscillatorHealthConfig] = None):
        """Initialize oscillator health monitor.

        Args:
            config: Configuration (uses defaults if None)
        """
        self.config = config or OscillatorHealthConfig()

        # History tracking for each oscillator
        self._phase_history: Dict[str, deque] = {}
        self._frequency_history: Dict[str, deque] = {}
        self._amplitude_history: Dict[str, deque] = {}

        # Initialize for known oscillators
        for osc in ["delta", "theta", "alpha", "beta", "gamma", "ripple"]:
            self._phase_history[osc] = deque(maxlen=self.config.history_length)
            self._frequency_history[osc] = deque(maxlen=self.config.history_length)
            self._amplitude_history[osc] = deque(maxlen=self.config.history_length)

    def check_health(
        self,
        phases: Dict[str, float],
        frequencies: Dict[str, float],
        amplitudes: Dict[str, float],
        signals: Optional[Dict[str, float]] = None,
        couplings: Optional[List[Any]] = None,
    ) -> OscillatorHealthReport:
        """Check oscillator health from current state.

        Args:
            phases: Current phases for each oscillator [0, 2π)
            frequencies: Current frequencies for each oscillator (Hz)
            amplitudes: Current effective amplitudes for each oscillator
            signals: Current signal values (optional)
            couplings: List of OscillatorCoupling objects (optional)

        Returns:
            OscillatorHealthReport with detected issues
        """
        cfg = self.config
        issues: List[OscillatorIssueReport] = []

        # Update history
        for osc in phases.keys():
            if osc not in self._phase_history:
                self._phase_history[osc] = deque(maxlen=cfg.history_length)
                self._frequency_history[osc] = deque(maxlen=cfg.history_length)
                self._amplitude_history[osc] = deque(maxlen=cfg.history_length)

            self._phase_history[osc].append(phases[osc])
            self._frequency_history[osc].append(frequencies.get(osc, 0.0))
            self._amplitude_history[osc].append(amplitudes.get(osc, 1.0))

        # =====================================================================
        # CHECK 1: Frequency Drift
        # =====================================================================
        freq_ranges = {
            "delta": cfg.delta_freq_range,
            "theta": cfg.theta_freq_range,
            "alpha": cfg.alpha_freq_range,
            "beta": cfg.beta_freq_range,
            "gamma": cfg.gamma_freq_range,
            "ripple": cfg.ripple_freq_range,
        }

        for osc, freq in frequencies.items():
            if osc not in freq_ranges:
                continue

            min_freq, max_freq = freq_ranges[osc]
            if freq < min_freq:
                deviation = min_freq - freq
                severity = 100 * deviation / min_freq
                issues.append(
                    OscillatorIssueReport(
                        issue_type=OscillatorIssue.FREQUENCY_DRIFT,
                        oscillator_name=osc,
                        severity=min(100, severity),
                        description=f"{osc} frequency too low: {freq:.2f} Hz < {min_freq:.2f} Hz",
                        recommendation=f"Increase {osc} frequency or check configuration",
                        metrics={"frequency": freq, "min": min_freq, "max": max_freq},
                    )
                )
            elif freq > max_freq:
                deviation = freq - max_freq
                severity = 100 * deviation / max_freq
                issues.append(
                    OscillatorIssueReport(
                        issue_type=OscillatorIssue.FREQUENCY_DRIFT,
                        oscillator_name=osc,
                        severity=min(100, severity),
                        description=f"{osc} frequency too high: {freq:.2f} Hz > {max_freq:.2f} Hz",
                        recommendation=f"Decrease {osc} frequency or check configuration",
                        metrics={"frequency": freq, "min": min_freq, "max": max_freq},
                    )
                )

        # =====================================================================
        # CHECK 2: Phase Locking (Stuck Oscillator)
        # =====================================================================
        for osc, phase_hist in self._phase_history.items():
            if len(phase_hist) < cfg.phase_lock_window:
                continue

            # Check if phase is changing
            recent_phases = list(phase_hist)[-cfg.phase_lock_window :]
            phase_changes = []
            for i in range(1, len(recent_phases)):
                # Handle phase wrapping (0 to 2π)
                delta = recent_phases[i] - recent_phases[i - 1]
                if delta > math.pi:
                    delta -= 2 * math.pi
                elif delta < -math.pi:
                    delta += 2 * math.pi
                phase_changes.append(abs(delta))

            avg_change = sum(phase_changes) / len(phase_changes)

            if avg_change < cfg.phase_change_min:
                severity = 80.0  # Phase locking is critical
                issues.append(
                    OscillatorIssueReport(
                        issue_type=OscillatorIssue.PHASE_LOCKING,
                        oscillator_name=osc,
                        severity=severity,
                        description=f"{osc} phase locked (avg change: {avg_change:.6f} rad/step)",
                        recommendation=f"Reset {osc} oscillator or check timestep configuration",
                        metrics={"avg_phase_change": avg_change, "threshold": cfg.phase_change_min},
                    )
                )

        # =====================================================================
        # CHECK 3: Abnormal Amplitude
        # =====================================================================
        for osc, amp in amplitudes.items():
            if amp < cfg.amplitude_min:
                severity = 100 * (cfg.amplitude_min - amp) / cfg.amplitude_min
                issues.append(
                    OscillatorIssueReport(
                        issue_type=OscillatorIssue.ABNORMAL_AMPLITUDE,
                        oscillator_name=osc,
                        severity=min(100, severity),
                        description=f"{osc} amplitude too low: {amp:.3f} < {cfg.amplitude_min:.3f}",
                        recommendation=f"Check coupling configuration or enable {osc}",
                        metrics={
                            "amplitude": amp,
                            "min": cfg.amplitude_min,
                            "max": cfg.amplitude_max,
                        },
                    )
                )
            elif amp > cfg.amplitude_max:
                severity = 100 * (amp - cfg.amplitude_max) / cfg.amplitude_max
                issues.append(
                    OscillatorIssueReport(
                        issue_type=OscillatorIssue.ABNORMAL_AMPLITUDE,
                        oscillator_name=osc,
                        severity=min(100, severity),
                        description=f"{osc} amplitude too high: {amp:.3f} > {cfg.amplitude_max:.3f}",
                        recommendation=f"Reduce coupling strength for {osc}",
                        metrics={
                            "amplitude": amp,
                            "min": cfg.amplitude_min,
                            "max": cfg.amplitude_max,
                        },
                    )
                )

        # =====================================================================
        # CHECK 4: Dead Oscillator (No Signal Variation)
        # =====================================================================
        if signals:
            for osc, signal in signals.items():
                if osc not in self._amplitude_history:
                    continue

                amp_hist = list(self._amplitude_history[osc])
                if len(amp_hist) >= 10:
                    recent_amps = amp_hist[-10:]
                    amp_variance = sum(
                        (a - sum(recent_amps) / len(recent_amps)) ** 2 for a in recent_amps
                    ) / len(recent_amps)

                    if amp_variance < 0.001:  # Very low variance
                        severity = 60.0
                        issues.append(
                            OscillatorIssueReport(
                                issue_type=OscillatorIssue.OSCILLATOR_DEAD,
                                oscillator_name=osc,
                                severity=severity,
                                description=f"{osc} shows no amplitude variation (variance: {amp_variance:.6f})",
                                recommendation=f"Check if {osc} is enabled and properly configured",
                                metrics={
                                    "amplitude_variance": amp_variance,
                                    "recent_amplitude": signal,
                                },
                            )
                        )

        # =====================================================================
        # CHECK 5: Coupling Health
        # =====================================================================
        if couplings:
            for coupling in couplings:
                osc_name = coupling.oscillator
                strength = coupling.coupling_strength

                if strength < cfg.coupling_strength_min:
                    severity = 40.0
                    issues.append(
                        OscillatorIssueReport(
                            issue_type=OscillatorIssue.COUPLING_FAILURE,
                            oscillator_name=osc_name,
                            severity=severity,
                            description=f"{osc_name} coupling too weak: {strength:.3f} < {cfg.coupling_strength_min:.3f}",
                            recommendation=f"Increase coupling strength for {osc_name}",
                            metrics={
                                "coupling_strength": strength,
                                "min": cfg.coupling_strength_min,
                            },
                        )
                    )
                elif strength > cfg.coupling_strength_max:
                    severity = 60.0
                    issues.append(
                        OscillatorIssueReport(
                            issue_type=OscillatorIssue.PATHOLOGICAL_COUPLING,
                            oscillator_name=osc_name,
                            severity=severity,
                            description=f"{osc_name} coupling too strong: {strength:.3f} > {cfg.coupling_strength_max:.3f}",
                            recommendation=f"Reduce coupling strength for {osc_name}",
                            metrics={
                                "coupling_strength": strength,
                                "max": cfg.coupling_strength_max,
                            },
                        )
                    )

        # =====================================================================
        # CHECK 6: Cross-Oscillator Synchrony (Theta-Gamma Coupling)
        # =====================================================================
        # Check for healthy theta-gamma phase-amplitude coupling
        if (
            "theta" in phases
            and "gamma" in amplitudes
            and len(self._phase_history.get("theta", [])) >= 20
        ):
            theta_phases = list(self._phase_history["theta"])[-20:]
            gamma_amps = list(self._amplitude_history["gamma"])[-20:]

            # Compute correlation between theta phase and gamma amplitude
            # (simplified measure of phase-amplitude coupling)
            if len(theta_phases) == len(gamma_amps):
                mean_phase = sum(theta_phases) / len(theta_phases)
                mean_amp = sum(gamma_amps) / len(gamma_amps)

                covariance = sum(
                    (p - mean_phase) * (a - mean_amp) for p, a in zip(theta_phases, gamma_amps)
                ) / len(theta_phases)
                phase_std = (
                    sum((p - mean_phase) ** 2 for p in theta_phases) / len(theta_phases)
                ) ** 0.5
                amp_std = (sum((a - mean_amp) ** 2 for a in gamma_amps) / len(gamma_amps)) ** 0.5

                if phase_std > 0 and amp_std > 0:
                    correlation = covariance / (phase_std * amp_std)

                    # Healthy theta-gamma coupling should show moderate correlation
                    if abs(correlation) < 0.2:  # Very weak coupling
                        severity = 30.0
                        issues.append(
                            OscillatorIssueReport(
                                issue_type=OscillatorIssue.SYNCHRONY_LOSS,
                                oscillator_name="theta-gamma",
                                severity=severity,
                                description=f"Weak theta-gamma coupling (correlation: {correlation:.3f})",
                                recommendation="Check cross-frequency coupling configuration",
                                metrics={"theta_gamma_correlation": correlation},
                            )
                        )

        # =====================================================================
        # Filter by severity threshold
        # =====================================================================
        issues = [i for i in issues if i.severity >= cfg.severity_threshold]

        # Compute overall health
        is_healthy = len(issues) == 0
        overall_severity = max([i.severity for i in issues], default=0.0)

        # Generate summary
        if is_healthy:
            summary = "✓ Oscillators are healthy"
        else:
            issue_types = list(set(i.oscillator_name for i in issues))
            summary = f"⚠ {len(issues)} oscillator issue(s): {', '.join(issue_types[:3])}"
            if len(issue_types) > 3:
                summary += f" +{len(issue_types) - 3} more"

        return OscillatorHealthReport(
            is_healthy=is_healthy,
            overall_severity=overall_severity,
            issues=issues,
            summary=summary,
            metrics={
                "phases": phases,
                "frequencies": frequencies,
                "amplitudes": amplitudes,
            },
        )

    def reset_history(self):
        """Reset oscillator history."""
        for osc in self._phase_history.keys():
            self._phase_history[osc].clear()
            self._frequency_history[osc].clear()
            self._amplitude_history[osc].clear()

    def get_oscillator_statistics(self, oscillator: str) -> Dict[str, float]:
        """Get statistics for a specific oscillator.

        Args:
            oscillator: Name of oscillator (e.g., 'theta', 'gamma')

        Returns:
            Dictionary with mean, std, min, max for frequency and amplitude
        """
        if oscillator not in self._frequency_history:
            return {}

        freq_hist = list(self._frequency_history[oscillator])
        amp_hist = list(self._amplitude_history[oscillator])

        if not freq_hist or not amp_hist:
            return {}

        def compute_stats(values):
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = variance**0.5
            return {
                "mean": mean,
                "std": std,
                "min": min(values),
                "max": max(values),
            }

        stats = {
            "frequency": compute_stats(freq_hist),
            "amplitude": compute_stats(amp_hist),
        }

        return stats

    def compute_phase_coherence(
        self,
        region1_phases: Dict[str, float],
        region2_phases: Dict[str, float],
        oscillator: str = "gamma",
    ) -> float:
        """Compute phase coherence between two regions for a specific oscillator.

        Phase coherence measures how synchronized two regions are at a given
        frequency. High coherence indicates coordinated activity (e.g., for
        cross-modal binding or working memory).

        Biological Context:
        ===================
        - **Gamma coherence**: Cross-modal binding (visual + auditory)
        - **Theta coherence**: Working memory (hippocampus-PFC)
        - **Alpha coherence**: Attention coordination across regions
        - **Beta coherence**: Motor-cognitive coordination

        Args:
            region1_phases: Phase dict for first region
            region2_phases: Phase dict for second region
            oscillator: Which oscillator to measure ('theta', 'gamma', etc.)

        Returns:
            Phase coherence [0, 1] where 1 = perfect synchrony

        Reference:
            Fries (2015): Rhythms for Cognition: Communication Through Coherence
        """
        if oscillator not in region1_phases or oscillator not in region2_phases:
            return 0.0

        phase1 = region1_phases[oscillator]
        phase2 = region2_phases[oscillator]

        # Compute phase difference
        phase_diff = phase1 - phase2

        # Normalize to [-π, π]
        while phase_diff > math.pi:
            phase_diff -= 2 * math.pi
        while phase_diff < -math.pi:
            phase_diff += 2 * math.pi

        # Convert phase difference to coherence [0, 1]
        # Perfect synchrony (phase_diff=0) → coherence=1
        # Complete desynchrony (phase_diff=π) → coherence=0
        coherence = (1.0 + math.cos(phase_diff)) / 2.0

        return coherence

    def compute_region_pair_coherence(
        self,
        region_phases: Dict[str, Dict[str, float]],
        region_pairs: Optional[List[tuple]] = None,
        oscillators: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Compute phase coherence for multiple region pairs and oscillators.

        This is the primary method for validating curriculum mechanisms:
        - Stage 1: Hippocampus-PFC theta coherence (working memory)
        - Stage 2+: Visual-auditory gamma coherence (cross-modal binding)

        Args:
            region_phases: Dict mapping region_name → {oscillator: phase}
                Example: {'hippocampus': {'theta': 1.2, 'gamma': 2.5},
                         'prefrontal': {'theta': 1.3, 'gamma': 2.7}}
            region_pairs: List of (region1, region2) tuples to measure
                If None, measures all pairs
            oscillators: List of oscillators to measure
                If None, uses ['theta', 'gamma'] (most important for curriculum)

        Returns:
            Dictionary: {(region1, region2): {oscillator: coherence}}
                Example: {('hippocampus', 'prefrontal'): {'theta': 0.85, 'gamma': 0.62}}
        """
        if oscillators is None:
            oscillators = ["theta", "gamma"]  # Most important for curriculum

        # Auto-generate all pairs if not specified
        if region_pairs is None:
            regions = list(region_phases.keys())
            region_pairs = []
            for i, r1 in enumerate(regions):
                for r2 in regions[i + 1 :]:
                    region_pairs.append((r1, r2))

        coherence_map: dict[str, dict[str, float]] = {}

        for region1, region2 in region_pairs:
            if region1 not in region_phases or region2 not in region_phases:
                continue

            pair_key = f"{region1}-{region2}"
            coherence_map[pair_key] = {}

            for osc in oscillators:
                coherence = self.compute_phase_coherence(
                    region_phases[region1],
                    region_phases[region2],
                    oscillator=osc,
                )
                coherence_map[pair_key][osc] = coherence

        return coherence_map

    def check_cross_region_synchrony(
        self,
        region_phases: Dict[str, Dict[str, float]],
        expected_synchrony: Optional[Dict[tuple, Dict[str, float]]] = None,
    ) -> List[OscillatorIssueReport]:
        """Check for expected cross-region synchrony and detect problems.

        Validates curriculum mechanisms:
        - Working memory tasks: Expect high hippocampus-PFC theta coherence
        - Cross-modal binding: Expect high visual-auditory gamma coherence

        Args:
            region_phases: Dict mapping region_name → {oscillator: phase}
            expected_synchrony: Dict of expected coherence thresholds
                Example: {('hippocampus', 'prefrontal'): {'theta': 0.6}}
                If None, uses default curriculum expectations

        Returns:
            List of issues if synchrony is below expected levels
        """
        issues = []

        # Default expectations from curriculum
        if expected_synchrony is None:
            expected_synchrony = {
                # Stage 1+: Working memory requires hippocampus-PFC theta coherence
                ("hippocampus", "prefrontal"): {"theta": 0.5},
                # Stage 2+: Cross-modal binding requires visual-auditory gamma coherence
                ("visual_cortex", "auditory_cortex"): {"gamma": 0.4},
                ("cortex", "hippocampus"): {"gamma": 0.3},  # General binding
            }

        # Compute actual coherence
        coherence_map = self.compute_region_pair_coherence(region_phases)

        # Check each expected synchrony
        for pair, osc_thresholds in expected_synchrony.items():
            region1, region2 = pair
            pair_key = f"{region1}-{region2}"
            reverse_key = f"{region2}-{region1}"

            # Check both orderings
            actual_coherence = coherence_map.get(pair_key) or coherence_map.get(reverse_key)

            if actual_coherence is None:
                continue  # Regions not present

            for osc, min_coherence in osc_thresholds.items():
                coherence = actual_coherence.get(osc, 0.0)

                if coherence < min_coherence:
                    severity = 60.0 * (min_coherence - coherence) / min_coherence
                    issues.append(
                        OscillatorIssueReport(
                            issue_type=OscillatorIssue.CROSS_REGION_DESYNCHRONY,
                            oscillator_name=f"{pair_key}-{osc}",
                            severity=min(100, severity),
                            description=f"Low {osc} coherence between {region1} and {region2}: {coherence:.3f} < {min_coherence:.3f}",
                            recommendation=f"Check {osc} oscillator synchronization or coupling strength between regions",
                            metrics={
                                "coherence": coherence,
                                "expected": min_coherence,
                            },
                        )
                    )

        return issues
