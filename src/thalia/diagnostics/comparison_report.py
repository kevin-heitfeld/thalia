"""Structured comparison report — per-metric deltas, issue diffs, regression flags.

Pure data module: no simulation dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple


# =============================================================================
# METRIC DELTA
# =============================================================================

Transition = Literal["ok", "fixed", "regressed", "still_oob", "new", "removed"]


@dataclass
class MetricDelta:
    """Comparison of a single scalar metric between two runs."""

    key: str
    old_value: Optional[float]
    new_value: Optional[float]
    bio_range: Optional[Tuple[float, float]] = None

    @property
    def delta(self) -> Optional[float]:
        if self.old_value is not None and self.new_value is not None:
            return self.new_value - self.old_value
        return None

    @property
    def old_in_range(self) -> Optional[bool]:
        if self.old_value is None or self.bio_range is None:
            return None
        return self.bio_range[0] <= self.old_value <= self.bio_range[1]

    @property
    def new_in_range(self) -> Optional[bool]:
        if self.new_value is None or self.bio_range is None:
            return None
        return self.bio_range[0] <= self.new_value <= self.bio_range[1]

    @property
    def transition(self) -> Transition:
        """Classify the change between runs."""
        if self.old_value is None and self.new_value is not None:
            return "new"
        if self.old_value is not None and self.new_value is None:
            return "removed"
        o, n = self.old_in_range, self.new_in_range
        if o is None or n is None:
            return "ok"
        if not o and n:
            return "fixed"
        if o and not n:
            return "regressed"
        if not o and not n:
            return "still_oob"
        return "ok"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "delta": self.delta,
            "bio_range": list(self.bio_range) if self.bio_range else None,
            "transition": self.transition,
        }


# =============================================================================
# ISSUE DIFF
# =============================================================================


@dataclass
class IssueDiff:
    """Changes in health issues at a single severity level between runs."""

    severity: str
    resolved: List[str]
    added: List[str]
    unchanged: List[str]

    @property
    def old_count(self) -> int:
        return len(self.resolved) + len(self.unchanged)

    @property
    def new_count(self) -> int:
        return len(self.added) + len(self.unchanged)

    @property
    def has_new_issues(self) -> bool:
        return len(self.added) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "old_count": self.old_count,
            "new_count": self.new_count,
            "resolved": sorted(self.resolved),
            "added": sorted(self.added),
            "unchanged_count": len(self.unchanged),
        }


# =============================================================================
# COMPARISON REPORT
# =============================================================================


@dataclass
class ComparisonReport:
    """Structured comparison between two diagnostics runs.

    Designed for programmatic regression detection and CI integration.
    """

    # Issue diffs by severity
    critical_issues: IssueDiff
    warnings: IssueDiff

    # Per-population firing rate comparisons
    population_deltas: Dict[str, MetricDelta]

    # Per-region firing rate comparisons
    region_deltas: Dict[str, MetricDelta]

    # Global oscillatory metrics
    oscillation_deltas: Dict[str, MetricDelta] = field(default_factory=dict)

    # Neuromodulator peak concentrations
    neuromodulator_deltas: Dict[str, MetricDelta] = field(default_factory=dict)

    # Connectivity jitter
    connectivity_deltas: Dict[str, MetricDelta] = field(default_factory=dict)

    # Learning metrics (STDP timing, DA-eligibility alignment)
    learning_deltas: Dict[str, MetricDelta] = field(default_factory=dict)

    # Inferred brain state transition
    brain_state_old: str = "unknown"
    brain_state_new: str = "unknown"

    # ── Derived properties ──────────────────────────────────────────────

    @property
    def n_ok(self) -> int:
        return sum(1 for d in self.population_deltas.values() if d.transition == "ok")

    @property
    def n_fixed(self) -> int:
        return sum(1 for d in self.population_deltas.values() if d.transition == "fixed")

    @property
    def n_regressed(self) -> int:
        return sum(1 for d in self.population_deltas.values() if d.transition == "regressed")

    @property
    def n_still_oob(self) -> int:
        return sum(1 for d in self.population_deltas.values() if d.transition == "still_oob")

    @property
    def has_regressions(self) -> bool:
        """True if any population regressed or new critical issues appeared."""
        return self.n_regressed > 0 or self.critical_issues.has_new_issues

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable representation for CI and trend tracking."""
        return {
            "critical_issues": self.critical_issues.to_dict(),
            "warnings": self.warnings.to_dict(),
            "population_summary": {
                "n_ok": self.n_ok,
                "n_fixed": self.n_fixed,
                "n_regressed": self.n_regressed,
                "n_still_oob": self.n_still_oob,
            },
            "population_deltas": {
                k: d.to_dict() for k, d in self.population_deltas.items()
            },
            "region_deltas": {
                k: d.to_dict() for k, d in self.region_deltas.items()
            },
            "oscillation_deltas": {
                k: d.to_dict() for k, d in self.oscillation_deltas.items()
            },
            "neuromodulator_deltas": {
                k: d.to_dict() for k, d in self.neuromodulator_deltas.items()
            },
            "connectivity_deltas": {
                k: d.to_dict() for k, d in self.connectivity_deltas.items()
            },
            "brain_state_transition": (
                f"{self.brain_state_old} → {self.brain_state_new}"
                if self.brain_state_old != self.brain_state_new
                else self.brain_state_new
            ),
            "has_regressions": self.has_regressions,
        }
