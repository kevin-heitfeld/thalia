"""Diagnostics report structures — health issues and the final report dataclass.

Pure data module: no simulation dependencies.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Dict, List, Literal, Optional

import numpy as np

from thalia.typing import RegionName

from .diagnostics_metrics import (
    ConnectivityStats,
    HomeostaticStats,
    LearningStats,
    OscillatoryStats,
    RegionStats,
)


# =============================================================================
# HEALTH CATEGORY
# =============================================================================


class HealthCategory(StrEnum):
    """Enumerated category tags for :class:`HealthIssue`."""

    PREFLIGHT = "preflight"
    FIRING = "firing"
    EI_BALANCE = "ei_balance"
    OSCILLATIONS = "oscillations"
    NEUROMODULATORS = "neuromodulators"
    HOMEOSTASIS = "homeostasis"
    CONNECTIVITY = "connectivity"
    CEREBELLAR = "cerebellar"
    THALAMUS = "thalamus"
    INTERNEURON_COVERAGE = "interneuron_coverage"
    LAMINAR = "laminar"
    LEARNING = "learning"


# =============================================================================
# HEALTH ISSUE
# =============================================================================


@dataclass
class HealthIssue:
    """A single health issue with severity, source category, and message."""

    severity: Literal["critical", "warning", "info"]
    category: HealthCategory
    message: str
    population: Optional[str] = None
    region: Optional[str] = None

    def to_dict(self) -> Dict[str, Optional[str]]:
        """Return a JSON-serializable dictionary."""
        return {
            "severity": self.severity,
            "category": self.category.value,
            "message": self.message,
            "population": self.population,
            "region": self.region,
        }


# =============================================================================
# HEALTH REPORT
# =============================================================================


@dataclass
class HealthReport:
    """Per-population and global health assessment."""

    critical_issues: List[str]
    warnings: List[str]

    population_status: Dict[str, str]

    n_populations_ok: int
    n_populations_low: int
    n_populations_high: int
    n_populations_unknown: int

    all_issues: List[HealthIssue] = field(default_factory=list)
    global_brain_state: str = "unknown/none/unknown"
    inferred_brain_state: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Return a structured JSON-serializable dictionary.

        Includes issues grouped by both category and severity for
        machine-readable CI integration, plus flat summary counts.
        """
        by_severity: Dict[str, List[Dict[str, Optional[str]]]] = defaultdict(list)
        by_category: Dict[str, List[Dict[str, Optional[str]]]] = defaultdict(list)
        for issue in self.all_issues:
            d = issue.to_dict()
            by_severity[issue.severity].append(d)
            by_category[issue.category.value].append(d)

        return {
            "global_brain_state": self.global_brain_state,
            "inferred_brain_state": self.inferred_brain_state,
            "summary": {
                "n_critical": len(by_severity.get("critical", [])),
                "n_warning": len(by_severity.get("warning", [])),
                "n_info": len(by_severity.get("info", [])),
                "n_total": len(self.all_issues),
            },
            "population_count": {
                "ok": self.n_populations_ok,
                "low": self.n_populations_low,
                "high": self.n_populations_high,
                "unknown": self.n_populations_unknown,
            },
            "population_status": self.population_status,
            "issues": [issue.to_dict() for issue in self.all_issues],
            "issues_by_severity": dict(by_severity),
            "issues_by_category": dict(by_category),
        }


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================


@dataclass
class PerformanceMetrics:
    """Runtime performance breakdown for a simulation + analysis pass.

    All time fields are in seconds.
    """

    wall_clock_s: float
    forward_s: float
    record_s: float
    analysis_s: float = 0.0
    n_timesteps: int = 0

    @property
    def us_per_step(self) -> float:
        """Microseconds of wall-clock time per simulation step."""
        return self.wall_clock_s / max(self.n_timesteps, 1) * 1e6

    @property
    def forward_pct(self) -> float:
        """Percentage of simulation time spent in ``brain.forward()``."""
        return self.forward_s / max(self.wall_clock_s, 1e-12) * 100.0

    @property
    def record_pct(self) -> float:
        """Percentage of simulation time spent in ``recorder.record()``."""
        return self.record_s / max(self.wall_clock_s, 1e-12) * 100.0

    @property
    def overhead_pct(self) -> float:
        """Percentage not accounted for by forward + record (input gen, etc.)."""
        return 100.0 - self.forward_pct - self.record_pct

    def to_dict(self) -> Dict[str, float]:
        """Return a JSON-serializable dictionary."""
        return {
            "wall_clock_s": round(self.wall_clock_s, 4),
            "forward_s": round(self.forward_s, 4),
            "record_s": round(self.record_s, 4),
            "analysis_s": round(self.analysis_s, 4),
            "n_timesteps": self.n_timesteps,
            "us_per_step": round(self.us_per_step, 1),
            "forward_pct": round(self.forward_pct, 1),
            "record_pct": round(self.record_pct, 1),
            "overhead_pct": round(self.overhead_pct, 1),
        }


# =============================================================================
# DIAGNOSTICS REPORT
# =============================================================================


@dataclass
class DiagnosticsReport:
    """Complete diagnostics report."""

    # Meta
    timestamp: float
    simulation_time_ms: float
    n_timesteps: int

    # Core data
    regions: Dict[RegionName, RegionStats]
    oscillations: OscillatoryStats
    connectivity: ConnectivityStats
    homeostasis: HomeostaticStats
    health: HealthReport

    transient_steps: int = 0

    neuromodulator_levels: Optional[Dict[str, np.ndarray]] = None

    # Raw traces kept for plotting
    raw_spike_counts: Optional[np.ndarray] = None
    raw_voltages: Optional[np.ndarray] = None
    voltage_sample_times_ms: Optional[np.ndarray] = None
    conductance_sample_times_ms: Optional[np.ndarray] = None
    raw_g_exc: Optional[np.ndarray] = None
    raw_g_inh: Optional[np.ndarray] = None
    pop_rate_binned: Optional[np.ndarray] = field(default=None, repr=False)
    region_rate_binned: Optional[np.ndarray] = None
    pop_keys: Optional[List[tuple[str, str]]] = None
    region_keys: Optional[List[str]] = None

    effective_synaptic_gain: Optional[Dict[str, float]] = None

    learning: Optional[LearningStats] = None

    performance: Optional[PerformanceMetrics] = None
