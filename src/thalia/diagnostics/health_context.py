"""Health check context — unified data bundle for all health-check functions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .diagnostics_config import HealthThresholds
from .diagnostics_metrics import (
    ConnectivityStats,
    HomeostaticStats,
    LearningStats,
    OscillatoryStats,
    PopulationStats,
    RegionStats,
)
from .diagnostics_report import HealthIssue
from .diagnostics_snapshot import RecorderSnapshot

from .brain_state_classifier import BrainState


@dataclass
class HealthCheckContext:
    """Bundles all data available to health-check functions.

    Every health check receives a single ``ctx: HealthCheckContext`` argument
    and pulls whichever fields it needs.  This eliminates per-check parameter
    variation and makes the :func:`~.analysis_health.assess_health`
    orchestrator a simple flat loop.

    Attributes
    ----------
    rec : RecorderSnapshot
        Full recording snapshot (neuron/region metadata + raw buffers).
    pop_stats : dict
        Per-population firing statistics keyed by ``(region, population)``.
    region_stats : dict
        Per-region aggregate statistics keyed by region name.
    connectivity : ConnectivityStats
        Axonal tract connectivity metrics.
    oscillations : OscillatoryStats
        Oscillatory decomposition, coherence, PAC, etc.
    homeostasis : HomeostaticStats
        Homeostatic gain and STP final-state data.
    learning : LearningStats or None
        Learning-specific metrics (``None`` when learning is inactive).
    issues : list of HealthIssue
        Shared mutable list — each check appends findings here.
    population_status : dict
        Mutable ``pop_key → bio_plausibility`` map populated by
        :func:`~.health_firing.check_population_firing`.
    thresholds : HealthThresholds
        Diagnostic thresholds (shortcut for ``rec.config.thresholds``).
    skip_sfa_check : bool
        Whether to suppress SFA-index warnings (ramp inputs).
    sensory_pattern : str
        Active sensory pattern name (e.g. ``"slow_wave"``).
    inferred_brain_state : BrainState
        Physiological state inferred from band-power ratios.
    """

    rec: RecorderSnapshot
    pop_stats: Dict[Tuple[str, str], PopulationStats]
    region_stats: Dict[str, RegionStats]
    connectivity: ConnectivityStats
    oscillations: OscillatoryStats
    homeostasis: HomeostaticStats
    learning: Optional[LearningStats]
    issues: List[HealthIssue] = field(default_factory=list)
    population_status: Dict[str, str] = field(default_factory=dict)
    thresholds: HealthThresholds = field(init=False)
    skip_sfa_check: bool = field(init=False)
    sensory_pattern: str = field(init=False)
    inferred_brain_state: BrainState = "unknown"

    def __post_init__(self) -> None:
        self.thresholds = self.rec.config.thresholds
        self.skip_sfa_check = self.rec.config.skip_sfa_health_check
        self.sensory_pattern = self.rec.config.sensory_pattern
