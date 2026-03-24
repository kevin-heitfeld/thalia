"""Brain diagnostics and analysis tools for Thalia."""

from __future__ import annotations

from .bio_ranges import (
    RegionSpec,
    EEG_BANDS,
    bio_range,
    ei_ratio_thresholds,
    expected_dominant_band,
    nm_tonic_range,
)
from .brain_protocol import BrainLike
from .diagnostics_snapshot import (
    RecorderSnapshot,
    RunningStats,
)
from .diagnostics_snapshot_io import (
    load_snapshot,
    save_report,
    save_snapshot,
)
from .diagnostics_text_report import (
    print_brain_config,
    print_neuron_populations,
    print_report,
    print_synaptic_weights,
)
from .diagnostics_recorder import (
    DiagnosticsRecorder,
)
from .diagnostics_config import (
    ConnectivityThresholds,
    DiagnosticsConfig,
    FiringThresholds,
    HealthThresholds,
    HomeostasisThresholds,
    LearningThresholds,
    NeuromodulatorThresholds,
    OscillationThresholds,
    RegionalThresholds,
)
from .diagnostics_metrics import (
    AvalancheStats,
    BetaBurstRegionStats,
    CerebellarCouplingStats,
    ConnectivityStats,
    HomeostaticStats,
    LaminarCascadeRegionStats,
    LearningStats,
    OscillatoryStats,
    PlvThetaStats,
    PopulationStats,
    RegionStats,
    STDPTimingStats,
    SpikeFieldResult,
    SwrCouplingRegionStats,
    SynapseLearningSummary,
    ThetaSequenceRegionStats,
    WeightDistStats,
)
from .diagnostics_report import (
    DiagnosticsReport,
    HealthCategory,
    HealthIssue,
    HealthReport,
    PerformanceMetrics,
)
from .rate_predictor import (
    InputSpec,
    RatePrediction,
    predict_rate,
)
from .region_test_runner import (
    RegionTestResult,
    RegionTestRunner,
)
from .sensory_patterns import (
    SENSORY_PATTERNS,
    WAKING_PATTERNS,
    SLEEP_PATTERNS,
    NEUTRAL_PATTERNS,
    make_sensory_input,
)
from .stp_calculator import (
    STPResult,
    stp_eq,
    stp_table,
)
from .health_preflight import (
    validate_brain,
)
from .simulation_loop import simulate
from .sweep import (
    run_single,
    run_sweep,
    DEFAULT_SWEEP_PATTERNS,
)
from .brain_state_classifier import (
    BrainState,
    classify_brain_state,
)
from .comparison import (
    compare_reports,
    run_comparison,
)
from .comparison_report import (
    ComparisonReport,
    IssueDiff,
    MetricDelta,
)
from .comparison_text import (
    format_comparison_text,
)

__all__ = [
    # Biological reference data
    "RegionSpec",
    "EEG_BANDS",
    "bio_range",
    "ei_ratio_thresholds",
    "expected_dominant_band",
    "nm_tonic_range",
    # Protocol
    "BrainLike",
    # Config & result types
    "ConnectivityThresholds",
    "DiagnosticsConfig",
    "FiringThresholds",
    "HealthThresholds",
    "HomeostasisThresholds",
    "LearningThresholds",
    "NeuromodulatorThresholds",
    "OscillationThresholds",
    "RegionalThresholds",
    "DiagnosticsReport",
    "AvalancheStats",
    "BetaBurstRegionStats",
    "CerebellarCouplingStats",
    "ConnectivityStats",
    "HealthCategory",
    "HealthIssue",
    "HealthReport",
    "HomeostaticStats",
    "LaminarCascadeRegionStats",
    "LearningStats",
    "OscillatoryStats",
    "PerformanceMetrics",
    "PlvThetaStats",
    "PopulationStats",
    "RecorderSnapshot",
    "RegionStats",
    "RunningStats",
    "STDPTimingStats",
    "SpikeFieldResult",
    "SwrCouplingRegionStats",
    "SynapseLearningSummary",
    "ThetaSequenceRegionStats",
    "WeightDistStats",
    # I/O helpers (report-only, no live brain state needed)
    "print_report",
    "save_report",
    "save_snapshot",
    "load_snapshot",
    # Brain summary helpers
    "print_brain_config",
    "print_neuron_populations",
    "print_synaptic_weights",
    # Rate predictor
    "InputSpec",
    "RatePrediction",
    "predict_rate",
    # Recorder
    "DiagnosticsRecorder",
    "RecorderSnapshot",
    # Region tests
    "RegionTestResult",
    "RegionTestRunner",
    # Sensory input patterns
    "SENSORY_PATTERNS",
    "WAKING_PATTERNS",
    "SLEEP_PATTERNS",
    "NEUTRAL_PATTERNS",
    "make_sensory_input",
    # Pre-flight validation
    "validate_brain",
    # Sweep mode
    "simulate",
    "run_single",
    "run_sweep",
    "DEFAULT_SWEEP_PATTERNS",
    # Brain state classifier
    "BrainState",
    "classify_brain_state",
    # Run comparison
    "compare_reports",
    "run_comparison",
    "ComparisonReport",
    "IssueDiff",
    "MetricDelta",
    "format_comparison_text",
    # STP equilibrium calculator
    "STPResult",
    "stp_eq",
    "stp_table",
]
