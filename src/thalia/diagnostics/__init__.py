"""Brain diagnostics and analysis tools for Thalia."""

from __future__ import annotations

from .analysis_tuning import (
    TractContribution,
    TuningGuidance,
    TuningReport,
    compute_tuning,
    print_tuning_report,
)
from .calibration_advisor import (
    CalibrationAdvice,
    CalibrationReport,
    ParameterRecommendation,
    compute_calibration_advice,
    print_calibration_advice,
)
from .bio_ranges import (
    RegionSpec,
    EEG_BANDS,
    bio_range,
    ei_ratio_thresholds,
    expected_dominant_band,
    nm_tonic_range,
)
from .diagnostics_io import (
    print_brain_config,
    print_neuron_populations,
    print_report,
    print_synaptic_weights,
    save_snapshot,
    load_snapshot,
    save_report,
)
from .diagnostics_recorder import (
    DiagnosticsRecorder,
)
from .diagnostics_types import (
    DiagnosticsConfig,
    ConnectivityStats,
    HealthThresholds,
    DiagnosticsReport,
    HealthCategory,
    HealthReport,
    HomeostaticStats,
    OscillatoryStats,
    PopulationStats,
    RecorderSnapshot,
    RegionStats,
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
from .sweep import (
    run_single,
    run_sweep,
)
from .triage import (
    run_triage,
)

__all__ = [
    # Biological reference data
    "REGION_SPECS",
    "RegionSpec",
    "EEG_BANDS",
    "bio_range",
    "ei_ratio_thresholds",
    "expected_dominant_band",
    "nm_tonic_range",
    # Config & result types
    "DiagnosticsConfig",
    "HealthThresholds",
    "DiagnosticsReport",
    "ConnectivityStats",
    "HealthCategory",
    "HealthReport",
    "HomeostaticStats",
    "OscillatoryStats",
    "PopulationStats",
    "RegionStats",
    # I/O helpers (report-only, no live brain state needed)
    "print_report",
    "save_snapshot",
    "load_snapshot",
    "save_report",
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
    # Sweep mode
    "run_single",
    "run_sweep",
    # Triage
    "run_triage",
    # STP equilibrium calculator
    "STPResult",
    "stp_eq",
    "stp_table",
    # Tuning guidance
    "TractContribution",
    "TuningGuidance",
    "TuningReport",
    "compute_tuning",
    "print_tuning_report",
    # Calibration advisor
    "CalibrationAdvice",
    "CalibrationReport",
    "ParameterRecommendation",
    "compute_calibration_advice",
    "print_calibration_advice",
]
