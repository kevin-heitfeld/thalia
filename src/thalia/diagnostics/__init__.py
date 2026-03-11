"""Brain diagnostics and analysis tools for Thalia."""

from __future__ import annotations

from .bio_ranges import (
    REGION_SPECS,
    RegionSpec,
    EEG_BANDS,
    bio_range,
    ei_ratio_thresholds,
    expected_dominant_band,
    nm_tonic_range,
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
    RegionStats,
)
from .diagnostics_io import (
    print_brain_config,
    print_neuron_populations,
    print_report,
    print_synaptic_weights,
    save,
)
from .diagnostics_recorder import (
    DiagnosticsRecorder,
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
from .sweep import (
    plot_sweep_comparison,
    run_sweep,
    simulate,
)
from .triage import (
    run_triage,
)
from .rate_predictor import (
    InputSpec,
    RatePrediction,
    predict_rate,
)
from .stp_calculator import (
    STPResult,
    stp_eq,
    stp_table,
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
    "save",
    # Brain summary helpers
    "print_brain_config",
    "print_neuron_populations",
    "print_synaptic_weights",
    # Recorder
    "DiagnosticsRecorder",
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
    "simulate",
    "run_sweep",
    "plot_sweep_comparison",
    # Triage
    "run_triage",
    # Analytical rate predictor
    "InputSpec",
    "RatePrediction",
    "predict_rate",
    # STP equilibrium calculator
    "STPResult",
    "stp_eq",
    "stp_table",
]
