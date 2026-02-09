"""
Diagnostics package for THALIA.

This package provides monitoring and analysis tools for network health:

- Health checks (activity levels, weight magnitudes, E/I balance)
- Performance profiling (timing, memory, throughput)
- Interactive dashboard for real-time monitoring
- Automatic diagnostic collection utilities
- Metacognitive monitoring (confidence estimation, abstention)
"""

from __future__ import annotations

from .auto_collect import (
    auto_diagnostics,
)
from .diagnostics import (
    DiagnosticLevel,
    MainDiagnosticsConfig,
    DiagnosticsManager,
    DiagnosticsUtils,
    compute_activity_metrics,
    compute_health_metrics,
    compute_plasticity_metrics,
)
from .health_monitor import (
    HealthConfig,
    HealthIssue,
    HealthMonitor,
    HealthReport,
    IssueReport,
)
from .oscillator_health import (
    OscillatorHealthConfig,
    OscillatorHealthMonitor,
    OscillatorHealthReport,
    OscillatorIssue,
    OscillatorIssueReport,
)

__all__ = [
    # Core diagnostics
    "DiagnosticLevel",
    "MainDiagnosticsConfig",
    "DiagnosticsManager",
    # Diagnostics utilities
    "DiagnosticsUtils",
    "compute_activity_metrics",
    "compute_health_metrics",
    "compute_plasticity_metrics",
    # Health monitoring
    "HealthConfig",
    "HealthMonitor",
    "HealthReport",
    "IssueReport",
    "HealthIssue",
    # Oscillator health monitoring
    "OscillatorHealthConfig",
    "OscillatorHealthMonitor",
    "OscillatorHealthReport",
    "OscillatorIssueReport",
    "OscillatorIssue",
    # Auto-collection utilities
    "auto_diagnostics",
]
