"""
Diagnostics package for THALIA.

This package provides monitoring and analysis tools for network health:

- Criticality monitoring (branching ratio, avalanche analysis)
- Health checks (activity levels, weight magnitudes, E/I balance)
- Performance profiling (timing, memory, throughput)
- Interactive dashboard for real-time monitoring
- Automatic diagnostic collection utilities
- Metacognitive monitoring (confidence estimation, abstention)
"""

from __future__ import annotations

from .auto_collect import auto_diagnostics
from .criticality import (
    AvalancheAnalyzer,
    CriticalityConfig,
    CriticalityMonitor,
    CriticalityState,
)
from .dashboard import Dashboard
from .health_monitor import (
    HealthConfig,
    HealthIssue,
    HealthMonitor,
    HealthReport,
    IssueReport,
)
from .metacognition import (
    CalibrationNetwork,
    ConfidenceEstimator,
    MetacognitiveMonitor,
    MetacognitiveMonitorConfig,
    MetacognitiveStage,
)
from .oscillator_health import (
    OscillatorHealthConfig,
    OscillatorHealthMonitor,
    OscillatorHealthReport,
    OscillatorIssue,
    OscillatorIssueReport,
)
from .performance_profiler import (
    PerformanceProfiler,
    PerformanceStats,
    quick_profile,
)

__all__ = [
    # Criticality monitoring
    "CriticalityConfig",
    "CriticalityMonitor",
    "CriticalityState",
    "AvalancheAnalyzer",
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
    # Performance profiling
    "PerformanceProfiler",
    "PerformanceStats",
    "quick_profile",
    # Dashboard
    "Dashboard",
    # Auto-collection utilities
    "auto_diagnostics",
    # Metacognitive monitoring
    "MetacognitiveMonitor",
    "MetacognitiveMonitorConfig",
    "MetacognitiveStage",
    "ConfidenceEstimator",
    "CalibrationNetwork",
]
