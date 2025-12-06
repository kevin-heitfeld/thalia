"""
Diagnostics package for THALIA.

This package provides monitoring and analysis tools for network health:

- Criticality monitoring (branching ratio, avalanche analysis)
- Health checks (activity levels, weight magnitudes, E/I balance)
- Performance profiling
- Interactive dashboard for real-time monitoring
"""

from .criticality import (
    CriticalityConfig,
    CriticalityMonitor,
    CriticalityState,
    AvalancheAnalyzer,
)
from .health_monitor import (
    HealthConfig,
    HealthMonitor,
    HealthReport,
    IssueReport,
    HealthIssue,
)
from .dashboard import Dashboard

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
    # Dashboard
    "Dashboard",
]
