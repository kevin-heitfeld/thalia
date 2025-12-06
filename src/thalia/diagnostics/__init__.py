"""
Diagnostics package for THALIA.

This package provides monitoring and analysis tools for network health:

- Criticality monitoring (branching ratio, avalanche analysis)
- Health checks (activity levels, weight magnitudes, E/I balance)
- Performance profiling
"""

from .criticality import (
    CriticalityConfig,
    CriticalityMonitor,
    CriticalityState,
    AvalancheAnalyzer,
)

__all__ = [
    "CriticalityConfig",
    "CriticalityMonitor",
    "CriticalityState",
    "AvalancheAnalyzer",
]
