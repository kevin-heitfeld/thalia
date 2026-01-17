"""
Visualization Tools for THALIA Training.

This module provides monitoring and diagnostics visualization:
- Training monitor (matplotlib-based)
- Live diagnostics
- Quick visualization utilities

Author: Thalia Project
Date: December 12, 2025
"""

from __future__ import annotations

from thalia.training.visualization.live_diagnostics import (
    LiveDiagnostics,
    quick_diagnostics,
)
from thalia.training.visualization.monitor import (
    TrainingMonitor,
    quick_monitor,
)

__all__ = [
    "TrainingMonitor",
    "quick_monitor",
    "LiveDiagnostics",
    "quick_diagnostics",
]
