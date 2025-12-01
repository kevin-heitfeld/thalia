"""
Diagnostic tools for understanding and debugging SNN behavior.

This module provides:
- DiagnosticConfig: Control what diagnostics are collected/reported
- MechanismConfig: Enable/disable specific neural mechanisms for ablation studies
- ExperimentDiagnostics: Central collector for all diagnostic data
- EligibilityTracker: Track eligibility trace dynamics for three-factor learning
- DopamineTracker: Track dopamine signal dynamics for reward-modulated learning
"""

from .config import DiagnosticConfig, DiagnosticLevel, MechanismConfig
from .collectors import (
    ExperimentDiagnostics,
    SpikeTimingAnalyzer,
    WeightChangeTracker,
    MechanismStateTracker,
    WinnerConsistencyTracker,
    EligibilityTracker,
    DopamineTracker,
)

__all__ = [
    "DiagnosticConfig",
    "DiagnosticLevel",
    "MechanismConfig",
    "ExperimentDiagnostics",
    "SpikeTimingAnalyzer",
    "WeightChangeTracker",
    "MechanismStateTracker",
    "WinnerConsistencyTracker",
    "EligibilityTracker",
    "DopamineTracker",
]
