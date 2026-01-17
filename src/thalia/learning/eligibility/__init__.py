"""Eligibility Trace Management.

Provides eligibility trace tracking for three-factor learning rules and STDP.
"""

from __future__ import annotations

from thalia.learning.eligibility.trace_manager import (
    EligibilityTraceManager,
    STDPConfig,
)

__all__ = [
    "EligibilityTraceManager",
    "STDPConfig",
]
