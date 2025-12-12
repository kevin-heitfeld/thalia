"""Eligibility Trace Management.

Provides eligibility trace tracking for three-factor learning rules and STDP.
"""

from thalia.learning.eligibility.trace_manager import (
    EligibilityTraceManager,
    STDPConfig,
)

__all__ = [
    "EligibilityTraceManager",
    "STDPConfig",
]
