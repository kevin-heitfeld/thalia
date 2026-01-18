"""
Thalamus Module - Sensory Relay and Attentional Gating.

This module provides the thalamic relay implementation with:
- Sensory information routing
- Top-down attentional modulation
- Burst/tonic mode switching
- TRN-mediated inhibition

Author: Thalia Project
Date: January 16, 2026
"""

from __future__ import annotations

from .thalamus import ThalamicRelay, ThalamicRelayState

__all__ = [
    "ThalamicRelay",
    "ThalamicRelayState",
]
