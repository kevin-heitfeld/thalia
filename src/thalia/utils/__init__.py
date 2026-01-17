"""Utility Functions.

General utility functions for the Thalia framework.
"""

from __future__ import annotations

from thalia.utils.core_utils import (
    assert_single_instance,
    clamp_weights,
    cosine_similarity_safe,
    ensure_1d,
    ones_like_config,
    zeros_like_config,
)
from thalia.utils.delay_buffer import CircularDelayBuffer
from thalia.utils.oscillator_utils import (
    compute_ach_recurrent_suppression,
    compute_learning_rate_modulation,
    compute_oscillator_modulated_gain,
    compute_theta_encoding_retrieval,
    compute_theta_gamma_coupling_gate,
)

__all__ = [
    "clamp_weights",
    "cosine_similarity_safe",
    "ensure_1d",
    "zeros_like_config",
    "ones_like_config",
    "assert_single_instance",
    "CircularDelayBuffer",
    "compute_theta_encoding_retrieval",
    "compute_ach_recurrent_suppression",
    "compute_theta_gamma_coupling_gate",
    "compute_oscillator_modulated_gain",
    "compute_learning_rate_modulation",
]
