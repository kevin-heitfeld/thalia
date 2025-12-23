"""Utility Functions.

General utility functions for the Thalia framework.
"""

from thalia.utils.core_utils import (
    clamp_weights,
    cosine_similarity_safe,
    ensure_1d,
    zeros_like_config,
    ones_like_config,
    assert_single_instance,
)
from thalia.utils.delay_buffer import CircularDelayBuffer
from thalia.utils.oscillator_utils import (
    compute_theta_encoding_retrieval,
    compute_ach_recurrent_suppression,
    compute_theta_gamma_coupling_gate,
    compute_oscillator_modulated_gain,
    compute_learning_rate_modulation,
)
from thalia.utils.time_constants import MS_PER_SECOND, SECONDS_PER_MS, TAU, TWO_PI

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
    "MS_PER_SECOND",
    "SECONDS_PER_MS",
    "TAU",
    "TWO_PI",
]
