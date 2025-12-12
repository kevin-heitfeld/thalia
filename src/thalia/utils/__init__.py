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

__all__ = [
    "clamp_weights",
    "cosine_similarity_safe",
    "ensure_1d",
    "zeros_like_config",
    "ones_like_config",
    "assert_single_instance",
]
