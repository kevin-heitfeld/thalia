"""Utility Functions.

General utility functions for the Thalia framework.
"""

from __future__ import annotations

from .delay_buffer import (
    CircularDelayBuffer,
)
from .neuromodulation import (
    compute_ne_gain,
)
from .numerical_validation import (
    set_numerical_validation,
    validate_finite,
)
from .oscillator_utils import (
    compute_ach_recurrent_suppression,
    compute_learning_rate_modulation,
)
from .spike_utils import (
    validate_spike_tensor,
    validate_spike_tensors,
    compute_firing_rate,
    compute_spike_count,
    cosine_similarity_safe,
)
from .weight_utils import (
    clamp_weights,
)

__all__ = [
    # Delay Buffer
    "CircularDelayBuffer",
    # Neuromodulation
    "compute_ne_gain",
    # Numerical Validation
    "set_numerical_validation",
    "validate_finite",
    # Oscillator Utilities
    "compute_ach_recurrent_suppression",
    "compute_learning_rate_modulation",
    # Spike Utilities
    "validate_spike_tensor",
    "validate_spike_tensors",
    "compute_firing_rate",
    "compute_spike_count",
    "cosine_similarity_safe",
    # Weight Utilities
    "clamp_weights",
]
