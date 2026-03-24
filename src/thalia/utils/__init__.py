"""Utility Functions.

General utility functions for the Thalia framework.
"""

from __future__ import annotations

from .delay_buffer import (
    CircularDelayBuffer,
    HeterogeneousDelayBuffer,
)
from .neuromodulation import (
    compute_ach_recurrent_suppression,
    compute_da_gain,
    compute_ne_gain,
)
from .numerical_validation import (
    set_numerical_validation,
    validate_finite,
)
from .reward_coding import (
    generate_reward_spikes,
)
from .rng import (
    philox_uniform,
    philox_gaussian,
)
from .spike_utils import (
    validate_spike_tensor,
    compute_firing_rate,
    compute_spike_count,
    cosine_similarity_safe,
)
from .synapse_dicts import (
    SynapseIdParameterDict,
    SynapseIdModuleDict,
    SynapseIdBufferDict,
)
from .tensor_utils import (
    decay_float,
    decay_tensor,
)
from .weight_utils import (
    clamp_weights,
)

__all__ = [
    # Delay Buffer
    "CircularDelayBuffer",
    "HeterogeneousDelayBuffer",
    # Neuromodulation
    "compute_ach_recurrent_suppression",
    "compute_da_gain",
    "compute_ne_gain",
    # Numerical Validation
    "set_numerical_validation",
    "validate_finite",
    # Reward Coding
    "generate_reward_spikes",
    # RNG
    "philox_uniform",
    "philox_gaussian",
    # Spike Utilities
    "validate_spike_tensor",
    "compute_firing_rate",
    "compute_spike_count",
    "cosine_similarity_safe",
    # SynapseId-keyed containers
    "SynapseIdParameterDict",
    "SynapseIdModuleDict",
    "SynapseIdBufferDict",
    # Tensor Utilities
    "decay_float",
    "decay_tensor",
    # Weight Utilities
    "clamp_weights",
]
