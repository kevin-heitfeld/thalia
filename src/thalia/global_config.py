"""Global configuration constants for Thalia."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GlobalConfig:
    """Global configuration constants for Thalia.

    This module centralizes global constants that affect the entire simulation,
    such as time conversion factors, global learning enable/disable flags, and
    synaptic weight scaling.
    """

    DEBUG: bool = False  # Set to True to enable runtime tensor validation (validate_spike_tensor etc.)
    """Enable expensive runtime assertions and tensor validation. Leave False in production."""

    DEFAULT_DEVICE: str = "cpu"  # Default device for tensors (can be overridden per brain/region)
    """Default device for tensors (can be overridden per brain/region)."""

    DEFAULT_DT_MS: float = 1.0
    """Default timestep in milliseconds (1.0 ms)."""

    HOMEOSTASIS_DISABLED: bool = False  # Set to True to disable homeostatic plasticity (intrinsic excitability, threshold adaptation, synaptic scaling)
    """Global homeostatic plasticity enable flag."""

    LEARNING_DISABLED: bool = True  # Set to True to disable all synaptic plasticity
    """Global learning/plasticity enable flag."""

    NEUROMODULATION_DISABLED: bool = False  # Set to True to disable all neuromodulator effects (DA, NE, ACh)
    """Global neuromodulation enable flag."""

    SYNAPTIC_WEIGHT_SCALE: float = 1.0  # Set to 0.0 to disable synaptic conductances and test intrinsic excitability alone
    """Global weight scale factor for conductance-based synapses."""
