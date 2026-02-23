"""Global configuration constants for Thalia."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GlobalConfig:
    """Global configuration constants for Thalia.

    This module centralizes global constants that affect the entire simulation,
    such as time conversion factors, global learning enable/disable flags, and
    synaptic weight scaling. These constants can be imported and used across all
    regions and components to ensure consistency.
    """

    DEFAULT_DT_MS = 1.0
    """Default timestep in milliseconds (1.0 ms)."""

    HOMEOSTASIS_DISABLED: bool = False  # Set to True to disable homeostatic plasticity (intrinsic excitability, threshold adaptation, synaptic scaling)
    """Global homeostatic plasticity enable flag."""

    LEARNING_DISABLED: bool = False  # Set to True to disable all synaptic plasticity
    """Global learning/plasticity enable flag."""

    NEUROMODULATION_DISABLED: bool = False  # Set to True to disable all neuromodulator effects (DA, NE, ACh)
    """Global neuromodulation enable flag."""

    SYNAPTIC_WEIGHT_SCALE: float = 1.0  # Set to 0.0 to disable synaptic conductances and test intrinsic excitability alone
    """Global weight scale factor for conductance-based synapses."""
