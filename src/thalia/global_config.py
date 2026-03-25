"""Global configuration constants for Thalia."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class _GlobalConfig:
    """Global configuration constants for Thalia.

    This module centralizes global constants that affect the entire simulation,
    such as time conversion factors, global learning enable/disable flags, and
    synaptic weight scaling.
    """

    DEBUG: bool = False  # Set to True to enable runtime tensor validation
    """Enable expensive runtime assertions and tensor validation. Leave False in production."""

    DEFAULT_DEVICE: str = "cpu"  # Default device for tensors (can be overridden per brain/region)
    """Default device for tensors (can be overridden per brain/region)."""

    DEFAULT_DT_MS: float = 1.0
    """Default timestep in milliseconds (1.0 ms)."""


GlobalConfig = _GlobalConfig()
