"""Programmatic stimulus pattern."""

from __future__ import annotations

from typing import Callable, Optional

import torch

from .base import StimulusPattern


class Programmatic(StimulusPattern):
    """Programmatic stimulus generated on-demand via function.

    Represents stimuli that are computed dynamically:
    - Algorithmic patterns (e.g., moving gratings)
    - Closed-loop feedback (input depends on output)
    - Infinite/unbounded sequences
    """

    def __init__(
        self,
        fn: Callable[[float], Optional[torch.Tensor]],
        duration_ms: Optional[float] = None,
        device: torch.device = torch.device("cpu"),
    ):
        """Initialize programmatic stimulus.

        Args:
            fn: Function taking time_ms and returning input tensor or None
            duration_ms: Optional duration limit (None = infinite)
            device: Device for tensors (required for shape/device queries)
        """
        self.fn = fn
        self.duration_ms = duration_ms
        self._device = device

        # Generate sample to infer shape
        sample = fn(0.0)
        if sample is not None:
            self._shape = sample.shape
        else:
            # Try a few timesteps to find non-None sample
            for t in range(100):
                sample = fn(float(t))
                if sample is not None:
                    self._shape = sample.shape
                    break
            else:
                raise ValueError("Programmatic stimulus returns None for first 100 timesteps")

    def get_input(self, timestep: int, dt_ms: float) -> Optional[torch.Tensor]:
        """Get input for timestep by calling function."""
        current_time_ms = timestep * dt_ms

        if self.duration_ms is not None and current_time_ms >= self.duration_ms:
            return None

        return self.fn(current_time_ms)

    def duration_timesteps(self, dt_ms: float) -> int:
        """Total duration in timesteps (infinite if duration_ms is None)."""
        if self.duration_ms is None:
            return int(1e9)  # Effectively infinite
        return int(self.duration_ms / dt_ms)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def shape(self) -> torch.Size:
        return self._shape
