"""Sustained (tonic) stimulus pattern."""

from __future__ import annotations

from typing import Optional

import torch

from .base import StimulusPattern


class Sustained(StimulusPattern):
    """Sustained stimulus held constant over duration.

    Represents tonic input like:
    - Visual pattern shown continuously
    - Constant pressure on skin
    - Sustained auditory tone
    """

    def __init__(
        self,
        pattern: torch.Tensor,
        duration_ms: float,
        onset_ms: float = 0.0,
    ):
        """Initialize sustained stimulus.

        Args:
            pattern: Input tensor to present [input_size]
            duration_ms: How long to present in milliseconds
            onset_ms: When to start presenting (default: 0.0)
        """
        self.pattern = pattern
        self.duration_ms = duration_ms
        self.onset_ms = onset_ms

    def get_input(self, timestep: int, dt_ms: float) -> Optional[torch.Tensor]:
        """Get input for timestep (pattern if within duration, else None)."""
        current_time_ms = timestep * dt_ms

        if self.onset_ms <= current_time_ms < self.onset_ms + self.duration_ms:
            return self.pattern
        return None

    def duration_timesteps(self, dt_ms: float) -> int:
        """Total duration in timesteps (onset + hold duration)."""
        return int((self.onset_ms + self.duration_ms) / dt_ms)

    @property
    def device(self) -> torch.device:
        return self.pattern.device

    @property
    def shape(self) -> torch.Size:
        return self.pattern.shape
