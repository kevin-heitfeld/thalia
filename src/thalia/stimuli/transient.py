"""Transient (phasic) stimulus pattern."""

from __future__ import annotations

from typing import Optional

import torch

from .base import StimulusPattern


class Transient(StimulusPattern):
    """Transient stimulus presented briefly then removed.

    Represents phasic input like:
    - Brief visual flash
    - Touch/tap on skin
    - Brief auditory click
    """

    def __init__(
        self,
        pattern: torch.Tensor,
        onset_ms: float = 0.0,
        duration_ms: float = 1.0,
    ):
        """Initialize transient stimulus.

        Args:
            pattern: Input tensor to present [input_size]
            onset_ms: When to present (default: 0.0)
            duration_ms: How long to present (default: 1.0ms, single timestep)
        """
        self.pattern = pattern
        self.onset_ms = onset_ms
        self.duration_ms = duration_ms

    def get_input(self, timestep: int, dt_ms: float) -> Optional[torch.Tensor]:
        """Get input for timestep (pattern if at onset, else None)."""
        current_time_ms = timestep * dt_ms

        if self.onset_ms <= current_time_ms < self.onset_ms + self.duration_ms:
            return self.pattern
        return None

    def duration_timesteps(self, dt_ms: float) -> int:
        """Total duration in timesteps (onset + brief pulse)."""
        return int((self.onset_ms + self.duration_ms) / dt_ms)

    @property
    def device(self) -> torch.device:
        return self.pattern.device

    @property
    def shape(self) -> torch.Size:
        return self.pattern.shape
