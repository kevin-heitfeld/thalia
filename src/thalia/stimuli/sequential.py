"""Sequential stimulus pattern."""

from typing import List, Optional
import torch
from .base import StimulusPattern


class Sequential(StimulusPattern):
    """Sequential stimulus with explicit per-timestep inputs.

    Represents time-varying signals like:
    - Video frames
    - Speech waveform
    - Hand-written trajectory
    - Sensory sequences in experiments

    Example:
        >>> # Movie frames
        >>> frames = [torch.randn(128) for _ in range(100)]
        >>> stim = Sequential(frames)
        >>>
        >>> input_t0 = stim.get_input(0, dt_ms=1.0)   # Returns frames[0]
        >>> input_t50 = stim.get_input(50, dt_ms=1.0)  # Returns frames[50]
        >>> input_t100 = stim.get_input(100, dt_ms=1.0) # Returns None
    """

    def __init__(
        self,
        frames: List[torch.Tensor],
        onset_ms: float = 0.0,
    ):
        """Initialize sequential stimulus.

        Args:
            frames: List of input tensors, one per timestep
            onset_ms: When to start sequence (default: 0.0)
        """
        if not frames:
            raise ValueError("Sequential stimulus requires at least one frame")

        self.frames = frames
        self.onset_ms = onset_ms
        self._device = frames[0].device
        self._shape = frames[0].shape

        # Validate all frames have same shape and device
        for i, frame in enumerate(frames[1:], 1):
            if frame.shape != self._shape:
                raise ValueError(
                    f"Frame {i} has shape {frame.shape}, expected {self._shape}"
                )
            if frame.device != self._device:
                raise ValueError(
                    f"Frame {i} on device {frame.device}, expected {self._device}"
                )

    def get_input(self, timestep: int, dt_ms: float) -> Optional[torch.Tensor]:
        """Get input for timestep (frames[t] if within sequence, else None)."""
        current_time_ms = timestep * dt_ms

        if current_time_ms < self.onset_ms:
            return None

        frame_index = int((current_time_ms - self.onset_ms) / dt_ms)

        if 0 <= frame_index < len(self.frames):
            return self.frames[frame_index]
        return None

    def duration_timesteps(self, dt_ms: float) -> int:
        """Total duration in timesteps (onset + sequence length)."""
        return int(self.onset_ms / dt_ms) + len(self.frames)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def shape(self) -> torch.Size:
        return self._shape
