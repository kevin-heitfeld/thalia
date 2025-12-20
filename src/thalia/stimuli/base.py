"""Base class for stimulus patterns."""

from abc import ABC, abstractmethod
from typing import Optional
import torch


class StimulusPattern(ABC):
    """Abstract base class for temporal stimulus patterns.

    All stimulus patterns must implement get_input(timestep, dt_ms) to provide
    the input tensor for a given timestep.

    This abstraction cleanly separates:
    - **What** the input is (pattern data)
    - **When** it's presented (temporal dynamics)
    - **How** it's generated (eager vs lazy)

    Subclasses:
        - Sustained: Constant input held over duration
        - Transient: Brief pulse at specific time
        - Sequential: Pre-defined sequence of inputs
        - Programmatic: Generated on-demand via function
    """

    @abstractmethod
    def get_input(self, timestep: int, dt_ms: float) -> Optional[torch.Tensor]:
        """Get input tensor for given timestep.

        Args:
            timestep: Current timestep index (0-based)
            dt_ms: Timestep duration in milliseconds

        Returns:
            Input tensor for this timestep, or None if no input
        """
        pass

    @abstractmethod
    def duration_timesteps(self, dt_ms: float) -> int:
        """Get total duration in timesteps.

        Args:
            dt_ms: Timestep duration in milliseconds

        Returns:
            Number of timesteps this stimulus lasts
        """
        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Device where stimulus tensors are located."""
        pass

    @property
    @abstractmethod
    def shape(self) -> torch.Size:
        """Shape of input tensors."""
        pass
