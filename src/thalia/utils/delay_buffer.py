"""
Circular Delay Buffer - Efficient axonal delay implementation.

This module provides a circular buffer for modeling deterministic axonal
conduction delays in clock-driven spiking neural network simulation.

Biological Motivation:
- Action potentials propagate at fixed velocities (1-100 m/s)
- Distance determines delay (deterministic, not stochastic)
- Delays don't change dynamically (myelin thickness is stable)
- Axonal arbors create fanout (one spike → many targets)

Implementation:
- O(1) read/write operations (no heap/priority queue)
- Cache-friendly (contiguous tensor storage)
- GPU-compatible (pure tensor operations)
- Growable (can expand buffer size dynamically)

Author: Thalia Project
Date: December 21, 2025
"""

from __future__ import annotations

import torch


class CircularDelayBuffer:
    """Efficient circular buffer for axonal conduction delays.

    Models deterministic axonal delays using a ring buffer. Spikes are written
    to the current position and read from a position `delay` steps in the past.

    The buffer automatically handles wrap-around using modulo arithmetic, making
    it efficient and simple to use.

    Memory: O(max_delay × size) per buffer
    Read/Write: O(1) per operation

    Example:
        >>> # 5ms delay for 128 neurons
        >>> buffer = CircularDelayBuffer(max_delay=5, size=128, device="cpu")
        >>>
        >>> # Each timestep: write current spikes, read delayed spikes
        >>> for t in range(100):
        ...     current_spikes = get_spikes()  # [128] binary tensor
        ...     buffer.write(current_spikes)
        ...     delayed_spikes = buffer.read(delay=5)  # Spikes from t-5
        ...     buffer.advance()

    Args:
        max_delay: Maximum delay in timesteps (buffer size = max_delay + 1)
        size: Size of spike vector (number of neurons)
        device: Torch device ('cpu', 'cuda', etc.)
        dtype: Data type for buffer (default: torch.bool for spikes)
    """

    def __init__(
        self,
        max_delay: int,
        size: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.bool,
    ):
        if max_delay < 0:
            raise ValueError(f"max_delay must be >= 0, got {max_delay}")
        if size <= 0:
            raise ValueError(f"size must be > 0, got {size}")

        self.max_delay = max_delay
        self.size = size
        self.device = device
        self.dtype = dtype

        # Buffer: [max_delay + 1, size]
        # +1 because we need to store: current, t-1, t-2, ..., t-max_delay
        self.buffer = torch.zeros(
            (max_delay + 1, size),
            dtype=dtype,
            device=device,
        )

        # Current write position (0 to max_delay, wraps around)
        self.ptr = 0

    def write(self, spikes: torch.Tensor) -> None:
        """Write spikes to current buffer position.

        Args:
            spikes: Spike tensor [size] to write

        Raises:
            ValueError: If spikes.shape[0] != self.size
        """
        if spikes.shape[0] != self.size:
            raise ValueError(
                f"Spike vector size mismatch: expected {self.size}, "
                f"got {spikes.shape[0]}"
            )

        # Write to current position
        self.buffer[self.ptr] = spikes.to(dtype=self.dtype, device=self.device)

    def read(self, delay: int) -> torch.Tensor:
        """Read spikes from `delay` timesteps ago.

        Args:
            delay: Number of timesteps in the past (0 to max_delay)
                  delay=0 returns current spikes (just written)
                  delay=1 returns spikes from previous timestep
                  delay=max_delay returns oldest spikes

        Returns:
            Spike tensor [size] from `delay` timesteps ago

        Raises:
            ValueError: If delay > max_delay or delay < 0
        """
        if delay < 0 or delay > self.max_delay:
            raise ValueError(
                f"Delay {delay} out of range [0, {self.max_delay}]"
            )

        # Calculate read index (wrap around using modulo)
        read_idx = (self.ptr - delay) % (self.max_delay + 1)
        return self.buffer[read_idx]

    def advance(self) -> None:
        """Advance buffer to next timestep.

        Call this AFTER write() and read() for each timestep.
        Increments pointer and wraps around at buffer boundary.
        """
        self.ptr = (self.ptr + 1) % (self.max_delay + 1)

    def reset(self) -> None:
        """Reset buffer to all zeros and pointer to 0."""
        self.buffer.zero_()
        self.ptr = 0

    def grow(self, new_size: int) -> None:
        """Grow buffer size (number of neurons).

        Expands the spike vector dimension while preserving existing data.
        New neurons are initialized to zeros.

        Args:
            new_size: New size (must be >= current size)

        Raises:
            ValueError: If new_size < current size
        """
        if new_size < self.size:
            raise ValueError(
                f"Cannot shrink buffer: new_size {new_size} < "
                f"current size {self.size}"
            )

        if new_size == self.size:
            return  # No-op

        # Create expanded buffer
        new_buffer = torch.zeros(
            (self.max_delay + 1, new_size),
            dtype=self.dtype,
            device=self.device,
        )

        # Copy existing data
        new_buffer[:, :self.size] = self.buffer

        # Replace buffer
        self.buffer = new_buffer
        self.size = new_size

    def to(self, device: str) -> CircularDelayBuffer:
        """Move buffer to different device.

        Args:
            device: Target device ('cpu', 'cuda', etc.)

        Returns:
            Self (for chaining)
        """
        if device != self.device:
            self.buffer = self.buffer.to(device)
            self.device = device
        return self

    def state_dict(self) -> dict:
        """Get buffer state for checkpointing.

        Returns:
            Dict containing buffer tensor and pointer position
        """
        return {
            "buffer": self.buffer,
            "ptr": self.ptr,
            "max_delay": self.max_delay,
            "size": self.size,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore buffer state from checkpoint.

        Args:
            state: Dict from state_dict()

        Raises:
            ValueError: If state dimensions don't match
        """
        if state["max_delay"] != self.max_delay:
            raise ValueError(
                f"max_delay mismatch: expected {self.max_delay}, "
                f"got {state['max_delay']}"
            )
        if state["size"] != self.size:
            raise ValueError(
                f"size mismatch: expected {self.size}, "
                f"got {state['size']}"
            )

        self.buffer = state["buffer"].to(dtype=self.dtype, device=self.device)
        self.ptr = state["ptr"]

    def __repr__(self) -> str:
        return (
            f"CircularDelayBuffer(max_delay={self.max_delay}, "
            f"size={self.size}, device='{self.device}', "
            f"dtype={self.dtype}, ptr={self.ptr})"
        )
