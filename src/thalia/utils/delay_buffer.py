"""
Circular Delay Buffer - Efficient axonal delay implementation.

This module provides a circular buffer for modeling deterministic axonal
conduction delays in clock-driven spiking neural network simulation.

Biological Motivation:
- Action potentials propagate at fixed velocities (1-100 m/s)
- Distance determines delay (deterministic, not stochastic)
- Delays don't change dynamically (myelin thickness is stable)
- Axonal arbors create fanout (one spike → many targets)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CircularDelayBuffer(nn.Module):
    """Efficient circular buffer for axonal conduction delays.

    Models deterministic axonal delays using a ring buffer. Spikes are written
    to the current position and read from a position `delay` steps in the past.

    The buffer automatically handles wrap-around using modulo arithmetic, making
    it efficient and simple to use.

    Memory: O(max_delay × size) per buffer
    Read/Write: O(1) per operation

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

        super().__init__()

        self.max_delay = max_delay
        self.size = size
        self.dtype = dtype

        # Buffer: [max_delay + 1, size]
        # +1 because we need to store: current, t-1, t-2, ..., t-max_delay
        self.register_buffer(
            "buffer",
            torch.zeros((max_delay + 1, size), dtype=dtype, device=device),
        )

        # Current write position (0 to max_delay, wraps around)
        self.ptr = 0

    @property
    def device(self) -> torch.device:  # type: ignore[override]
        """Device where the buffer tensor resides."""
        return self.buffer.device

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
            raise ValueError(f"Delay {delay} out of range [0, {self.max_delay}]")

        # Calculate read index (wrap around using modulo)
        read_idx = (self.ptr - delay) % (self.max_delay + 1)
        return self.buffer[read_idx]

    def write(self, spikes: torch.Tensor) -> None:
        """Write spikes to current buffer position.

        Args:
            spikes: Spike tensor [size] to write

        Raises:
            ValueError: If spikes.shape[0] != self.size
        """
        if spikes.shape[0] != self.size:
            raise ValueError(
                f"Spike vector size mismatch: expected {self.size}, " f"got {spikes.shape[0]}"
            )

        # Write to current position
        self.buffer[self.ptr] = spikes.to(dtype=self.dtype, device=self.buffer.device)

    def advance(self) -> None:
        """Advance buffer to next timestep.

        Call this AFTER write() and read() for each timestep.
        Increments pointer and wraps around at buffer boundary.
        """
        self.ptr = (self.ptr + 1) % (self.max_delay + 1)

    def write_and_advance(self, spikes: torch.Tensor) -> None:
        """Convenience method to write spikes and advance in one step."""
        self.write(spikes)
        self.advance()

    def resize_for_new_dt(self, new_dt_ms: float, delay_ms: float, old_dt_ms: float) -> None:
        """Resize buffer when simulation timestep changes.

        When dt changes, the number of steps required to implement a fixed
        delay in milliseconds also changes:
            delay_steps = delay_ms / dt_ms

        This method resizes the buffer and interpolates existing spike history
        to preserve temporal information.

        Args:
            new_dt_ms: New simulation timestep in milliseconds
            delay_ms: Delay duration in milliseconds (fixed)
            old_dt_ms: Previous simulation timestep in milliseconds

        Raises:
            ValueError: If new_dt_ms or delay_ms are <= 0
        """
        if new_dt_ms <= 0:
            raise ValueError(f"new_dt_ms must be > 0, got {new_dt_ms}")
        if delay_ms < 0:
            raise ValueError(f"delay_ms must be >= 0, got {delay_ms}")
        if old_dt_ms <= 0:
            raise ValueError(f"old_dt_ms must be > 0, got {old_dt_ms}")

        # Calculate new delay in steps
        new_delay_steps = int(delay_ms / new_dt_ms)

        if new_delay_steps == self.max_delay:
            return  # No resize needed

        # Extract current buffer history in chronological order
        # Start from oldest (ptr - max_delay) to newest (ptr)
        history = []
        for i in range(self.max_delay + 1):
            idx = (self.ptr - self.max_delay + i) % (self.max_delay + 1)
            history.append(self.buffer[idx])
        history_tensor = torch.stack(history, dim=0)  # [old_steps+1, size]

        # Interpolate to new length (if needed)
        if new_delay_steps != self.max_delay:
            # Convert to float for interpolation
            history_float = history_tensor.float().unsqueeze(0)  # [1, old_steps+1, size]

            # Interpolate along time dimension
            new_length = new_delay_steps + 1
            if new_length != history_float.shape[1]:
                # Permute to [1, size, old_steps+1] for interpolation
                history_float = history_float.permute(0, 2, 1)
                # Interpolate
                interpolated = torch.nn.functional.interpolate(
                    history_float,
                    size=new_length,
                    mode="linear",
                    align_corners=True if new_length > 1 else None,
                )
                # Permute back to [1, new_steps+1, size]
                interpolated = interpolated.permute(0, 2, 1).squeeze(0)
            else:
                interpolated = history_float.squeeze(0)

            # Convert back to original dtype (threshold at 0.5 for binary spikes)
            if self.dtype == torch.bool:
                new_history = (interpolated > 0.5).to(dtype=self.dtype)
            else:
                new_history = interpolated.to(dtype=self.dtype)
        else:
            new_history = history_tensor

        # Create new buffer with new size
        self.register_buffer(
            "buffer",
            torch.zeros((new_delay_steps + 1, self.size), dtype=self.dtype, device=self.buffer.device),
        )
        self.max_delay = new_delay_steps

        # Copy interpolated history into new buffer
        self.buffer[:] = new_history

        # Reset pointer to end (most recent is at buffer[-1])
        self.ptr = new_delay_steps


class HeterogeneousDelayBuffer(nn.Module):
    """Circular buffer with per-neuron heterogeneous delays.

    Unlike CircularDelayBuffer (uniform delay for all neurons), this buffer
    allows each neuron to have its own delay value, modeling biological
    heterogeneity in axonal conduction velocities due to:
    - Variable myelination (0.5-120 m/s conduction velocity)
    - Different fiber diameters (thin=slow, thick=fast)
    - Path length differences within a tract (branching axons)

    This heterogeneity is critical for breaking pathological synchronization
    that can occur with uniform delays.

    Memory: O(max_delay × size) per buffer (same as CircularDelayBuffer)
    Read: O(size) per operation (must gather from different buffer positions)
    Write: O(1) per operation

    Args:
        delays: Per-neuron delays in timesteps [size] (integers >= 0)
        size: Number of neurons
        device: Torch device ('cpu', 'cuda', etc.)
        dtype: Data type for buffer (default: torch.bool for spikes)
    """

    def __init__(
        self,
        delays: torch.Tensor,
        size: int,
        device: str = "cpu",
        dtype: torch.dtype = torch.bool,
    ):
        if delays.shape[0] != size:
            raise ValueError(f"delays.shape[0] ({delays.shape[0]}) must equal size ({size})")

        super().__init__()

        self.size = size
        self.dtype = dtype

        # Store per-neuron delays [size]
        self.register_buffer("delays", delays.long().to(device))

        # Maximum delay determines buffer depth
        self.max_delay = int(self.delays.max().item())

        if self.max_delay < 0:
            raise ValueError(f"All delays must be >= 0, got min={self.delays.min().item()}")

        # Buffer: [max_delay + 1, size]
        self.register_buffer(
            "buffer",
            torch.zeros((self.max_delay + 1, size), dtype=dtype, device=device),
        )

        # Current write position
        self.ptr = 0

    @property
    def device(self) -> torch.device:  # type: ignore[override]
        """Device where the buffer tensor resides."""
        return self.buffer.device

    def read_heterogeneous(self) -> torch.Tensor:
        """Read spikes with per-neuron delays (VECTORIZED).

        Returns:
            Spike tensor [size] where each element comes from its specific delay
        """
        # Calculate read indices for each neuron: (ptr - delay[i]) % buffer_size
        read_indices = (self.ptr - self.delays) % (self.max_delay + 1)

        # VECTORIZED: Use advanced indexing to gather all spikes at once
        # This replaces the Python loop with a single torch operation
        # buffer[read_indices, arange(size)] efficiently gathers per-neuron delayed spikes
        neuron_indices = torch.arange(self.size, device=self.buffer.device)
        output = self.buffer[read_indices, neuron_indices]

        return output

    def write(self, spikes: torch.Tensor) -> None:
        """Write spikes to current buffer position.

        Args:
            spikes: Spike tensor [size] to write
        """
        if spikes.shape[0] != self.size:
            raise ValueError(
                f"Spike vector size mismatch: expected {self.size}, got {spikes.shape[0]}"
            )

        self.buffer[self.ptr] = spikes.to(dtype=self.dtype, device=self.buffer.device)

    def advance(self) -> None:
        """Advance buffer to next timestep."""
        self.ptr = (self.ptr + 1) % (self.max_delay + 1)

    def write_and_advance(self, spikes: torch.Tensor) -> None:
        """Convenience method to write spikes and advance in one step."""
        self.write(spikes)
        self.advance()

    def resize_for_new_dt(self, new_dt_ms: float, delay_ms: float, old_dt_ms: float) -> None:
        """Resize buffer when simulation timestep changes.

        For heterogeneous delays, we rescale the per-neuron delay distribution
        to match the new timestep while preserving the relative heterogeneity.

        Args:
            new_dt_ms: New simulation timestep in milliseconds
            delay_ms: Mean delay duration in milliseconds (fixed)
            old_dt_ms: Previous simulation timestep in milliseconds
        """
        if new_dt_ms <= 0:
            raise ValueError(f"new_dt_ms must be > 0, got {new_dt_ms}")
        if delay_ms < 0:
            raise ValueError(f"delay_ms must be >= 0, got {delay_ms}")
        if old_dt_ms <= 0:
            raise ValueError(f"old_dt_ms must be > 0, got {old_dt_ms}")

        # Calculate new delays in steps, preserving relative distribution
        # delays_ms = delays_steps * old_dt_ms
        # new_delays_steps = delays_ms / new_dt_ms
        delays_ms = self.delays.float() * old_dt_ms
        new_delays_steps = (delays_ms / new_dt_ms).long()
        new_delays_steps = torch.clamp(new_delays_steps, min=0)

        # Update delay array
        self.register_buffer("delays", new_delays_steps.to(self.buffer.device))

        # Update max_delay
        new_max_delay = int(self.delays.max().item())

        if new_max_delay == self.max_delay:
            return  # No resize needed

        # Extract current buffer history in chronological order
        history = []
        for i in range(self.max_delay + 1):
            idx = (self.ptr - self.max_delay + i) % (self.max_delay + 1)
            history.append(self.buffer[idx])
        history_tensor = torch.stack(history, dim=0)  # [old_steps+1, size]

        # Interpolate to new length if needed
        if new_max_delay != self.max_delay:
            history_float = history_tensor.float().unsqueeze(0)  # [1, old_steps+1, size]
            new_length = new_max_delay + 1

            if new_length != history_float.shape[1]:
                # Permute to [1, size, old_steps+1] for interpolation
                history_float = history_float.permute(0, 2, 1)
                # Interpolate
                interpolated = torch.nn.functional.interpolate(
                    history_float,
                    size=new_length,
                    mode="linear",
                    align_corners=True if new_length > 1 else None,
                )
                # Permute back to [1, new_steps+1, size]
                interpolated = interpolated.permute(0, 2, 1).squeeze(0)
            else:
                interpolated = history_float.squeeze(0)

            # Convert back to original dtype
            if self.dtype == torch.bool:
                new_history = (interpolated > 0.5).to(dtype=self.dtype)
            else:
                new_history = interpolated.to(dtype=self.dtype)
        else:
            new_history = history_tensor

        # Create new buffer with new size
        self.register_buffer(
            "buffer",
            torch.zeros((new_max_delay + 1, self.size), dtype=self.dtype, device=self.buffer.device),
        )
        self.max_delay = new_max_delay

        # Copy interpolated history into new buffer
        self.buffer[:] = new_history

        # Reset pointer to end
        self.ptr = new_max_delay
