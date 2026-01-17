"""
Spike trace utilities for STDP and eligibility-based learning.

This module provides reusable trace management for spike-timing dependent
plasticity and related learning rules. Traces are exponentially decaying
signals that accumulate spike history for computing synaptic updates.

Usage:
    # Standalone trace object (for new code)
    trace = SpikeTrace(size=100, tau=20.0)
    for spikes in spike_train:
        current_trace = trace.update(spikes, dt=1.0)

    # Functional API (for existing code with state dataclasses)
    trace = update_trace(trace, spikes, tau=20.0, dt=1.0)

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from thalia.config.base import BaseConfig

# =============================================================================
# Functional API (for integrating with existing state dataclasses)
# =============================================================================


def update_trace(
    trace: torch.Tensor,
    spikes: torch.Tensor,
    tau: float,
    dt: float = 1.0,
    decay_type: str = "exponential",
    amplitude: float = 1.0,
) -> torch.Tensor:
    """Update a spike trace with exponential decay.

    This is the functional version for use with existing state dataclasses.
    The trace tensor is updated in-place and also returned.

    Args:
        trace: Current trace tensor [size] or [batch, size]
        spikes: New spikes to accumulate [size] or [batch, size]
        tau: Time constant for decay (ms)
        dt: Time step (ms)
        decay_type: 'exponential' or 'linear'
        amplitude: Spike contribution amplitude

    Returns:
        Updated trace tensor (same object, modified in-place)

    Example:
        >>> # In a region's forward() method:
        >>> self.state.l4_trace = update_trace(
        ...     self.state.l4_trace, l4_spikes, tau=20.0, dt=1.0
        ... )
    """
    if decay_type == "exponential":
        decay = torch.exp(torch.tensor(-dt / tau, device=trace.device))
    else:
        decay = torch.tensor(max(0.0, 1.0 - dt / tau), device=trace.device)

    # Handle batch dimension mismatch
    # If spikes has larger batch than trace, we need to expand trace first
    if spikes.dim() == 2 and trace.dim() == 2:
        if spikes.shape[0] != trace.shape[0]:
            # Create new tensor with correct batch size
            new_trace = trace.expand(spikes.shape[0], -1).clone()
            new_trace.mul_(decay).add_(spikes.float() * amplitude)
            # Copy back to original (resize if needed)
            trace.resize_(spikes.shape[0], trace.shape[1])
            trace.copy_(new_trace)
            return trace

    # Normal in-place update
    trace.mul_(decay).add_(spikes.float() * amplitude)
    return trace


def compute_decay(tau: float, dt: float = 1.0, decay_type: str = "exponential") -> float:
    """Compute decay factor for a given tau and dt.

    Args:
        tau: Time constant (ms)
        dt: Time step (ms)
        decay_type: 'exponential' or 'linear'

    Returns:
        Decay factor (multiply trace by this each timestep)
    """
    if decay_type == "exponential":
        import math

        return math.exp(-dt / tau)
    else:
        return max(0.0, 1.0 - dt / tau)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class TraceConfig(BaseConfig):
    """Configuration for spike traces.

    Inherits device, dtype, seed from BaseConfig.

    Attributes:
        tau: Time constant for exponential decay (ms)
        decay_type: 'exponential' or 'linear'
        amplitude: Spike contribution amplitude (default 1.0)
    """

    tau: float = 20.0
    decay_type: str = "exponential"
    amplitude: float = 1.0


# =============================================================================
# Object-Oriented API (for new code)
# =============================================================================


class SpikeTrace(nn.Module):
    """Manages exponentially decaying spike traces for STDP.

    Spike traces are used in trace-based STDP to approximate the
    timing-dependent learning window. When a neuron spikes, its trace
    increases; between spikes, the trace decays exponentially.

    The trace value at time t represents the "eligibility" of that
    neuron for synaptic modification based on recent spiking history.

    Attributes:
        tau: Time constant for decay (ms)
        trace: Current trace values [size] or [batch, size]

    Example:
        >>> pre_trace = SpikeTrace(n_pre=100, tau=20.0)
        >>> post_trace = SpikeTrace(n_post=50, tau=20.0)
        >>>
        >>> # During forward pass
        >>> pre_t = pre_trace.update(pre_spikes, dt=1.0)
        >>> post_t = post_trace.update(post_spikes, dt=1.0)
        >>>
        >>> # STDP update
        >>> ltp = torch.outer(post_spikes, pre_t)  # pre before post
        >>> ltd = torch.outer(post_t, pre_spikes)  # post before pre
    """

    def __init__(
        self,
        size: int,
        tau: float = 20.0,
        decay_type: str = "exponential",
        amplitude: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize spike trace.

        Args:
            size: Number of neurons to track
            tau: Time constant for exponential decay (ms)
            decay_type: 'exponential' (exp(-dt/tau)) or 'linear' (1 - dt/tau)
            amplitude: Spike contribution to trace (default 1.0)
            device: Torch device for tensors
            dtype: Torch dtype for tensors
        """
        super().__init__()
        self.size = size
        self.tau = tau
        self.decay_type = decay_type
        self.amplitude = amplitude

        # Register trace as buffer (not a parameter, but saved with model)
        self.register_buffer("trace", torch.zeros(size, dtype=dtype, device=device))
        self.trace: torch.Tensor  # Type annotation for mypy

        # Cache decay factor for efficiency (recomputed if tau changes)
        self._cached_dt: Optional[float] = None
        self._cached_decay: Optional[torch.Tensor] = None

    def _get_decay(self, dt: float) -> torch.Tensor:
        """Get decay factor, using cache if possible."""
        if self._cached_dt != dt or self._cached_decay is None:
            if self.decay_type == "exponential":
                self._cached_decay = torch.exp(
                    torch.tensor(-dt / self.tau, device=self.trace.device)
                )
            else:  # linear
                self._cached_decay = torch.tensor(
                    max(0.0, 1.0 - dt / self.tau), device=self.trace.device
                )
            self._cached_dt = dt
        return self._cached_decay

    def update(
        self,
        spikes: torch.Tensor,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """Decay trace and add new spikes.

        Args:
            spikes: Binary spike tensor [size] or [batch, size]
            dt: Time step in ms

        Returns:
            Updated trace tensor (same shape as input)
        """
        decay = self._get_decay(dt)

        # Handle batched input
        if spikes.dim() == 2:
            # Expand trace to batch if needed
            if self.trace.dim() == 1:
                batch_size = spikes.shape[0]
                self.trace = self.trace.unsqueeze(0).expand(batch_size, -1).clone()

        # Decay and accumulate
        self.trace = self.trace * decay + spikes.float() * self.amplitude

        return self.trace

    def reset_state(self) -> None:
        """Reset trace to zeros.

        Always resets to batch_size=1 per THALIA's single-instance architecture.
        """
        self.trace.zero_()

    def get_trace(self) -> torch.Tensor:
        """Get current trace values."""
        return self.trace

    def __repr__(self) -> str:
        return f"SpikeTrace(size={self.size}, tau={self.tau}, " f"decay_type='{self.decay_type}')"


class PairedTraces(nn.Module):
    """Manages paired pre/post traces for STDP learning.

    Convenience class that manages both presynaptic and postsynaptic
    traces together, which is the common use case for STDP.

    Example:
        >>> traces = PairedTraces(n_pre=100, n_post=50, tau=20.0)
        >>>
        >>> # Update both traces
        >>> pre_t, post_t = traces.update(pre_spikes, post_spikes, dt=1.0)
        >>>
        >>> # Compute STDP
        >>> ltp = A_plus * torch.outer(post_spikes, pre_t)
        >>> ltd = A_minus * torch.outer(post_t, pre_spikes)
        >>> dw = ltp - ltd
    """

    def __init__(
        self,
        n_pre: int,
        n_post: int,
        tau_pre: float = 20.0,
        tau_post: float = 20.0,
        decay_type: str = "exponential",
        device: Optional[torch.device] = None,
    ):
        """Initialize paired traces.

        Args:
            n_pre: Number of presynaptic neurons
            n_post: Number of postsynaptic neurons
            tau_pre: Pre trace time constant (ms)
            tau_post: Post trace time constant (ms)
            decay_type: 'exponential' or 'linear'
            device: Torch device
        """
        super().__init__()
        self.pre_trace = SpikeTrace(size=n_pre, tau=tau_pre, decay_type=decay_type, device=device)
        self.post_trace = SpikeTrace(
            size=n_post, tau=tau_post, decay_type=decay_type, device=device
        )

    def update(
        self,
        pre_spikes: torch.Tensor,
        post_spikes: torch.Tensor,
        dt: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Update both traces with new spikes.

        Args:
            pre_spikes: Presynaptic spikes [n_pre] or [batch, n_pre]
            post_spikes: Postsynaptic spikes [n_post] or [batch, n_post]
            dt: Time step in ms

        Returns:
            Tuple of (pre_trace, post_trace)
        """
        pre_t = self.pre_trace.update(pre_spikes, dt)
        post_t = self.post_trace.update(post_spikes, dt)
        return pre_t, post_t

    def reset_state(self, batch_size: Optional[int] = None) -> None:
        """Reset both traces."""
        self.pre_trace.reset_state()  # SpikeTrace.reset_state() takes no args
        self.post_trace.reset_state()  # SpikeTrace.reset_state() takes no args

    def get_traces(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current trace values."""
        return self.pre_trace.get_trace(), self.post_trace.get_trace()


# =============================================================================
# STDP Helpers
# =============================================================================


def compute_stdp_update(
    pre_spikes: torch.Tensor,
    post_spikes: torch.Tensor,
    pre_trace: torch.Tensor,
    post_trace: torch.Tensor,
    a_plus: float = 0.01,
    a_minus: float = 0.012,
    ltd_factor: float = 1.0,
) -> torch.Tensor:
    """Compute STDP weight update from spikes and traces.

    Standard trace-based STDP:
        LTP: A+ × post_spike × pre_trace (pre before post → strengthen)
        LTD: A- × pre_spike × post_trace (post before pre → weaken)

    Args:
        pre_spikes: Presynaptic spikes [n_pre] or [batch, n_pre]
        post_spikes: Postsynaptic spikes [n_post] or [batch, n_post]
        pre_trace: Presynaptic trace [n_pre] or [batch, n_pre]
        post_trace: Postsynaptic trace [n_post] or [batch, n_post]
        a_plus: LTP amplitude
        a_minus: LTD amplitude
        ltd_factor: Multiplier for LTD (use <1.0 for asymmetric STDP)

    Returns:
        Weight change matrix [n_post, n_pre] (to add to weights)
    """
    # Ensure 1D for outer product
    pre_spikes = pre_spikes.squeeze()
    post_spikes = post_spikes.squeeze()
    pre_trace = pre_trace.squeeze()
    post_trace = post_trace.squeeze()

    # LTP: post fires, pre was recently active
    ltp = a_plus * torch.outer(post_spikes.float(), pre_trace)

    # LTD: pre fires, post was recently active
    ltd = a_minus * ltd_factor * torch.outer(post_trace, pre_spikes.float())

    return ltp - ltd


# =============================================================================
# Factory Functions
# =============================================================================


def create_trace(
    size: int,
    config: Optional[TraceConfig] = None,
    device: Optional[torch.device] = None,
) -> SpikeTrace:
    """Create a SpikeTrace from configuration.

    Args:
        size: Number of neurons
        config: TraceConfig or None for defaults
        device: Torch device

    Returns:
        Configured SpikeTrace instance
    """
    cfg = config or TraceConfig()
    return SpikeTrace(
        size=size,
        tau=cfg.tau,
        decay_type=cfg.decay_type,
        amplitude=cfg.amplitude,
        device=device,
    )
