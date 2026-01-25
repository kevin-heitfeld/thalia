"""
Spike trace utilities for STDP and eligibility-based learning.

This module provides reusable trace management for spike-timing dependent
plasticity and related learning rules. Traces are exponentially decaying
signals that accumulate spike history for computing synaptic updates.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

import torch


def update_trace(
    trace: torch.Tensor,
    spikes: torch.Tensor,
    tau: float,
    dt_ms: float = 1.0,
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
        dt_ms: Time step (ms)
        decay_type: 'exponential' or 'linear'
        amplitude: Spike contribution amplitude

    Returns:
        Updated trace tensor (same object, modified in-place)
    """
    if decay_type == "exponential":
        decay = torch.exp(torch.tensor(-dt_ms / tau, device=trace.device))
    else:
        decay = torch.tensor(max(0.0, 1.0 - dt_ms / tau), device=trace.device)

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
