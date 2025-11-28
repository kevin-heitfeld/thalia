"""Rate coding: convert values to spike rates."""

import torch


def rate_encode(
    values: torch.Tensor,
    duration: int,
    max_rate: float = 100.0,  # Hz
    dt: float = 1.0,  # ms
) -> torch.Tensor:
    """Encode values as spike rates over time.
    
    Higher values = more frequent spikes (up to max_rate).
    
    Args:
        values: Input values in [0, 1], shape (batch, n_inputs)
        duration: Number of timesteps
        max_rate: Maximum firing rate in Hz
        dt: Timestep in ms
        
    Returns:
        Spike trains, shape (duration, batch, n_inputs)
    """
    # Convert rate to probability per timestep
    # P(spike) = rate * dt / 1000
    spike_prob = values * max_rate * dt / 1000.0
    spike_prob = torch.clamp(spike_prob, 0, 1)
    
    # Generate spikes
    spikes = torch.rand(duration, *values.shape, device=values.device) < spike_prob
    
    return spikes.float()
