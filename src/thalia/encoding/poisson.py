"""Poisson spike generation."""

import torch


def poisson_encode(
    rates: torch.Tensor,
    duration: int,
    dt: float = 1.0,
) -> torch.Tensor:
    """Generate Poisson spike trains from firing rates.
    
    Args:
        rates: Firing rates in Hz, shape (batch, n_neurons)
        duration: Number of timesteps
        dt: Timestep in ms
        
    Returns:
        Spike trains, shape (duration, batch, n_neurons)
    """
    # Convert rate to expected spikes per timestep
    lambda_per_step = rates * dt / 1000.0
    
    # Poisson sampling (approximated as Bernoulli for small lambda)
    # For proper Poisson: torch.poisson(lambda_per_step)
    # But for SNNs, we typically want 0/1 spikes per step
    spikes = torch.rand(duration, *rates.shape, device=rates.device) < lambda_per_step
    
    return spikes.float()


def generate_spike_train(
    rate: float,
    duration: int,
    n_neurons: int = 1,
    dt: float = 1.0,
    device: str = "cpu",
) -> torch.Tensor:
    """Generate a single Poisson spike train.
    
    Args:
        rate: Firing rate in Hz
        duration: Number of timesteps
        n_neurons: Number of neurons
        dt: Timestep in ms
        device: Torch device
        
    Returns:
        Spike train, shape (duration, 1, n_neurons)
    """
    rates = torch.full((1, n_neurons), rate, device=device)
    return poisson_encode(rates, duration, dt)
