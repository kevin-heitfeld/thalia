"""
Temporal input pattern generation for spiking neural networks.

This module provides functions to create various temporal spike patterns
for training and testing SNNs. All patterns are designed with biological
plausibility in mind.
"""

import torch
from typing import Optional, Union


def create_temporal_pattern(
    n_neurons: int, 
    duration_ms: float, 
    pattern_type: str = "sequential",
    start_neuron: int = 0, 
    gap_duration_ms: float = 50.0,
    phase_duration_ms: float = 8.0,
    spike_duration_ms: float = 5.0,
    dt: float = 1.0,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor:
    """Create a temporal spike pattern with sub-millisecond resolution.

    All time parameters are in MILLISECONDS. The dt parameter controls temporal resolution.

    Args:
        n_neurons: Number of neurons in the pattern
        duration_ms: Total duration in milliseconds
        pattern_type: Type of pattern to generate:
            - "sequential": Each neuron fires in sequence
            - "circular": Continuous cycling through neurons
            - "gapped": Sequential with silent gaps between repetitions
            - "burst": All neurons fire simultaneously
            - "random": Random sparse activity
        start_neuron: Which neuron fires first (for sequential patterns)
        gap_duration_ms: Duration of silent gap in milliseconds (for "gapped" pattern)
        phase_duration_ms: Time allocated to each neuron's phase (default: 8ms)
        spike_duration_ms: How long each neuron fires within its phase (default: 5ms)
        dt: Simulation timestep in milliseconds (default: 1.0)
        device: Torch device for the output tensor

    Returns:
        Tensor of shape (n_timesteps, n_neurons) with binary spike indicators
        
    Example:
        >>> # Create a 100ms gapped pattern with 20 neurons at 0.1ms resolution
        >>> pattern = create_temporal_pattern(
        ...     n_neurons=20, duration_ms=100, pattern_type="gapped",
        ...     gap_duration_ms=50, dt=0.1
        ... )
        >>> print(pattern.shape)  # (1000, 20)
    """
    # Convert ms to timesteps
    n_timesteps = int(duration_ms / dt)
    gap_timesteps = int(gap_duration_ms / dt)
    phase_timesteps = int(phase_duration_ms / dt)
    spike_timesteps = int(spike_duration_ms / dt)

    spikes = torch.zeros(n_timesteps, n_neurons)

    if pattern_type == "sequential":
        # Sequential activation - each neuron fires multiple times in its time window
        window_size = n_timesteps // n_neurons
        for i in range(n_neurons):
            neuron_idx = (start_neuron + i) % n_neurons
            start_time = i * window_size
            # Each neuron fires 3 times in its window
            for offset in [0, window_size // 3, 2 * window_size // 3]:
                spike_time = start_time + offset
                if spike_time < n_timesteps:
                    spikes[spike_time, neuron_idx] = 1.0

    elif pattern_type == "circular":
        # Circular/continuous pattern: 0 -> 1 -> 2 -> ... -> n-1 -> 0 -> 1 -> ...
        cycle_timesteps = n_neurons * phase_timesteps  # One full cycle
        for t in range(n_timesteps):
            cycle_position = t % cycle_timesteps
            neuron_idx = cycle_position // phase_timesteps
            within_window = cycle_position % phase_timesteps
            if within_window < spike_timesteps and neuron_idx < n_neurons:
                spikes[t, neuron_idx] = 1.0

    elif pattern_type == "gapped":
        # Gapped pattern with silent gap between sequence presentations
        sequence_timesteps = n_neurons * phase_timesteps
        cycle_with_gap = sequence_timesteps + gap_timesteps

        for t in range(n_timesteps):
            cycle_position = t % cycle_with_gap
            if cycle_position < sequence_timesteps:
                neuron_idx = cycle_position // phase_timesteps
                within_window = cycle_position % phase_timesteps
                if within_window < spike_timesteps and neuron_idx < n_neurons:
                    spikes[t, neuron_idx] = 1.0

    elif pattern_type == "burst":
        # Synchronous burst - all neurons fire together
        burst_timesteps = int(5.0 / dt)
        spikes[0:burst_timesteps, :] = (torch.rand(burst_timesteps, n_neurons) > 0.5).float()

    elif pattern_type == "random":
        # Random sparse activity
        spikes = (torch.rand(n_timesteps, n_neurons) > 0.95).float()

    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")

    if device is not None:
        spikes = spikes.to(device)
        
    return spikes


def create_poisson_pattern(
    n_neurons: int,
    duration_ms: float,
    firing_rate_hz: float = 10.0,
    dt: float = 1.0,
    device: Optional[Union[torch.device, str]] = None,
) -> torch.Tensor:
    """Create a Poisson spike pattern with specified firing rate.
    
    Args:
        n_neurons: Number of neurons
        duration_ms: Duration in milliseconds
        firing_rate_hz: Average firing rate in Hz
        dt: Timestep in ms
        device: Torch device
        
    Returns:
        Tensor of shape (n_timesteps, n_neurons) with Poisson-distributed spikes
    """
    n_timesteps = int(duration_ms / dt)
    
    # Convert firing rate to probability per timestep
    # rate_hz = spikes_per_second, dt in ms
    prob_per_timestep = firing_rate_hz * dt / 1000.0
    
    spikes = (torch.rand(n_timesteps, n_neurons) < prob_per_timestep).float()
    
    if device is not None:
        spikes = spikes.to(device)
        
    return spikes
