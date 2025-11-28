"""Raster plot visualization for spike trains."""

from typing import Optional

import torch


def plot_raster(
    spikes: torch.Tensor,
    dt: float = 1.0,
    neuron_ids: Optional[list[int]] = None,
    title: str = "Spike Raster",
    ax=None,
):
    """Create a raster plot of spike activity.
    
    Args:
        spikes: Spike tensor, shape (time, batch, neurons) or (time, neurons)
        dt: Timestep in ms for time axis
        neuron_ids: Subset of neuron indices to plot
        title: Plot title
        ax: Matplotlib axes (creates new if None)
        
    Returns:
        Matplotlib axes object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for visualization. Install with: pip install matplotlib")
    
    # Handle different input shapes
    if spikes.dim() == 3:
        # (time, batch, neurons) -> take first batch
        spikes = spikes[:, 0, :]
    
    spikes = spikes.cpu().numpy()
    n_time, n_neurons = spikes.shape
    
    if neuron_ids is not None:
        spikes = spikes[:, neuron_ids]
        n_neurons = len(neuron_ids)
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Find spike times and neuron indices
    times, neurons = spikes.nonzero()
    times = times * dt  # Convert to ms
    
    ax.scatter(times, neurons, s=1, c='black', marker='|')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Neuron")
    ax.set_title(title)
    ax.set_xlim(0, n_time * dt)
    ax.set_ylim(-0.5, n_neurons - 0.5)
    
    return ax
