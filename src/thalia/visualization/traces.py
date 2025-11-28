"""Membrane potential trace visualization."""

from typing import Optional

import torch


def plot_membrane_traces(
    membrane: torch.Tensor,
    spikes: Optional[torch.Tensor] = None,
    dt: float = 1.0,
    neuron_ids: Optional[list[int]] = None,
    threshold: float = -50.0,
    title: str = "Membrane Potential",
    ax=None,
):
    """Plot membrane potential traces over time.
    
    Args:
        membrane: Membrane potential, shape (time, batch, neurons) or (time, neurons)
        spikes: Optional spike tensor to mark spike times
        dt: Timestep in ms
        neuron_ids: Subset of neurons to plot
        threshold: Spike threshold for reference line
        title: Plot title
        ax: Matplotlib axes
        
    Returns:
        Matplotlib axes object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for visualization")
    
    # Handle shapes
    if membrane.dim() == 3:
        membrane = membrane[:, 0, :]
    
    membrane = membrane.cpu().numpy()
    n_time, n_neurons = membrane.shape
    
    if neuron_ids is None:
        neuron_ids = list(range(min(5, n_neurons)))  # Default: first 5
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    times = torch.arange(n_time) * dt
    
    for i, nid in enumerate(neuron_ids):
        ax.plot(times, membrane[:, nid], label=f"Neuron {nid}")
    
    ax.axhline(y=threshold, color='r', linestyle='--', alpha=0.5, label="Threshold")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Membrane Potential (mV)")
    ax.set_title(title)
    ax.legend()
    
    return ax


def plot_activity_heatmap(
    activity: torch.Tensor,
    dt: float = 1.0,
    title: str = "Network Activity",
    ax=None,
):
    """Plot activity as a heatmap.
    
    Args:
        activity: Activity tensor, shape (time, neurons)
        dt: Timestep in ms
        title: Plot title
        ax: Matplotlib axes
        
    Returns:
        Matplotlib axes object
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for visualization")
    
    if activity.dim() == 3:
        activity = activity[:, 0, :]
    
    activity = activity.cpu().numpy()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(activity.T, aspect='auto', cmap='hot', origin='lower')
    ax.set_xlabel("Time (steps)")
    ax.set_ylabel("Neuron")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="Activity")
    
    return ax
