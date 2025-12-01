"""Weight matrix visualization functions.

This module provides functions for visualizing synaptic weight matrices,
their evolution over training, and learned structure.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import torch


def plot_weight_matrix(
    weights: Union[torch.Tensor, np.ndarray],
    ax: Optional[Axes] = None,
    title: str = "Weight Matrix",
    xlabel: str = "Input Neuron",
    ylabel: str = "Output Neuron",
    cmap: str = "RdBu_r",
    colorbar: bool = True,
    highlight_diagonal: bool = False,
    diagonal_offset: int = 0,
) -> Axes:
    """Plot a weight matrix as a heatmap.

    Args:
        weights: Weight matrix (n_output, n_input) as tensor or array
        ax: Matplotlib axes to plot on. Creates new if None.
        title: Plot title
        xlabel: Label for x-axis
        ylabel: Label for y-axis
        cmap: Colormap name
        colorbar: Whether to add a colorbar
        highlight_diagonal: Whether to mark expected diagonal positions
        diagonal_offset: Offset for diagonal highlighting (0 = main diagonal)

    Returns:
        The axes object
    """
    if ax is None:
        _, ax = plt.subplots()

    # Convert to numpy if needed
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()

    im = ax.imshow(weights, aspect='auto', cmap=cmap)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if colorbar:
        plt.colorbar(im, ax=ax)

    if highlight_diagonal:
        n_output = weights.shape[0]
        for i in range(n_output):
            target = i + diagonal_offset
            if 0 <= target < weights.shape[1]:
                ax.plot(target, i, 'r+', markersize=10, markeredgewidth=2)

    return ax


def plot_recurrent_weights(
    weights: Union[torch.Tensor, np.ndarray],
    ax: Optional[Axes] = None,
    title: str = "Recurrent Weights",
    highlight_chain: bool = True,
    chain_offset: int = 1,
) -> Axes:
    """Plot recurrent weight matrix with chain structure highlighting.

    Args:
        weights: Recurrent weight matrix (n_output, n_output)
        ax: Matplotlib axes to plot on
        title: Plot title
        highlight_chain: Whether to mark expected i→i+offset connections
        chain_offset: Expected chain offset (1 for i→i+1)

    Returns:
        The axes object
    """
    if ax is None:
        _, ax = plt.subplots()

    if isinstance(weights, torch.Tensor):
        weights = weights.cpu().numpy()

    im = ax.imshow(weights, aspect='auto', cmap='Blues')
    ax.set_xlabel("To Neuron")
    ax.set_ylabel("From Neuron")
    ax.set_title(title)
    plt.colorbar(im, ax=ax)

    if highlight_chain:
        n = weights.shape[0]
        for i in range(n - chain_offset):
            ax.plot(i + chain_offset, i, 'r+', markersize=10, markeredgewidth=2)

    return ax


def plot_learning_curve(
    spike_counts: List[float],
    ax: Optional[Axes] = None,
    title: str = "Learning Curve",
    window: int = 10,
    show_moving_avg: bool = True,
) -> Axes:
    """Plot output spike counts over training with moving average.

    Args:
        spike_counts: List of spike counts per cycle
        ax: Matplotlib axes to plot on
        title: Plot title
        window: Moving average window size
        show_moving_avg: Whether to show moving average line

    Returns:
        The axes object
    """
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(spike_counts, 'b-', alpha=0.7)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Output Spikes")
    ax.set_title(title)

    if show_moving_avg and len(spike_counts) >= window:
        ma = np.convolve(spike_counts, np.ones(window)/window, mode='valid')
        ax.plot(np.arange(window-1, len(spike_counts)), ma, 'r-',
                linewidth=2, label='Moving Avg')
        ax.legend()

    return ax


def plot_weight_evolution(
    weight_history: List[np.ndarray],
    ax: Optional[Axes] = None,
    title: str = "Weight Evolution",
    sample_interval: int = 10,
) -> Axes:
    """Plot weight mean and std evolution over training.

    Args:
        weight_history: List of weight snapshots (numpy arrays)
        ax: Matplotlib axes to plot on
        title: Plot title
        sample_interval: Interval between weight samples (for x-axis)

    Returns:
        The axes object
    """
    if ax is None:
        _, ax = plt.subplots()

    weight_means = [w.mean() for w in weight_history]
    weight_stds = [w.std() for w in weight_history]
    x = np.arange(len(weight_means)) * sample_interval

    ax.plot(x, weight_means, 'b-', linewidth=2)
    ax.fill_between(
        x,
        np.array(weight_means) - np.array(weight_stds),
        np.array(weight_means) + np.array(weight_stds),
        alpha=0.3
    )
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Weight")
    ax.set_title(title)

    return ax


def plot_learned_mapping(
    weights: Union[torch.Tensor, np.ndarray],
    expected_mapping: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None,
    title: str = "Learned vs Expected Mapping",
    n_output: Optional[int] = None,
) -> Axes:
    """Plot comparison of learned vs expected input→output mapping.

    Args:
        weights: Weight matrix (n_output, n_input)
        expected_mapping: Array of expected strongest input for each output
        ax: Matplotlib axes to plot on
        title: Plot title
        n_output: Number of output neurons (inferred if None)

    Returns:
        The axes object
    """
    if ax is None:
        _, ax = plt.subplots()

    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()

    if n_output is None:
        n_output = weights.shape[0]

    # Find strongest input for each output
    max_inputs = np.argmax(weights, axis=1)

    # Bar positions
    x = np.arange(n_output)

    ax.bar(x - 0.2, max_inputs, width=0.4, label='Learned', color='blue', alpha=0.7)

    if expected_mapping is not None:
        ax.bar(x + 0.2, expected_mapping, width=0.4,
               label='Expected', color='red', alpha=0.5)
        ax.legend()

    ax.set_xlabel("Output Neuron")
    ax.set_ylabel("Strongest Input")
    ax.set_title(title)
    ax.set_xticks(range(n_output))

    return ax


def create_training_summary_figure(
    initial_weights: Union[torch.Tensor, np.ndarray],
    final_weights: Union[torch.Tensor, np.ndarray],
    recurrent_weights: Union[torch.Tensor, np.ndarray],
    spike_counts: List[float],
    weight_history: List[np.ndarray],
    title: str = "Training Summary",
    expected_mapping: Optional[np.ndarray] = None,
) -> Tuple[Figure, np.ndarray]:
    """Create a 2x3 summary figure of training results.

    Args:
        initial_weights: Initial feedforward weights
        final_weights: Final feedforward weights
        recurrent_weights: Learned recurrent weights
        spike_counts: Spike counts per cycle
        weight_history: Weight snapshots over training
        title: Main figure title
        expected_mapping: Expected input mapping for comparison

    Returns:
        Tuple of (figure, axes array)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # 1. Initial weights
    plot_weight_matrix(initial_weights, ax=axes[0, 0], title="Initial Weights (Random)")

    # 2. Final weights
    plot_weight_matrix(final_weights, ax=axes[0, 1], title="Final Feedforward Weights")

    # 3. Recurrent weights
    plot_recurrent_weights(recurrent_weights, ax=axes[0, 2], title="Learned Recurrent Weights")

    # 4. Learning curve
    plot_learning_curve(spike_counts, ax=axes[1, 0])

    # 5. Weight evolution
    plot_weight_evolution(weight_history, ax=axes[1, 1])

    # 6. Learned mapping
    plot_learned_mapping(final_weights, expected_mapping=expected_mapping, ax=axes[1, 2])

    plt.tight_layout()

    return fig, axes
