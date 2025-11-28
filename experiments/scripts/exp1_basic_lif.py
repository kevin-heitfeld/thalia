#!/usr/bin/env python3
"""Experiment 1: Basic LIF Network

Create 100 LIF neurons with random sparse connectivity,
inject current, observe spiking, and visualize spike raster.

This validates that our core SNN infrastructure works correctly.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from thalia.core import LIFNeuron, LIFConfig, SNNLayer


def run_experiment():
    """Run the basic LIF network experiment."""
    print("=" * 60)
    print("Experiment 1: Basic LIF Network")
    print("=" * 60)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    n_neurons = 100
    n_timesteps = 500
    dt = 1.0  # ms

    # Create LIF configuration with varying time constants
    config = LIFConfig(
        tau_mem=20.0,  # 20ms membrane time constant
        v_threshold=1.0,
        v_reset=0.0,
        v_rest=0.0,
        noise_std=0.1,  # Add some noise for variability
    )

    print(f"\nNetwork Configuration:")
    print(f"  Neurons: {n_neurons}")
    print(f"  Timesteps: {n_timesteps} ({n_timesteps * dt}ms)")
    print(f"  τ_mem: {config.tau_mem}ms")
    print(f"  Threshold: {config.v_threshold}")

    # Create recurrent layer with sparse connectivity
    layer = SNNLayer(
        n_neurons=n_neurons,
        neuron_config=config,
        recurrent=True,
        recurrent_connectivity=0.1,  # 10% connectivity
    ).to(device)

    # Count connections (handle case where recurrent_synapses exists)
    if layer.recurrent_synapses is not None:
        n_connections = (layer.recurrent_synapses.weight != 0).sum().item()
        print(f"  Recurrent connections: {n_connections} ({100*n_connections/(n_neurons**2):.1f}%)")

    # Storage for results
    spike_times = []  # List of (neuron_idx, time)
    membrane_history = []

    # Reset layer state
    layer.reset_state(batch_size=1)

    print(f"\nRunning simulation...")

    # Simulation loop
    for t in range(n_timesteps):
        # Create input current
        # Constant base current + time-varying component
        base_current = 0.15  # Lower base current for reasonable rates
        modulation = 0.1 * np.sin(2 * np.pi * t / 100)  # 100ms period
        
        # Random subset gets stronger input
        input_current = torch.ones(1, n_neurons, device=device) * (base_current + modulation)
        
        # Add some random input to specific neurons
        if t % 50 == 0:  # Every 50ms, stimulate random neurons
            stim_neurons = torch.randperm(n_neurons)[:10]
            input_current[0, stim_neurons] += 0.5        # Forward pass - returns (spikes, voltages)
        spikes, voltages = layer(external_current=input_current)

        # Record spikes
        spike_indices = spikes[0].nonzero(as_tuple=True)[0]
        for idx in spike_indices:
            spike_times.append((idx.item(), t))

        # Record membrane potentials (every 5th step to save memory)
        if t % 5 == 0:
            membrane_history.append(voltages.detach().clone().cpu().numpy())

    print(f"  Total spikes: {len(spike_times)}")
    print(f"  Average firing rate: {len(spike_times) / n_neurons / (n_timesteps * dt / 1000):.1f} Hz")

    # Analysis
    print("\n" + "=" * 60)
    print("Analysis")
    print("=" * 60)

    # Spike count per neuron
    spike_counts = np.zeros(n_neurons)
    for neuron_idx, _ in spike_times:
        spike_counts[neuron_idx] += 1

    print(f"\nFiring Statistics:")
    print(f"  Min spikes/neuron: {spike_counts.min():.0f}")
    print(f"  Max spikes/neuron: {spike_counts.max():.0f}")
    print(f"  Mean spikes/neuron: {spike_counts.mean():.1f}")
    print(f"  Std spikes/neuron: {spike_counts.std():.1f}")
    print(f"  Silent neurons: {(spike_counts == 0).sum()}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Experiment 1: Basic LIF Network", fontsize=14, fontweight='bold')

    # 1. Spike raster plot
    ax1 = axes[0, 0]
    if spike_times:
        neurons, times = zip(*spike_times)
        ax1.scatter(times, neurons, s=1, c='black', alpha=0.5)
    ax1.set_xlabel("Time (ms)")
    ax1.set_ylabel("Neuron Index")
    ax1.set_title("Spike Raster Plot")
    ax1.set_xlim(0, n_timesteps)
    ax1.set_ylim(0, n_neurons)

    # 2. Population firing rate over time
    ax2 = axes[0, 1]
    bin_size = 10  # 10ms bins
    time_bins = np.arange(0, n_timesteps + bin_size, bin_size)
    spike_times_array = np.array([t for _, t in spike_times])
    if len(spike_times_array) > 0:
        hist, _ = np.histogram(spike_times_array, bins=time_bins)
        rate = hist / (bin_size / 1000) / n_neurons  # Hz
        ax2.plot(time_bins[:-1] + bin_size/2, rate, 'b-', linewidth=1.5)
        ax2.fill_between(time_bins[:-1] + bin_size/2, rate, alpha=0.3)
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Population Rate (Hz)")
    ax2.set_title("Population Firing Rate")
    ax2.set_xlim(0, n_timesteps)

    # 3. Firing rate histogram
    ax3 = axes[1, 0]
    firing_rates = spike_counts / (n_timesteps * dt / 1000)  # Convert to Hz
    ax3.hist(firing_rates, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax3.axvline(firing_rates.mean(), color='red', linestyle='--', label=f'Mean: {firing_rates.mean():.1f} Hz')
    ax3.set_xlabel("Firing Rate (Hz)")
    ax3.set_ylabel("Count")
    ax3.set_title("Firing Rate Distribution")
    ax3.legend()

    # 4. Sample membrane potential traces
    ax4 = axes[1, 1]
    membrane_history = np.array(membrane_history)  # (n_samples, 1, n_neurons)
    sample_neurons = [0, 25, 50, 75, 99]
    time_axis = np.arange(0, n_timesteps, 5)
    for i, neuron_idx in enumerate(sample_neurons):
        trace = membrane_history[:, 0, neuron_idx]
        ax4.plot(time_axis, trace + i * 1.5, label=f'Neuron {neuron_idx}')
    ax4.set_xlabel("Time (ms)")
    ax4.set_ylabel("Membrane Potential (offset)")
    ax4.set_title("Sample Membrane Potential Traces")
    ax4.legend(loc='upper right', fontsize=8)

    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "exp1_basic_lif.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")

    plt.show()

    # Success criteria
    print("\n" + "=" * 60)
    print("Success Criteria Check")
    print("=" * 60)

    criteria = [
        ("LIF neurons fire correctly", len(spike_times) > 100),
        ("Network simulates without errors", True),  # We got here!
        ("Visualization works", output_path.exists()),
        ("Neurons show activity", firing_rates.mean() > 0),
        ("Not all neurons silent", (spike_counts > 0).sum() > n_neurons * 0.5),
    ]

    all_passed = True
    for name, passed in criteria:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        all_passed = all_passed and passed

    print("\n" + ("🎉 All criteria passed!" if all_passed else "⚠️ Some criteria failed"))

    return all_passed


if __name__ == "__main__":
    run_experiment()
