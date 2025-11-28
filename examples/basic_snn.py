#!/usr/bin/env python3
"""
Example: Basic SNN Simulation

Demonstrates creating a simple spiking neural network, applying input,
and observing output spikes.
"""

import torch
from thalia.core.network import SNNNetwork
from thalia.encoding.poisson import poisson_encode


def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    batch_size = 16
    n_input = 100
    n_hidden = 50
    n_output = 10
    duration = 100  # timesteps
    dt = 1.0  # ms

    # Create network
    print(f"\nCreating network: {n_input} -> {n_hidden} -> {n_output}")
    network = SNNNetwork(
        layer_sizes=[n_input, n_hidden, n_output],
        recurrent=True,
        recurrent_connectivity=0.1
    )
    network.to(device)
    network.reset_state(batch_size=batch_size)

    # Generate Poisson input (50 Hz background rate)
    print(f"Generating Poisson input at 50 Hz for {duration} ms...")
    input_rates = torch.ones(batch_size, n_input, device=device) * 50  # Hz
    input_spikes = poisson_encode(input_rates, duration=duration, dt=dt)

    # Run simulation
    print(f"Running simulation for {duration} timesteps...")
    all_output_spikes = []
    all_hidden_spikes = []

    for t in range(duration):
        # Get input for this timestep
        x = input_spikes[t]

        # Forward pass
        output, layer_spikes = network(x)
        all_output_spikes.append(output)
        all_hidden_spikes.append(layer_spikes[0])  # First layer = hidden

    # Stack results
    output_spikes = torch.stack(all_output_spikes)  # (time, batch, neurons)
    hidden_spikes = torch.stack(all_hidden_spikes)

    # Analyze results
    print("\n" + "=" * 50)
    print("Results:")
    print("=" * 50)

    # Input statistics
    total_input = input_spikes.sum().item()
    input_rate = total_input / (duration * batch_size * n_input) * 1000 / dt
    print(f"\nInput layer:")
    print(f"  Total spikes: {total_input:.0f}")
    print(f"  Mean rate: {input_rate:.1f} Hz")

    # Hidden layer statistics
    total_hidden = hidden_spikes.sum().item()
    hidden_rate = total_hidden / (duration * batch_size * n_hidden) * 1000 / dt
    print(f"\nHidden layer ({n_hidden} neurons):")
    print(f"  Total spikes: {total_hidden:.0f}")
    print(f"  Mean rate: {hidden_rate:.1f} Hz")

    # Output statistics
    total_output = output_spikes.sum().item()
    output_rate = total_output / (duration * batch_size * n_output) * 1000 / dt
    print(f"\nOutput layer ({n_output} neurons):")
    print(f"  Total spikes: {total_output:.0f}")
    print(f"  Mean rate: {output_rate:.1f} Hz")

    # Per-neuron output activity
    print(f"\nPer-neuron output activity (averaged over batch):")
    neuron_rates = output_spikes.sum(dim=(0, 1)) / (duration * batch_size) * 1000 / dt
    for i, rate in enumerate(neuron_rates):
        bar = "█" * int(rate / 5)
        print(f"  Neuron {i}: {rate:5.1f} Hz {bar}")

    print("\nDone!")


if __name__ == "__main__":
    main()
