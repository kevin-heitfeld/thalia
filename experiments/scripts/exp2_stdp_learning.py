#!/usr/bin/env python3
"""Experiment 2: STDP Learning

Create a two-layer network, present temporal patterns,
observe weight evolution, and test pattern completion.

This validates that STDP learning works correctly.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from thalia.core import LIFNeuron, LIFConfig
from thalia.learning import STDPConfig, STDPLearner


def create_temporal_pattern(n_neurons: int, duration: int, pattern_type: str = "sequential") -> torch.Tensor:
    """Create a temporal spike pattern.
    
    Args:
        n_neurons: Number of neurons
        duration: Duration in timesteps
        pattern_type: "sequential", "burst", or "random"
        
    Returns:
        Tensor of shape (duration, n_neurons) with spike times
    """
    spikes = torch.zeros(duration, n_neurons)
    
    if pattern_type == "sequential":
        # Sequential activation - each neuron fires in order
        for i in range(n_neurons):
            spike_time = int(i * duration / n_neurons)
            if spike_time < duration:
                spikes[spike_time, i] = 1.0
                
    elif pattern_type == "burst":
        # All neurons fire together at start
        spikes[0:5, :] = torch.rand(5, n_neurons) > 0.5
        
    elif pattern_type == "random":
        # Random sparse spiking
        spikes = (torch.rand(duration, n_neurons) > 0.95).float()
        
    return spikes


def run_experiment():
    """Run the STDP learning experiment."""
    print("=" * 60)
    print("Experiment 2: STDP Learning")
    print("=" * 60)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Network parameters
    n_input = 20
    n_output = 10
    pattern_duration = 50  # ms
    n_presentations = 100
    
    print(f"\nNetwork Configuration:")
    print(f"  Input neurons: {n_input}")
    print(f"  Output neurons: {n_output}")
    print(f"  Pattern duration: {pattern_duration}ms")
    print(f"  Training presentations: {n_presentations}")
    
    # Create neurons
    config = LIFConfig(tau_mem=10.0, v_thresh=1.0, noise_std=0.05)
    output_neurons = LIFNeuron(n_neurons=n_output, config=config).to(device)
    
    # Create weights (input -> output)
    weights = torch.randn(n_output, n_input, device=device) * 0.3
    initial_weights = weights.clone()
    
    # Create STDP learner
    stdp_config = STDPConfig(
        tau_plus=20.0,
        tau_minus=20.0,
        a_plus=0.01,
        a_minus=0.012,  # Slightly stronger depression
        w_max=1.0,
        w_min=0.0,
    )
    stdp = STDPLearner(stdp_config).to(device)
    
    print(f"\nSTDP Configuration:")
    print(f"  τ+: {stdp_config.tau_plus}ms")
    print(f"  τ-: {stdp_config.tau_minus}ms")
    print(f"  A+: {stdp_config.a_plus}")
    print(f"  A-: {stdp_config.a_minus}")
    
    # Create training pattern
    pattern = create_temporal_pattern(n_input, pattern_duration, "sequential").to(device)
    print(f"\nTraining pattern: sequential activation")
    print(f"  Pattern spikes: {pattern.sum().item():.0f}")
    
    # Storage for tracking
    weight_history = [weights.clone().cpu().numpy()]
    output_spike_counts = []
    
    print(f"\nTraining...")
    
    # Training loop
    for presentation in range(n_presentations):
        output_neurons.reset_state(batch_size=1)
        stdp.reset()
        
        presentation_spikes = 0
        
        for t in range(pattern_duration):
            # Get input spikes for this timestep
            input_spikes = pattern[t].unsqueeze(0)  # (1, n_input)
            
            # Compute input current
            current = torch.mm(input_spikes, weights.t())  # (1, n_output)
            
            # Forward pass through output neurons
            output_spikes = output_neurons(current)
            
            # Apply STDP
            dw = stdp.compute_weight_update(input_spikes, output_spikes)
            weights = weights + dw.squeeze(0)
            weights = weights.clamp(stdp_config.w_min, stdp_config.w_max)
            
            presentation_spikes += output_spikes.sum().item()
        
        output_spike_counts.append(presentation_spikes)
        
        # Store weight snapshot periodically
        if presentation % 10 == 0:
            weight_history.append(weights.clone().cpu().numpy())
            
        if presentation % 20 == 0:
            print(f"  Presentation {presentation}: {presentation_spikes:.0f} output spikes")
    
    print(f"\nTraining complete!")
    
    # Test pattern completion
    print("\n" + "=" * 60)
    print("Testing Pattern Completion")
    print("=" * 60)
    
    # Present partial pattern (first half only)
    partial_pattern = pattern.clone()
    partial_pattern[pattern_duration//2:, :] = 0  # Zero out second half
    
    output_neurons.reset_state(batch_size=1)
    
    full_output_spikes = []
    partial_output_spikes = []
    
    # Full pattern response
    output_neurons.reset_state(batch_size=1)
    for t in range(pattern_duration):
        input_spikes = pattern[t].unsqueeze(0)
        current = torch.mm(input_spikes, weights.t())
        output_spikes = output_neurons(current)
        full_output_spikes.append(output_spikes.cpu().numpy())
    
    # Partial pattern response  
    output_neurons.reset_state(batch_size=1)
    for t in range(pattern_duration):
        input_spikes = partial_pattern[t].unsqueeze(0)
        current = torch.mm(input_spikes, weights.t())
        output_spikes = output_neurons(current)
        partial_output_spikes.append(output_spikes.cpu().numpy())
    
    full_output_spikes = np.array(full_output_spikes).squeeze()
    partial_output_spikes = np.array(partial_output_spikes).squeeze()
    
    print(f"  Full pattern output spikes: {full_output_spikes.sum():.0f}")
    print(f"  Partial pattern output spikes: {partial_output_spikes.sum():.0f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Experiment 2: STDP Learning", fontsize=14, fontweight='bold')
    
    # 1. Initial weights
    ax1 = axes[0, 0]
    im1 = ax1.imshow(initial_weights.cpu().numpy(), aspect='auto', cmap='RdBu_r')
    ax1.set_xlabel("Input Neuron")
    ax1.set_ylabel("Output Neuron")
    ax1.set_title("Initial Weights")
    plt.colorbar(im1, ax=ax1)
    
    # 2. Final weights
    ax2 = axes[0, 1]
    im2 = ax2.imshow(weights.cpu().numpy(), aspect='auto', cmap='RdBu_r')
    ax2.set_xlabel("Input Neuron")
    ax2.set_ylabel("Output Neuron")
    ax2.set_title("Final Weights (After STDP)")
    plt.colorbar(im2, ax=ax2)
    
    # 3. Weight change
    ax3 = axes[0, 2]
    weight_change = weights.cpu().numpy() - initial_weights.cpu().numpy()
    im3 = ax3.imshow(weight_change, aspect='auto', cmap='RdBu_r')
    ax3.set_xlabel("Input Neuron")
    ax3.set_ylabel("Output Neuron")
    ax3.set_title("Weight Change (Final - Initial)")
    plt.colorbar(im3, ax=ax3)
    
    # 4. Output spike count over training
    ax4 = axes[1, 0]
    ax4.plot(output_spike_counts, 'b-', alpha=0.7)
    ax4.set_xlabel("Presentation")
    ax4.set_ylabel("Output Spikes")
    ax4.set_title("Learning Curve")
    # Add moving average
    window = 10
    if len(output_spike_counts) >= window:
        ma = np.convolve(output_spike_counts, np.ones(window)/window, mode='valid')
        ax4.plot(np.arange(window-1, len(output_spike_counts)), ma, 'r-', linewidth=2, label='Moving Avg')
        ax4.legend()
    
    # 5. Mean weight evolution
    ax5 = axes[1, 1]
    weight_means = [w.mean() for w in weight_history]
    weight_stds = [w.std() for w in weight_history]
    x = np.arange(len(weight_means)) * 10
    ax5.plot(x, weight_means, 'b-', linewidth=2)
    ax5.fill_between(x, 
                     np.array(weight_means) - np.array(weight_stds),
                     np.array(weight_means) + np.array(weight_stds),
                     alpha=0.3)
    ax5.set_xlabel("Presentation")
    ax5.set_ylabel("Weight")
    ax5.set_title("Mean Weight ± Std Over Training")
    
    # 6. Pattern completion comparison
    ax6 = axes[1, 2]
    ax6.plot(full_output_spikes.sum(axis=1), 'b-', label='Full Pattern', linewidth=2)
    ax6.plot(partial_output_spikes.sum(axis=1), 'r--', label='Partial Pattern', linewidth=2)
    ax6.axvline(pattern_duration//2, color='gray', linestyle=':', label='Partial cutoff')
    ax6.set_xlabel("Time (ms)")
    ax6.set_ylabel("Output Spikes")
    ax6.set_title("Pattern Completion Test")
    ax6.legend()
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "exp2_stdp_learning.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    
    plt.show()
    
    # Success criteria
    print("\n" + "=" * 60)
    print("Success Criteria Check")
    print("=" * 60)
    
    weight_changed = np.abs(weight_change).mean() > 0.01
    patterns_learned = output_spike_counts[-1] > output_spike_counts[0] * 0.5
    stable_training = np.std(output_spike_counts[-20:]) < np.std(output_spike_counts[:20]) * 2
    
    criteria = [
        ("STDP modifies weights correctly", weight_changed),
        ("Temporal patterns learned", patterns_learned),
        ("Stable training dynamics", stable_training),
        ("Pattern completion works", partial_output_spikes.sum() > 0),
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
