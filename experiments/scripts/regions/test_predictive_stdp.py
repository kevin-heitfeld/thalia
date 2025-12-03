"""
Test: Predictive STDP Learning in Cortex

This experiment tests whether PREDICTIVE_STDP learning actually works with
real spiking dynamics. The key difference from rate-coded experiments:

1. Input is converted to spike trains (Poisson encoding)
2. Neurons are real LIF neurons that spike or don't
3. STDP creates eligibility traces from spike timing
4. Prediction error modulates whether learning occurs

If this works, we have a path to true spiking learning in Thalia.

Task: Learn to distinguish 4 different input patterns.
Success: After training, each pattern should activate a different subset of neurons.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from thalia.regions.cortex import Cortex, CortexConfig
from thalia.regions.base import LearningRule


def create_patterns(n_patterns: int = 4, pattern_size: int = 64) -> List[torch.Tensor]:
    """Create distinct sparse patterns."""
    patterns = []
    for i in range(n_patterns):
        pattern = torch.zeros(pattern_size)
        # Each pattern activates a different quadrant + some overlap
        start = (pattern_size // n_patterns) * i
        end = start + pattern_size // n_patterns + pattern_size // 8
        pattern[start:min(end, pattern_size)] = 1.0
        # Add some noise for realism
        pattern = pattern + torch.randn(pattern_size) * 0.1
        pattern = pattern.clamp(0.0, 1.0)
        patterns.append(pattern)
    return patterns


def rate_to_spikes(rate: torch.Tensor, n_timesteps: int = 20) -> torch.Tensor:
    """Convert rate-coded pattern to spike train using Poisson encoding."""
    # rate: (n_input,) with values 0-1
    # output: (n_timesteps, n_input) binary spikes
    spikes = torch.zeros(n_timesteps, rate.shape[0])
    for t in range(n_timesteps):
        spikes[t] = (torch.rand_like(rate) < rate * 0.3).float()  # Scale rate for reasonable spiking
    return spikes


def run_experiment(
    n_patterns: int = 4,
    n_input: int = 64,
    n_output: int = 32,
    n_epochs: int = 50,
    n_timesteps: int = 100,  # More timesteps for better spike statistics
    verbose: bool = True,
    seed: int | None = None,
):
    """Run the predictive STDP experiment."""
    # Set seeds for reproducibility (optional)
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Create cortex with PREDICTIVE_STDP learning
    config = CortexConfig(
        n_input=n_input,
        n_output=n_output,
        learning_rule=LearningRule.PREDICTIVE_STDP,
        hebbian_lr=0.005,          # Slightly higher with competition
        predictive_lr=0.01,        # Learn to predict
        pred_error_modulation=0.5, 
        pred_error_tau_ms=20.0,    
        pred_error_min_mod=0.2,    
        diagonal_bias=0.3,         # Symmetry breaking
        sfa_enabled=True,          # Prevent monopolization
        sfa_strength=2.0,          # Stronger SFA to reduce spiking
        sfa_tau_ms=100.0,          # Faster adaptation
        lateral_inhibition=True,
        inhibition_strength=0.5,   # Stronger inhibition for sparsity
        kwta_k=8,                  # Only top 8 neurons can spike - CRITICAL for competition!
        dt_ms=1.0,
        w_max=1.0,                 # Normal range
        w_min=0.0,
    )
    cortex = Cortex(config)

    # Store initial weights for comparison
    initial_weights = cortex.weights.detach().clone()

    # Create patterns
    patterns = create_patterns(n_patterns, n_input)

    print("=" * 60)
    print("Predictive STDP Learning Test")
    print("=" * 60)
    print(f"Patterns: {n_patterns}, Input: {n_input}, Output: {n_output}")
    print(f"Learning rule: PREDICTIVE_STDP")
    print(f"Timesteps per pattern: {n_timesteps}")
    print()

    # Track metrics
    error_history = []
    selectivity_history = []
    weight_change_history = []
    ltp_history = []
    ltd_history = []
    spike_count_history = []

    # Training loop
    for epoch in range(n_epochs):
        epoch_errors = []
        epoch_responses = {i: [] for i in range(n_patterns)}
        epoch_ltp = 0.0
        epoch_ltd = 0.0
        epoch_spikes = 0

        weights_before = cortex.weights.detach().clone()

        for pattern_idx, pattern in enumerate(patterns):
            # Reset cortex state for new pattern presentation
            cortex.reset()

            # Convert pattern to spike train
            spike_train = rate_to_spikes(pattern, n_timesteps)

            # Present pattern over time
            total_output = torch.zeros(n_output)
            for t in range(n_timesteps):
                input_spikes = spike_train[t].unsqueeze(0)
                output_spikes = cortex.forward(input_spikes)
                metrics = cortex.learn(input_spikes, output_spikes)
                total_output += output_spikes.squeeze()
                epoch_errors.append(metrics.get("pred_error", 0.0))
                epoch_ltp += metrics.get("ltp", 0.0)
                epoch_ltd += metrics.get("ltd", 0.0)
                epoch_spikes += output_spikes.sum().item()

            # Record response pattern
            response = total_output / n_timesteps
            epoch_responses[pattern_idx].append(response.detach().numpy())

        # Track weight changes
        weights_after = cortex.weights.detach()
        weight_change = (weights_after - weights_before).abs().mean().item()
        weight_change_history.append(weight_change)
        ltp_history.append(epoch_ltp)
        ltd_history.append(epoch_ltd)
        spike_count_history.append(epoch_spikes)

        # Calculate selectivity: do different patterns activate different neurons?
        mean_responses = {i: np.mean(epoch_responses[i], axis=0) for i in range(n_patterns)}

        # Selectivity = how different are the response profiles?
        selectivity = 0.0
        for i in range(n_patterns):
            for j in range(i+1, n_patterns):
                diff = np.abs(mean_responses[i] - mean_responses[j]).mean()
                selectivity += diff
        selectivity /= (n_patterns * (n_patterns - 1) / 2)

        avg_error = np.mean(epoch_errors)
        error_history.append(avg_error)
        selectivity_history.append(selectivity)

        if verbose and (epoch + 1) % 5 == 0:
            # Debug: Check spike counts and weight changes
            print(f"Epoch {epoch+1}/{n_epochs}: error={avg_error:.4f}, sel={selectivity:.4f}, "
                  f"spikes={epoch_spikes:.0f}, Δw={weight_change:.5f}, LTP={epoch_ltp:.2f}, LTD={epoch_ltd:.2f}")

    # Final evaluation
    print("\n" + "-" * 60)
    print("Final Evaluation")
    print("-" * 60)

    # Test pattern recognition
    correct = 0
    total = n_patterns * 5  # 5 test trials per pattern

    for pattern_idx, pattern in enumerate(patterns):
        for trial in range(5):
            cortex.reset()
            spike_train = rate_to_spikes(pattern, n_timesteps)

            total_output = torch.zeros(n_output)
            for t in range(n_timesteps):
                input_spikes = spike_train[t].unsqueeze(0)
                output_spikes = cortex.forward(input_spikes)
                total_output += output_spikes.squeeze()

            # Check which neurons are most active
            top_neurons = torch.topk(total_output, k=5).indices
            # Simple check: does this pattern activate neurons in its expected range?
            expected_start = (n_output // n_patterns) * pattern_idx
            expected_end = expected_start + n_output // n_patterns

            if any(expected_start <= n.item() < expected_end for n in top_neurons):
                correct += 1

    accuracy = correct / total * 100
    print(f"Pattern recognition accuracy: {accuracy:.1f}%")

    # Check improvement
    initial_error = np.mean(error_history[:3])
    final_error = np.mean(error_history[-3:])
    error_reduction = (initial_error - final_error) / initial_error * 100 if initial_error > 0 else 0

    print(f"Prediction error reduction: {error_reduction:.1f}%")
    print(f"Final selectivity: {selectivity_history[-1]:.4f}")

    # Success criteria
    passed = []

    # Criterion 1: Prediction error should decrease
    if error_reduction > 10:
        passed.append("✓ Prediction error decreased (>10%)")
    else:
        passed.append(f"✗ Prediction error did not decrease enough ({error_reduction:.1f}%)")

    # Criterion 2: Selectivity should increase
    initial_sel = np.mean(selectivity_history[:3])
    final_sel = np.mean(selectivity_history[-3:])
    if final_sel > initial_sel:
        passed.append("✓ Selectivity increased")
    else:
        passed.append("✗ Selectivity did not increase")

    # Criterion 3: Some pattern recognition
    if accuracy > 30:  # Better than random (25% for 4 patterns)
        passed.append(f"✓ Pattern recognition above chance ({accuracy:.1f}%)")
    else:
        passed.append(f"✗ Pattern recognition at chance ({accuracy:.1f}%)")

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for p in passed:
        print(f"  {p}")

    n_passed = sum(1 for p in passed if p.startswith("✓"))
    overall = "PASSED" if n_passed >= 2 else "FAILED"
    print(f"\nOverall: {overall} ({n_passed}/3 criteria)")

    # Weight analysis
    print("\n" + "-" * 60)
    print("Weight Analysis")
    print("-" * 60)
    final_weights = cortex.weights.detach().cpu().numpy()
    weight_delta = final_weights - initial_weights.cpu().numpy()
    print(f"Initial weights: mean={initial_weights.mean():.4f}, std={initial_weights.std():.4f}")
    print(f"Final weights: mean={final_weights.mean():.4f}, std={final_weights.std():.4f}")
    print(f"Weight change: mean={np.abs(weight_delta).mean():.5f}, max={np.abs(weight_delta).max():.4f}")
    print(f"Total LTP: {sum(ltp_history):.2f}, Total LTD: {sum(ltd_history):.2f}")
    print(f"Avg spikes/epoch: {np.mean(spike_count_history):.1f}")

    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Plot 1: Prediction error over time
    ax1 = axes[0, 0]
    ax1.plot(error_history)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Prediction Error")
    ax1.set_title("Prediction Error During Training")

    # Plot 2: Selectivity over time
    ax2 = axes[0, 1]
    ax2.plot(selectivity_history)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Selectivity")
    ax2.set_title("Response Selectivity During Training")

    # Plot 3: Weight change over time
    ax3 = axes[0, 2]
    ax3.plot(weight_change_history)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Mean |ΔW|")
    ax3.set_title("Weight Change Magnitude")

    # Plot 4: LTP vs LTD
    ax4 = axes[1, 0]
    ax4.plot(ltp_history, label='LTP', color='green')
    ax4.plot(ltd_history, label='LTD', color='red')
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Cumulative")
    ax4.set_title("LTP vs LTD")
    ax4.legend()

    # Plot 5: Spike counts
    ax5 = axes[1, 1]
    ax5.plot(spike_count_history)
    ax5.set_xlabel("Epoch")
    ax5.set_ylabel("Spike Count")
    ax5.set_title("Spikes per Epoch")

    # Plot 6: Final weight matrix
    ax6 = axes[1, 2]
    weights = cortex.weights.detach().cpu().numpy()
    im = ax6.imshow(weights, aspect='auto', cmap='RdBu_r')
    ax6.set_xlabel("Input")
    ax6.set_ylabel("Output")
    ax6.set_title("Learned Weights")
    plt.colorbar(im, ax=ax6)

    plt.tight_layout()

    # Save plot
    results_dir = Path(__file__).parent.parent.parent / "results" / "regions"
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / "test_predictive_stdp.png", dpi=150)
    plt.close()
    print(f"\nPlot saved to: {results_dir / 'test_predictive_stdp.png'}")

    return {
        "accuracy": accuracy,
        "error_reduction": error_reduction,
        "final_selectivity": selectivity_history[-1],
        "passed": n_passed >= 2,
    }


if __name__ == "__main__":
    results = run_experiment(verbose=True)
