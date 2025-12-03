"""
Test: Reward-Modulated STDP Learning in Striatum

This experiment tests whether REWARD_MODULATED_STDP learning works with
real spiking dynamics. The key difference from rate-coded THREE_FACTOR:

1. Input is converted to spike trains (Poisson encoding)
2. Neurons are real LIF neurons that spike or don't
3. STDP creates eligibility traces from spike TIMING (not just correlation)
4. Dopamine modulates whether learning occurs (third factor)

If this works, we have spike-based reinforcement learning in Thalia.

Task: Learn which action (output neuron) to select for each input pattern.
Reward: +1 if correct action selected, -1 if wrong action selected.
Success: After training, action selection accuracy should exceed chance.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from thalia.regions.striatum import Striatum, StriatumConfig
from thalia.regions.base import LearningRule


def create_patterns(n_patterns: int = 4, pattern_size: int = 64) -> List[torch.Tensor]:
    """Create distinct sparse patterns."""
    patterns = []
    for i in range(n_patterns):
        pattern = torch.zeros(pattern_size)
        # Each pattern activates a different quadrant
        start = (pattern_size // n_patterns) * i
        end = start + pattern_size // n_patterns
        pattern[start:end] = 1.0
        # Add some noise for realism
        pattern = pattern + torch.randn(pattern_size) * 0.1
        pattern = pattern.clamp(0.0, 1.0)
        patterns.append(pattern)
    return patterns


def rate_to_spikes(rate: torch.Tensor, n_timesteps: int = 20) -> torch.Tensor:
    """Convert rate-coded pattern to spike train using Poisson encoding."""
    spikes = torch.zeros(n_timesteps, rate.shape[0])
    for t in range(n_timesteps):
        spikes[t] = (torch.rand_like(rate) < rate * 0.3).float()
    return spikes


def run_experiment(
    n_patterns: int = 2,  # Simpler task: 2 patterns
    n_input: int = 64,
    n_actions: int = 2,   # 2 actions
    n_epochs: int = 100,
    n_timesteps: int = 50,  # Timesteps per trial
    verbose: bool = True,
    seed: int | None = None,
):
    """Run the reward-modulated STDP experiment."""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # Create striatum - try both learning rules
    use_stdp = False  # Set to False to test THREE_FACTOR baseline
    
    if use_stdp:
        config = StriatumConfig(
            n_input=n_input,
            n_output=n_actions,
            learning_rule=LearningRule.REWARD_MODULATED_STDP,
            stdp_lr=0.02,
            stdp_tau_ms=20.0,
            eligibility_tau_ms=100.0,
            dopamine_burst=2.0,
            dopamine_dip=-0.5,
            lateral_inhibition=True,
            inhibition_strength=0.3,
            dt_ms=1.0,
            w_max=1.0,
            w_min=0.0,
        )
    else:
        # Baseline: THREE_FACTOR (rate-coded)
        config = StriatumConfig(
            n_input=n_input,
            n_output=n_actions,
            learning_rule=LearningRule.THREE_FACTOR,
            three_factor_lr=0.05,
            eligibility_tau_ms=100.0,
            dopamine_burst=2.0,
            dopamine_dip=-0.5,
            lateral_inhibition=True,
            inhibition_strength=0.3,
            dt_ms=1.0,
            w_max=1.0,
            w_min=0.0,
        )
    striatum = Striatum(config)
    print(f"Using learning rule: {config.learning_rule.name}")
    
    # NO warm-start - learn from scratch

    # Store initial weights
    initial_weights = striatum.weights.detach().clone()

    # Create patterns (each pattern has a "correct" action)
    patterns = create_patterns(n_patterns, n_input)
    correct_actions = list(range(n_patterns))  # Pattern i → Action i

    print("=" * 60)
    print("Reward-Modulated STDP Learning Test")
    print("=" * 60)
    print(f"Patterns: {n_patterns}, Input: {n_input}, Actions: {n_actions}")
    print(f"Learning rule: REWARD_MODULATED_STDP")
    print(f"Timesteps per trial: {n_timesteps}")
    print()

    # Track metrics
    accuracy_history = []
    reward_history = []
    dopamine_history = []
    weight_change_history = []

    # Training loop
    for epoch in range(n_epochs):
        epoch_correct = 0
        epoch_rewards = []
        epoch_dopamine = []
        weights_before = striatum.weights.detach().clone()

        # Shuffle pattern presentation order
        pattern_order = np.random.permutation(n_patterns)

        for pattern_idx in pattern_order:
            pattern = patterns[pattern_idx]
            correct_action = correct_actions[pattern_idx]

            # Reset striatum for new trial
            striatum.reset()

            # Convert pattern to spike train
            spike_train = rate_to_spikes(pattern, n_timesteps)

            # Present pattern over time and accumulate eligibility
            total_output = torch.zeros(n_actions)
            for t in range(n_timesteps):
                input_spikes = spike_train[t].unsqueeze(0)
                output_spikes = striatum.forward(input_spikes)
                total_output += output_spikes.squeeze()

                # Build up eligibility during presentation (NO reward yet)
                # Just update traces, don't apply dopamine
                striatum.learn(input_spikes, output_spikes, reward=None)

            # Determine selected action (most active neuron)
            selected_action = total_output.argmax().item()

            # Compute reward based on action correctness
            if selected_action == correct_action:
                reward = 1.0
                epoch_correct += 1
            else:
                reward = -1.0

            epoch_rewards.append(reward)

            # NOW deliver reward - this triggers the actual weight update
            # Use the last spikes as context (eligibility is already accumulated)
            last_input = spike_train[-1].unsqueeze(0)
            last_output = torch.zeros(n_actions).unsqueeze(0)
            last_output[0, selected_action] = 1.0  # Mark which action was taken
            
            metrics = striatum.learn(last_input, last_output, reward=reward)
            epoch_dopamine.append(metrics.get("dopamine", 0.0))

        # Track metrics
        accuracy = epoch_correct / n_patterns * 100
        accuracy_history.append(accuracy)
        reward_history.append(np.mean(epoch_rewards))
        dopamine_history.append(np.mean(epoch_dopamine))

        weight_change = (striatum.weights.detach() - weights_before).abs().mean().item()
        weight_change_history.append(weight_change)

        if verbose and (epoch + 1) % 10 == 0:
            # Debug: show which actions are being selected
            w_means = striatum.weights.mean(dim=1).tolist()
            print(f"Epoch {epoch+1}/{n_epochs}: accuracy={accuracy:.1f}%, "
                  f"reward={np.mean(epoch_rewards):.2f}, Δw={weight_change:.5f}, "
                  f"w_per_action={[f'{w:.3f}' for w in w_means]}")

    # Final evaluation (multiple test trials)
    print("\n" + "-" * 60)
    print("Final Evaluation")
    print("-" * 60)

    test_correct = 0
    n_test_trials = n_patterns * 10

    for _ in range(n_test_trials // n_patterns):
        for pattern_idx, pattern in enumerate(patterns):
            correct_action = correct_actions[pattern_idx]
            striatum.reset()

            spike_train = rate_to_spikes(pattern, n_timesteps)
            total_output = torch.zeros(n_actions)

            for t in range(n_timesteps):
                input_spikes = spike_train[t].unsqueeze(0)
                output_spikes = striatum.forward(input_spikes)
                total_output += output_spikes.squeeze()

            selected_action = total_output.argmax().item()
            if selected_action == correct_action:
                test_correct += 1

    test_accuracy = test_correct / n_test_trials * 100
    print(f"Test accuracy: {test_accuracy:.1f}%")

    # Check improvement
    initial_acc = np.mean(accuracy_history[:5])
    final_acc = np.mean(accuracy_history[-5:])
    improvement = final_acc - initial_acc

    print(f"Initial accuracy: {initial_acc:.1f}%")
    print(f"Final accuracy: {final_acc:.1f}%")
    print(f"Improvement: {improvement:.1f}%")

    # Success criteria
    passed = []
    chance = 100 / n_patterns

    if test_accuracy > chance + 10:
        passed.append(f"✓ Test accuracy ({test_accuracy:.1f}%) above chance+10% ({chance+10:.1f}%)")
    else:
        passed.append(f"✗ Test accuracy ({test_accuracy:.1f}%) not above chance+10% ({chance+10:.1f}%)")

    if final_acc > initial_acc + 5:
        passed.append(f"✓ Learning occurred (final > initial by {improvement:.1f}%)")
    else:
        passed.append(f"✗ No significant learning (improvement only {improvement:.1f}%)")

    if np.mean(reward_history[-10:]) > np.mean(reward_history[:10]):
        passed.append("✓ Reward increased over training")
    else:
        passed.append("✗ Reward did not increase over training")

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
    final_weights = striatum.weights.detach().cpu().numpy()
    weight_delta = final_weights - initial_weights.cpu().numpy()
    print(f"Initial weights: mean={initial_weights.mean():.4f}, std={initial_weights.std():.4f}")
    print(f"Final weights: mean={final_weights.mean():.4f}, std={final_weights.std():.4f}")
    print(f"Weight change: mean={np.abs(weight_delta).mean():.5f}, max={np.abs(weight_delta).max():.4f}")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Plot 1: Accuracy over time
    ax1 = axes[0, 0]
    ax1.plot(accuracy_history)
    ax1.axhline(y=chance, color='r', linestyle='--', label=f'Chance ({chance:.0f}%)')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Action Selection Accuracy")
    ax1.legend()

    # Plot 2: Reward over time
    ax2 = axes[0, 1]
    ax2.plot(reward_history)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Mean Reward")
    ax2.set_title("Reward During Training")

    # Plot 3: Weight change over time
    ax3 = axes[1, 0]
    ax3.plot(weight_change_history)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Mean |ΔW|")
    ax3.set_title("Weight Change Magnitude")

    # Plot 4: Final weight matrix
    ax4 = axes[1, 1]
    weights = striatum.weights.detach().cpu().numpy()
    im = ax4.imshow(weights, aspect='auto', cmap='RdBu_r')
    ax4.set_xlabel("Input")
    ax4.set_ylabel("Action")
    ax4.set_title("Learned Weights")
    plt.colorbar(im, ax=ax4)

    plt.tight_layout()

    # Save plot
    results_dir = Path(__file__).parent.parent.parent / "results" / "regions"
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / "test_reward_modulated_stdp.png", dpi=150)
    plt.close()
    print(f"\nPlot saved to: {results_dir / 'test_reward_modulated_stdp.png'}")

    return {
        "test_accuracy": test_accuracy,
        "final_accuracy": final_acc,
        "improvement": improvement,
        "passed": n_passed >= 2,
    }


if __name__ == "__main__":
    results = run_experiment(verbose=True)
