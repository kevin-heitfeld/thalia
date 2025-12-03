"""
Cortex → Striatum Integration Demo

This demo shows how two brain regions work together:

1. **Cortex** (PREDICTIVE_STDP):
   - Learns to recognize visual patterns
   - Develops selective feature detectors
   - Outputs sparse, informative representations

2. **Striatum** (REWARD_MODULATED_STDP):
   - Receives Cortex output as input
   - Learns to select correct actions via reinforcement
   - Dopamine modulates learning based on reward

Together, they form a complete sensory-to-action pipeline:
Input Pattern → Cortex (feature extraction) → Striatum (action selection) → Reward

This is a simplified model of the cortico-basal ganglia loop that underlies
habit learning and decision-making in biological brains.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from thalia.regions.cortex import Cortex, CortexConfig
from thalia.regions.striatum import Striatum, StriatumConfig
from thalia.regions.base import LearningRule


def create_visual_patterns(n_patterns: int = 4, size: int = 8) -> List[torch.Tensor]:
    """Create distinct visual patterns (like simple shapes)."""
    patterns = []
    flat_size = size * size

    for i in range(n_patterns):
        pattern = torch.zeros(flat_size)

        # Simple half patterns for clear discrimination
        if i == 0:
            # Left half active
            for row in range(size):
                for col in range(size // 2):
                    pattern[row * size + col] = 1.0
        elif i == 1:
            # Right half active
            for row in range(size):
                for col in range(size // 2, size):
                    pattern[row * size + col] = 1.0
        elif i == 2:
            # Top half active
            for row in range(size // 2):
                for col in range(size):
                    pattern[row * size + col] = 1.0
        elif i == 3:
            # Bottom half active
            for row in range(size // 2, size):
                for col in range(size):
                    pattern[row * size + col] = 1.0

        patterns.append(pattern)

    return patterns


def rate_to_spikes(rate: torch.Tensor, n_timesteps: int = 20) -> torch.Tensor:
    """Convert rate-coded pattern to spike train."""
    spikes = torch.zeros(n_timesteps, rate.shape[0])
    for t in range(n_timesteps):
        spikes[t] = (torch.rand_like(rate) < rate * 0.3).float()
    return spikes


def temporal_to_spikes(pattern: torch.Tensor, n_timesteps: int = 30) -> torch.Tensor:
    """
    Convert pattern to temporal/first-spike coded spike train.

    Higher intensity → earlier spike (more salient features fire first).
    This is biologically plausible and carries more information than rate coding.
    """
    spikes = torch.zeros(n_timesteps, pattern.shape[0])

    # Compute spike times: high intensity = early spike, low = late/never
    # Map intensity [0,1] to spike time [1, n_timesteps] (inverted)
    spike_times = ((1.0 - pattern) * (n_timesteps - 1) + 1).long()

    # Only active neurons spike (pattern > 0)
    active_mask = pattern > 0.1

    for i in range(pattern.shape[0]):
        if active_mask[i]:
            t = int(min(spike_times[i].item(), n_timesteps - 1))
            # Spike at computed time, with some jitter for realism
            jitter = torch.randint(-2, 3, (1,)).item()
            t = int(max(0, min(n_timesteps - 1, t + jitter)))
            spikes[t, i] = 1.0

            # Optional: add a second spike later for sustained activity
            if t + 5 < n_timesteps:
                spikes[t + 5, i] = 1.0

    return spikes


def run_demo(
    n_patterns: int = 4,  # HARDER: 4 patterns (left, right, top, bottom)
    n_epochs: int = 500,  # More epochs for learning both regions
    n_timesteps: int = 30,  # Moderate trial length
    seed: int = 42,
    verbose: bool = True,
    use_temporal_coding: bool = False,  # Use rate coding (simpler)
):
    """Run the Cortex → Striatum integration demo."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # =========================================================================
    # Setup: Create both regions
    # =========================================================================

    input_size = 256   # 16x16 visual field
    cortex_size = 256  # MORE cortex neurons for 4 patterns
    n_actions = n_patterns  # One action per pattern

    # Cortex: Feature extraction with COMPETITIVE STDP
    # Enable learning with synaptic scaling to prevent saturation
    cortex_config = CortexConfig(
        n_input=input_size,
        n_output=cortex_size,
        learning_rule=LearningRule.STDP,  # Standard STDP with competition
        hebbian_lr=0.005,  # LOWER learning rate to prevent saturation
        stdp_tau_plus_ms=20.0,
        stdp_tau_minus_ms=25.0,  # Slightly longer LTD for balance
        lateral_inhibition=True,
        inhibition_strength=0.8,  # Strong competition
        kwta_k=int(cortex_size * 0.10),  # Only 10% neurons fire (sparse)
        synaptic_scaling_enabled=True,  # PREVENT weight saturation
        synaptic_scaling_target=0.4,  # Target weight level
        synaptic_scaling_tau=50.0,  # Adaptation time constant
        dt_ms=1.0,
        w_max=1.0,
        w_min=0.0,
    )
    cortex = Cortex(cortex_config)

    # Striatum: Action selection with reward-modulated STDP
    # Higher learning rate for RL, slower exploration decay for 4 patterns
    striatum_config = StriatumConfig(
        n_input=cortex_size,  # Receives from Cortex!
        n_output=n_actions,
        learning_rule=LearningRule.REWARD_MODULATED_STDP,
        stdp_lr=0.02,  # Lower LR for more patterns
        stdp_tau_ms=20.0,
        eligibility_tau_ms=300.0,  # Longer eligibility for complex task
        dopamine_burst=2.0,  # Less extreme modulation
        dopamine_dip=-0.5,
        lateral_inhibition=True,
        inhibition_strength=1.0,
        dt_ms=1.0,
        w_max=1.0,
        w_min=0.0,
        exploration_epsilon=0.7,  # More exploration for 4 patterns
        exploration_decay=0.995,  # Slower decay
        min_epsilon=0.05,
    )
    striatum = Striatum(striatum_config)

    # Create patterns (left, right, top, bottom halves)
    patterns = create_visual_patterns(n_patterns, size=16)
    correct_actions = list(range(n_patterns))

    # =========================================================================
    # Initialize Cortex with RANDOM weights (let STDP learn!)
    # =========================================================================
    # Start with small random weights - STDP will specialize neurons
    size = 16  # Visual field is 16x16
    with torch.no_grad():
        # Random initialization with small weights
        cortex.weights = torch.rand_like(cortex.weights) * 0.3 + 0.1
        # Different random seed for Striatum
        striatum.weights = torch.rand_like(striatum.weights) * 0.3 + 0.1
    
    print(f"Initialized with random weights:")
    print(f"  Cortex: mean={cortex.weights.mean():.3f}, std={cortex.weights.std():.3f}")
    print(f"  Striatum: mean={striatum.weights.mean():.3f}, std={striatum.weights.std():.3f}")
    
    coding_type = "Temporal (first-spike)" if use_temporal_coding else "Rate"

    print("=" * 70)
    print("Cortex → Striatum Integration Demo (4 PATTERNS, BOTH LEARNING)")
    print("=" * 70)
    print(f"Visual input: {input_size} neurons (16x16)")
    print(f"Cortex (STDP + synaptic scaling): {input_size} → {cortex_size}")
    print(f"Striatum (REWARD_MODULATED_STDP): {cortex_size} → {n_actions}")
    print(f"Patterns: {n_patterns} (left, right, top, bottom)")
    print(f"Epochs: {n_epochs}")
    print(f"Coding: {coding_type}, Timesteps: {n_timesteps}")
    print()

    # Track metrics
    accuracy_history = []
    cortex_sparsity = []
    weight_histories = {"cortex": [], "striatum": []}

    # =========================================================================
    # Training Loop
    # =========================================================================

    for epoch in range(n_epochs):
        epoch_correct = 0
        epoch_cortex_active = []

        # Shuffle pattern order
        pattern_order = np.random.permutation(n_patterns)

        for pattern_idx in pattern_order:
            pattern = patterns[pattern_idx]
            correct_action = correct_actions[pattern_idx]

            # Reset both regions for new trial
            cortex.reset()
            striatum.reset()

            # Convert to spike train - TEMPORAL or RATE coding
            if use_temporal_coding:
                spike_train = temporal_to_spikes(pattern, n_timesteps)
            else:
                spike_train = rate_to_spikes(pattern, n_timesteps)

            # Process through network
            cortex_output_total = torch.zeros(cortex_size)
            striatum_output_total = torch.zeros(n_actions)
            striatum_drive_total = torch.zeros(n_actions)
            timestep_sparsity = []  # Track per-timestep sparsity
            
            # For early-spike weighted action selection
            early_cortex_activity = torch.zeros(cortex_size)
            early_drive = torch.zeros(n_actions)

            for t in range(n_timesteps):
                # Input → Cortex (for feature extraction)
                input_spikes = spike_train[t].unsqueeze(0)
                cortex_output = cortex.forward(input_spikes)

                # Cortex → Striatum (hierarchical flow!)
                striatum_output = striatum.forward(cortex_output, explore=True)

                # Accumulate
                cortex_output_total += cortex_output.squeeze()
                striatum_output_total += striatum_output.squeeze()

                # Track per-timestep sparsity
                n_active_t = (cortex_output.squeeze() > 0).sum().item()
                timestep_sparsity.append(n_active_t / cortex_size)

                # Temporal weighting: earlier spikes matter more
                # Exponential decay: weight = e^(-t/tau)
                temporal_weight = np.exp(-t / 10.0)  # tau=10 timesteps
                weighted_input = torch.matmul(cortex_output, striatum.weights.T)
                striatum_drive_total += weighted_input.squeeze() * temporal_weight
                
                # Track early activity (first 10 timesteps)
                if t < 10:
                    early_cortex_activity += cortex_output.squeeze()
                    early_drive += weighted_input.squeeze()

                # Update STDP eligibility (no reward yet)
                cortex.learn(input_spikes, cortex_output, reward=None)
                striatum.learn(cortex_output, striatum_output, reward=None)

            # Debug: Check cortex representation selectivity
            if epoch == 0 and verbose:
                total_active = (early_cortex_activity > 0).sum().item()
                print(f"  Pattern {pattern_idx}: cortex_active={total_active}, striatum_spikes={striatum_output_total.tolist()}")

            # Track cortex sparsity (per-timestep average, not cumulative)
            avg_sparsity = np.mean(timestep_sparsity)
            epoch_cortex_active.append(avg_sparsity)

            # Determine action using Striatum spike counts (matches internal selection)
            selected_action = int(striatum_output_total.argmax().item())

            # Compute reward
            if selected_action == correct_action:
                reward = 1.0
                epoch_correct += 1
            else:
                reward = -1.0

            # Deliver reward to Striatum (triggers weight update)
            striatum.deliver_reward(reward)

            # Cortex also gets reward signal for attention/salience
            # (optional - cortex can learn without reward)
            cortex.learn(
                spike_train[-1].unsqueeze(0),
                cortex_output,
                reward=reward,
            )

        # Track metrics
        accuracy = epoch_correct / n_patterns * 100
        accuracy_history.append(accuracy)
        cortex_sparsity.append(np.mean(epoch_cortex_active))

        weight_histories["cortex"].append(cortex.weights.detach().mean().item())
        weight_histories["striatum"].append(striatum.weights.detach().mean().item())

        # Decay exploration
        striatum.decay_exploration()

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}: "
                  f"accuracy={accuracy:.0f}%, "
                  f"eps={striatum.current_epsilon:.2f}, "
                  f"cortex_sparsity={np.mean(epoch_cortex_active):.2f}")

    # =========================================================================
    # Final Evaluation (no exploration)
    # =========================================================================

    print("\n" + "-" * 70)
    print("Final Evaluation (no exploration)")
    print("-" * 70)

    test_correct = 0
    n_test = n_patterns * 10

    for _ in range(n_test // n_patterns):
        for pattern_idx, pattern in enumerate(patterns):
            correct_action = correct_actions[pattern_idx]

            cortex.reset()
            striatum.reset()

            # Use same coding as training
            if use_temporal_coding:
                spike_train = temporal_to_spikes(pattern, n_timesteps)
            else:
                spike_train = rate_to_spikes(pattern, n_timesteps)

            striatum_drive = torch.zeros(n_actions)

            for t in range(n_timesteps):
                input_spikes = spike_train[t].unsqueeze(0)

                # Input → Cortex → Striatum (hierarchical flow)
                cortex_output = cortex.forward(input_spikes)
                striatum_output = striatum.forward(cortex_output, explore=False)

                weighted_input = torch.matmul(cortex_output, striatum.weights.T)
                striatum_drive += weighted_input.squeeze()

            selected = int(striatum_drive.argmax().item())
            if selected == correct_action:
                test_correct += 1

    test_accuracy = test_correct / n_test * 100
    print(f"Test accuracy: {test_accuracy:.1f}%")

    # =========================================================================
    # Analysis
    # =========================================================================

    initial_acc = np.mean(accuracy_history[:5])
    final_acc = np.mean(accuracy_history[-5:])
    improvement = final_acc - initial_acc

    print(f"\nLearning Summary:")
    print(f"  Initial accuracy: {initial_acc:.1f}%")
    print(f"  Final accuracy: {final_acc:.1f}%")
    print(f"  Improvement: {improvement:.1f}%")

    # Check weight structure
    print(f"\nCortex Weight Structure:")
    print(f"  Mean: {cortex.weights.mean():.3f}, Std: {cortex.weights.std():.3f}")

    print(f"\nStriatum Weight Structure:")
    for action in range(n_actions):
        w_mean = striatum.weights[action].mean().item()
        w_std = striatum.weights[action].std().item()
        print(f"  Action {action}: mean={w_mean:.3f}, std={w_std:.3f}")

    # Success criteria
    passed = test_accuracy > 100 / n_patterns + 15  # Above chance + 15%

    print(f"\n{'✓ SUCCESS' if passed else '✗ FAILED'}: Test accuracy {test_accuracy:.1f}%")

    # =========================================================================
    # Visualization
    # =========================================================================

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Accuracy over time
    ax1 = axes[0, 0]
    ax1.plot(accuracy_history, 'b-', alpha=0.7)
    ax1.axhline(y=100/n_patterns, color='r', linestyle='--', label='Chance')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Action Selection Accuracy")
    ax1.legend()

    # Plot 2: Cortex sparsity over time
    ax2 = axes[0, 1]
    ax2.plot(cortex_sparsity, 'g-', alpha=0.7)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Fraction Active")
    ax2.set_title("Cortex Sparsity (lower = more selective)")

    # Plot 3: Weight evolution
    ax3 = axes[1, 0]
    ax3.plot(weight_histories["cortex"], label="Cortex")
    ax3.plot(weight_histories["striatum"], label="Striatum")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Mean Weight")
    ax3.set_title("Weight Evolution")
    ax3.legend()

    # Plot 4: Striatum weights
    ax4 = axes[1, 1]
    im = ax4.imshow(striatum.weights.detach().cpu().numpy(), aspect='auto', cmap='RdBu_r')
    ax4.set_xlabel("Cortex Neuron")
    ax4.set_ylabel("Action")
    ax4.set_title("Striatum Weight Matrix")
    plt.colorbar(im, ax=ax4)

    plt.tight_layout()

    # Save figure
    results_dir = project_root / "experiments" / "results" / "integration"
    results_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(results_dir / "cortex_striatum_demo.png", dpi=150)
    print(f"\nPlot saved to: {results_dir / 'cortex_striatum_demo.png'}")

    plt.show()

    return {
        "test_accuracy": test_accuracy,
        "improvement": improvement,
        "passed": passed,
    }


if __name__ == "__main__":
    # Run multiple times with different seeds to verify robustness
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--multi":
        print("=" * 60)
        print("RUNNING MULTIPLE SEEDS FOR ROBUSTNESS CHECK")
        print("=" * 60)
        successes = 0
        for seed in [42, 123, 456, 789, 2025]:
            torch.manual_seed(seed)
            results = run_demo(verbose=False)
            acc = results["test_accuracy"]
            status = "✓" if acc >= 75.0 else "✗"
            print(f"Seed {seed}: Test accuracy = {acc:.1f}% {status}")
            if acc >= 75.0:
                successes += 1
        print(f"\nPassed: {successes}/5 seeds")
    else:
        results = run_demo(verbose=True)
