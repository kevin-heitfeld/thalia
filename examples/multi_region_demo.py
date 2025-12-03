"""
Multi-Region Brain Integration Demo
====================================

This demo demonstrates all 5 brain regions working together in a hierarchical
spike-based learning system:

1. CORTEX: Visual feature extraction (unsupervised STDP)
2. HIPPOCAMPUS: Episodic memory storage (one-shot theta-gated STDP)
3. PREFRONTAL: Working memory and context (dopamine-gated STDP)
4. STRIATUM: Action selection (reward-modulated STDP)
5. CEREBELLUM: Motor refinement (error-corrective STDP)

Task: Context-dependent pattern classification
===============================================
- Visual input → Cortex extracts features
- PFC maintains context (which rule to apply)
- Hippocampus stores pattern-context associations
- Striatum selects action based on context + features
- Cerebellum refines the motor output

The key insight: each region uses spike-based learning with different
modulation signals (unsupervised, theta-gated, dopamine-gated, reward, error).
"""

import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Import all brain regions
from thalia.regions.cortex import Cortex, CortexConfig
from thalia.regions.hippocampus import Hippocampus, HippocampusConfig
from thalia.regions.prefrontal import Prefrontal, PrefrontalConfig
from thalia.regions.striatum import Striatum, StriatumConfig
from thalia.regions.cerebellum import Cerebellum, CerebellumConfig
from thalia.regions.base import LearningRule


def create_context_patterns(
    n_patterns: int = 4,
    input_size: int = 256,  # 16x16 visual field
) -> Tuple[List[torch.Tensor], List[int]]:
    """Create visual patterns with associated labels.

    Patterns are created by activating different regions of the visual field:
    - Pattern 0: Left half active
    - Pattern 1: Right half active
    - Pattern 2: Top half active
    - Pattern 3: Bottom half active
    """
    patterns = []
    labels = []

    grid_size = int(input_size ** 0.5)  # 16 for 256

    for label in range(n_patterns):
        pattern = torch.zeros(input_size)

        if label == 0:  # Left half
            for row in range(grid_size):
                for col in range(grid_size // 2):
                    pattern[row * grid_size + col] = 1.0
        elif label == 1:  # Right half
            for row in range(grid_size):
                for col in range(grid_size // 2, grid_size):
                    pattern[row * grid_size + col] = 1.0
        elif label == 2:  # Top half
            for row in range(grid_size // 2):
                for col in range(grid_size):
                    pattern[row * grid_size + col] = 1.0
        elif label == 3:  # Bottom half
            for row in range(grid_size // 2, grid_size):
                for col in range(grid_size):
                    pattern[row * grid_size + col] = 1.0

        patterns.append(pattern)
        labels.append(label)

    return patterns, labels


def create_context_signal(context_id: int, n_contexts: int = 2) -> torch.Tensor:
    """Create a context signal for PFC.

    Context 0: Rule A (e.g., respond to left/right)
    Context 1: Rule B (e.g., respond to top/bottom)
    """
    context = torch.zeros(n_contexts * 4)  # 4 neurons per context
    start_idx = context_id * 4
    context[start_idx:start_idx + 4] = 1.0
    return context


def rate_to_spikes(
    rate_pattern: torch.Tensor,
    n_timesteps: int,
    max_rate: float = 0.8,
) -> torch.Tensor:
    """Convert rate-coded pattern to spike train."""
    spikes = torch.zeros(n_timesteps, *rate_pattern.shape)
    for t in range(n_timesteps):
        spikes[t] = (torch.rand_like(rate_pattern) < rate_pattern * max_rate).float()
    return spikes


class MultiBrainSystem:
    """Integrated multi-region brain system."""

    def __init__(
        self,
        input_size: int = 256,
        cortex_size: int = 256,  # Match working demo
        hippocampus_size: int = 64,
        pfc_size: int = 32,
        n_actions: int = 4,
        n_contexts: int = 2,
        device: str = "cpu",
    ):
        self.input_size = input_size
        self.n_actions = n_actions
        self.n_contexts = n_contexts
        self.device = device

        # ======================================================================
        # Create brain regions (matched to working demo settings)
        # ======================================================================

        # 1. CORTEX: Visual feature extraction (same as working demo)
        self.cortex = Cortex(CortexConfig(
            n_input=input_size,
            n_output=cortex_size,
            hebbian_lr=0.005,  # Lower LR like working demo
            synaptic_scaling_enabled=True,
            synaptic_scaling_target=0.4,
            kwta_k=max(1, int(cortex_size * 0.10)),  # 10% sparsity
            lateral_inhibition=True,
            inhibition_strength=0.8,
            device=device,
        ))

        # 2. HIPPOCAMPUS: Episodic memory
        self.hippocampus = Hippocampus(HippocampusConfig(
            n_input=cortex_size,
            n_output=hippocampus_size,
            stdp_lr=0.1,
            sparsity_target=0.1,
            soft_bounds=True,
            synaptic_scaling_enabled=True,
            device=device,
        ))

        # 3. PREFRONTAL CORTEX: Working memory
        self.pfc = Prefrontal(PrefrontalConfig(
            n_input=n_contexts * 4,
            n_output=pfc_size,
            stdp_lr=0.02,
            dopamine_baseline=0.3,
            gate_threshold=0.4,
            soft_bounds=True,
            synaptic_scaling_enabled=True,
            device=device,
        ))

        # 4. STRIATUM: Action selection (matched to working demo)
        self.striatum = Striatum(StriatumConfig(
            n_input=cortex_size,
            n_output=n_actions,
            learning_rule=LearningRule.REWARD_MODULATED_STDP,
            stdp_lr=0.02,  # Lower LR like working demo
            stdp_tau_ms=20.0,
            eligibility_tau_ms=300.0,
            dopamine_burst=2.0,
            dopamine_dip=-0.5,
            lateral_inhibition=True,
            inhibition_strength=1.0,
            exploration_epsilon=0.7,  # Higher exploration
            exploration_decay=0.995,  # Slower decay
            min_epsilon=0.05,
            device=device,
        ))

        # 5. CEREBELLUM: Motor refinement
        self.cerebellum = Cerebellum(CerebellumConfig(
            n_input=n_actions,
            n_output=n_actions,
            stdp_lr=0.05,
            soft_bounds=True,
            synaptic_scaling_enabled=True,
            device=device,
        ))

        # Initialize weights with random values (like working demo)
        with torch.no_grad():
            self.cortex.weights.data = torch.rand(cortex_size, input_size) * 0.3 + 0.1
            self.striatum.weights.data = torch.rand(n_actions, cortex_size) * 0.3 + 0.1

        print("Multi-Brain System Initialized:")
        print(f"  Cortex: {input_size} → {cortex_size}")
        print(f"  Hippocampus: {cortex_size} → {hippocampus_size}")
        print(f"  PFC: {n_contexts * 4} → {pfc_size}")
        print(f"  Striatum: {cortex_size} → {n_actions} (direct Cortex pathway)")
        print(f"  Cerebellum: {n_actions} → {n_actions}")
        print(f"  Initial weights: Cortex mean={self.cortex.weights.mean():.3f}, "
              f"Striatum mean={self.striatum.weights.mean():.3f}")

    def reset(self):
        """Reset all regions for a new trial."""
        self.cortex.reset()
        self.hippocampus.reset()
        self.pfc.reset()
        self.striatum.reset()
        self.cerebellum.reset()

    def forward_trial(
        self,
        visual_input: torch.Tensor,
        context_signal: torch.Tensor,
        n_timesteps: int = 20,
        dopamine_signal: float = 0.0,
        explore: bool = True,
    ) -> Dict[str, Any]:
        """Run a complete trial through all brain regions.

        Args:
            visual_input: Visual pattern [input_size]
            context_signal: Context cue [n_contexts * 4]
            n_timesteps: Number of simulation timesteps
            dopamine_signal: DA signal for PFC gating
            explore: Whether to use exploration in striatum

        Returns:
            Dictionary with outputs from all regions
        """
        # Convert to spike trains
        visual_spikes = rate_to_spikes(visual_input, n_timesteps)
        context_spikes = rate_to_spikes(context_signal, n_timesteps, max_rate=0.5)

        # Accumulate outputs over time
        cortex_total = torch.zeros(self.cortex.config.n_output)
        hippo_total = torch.zeros(self.hippocampus.config.n_output)
        pfc_total = torch.zeros(self.pfc.config.n_output)
        striatum_total = torch.zeros(self.striatum.config.n_output)
        cerebellum_total = torch.zeros(self.cerebellum.config.n_output)

        for t in range(n_timesteps):
            # 1. Visual processing through Cortex
            cortex_out = self.cortex.forward(visual_spikes[t].unsqueeze(0))
            cortex_total += cortex_out.squeeze()

            # 2. PFC maintains context with dopamine gating (parallel pathway)
            pfc_out = self.pfc.forward(
                context_spikes[t].unsqueeze(0),
                dopamine_signal=dopamine_signal,
            )
            pfc_total += pfc_out.squeeze()

            # 3. Hippocampus receives ONLY cortex for episodic storage
            hippo_out = self.hippocampus.forward(cortex_out)
            hippo_total += hippo_out.squeeze()

            # 4. Striatum receives ONLY cortex for action selection
            # (Direct pathway - same as working demo)
            striatum_out = self.striatum.forward(cortex_out, explore=explore)
            striatum_total += striatum_out.squeeze()

            # 5. Cerebellum refines motor output
            cerebellum_out = self.cerebellum.forward(striatum_out)
            cerebellum_total += cerebellum_out.squeeze()

        # Determine selected action from striatum spike counts
        selected_action = int(striatum_total.argmax().item())

        return {
            "cortex_spikes": cortex_total,
            "hippocampus_spikes": hippo_total,
            "pfc_spikes": pfc_total,
            "striatum_spikes": striatum_total,
            "cerebellum_spikes": cerebellum_total,
            "selected_action": selected_action,
            "exploring": self.striatum.exploring,
        }

    def learn_trial(
        self,
        visual_input: torch.Tensor,
        context_signal: torch.Tensor,
        target_action: int,
        reward: float,
        n_timesteps: int = 20,
    ) -> Dict[str, Any]:
        """Run a learning trial with all plasticity enabled.

        Args:
            visual_input: Visual pattern
            context_signal: Context cue
            target_action: Correct action for supervised learning
            reward: Reward signal for RL
            n_timesteps: Number of timesteps

        Returns:
            Learning metrics from all regions
        """
        # Forward pass with exploration
        outputs = self.forward_trial(
            visual_input,
            context_signal,
            n_timesteps=n_timesteps,
            dopamine_signal=0.5 if reward > 0 else -0.3,
            explore=True,
        )

        selected = outputs["selected_action"]
        correct = (selected == target_action)

        # ======================================================================
        # Apply learning to all regions
        # ======================================================================

        # 1. CORTEX: Unsupervised STDP (always learns)
        cortex_spikes = (outputs["cortex_spikes"] > 0).float().unsqueeze(0)
        visual_spikes = (visual_input > 0.5).float().unsqueeze(0)
        self.cortex.learn(visual_spikes, cortex_spikes)

        # 2. HIPPOCAMPUS: One-shot episodic storage
        hippo_spikes = (outputs["hippocampus_spikes"] > 0).float().unsqueeze(0)
        self.hippocampus.learn(
            cortex_spikes,  # Only cortex input
            hippo_spikes,
            force_encoding=True,
        )

        # 3. PREFRONTAL: Dopamine-gated learning
        pfc_spikes = (outputs["pfc_spikes"] > 0).float().unsqueeze(0)
        self.pfc.learn(
            context_signal.unsqueeze(0),
            pfc_spikes,
            reward=reward,
        )

        # 4. STRIATUM: Reward-modulated STDP
        # Reward +1 for correct, -1 for incorrect (clear signal)
        self.striatum.deliver_reward(1.0 if correct else -1.0)

        # 5. CEREBELLUM: Error-corrective learning
        target_tensor = torch.zeros(1, self.n_actions)
        target_tensor[0, target_action] = 1.0
        cerebellum_spikes = (outputs["cerebellum_spikes"] > 0).float().unsqueeze(0)
        self.cerebellum.learn(
            outputs["striatum_spikes"].unsqueeze(0),
            cerebellum_spikes,
            target=target_tensor,
        )

        return {
            "selected_action": selected,
            "target_action": target_action,
            "correct": correct,
            "reward": reward,
            "exploring": outputs["exploring"],
        }


def run_demo(
    n_epochs: int = 500,  # Same as working demo
    n_timesteps: int = 30,  # Same as working demo
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the multi-region integration demo."""

    print("=" * 70)
    print("Multi-Region Brain Integration Demo")
    print("=" * 70)

    # Create patterns and system
    patterns, labels = create_context_patterns(n_patterns=4)
    system = MultiBrainSystem(
        input_size=256,
        cortex_size=128,
        hippocampus_size=64,
        pfc_size=32,
        n_actions=4,
        n_contexts=2,
    )

    # Training history
    accuracy_history = []

    print(f"\nTraining for {n_epochs} epochs...")
    print("-" * 70)

    for epoch in range(n_epochs):
        correct_count = 0
        total_count = 0

        for pattern_idx in range(len(patterns)):
            # Start simple: no context switching, just pattern → action
            context_id = 0  # Fixed context for now

            # Target action is just the pattern index
            target_action = pattern_idx

            system.reset()

            # Create context signal
            context = create_context_signal(context_id, n_contexts=2)

            # Run learning trial - reward based on correctness
            result = system.learn_trial(
                visual_input=patterns[pattern_idx],
                context_signal=context,
                target_action=target_action,
                reward=1.0,  # Reward for correct responses
                n_timesteps=n_timesteps,
            )

            if result["correct"]:
                correct_count += 1
            total_count += 1

        # Decay exploration
        system.striatum.decay_exploration()

        accuracy = 100.0 * correct_count / total_count
        accuracy_history.append(accuracy)

        if verbose and (epoch % 20 == 0 or epoch == n_epochs - 1):
            eps = system.striatum.current_epsilon
            print(f"Epoch {epoch:3d}: Accuracy={accuracy:5.1f}%, Epsilon={eps:.3f}")

    # ======================================================================
    # Final Evaluation (no exploration)
    # ======================================================================
    print("\n" + "-" * 70)
    print("Final Evaluation (no exploration)")
    print("-" * 70)

    correct_count = 0
    total_count = 0

    # Test with context 0 only (simple classification)
    context_id = 0
    context = create_context_signal(context_id, n_contexts=2)

    for pattern_idx in range(len(patterns)):
        target_action = pattern_idx  # Simple: pattern → action

        system.reset()

        result = system.forward_trial(
            visual_input=patterns[pattern_idx],
            context_signal=context,
            n_timesteps=n_timesteps,
            dopamine_signal=0.0,
            explore=False,
        )

        if result["selected_action"] == target_action:
            correct_count += 1
        total_count += 1

        if verbose:
            status = "✓" if result["selected_action"] == target_action else "✗"
            print(f"  Pattern {pattern_idx}: "
                  f"Selected={result['selected_action']}, "
                  f"Target={target_action} {status}")

    test_accuracy = 100.0 * correct_count / total_count
    print(f"\nTest Accuracy: {test_accuracy:.1f}%")

    # ======================================================================
    # Plot results
    # ======================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Training accuracy
    axes[0].plot(accuracy_history)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Training Accuracy (%)")
    axes[0].set_title("Multi-Region Learning Progress")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=100, color='g', linestyle='--', alpha=0.5, label='Perfect')
    axes[0].legend()

    # Weight visualization for each region
    region_weights = {
        "Cortex": system.cortex.weights.data.mean().item(),
        "Hippocampus": system.hippocampus.weights.data.mean().item(),
        "PFC": system.pfc.weights.data.mean().item(),
        "Striatum": system.striatum.weights.data.mean().item(),
        "Cerebellum": system.cerebellum.weights.data.mean().item(),
    }

    axes[1].bar(region_weights.keys(), region_weights.values())
    axes[1].set_ylabel("Mean Weight")
    axes[1].set_title("Final Weight Statistics by Region")
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    # Save plot
    save_dir = Path(__file__).parent.parent / "experiments" / "results" / "integration"
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / "multi_region_demo.png", dpi=150)
    print(f"\nPlot saved to: {save_dir / 'multi_region_demo.png'}")

    plt.close()

    # Summary
    if test_accuracy >= 75.0:
        print(f"\n✓ SUCCESS: Test accuracy {test_accuracy:.1f}%")
    else:
        print(f"\n✗ NEEDS TUNING: Test accuracy {test_accuracy:.1f}%")

    return {
        "test_accuracy": test_accuracy,
        "final_training_accuracy": accuracy_history[-1],
        "accuracy_history": accuracy_history,
        "region_weights": region_weights,
    }


if __name__ == "__main__":
    results = run_demo(verbose=True)
