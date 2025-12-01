#!/usr/bin/env python3
"""
Demo: Brain Regions Comparison

This demonstrates the five specialized brain regions in Thalia,
each with its own learning algorithm optimized for different tasks:

1. CORTEX (unsupervised): Discovers structure through Hebbian learning
2. CEREBELLUM (supervised): Learns from explicit targets (delta rule)
3. STRIATUM (reinforcement): Three-factor rule with dopamine
4. HIPPOCAMPUS (episodic): One-shot learning with theta gating
5. PREFRONTAL (working memory): Gated maintenance and rule learning

Run this to see the difference in action!
"""

import torch
from thalia.regions import (
    Cortex, CortexConfig,
    Cerebellum, CerebellumConfig,
    Striatum, StriatumConfig,
    Hippocampus, HippocampusConfig,
    Prefrontal, PrefrontalConfig,
)


def demo_cortex():
    """Demonstrate unsupervised learning in Cortex."""
    print("\n" + "=" * 70)
    print("1. CORTEX: Unsupervised Hebbian Learning")
    print("=" * 70)

    config = CortexConfig(
        n_input=10,
        n_output=5,
        learning_rate=0.02,
        hebbian_lr=0.01,  # Lower learning rate to prevent runaway
        inhibition_strength=2.0,  # Much stronger lateral inhibition
        heterosynaptic_ratio=0.5,  # More aggressive unlearning
        device="cpu",
    )
    cortex = Cortex(config)

    print("\nCortex learns through correlation, not teaching signals.")
    print("It strengthens connections for frequently co-active patterns.")

    initial_weights = cortex.weights.clone().detach()
    print(f"\nInitial weights range: [{initial_weights.min():.3f}, {initial_weights.max():.3f}]")

    n_trials = 200  # More trials for stable learning
    n_timesteps = 15  # More timesteps to allow inhibition to settle

    for trial in range(n_trials):
        # Create structured input patterns
        input_pattern = torch.zeros(1, 10)
        active_inputs = trial % 5 * 2  # Cycle through 0, 2, 4, 6, 8
        input_pattern[0, active_inputs:active_inputs + 2] = 1.0

        cortex.reset()

        # Run multiple timesteps to allow neurons to spike AND inhibit each other
        winner_spikes = torch.zeros(1, 5)
        for t in range(n_timesteps):
            output = cortex.forward(input_pattern)
            # Only count first spike (winner) not subsequent
            if t >= 5 and winner_spikes.sum() == 0:  # Wait for conductance buildup
                winner_spikes = output.clone()

        # Learn from winner only
        if winner_spikes.sum() > 0:
            cortex.learn(input_pattern, winner_spikes)

    final_weights = cortex.weights.detach()
    print(f"Final weights range: [{final_weights.min():.3f}, {final_weights.max():.3f}]")

    # Show learned structure
    print("\nLearned selectivity (which inputs each neuron prefers):")
    for neuron in range(5):
        preferred = final_weights[neuron].argmax().item()
        strength = final_weights[neuron].max().item()
        print(f"  Neuron {neuron}: prefers input {preferred} (strength={strength:.3f})")

    print("\n→ Cortex discovers correlational structure WITHOUT explicit teaching")


def demo_cerebellum():
    """Demonstrate supervised learning in Cerebellum."""
    print("\n" + "=" * 70)
    print("2. CEREBELLUM: Supervised Error-Corrective Learning")
    print("=" * 70)

    config = CerebellumConfig(
        n_input=10,
        n_output=5,
        learning_rate=0.3,  # Higher for faster learning
        device="cpu",
    )
    cerebellum = Cerebellum(config)

    print("\nCerebellum learns from explicit targets (error signals).")
    print("It can learn arbitrary mappings through the delta rule.")

    # Define an arbitrary (shuffled) mapping:
    # Input 0-1 → Neuron 3, Input 2-3 → Neuron 0, etc.
    target_mapping = {0: 3, 1: 0, 2: 4, 3: 1, 4: 2}

    print(f"\nTarget mapping (arbitrary): {target_mapping}")

    n_trials = 200  # More trials for learning
    n_timesteps = 15  # More timesteps to ensure spiking
    correct = 0

    for trial in range(n_trials):
        phase = trial % 5
        input_pattern = torch.zeros(1, 10)
        input_pattern[0, phase * 2:phase * 2 + 2] = 1.0

        target_neuron = target_mapping[phase]
        target = torch.zeros(1, 5)
        target[0, target_neuron] = 1.0

        cerebellum.reset()

        # Track accumulated spikes to find winner
        total_spikes = torch.zeros(1, 5)
        first_winner = None

        for t in range(n_timesteps):
            output = cerebellum.forward(input_pattern)
            total_spikes += output
            # Record first neuron to spike
            if first_winner is None and output.sum() > 0:
                first_winner = output.argmax().item()

        # Use first winner if any, else argmax of membrane
        if first_winner is not None:
            winner = first_winner
        elif cerebellum.neurons.membrane is not None:
            winner = cerebellum.neurons.membrane.argmax().item()
        else:
            winner = 0

        if winner == target_neuron:
            correct += 1

        # Learn using learn_from_phase for correct error handling
        cerebellum.learn_from_phase(
            cerebellum.input_trace, winner, target_neuron
        )

    print(f"\nAccuracy over {n_trials} trials: {correct}/{n_trials} = {correct / n_trials * 100:.1f}%")

    # Test final learned mapping
    print("\nFinal learned mapping:")
    for phase in range(5):
        input_pattern = torch.zeros(1, 10)
        input_pattern[0, phase * 2:phase * 2 + 2] = 1.0

        cerebellum.reset()

        # Track first spike
        first_winner = None
        for _ in range(n_timesteps):
            output = cerebellum.forward(input_pattern)
            if first_winner is None and output.sum() > 0:
                first_winner = int(output.argmax().item())

        # Use first winner or membrane argmax
        if first_winner is not None:
            winner = first_winner
        elif cerebellum.neurons.membrane is not None:
            winner = int(cerebellum.neurons.membrane.argmax().item())
        else:
            winner = 0

        target = target_mapping[phase]

        match = "✓" if winner == target else "✗"
        print(f"  Input {phase * 2}-{phase * 2 + 1} → Neuron {winner} (target: {target}) {match}")

    print("\n→ Cerebellum learns arbitrary mappings via explicit error signals")


def demo_striatum():
    """Demonstrate three-factor RL in Striatum."""
    print("\n" + "=" * 70)
    print("3. STRIATUM: Three-Factor Reinforcement Learning")
    print("=" * 70)

    config = StriatumConfig(
        n_input=10,
        n_output=4,  # 4 actions
        eligibility_tau_ms=200.0,  # Faster trace for demo
        three_factor_lr=0.3,  # Higher LR for faster learning
        device="cpu",
    )
    striatum = Striatum(config)

    print("\nStriatum learns from reward/punishment (dopamine).")
    print("It uses: Δw = eligibility × dopamine")

    # Simple bandit task: action 2 gives reward
    best_action = 2

    print(f"\nSecret: Action {best_action} gives reward (+1)")
    print("Striatum must discover this through trial and error.\n")

    n_trials = 100  # More trials for learning
    n_timesteps = 10  # Multiple timesteps per trial
    action_counts = {i: 0 for i in range(4)}
    rewards_per_10: list[float] = []
    rewards = 0.0

    for trial in range(n_trials):
        # Same input each time (context)
        input_pattern = torch.ones(1, 10) * 0.5

        striatum.reset()

        # Run timesteps to build eligibility and get neural activity
        accumulated_activity = torch.zeros(1, 4)
        for _ in range(n_timesteps):
            output = striatum.forward(input_pattern)
            # Use membrane potential for action values, not spikes
            if striatum.neurons.membrane is not None:
                accumulated_activity += striatum.neurons.membrane.detach()

        # Choose action based on accumulated membrane potential (not binary spikes)
        action_values = accumulated_activity.flatten()
        action_probs = torch.softmax(action_values * 2, dim=0)
        action = int(torch.multinomial(action_probs, 1).item())
        action_counts[action] += 1

        # Get reward
        reward = 1.0 if action == best_action else 0.0
        rewards += reward

        # Create one-hot output for the chosen action
        action_output = torch.zeros(1, 4)
        action_output[0, action] = 1.0

        # Learn from reward
        striatum.learn(input_pattern, action_output, reward=reward)

        if (trial + 1) % 10 == 0:
            rewards_per_10.append(rewards)
            rewards = 0.0

    print("Action selection frequency:")
    for action, count in action_counts.items():
        bar = "█" * (count // 2)
        marker = " ←BEST" if action == best_action else ""
        print(f"  Action {action}: {count:3d} {bar}{marker}")

    print(f"\nRewards per 10 trials: {rewards_per_10}")
    print("\n→ Striatum learns to select rewarding actions through trial-and-error")


def demo_hippocampus():
    """Demonstrate one-shot learning in Hippocampus."""
    print("\n" + "=" * 70)
    print("4. HIPPOCAMPUS: One-Shot Episodic Learning")
    print("=" * 70)

    config = HippocampusConfig(
        n_input=20,
        n_output=20,  # Auto-associative
        learning_rate=1.0,  # Very high for one-shot
        recurrent_strength=2.0,  # Strong recurrence
        sparsity_target=0.3,  # Allow more active neurons
        device="cpu",
    )
    hippocampus = Hippocampus(config)

    print("\nHippocampus can store patterns in ONE exposure.")
    print("It uses theta-phase gating for encoding vs retrieval.")

    # Create 3 distinct sparse patterns (non-overlapping)
    patterns: list[torch.Tensor] = []
    for i in range(3):
        p = torch.zeros(1, 20)
        p[0, i * 5:(i + 1) * 5] = 1.0
        patterns.append(p)

    print("\nStoring 3 patterns (one-shot each)...")
    for i, pattern in enumerate(patterns):
        hippocampus.reset_state(1)

        # Run multiple steps with learning
        n_steps = 20
        for _ in range(n_steps):
            output = hippocampus.forward(pattern)
            hippocampus.learn(pattern, pattern, force_encoding=True)  # Auto-associative

        # Check stored weight strength for this pattern's neurons
        pattern_neurons = list(range(i * 5, (i + 1) * 5))
        ff_strength = hippocampus.weights.data[pattern_neurons][:, pattern_neurons].mean().item()
        rec_strength = hippocampus.rec_weights.data[pattern_neurons][:, pattern_neurons].mean().item()
        print(f"  Pattern {i}: stored (ff={ff_strength:.2f}, rec={rec_strength:.2f})")

    # Test recall with partial cues
    print("\nRecalling with partial cues...")
    for i, pattern in enumerate(patterns):
        # Create partial cue (2 of 5 neurons)
        cue = torch.zeros(1, 20)
        cue[0, i * 5:i * 5 + 2] = 1.0

        hippocampus.reset_state(1)

        # Pattern completion via multiple iterations
        current = cue.clone()
        n_recall_steps = 30

        for step in range(n_recall_steps):
            # Forward with current activity
            output = hippocampus.forward(current)

            # Blend output with persistent cue for stability
            if output.sum() > 0:
                current = (output + cue).clamp(0, 1)  # Keep cue active
            else:
                # If no spikes, use membrane potential as activity
                if hippocampus.neurons.membrane is not None:
                    membrane = hippocampus.neurons.membrane
                    # Normalize membrane to 0-1 range
                    current = (membrane - membrane.min()) / (membrane.max() - membrane.min() + 1e-6)
                    current = (current + cue).clamp(0, 1)

        # Check overlap - use membrane potential if no spikes
        if current.sum() > 0:
            # Binarize at threshold
            final = (current > 0.3).float()
        else:
            final = torch.zeros_like(pattern)

        overlap = (final * pattern).sum() / pattern.sum()
        recovered = int(final.sum().item())
        print(f"  Pattern {i}: {overlap.item() * 100:.1f}% recovered ({recovered}/5 active)")

    print("\n→ Hippocampus stores episodes rapidly and recalls from partial cues")


def demo_prefrontal():
    """Demonstrate gated working memory in Prefrontal Cortex."""
    print("\n" + "=" * 70)
    print("5. PREFRONTAL: Gated Working Memory")
    print("=" * 70)

    config = PrefrontalConfig(
        n_input=10,
        n_output=10,
        wm_decay_tau_ms=500.0,  # Slower decay for visible maintenance
        gate_threshold=0.3,  # Lower threshold for easier gating
        recurrent_strength=0.5,  # Self-sustaining activity
        device="cpu",
    )
    pfc = Prefrontal(config)

    print("\nPrefrontal cortex maintains information in working memory.")
    print("Dopamine gates what enters/updates WM:")
    print("  - High DA → gate OPEN → update WM with new input")
    print("  - Low DA → gate CLOSED → maintain current WM")

    # Demo 1: Store pattern with high DA
    print("\n--- Storing Pattern with High Dopamine ---")
    pfc.reset_state(1)

    pattern = torch.zeros(1, 10)
    pattern[0, :5] = 1.0  # First 5 neurons active

    # Use set_context for direct WM loading (most reliable)
    pfc.set_context(pattern)
    print(f"WM after direct context set: {pfc.get_working_memory().mean().item():.3f}")

    # Also show gated update with high DA
    pfc.reset_state(1)
    n_store_steps = 20
    for _ in range(n_store_steps):
        pfc.forward(pattern, dopamine_signal=0.8)  # High DA opens gate

    wm_activity = pfc.get_working_memory()
    pattern_match = (wm_activity * pattern).sum() / pattern.sum()
    print(f"WM after {n_store_steps} gated updates: {wm_activity.mean().item():.3f} (pattern match: {pattern_match.item():.2f})")

    # Demo 2: Maintenance with low DA
    print("\n--- Maintaining with Low Dopamine ---")

    # Store a pattern first
    pfc.set_context(pattern * 0.5)  # Store at 50% strength
    initial_wm = pfc.get_working_memory().clone()
    initial_activity = initial_wm.mean().item()

    # Run maintenance (no input, low DA)
    metrics = pfc.maintain(n_steps=10, dt=1.0)

    print(f"Initial activity: {initial_activity:.3f}")
    print(f"Retention after 10 steps: {metrics['retention'] * 100:.1f}%")
    print(f"Final activity: {metrics['final_activity']:.3f}")

    # Demo 3: Distractor rejection
    print("\n--- Distractor Rejection (Low DA) ---")
    pfc.reset_state(1)

    # First store pattern A directly
    pattern_a = torch.zeros(1, 10)
    pattern_a[0, :5] = 1.0
    pfc.set_context(pattern_a * 0.8)  # Store at 80% strength

    wm_after_a = pfc.get_working_memory().clone()
    print(f"WM after storing pattern A: {wm_after_a.mean().item():.3f}")

    # Now present distractor B with LOW DA (should be rejected)
    pattern_b = torch.zeros(1, 10)
    pattern_b[0, 5:] = 1.0  # Different pattern (neurons 5-9)
    for _ in range(10):
        pfc.forward(pattern_b, dopamine_signal=-0.5)  # Low DA = gate closed

    wm_after_b = pfc.get_working_memory()

    # Check if A is still in WM (not replaced by B)
    overlap_a = torch.nn.functional.cosine_similarity(
        wm_after_a.flatten(), wm_after_b.flatten(), dim=0
    ).item()

    # Check if B intruded
    overlap_b = (wm_after_b * pattern_b).sum() / pattern_b.sum()

    print(f"Pattern A retained: {overlap_a * 100:.1f}% similarity")
    print(f"Pattern B intrusion: {overlap_b.item() * 100:.1f}%")

    print("\n→ Prefrontal cortex uses dopamine gating for selective working memory")


def demo_comparison():
    """Show the key differences between all regions."""
    print("\n" + "=" * 70)
    print("SUMMARY: Why Different Brain Regions Exist")
    print("=" * 70)

    print("""
┌─────────────┬──────────────────────┬────────────────────────────────────┐
│   REGION    │    LEARNING RULE     │           BEST FOR                 │
├─────────────┼──────────────────────┼────────────────────────────────────┤
│ CORTEX      │ Unsupervised Hebbian │ Feature extraction, clustering     │
│             │ + BCM homeostasis    │ Finding structure in data          │
├─────────────┼──────────────────────┼────────────────────────────────────┤
│ CEREBELLUM  │ Supervised Delta     │ Precise input→output mapping       │
│             │ (climbing fibers)    │ Motor control, timing              │
├─────────────┼──────────────────────┼────────────────────────────────────┤
│ STRIATUM    │ Three-Factor RL      │ Action selection, habits           │
│             │ (eligibility × DA)   │ Learning from reward               │
├─────────────┼──────────────────────┼────────────────────────────────────┤
│ HIPPOCAMPUS │ One-Shot Hebbian     │ Episodic memory, sequences         │
│             │ + theta gating       │ Rapid memorization                 │
├─────────────┼──────────────────────┼────────────────────────────────────┤
│ PREFRONTAL  │ Gated Hebbian        │ Working memory, rules              │
│             │ (DA gates WM)        │ Executive control                  │
└─────────────┴──────────────────────┴────────────────────────────────────┘

KEY INSIGHT: The brain uses DIFFERENT algorithms for DIFFERENT problems!
- No single learning rule is optimal for everything
- Each region evolved to solve specific computational challenges
- This is why we need specialized architectures

WHEN TO USE EACH:
  • Cortex: "I have unlabeled data and want to find patterns"
  • Cerebellum: "I have input-output pairs and want precise mapping"
  • Striatum: "I have rewards/punishments and want to learn actions"
  • Hippocampus: "I need to remember things quickly (few exposures)"
  • Prefrontal: "I need to hold and manipulate information over time"
""")


def main():
    """Run all demos."""
    print("\n" + "█" * 70)
    print("THALIA BRAIN REGIONS DEMO")
    print("Demonstrating 5 Specialized Learning Algorithms")
    print("█" * 70)

    demo_cortex()
    demo_cerebellum()
    demo_striatum()
    demo_hippocampus()
    demo_prefrontal()
    demo_comparison()

    print("\n" + "█" * 70)
    print("Demo complete! Each region learns differently because")
    print("different problems require different algorithms.")
    print("█" * 70 + "\n")


if __name__ == "__main__":
    main()
