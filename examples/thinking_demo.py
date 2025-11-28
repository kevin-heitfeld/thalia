#!/usr/bin/env python3
"""
Thinking SNN Demo

Demonstrates the ThinkingSNN - the integrated thinking system that combines:
- Attractor-based concept storage
- Working memory with limited slots
- STDP and reward-modulated learning
- Homeostatic plasticity
- Activity tracking and trajectory analysis

This is the core of THALIA: a network that can genuinely "think" by
navigating between concepts through attractor dynamics.
"""

import torch
from thalia.cognition import ThinkingSNN, ThinkingConfig

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print()


def demo_basic_thinking():
    """Demonstrate basic thinking with concept storage and navigation."""
    print("=" * 60)
    print("Demo 1: Basic Thinking with Concepts")
    print("=" * 60)

    # Create a thinker with 64 concept neurons and 4 memory slots
    config = ThinkingConfig(
        n_concepts=64,
        n_wm_slots=4,
        wm_slot_size=16,
        enable_learning=False,  # Start simple
        noise_std=0.05,
    )
    thinker = ThinkingSNN(config)
    thinker.to(device)

    # Store some concepts
    print("\n--- Storing Concepts ---")

    # Create concept patterns (sparse activation patterns)
    concept_indices = {}
    for name in ["cat", "dog", "car", "bicycle"]:
        pattern = torch.zeros(64, device=device)
        # Each concept activates a random subset of neurons
        active_indices = torch.randperm(64, device=device)[:16]
        pattern[active_indices] = 1.0
        idx = thinker.store_concept(pattern, name)
        concept_indices[name] = idx
        print(f"  Stored concept: '{name}' at index {idx}")

    # Associate related concepts
    print("\n--- Associating Concepts ---")
    thinker.associate_concepts(concept_indices["cat"], concept_indices["dog"], strength=0.8)
    thinker.associate_concepts(concept_indices["car"], concept_indices["bicycle"], strength=0.8)
    print("  Associated: cat <-> dog (pets)")
    print("  Associated: car <-> bicycle (vehicles)")

    # Think for a while, starting from 'cat'
    print("\n--- Thinking Process ---")
    thinker.reset_state()
    cat_pattern = thinker.concepts.patterns[concept_indices["cat"]].unsqueeze(0)
    thinker.attend_to(cat_pattern)

    for step in range(50):
        state = thinker.think()
        if step % 10 == 0:
            print(f"  Step {step:3d}: concept={state.concept_name or 'none':10s} "
                  f"energy={state.energy:.4f} active_neurons={state.spikes.sum().item():.0f}")

    # Show thought trajectory
    print("\n--- Thought Trajectory ---")
    trajectory = thinker.get_trajectory()
    print(f"  States visited: {trajectory.get_sequence()}")
    print(f"  Transitions: {trajectory.get_transitions()}")
    if trajectory.mean_dwell_time() > 0:
        print(f"  Mean dwell time: {trajectory.mean_dwell_time():.1f} steps")


def demo_working_memory():
    """Demonstrate working memory operations."""
    print("\n" + "=" * 60)
    print("Demo 2: Working Memory Operations")
    print("=" * 60)

    config = ThinkingConfig(
        n_concepts=64,
        n_wm_slots=4,
        wm_slot_size=16,
        enable_learning=False,
    )
    thinker = ThinkingSNN(config)
    thinker.to(device)

    # Store concepts
    concepts = ["apple", "banana", "cherry"]
    concept_indices = {}
    patterns = {}
    for name in concepts:
        pattern = torch.zeros(64, device=device)
        pattern[torch.randperm(64, device=device)[:16]] = 1.0
        idx = thinker.store_concept(pattern, name)
        concept_indices[name] = idx
        patterns[name] = pattern

    print("\n--- Loading to Working Memory ---")

    # Load patterns into memory slots
    for i, name in enumerate(concepts):
        # Create a smaller pattern for working memory (wm_slot_size=16)
        wm_pattern = torch.zeros(16, device=device)
        wm_pattern[torch.randperm(16, device=device)[:6]] = 1.0
        thinker.load_to_memory(i, wm_pattern.unsqueeze(0), name)
        print(f"  Loaded '{name}' to slot {i}")

    # Check memory status
    print("\n--- Memory Status ---")
    status = thinker.working_memory.get_status()
    for slot_status in status["slots"]:
        print(f"  Slot {slot_status['index']}: active={slot_status['active']}, "
              f"label={slot_status['label']}")

    # Read from memory
    print("\n--- Reading from Memory ---")
    content = thinker.read_from_memory(0)
    if content is not None:
        print(f"  Slot 0 content shape: {content.shape}")
        print(f"  Non-zero elements: {(content > 0.1).sum().item()}")


def demo_learning():
    """Demonstrate learning during thinking."""
    print("\n" + "=" * 60)
    print("Demo 3: Learning During Thinking")
    print("=" * 60)

    config = ThinkingConfig(
        n_concepts=64,
        n_wm_slots=4,
        wm_slot_size=16,
        enable_learning=True,
        enable_homeostasis=True,
    )
    thinker = ThinkingSNN(config)
    thinker.to(device)

    # Store concepts
    pattern_a = torch.zeros(64, device=device)
    pattern_a[:32] = 1.0  # First half active

    pattern_b = torch.zeros(64, device=device)
    pattern_b[32:] = 1.0  # Second half active

    idx_a = thinker.store_concept(pattern_a, "A")
    idx_b = thinker.store_concept(pattern_b, "B")

    # Get initial weight statistics
    initial_weights = thinker.concepts.weights.clone()

    print("\n--- Thinking with Learning ---")
    thinker.reset_state()
    thinker.attend_to(pattern_a.unsqueeze(0))

    for step in range(100):
        state = thinker.think()
        if step % 25 == 0:
            print(f"  Step {step:3d}: concept={state.concept_name or 'none':5s} "
                  f"energy={state.energy:.4f}")

    # Compare weights
    weight_change = (thinker.concepts.weights - initial_weights).abs().mean().item()
    print(f"\n  Mean absolute weight change: {weight_change:.6f}")

    # Give reward signal
    print("\n--- Applying Reward Signal ---")
    weights_before_reward = thinker.concepts.weights.clone()
    thinker.set_goal(1.0)  # Use set_goal to set reward signal

    # The reward is applied on next think step
    thinker.think()

    reward_weight_change = (thinker.concepts.weights - weights_before_reward).abs().mean().item()
    print(f"  Weight change from reward: {reward_weight_change:.6f}")


def demo_thought_chain():
    """Demonstrate generating a chain of thoughts."""
    print("\n" + "=" * 60)
    print("Demo 4: Thought Chain Generation")
    print("=" * 60)

    config = ThinkingConfig(
        n_concepts=64,
        n_wm_slots=4,
        wm_slot_size=16,
        enable_learning=False,
        noise_std=0.1,  # More noise for exploration
    )
    thinker = ThinkingSNN(config)
    thinker.to(device)

    # Create an associative network of concepts
    concepts = ["morning", "coffee", "work", "lunch", "meeting", "evening"]
    concept_indices = {}
    for name in concepts:
        pattern = torch.zeros(64, device=device)
        pattern[torch.randperm(64, device=device)[:16]] = 1.0
        idx = thinker.store_concept(pattern, name)
        concept_indices[name] = idx

    # Create temporal associations (daily routine)
    associations = [
        ("morning", "coffee"),
        ("coffee", "work"),
        ("work", "lunch"),
        ("lunch", "meeting"),
        ("meeting", "evening"),
    ]

    for a, b in associations:
        thinker.associate_concepts(concept_indices[a], concept_indices[b], strength=0.7)

    print("\n--- Generating Thought Chain ---")
    thinker.reset_state()
    chain = thinker.generate_thought_chain(
        steps=200,
        start_concept=concept_indices["morning"],
    )

    print(f"  Thought chain: {' -> '.join(chain) if chain else '(empty)'}")
    print(f"  Chain length: {len(chain)}")


def demo_activity_projection():
    """Demonstrate activity tracking and dimensionality reduction."""
    print("\n" + "=" * 60)
    print("Demo 5: Activity Projection (PCA)")
    print("=" * 60)

    config = ThinkingConfig(
        n_concepts=64,
        n_wm_slots=4,
        wm_slot_size=16,
        enable_learning=False,
        noise_std=0.05,
    )
    thinker = ThinkingSNN(config)
    thinker.to(device)

    # Store a few concepts
    concept_indices = {}
    for i, name in enumerate(["alpha", "beta", "gamma"]):
        pattern = torch.zeros(64, device=device)
        start = i * 20
        pattern[start:start+20] = 1.0  # Each concept uses different neurons
        idx = thinker.store_concept(pattern, name)
        concept_indices[name] = idx

    # Associate them in a cycle
    thinker.associate_concepts(concept_indices["alpha"], concept_indices["beta"], strength=0.6)
    thinker.associate_concepts(concept_indices["beta"], concept_indices["gamma"], strength=0.6)
    thinker.associate_concepts(concept_indices["gamma"], concept_indices["alpha"], strength=0.6)

    # Think and track
    print("\n--- Recording Activity ---")
    thinker.reset_state()
    alpha_pattern = thinker.concepts.patterns[concept_indices["alpha"]].unsqueeze(0)
    thinker.attend_to(alpha_pattern)

    for _ in range(100):
        thinker.think()

    # Project to 2D
    print("\n--- Projecting to 2D ---")
    result = thinker.project_activity(n_components=2)

    if result is not None:
        projected, _ = result
        print(f"  Projected shape: {projected.shape}")
        print(f"  First 5 points:")
        for i in range(min(5, projected.shape[0])):
            print(f"    Point {i}: ({projected[i, 0]:.4f}, {projected[i, 1]:.4f})")
    else:
        print("  Not enough data for projection")

    # Get activity history
    history = thinker.get_activity_history()
    print(f"\n  Full activity history shape: {history.shape}")


def demo_goals():
    """Demonstrate goal-directed thinking."""
    print("\n" + "=" * 60)
    print("Demo 6: Goal-Directed Thinking")
    print("=" * 60)

    config = ThinkingConfig(
        n_concepts=64,
        n_wm_slots=4,
        wm_slot_size=16,
        enable_learning=False,
        noise_std=0.03,
    )
    thinker = ThinkingSNN(config)
    thinker.to(device)

    # Create concepts
    concepts = ["start", "intermediate", "goal"]
    concept_indices = {}
    for name in concepts:
        pattern = torch.zeros(64, device=device)
        pattern[torch.randperm(64, device=device)[:20]] = 1.0
        idx = thinker.store_concept(pattern, name)
        concept_indices[name] = idx

    # Create a path
    thinker.associate_concepts(concept_indices["start"], concept_indices["intermediate"], strength=0.7)
    thinker.associate_concepts(concept_indices["intermediate"], concept_indices["goal"], strength=0.7)

    print("\n--- Setting Goal ---")
    thinker.set_goal(0.5)  # Positive reward signal
    print("  Goal signal set to 0.5")

    print("\n--- Thinking Toward Goal ---")
    thinker.reset_state()
    start_pattern = thinker.concepts.patterns[concept_indices["start"]].unsqueeze(0)
    thinker.attend_to(start_pattern)

    goal_reached = False
    for step in range(50):
        state = thinker.think()
        if step % 10 == 0:
            print(f"  Step {step:3d}: concept={state.concept_name or 'none':15s}")
        if state.concept_name == "goal":
            print(f"\n  Goal reached at step {step}!")
            goal_reached = True
            break

    if not goal_reached:
        print("\n  Goal not reached in 50 steps (try adjusting parameters)")


def demo_concept_change_callback():
    """Demonstrate callback on concept changes."""
    print("\n" + "=" * 60)
    print("Demo 7: Concept Change Callbacks")
    print("=" * 60)

    config = ThinkingConfig(
        n_concepts=64,
        n_wm_slots=4,
        wm_slot_size=16,
        enable_learning=False,
        noise_std=0.05,
    )
    thinker = ThinkingSNN(config)
    thinker.to(device)

    # Track concept changes
    transitions = []

    def on_change(concept_idx, concept_name):
        transitions.append((concept_idx, concept_name))
        print(f"  Changed to concept {concept_idx}: '{concept_name}'")

    thinker.on_concept_change(on_change)

    # Store and associate concepts
    concept_indices = {}
    for name in ["red", "blue", "green"]:
        pattern = torch.zeros(64, device=device)
        pattern[torch.randperm(64, device=device)[:18]] = 1.0
        idx = thinker.store_concept(pattern, name)
        concept_indices[name] = idx

    thinker.associate_concepts(concept_indices["red"], concept_indices["blue"], strength=0.5)
    thinker.associate_concepts(concept_indices["blue"], concept_indices["green"], strength=0.5)
    thinker.associate_concepts(concept_indices["green"], concept_indices["red"], strength=0.5)

    print("\n--- Watching Concept Transitions ---")
    thinker.reset_state()
    red_pattern = thinker.concepts.patterns[concept_indices["red"]].unsqueeze(0)
    thinker.attend_to(red_pattern)

    for _ in range(80):
        thinker.think()

    print(f"\n  Total transitions recorded: {len(transitions)}")


def demo_think_until_stable():
    """Demonstrate thinking until stable."""
    print("\n" + "=" * 60)
    print("Demo 8: Think Until Stable")
    print("=" * 60)

    config = ThinkingConfig(
        n_concepts=64,
        n_wm_slots=4,
        wm_slot_size=16,
        enable_learning=False,
        noise_std=0.02,  # Lower noise for stability
    )
    thinker = ThinkingSNN(config)
    thinker.to(device)

    # Store one concept
    pattern = torch.zeros(64, device=device)
    pattern[:30] = 1.0
    thinker.store_concept(pattern, "stable")

    print("\n--- Thinking Until Stable ---")
    thinker.reset_state()
    thinker.attend_to(pattern.unsqueeze(0))

    steps_taken = thinker.think_until_stable(max_steps=200, stability_window=30)
    print(f"  Stabilized after {steps_taken} steps")

    # Get final state
    trajectory = thinker.get_trajectory()
    print(f"  Final trajectory: {trajectory.get_sequence()}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("   THALIA - Thinking Architecture Demo")
    print("   Integrated SNN with Concepts, Memory, and Learning")
    print("=" * 60)

    demo_basic_thinking()
    demo_working_memory()
    demo_learning()
    demo_thought_chain()
    demo_activity_projection()
    demo_goals()
    demo_concept_change_callback()
    demo_think_until_stable()

    print("\n" + "=" * 60)
    print("   Demo Complete!")
    print("=" * 60)
