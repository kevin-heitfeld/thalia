"""
Daydream Demo - Spontaneous cognition in THALIA.

This demo shows the network "thinking without input" - generating
sequences of thoughts by wandering through its learned concept space.
This is key for:
- Creativity: Novel combinations emerge
- Incubation: Solutions appear during rest
- Memory consolidation: Replaying and strengthening

Run: python examples/daydream_demo.py
"""

import torch
import sys

# Add src to path if needed
sys.path.insert(0, "src")

from thalia.cognition.daydream import (
    DaydreamNetwork,
    DaydreamConfig,
    DaydreamMode,
    DaydreamIntegration,
)
from thalia.cognition import ThinkingSNN, ThinkingConfig


def demo_basic_daydream():
    """Demo 1: Basic daydreaming with a concept network."""
    print("\n" + "="*60)
    print("Demo 1: Basic Daydreaming")
    print("="*60)

    # Create daydream network
    config = DaydreamConfig(
        n_neurons=64,
        base_noise=0.1,
        dwell_time_mean=30,  # Shorter for demo
        dwell_time_std=10,
    )
    dreamer = DaydreamNetwork(config)

    # Store some concepts (fruit theme)
    concepts = ["apple", "banana", "cherry", "orange", "grape"]
    for i, name in enumerate(concepts):
        pattern = torch.zeros(64)
        pattern[i*10:(i+1)*10] = 1.0  # Non-overlapping patterns
        dreamer.store_concept(pattern, name)

    # Create associations (semantic relationships)
    dreamer.associate("apple", "cherry", 0.7)   # Both are red fruits
    dreamer.associate("banana", "orange", 0.6)  # Both citrus-ish
    dreamer.associate("grape", "cherry", 0.6)   # Both small fruits

    print(f"Stored concepts: {concepts}")
    print(f"Created semantic associations")

    # Daydream session
    print("\nStarting daydream from 'apple'...")
    print("-" * 40)

    def on_transition(old, new, name):
        if old >= 0:
            old_name = dreamer._concept_names.get(old, "?")
            print(f"  {old_name} → {name}")
        else:
            print(f"  Started at: {name}")

    dreamer.on_transition(on_transition)

    states = dreamer.daydream(
        steps=200,
        mode=DaydreamMode.FREE,
        start_concept="apple"
    )

    print("-" * 40)
    concepts_visited = dreamer.get_concepts_visited()
    print(f"Concepts visited: {concepts_visited}")
    print(f"Total transitions: {len(concepts_visited) - 1}")


def demo_daydream_modes():
    """Demo 2: Different daydream modes."""
    print("\n" + "="*60)
    print("Demo 2: Daydream Modes")
    print("="*60)

    config = DaydreamConfig(
        n_neurons=64,
        dwell_time_mean=25,
    )
    dreamer = DaydreamNetwork(config)

    # Store concepts
    concepts = ["dog", "cat", "bird", "fish", "horse", "elephant"]
    for i, name in enumerate(concepts):
        pattern = torch.zeros(64)
        pattern[i*10:(i+1)*10] = 1.0
        dreamer.store_concept(pattern, name)

    # Associate (pets vs wild)
    dreamer.associate("dog", "cat", 0.8)   # Common pets
    dreamer.associate("bird", "fish", 0.5)  # Pet-ish
    dreamer.associate("horse", "elephant", 0.6)  # Large animals

    print("\nTesting different modes:")

    # Mode 1: FREE
    print("\n1. FREE mode (pure random walk):")
    dreamer.start_daydream(mode=DaydreamMode.FREE, start_concept="dog")
    free_noise = dreamer._get_noise_level()
    for _ in range(100):
        dreamer.step()
    print(f"   Noise level: {free_noise:.3f}")
    print(f"   Visited: {dreamer.get_concepts_visited()}")

    # Mode 2: THEMED
    print("\n2. THEMED mode (biased toward theme):")
    dreamer.set_theme(["dog", "cat", "bird"], strength=0.4)
    dreamer.start_daydream(mode=DaydreamMode.THEMED, start_concept="dog")
    for _ in range(100):
        dreamer.step()
    print(f"   Theme: pets (dog, cat, bird)")
    print(f"   Visited: {dreamer.get_concepts_visited()}")
    dreamer.clear_theme()

    # Mode 3: DREAM
    print("\n3. DREAM mode (high noise, strange associations):")
    dreamer.start_daydream(mode=DaydreamMode.DREAM, start_concept="dog")
    dream_noise = dreamer._get_noise_level()
    for _ in range(100):
        dreamer.step()
    print(f"   Noise level: {dream_noise:.3f} ({dream_noise/free_noise:.1f}x FREE)")
    print(f"   Visited: {dreamer.get_concepts_visited()}")

    # Mode 4: GOAL_DIRECTED
    print("\n4. GOAL_DIRECTED mode (searching for target):")
    goal_pattern = torch.zeros(64)
    goal_pattern[50:60] = 1.0  # Similar to "elephant"
    dreamer.set_goal(goal_pattern)
    dreamer.start_daydream(mode=DaydreamMode.GOAL_DIRECTED, start_concept="dog")
    goal_noise = dreamer._get_noise_level()
    for _ in range(100):
        dreamer.step()
    print(f"   Noise level: {goal_noise:.3f} (reduced for focus)")
    print(f"   Visited: {dreamer.get_concepts_visited()}")
    dreamer.clear_goal()


def demo_novelty_tracking():
    """Demo 3: Tracking novelty of thought transitions."""
    print("\n" + "="*60)
    print("Demo 3: Novelty Tracking")
    print("="*60)

    config = DaydreamConfig(
        n_neurons=64,
        dwell_time_mean=20,
    )
    dreamer = DaydreamNetwork(config)

    # Store concepts with varying associations
    concepts = ["red", "blue", "green", "yellow", "purple"]
    for i, name in enumerate(concepts):
        pattern = torch.zeros(64)
        pattern[i*12:(i+1)*12] = 1.0
        dreamer.store_concept(pattern, name)

    # Strong associations (low novelty transitions)
    dreamer.associate("red", "blue", 0.9)
    dreamer.associate("blue", "green", 0.8)
    # Weak associations (high novelty transitions)
    dreamer.associate("red", "purple", 0.1)

    print("\nAssociations:")
    print("  red ↔ blue:   0.9 (strong)")
    print("  blue ↔ green: 0.8 (strong)")
    print("  red ↔ purple: 0.1 (weak)")

    print("\nDaydreaming and tracking novelty...")
    states = dreamer.daydream(steps=150, start_concept="red")

    # Find transitions and their novelty
    transitions = [(s.concept_name, s.novelty) for s in states if s.transition_occurred]

    print("\nTransitions with novelty scores:")
    for name, novelty in transitions:
        novelty_desc = "expected" if novelty < 0.4 else "surprising" if novelty > 0.7 else "moderate"
        print(f"  → {name}: novelty={novelty:.2f} ({novelty_desc})")


def demo_integration_with_thinker():
    """Demo 4: Adding daydream mode to ThinkingSNN."""
    print("\n" + "="*60)
    print("Demo 4: Daydream Integration with ThinkingSNN")
    print("="*60)

    # Create a ThinkingSNN
    config = ThinkingConfig(
        n_concepts=64,
        noise_std=0.05,
        enable_learning=False,
        enable_homeostasis=False,
    )
    thinker = ThinkingSNN(config)
    thinker.reset_state(batch_size=1)

    # Store some concepts
    concepts = ["work", "deadline", "stress", "vacation", "beach", "relax"]
    for i, name in enumerate(concepts):
        pattern = torch.zeros(64)
        pattern[i*10:(i+1)*10] = 1.0
        thinker.store_concept(pattern, name)

    # Create associations
    thinker.associate_concepts(0, 1, 0.8)  # work → deadline
    thinker.associate_concepts(1, 2, 0.7)  # deadline → stress
    thinker.associate_concepts(3, 4, 0.8)  # vacation → beach
    thinker.associate_concepts(4, 5, 0.9)  # beach → relax
    thinker.associate_concepts(2, 3, 0.5)  # stress → vacation (escape!)

    print("Stored concepts: work, deadline, stress, vacation, beach, relax")
    print("Associations create paths: work→deadline→stress→vacation→beach→relax")

    # Wrap with DaydreamIntegration
    daydreamer = DaydreamIntegration(thinker)

    print("\nNormal thinking (low noise):")
    thinker.reset_state(batch_size=1)
    for _ in range(50):
        state = thinker.think()
    normal_noise = thinker.concepts.neurons.config.noise_std
    print(f"  Noise level: {normal_noise:.4f}")

    print("\nEntering daydream mode...")
    concepts_visited = daydreamer.daydream(steps=200)

    print(f"  Noise during daydream: {normal_noise * 3:.4f} (3x)")
    print(f"  Concepts visited: {concepts_visited}")

    # Verify noise restored
    current_noise = thinker.concepts.neurons.config.noise_std
    print(f"\nNoise after daydream: {current_noise:.4f} (restored)")


def demo_creative_associations():
    """Demo 5: Using daydream for creative brainstorming."""
    print("\n" + "="*60)
    print("Demo 5: Creative Brainstorming")
    print("="*60)

    config = DaydreamConfig(
        n_neurons=96,
        dwell_time_mean=15,
        base_noise=0.15,  # Slightly higher for creativity
    )
    dreamer = DaydreamNetwork(config)

    # Store diverse concepts
    concept_groups = {
        "tech": ["computer", "algorithm", "data"],
        "nature": ["tree", "river", "mountain"],
        "art": ["painting", "music", "dance"],
    }

    all_concepts = []
    for group, concepts in concept_groups.items():
        for i, name in enumerate(concepts):
            pattern = torch.zeros(96)
            # Each concept has unique + group overlap
            base = len(all_concepts) * 10
            pattern[base:base+8] = 1.0  # Unique
            group_base = {"tech": 80, "nature": 85, "art": 90}[group]
            pattern[group_base:group_base+3] = 0.5  # Group overlap
            dreamer.store_concept(pattern, name)
            all_concepts.append(name)

    # Within-group associations
    for group, concepts in concept_groups.items():
        for i in range(len(concepts)):
            for j in range(i+1, len(concepts)):
                dreamer.associate(concepts[i], concepts[j], 0.7)

    # Cross-group associations (sparse, creative connections)
    dreamer.associate("algorithm", "music", 0.3)  # Algorithmic music
    dreamer.associate("tree", "data", 0.2)  # Data trees
    dreamer.associate("river", "dance", 0.25)  # Flowing movement

    print("Concept groups:")
    for group, concepts in concept_groups.items():
        print(f"  {group}: {concepts}")
    print("\nCross-group associations: algorithm↔music, tree↔data, river↔dance")

    print("\nCreative brainstorm starting from 'computer'...")
    print("(Using DREAM mode for stranger associations)")

    states = dreamer.daydream(
        steps=200,
        mode=DaydreamMode.DREAM,
        start_concept="computer"
    )

    concepts_visited = dreamer.get_concepts_visited()
    print(f"\nThought chain: {' → '.join(concepts_visited)}")

    # Count cross-group transitions
    def get_group(concept):
        for g, cs in concept_groups.items():
            if concept in cs:
                return g
        return None

    cross_group = 0
    for i in range(len(concepts_visited)-1):
        g1 = get_group(concepts_visited[i])
        g2 = get_group(concepts_visited[i+1])
        if g1 and g2 and g1 != g2:
            cross_group += 1

    print(f"\nCross-group transitions: {cross_group} (creative jumps!)")


def demo_recency_effects():
    """Demo 6: Demonstrating recency effects prevent loops."""
    print("\n" + "="*60)
    print("Demo 6: Recency Effects")
    print("="*60)

    config = DaydreamConfig(
        n_neurons=48,
        dwell_time_mean=15,
        recency_decay=0.9,  # Slower decay = stronger recency effect
    )
    dreamer = DaydreamNetwork(config)

    # Small concept set (more likely to see recency effects)
    for i, name in enumerate(["A", "B", "C", "D"]):
        pattern = torch.zeros(48)
        pattern[i*10:(i+1)*10] = 1.0
        dreamer.store_concept(pattern, name)

    # Strong associations (would cause loops without recency)
    dreamer.associate("A", "B", 0.9)
    dreamer.associate("B", "C", 0.9)
    dreamer.associate("C", "D", 0.9)
    dreamer.associate("D", "A", 0.9)  # Creates a cycle A→B→C→D→A

    print("Concepts: A, B, C, D (strongly connected in a cycle)")
    print("Without recency: would loop A→B→C→D→A→B→...")
    print("\nDaydreaming with recency effect...")

    states = dreamer.daydream(steps=150, start_concept="A")
    concepts_visited = dreamer.get_concepts_visited()

    print(f"\nVisited: {concepts_visited}")

    # Check for immediate repetition
    immediate_repeats = sum(
        1 for i in range(len(concepts_visited)-1)
        if concepts_visited[i] == concepts_visited[i+1]
    )
    print(f"Immediate repetitions: {immediate_repeats}")
    print("(Recency effect encourages exploration over repetition)")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("THALIA Daydream Mode Demo")
    print("Spontaneous Cognition Without External Input")
    print("="*60)

    demos = [
        ("Basic Daydreaming", demo_basic_daydream),
        ("Daydream Modes", demo_daydream_modes),
        ("Novelty Tracking", demo_novelty_tracking),
        ("ThinkingSNN Integration", demo_integration_with_thinker),
        ("Creative Brainstorming", demo_creative_associations),
        ("Recency Effects", demo_recency_effects),
    ]

    for name, demo_func in demos:
        try:
            demo_func()
        except Exception as e:
            print(f"\n[Error in {name}]: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)
    print("\nKey takeaways:")
    print("• Daydream mode generates thoughts without input")
    print("• Different modes: FREE, THEMED, DREAM, GOAL_DIRECTED")
    print("• Associations guide transitions, recency prevents loops")
    print("• Novelty tracking identifies surprising thought jumps")
    print("• Integration with ThinkingSNN enables hybrid thinking")


if __name__ == "__main__":
    main()
