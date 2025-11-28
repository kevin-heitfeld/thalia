"""
Attractor Network Demo

Demonstrates pattern storage, recall, and concept associations in
a spiking attractor network. This is the foundation for "thought"
in THALIA - concepts are stored as stable attractor states.
"""

import torch
from thalia.dynamics import AttractorNetwork, AttractorConfig, ActivityTracker, ThoughtTrajectory
from thalia.dynamics.attractor import ConceptNetwork


def demo_pattern_storage():
    """Demonstrate storing and recalling patterns."""
    print("=" * 60)
    print("Pattern Storage and Recall Demo")
    print("=" * 60)
    
    # Create network
    config = AttractorConfig(
        n_neurons=100,
        noise_std=0.01,  # Low noise for stable attractors
        sparsity=0.1,    # 10% of neurons active per pattern
    )
    network = AttractorNetwork(config)
    
    print(f"Created attractor network with {config.n_neurons} neurons")
    
    # Create sparse patterns (10% of neurons active)
    torch.manual_seed(42)
    patterns = []
    for i in range(3):
        pattern = (torch.rand(100) < 0.1).float()
        patterns.append(pattern)
        network.store_pattern(pattern)
        print(f"Stored pattern {i}: {int(pattern.sum())} active neurons")
    
    print(f"\nTotal patterns stored: {len(network.patterns)}")
    
    # Test recall with partial cue
    print("\n--- Pattern Completion Test ---")
    cue = patterns[0].clone()
    cue[50:] = 0  # Only first half of pattern
    print(f"Cue has {int(cue.sum())} active neurons (partial pattern)")
    
    recalled = network.recall(cue, steps=100, cue_strength=2.0)
    
    # Check similarity to stored patterns
    similarity = network.similarity_to_patterns(recalled)
    print(f"Similarity to stored patterns: {similarity.tolist()}")
    print(f"Best match: Pattern {similarity.argmax().item()}")
    
    # Energy at recalled state
    energy = network.energy(recalled.squeeze())
    print(f"Energy at recalled state: {energy.item():.4f}")


def demo_concept_network():
    """Demonstrate named concepts and associations."""
    print("\n" + "=" * 60)
    print("Concept Network Demo")
    print("=" * 60)
    
    # Create concept network
    config = AttractorConfig(n_neurons=100, noise_std=0.02)
    network = ConceptNetwork(config)
    
    # Create concept patterns
    torch.manual_seed(123)
    
    # Each concept has a distinct sparse pattern
    concepts = {
        "apple": (torch.rand(100) < 0.1).float(),
        "red": (torch.rand(100) < 0.1).float(),
        "fruit": (torch.rand(100) < 0.1).float(),
        "banana": (torch.rand(100) < 0.1).float(),
        "yellow": (torch.rand(100) < 0.1).float(),
    }
    
    # Store concepts
    indices = {}
    for name, pattern in concepts.items():
        idx = network.store_concept(pattern, name)
        indices[name] = idx
        print(f"Stored concept '{name}' at index {idx}")
    
    # Create semantic associations
    print("\n--- Creating Associations ---")
    
    # Apple is associated with red and fruit
    network.associate(indices["apple"], indices["red"], strength=1.0)
    network.associate(indices["apple"], indices["fruit"], strength=1.0)
    print("Associated: apple <-> red, apple <-> fruit")
    
    # Banana is associated with yellow and fruit
    network.associate(indices["banana"], indices["yellow"], strength=1.0)
    network.associate(indices["banana"], indices["fruit"], strength=1.0)
    print("Associated: banana <-> yellow, banana <-> fruit")
    
    print(f"\nTotal associations: {len(network.associations) // 2}")
    
    # Test concept recall
    print("\n--- Concept Recall ---")
    
    # Activate "apple" and see what it recalls
    apple_pattern = concepts["apple"]
    network.recall(apple_pattern, steps=50)
    
    idx, name = network.active_concept()
    print(f"After activating 'apple': active concept = '{name}' (idx {idx})")
    
    # Check similarity to all concepts
    recent_activity = torch.stack(network.activity_history[-20:]).mean(dim=0)
    sim = network.similarity_to_patterns(recent_activity)
    
    print("\nSimilarity to all concepts:")
    for concept_name, concept_idx in indices.items():
        print(f"  {concept_name}: {sim[concept_idx].item():.3f}")


def demo_activity_tracking():
    """Demonstrate tracking neural activity and finding transitions."""
    print("\n" + "=" * 60)
    print("Activity Tracking Demo")
    print("=" * 60)
    
    # Create network with noise for spontaneous transitions
    config = AttractorConfig(
        n_neurons=100,
        noise_std=0.1,  # Higher noise for more transitions
    )
    network = AttractorNetwork(config)
    tracker = ActivityTracker()
    
    # Store distinct patterns
    torch.manual_seed(42)
    patterns = []
    for i in range(3):
        pattern = torch.zeros(100)
        # Each pattern activates a different region
        start = i * 30
        pattern[start:start+10] = 1
        patterns.append(pattern)
        network.store_pattern(pattern)
    
    print(f"Stored {len(patterns)} patterns with non-overlapping active regions")
    
    # Run network and track activity
    print("\n--- Simulating Network Activity ---")
    network.reset_state(batch_size=1)
    
    # Apply different patterns as cues
    for t in range(100):
        # Switch cue every 30 steps
        if t < 30:
            cue = patterns[0].unsqueeze(0) * 0.5
        elif t < 60:
            cue = patterns[1].unsqueeze(0) * 0.5
        else:
            cue = patterns[2].unsqueeze(0) * 0.5
        
        spikes, _ = network(cue)
        tracker.record(spikes)
    
    print(f"Recorded {len(tracker.history)} timesteps")
    
    # Analyze trajectory
    trajectory = tracker.get_trajectory()
    print(f"Trajectory shape: {trajectory.shape}")
    
    # Project to 3D using PCA
    coords, variance = tracker.project_pca(n_components=3)
    print(f"\nPCA projection shape: {coords.shape}")
    print(f"Explained variance: {[f'{v:.2%}' for v in variance.tolist()]}")
    
    # Find transitions
    transitions = tracker.find_transitions(patterns, threshold=0.5)
    print(f"\nFound {len(transitions)} transitions:")
    for t, from_p, to_p in transitions:
        print(f"  t={t}: pattern {from_p} -> pattern {to_p}")


def demo_thought_trajectory():
    """Demonstrate thought trajectory representation."""
    print("\n" + "=" * 60)
    print("Thought Trajectory Demo")
    print("=" * 60)
    
    trajectory = ThoughtTrajectory()
    
    # Simulate a sequence of "thoughts" (attractor visits)
    thoughts = [
        (0, "coffee"),
        (100, "morning"),
        (200, "work"),
        (350, "meeting"),
        (400, "coffee"),  # Back to coffee
    ]
    
    print("Simulated thought sequence:")
    for time, thought in thoughts:
        trajectory.add_state(thoughts.index((time, thought)), time)
        print(f"  t={time}: thinking about '{thought}'")
    
    print(f"\nState sequence: {trajectory.get_sequence()}")
    print(f"Transitions: {trajectory.get_transitions()}")
    print(f"Mean dwell time: {trajectory.mean_dwell_time():.1f} ms")
    print(f"\nTrajectory: {trajectory}")


def main():
    """Run all demos."""
    demo_pattern_storage()
    demo_concept_network()
    demo_activity_tracking()
    demo_thought_trajectory()
    
    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
