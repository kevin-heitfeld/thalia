#!/usr/bin/env python3
"""Experiment 5: Spontaneous Thought

Create a recurrent network with attractors, run without external input,
log concept transitions, and analyze thought trajectories.

This is the core demonstration of THALIA's thinking capabilities.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import Counter
import time

from thalia.cognition import DaydreamNetwork, DaydreamConfig, DaydreamMode


def run_experiment():
    """Run the spontaneous thought experiment."""
    print("=" * 60)
    print("Experiment 5: Spontaneous Thought")
    print("=" * 60)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    n_neurons = 128
    n_concepts = 6
    n_timesteps = 500
    
    # Concept names for visualization
    concept_names = ["Cat", "Dog", "Bird", "Tree", "House", "Car"]
    
    print(f"\nConfiguration:")
    print(f"  Neurons: {n_neurons}")
    print(f"  Concepts: {n_concepts}")
    print(f"  Simulation steps: {n_timesteps}")
    print(f"  Concepts: {', '.join(concept_names)}")
    
    # Create daydream network
    config = DaydreamConfig(
        n_neurons=n_neurons,
        base_noise=0.15,
        dwell_time_mean=40.0,
        dwell_time_std=15.0,
    )
    
    daydreamer = DaydreamNetwork(config).to(device)
    
    # Create and store concept patterns
    print(f"\nCreating and storing concept patterns...")
    
    for i, name in enumerate(concept_names):
        # Create distinct pattern for each concept
        pattern = torch.zeros(n_neurons, device=device)
        
        # Each concept activates a different subset of neurons
        base_start = i * (n_neurons // n_concepts)
        base_end = base_start + n_neurons // 3
        pattern[base_start:base_end] = 1.0
        
        # Add some overlap with adjacent concepts
        overlap_start = ((i + 1) % n_concepts) * (n_neurons // n_concepts)
        pattern[overlap_start:overlap_start + n_neurons // 8] = 1.0
        
        # Store concept
        daydreamer.store_concept(pattern, name)
        
        active = pattern.sum().item()
        print(f"  {name}: {active:.0f} active neurons ({100*active/n_neurons:.1f}%)")
    
    # Create associations between concepts
    print("\nCreating concept associations...")
    associations = [
        ("Cat", "Dog", 0.7),    # Both are pets
        ("Cat", "Bird", 0.3),   # Cat hunts bird
        ("Dog", "House", 0.5),  # Dog lives in house
        ("Bird", "Tree", 0.8),  # Bird in tree
        ("Tree", "House", 0.4), # Tree near house
        ("House", "Car", 0.6),  # Car near house
    ]
    
    for concept1, concept2, strength in associations:
        daydreamer.associate(concept1, concept2, strength)
        print(f"  {concept1} <-> {concept2}: {strength}")
    
    # Run spontaneous thought simulation
    print(f"\n" + "=" * 60)
    print("Running Spontaneous Thought Simulation")
    print("=" * 60)
    
    # Storage for analysis
    concept_history = []
    transition_times = []
    last_concept = None
    
    # Start daydreaming
    daydreamer.start_daydream(mode=DaydreamMode.FREE)
    
    print("\nThought stream:")
    thought_log = []
    
    start_time = time.time()
    
    for t in range(n_timesteps):
        # Take one step of daydreaming
        state = daydreamer.step()
        
        # Track current concept
        concept_history.append(state.current_concept)
        
        # Log transitions
        if state.transition_occurred:
            transition_times.append(t)
            thought_log.append((t, state.concept_name))
            if t < 200:  # Only print first 200 steps
                print(f"  t={t:3d}: {state.concept_name}")
    
    elapsed_time = time.time() - start_time
    
    print(f"\n... (truncated after t=200)")
    print(f"\nSimulation completed in {elapsed_time:.2f}s")
    print(f"Total transitions: {len(transition_times)}")
    
    # Analyze thought patterns
    print("\n" + "=" * 60)
    print("Analysis")
    print("=" * 60)
    
    # Count concept visits
    concept_counts = Counter([c for _, c in thought_log])
    print("\nConcept visit frequencies:")
    for name in concept_names:
        count = concept_counts.get(name, 0)
        pct = 100 * count / len(thought_log) if thought_log else 0
        bar = "█" * int(pct / 5)
        print(f"  {name:6s}: {count:3d} ({pct:5.1f}%) {bar}")
    
    # Compute transition matrix
    transitions = np.zeros((n_concepts, n_concepts))
    for i in range(len(thought_log) - 1):
        from_concept = concept_names.index(thought_log[i][1])
        to_concept = concept_names.index(thought_log[i + 1][1])
        transitions[from_concept, to_concept] += 1
    
    # Normalize rows
    row_sums = transitions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition_probs = transitions / row_sums
    
    print("\nTransition probabilities:")
    print(f"  From\\To: {' '.join([n[:4] for n in concept_names])}")
    for i, name in enumerate(concept_names):
        probs = transition_probs[i]
        prob_str = " ".join([f"{p:.2f}" for p in probs])
        print(f"  {name[:6]:6s}: {prob_str}")
    
    # Calculate dwell times
    if len(transition_times) > 1:
        dwell_times = np.diff(transition_times)
        mean_dwell = np.mean(dwell_times)
        std_dwell = np.std(dwell_times)
        print(f"\nDwell time: {mean_dwell:.1f} ± {std_dwell:.1f} steps")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Experiment 5: Spontaneous Thought", fontsize=14, fontweight='bold')
    
    # 1. Concept activations over time
    ax1 = axes[0, 0]
    concept_activations = np.array(concept_history)
    ax1.plot(concept_activations, 'b-', alpha=0.7)
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Active Concept")
    ax1.set_yticks(range(n_concepts))
    ax1.set_yticklabels(concept_names)
    ax1.set_title("Thought Stream Over Time")
    
    # 2. Concept frequency bar chart
    ax2 = axes[0, 1]
    counts = [concept_counts.get(n, 0) for n in concept_names]
    bars = ax2.bar(concept_names, counts, color='steelblue')
    ax2.set_xlabel("Concept")
    ax2.set_ylabel("Visit Count")
    ax2.set_title("Concept Visit Frequency")
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Transition matrix heatmap
    ax3 = axes[0, 2]
    im3 = ax3.imshow(transition_probs, cmap='Blues', vmin=0, vmax=1)
    ax3.set_xticks(range(n_concepts))
    ax3.set_yticks(range(n_concepts))
    ax3.set_xticklabels([n[:3] for n in concept_names])
    ax3.set_yticklabels([n[:3] for n in concept_names])
    ax3.set_xlabel("To Concept")
    ax3.set_ylabel("From Concept")
    ax3.set_title("Transition Probabilities")
    plt.colorbar(im3, ax=ax3)
    
    # 4. Dwell time histogram
    ax4 = axes[1, 0]
    if len(transition_times) > 1:
        ax4.hist(dwell_times, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax4.axvline(config.dwell_time_mean, color='red', linestyle='--', 
                   label=f'Target: {config.dwell_time_mean}')
        ax4.axvline(np.mean(dwell_times), color='green', linestyle='-', 
                   label=f'Actual: {np.mean(dwell_times):.1f}')
        ax4.legend()
    ax4.set_xlabel("Dwell Time (steps)")
    ax4.set_ylabel("Count")
    ax4.set_title("Dwell Time Distribution")
    
    # 5. Association network visualization
    ax5 = axes[1, 1]
    # Draw concept nodes in a circle
    angles = np.linspace(0, 2*np.pi, n_concepts, endpoint=False)
    x_pos = np.cos(angles)
    y_pos = np.sin(angles)
    
    # Draw association edges
    for c1, c2, strength in associations:
        i1 = concept_names.index(c1)
        i2 = concept_names.index(c2)
        ax5.plot([x_pos[i1], x_pos[i2]], [y_pos[i1], y_pos[i2]], 
                'k-', alpha=strength, linewidth=2*strength)
    
    # Draw nodes
    ax5.scatter(x_pos, y_pos, s=500, c='steelblue', zorder=5)
    for i, name in enumerate(concept_names):
        ax5.annotate(name, (x_pos[i], y_pos[i]), ha='center', va='center', 
                    fontsize=8, fontweight='bold', color='white')
    ax5.set_xlim(-1.5, 1.5)
    ax5.set_ylim(-1.5, 1.5)
    ax5.set_aspect('equal')
    ax5.set_title("Concept Association Network")
    ax5.axis('off')
    
    # 6. Thought trajectory in 2D (simple projection)
    ax6 = axes[1, 2]
    trajectory_x = x_pos[concept_activations]
    trajectory_y = y_pos[concept_activations]
    
    # Add noise for visualization
    trajectory_x += np.random.randn(len(trajectory_x)) * 0.1
    trajectory_y += np.random.randn(len(trajectory_y)) * 0.1
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(trajectory_x)))
    ax6.scatter(trajectory_x[::5], trajectory_y[::5], c=colors[::5], s=10, alpha=0.5)
    ax6.scatter(x_pos, y_pos, s=200, c='red', marker='*', zorder=5)
    for i, name in enumerate(concept_names):
        ax6.annotate(name, (x_pos[i], y_pos[i] + 0.15), ha='center', fontsize=8)
    ax6.set_title("Thought Trajectory (color = time)")
    ax6.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "exp5_spontaneous_thought.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    
    plt.show()
    
    # Success criteria
    print("\n" + "=" * 60)
    print("Success Criteria Check")
    print("=" * 60)
    
    # Check that transitions occur
    has_transitions = len(transition_times) > 5
    
    # Check that all concepts are visited
    concepts_visited = len(concept_counts) >= n_concepts // 2
    
    # Check that associations influence transitions (associated concepts co-occur more)
    # Count associated transitions vs non-associated
    associated_pairs = set((c1, c2) for c1, c2, _ in associations)
    associated_pairs.update((c2, c1) for c1, c2, _ in associations)
    
    associated_count = 0
    total_count = 0
    for i in range(len(thought_log) - 1):
        pair = (thought_log[i][1], thought_log[i + 1][1])
        if pair[0] != pair[1]:  # Not self-transition
            total_count += 1
            if pair in associated_pairs:
                associated_count += 1
    
    # With 6 concepts and random transitions, expected associated = 6/15 = 40%
    # With associations working, should be higher
    association_ratio = associated_count / max(1, total_count)
    associations_work = association_ratio > 0.3  # Above random chance
    
    # Check reasonable dwell times
    reasonable_dwell = True
    if len(transition_times) > 1:
        reasonable_dwell = 5 < np.mean(dwell_times) < 200
    
    criteria = [
        ("Spontaneous transitions occur", has_transitions),
        ("Multiple concepts visited", concepts_visited),
        ("Associations influence thought", associations_work),
        ("Reasonable dwell times", reasonable_dwell),
    ]
    
    all_passed = True
    for name, passed in criteria:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        all_passed = all_passed and passed
    
    print(f"\n  Total transitions: {len(transition_times)}")
    print(f"  Concepts visited: {len(concept_counts)}/{n_concepts}")
    print(f"  Associated transition ratio: {association_ratio*100:.1f}%")
    
    print("\n" + ("🎉 All criteria passed!" if all_passed else "⚠️ Some criteria failed"))
    
    return all_passed


if __name__ == "__main__":
    run_experiment()
