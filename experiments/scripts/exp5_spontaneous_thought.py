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

from thalia.dynamics import AttractorSNN, AttractorConfig
from thalia.cognition import DaydreamEngine, DaydreamConfig, DaydreamMode
from thalia.memory import WorkingMemorySNN


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
    
    # Create attractor network for concept storage
    attractor_config = AttractorConfig(
        n_neurons=n_neurons,
        tau_mem=20.0,
        noise_std=0.1,
        learning_rate=0.5,
    )
    
    attractor_net = AttractorSNN(attractor_config).to(device)
    
    # Create concept patterns
    print(f"\nCreating and storing concept patterns...")
    concepts = {}
    
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
        
        concepts[name] = pattern
        attractor_net.store_pattern(pattern)
        
        active = pattern.sum().item()
        print(f"  {name}: {active:.0f} active neurons ({100*active/n_neurons:.1f}%)")
    
    # Create daydream engine
    daydream_config = DaydreamConfig(
        noise_scale=0.15,
        temperature=1.2,
        min_dwell_time=20,
        max_dwell_time=80,
        novelty_weight=0.5,
    )
    
    daydreamer = DaydreamEngine(
        attractor_network=attractor_net,
        config=daydream_config,
    ).to(device)
    
    # Run spontaneous thought simulation
    print(f"\n" + "=" * 60)
    print("Running Spontaneous Thought Simulation")
    print("=" * 60)
    
    # Storage for analysis
    trajectory = []
    concept_activations = {name: [] for name in concept_names}
    transition_times = []
    last_concept = None
    
    # Reset and run
    daydreamer.reset()
    
    print(f"\nThinking...")
    start_time = time.time()
    
    for t in range(n_timesteps):
        # Step the daydreamer (no external input)
        state = daydreamer.step()
        
        # Compute similarity to each concept
        similarities = {}
        for name, pattern in concepts.items():
            # Cosine similarity
            sim = torch.cosine_similarity(
                state.view(-1), 
                pattern.view(-1), 
                dim=0
            ).item()
            similarities[name] = sim
            concept_activations[name].append(sim)
        
        # Identify current dominant concept
        current_concept = max(similarities, key=similarities.get)
        trajectory.append(current_concept)
        
        # Detect transitions
        if current_concept != last_concept:
            if last_concept is not None:
                transition_times.append(t)
                if len(transition_times) <= 10:
                    print(f"  t={t:3d}: {last_concept} → {current_concept}")
            last_concept = current_concept
    
    elapsed = time.time() - start_time
    print(f"\nSimulation complete in {elapsed:.2f}s")
    print(f"  {n_timesteps / elapsed:.1f} steps/sec")
    
    # Analyze thought patterns
    print(f"\n" + "=" * 60)
    print("Thought Pattern Analysis")
    print("=" * 60)
    
    # Transition statistics
    n_transitions = len(transition_times)
    print(f"\nTransition Statistics:")
    print(f"  Total transitions: {n_transitions}")
    if n_transitions > 0:
        dwell_times = np.diff([0] + transition_times + [n_timesteps])
        print(f"  Mean dwell time: {np.mean(dwell_times):.1f} steps")
        print(f"  Min dwell time: {np.min(dwell_times)} steps")
        print(f"  Max dwell time: {np.max(dwell_times)} steps")
    
    # Concept visit frequency
    concept_counts = Counter(trajectory)
    print(f"\nConcept Visit Frequency:")
    for name in concept_names:
        count = concept_counts[name]
        pct = 100 * count / n_timesteps
        print(f"  {name}: {count} ({pct:.1f}%)")
    
    # Transition matrix
    transition_matrix = np.zeros((n_concepts, n_concepts))
    for i in range(len(trajectory) - 1):
        from_idx = concept_names.index(trajectory[i])
        to_idx = concept_names.index(trajectory[i + 1])
        if from_idx != to_idx:
            transition_matrix[from_idx, to_idx] += 1
    
    # Normalize
    row_sums = transition_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transition_probs = transition_matrix / row_sums
    
    print(f"\nTransition Probabilities:")
    for i, from_name in enumerate(concept_names):
        transitions = []
        for j, to_name in enumerate(concept_names):
            if transition_probs[i, j] > 0.1:
                transitions.append(f"{to_name}:{transition_probs[i,j]:.2f}")
        if transitions:
            print(f"  {from_name} → {', '.join(transitions)}")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Experiment 5: Spontaneous Thought", fontsize=14, fontweight='bold')
    
    # 1. Concept activation over time
    ax1 = axes[0, 0]
    colors = plt.cm.Set2(np.linspace(0, 1, n_concepts))
    for i, name in enumerate(concept_names):
        ax1.plot(concept_activations[name], label=name, color=colors[i], alpha=0.7)
    ax1.set_xlabel("Time (steps)")
    ax1.set_ylabel("Concept Similarity")
    ax1.set_title("Concept Activations Over Time")
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_xlim(0, n_timesteps)
    
    # 2. Thought trajectory (categorical)
    ax2 = axes[0, 1]
    concept_indices = [concept_names.index(c) for c in trajectory]
    ax2.plot(concept_indices, 'k-', alpha=0.3, linewidth=0.5)
    ax2.scatter(range(len(trajectory)), concept_indices, c=concept_indices, 
                cmap='Set2', s=1, alpha=0.5)
    ax2.set_xlabel("Time (steps)")
    ax2.set_ylabel("Concept")
    ax2.set_yticks(range(n_concepts))
    ax2.set_yticklabels(concept_names)
    ax2.set_title("Thought Trajectory")
    ax2.set_xlim(0, n_timesteps)
    
    # 3. Visit frequency pie chart
    ax3 = axes[0, 2]
    sizes = [concept_counts[name] for name in concept_names]
    ax3.pie(sizes, labels=concept_names, colors=colors, autopct='%1.1f%%')
    ax3.set_title("Concept Visit Frequency")
    
    # 4. Transition matrix heatmap
    ax4 = axes[1, 0]
    im4 = ax4.imshow(transition_probs, cmap='Blues', vmin=0, vmax=1)
    ax4.set_xticks(range(n_concepts))
    ax4.set_yticks(range(n_concepts))
    ax4.set_xticklabels(concept_names, rotation=45, ha='right')
    ax4.set_yticklabels(concept_names)
    ax4.set_xlabel("To Concept")
    ax4.set_ylabel("From Concept")
    ax4.set_title("Transition Probabilities")
    plt.colorbar(im4, ax=ax4)
    
    # 5. Dwell time distribution
    ax5 = axes[1, 1]
    if len(dwell_times) > 1:
        ax5.hist(dwell_times, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax5.axvline(np.mean(dwell_times), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(dwell_times):.1f}')
        ax5.legend()
    ax5.set_xlabel("Dwell Time (steps)")
    ax5.set_ylabel("Count")
    ax5.set_title("Dwell Time Distribution")
    
    # 6. Thought graph
    ax6 = axes[1, 2]
    # Draw concept nodes
    angles = np.linspace(0, 2*np.pi, n_concepts, endpoint=False)
    x_pos = np.cos(angles)
    y_pos = np.sin(angles)
    
    # Draw edges (transitions)
    for i in range(n_concepts):
        for j in range(n_concepts):
            if transition_probs[i, j] > 0.05:
                weight = transition_probs[i, j]
                ax6.annotate("", 
                            xy=(x_pos[j], y_pos[j]),
                            xytext=(x_pos[i], y_pos[i]),
                            arrowprops=dict(arrowstyle="->", 
                                          color='gray',
                                          alpha=weight,
                                          lw=weight*3))
    
    # Draw nodes
    node_sizes = [300 * concept_counts[name] / n_timesteps + 100 for name in concept_names]
    ax6.scatter(x_pos, y_pos, s=node_sizes, c=range(n_concepts), cmap='Set2', 
                edgecolor='black', linewidth=2, zorder=5)
    
    for i, name in enumerate(concept_names):
        ax6.annotate(name, (x_pos[i], y_pos[i]), ha='center', va='center',
                    fontsize=8, fontweight='bold')
    
    ax6.set_xlim(-1.5, 1.5)
    ax6.set_ylim(-1.5, 1.5)
    ax6.set_aspect('equal')
    ax6.axis('off')
    ax6.set_title("Thought Graph\n(node size = visit frequency)")
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "exp5_spontaneous_thought.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    
    plt.show()
    
    # Log a sample thought stream
    print(f"\n" + "=" * 60)
    print("Sample Thought Stream (first 100 steps)")
    print("=" * 60)
    
    current = trajectory[0]
    stream_parts = [current]
    for i, concept in enumerate(trajectory[1:100], 1):
        if concept != current:
            stream_parts.append(f"({i}) → {concept}")
            current = concept
    
    print(" ".join(stream_parts))
    
    # Success criteria
    print(f"\n" + "=" * 60)
    print("Success Criteria Check")
    print("=" * 60)
    
    has_transitions = n_transitions > 5
    multiple_concepts = len(set(trajectory)) >= 3
    reasonable_dwell = 10 < np.mean(dwell_times) < 100 if n_transitions > 0 else False
    coherent_flow = n_transitions < n_timesteps / 5  # Not switching every step
    
    criteria = [
        ("Recurrent network with attractors", True),
        ("No external input (spontaneous)", True),
        ("Multiple concept transitions", has_transitions),
        ("Visits multiple concepts", multiple_concepts),
        ("Reasonable dwell times", reasonable_dwell),
        ("Coherent thought flow (not random)", coherent_flow),
    ]
    
    all_passed = True
    for name, passed in criteria:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        all_passed = all_passed and passed
    
    print("\n" + ("🎉 All criteria passed! THALIA is thinking!" if all_passed 
                 else "⚠️ Some criteria failed"))
    
    return all_passed


if __name__ == "__main__":
    run_experiment()
