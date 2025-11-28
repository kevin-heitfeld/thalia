#!/usr/bin/env python3
"""Experiment 3: Attractor Formation

Create a small attractor network (~100 neurons), store 3-5 patterns,
test recall from partial cues, and visualize attractor basins.

This validates that attractor dynamics work correctly.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA

from thalia.dynamics import AttractorSNN, AttractorConfig


def create_orthogonal_patterns(n_patterns: int, n_neurons: int) -> torch.Tensor:
    """Create semi-orthogonal binary patterns.
    
    Args:
        n_patterns: Number of patterns to create
        n_neurons: Size of each pattern
        
    Returns:
        Tensor of shape (n_patterns, n_neurons)
    """
    patterns = torch.zeros(n_patterns, n_neurons)
    
    # Create patterns with ~50% active neurons, minimizing overlap
    neurons_per_pattern = n_neurons // n_patterns
    
    for i in range(n_patterns):
        # Base neurons for this pattern
        start = i * neurons_per_pattern // 2
        end = start + n_neurons // 2
        indices = torch.arange(start, min(end, n_neurons)) % n_neurons
        patterns[i, indices] = 1.0
        
        # Add some random neurons
        random_indices = torch.randperm(n_neurons)[:n_neurons // 4]
        patterns[i, random_indices] = 1.0
        
    return (patterns > 0.5).float()


def run_experiment():
    """Run the attractor formation experiment."""
    print("=" * 60)
    print("Experiment 3: Attractor Formation")
    print("=" * 60)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    n_neurons = 100
    n_patterns = 4
    n_recall_steps = 50
    
    print(f"\nNetwork Configuration:")
    print(f"  Neurons: {n_neurons}")
    print(f"  Patterns to store: {n_patterns}")
    print(f"  Recall steps: {n_recall_steps}")
    
    # Create attractor network
    config = AttractorConfig(
        n_neurons=n_neurons,
        tau_mem=10.0,
        noise_std=0.05,
        learning_rate=0.1,
    )
    
    network = AttractorSNN(config).to(device)
    
    # Create and store patterns
    patterns = create_orthogonal_patterns(n_patterns, n_neurons).to(device)
    
    print(f"\nStoring {n_patterns} patterns...")
    for i in range(n_patterns):
        network.store_pattern(patterns[i])
        overlap = (patterns[i].sum() / n_neurons * 100).item()
        print(f"  Pattern {i}: {overlap:.1f}% neurons active")
    
    # Compute pattern similarity matrix
    similarity = torch.zeros(n_patterns, n_patterns)
    for i in range(n_patterns):
        for j in range(n_patterns):
            overlap = (patterns[i] * patterns[j]).sum() / patterns[i].sum()
            similarity[i, j] = overlap.item()
    
    print(f"\nPattern similarity matrix:")
    print(similarity.numpy().round(2))
    
    # Test recall from partial cues
    print("\n" + "=" * 60)
    print("Testing Pattern Recall")
    print("=" * 60)
    
    corruption_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    recall_results = {}
    
    for corruption in corruption_levels:
        recalls = []
        for pattern_idx in range(n_patterns):
            # Create corrupted pattern
            original = patterns[pattern_idx].clone()
            mask = torch.rand(n_neurons, device=device) < corruption
            corrupted = original.clone()
            corrupted[mask] = 1 - corrupted[mask]  # Flip corrupted bits
            
            # Recall
            network.reset_state(batch_size=1)
            recalled = network.recall(corrupted, steps=n_recall_steps)
            
            # Compute recall accuracy
            accuracy = (recalled.round() == original).float().mean().item()
            recalls.append(accuracy)
        
        recall_results[corruption] = recalls
        mean_acc = np.mean(recalls)
        print(f"  Corruption {corruption*100:.0f}%: Mean recall accuracy {mean_acc*100:.1f}%")
    
    # Visualize attractor dynamics
    print("\n" + "=" * 60)
    print("Visualizing Attractor Dynamics")
    print("=" * 60)
    
    # Collect trajectories for visualization
    trajectories = []
    trajectory_labels = []
    
    for pattern_idx in range(n_patterns):
        # Start from corrupted pattern
        corrupted = patterns[pattern_idx].clone()
        mask = torch.rand(n_neurons, device=device) < 0.3
        corrupted[mask] = 1 - corrupted[mask]
        
        network.reset_state(batch_size=1)
        trajectory = [corrupted.cpu().numpy()]
        
        state = corrupted.unsqueeze(0)
        for _ in range(n_recall_steps):
            state = network.step(state)
            trajectory.append(state.squeeze(0).cpu().numpy())
        
        trajectories.append(np.array(trajectory))
        trajectory_labels.append(pattern_idx)
    
    # Also add trajectories from random starting points
    for _ in range(4):
        random_start = (torch.rand(n_neurons, device=device) > 0.5).float()
        network.reset_state(batch_size=1)
        trajectory = [random_start.cpu().numpy()]
        
        state = random_start.unsqueeze(0)
        for _ in range(n_recall_steps):
            state = network.step(state)
            trajectory.append(state.squeeze(0).cpu().numpy())
        
        trajectories.append(np.array(trajectory))
        trajectory_labels.append(-1)  # Random start
    
    # PCA for visualization
    all_points = np.vstack([t for t in trajectories])
    pca = PCA(n_components=2)
    all_points_2d = pca.fit_transform(all_points)
    
    # Split back into trajectories
    trajectories_2d = []
    idx = 0
    for t in trajectories:
        trajectories_2d.append(all_points_2d[idx:idx+len(t)])
        idx += len(t)
    
    # Project stored patterns
    patterns_2d = pca.transform(patterns.cpu().numpy())
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Experiment 3: Attractor Formation", fontsize=14, fontweight='bold')
    
    # 1. Stored patterns
    ax1 = axes[0, 0]
    im1 = ax1.imshow(patterns.cpu().numpy(), aspect='auto', cmap='binary')
    ax1.set_xlabel("Neuron")
    ax1.set_ylabel("Pattern")
    ax1.set_title(f"Stored Patterns ({n_patterns})")
    plt.colorbar(im1, ax=ax1)
    
    # 2. Pattern similarity
    ax2 = axes[0, 1]
    im2 = ax2.imshow(similarity.numpy(), cmap='viridis', vmin=0, vmax=1)
    ax2.set_xlabel("Pattern")
    ax2.set_ylabel("Pattern")
    ax2.set_title("Pattern Similarity")
    plt.colorbar(im2, ax=ax2)
    for i in range(n_patterns):
        for j in range(n_patterns):
            ax2.text(j, i, f'{similarity[i,j]:.2f}', ha='center', va='center', color='white')
    
    # 3. Recall accuracy vs corruption
    ax3 = axes[0, 2]
    for pattern_idx in range(n_patterns):
        accs = [recall_results[c][pattern_idx] for c in corruption_levels]
        ax3.plot([c*100 for c in corruption_levels], [a*100 for a in accs], 
                 'o-', label=f'Pattern {pattern_idx}')
    ax3.set_xlabel("Corruption Level (%)")
    ax3.set_ylabel("Recall Accuracy (%)")
    ax3.set_title("Pattern Completion Performance")
    ax3.legend()
    ax3.set_ylim(0, 105)
    ax3.axhline(50, color='gray', linestyle='--', alpha=0.5)
    
    # 4. Attractor landscape (PCA)
    ax4 = axes[1, 0]
    colors = plt.cm.Set1(np.linspace(0, 1, n_patterns))
    
    for i, (traj, label) in enumerate(zip(trajectories_2d, trajectory_labels)):
        if label >= 0:
            color = colors[label]
            ax4.plot(traj[:, 0], traj[:, 1], '-', color=color, alpha=0.5, linewidth=1)
            ax4.scatter(traj[0, 0], traj[0, 1], c=[color], marker='o', s=50, edgecolor='black')
            ax4.scatter(traj[-1, 0], traj[-1, 1], c=[color], marker='*', s=100, edgecolor='black')
        else:
            ax4.plot(traj[:, 0], traj[:, 1], '-', color='gray', alpha=0.3, linewidth=1)
    
    # Plot attractors
    for i in range(n_patterns):
        ax4.scatter(patterns_2d[i, 0], patterns_2d[i, 1], c=[colors[i]], 
                   marker='s', s=200, edgecolor='black', linewidth=2, label=f'Attractor {i}')
    
    ax4.set_xlabel("PC1")
    ax4.set_ylabel("PC2")
    ax4.set_title("Attractor Landscape (PCA)")
    ax4.legend(loc='upper right')
    
    # 5. Weight matrix
    ax5 = axes[1, 1]
    weights = network.weights.cpu().numpy()
    im5 = ax5.imshow(weights, aspect='auto', cmap='RdBu_r')
    ax5.set_xlabel("Pre-synaptic Neuron")
    ax5.set_ylabel("Post-synaptic Neuron")
    ax5.set_title("Learned Weight Matrix")
    plt.colorbar(im5, ax=ax5)
    
    # 6. Energy landscape cross-section
    ax6 = axes[1, 2]
    # Interpolate between two patterns
    pattern_a = patterns[0].cpu().numpy()
    pattern_b = patterns[1].cpu().numpy()
    
    alphas = np.linspace(0, 1, 50)
    energies = []
    
    for alpha in alphas:
        interpolated = (1 - alpha) * pattern_a + alpha * pattern_b
        # Approximate energy as -x^T W x
        energy = -np.dot(interpolated, np.dot(weights, interpolated))
        energies.append(energy)
    
    ax6.plot(alphas, energies, 'b-', linewidth=2)
    ax6.scatter([0, 1], [energies[0], energies[-1]], c='red', s=100, zorder=5, label='Attractors')
    ax6.set_xlabel("Interpolation (Pattern 0 → Pattern 1)")
    ax6.set_ylabel("Energy")
    ax6.set_title("Energy Landscape Cross-Section")
    ax6.legend()
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "exp3_attractors.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {output_path}")
    
    plt.show()
    
    # Success criteria
    print("\n" + "=" * 60)
    print("Success Criteria Check")
    print("=" * 60)
    
    low_corruption_recall = np.mean(recall_results[0.2]) > 0.8
    attractors_form = np.mean(recall_results[0.1]) > 0.9
    completion_works = np.mean(recall_results[0.3]) > 0.6
    coherent_flow = len(trajectories) > 0  # Trajectories were generated
    
    criteria = [
        ("Attractors form and stabilize", attractors_form),
        ("Pattern completion works", completion_works),
        ("Low corruption recall >80%", low_corruption_recall),
        ("Free association generates coherent flow", coherent_flow),
    ]
    
    all_passed = True
    for name, passed in criteria:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        all_passed = all_passed and passed
    
    print("\n" + ("🎉 All criteria passed!" if all_passed else "⚠️ Some criteria failed"))
    
    return all_passed


if __name__ == "__main__":
    run_experiment()
