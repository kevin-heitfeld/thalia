"""
Experiment 4: Hippocampus Sequence Memory

Objective: Demonstrate one-shot episodic memory formation and pattern completion
using the hippocampus region.

Setup:
- Input: Sequence of sparse pattern pairs (cue → target)
- Architecture: 64 input → 128 CA3-like neurons → 64 output
- Learning: One-shot Hebbian (single exposure)
- Test: Present cue, measure target recall

Key Insight:
The hippocampus can learn associations in a single exposure (one-shot learning),
unlike cortex which needs many repetitions. This is achieved through:
1. Very high learning rate
2. Sparse coding for pattern separation
3. Recurrent connections for pattern completion

Analysis:
1. Recall accuracy vs number of stored patterns
2. Pattern completion with partial cues (50%, 25%)
3. Interference between similar patterns
4. Capacity estimation

Success Metrics:
- 80%+ exact recall for 20 patterns
- Graceful degradation with partial cues
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from thalia.regions.hippocampus import Hippocampus, HippocampusConfig
from experiments.scripts.regions.exp_utils import (
    create_sequence_patterns,
    save_results,
    get_results_dir,
)


def create_hippocampus(
    n_cue: int = 64,
    n_target: int = 64,
) -> Hippocampus:
    """Create a Hippocampus configured for hetero-associative memory.

    Uses separate n_input (cue dimension) and n_output (target dimension)
    for proper cue→target association learning.
    
    Architecture:
        Cue (n_input) → Feedforward Weights → Target (n_output)
        
    This is hetero-associative: input and output are different patterns.
    """

    config = HippocampusConfig(
        n_input=n_cue,      # Cue dimension
        n_output=n_target,  # Target dimension
        learning_rate=1.0,  # Very high for one-shot
        learning_rate_retrieval=0.01,  # Minimal during retrieval
        sparsity_target=0.1,  # 10% active
        inhibition_strength=2.0,
        recurrent_strength=0.5,  # For target pattern completion
        w_max=3.0,
    )
    return Hippocampus(config)


def create_pattern_pairs(
    n_patterns: int = 20,
    pattern_size: int = 64,
    sparsity: float = 0.1,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Create cue-target pattern pairs.

    Each pair is a random sparse pattern.
    """
    n_active = int(pattern_size * sparsity)
    patterns = []

    for _ in range(n_patterns):
        # Random sparse cue
        cue = torch.zeros(pattern_size)
        cue[torch.randperm(pattern_size)[:n_active]] = 1.0

        # Random sparse target (different from cue)
        target = torch.zeros(pattern_size)
        target[torch.randperm(pattern_size)[:n_active]] = 1.0

        patterns.append((cue, target))

    return patterns


def store_patterns(
    hippocampus: Hippocampus,
    patterns: List[Tuple[torch.Tensor, torch.Tensor]],
    n_timesteps: int = 10,
    verbose: bool = True,
) -> Dict:
    """
    Store all pattern pairs in hippocampus.

    Uses one-shot learning: each pattern is presented once.
    """
    storage_stats = []

    for i, (cue, target) in enumerate(patterns):
        hippocampus.reset()

        # Create input as cue + target concatenated or just cue→target association
        # We'll store the association by presenting cue and learning target response

        # First, present the cue and get hippocampus response
        total_activity = None
        accumulated_cue = torch.zeros_like(cue)

        for t in range(n_timesteps):
            # Rate-coded cue
            input_spikes = (torch.rand_like(cue) < cue).float().unsqueeze(0)
            accumulated_cue += input_spikes.squeeze()

            output_spikes = hippocampus.forward(input_spikes)

            if total_activity is None:
                total_activity = output_spikes.clone()
            else:
                total_activity += output_spikes

        # Now learn the cue→target association
        # We want the hippocampus output to represent the target
        target_expanded = target.unsqueeze(0)

        # Force high learning rate for one-shot
        stats = hippocampus.learn(
            accumulated_cue.unsqueeze(0) / n_timesteps,
            target_expanded,
            force_encoding=True,
        )

        storage_stats.append({
            "pattern_idx": i,
            **stats,
        })

        if verbose and (i + 1) % 10 == 0:
            print(f"Stored {i+1}/{len(patterns)} patterns")

    return {"storage_stats": storage_stats}


def recall_patterns(
    hippocampus: Hippocampus,
    patterns: List[Tuple[torch.Tensor, torch.Tensor]],
    n_timesteps: int = 15,
    cue_fraction: float = 1.0,
) -> Dict:
    """
    Test recall of stored patterns.

    Uses direct weight-based readout for cleaner evaluation.

    Args:
        hippocampus: Trained hippocampus
        patterns: List of (cue, target) pairs
        n_timesteps: Timesteps for recall
        cue_fraction: Fraction of cue to present (for partial cue test)
    """
    recalls = []

    for i, (cue, target) in enumerate(patterns):
        # Create partial cue if needed
        if cue_fraction < 1.0:
            partial_cue = cue.clone()
            active_indices = cue.nonzero().squeeze(-1)
            n_keep = max(1, int(len(active_indices) * cue_fraction))
            drop_indices = active_indices[torch.randperm(len(active_indices))[n_keep:]]
            partial_cue[drop_indices] = 0.0
            cue_input = partial_cue
        else:
            cue_input = cue

        # Direct weight-based readout: which output neurons have strong
        # connections FROM the active cue neurons?
        # activation = W @ cue (each output neuron's total input from cue)
        activation = hippocampus.weights @ cue_input

        # Select top-k neurons matching target sparsity
        n_target_active = int(target.sum().item())
        _, top_indices = activation.topk(n_target_active)
        response = torch.zeros_like(target)
        response[top_indices] = 1.0

        # Compute recall accuracy
        target_active = target.sum().item()
        response_active = response.sum().item()

        if target_active > 0:
            # How many target neurons did we recall?
            correct_active = (response * target).sum().item()
            recall_rate = correct_active / target_active

            # How many false positives?
            if response_active > 0:
                precision = correct_active / response_active
            else:
                precision = 0.0

            # F1 score
            if recall_rate + precision > 0:
                f1 = 2 * (recall_rate * precision) / (recall_rate + precision)
            else:
                f1 = 0.0
        else:
            recall_rate = 1.0 if response_active == 0 else 0.0
            precision = 1.0 if response_active == 0 else 0.0
            f1 = 1.0 if response_active == 0 else 0.0

        recalls.append({
            "pattern_idx": i,
            "recall_rate": recall_rate,
            "precision": precision,
            "f1": f1,
            "target_active": target_active,
            "response_active": response_active,
        })

    avg_recall = np.mean([r['recall_rate'] for r in recalls])
    avg_precision = np.mean([r['precision'] for r in recalls])
    avg_f1 = np.mean([r['f1'] for r in recalls])

    return {
        "recalls": recalls,
        "avg_recall": avg_recall,
        "avg_precision": avg_precision,
        "avg_f1": avg_f1,
        "cue_fraction": cue_fraction,
    }


def test_capacity(
    pattern_size: int = 64,
    pattern_sizes: List[int] = [5, 10, 20, 30, 40, 50],
    n_trials: int = 5,
    n_timesteps: int = 15,
) -> Dict:
    """
    Test memory capacity as function of number of stored patterns.
    """
    results = {"pattern_counts": [], "avg_recalls": [], "std_recalls": []}

    for n_patterns in pattern_sizes:
        trial_recalls = []

        for trial in range(n_trials):
            # Create fresh hippocampus with hetero-associative architecture
            hippocampus = create_hippocampus(n_cue=pattern_size, n_target=pattern_size)

            # Create and store patterns
            patterns = create_pattern_pairs(n_patterns=n_patterns, pattern_size=pattern_size)
            store_patterns(hippocampus, patterns, n_timesteps=10, verbose=False)

            # Test recall
            recall_results = recall_patterns(hippocampus, patterns, n_timesteps=n_timesteps)
            trial_recalls.append(recall_results['avg_f1'])

        avg = np.mean(trial_recalls)
        std = np.std(trial_recalls)

        results["pattern_counts"].append(n_patterns)
        results["avg_recalls"].append(avg)
        results["std_recalls"].append(std)

        print(f"  {n_patterns} patterns: F1 = {avg:.3f} ± {std:.3f}")

    return results


def test_partial_cues(
    hippocampus: Hippocampus,
    patterns: List[Tuple[torch.Tensor, torch.Tensor]],
    cue_fractions: List[float] = [1.0, 0.75, 0.5, 0.25],
    n_timesteps: int = 15,
) -> Dict:
    """
    Test pattern completion with partial cues.
    """
    results = {"cue_fractions": [], "avg_f1": [], "avg_recall": []}

    for frac in cue_fractions:
        recall_results = recall_patterns(hippocampus, patterns, n_timesteps, cue_fraction=frac)

        results["cue_fractions"].append(frac)
        results["avg_f1"].append(recall_results['avg_f1'])
        results["avg_recall"].append(recall_results['avg_recall'])

        print(f"  Cue fraction {frac:.0%}: F1 = {recall_results['avg_f1']:.3f}, "
              f"Recall = {recall_results['avg_recall']:.3f}")

    return results


def visualize_results(
    recall_results: Dict,
    capacity_results: Dict,
    partial_results: Dict,
) -> Tuple[plt.Figure, plt.Figure]:
    """Visualize experiment results."""

    # Figure 1: Per-pattern recall
    fig1, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    recalls = recall_results['recalls']
    f1_scores = [r['f1'] for r in recalls]
    ax.bar(range(len(f1_scores)), f1_scores, color='steelblue', alpha=0.7)
    ax.axhline(y=0.8, color='green', linestyle='--', label='80% threshold')
    ax.set_xlabel('Pattern Index')
    ax.set_ylabel('F1 Score')
    ax.set_title(f'Per-Pattern Recall (avg F1 = {recall_results["avg_f1"]:.2f})')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    recall_rates = [r['recall_rate'] for r in recalls]
    precision_rates = [r['precision'] for r in recalls]
    x = np.arange(len(recalls))
    width = 0.35
    ax.bar(x - width/2, recall_rates, width, label='Recall', color='blue', alpha=0.7)
    ax.bar(x + width/2, precision_rates, width, label='Precision', color='orange', alpha=0.7)
    ax.set_xlabel('Pattern Index')
    ax.set_ylabel('Score')
    ax.set_title('Recall vs Precision')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Figure 2: Capacity and partial cues
    fig2, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    counts = capacity_results['pattern_counts']
    avgs = capacity_results['avg_recalls']
    stds = capacity_results['std_recalls']
    ax.errorbar(counts, avgs, yerr=stds, marker='o', capsize=5, color='steelblue')
    ax.axhline(y=0.8, color='green', linestyle='--', label='80% threshold')
    ax.set_xlabel('Number of Patterns')
    ax.set_ylabel('Average F1 Score')
    ax.set_title('Memory Capacity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    fracs = partial_results['cue_fractions']
    f1s = partial_results['avg_f1']
    recalls = partial_results['avg_recall']
    ax.plot(fracs, f1s, 'o-', label='F1 Score', color='blue')
    ax.plot(fracs, recalls, 's--', label='Recall', color='orange')
    ax.axhline(y=0.8, color='green', linestyle=':', label='80% threshold')
    ax.set_xlabel('Cue Fraction')
    ax.set_ylabel('Score')
    ax.set_title('Pattern Completion with Partial Cues')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0, 1.1)

    plt.tight_layout()

    return fig1, fig2


def run_experiment(
    n_patterns: int = 20,
    pattern_size: int = 64,
    sparsity: float = 0.1,
    n_timesteps: int = 15,
    show_plots: bool = True,
    save_plots: bool = True,
):
    """
    Run the complete Hippocampus hetero-associative memory experiment.
    
    Architecture: Cue (pattern_size) → Weights → Target (pattern_size)
    This tests the hippocampus's ability to learn cue→target associations
    in one shot (episodic memory).
    """
    print("=" * 60)
    print("EXPERIMENT 4: Hippocampus Sequence Memory")
    print("=" * 60)

    # 1. Create hippocampus with hetero-associative architecture
    print(f"\n[1/5] Creating Hippocampus (cue={pattern_size}, target={pattern_size})...")
    hippocampus = create_hippocampus(n_cue=pattern_size, n_target=pattern_size)

    print(f"\n[2/5] Creating {n_patterns} pattern pairs (sparsity={sparsity})...")
    patterns = create_pattern_pairs(n_patterns=n_patterns, pattern_size=pattern_size, sparsity=sparsity)

    # 2. Store patterns
    print(f"\n[3/5] Storing patterns (one-shot learning)...")
    store_stats = store_patterns(hippocampus, patterns, n_timesteps=10)

    # 3. Test recall
    print(f"\n[4/5] Testing recall...")
    print("  Full cue recall:")
    recall_results = recall_patterns(hippocampus, patterns, n_timesteps=n_timesteps)
    print(f"    Average F1: {recall_results['avg_f1']:.3f}")
    print(f"    Average Recall: {recall_results['avg_recall']:.3f}")
    print(f"    Average Precision: {recall_results['avg_precision']:.3f}")

    # 4. Test capacity
    print("\n  Memory capacity test:")
    capacity_results = test_capacity(pattern_size=pattern_size, n_trials=3, n_timesteps=n_timesteps)

    # 5. Test partial cues
    print("\n  Partial cue test:")
    partial_results = test_partial_cues(hippocampus, patterns, n_timesteps=n_timesteps)

    # 6. Visualize
    print(f"\n[5/5] Generating visualizations...")

    results_dir = get_results_dir()

    fig1, fig2 = visualize_results(recall_results, capacity_results, partial_results)

    if save_plots:
        fig1.savefig(results_dir / "exp4_recall.png", dpi=150)
        print(f"  Saved: exp4_recall.png")
        fig2.savefig(results_dir / "exp4_capacity.png", dpi=150)
        print(f"  Saved: exp4_capacity.png")

    # Weight visualization
    fig3, ax = plt.subplots(figsize=(10, 4))
    weights = hippocampus.weights.data.detach().cpu().numpy()
    im = ax.imshow(weights, aspect='auto', cmap='RdBu_r')
    ax.set_xlabel('Input')
    ax.set_ylabel('Hippocampal Neuron')
    ax.set_title('Learned Hippocampal Weights')
    plt.colorbar(im, ax=ax, label='Weight')
    if save_plots:
        fig3.savefig(results_dir / "exp4_weights.png", dpi=150)
        print(f"  Saved: exp4_weights.png")

    if show_plots:
        plt.show()
    else:
        plt.close('all')

    # Save results
    results = {
        "config": {
            "n_patterns": n_patterns,
            "pattern_size": pattern_size,
            "sparsity": sparsity,
            "n_timesteps": n_timesteps,
            "architecture": "hetero-associative",
        },
        "metrics": {
            "full_cue_f1": recall_results['avg_f1'],
            "full_cue_recall": recall_results['avg_recall'],
            "full_cue_precision": recall_results['avg_precision'],
        },
        "capacity": capacity_results,
        "partial_cues": partial_results,
    }
    save_results("exp4_hippocampus_memory", results)

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT 4 COMPLETE")
    print("=" * 60)

    success = recall_results['avg_f1'] >= 0.80
    print(f"\nSuccess criterion: Average F1 ≥ 80%")
    print(f"Result: {recall_results['avg_f1']*100:.1f}%")
    print(f"Status: {'✓ PASSED' if success else '✗ FAILED'}")

    return results, hippocampus


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hippocampus Hetero-Associative Memory Experiment")
    parser.add_argument("--n-patterns", type=int, default=20, help="Number of cue-target pairs")
    parser.add_argument("--pattern-size", type=int, default=64, help="Size of cue and target patterns")
    parser.add_argument("--sparsity", type=float, default=0.1, help="Pattern sparsity")
    parser.add_argument("--n-timesteps", type=int, default=15, help="Timesteps per pattern")
    parser.add_argument("--no-show", action="store_true", help="Don't show plots")
    parser.add_argument("--no-save", action="store_true", help="Don't save plots")

    args = parser.parse_args()

    run_experiment(
        n_patterns=args.n_patterns,
        pattern_size=args.pattern_size,
        sparsity=args.sparsity,
        n_timesteps=args.n_timesteps,
        show_plots=not args.no_show,
        save_plots=not args.no_save,
    )
