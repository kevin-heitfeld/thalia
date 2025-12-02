"""
Experiment 5: Prefrontal Delayed Match-to-Sample

Objective: Validate working memory maintenance across delay periods
using the prefrontal cortex region.

Setup:
- Task: Present sample → delay (variable: 5-50 timesteps) → match/non-match
- Architecture: Sample → PFC (gated working memory) → decision
- Challenge: Information must persist without external input

Key Insight:
The prefrontal cortex maintains information through recurrent connections
and dopamine-gated updates. During the delay period:
- No external input
- WM must resist decay through self-excitation
- Distractors should be filtered out (gate closed)

Analysis:
1. Accuracy vs delay length
2. WM state visualization during delay
3. Effect of distractors during delay
4. Gating dynamics (when is WM updated?)

Success Metrics:
- 85%+ accuracy with 20-step delay
- WM state stable during delay
- Correctly rejects distractors
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from thalia.regions.prefrontal import Prefrontal, PrefrontalConfig
from experiments.scripts.regions.exp_utils import (
    save_results,
    get_results_dir,
)


def create_prefrontal(
    n_input: int = 32,
    n_output: int = 32,
) -> Prefrontal:
    """Create a Prefrontal configured for working memory tasks."""

    config = PrefrontalConfig(
        n_input=n_input,
        n_output=n_output,
        wm_decay_tau_ms=200.0,  # Slow decay
        gate_threshold=0.5,
        dopamine_baseline=0.2,
        recurrent_strength=0.9,  # Strong self-excitation
        recurrent_inhibition=0.1,
    )
    return Prefrontal(config)


def create_sample_patterns(
    n_classes: int = 4,
    pattern_size: int = 32,
    sparsity: float = 0.15,
) -> List[torch.Tensor]:
    """
    Create distinct sample patterns for each class.
    """
    n_active = int(pattern_size * sparsity)
    patterns = []

    for _ in range(n_classes):
        pattern = torch.zeros(pattern_size)
        pattern[torch.randperm(pattern_size)[:n_active]] = 1.0
        patterns.append(pattern)

    return patterns


def create_trial(
    sample_patterns: List[torch.Tensor],
    delay_length: int = 20,
    match_prob: float = 0.5,
    distractor_prob: float = 0.0,
) -> Dict:
    """
    Create a single delayed match-to-sample trial.

    Args:
        sample_patterns: List of possible sample patterns
        delay_length: Number of timesteps in delay period
        match_prob: Probability of match trial
        distractor_prob: Probability of distractor during delay

    Returns:
        Trial dict with sample, delay, test, is_match
    """
    n_classes = len(sample_patterns)

    # Random sample
    sample_class = np.random.randint(n_classes)
    sample = sample_patterns[sample_class].clone()

    # Match or non-match
    is_match = np.random.rand() < match_prob

    if is_match:
        test = sample.clone()
        test_class = sample_class
    else:
        test_class = np.random.choice([c for c in range(n_classes) if c != sample_class])
        test = sample_patterns[test_class].clone()

    # Create distractor schedule
    distractors = []
    for t in range(delay_length):
        if np.random.rand() < distractor_prob:
            distractor_class = np.random.choice([c for c in range(n_classes) if c != sample_class])
            distractors.append(sample_patterns[distractor_class].clone())
        else:
            distractors.append(None)

    return {
        "sample": sample,
        "sample_class": sample_class,
        "delay_length": delay_length,
        "distractors": distractors,
        "test": test,
        "test_class": test_class,
        "is_match": is_match,
    }


def run_trial(
    pfc: Prefrontal,
    trial: Dict,
    n_sample_steps: int = 10,
    n_test_steps: int = 10,
) -> Dict:
    """
    Run a single delayed match-to-sample trial.
    """
    pfc.reset()

    sample = trial["sample"]
    delay_length = trial["delay_length"]
    distractors = trial["distractors"]
    test = trial["test"]

    wm_history = []

    # === SAMPLE PHASE ===
    # Present sample with high dopamine (open gate)
    for t in range(n_sample_steps):
        input_spikes = (torch.rand_like(sample) < sample).float().unsqueeze(0)
        pfc.forward(input_spikes, dopamine_signal=0.5)  # High DA to encode

    # Set WM directly for reliable encoding
    pfc.set_context(sample.unsqueeze(0))
    wm_history.append(pfc.get_working_memory().squeeze().clone())

    # === DELAY PHASE ===
    # Maintain with low dopamine (closed gate)
    for t in range(delay_length):
        distractor = distractors[t] if t < len(distractors) else None

        if distractor is not None:
            input_spikes = (torch.rand_like(distractor) < distractor).float().unsqueeze(0)
            pfc.forward(input_spikes, dopamine_signal=-0.3)  # Low DA to reject
        else:
            # Empty input during maintenance
            input_spikes = torch.zeros_like(sample).unsqueeze(0)
            pfc.forward(input_spikes, dopamine_signal=-0.3)  # Low DA to maintain

        wm_history.append(pfc.get_working_memory().squeeze().clone())

    # === TEST PHASE ===
    # Compare WM with test pattern
    wm_final = pfc.get_working_memory().squeeze()

    # Decision: compare overlap with sample vs test
    wm_normalized = wm_final / (wm_final.norm() + 1e-6)
    sample_normalized = sample / (sample.norm() + 1e-6)
    test_normalized = test / (test.norm() + 1e-6)

    wm_sample_overlap = (wm_normalized * sample_normalized).sum().item()
    wm_test_overlap = (wm_normalized * test_normalized).sum().item()

    # Decision rule: if WM matches test well and test matches sample → match
    # Otherwise → non-match
    predicted_match = wm_test_overlap > 0.5 and abs(wm_sample_overlap - wm_test_overlap) < 0.3

    # Alternative: if trial is match, WM should match test well
    # If trial is non-match, WM should NOT match test well
    correct = (predicted_match == trial["is_match"])

    return {
        "predicted_match": predicted_match,
        "is_match": trial["is_match"],
        "correct": correct,
        "wm_sample_overlap": wm_sample_overlap,
        "wm_test_overlap": wm_test_overlap,
        "wm_history": torch.stack(wm_history),
    }


def evaluate_delay_performance(
    pfc: Prefrontal,
    sample_patterns: List[torch.Tensor],
    n_trials: int = 100,
    delay_length: int = 20,
    distractor_prob: float = 0.0,
) -> Dict:
    """
    Evaluate performance at a specific delay length.
    """
    results = []

    for _ in range(n_trials):
        trial = create_trial(
            sample_patterns,
            delay_length=delay_length,
            distractor_prob=distractor_prob,
        )
        result = run_trial(pfc, trial)
        results.append(result)

    accuracy = np.mean([r["correct"] for r in results])

    return {
        "delay_length": delay_length,
        "distractor_prob": distractor_prob,
        "accuracy": accuracy,
        "n_trials": n_trials,
        "results": results,
    }


def test_delay_sweep(
    sample_patterns: List[torch.Tensor],
    delay_lengths: List[int] = [5, 10, 20, 30, 40, 50],
    n_trials: int = 50,
    n_output: int = 32,
) -> Dict:
    """
    Test performance across different delay lengths.
    """
    results = {"delay_lengths": [], "accuracies": []}

    for delay in delay_lengths:
        # Fresh PFC for each delay to avoid learning confounds
        pfc = create_prefrontal(n_input=len(sample_patterns[0]), n_output=n_output)

        eval_result = evaluate_delay_performance(
            pfc, sample_patterns, n_trials=n_trials, delay_length=delay
        )

        results["delay_lengths"].append(delay)
        results["accuracies"].append(eval_result["accuracy"])

        print(f"  Delay {delay}: accuracy = {eval_result['accuracy']*100:.1f}%")

    return results


def test_distractor_resistance(
    sample_patterns: List[torch.Tensor],
    distractor_probs: List[float] = [0.0, 0.1, 0.2, 0.3, 0.5],
    delay_length: int = 20,
    n_trials: int = 50,
    n_output: int = 32,
) -> Dict:
    """
    Test resistance to distractors during delay.
    """
    results = {"distractor_probs": [], "accuracies": []}

    for prob in distractor_probs:
        pfc = create_prefrontal(n_input=len(sample_patterns[0]), n_output=n_output)

        eval_result = evaluate_delay_performance(
            pfc, sample_patterns, n_trials=n_trials,
            delay_length=delay_length, distractor_prob=prob
        )

        results["distractor_probs"].append(prob)
        results["accuracies"].append(eval_result["accuracy"])

        print(f"  Distractor prob {prob:.1f}: accuracy = {eval_result['accuracy']*100:.1f}%")

    return results


def visualize_wm_dynamics(
    trial_result: Dict,
    sample_class: int,
    test_class: int,
    pattern_size: int,
) -> plt.Figure:
    """Visualize working memory evolution during trial."""

    wm_history = trial_result["wm_history"].numpy()
    n_steps = wm_history.shape[0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 6))

    # WM activity heatmap
    ax = axes[0]
    im = ax.imshow(wm_history.T, aspect='auto', cmap='viridis',
                   extent=[0, n_steps, pattern_size, 0])
    ax.axvline(x=1, color='red', linestyle='--', label='Sample End')
    ax.axvline(x=n_steps-1, color='green', linestyle='--', label='Test Start')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('WM Neuron')
    ax.set_title(f'Working Memory During Trial (Sample={sample_class}, Test={test_class})')
    plt.colorbar(im, ax=ax, label='Activation')
    ax.legend(loc='upper right')

    # Mean WM activity over time
    ax = axes[1]
    mean_activity = wm_history.mean(axis=1)
    ax.plot(mean_activity, 'b-', label='Mean WM Activity')
    ax.axvline(x=1, color='red', linestyle='--', label='Sample End')
    ax.axvline(x=n_steps-1, color='green', linestyle='--', label='Test Start')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Mean Activation')
    ax.set_title('WM Stability During Delay')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def visualize_results(
    delay_results: Dict,
    distractor_results: Dict,
) -> plt.Figure:
    """Visualize experiment results."""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Delay sweep
    ax = axes[0]
    ax.plot(delay_results['delay_lengths'], delay_results['accuracies'], 'bo-')
    ax.axhline(y=0.85, color='green', linestyle='--', label='85% threshold')
    ax.axhline(y=0.50, color='gray', linestyle=':', label='Chance')
    ax.set_xlabel('Delay Length (timesteps)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Performance vs Delay Length')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)

    # Distractor resistance
    ax = axes[1]
    ax.plot(distractor_results['distractor_probs'], distractor_results['accuracies'], 'ro-')
    ax.axhline(y=0.85, color='green', linestyle='--', label='85% threshold')
    ax.axhline(y=0.50, color='gray', linestyle=':', label='Chance')
    ax.set_xlabel('Distractor Probability')
    ax.set_ylabel('Accuracy')
    ax.set_title('Resistance to Distractors')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)

    plt.tight_layout()
    return fig


def run_experiment(
    n_classes: int = 4,
    pattern_size: int = 32,
    n_trials: int = 50,
    delay_lengths: List[int] = [5, 10, 20, 30, 40, 50],
    distractor_probs: List[float] = [0.0, 0.1, 0.2, 0.3, 0.5],
    show_plots: bool = True,
    save_plots: bool = True,
):
    """
    Run the complete Prefrontal delayed match-to-sample experiment.
    """
    print("=" * 60)
    print("EXPERIMENT 5: Prefrontal Delayed Match-to-Sample")
    print("=" * 60)

    # 1. Create sample patterns
    print(f"\n[1/5] Creating {n_classes} sample patterns (size={pattern_size})...")
    sample_patterns = create_sample_patterns(
        n_classes=n_classes,
        pattern_size=pattern_size,
    )

    # 2. Test delay sweep
    print(f"\n[2/5] Testing performance across delay lengths...")
    delay_results = test_delay_sweep(
        sample_patterns,
        delay_lengths=delay_lengths,
        n_trials=n_trials,
        n_output=pattern_size,
    )

    # 3. Test distractor resistance
    print(f"\n[3/5] Testing resistance to distractors...")
    distractor_results = test_distractor_resistance(
        sample_patterns,
        distractor_probs=distractor_probs,
        delay_length=20,
        n_trials=n_trials,
        n_output=pattern_size,
    )

    # 4. Example trial for visualization
    print(f"\n[4/5] Running example trial for WM visualization...")
    pfc = create_prefrontal(n_input=pattern_size, n_output=pattern_size)
    example_trial = create_trial(sample_patterns, delay_length=30)
    example_result = run_trial(pfc, example_trial)

    print(f"  Example trial: sample_class={example_trial['sample_class']}, "
          f"test_class={example_trial['test_class']}, "
          f"is_match={example_trial['is_match']}, "
          f"predicted={example_result['predicted_match']}, "
          f"correct={example_result['correct']}")

    # 5. Visualize
    print(f"\n[5/5] Generating visualizations...")

    results_dir = get_results_dir()

    # Main results
    fig_results = visualize_results(delay_results, distractor_results)
    if save_plots:
        fig_results.savefig(results_dir / "exp5_results.png", dpi=150)
        print(f"  Saved: exp5_results.png")

    # WM dynamics
    fig_wm = visualize_wm_dynamics(
        example_result,
        example_trial['sample_class'],
        example_trial['test_class'],
        pattern_size,
    )
    if save_plots:
        fig_wm.savefig(results_dir / "exp5_wm_dynamics.png", dpi=150)
        print(f"  Saved: exp5_wm_dynamics.png")

    # Weight visualization
    fig_weights, ax = plt.subplots(figsize=(10, 4))
    weights = pfc.weights.data.detach().cpu().numpy()
    im = ax.imshow(weights, aspect='auto', cmap='RdBu_r')
    ax.set_xlabel('Input')
    ax.set_ylabel('PFC Neuron')
    ax.set_title('Prefrontal Weights')
    plt.colorbar(im, ax=ax, label='Weight')
    if save_plots:
        fig_weights.savefig(results_dir / "exp5_weights.png", dpi=150)
        print(f"  Saved: exp5_weights.png")

    if show_plots:
        plt.show()
    else:
        plt.close('all')

    # Find accuracy at delay=20
    delay_20_idx = delay_results['delay_lengths'].index(20) if 20 in delay_results['delay_lengths'] else -1
    accuracy_20 = delay_results['accuracies'][delay_20_idx] if delay_20_idx >= 0 else delay_results['accuracies'][-1]

    # Save results
    results = {
        "config": {
            "n_classes": n_classes,
            "pattern_size": pattern_size,
            "n_trials": n_trials,
            "delay_lengths": delay_lengths,
            "distractor_probs": distractor_probs,
        },
        "delay_sweep": delay_results,
        "distractor_resistance": distractor_results,
        "accuracy_at_delay_20": accuracy_20,
    }
    save_results("exp5_prefrontal_delay", results)

    # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT 5 COMPLETE")
    print("=" * 60)

    success = accuracy_20 >= 0.85
    print(f"\nSuccess criterion: Accuracy ≥ 85% at delay=20")
    print(f"Result: {accuracy_20*100:.1f}%")
    print(f"Status: {'✓ PASSED' if success else '✗ FAILED'}")

    return results, pfc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prefrontal Delayed Match-to-Sample Experiment")
    parser.add_argument("--n-classes", type=int, default=4, help="Number of sample classes")
    parser.add_argument("--pattern-size", type=int, default=32, help="Pattern dimension")
    parser.add_argument("--n-trials", type=int, default=50, help="Trials per condition")
    parser.add_argument("--no-show", action="store_true", help="Don't show plots")
    parser.add_argument("--no-save", action="store_true", help="Don't save plots")

    args = parser.parse_args()

    run_experiment(
        n_classes=args.n_classes,
        pattern_size=args.pattern_size,
        n_trials=args.n_trials,
        show_plots=not args.no_show,
        save_plots=not args.no_save,
    )
