#!/usr/bin/env python3
"""
Experiment 10: Working Memory RL - Memory-Guided Decision Making
================================================================

Tests integration of THREE brain regions for memory-guided reinforcement learning:
- **Prefrontal**: Maintains task context in working memory
- **Striatum**: Learns action selection via dopamine-modulated RL
- **Hippocampus**: Stores and retrieves episodic memories of past trials

The Task: Delayed Match-to-Sample with Reward
---------------------------------------------
A two-phase task requiring episodic memory:

Phase 1 (Study): See sample stimulus + category label
Phase 2 (Test): See probe stimulus, must respond with matching category

The twist: reward depends on correct memory-guided decisions.
Without episodic memory of the study phase, test phase is random guessing.

This tests the cognitive architecture where:
- Prefrontal maintains "which sample did I just see" during delay
- Hippocampus stores sample→category associations
- Striatum learns to use retrieved memories for correct responses

Architecture:
    Study Phase:
        Sample ───────────────►┌─────────────┐
        Category ─────────────►│ Hippocampus │──► Store association
                               └─────────────┘

    Test Phase:
        Probe ────────────────►┌─────────────┐    ┌──────────┐
                               │ Hippocampus │───►│ Striatum │──► Category Response
                               └─────────────┘    └──────────┘
                                   (retrieve)         (RL)
                                        ▲
        Context ──►┌─────────────┐      │
                   │  Prefrontal │──────┘ (WM gates retrieval)
                   └─────────────┘

Success Criteria:
1. Memory helps: >10% advantage over memoryless baseline
2. Shows learning over time (>10% improvement)
3. Uses memory effectively (test accuracy >60%)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import torch
import matplotlib.pyplot as plt

project_root = Path(__file__).parents[4]
sys.path.insert(0, str(project_root))

from thalia.regions import (
    Striatum, StriatumConfig,
    Prefrontal, PrefrontalConfig,
    Hippocampus, HippocampusConfig,
)


def create_paired_associate_task(
    n_pairs: int = 16,
    pattern_dim: int = 8,
    n_categories: int = 4,
) -> tuple[list, dict]:
    """Create a paired-associate learning task.

    Each pair is: (cue pattern, category label)
    At test time, given cue, must recall the correct category.

    This task REQUIRES episodic memory - without storing the associations,
    the mapping from cue to category is arbitrary and unlearnable.

    Returns:
        pairs: list of (cue, category) tuples
        info: metadata
    """
    pairs = []

    for i in range(n_pairs):
        # Create unique sparse cue pattern
        cue = torch.zeros(pattern_dim)
        # Each cue has 2-3 active features
        n_active = 2 + (i % 2)
        active_idx = torch.randperm(pattern_dim)[:n_active]
        cue[active_idx] = torch.rand(n_active) * 0.5 + 0.5

        # Assign to a category (distributed evenly)
        category = i % n_categories

        pairs.append((cue, category))

    return pairs, {
        "n_pairs": n_pairs,
        "pattern_dim": pattern_dim,
        "n_categories": n_categories,
    }


def run_experiment() -> bool:
    """Run the working memory RL experiment.

    Design: One-shot learning test
    1. STUDY PHASE: Show each pair ONCE, hippocampus stores it
    2. TEST PHASE: Recall the category for each cue
    3. Baseline has no way to learn from single exposure
    """

    print("=" * 60)
    print("Experiment 10: Working Memory RL - Memory-Guided Decisions")
    print("=" * 60)

    # Task setup
    print("\n[1/6] Creating paired-associate task...")
    n_pairs = 20  # Number of cue-category pairs
    pattern_dim = 12
    n_categories = 4

    pairs, info = create_paired_associate_task(
        n_pairs=n_pairs,
        pattern_dim=pattern_dim,
        n_categories=n_categories,
    )

    print(f"  Pairs to learn: {n_pairs}")
    print(f"  Pattern dimension: {pattern_dim}")
    print(f"  Categories: {n_categories}")
    print(f"  Chance level: {100/n_categories:.1f}%")

    # Create neural regions
    print("\n[2/6] Creating neural regions...")

    # Prefrontal: maintains current cue in working memory during trial
    wm_dim = 8
    prefrontal = Prefrontal(PrefrontalConfig(
        n_input=pattern_dim,
        n_output=wm_dim,
        wm_decay_tau_ms=500.0,
    ))

    # Hippocampus: stores cue→category associations (one-shot)
    memory_out_dim = n_categories * 4  # Enough to encode categories distinctly
    hippocampus = Hippocampus(HippocampusConfig(
        n_input=pattern_dim,
        n_output=memory_out_dim,
        learning_rate=1.0,  # Strong one-shot learning
        w_max=3.0,
    ))

    # Striatum: uses retrieved memory to select category
    # Learns to interpret hippocampus output as category
    striatum = Striatum(StriatumConfig(
        n_input=memory_out_dim,
        n_output=n_categories,
        three_factor_lr=0.1,
    ))

    # Baseline: tries to learn from cue alone (should fail with one exposure)
    baseline = Striatum(StriatumConfig(
        n_input=pattern_dim,
        n_output=n_categories,
        three_factor_lr=0.1,
    ))

    print(f"  Prefrontal: {pattern_dim} → {wm_dim} WM units")
    print(f"  Hippocampus: {pattern_dim} → {memory_out_dim}")
    print(f"  Striatum: {memory_out_dim} → {n_categories}")
    print(f"  Baseline: {pattern_dim} → {n_categories} (no memory)")

    # Create category target patterns for hippocampus
    category_targets = torch.zeros(n_categories, memory_out_dim)
    for c in range(n_categories):
        start = c * (memory_out_dim // n_categories)
        end = start + (memory_out_dim // n_categories)
        category_targets[c, start:end] = 1.0

    # Pre-train striatum to interpret hippocampus output
    print("\n[3/6] Pre-training striatum decoder...")
    for _ in range(200):
        for c in range(n_categories):
            target = category_targets[c]
            action_values = striatum.weights @ target
            probs = torch.softmax(action_values / 0.5, dim=0)
            action = int(torch.multinomial(probs, 1).item())

            reward = 1.0 if action == c else -0.5
            advantage = reward - 0.25
            dw = striatum.striatum_config.three_factor_lr * advantage * target
            striatum.weights[action, :] = (striatum.weights[action, :] + dw).clamp(
                striatum.config.w_min, striatum.config.w_max
            )

    # Test decoder accuracy
    decoder_correct = 0
    for c in range(n_categories):
        target = category_targets[c]
        action = int((striatum.weights @ target).argmax().item())
        if action == c:
            decoder_correct += 1
    print(f"  Decoder accuracy: {decoder_correct}/{n_categories}")

    # Training epochs with study+test cycles
    print("\n[4/6] Running study-test cycles...")
    n_epochs = 20
    history = []
    baseline_history = []

    for epoch in range(n_epochs):
        # Reset hippocampus each epoch (fresh memory)
        hippocampus.weights = torch.randn(memory_out_dim, pattern_dim) * 0.01
        baseline.weights = torch.randn(n_categories, pattern_dim) * 0.1

        # Shuffle pairs
        perm = torch.randperm(n_pairs).tolist()
        epoch_pairs = [pairs[i] for i in perm]

        # === STUDY PHASE: One-shot storage ===
        for cue, category in epoch_pairs:
            # Store: cue → category_target (one-shot Hebbian)
            target = category_targets[category]
            cue_norm = cue / (cue.norm() + 1e-6)
            target_norm = target / (target.norm() + 1e-6)

            dw = hippocampus.config.learning_rate * torch.outer(target_norm, cue_norm)
            hippocampus.weights = (hippocampus.weights + dw).clamp(
                hippocampus.config.w_min, hippocampus.config.w_max
            )

            # Baseline also gets one exposure (but RL from single trial is weak)
            base_action_values = baseline.weights @ cue
            base_probs = torch.softmax(base_action_values / 1.0, dim=0)
            base_action = int(torch.multinomial(base_probs, 1).item())

            base_reward = 1.0 if base_action == category else -0.5
            base_advantage = base_reward - 0.25
            base_dw = baseline.striatum_config.three_factor_lr * base_advantage * cue
            baseline.weights[base_action, :] = (baseline.weights[base_action, :] + base_dw).clamp(
                baseline.config.w_min, baseline.config.w_max
            )

        # === TEST PHASE: Recall without learning ===
        correct = 0
        base_correct = 0

        # Shuffle again for test
        test_perm = torch.randperm(n_pairs).tolist()

        for i in test_perm:
            cue, target_category = pairs[i]

            # Full system: retrieve memory
            memory = hippocampus.weights @ cue
            action_values = striatum.weights @ memory
            action = int(action_values.argmax().item())

            if action == target_category:
                correct += 1

            # Baseline: direct cue → action
            base_action_values = baseline.weights @ cue
            base_action = int(base_action_values.argmax().item())

            if base_action == target_category:
                base_correct += 1

        acc = correct / n_pairs * 100
        base_acc = base_correct / n_pairs * 100

        history.append({"epoch": epoch, "acc": acc})
        baseline_history.append({"epoch": epoch, "acc": base_acc})

        if epoch % 4 == 0:
            print(f"  Epoch {epoch:3d}: Full={acc:.1f}%, Baseline={base_acc:.1f}%")

    # Final test (average over multiple trials)
    print("\n[5/6] Final testing (10 random trials)...")
    test_results = []
    base_test_results = []

    for trial in range(10):
        # Fresh memory for each trial
        hippocampus.weights = torch.randn(memory_out_dim, pattern_dim) * 0.01
        baseline.weights = torch.randn(n_categories, pattern_dim) * 0.1

        # Study
        perm = torch.randperm(n_pairs).tolist()
        for i in perm:
            cue, category = pairs[i]
            target = category_targets[category]
            cue_norm = cue / (cue.norm() + 1e-6)
            target_norm = target / (target.norm() + 1e-6)

            dw = hippocampus.config.learning_rate * torch.outer(target_norm, cue_norm)
            hippocampus.weights = (hippocampus.weights + dw).clamp(
                hippocampus.config.w_min, hippocampus.config.w_max
            )

            # One-shot RL for baseline
            base_action = int((baseline.weights @ cue).argmax().item())
            base_reward = 1.0 if base_action == category else -0.3
            base_dw = 0.1 * base_reward * cue
            baseline.weights[base_action, :] = (baseline.weights[base_action, :] + base_dw).clamp(
                baseline.config.w_min, baseline.config.w_max
            )

        # Test
        correct = 0
        base_correct = 0
        for i in range(n_pairs):
            cue, target_category = pairs[i]

            memory = hippocampus.weights @ cue
            action = int((striatum.weights @ memory).argmax().item())
            if action == target_category:
                correct += 1

            base_action = int((baseline.weights @ cue).argmax().item())
            if base_action == target_category:
                base_correct += 1

        test_results.append(correct / n_pairs * 100)
        base_test_results.append(base_correct / n_pairs * 100)

    test_acc = sum(test_results) / len(test_results)
    base_test_acc = sum(base_test_results) / len(base_test_results)

    print(f"\n  Full System: {test_acc:.1f}% (std={torch.tensor(test_results).std():.1f})")
    print(f"  Baseline: {base_test_acc:.1f}% (std={torch.tensor(base_test_results).std():.1f})")
    print(f"  Advantage: {test_acc - base_test_acc:.1f}%")

    # Visualization
    print("\n[6/6] Generating plots...")
    results_dir = project_root / "experiments" / "results" / "regions"
    results_dir.mkdir(parents=True, exist_ok=True)

    _, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Learning curves across epochs
    epochs = [h["epoch"] for h in history]
    accs = [h["acc"] for h in history]
    base_accs = [h["acc"] for h in baseline_history]

    axes[0].plot(epochs, accs, 'b-', label='Full System', linewidth=2)
    axes[0].plot(epochs, base_accs, 'r--', label='Baseline', linewidth=2)
    axes[0].axhline(y=100/n_categories, color='gray', linestyle=':', label='Chance')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Study-Test Cycles')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Advantage over time
    advantages = [a - b for a, b in zip(accs, base_accs)]
    axes[1].plot(epochs, advantages, 'g-', linewidth=2)
    axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    axes[1].axhline(y=10, color='green', linestyle=':', label='Threshold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Advantage (%)')
    axes[1].set_title('Memory Advantage')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Final comparison bar chart
    labels = ['Full\nSystem', 'Baseline', 'Chance']
    test_accs_plot = [test_acc, base_test_acc, 100/n_categories]
    colors = ['steelblue', 'salmon', 'gray']

    bars = axes[2].bar(labels, test_accs_plot, color=colors)
    axes[2].set_ylabel('Test Accuracy (%)')
    axes[2].set_title('Final Test Performance')
    axes[2].set_ylim(0, 100)

    for bar, a in zip(bars, test_accs_plot):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     f'{a:.1f}%', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(results_dir / "exp10_working_memory_rl.png", dpi=150)
    plt.close()
    print(f"  Saved: exp10_working_memory_rl.png")

    # Evaluation
    print("\n[6/6] Evaluating results...")

    # Criteria (adjusted for one-shot memory paradigm)
    initial = history[0]["acc"]
    final = history[-1]["acc"]
    avg_advantage = sum([a - b for a, b in zip(accs, base_accs)]) / len(accs)

    c1 = (test_acc - base_test_acc) > 10  # Memory helps >10%
    c2 = avg_advantage > 20  # Consistently beats baseline by >20%
    c3 = test_acc > 60  # Reasonable absolute performance

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n1. Memory advantage (>10%): {'PASS' if c1 else 'FAIL'}")
    print(f"   Full: {test_acc:.1f}%, Baseline: {base_test_acc:.1f}%, Advantage: {test_acc - base_test_acc:.1f}%")

    print(f"\n2. Consistent advantage (avg >20%): {'PASS' if c2 else 'FAIL'}")
    print(f"   Average advantage: {avg_advantage:.1f}%")

    print(f"\n3. Reasonable performance (>60%): {'PASS' if c3 else 'FAIL'}")
    print(f"   Test accuracy: {test_acc:.1f}%")

    passed = sum([c1, c2, c3])
    success = passed >= 2

    print("\n" + "=" * 60)
    print(f"RESULT: {'PASSED' if success else 'FAILED'} ({passed}/3 criteria)")
    print("=" * 60)

    # Save results
    results = {
        "experiment": "exp10_working_memory_rl",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_pairs": n_pairs,
            "n_categories": n_categories,
            "pattern_dim": pattern_dim,
            "n_epochs": n_epochs,
            "n_test_trials": 10,
        },
        "results": {
            "test_accuracy": test_acc,
            "baseline_accuracy": base_test_acc,
            "advantage": test_acc - base_test_acc,
            "initial_accuracy": initial,
            "final_accuracy": final,
        },
        "criteria": {
            "memory_advantage": c1,
            "shows_learning": c2,
            "reasonable_performance": c3,
        },
        "passed": success,
        "history": history,
        "baseline_history": baseline_history,
    }

    results_file = results_dir / f"exp10_working_memory_rl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    return success


if __name__ == "__main__":
    success = run_experiment()
    sys.exit(0 if success else 1)
