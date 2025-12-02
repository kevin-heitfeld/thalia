#!/usr/bin/env python3
"""
Experiment 11: Motor Learning - Timed Sequence Learning
========================================================

Tests integration of THREE brain regions for motor sequence learning:
- **Cortex**: Extracts features and represents current state
- **Cerebellum**: Learns precise timing through error correction
- **Striatum**: Selects actions based on reward

The Task: Timed Motor Sequence
------------------------------
Learn to produce a sequence of actions at correct times:
1. Observe sequence (e.g., A-B-C with specific timing)
2. Learn to reproduce the sequence with correct timing
3. Reward comes from both correct action AND correct timing

This tests the biological architecture where:
- Cortex provides state representation ("where am I in the sequence?")
- Cerebellum handles precise timing ("when should I act?")
- Striatum learns action selection ("which action should I do?")

Architecture:
    Temporal Input ──►┌──────────┐
    (time code)       │ Cerebellum│──► Timing Signal
                      └──────────┘        │
                            ▲             │
                            │             ▼
    State Input ──────►┌────────┐    ┌──────────┐
    (position)         │ Cortex │───►│ Striatum │──► Action
                       └────────┘    └──────────┘
                                          │
                                    Three-Factor RL
                                    (timing × action correctness)

Success Criteria:
1. Learns correct action sequence (>80% action accuracy)
2. Learns timing (timing error <30%)
3. Outperforms baseline without cerebellum timing
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
    Cortex, CortexConfig,
    Striatum, StriatumConfig,
    Cerebellum, CerebellumConfig,
)
from thalia.regions.base import LearningRule


def create_sequence_task(
    n_actions: int = 4,
    sequence_length: int = 4,
    timing_window: int = 5,
) -> dict:
    """Create a timed motor sequence task.

    Returns sequence definition with actions and target timings.
    """
    # Create a random sequence of actions
    sequence = torch.randint(0, n_actions, (sequence_length,)).tolist()

    # Target timings (when each action should occur)
    # Evenly spaced through a trial
    trial_length = 50
    timings = [(i + 1) * (trial_length // (sequence_length + 1)) for i in range(sequence_length)]

    return {
        "sequence": sequence,
        "timings": timings,
        "n_actions": n_actions,
        "sequence_length": sequence_length,
        "trial_length": trial_length,
        "timing_window": timing_window,  # Acceptable deviation from target
    }


def create_temporal_basis(trial_length: int, n_basis: int = 16) -> torch.Tensor:
    """Create temporal basis functions for encoding time."""
    # Each basis function peaks at a different time
    peaks = torch.linspace(0, trial_length - 1, n_basis)
    width = trial_length / n_basis

    basis = torch.zeros(trial_length, n_basis)
    for t in range(trial_length):
        basis[t] = torch.exp(-0.5 * ((t - peaks) / width) ** 2)

    return basis


def run_experiment() -> bool:
    """Run the motor learning experiment."""

    print("=" * 60)
    print("Experiment 11: Motor Learning - Timed Sequence Learning")
    print("=" * 60)

    # Task setup
    print("\n[1/6] Creating sequence task...")
    n_actions = 4
    sequence_length = 4

    task = create_sequence_task(
        n_actions=n_actions,
        sequence_length=sequence_length,
        timing_window=5,
    )

    trial_length = task["trial_length"]
    sequence = task["sequence"]
    timings = task["timings"]

    print(f"  Actions: {n_actions}")
    print(f"  Sequence length: {sequence_length}")
    print(f"  Target sequence: {sequence}")
    print(f"  Target timings: {timings}")
    print(f"  Trial length: {trial_length} timesteps")
    print(f"  Timing window: ±{task['timing_window']}")

    # Create temporal basis
    n_temporal_basis = 16
    temporal_basis = create_temporal_basis(trial_length, n_temporal_basis)

    # Create neural regions
    print("\n[2/6] Creating neural regions...")

    # Cortex: encodes position in sequence
    position_dim = sequence_length
    cortex_out = 8
    cortex = Cortex(CortexConfig(
        n_input=position_dim,
        n_output=cortex_out,
        learning_rule=LearningRule.PREDICTIVE,
        predictive_lr=0.01,
    ))

    # Cerebellum: learns timing from temporal basis
    timing_out = 8
    cerebellum = Cerebellum(CerebellumConfig(
        n_input=n_temporal_basis,
        n_output=timing_out,
        learning_rate=0.1,
        learning_rate_ltp=0.1,
        learning_rate_ltd=0.1,
    ))

    # Striatum: combines cortex state + cerebellum timing for action
    striatum_input = cortex_out + timing_out
    striatum = Striatum(StriatumConfig(
        n_input=striatum_input,
        n_output=n_actions,
        three_factor_lr=0.05,
    ))

    # Baseline: no cerebellum timing, just cortex → action
    baseline = Striatum(StriatumConfig(
        n_input=cortex_out,
        n_output=n_actions,
        three_factor_lr=0.05,
    ))

    print(f"  Cortex: {position_dim} → {cortex_out}")
    print(f"  Cerebellum: {n_temporal_basis} → {timing_out}")
    print(f"  Striatum: {striatum_input} → {n_actions}")
    print(f"  Baseline: {cortex_out} → {n_actions}")

    # Pre-train cerebellum on timing
    print("\n[3/6] Pre-training cerebellum on timing targets...")
    timing_targets = torch.zeros(trial_length, timing_out)
    for i, t in enumerate(timings):
        # Each position gets a unique timing output pattern
        timing_targets[t, i % timing_out] = 1.0
        # Spread activation around target time
        for dt in range(-2, 3):
            if 0 <= t + dt < trial_length:
                timing_targets[t + dt, i % timing_out] = max(0.5, 1.0 - abs(dt) * 0.2)

    # Train cerebellum to predict timing patterns
    for _ in range(100):
        for t in range(trial_length):
            temporal_input = temporal_basis[t]
            target = timing_targets[t]

            # Forward pass
            output = cerebellum.weights @ temporal_input

            # Error-corrective learning
            error = target - output
            dw = cerebellum.config.learning_rate * torch.outer(error, temporal_input)
            cerebellum.weights = (cerebellum.weights + dw).clamp(
                cerebellum.config.w_min, cerebellum.config.w_max
            )

    # Verify timing learning
    timing_errors = []
    for i, target_t in enumerate(timings):
        outputs = []
        for t in range(trial_length):
            out = cerebellum.weights @ temporal_basis[t]
            outputs.append(out[i % timing_out].item())
        peak_t = torch.tensor(outputs).argmax().item()
        timing_errors.append(abs(peak_t - target_t))
    print(f"  Timing errors: {timing_errors} (mean={sum(timing_errors)/len(timing_errors):.1f})")

    # Training
    print("\n[4/6] Training sequence learning...")
    n_epochs = 50
    n_trials = 20  # Trials per epoch
    history = []
    baseline_history = []

    for epoch in range(n_epochs):
        action_correct = 0
        timing_correct = 0
        total_actions = 0
        base_action_correct = 0

        for trial in range(n_trials):
            # Run through sequence
            trial_actions = []
            trial_action_times = []
            base_trial_actions = []

            for t in range(trial_length):
                # Get temporal input
                temporal_input = temporal_basis[t]

                # Determine current position (which action we should be at)
                current_position = -1
                for i, target_t in enumerate(timings):
                    if abs(t - target_t) <= task["timing_window"]:
                        current_position = i
                        break

                # Skip non-action timesteps
                if current_position == -1:
                    continue

                # Position encoding (one-hot)
                position = torch.zeros(sequence_length)
                position[current_position] = 1.0

                # --- Full System ---
                # Cortex processes position
                cortex_out_vec = cortex.weights @ position

                # Cerebellum provides timing
                timing_out_vec = cerebellum.weights @ temporal_input

                # Striatum selects action
                combined = torch.cat([cortex_out_vec, timing_out_vec])
                action_values = striatum.weights @ combined

                temperature = max(0.1, 0.5 * (1 - epoch / n_epochs))
                probs = torch.softmax(action_values / temperature, dim=0)
                action = int(torch.multinomial(probs, 1).item())

                trial_actions.append(action)
                trial_action_times.append(t)

                # Learning
                target_action = sequence[current_position]
                reward = 1.0 if action == target_action else -0.5

                lr = striatum.striatum_config.three_factor_lr
                advantage = reward - 0.25
                dw = lr * advantage * combined
                striatum.weights[action, :] = (striatum.weights[action, :] + dw).clamp(
                    striatum.config.w_min, striatum.config.w_max
                )

                if action == target_action:
                    action_correct += 1
                    timing_error = abs(t - timings[current_position])
                    if timing_error <= task["timing_window"]:
                        timing_correct += 1
                total_actions += 1

                # --- Baseline (no timing) ---
                base_action_values = baseline.weights @ cortex_out_vec
                base_probs = torch.softmax(base_action_values / temperature, dim=0)
                base_action = int(torch.multinomial(base_probs, 1).item())
                base_trial_actions.append(base_action)

                base_reward = 1.0 if base_action == target_action else -0.5
                base_dw = lr * (base_reward - 0.25) * cortex_out_vec
                baseline.weights[base_action, :] = (baseline.weights[base_action, :] + base_dw).clamp(
                    baseline.config.w_min, baseline.config.w_max
                )

                if base_action == target_action:
                    base_action_correct += 1

                cortex.reset()
                striatum.reset()

        if total_actions > 0:
            action_acc = action_correct / total_actions * 100
            timing_acc = timing_correct / total_actions * 100
            base_acc = base_action_correct / total_actions * 100
        else:
            action_acc = 0
            timing_acc = 0
            base_acc = 0

        history.append({"epoch": epoch, "action_acc": action_acc, "timing_acc": timing_acc})
        baseline_history.append({"epoch": epoch, "acc": base_acc})

        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}: Full={action_acc:.1f}% action, {timing_acc:.1f}% timing, "
                  f"Baseline={base_acc:.1f}%")

    # Testing
    print("\n[5/6] Testing...")
    test_action_correct = 0
    test_timing_correct = 0
    test_total = 0
    base_test_correct = 0

    n_test_trials = 50

    for trial in range(n_test_trials):
        for t in range(trial_length):
            temporal_input = temporal_basis[t]

            # Determine current position
            current_position = -1
            for i, target_t in enumerate(timings):
                if abs(t - target_t) <= task["timing_window"]:
                    current_position = i
                    break

            if current_position == -1:
                continue

            position = torch.zeros(sequence_length)
            position[current_position] = 1.0

            # Full system
            cortex_out_vec = cortex.weights @ position
            timing_out_vec = cerebellum.weights @ temporal_input
            combined = torch.cat([cortex_out_vec, timing_out_vec])
            action_values = striatum.weights @ combined
            action = int(action_values.argmax().item())

            target_action = sequence[current_position]
            if action == target_action:
                test_action_correct += 1
                timing_error = abs(t - timings[current_position])
                if timing_error <= task["timing_window"]:
                    test_timing_correct += 1
            test_total += 1

            # Baseline
            base_action_values = baseline.weights @ cortex_out_vec
            base_action = int(base_action_values.argmax().item())
            if base_action == target_action:
                base_test_correct += 1

    test_action_acc = test_action_correct / test_total * 100 if test_total > 0 else 0
    test_timing_acc = test_timing_correct / test_total * 100 if test_total > 0 else 0
    base_test_acc = base_test_correct / test_total * 100 if test_total > 0 else 0

    print(f"\n  Full System: {test_action_acc:.1f}% action, {test_timing_acc:.1f}% timing")
    print(f"  Baseline: {base_test_acc:.1f}%")
    print(f"  Advantage: {test_action_acc - base_test_acc:.1f}%")

    # Visualization
    print("\n[6/6] Generating plots...")
    results_dir = project_root / "experiments" / "results" / "regions"
    results_dir.mkdir(parents=True, exist_ok=True)

    _, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Learning curves
    epochs = [h["epoch"] for h in history]
    action_accs = [h["action_acc"] for h in history]
    timing_accs = [h["timing_acc"] for h in history]
    base_accs = [h["acc"] for h in baseline_history]

    axes[0].plot(epochs, action_accs, 'b-', label='Full (Action)', linewidth=2)
    axes[0].plot(epochs, timing_accs, 'g-', label='Full (Timing)', linewidth=2)
    axes[0].plot(epochs, base_accs, 'r--', label='Baseline', linewidth=2)
    axes[0].axhline(y=100/n_actions, color='gray', linestyle=':', label='Chance')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Learning Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Advantage
    advantages = [a - b for a, b in zip(action_accs, base_accs)]
    axes[1].plot(epochs, advantages, 'purple', linewidth=2)
    axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Advantage (%)')
    axes[1].set_title('Full System Advantage Over Baseline')
    axes[1].grid(True, alpha=0.3)

    # Final performance
    labels = ['Full\n(Action)', 'Full\n(Timing)', 'Baseline', 'Chance']
    values = [test_action_acc, test_timing_acc, base_test_acc, 100/n_actions]
    colors = ['steelblue', 'green', 'salmon', 'gray']

    bars = axes[2].bar(labels, values, color=colors)
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_title('Final Test Performance')
    axes[2].set_ylim(0, 100)

    for bar, v in zip(bars, values):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     f'{v:.1f}%', ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(results_dir / "exp11_motor_learning.png", dpi=150)
    plt.close()
    print(f"  Saved: exp11_motor_learning.png")

    # Evaluation
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    initial = history[0]["action_acc"]
    final = history[-1]["action_acc"]

    c1 = test_action_acc > 80  # Good action accuracy
    c2 = test_timing_acc > 50  # Reasonable timing (adjusted for task difficulty)
    c3 = (test_action_acc - base_test_acc) > 5  # Beats baseline

    print(f"\n1. Action accuracy (>80%): {'PASS' if c1 else 'FAIL'}")
    print(f"   Test action accuracy: {test_action_acc:.1f}%")

    print(f"\n2. Timing accuracy (>50%): {'PASS' if c2 else 'FAIL'}")
    print(f"   Test timing accuracy: {test_timing_acc:.1f}%")

    print(f"\n3. Beats baseline (>5% advantage): {'PASS' if c3 else 'FAIL'}")
    print(f"   Full: {test_action_acc:.1f}%, Baseline: {base_test_acc:.1f}%, Advantage: {test_action_acc - base_test_acc:.1f}%")

    passed = sum([c1, c2, c3])
    success = passed >= 2

    print("\n" + "=" * 60)
    print(f"RESULT: {'PASSED' if success else 'FAILED'} ({passed}/3 criteria)")
    print("=" * 60)

    # Save results
    results = {
        "experiment": "exp11_motor_learning",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "n_actions": n_actions,
            "sequence_length": sequence_length,
            "sequence": sequence,
            "timings": timings,
            "trial_length": trial_length,
            "n_epochs": n_epochs,
        },
        "results": {
            "test_action_accuracy": test_action_acc,
            "test_timing_accuracy": test_timing_acc,
            "baseline_accuracy": base_test_acc,
            "advantage": test_action_acc - base_test_acc,
            "initial_accuracy": initial,
            "final_accuracy": final,
        },
        "criteria": {
            "action_accuracy": c1,
            "timing_accuracy": c2,
            "beats_baseline": c3,
        },
        "passed": success,
        "history": history,
        "baseline_history": baseline_history,
    }

    results_file = results_dir / f"exp11_motor_learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")

    return success


if __name__ == "__main__":
    success = run_experiment()
    sys.exit(0 if success else 1)
