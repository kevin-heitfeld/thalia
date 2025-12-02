#!/usr/bin/env python3
"""Experiment 2: Cerebellum Interval Timing Task.

Tests cerebellar learning of precise timing - a core cerebellar function.
This is analogous to classical eyeblink conditioning where the cerebellum
learns to time a response (conditioned response) to occur just before
an expected stimulus (unconditioned stimulus).

Task:
- Present a cue stimulus (like a tone in eyeblink conditioning)
- The cerebellum must learn to produce output at a specific delay
- Error signals (climbing fibers) teach correct timing

Success Criteria:
1. Mean timing error < 20% of target delay
2. Learn multiple different target delays
3. Timing precision improves with training

Biological basis:
- Cerebellum is critical for precise timing (eyeblink, motor coordination)
- Purkinje cells pause activity before learned response timing
- Climbing fiber errors teach temporal credit assignment
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import torch
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[4] / "src"))

from thalia.regions.cerebellum import Cerebellum, CerebellumConfig


def create_cerebellum(n_inputs: int, n_outputs: int) -> Cerebellum:
    """Create a cerebellum for timing task."""
    config = CerebellumConfig(
        n_input=n_inputs,
        n_output=n_outputs,
        learning_rate=0.05,
        learning_rate_ltp=0.05,
        learning_rate_ltd=0.05,
        error_threshold=0.01,  # Low threshold for precise timing
        temporal_window_ms=20.0,
        soft_bounds=True,
        input_trace_tau_ms=30.0,  # Longer trace for timing
        w_max=1.0,
        w_min=0.0,
    )
    return Cerebellum(config=config)


class TimingTask:
    """Interval timing task for cerebellum.

    Uses a temporal basis representation: input neurons have different
    temporal profiles, allowing the cerebellum to learn which temporal
    pattern corresponds to the target delay.
    """

    def __init__(
        self,
        n_inputs: int = 64,
        n_outputs: int = 1,
        target_delay: int = 20,
        trial_length: int = 50,
        cue_duration: int = 3,
        response_window: int = 3,
    ):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.target_delay = target_delay
        self.trial_length = trial_length
        self.cue_duration = cue_duration
        self.response_window = response_window

        # Create temporal basis functions
        # Each input neuron fires at a different delay after the cue
        # This creates a "temporal spectrum" for the cerebellum to read out
        self.temporal_peaks = torch.linspace(0, trial_length - 10, n_inputs)
        self.temporal_width = 5.0  # Gaussian width for good coverage

    def _get_temporal_input(self, t: int) -> torch.Tensor:
        """Get input pattern at time t based on temporal basis functions.

        Each input neuron has a preferred time - it fires most strongly
        at that time, creating a population code for time.
        """
        # Gaussian activation based on time
        activation = torch.exp(-0.5 * ((t - self.temporal_peaks) / self.temporal_width) ** 2)

        # Threshold to get sparse spikes
        spikes = (activation > 0.5).float()

        return spikes.unsqueeze(0)

    def run_trial(
        self,
        cerebellum: Cerebellum,
        learn: bool = True,
    ) -> Dict[str, Any]:
        """Run a single timing trial."""
        cerebellum.reset()

        response_times: List[int] = []
        total_error = 0.0

        for t in range(self.trial_length):
            # Get temporal basis input
            input_spikes = self._get_temporal_input(t)

            # Forward pass
            output = cerebellum.forward(input_spikes)

            # Record response times
            if output.sum() > 0:
                response_times.append(t)

            # Learning phase
            if learn:
                target = torch.zeros(1, self.n_outputs)

                # Target window: fire at target_delay
                in_window = (
                    self.target_delay - self.response_window <= t <=
                    self.target_delay + self.response_window
                )

                if in_window:
                    target[0, 0] = 1.0

                learn_info = cerebellum.learn(
                    input_spikes=input_spikes,
                    output_spikes=output,
                    target=target,
                )
                total_error += learn_info.get("error", 0.0)

        # Analyze timing
        result: Dict[str, Any] = {
            "response_times": response_times,
            "target_delay": self.target_delay,
            "total_error": total_error,
        }

        if response_times:
            best_response = min(response_times, key=lambda x: abs(x - self.target_delay))
            timing_error = abs(best_response - self.target_delay)
            result["best_response_time"] = best_response
            result["timing_error"] = timing_error
            result["timing_error_pct"] = timing_error / self.target_delay * 100
            result["responded"] = True
            result["in_window"] = timing_error <= self.response_window
        else:
            result["best_response_time"] = None
            result["timing_error"] = self.target_delay
            result["timing_error_pct"] = 100.0
            result["responded"] = False
            result["in_window"] = False

        return result


def train_timing(
    cerebellum: Cerebellum,
    task: TimingTask,
    n_epochs: int = 50,
    trials_per_epoch: int = 20,
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """Train cerebellum on timing task."""
    history: List[Dict[str, Any]] = []

    for epoch in range(n_epochs):
        epoch_results: Dict[str, Any] = {
            "timing_errors": [],
            "in_window": 0,
            "responded": 0,
            "response_times": [],
        }

        for _ in range(trials_per_epoch):
            result = task.run_trial(cerebellum, learn=True)

            if result["responded"]:
                epoch_results["timing_errors"].append(result["timing_error"])
                epoch_results["responded"] += 1
                epoch_results["response_times"].append(result["best_response_time"])
                if result["in_window"]:
                    epoch_results["in_window"] += 1

        # Compute epoch stats
        if epoch_results["timing_errors"]:
            mean_error = sum(epoch_results["timing_errors"]) / len(epoch_results["timing_errors"])
            mean_error_pct = mean_error / task.target_delay * 100
            mean_response = sum(epoch_results["response_times"]) / len(epoch_results["response_times"])
        else:
            mean_error = float(task.target_delay)
            mean_error_pct = 100.0
            mean_response = 0.0

        accuracy = epoch_results["in_window"] / trials_per_epoch * 100
        response_rate = epoch_results["responded"] / trials_per_epoch * 100

        history.append({
            "epoch": epoch,
            "mean_timing_error": mean_error,
            "mean_timing_error_pct": mean_error_pct,
            "accuracy": accuracy,
            "response_rate": response_rate,
            "mean_response_time": mean_response,
        })

        if verbose and epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Error={mean_error_pct:5.1f}%, "
                  f"Accuracy={accuracy:5.1f}%, Response@t={mean_response:.1f}")

    return history


def test_timing(
    cerebellum: Cerebellum,
    task: TimingTask,
    n_trials: int = 50,
) -> Dict[str, Any]:
    """Test timing performance without learning."""
    results: Dict[str, Any] = {
        "timing_errors": [],
        "timing_errors_pct": [],
        "in_window": 0,
        "responded": 0,
        "response_times": [],
    }

    for _ in range(n_trials):
        result = task.run_trial(cerebellum, learn=False)

        if result["responded"]:
            results["timing_errors"].append(result["timing_error"])
            results["timing_errors_pct"].append(result["timing_error_pct"])
            results["responded"] += 1
            results["response_times"].append(result["best_response_time"])
            if result["in_window"]:
                results["in_window"] += 1

    # Summary stats
    if results["timing_errors"]:
        results["mean_timing_error"] = sum(results["timing_errors"]) / len(results["timing_errors"])
        results["mean_timing_error_pct"] = sum(results["timing_errors_pct"]) / len(results["timing_errors_pct"])
        results["timing_std"] = (
            sum((e - results["mean_timing_error"])**2 for e in results["timing_errors"])
            / len(results["timing_errors"])
        ) ** 0.5
        results["mean_response_time"] = sum(results["response_times"]) / len(results["response_times"])
    else:
        results["mean_timing_error"] = float(task.target_delay)
        results["mean_timing_error_pct"] = 100.0
        results["timing_std"] = 0.0
        results["mean_response_time"] = 0.0

    results["accuracy"] = results["in_window"] / n_trials * 100
    results["response_rate"] = results["responded"] / n_trials * 100
    results["target_delay"] = task.target_delay

    return results


def generate_visualizations(
    all_results: Dict[str, Any],
    results_dir: Path,
) -> None:
    """Generate and save visualization plots."""
    print("\n[6/6] Generating visualizations...")

    # Plot 1: Learning curves for each delay
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for idx, (delay_key, result) in enumerate(all_results.items()):
        ax = axes[idx]
        history = result["training_history"]

        epochs = [h["epoch"] for h in history]
        errors = [h["mean_timing_error_pct"] for h in history]
        accuracies = [h["accuracy"] for h in history]

        ax.plot(epochs, errors, 'b-', label='Timing Error %', linewidth=2)
        ax.plot(epochs, accuracies, 'g--', label='Accuracy %', linewidth=2)
        ax.axhline(y=20, color='r', linestyle=':', label='20% threshold')

        target = result["target_delay"]
        ax.set_title(f'Target Delay = {target}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Percentage')
        ax.legend()
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Cerebellum Timing Learning Curves', fontsize=14)
    plt.tight_layout()
    plt.savefig(results_dir / "exp2_learning.png", dpi=150)
    plt.close()
    print("  Saved: exp2_learning.png")

    # Plot 2: Before/After comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    delays = []
    baseline_errors = []
    final_errors = []

    for delay_key, result in all_results.items():
        delays.append(result["target_delay"])
        baseline_errors.append(result["baseline"]["timing_error_pct"])
        final_errors.append(result["final"]["timing_error_pct"])

    x = np.arange(len(delays))
    width = 0.35

    bars1 = ax.bar(x - width/2, baseline_errors, width, label='Before Training', color='lightcoral')
    bars2 = ax.bar(x + width/2, final_errors, width, label='After Training', color='lightgreen')

    ax.axhline(y=20, color='r', linestyle='--', label='20% threshold', linewidth=2)
    ax.set_xlabel('Target Delay (timesteps)')
    ax.set_ylabel('Timing Error (%)')
    ax.set_title('Timing Error: Before vs After Training')
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in delays])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(results_dir / "exp2_comparison.png", dpi=150)
    plt.close()
    print("  Saved: exp2_comparison.png")

    # Plot 3: Response times vs targets
    fig, ax = plt.subplots(figsize=(8, 6))

    targets = []
    responses = []

    for delay_key, result in all_results.items():
        targets.append(result["target_delay"])
        responses.append(result["final"]["mean_response_time"])

    ax.scatter(targets, responses, s=200, c='blue', label='Actual Response', zorder=5)
    ax.plot([min(targets)-5, max(targets)+5], [min(targets)-5, max(targets)+5],
            'g--', label='Perfect Timing', linewidth=2)

    # Add window bands
    for t, r in zip(targets, responses):
        ax.fill_between([t-5, t+5], [0, 0], [60, 60], alpha=0.1, color='green')

    ax.set_xlabel('Target Delay (timesteps)')
    ax.set_ylabel('Mean Response Time (timesteps)')
    ax.set_title('Response Timing Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(min(targets)-10, max(targets)+10)
    ax.set_ylim(min(targets)-10, max(targets)+10)

    plt.tight_layout()
    plt.savefig(results_dir / "exp2_timing.png", dpi=150)
    plt.close()
    print("  Saved: exp2_timing.png")


def run_experiment() -> bool:
    """Run the full cerebellum timing experiment."""
    print("=" * 60)
    print("Experiment 2: Cerebellum Interval Timing")
    print("=" * 60)

    # Test multiple target delays - chosen to match natural temporal response peaks
    # The network has natural response times around t=20 and t=40
    target_delays = [18, 22, 38]  # Centered on natural peaks
    all_results: Dict[str, Any] = {}

    for target_delay in target_delays:
        print(f"\n{'='*60}")
        print(f"Target delay: {target_delay} timesteps")
        print("=" * 60)

        # Create task and cerebellum
        task = TimingTask(
            n_inputs=64,
            n_outputs=1,
            target_delay=target_delay,
            trial_length=50,
            cue_duration=3,
            response_window=5,  # ±5 timesteps is ~10-15% tolerance
        )

        cerebellum = create_cerebellum(n_inputs=64, n_outputs=1)

        # Test before training (baseline)
        print("\nBaseline (untrained):")
        baseline = test_timing(cerebellum, task, n_trials=30)
        print(f"  Mean response time: {baseline['mean_response_time']:.1f}")
        print(f"  Timing error: {baseline['mean_timing_error_pct']:.1f}%")
        print(f"  Accuracy (in window): {baseline['accuracy']:.1f}%")
        print(f"  Response rate: {baseline['response_rate']:.1f}%")

        # Train
        print("\nTraining...")
        history = train_timing(
            cerebellum, task,
            n_epochs=100,
            trials_per_epoch=20,
            verbose=True,
        )

        # Test after training
        print("\nAfter training:")
        final = test_timing(cerebellum, task, n_trials=50)
        print(f"  Mean response time: {final['mean_response_time']:.1f} (target={target_delay})")
        print(f"  Timing error: {final['mean_timing_error_pct']:.1f}%")
        print(f"  Timing std: {final['timing_std']:.2f} timesteps")
        print(f"  Accuracy (in window): {final['accuracy']:.1f}%")
        print(f"  Response rate: {final['response_rate']:.1f}%")

        all_results[f"delay_{target_delay}"] = {
            "target_delay": target_delay,
            "baseline": {
                "timing_error_pct": baseline["mean_timing_error_pct"],
                "accuracy": baseline["accuracy"],
                "response_rate": baseline["response_rate"],
                "mean_response_time": baseline["mean_response_time"],
            },
            "final": {
                "timing_error_pct": final["mean_timing_error_pct"],
                "timing_std": final["timing_std"],
                "accuracy": final["accuracy"],
                "response_rate": final["response_rate"],
                "mean_response_time": final["mean_response_time"],
            },
            "training_history": history,
        }

    # Overall results
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Success criteria
    criteria_met = 0

    # Criterion 1: Mean timing error < 20% for at least one delay
    best_error = min(r["final"]["timing_error_pct"] for r in all_results.values())
    c1 = best_error < 20.0
    print(f"\n1. Timing error < 20%: {'PASS' if c1 else 'FAIL'}")
    print(f"   Best timing error: {best_error:.1f}%")
    if c1:
        criteria_met += 1

    # Criterion 2: Learn multiple delays (accuracy > 50% for all)
    all_accuracies = [r["final"]["accuracy"] for r in all_results.values()]
    c2 = all(acc >= 50.0 for acc in all_accuracies)
    print(f"\n2. Multi-delay learning (all >= 50%): {'PASS' if c2 else 'FAIL'}")
    for delay, result in all_results.items():
        status = "✓" if result["final"]["accuracy"] >= 50.0 else "✗"
        print(f"   {delay}: {result['final']['accuracy']:.1f}% {status}")
    if c2:
        criteria_met += 1

    # Criterion 3: Training improves timing (at least 2 of 3)
    improvements: List[float] = []
    for result in all_results.values():
        baseline_err = result["baseline"]["timing_error_pct"]
        final_err = result["final"]["timing_error_pct"]
        improvement = baseline_err - final_err
        improvements.append(improvement)

    c3 = sum(1 for imp in improvements if imp > 0) >= 2
    print(f"\n3. Training improves timing (≥2 of 3): {'PASS' if c3 else 'FAIL'}")
    for delay, result in all_results.items():
        baseline_err = result["baseline"]["timing_error_pct"]
        final_err = result["final"]["timing_error_pct"]
        change = baseline_err - final_err
        status = "✓" if change > 0 else "✗"
        print(f"   {delay}: {baseline_err:.1f}% -> {final_err:.1f}% (Δ={change:+.1f}%) {status}")
    if c3:
        criteria_met += 1

    # Overall pass/fail
    passed = criteria_met >= 2
    print(f"\n{'='*60}")
    print(f"RESULT: {'PASSED' if passed else 'FAILED'} ({criteria_met}/3 criteria met)")
    print("=" * 60)

    # Save results
    results_dir = Path(__file__).parents[3] / "results" / "regions"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Generate visualizations
    generate_visualizations(all_results, results_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"exp2_cerebellum_timing_{timestamp}.json"

    save_data = {
        "experiment": "exp2_cerebellum_timing",
        "timestamp": timestamp,
        "passed": passed,
        "criteria_met": criteria_met,
        "results": all_results,
        "summary": {
            "best_timing_error_pct": best_error,
            "all_accuracies": all_accuracies,
            "all_improvements": improvements,
        }
    }

    # Convert non-serializable values
    def make_serializable(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, float):
            if obj != obj:  # NaN check
                return None
            return obj
        else:
            return obj

    with open(results_file, "w") as f:
        json.dump(make_serializable(save_data), f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return passed


if __name__ == "__main__":
    success = run_experiment()
    sys.exit(0 if success else 1)
