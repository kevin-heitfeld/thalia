#!/usr/bin/env python3
"""
Experiment 15: Metacognition - Self-Monitoring and Uncertainty

Tests emergent metacognitive abilities from integrated brain regions:
1. Prefrontal cortex monitors processing across regions
2. Hippocampus tracks memory confidence
3. Cortex provides pattern recognition feedback
4. System recognizes its own limitations

Success criteria:
1. Confidence calibration: High confidence correlates with correctness (r > 0.3)
2. Uncertainty recognition: Lower confidence on novel/ambiguous inputs
3. Error detection: System predicts its own errors better than chance

This demonstrates self-awareness emerging from region integration.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

# Add src to path
project_root = Path(__file__).parents[4]
sys.path.insert(0, str(project_root))

from thalia.regions import (
    Cortex, CortexConfig,
    Hippocampus, HippocampusConfig,
    Prefrontal, PrefrontalConfig,
)
from thalia.regions.base import LearningRule


def run_experiment() -> bool:
    """Run metacognition experiment."""
    print("=" * 60)
    print("Experiment 15: Metacognition - Self-Monitoring and Uncertainty")
    print("=" * 60)

    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Parameters
    input_dim = 16
    hidden_dim = 32
    n_patterns = 20
    n_novel = 10
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.5]

    # Phase 1: Create integrated brain system
    print("\n[1/6] Creating metacognitive brain system...")

    # Cortex: pattern recognition (input_dim -> hidden_dim)
    cortex = Cortex(CortexConfig(
        n_input=input_dim,
        n_output=hidden_dim,
        learning_rule=LearningRule.HEBBIAN,
    ))

    # Hippocampus: memory storage (hidden_dim -> hidden_dim, autoassociative)
    hippocampus = Hippocampus(HippocampusConfig(
        n_input=hidden_dim,
        n_output=hidden_dim,
        learning_rate=0.5,
    ))

    # Prefrontal: metacognitive monitor (hidden_dim*2 -> 1)
    # Takes concatenated cortex output and hippocampus output
    prefrontal = Prefrontal(PrefrontalConfig(
        n_input=hidden_dim * 2,
        n_output=1,
    ))

    print(f"  Cortex: {input_dim} → {hidden_dim}")
    print(f"  Hippocampus: {hidden_dim} → {hidden_dim}")
    print(f"  Prefrontal (monitor): {hidden_dim * 2} → 1")

    # Phase 2: Learn patterns
    print("\n[2/6] Learning known patterns...")

    known_patterns: List[torch.Tensor] = []
    for _ in range(n_patterns):
        # Create distinct sparse patterns
        pattern = torch.zeros(input_dim)
        active_units = np.random.choice(input_dim, size=4, replace=False)
        pattern[active_units] = 1.0
        known_patterns.append(pattern)

    # Store patterns: train cortex and hippocampus
    cortex_outputs: List[torch.Tensor] = []

    for epoch in range(100):
        for pattern in known_patterns:
            # Cortex encodes pattern
            cortex_out = torch.tanh(cortex.weights @ pattern)

            # Hebbian learning in cortex
            pattern_norm = pattern / (pattern.norm() + 1e-6)
            cortex_norm = cortex_out / (cortex_out.norm() + 1e-6)
            dw_cortex = 0.1 * torch.outer(cortex_norm, pattern_norm)
            cortex.weights = (cortex.weights + dw_cortex * 0.05).clamp(-2, 2)

            # Store in hippocampus (autoassociative)
            hipp_out = torch.tanh(hippocampus.weights @ cortex_out)
            dw_hipp = 0.2 * torch.outer(hipp_out, cortex_out)
            hippocampus.weights = (hippocampus.weights + dw_hipp * 0.05).clamp(-2, 2)

    # Cache cortex outputs for known patterns
    for pattern in known_patterns:
        cortex_out = torch.tanh(cortex.weights @ pattern)
        cortex_outputs.append(cortex_out.detach().clone())

    print(f"  Learned {n_patterns} patterns")

    # Phase 3: Train prefrontal to monitor confidence
    print("\n[3/6] Training prefrontal confidence monitor...")

    # Collect training data for metacognition
    metacog_data: List[Tuple[torch.Tensor, float, float]] = []

    for _ in range(500):
        # Pick random known pattern
        pattern_idx = np.random.randint(n_patterns)
        pattern = known_patterns[pattern_idx].clone()

        # Sometimes add noise (making it harder)
        noise_level = float(np.random.choice(noise_levels))
        if noise_level > 0:
            noise = torch.randn(input_dim) * noise_level
            noisy_pattern = pattern + noise
        else:
            noisy_pattern = pattern

        # Forward through system
        cortex_out = torch.tanh(cortex.weights @ noisy_pattern)
        hipp_out = torch.tanh(hippocampus.weights @ cortex_out)

        # Measure retrieval quality (how close to stored representation?)
        stored_cortex = cortex_outputs[pattern_idx]
        retrieval_sim = float((cortex_out @ stored_cortex) /
                              (cortex_out.norm() * stored_cortex.norm() + 1e-6))

        # Ground truth: good retrieval = similar to stored representation
        correct = retrieval_sim > 0.8

        # Combine cortex and hippocampus for prefrontal input
        combined = torch.cat([cortex_out, hipp_out])
        metacog_data.append((combined.detach(), 1.0 if correct else 0.0, noise_level))

    # Train prefrontal to predict correctness
    for epoch in range(100):
        for combined, target, _ in metacog_data:
            # Prefrontal predicts confidence
            raw_conf = prefrontal.weights @ combined
            confidence = torch.sigmoid(raw_conf)

            # Learn to predict correctness
            error = target - float(confidence.item())
            with torch.no_grad():
                prefrontal.weights += 0.05 * error * combined.unsqueeze(0)
                prefrontal.weights.clamp_(-2, 2)

    print("  Prefrontal trained to monitor confidence")

    # Phase 4: Test confidence calibration
    print("\n[4/6] Testing confidence calibration...")

    confidences: List[float] = []
    correctness: List[int] = []
    noise_info: List[float] = []

    for pattern_idx, pattern in enumerate(known_patterns):
        stored_cortex = cortex_outputs[pattern_idx]

        for noise_level in noise_levels:
            # Add noise
            if noise_level > 0:
                noisy = pattern + torch.randn(input_dim) * noise_level
            else:
                noisy = pattern.clone()

            # Forward through system
            cortex_out = torch.tanh(cortex.weights @ noisy)
            hipp_out = torch.tanh(hippocampus.weights @ cortex_out)
            combined = torch.cat([cortex_out, hipp_out])

            # Get confidence from prefrontal
            raw_conf = prefrontal.weights @ combined
            confidence = float(torch.sigmoid(raw_conf).item())

            # Measure actual correctness
            retrieval_sim = float((cortex_out @ stored_cortex) /
                                  (cortex_out.norm() * stored_cortex.norm() + 1e-6))
            is_correct = retrieval_sim > 0.8

            confidences.append(confidence)
            correctness.append(1 if is_correct else 0)
            noise_info.append(noise_level)

    # Calculate correlation
    if len(confidences) > 1:
        correlation = float(np.corrcoef(confidences, correctness)[0, 1])
    else:
        correlation = 0.0

    print(f"  Confidence-correctness correlation: {correlation:.3f}")

    # Phase 5: Test on novel patterns
    print("\n[5/6] Testing uncertainty on novel inputs...")

    novel_patterns: List[torch.Tensor] = []
    for _ in range(n_novel):
        pattern = torch.randn(input_dim) * 0.5  # Random novel patterns
        novel_patterns.append(pattern)

    known_confidences: List[float] = []
    novel_confidences: List[float] = []

    # Confidence on known patterns (clean, no noise)
    for pattern in known_patterns[:10]:
        cortex_out = torch.tanh(cortex.weights @ pattern)
        hipp_out = torch.tanh(hippocampus.weights @ cortex_out)
        combined = torch.cat([cortex_out, hipp_out])
        raw_conf = prefrontal.weights @ combined
        conf = float(torch.sigmoid(raw_conf).item())
        known_confidences.append(conf)

    # Confidence on novel patterns
    for pattern in novel_patterns:
        cortex_out = torch.tanh(cortex.weights @ pattern)
        hipp_out = torch.tanh(hippocampus.weights @ cortex_out)
        combined = torch.cat([cortex_out, hipp_out])
        raw_conf = prefrontal.weights @ combined
        conf = float(torch.sigmoid(raw_conf).item())
        novel_confidences.append(conf)

    avg_known_conf = float(np.mean(known_confidences))
    avg_novel_conf = float(np.mean(novel_confidences))
    uncertainty_recognition = avg_known_conf > avg_novel_conf

    print(f"  Known pattern confidence: {avg_known_conf:.3f}")
    print(f"  Novel pattern confidence: {avg_novel_conf:.3f}")
    print(f"  Recognizes uncertainty: {uncertainty_recognition}")

    # Phase 6: Test error prediction
    print("\n[6/6] Testing error prediction...")

    # Inject deliberate errors and see if system can predict them
    error_predictions: List[bool] = []
    actual_errors: List[bool] = []

    for _ in range(50):
        # Pick pattern and add varying noise
        pattern_idx = np.random.randint(n_patterns)
        pattern = known_patterns[pattern_idx]
        stored_cortex = cortex_outputs[pattern_idx]

        noise = float(np.random.uniform(0, 0.5))
        noisy = pattern + torch.randn(input_dim) * noise

        cortex_out = torch.tanh(cortex.weights @ noisy)
        hipp_out = torch.tanh(hippocampus.weights @ cortex_out)
        combined = torch.cat([cortex_out, hipp_out])

        raw_conf = prefrontal.weights @ combined
        confidence = float(torch.sigmoid(raw_conf).item())

        # System predicts error when confidence is low
        predicts_error = confidence < 0.5

        # Check actual error
        retrieval_sim = float((cortex_out @ stored_cortex) /
                              (cortex_out.norm() * stored_cortex.norm() + 1e-6))
        actual_error = retrieval_sim < 0.8

        error_predictions.append(predicts_error)
        actual_errors.append(actual_error)

    # Calculate error prediction accuracy
    correct_predictions = sum(1 for p, a in zip(error_predictions, actual_errors) if p == a)
    error_prediction_acc = correct_predictions / len(error_predictions) * 100

    print(f"  Error prediction accuracy: {error_prediction_acc:.1f}%")

    # Generate plots
    print("\n[7/7] Generating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Confidence vs noise
    ax1 = axes[0, 0]
    noise_groups: dict[float, List[float]] = {}
    for conf, noise in zip(confidences, noise_info):
        if noise not in noise_groups:
            noise_groups[noise] = []
        noise_groups[noise].append(conf)

    noise_vals = sorted(noise_groups.keys())
    conf_means = [float(np.mean(noise_groups[n])) for n in noise_vals]
    conf_stds = [float(np.std(noise_groups[n])) for n in noise_vals]

    ax1.errorbar(noise_vals, conf_means, yerr=conf_stds, marker='o', capsize=5)
    ax1.set_xlabel('Noise Level')
    ax1.set_ylabel('Confidence')
    ax1.set_title('Confidence Decreases with Noise')
    ax1.set_ylim([0, 1])

    # Plot 2: Known vs Novel confidence
    ax2 = axes[0, 1]
    ax2.bar(['Known\nPatterns', 'Novel\nPatterns'],
            [avg_known_conf, avg_novel_conf],
            color=['steelblue', 'coral'], alpha=0.7)
    ax2.set_ylabel('Average Confidence')
    ax2.set_title('Uncertainty Recognition')
    ax2.set_ylim([0, 1])

    # Plot 3: Confidence-Correctness scatter
    ax3 = axes[1, 0]
    correct_mask = [c == 1 for c in correctness]
    incorrect_mask = [c == 0 for c in correctness]

    correct_confs = [c for c, m in zip(confidences, correct_mask) if m]
    incorrect_confs = [c for c, m in zip(confidences, incorrect_mask) if m]

    if correct_confs:
        ax3.hist(correct_confs, bins=10, alpha=0.6, label='Correct', color='green')
    if incorrect_confs:
        ax3.hist(incorrect_confs, bins=10, alpha=0.6, label='Incorrect', color='red')
    ax3.set_xlabel('Confidence')
    ax3.set_ylabel('Count')
    ax3.set_title(f'Confidence Distribution (r={correlation:.2f})')
    ax3.legend()

    # Plot 4: Error prediction confusion
    ax4 = axes[1, 1]
    true_pos = sum(1 for p, a in zip(error_predictions, actual_errors) if p and a)
    false_pos = sum(1 for p, a in zip(error_predictions, actual_errors) if p and not a)
    true_neg = sum(1 for p, a in zip(error_predictions, actual_errors) if not p and not a)
    false_neg = sum(1 for p, a in zip(error_predictions, actual_errors) if not p and a)

    confusion = np.array([[true_neg, false_pos], [false_neg, true_pos]])
    ax4.imshow(confusion, cmap='Blues')
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(['Pred: OK', 'Pred: Error'])
    ax4.set_yticklabels(['Actual: OK', 'Actual: Error'])
    ax4.set_title(f'Error Prediction ({error_prediction_acc:.0f}% accurate)')
    for i in range(2):
        for j in range(2):
            ax4.text(j, i, str(confusion[i, j]), ha='center', va='center')

    plt.tight_layout()

    # Path: experiments/scripts/regions/phase4 -> experiments/results/regions
    results_dir = Path(__file__).parent.parent.parent.parent / "results" / "regions"
    results_dir.mkdir(parents=True, exist_ok=True)
    plot_path = results_dir / "exp15_metacognition.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"  Saved: exp15_metacognition.png")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Criterion 1: Confidence calibration (r > 0.3)
    calib_pass = correlation > 0.3
    print(f"\n1. Confidence calibration (r > 0.3): {'PASS' if calib_pass else 'FAIL'}")
    print(f"   Correlation: {correlation:.3f}")

    # Criterion 2: Uncertainty recognition (known > novel confidence)
    uncert_pass = bool(uncertainty_recognition)
    print(f"\n2. Uncertainty recognition: {'PASS' if uncert_pass else 'FAIL'}")
    print(f"   Known: {avg_known_conf:.3f}, Novel: {avg_novel_conf:.3f}")

    # Criterion 3: Error prediction (> 55% = better than chance)
    error_pass = error_prediction_acc > 55.0
    print(f"\n3. Error prediction (> 55%): {'PASS' if error_pass else 'FAIL'}")
    print(f"   Accuracy: {error_prediction_acc:.1f}%")

    n_passed = sum([calib_pass, uncert_pass, error_pass])
    success = n_passed >= 2  # Need 2/3 to pass

    print("\n" + "=" * 60)
    print(f"RESULT: {'PASSED' if success else 'FAILED'} ({n_passed}/3 criteria)")
    print("=" * 60)

    # Save results
    results: dict[str, Any] = {
        'experiment': 'exp15_metacognition',
        'phase': 4,
        'description': 'Metacognition - Self-Monitoring and Uncertainty',
        'metrics': {
            'confidence_calibration': {
                'correlation': correlation,
                'target': 0.3,
                'passed': bool(calib_pass)
            },
            'uncertainty_recognition': {
                'known_confidence': avg_known_conf,
                'novel_confidence': avg_novel_conf,
                'passed': bool(uncert_pass)
            },
            'error_prediction': {
                'accuracy': error_prediction_acc,
                'target': 55.0,
                'passed': bool(error_pass)
            }
        },
        'criteria_passed': n_passed,
        'success': bool(success)
    }

    # Use the same results_dir we already created for plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = results_dir / f"exp15_metacognition_{timestamp}.json"

    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {result_file}")

    return success


if __name__ == "__main__":
    success = run_experiment()
    sys.exit(0 if success else 1)
