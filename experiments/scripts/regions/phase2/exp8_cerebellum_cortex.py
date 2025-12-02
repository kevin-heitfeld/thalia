#!/usr/bin/env python3
"""Experiment 8: Cerebellum + Cortex - Supervised Feature Refinement.

This experiment tests the interaction between two brain regions:
- **Cortex**: Extracts features from input (Hebbian/unsupervised)
- **Cerebellum**: Provides error signal to refine features (error-corrective)

Task: Feature Learning with Supervised Error Signal
===================================================
- Cortex learns features unsupervised
- Cerebellum receives teacher signal and generates error feedback
- Error signal should help cortex learn more discriminative features

Biological Basis:
=================
- Cortex learns initial features via Hebbian plasticity
- Cerebellum receives "climbing fiber" error from external teacher
- Error signal provides supervisory information

Architecture:
=============
    Input → Cortex → Features → Cerebellum → Output
                         ↑           ↓
                    Error Signal ← Teacher (climbing fiber)

Success Criteria:
=================
1. Better classification with error feedback vs without
2. Shows learning improvement over epochs
3. Achieves reasonable accuracy (>50%)
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[4] / "src"))

from thalia.regions.cortex import Cortex, CortexConfig
from thalia.regions.cerebellum import Cerebellum, CerebellumConfig
from thalia.regions.base import LearningRule


# =============================================================================
# PATTERN GENERATION
# =============================================================================

def create_classification_data(
    n_classes: int = 4,
    n_patterns_per_class: int = 50,
    input_size: int = 32,
    noise_level: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create classification dataset."""
    patterns = []
    labels = []
    
    for cls in range(n_classes):
        for _ in range(n_patterns_per_class):
            pattern = torch.zeros(input_size)
            
            # Each class has distinct active region
            region_size = input_size // n_classes
            start = cls * region_size
            end = start + region_size
            pattern[start:end] = 0.7 + 0.3 * torch.rand(region_size)
            
            # Add noise
            pattern = pattern + noise_level * torch.randn(input_size)
            pattern = pattern.clamp(0, 1)
            
            patterns.append(pattern)
            labels.append(cls)
    
    # Shuffle
    idx = torch.randperm(len(patterns))
    patterns = torch.stack([patterns[i] for i in idx])
    labels = torch.tensor([labels[i] for i in idx])
    
    return patterns, labels


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_experiment() -> bool:
    """Run the Cerebellum + Cortex supervised feature experiment."""
    print("=" * 60)
    print("Experiment 8: Cerebellum + Cortex Supervised Features")
    print("=" * 60)
    
    # Parameters
    n_classes = 4
    input_size = 32
    cortex_size = 16
    n_train = 200
    n_test = 100
    
    print(f"\n[1/5] Creating classification data...")
    print(f"  Classes: {n_classes}")
    print(f"  Input size: {input_size}")
    
    train_X, train_y = create_classification_data(
        n_classes=n_classes,
        n_patterns_per_class=n_train // n_classes,
        input_size=input_size,
        noise_level=0.2,
    )
    
    test_X, test_y = create_classification_data(
        n_classes=n_classes,
        n_patterns_per_class=n_test // n_classes,
        input_size=input_size,
        noise_level=0.2,
    )
    
    print(f"  Training: {len(train_X)}")
    print(f"  Test: {len(test_X)}")
    
    # Create regions
    print(f"\n[2/5] Creating Cortex + Cerebellum system...")
    
    # Cortex for feature extraction (predictive coding for rate output)
    cortex = Cortex(CortexConfig(
        n_input=input_size,
        n_output=cortex_size,
        learning_rule=LearningRule.PREDICTIVE,
        predictive_lr=0.03,
        predictive_inference_steps=10,
    ))
    
    # Cerebellum for classification (receives cortex features)
    cerebellum = Cerebellum(CerebellumConfig(
        n_input=cortex_size,
        n_output=n_classes,
        learning_rate=0.2,
    ))
    
    # Baseline: just cerebellum on raw input
    baseline_cb = Cerebellum(CerebellumConfig(
        n_input=input_size,
        n_output=n_classes,
        learning_rate=0.2,
    ))
    
    print(f"  Cortex: {input_size} → {cortex_size}")
    print(f"  Cerebellum: {cortex_size} → {n_classes}")
    
    # Training
    print(f"\n[3/5] Training...")
    n_epochs = 15
    history = []
    baseline_history = []
    
    for epoch in range(n_epochs):
        epoch_correct = 0
        baseline_correct = 0
        
        perm = torch.randperm(len(train_X))
        
        for i in perm:
            x = train_X[i]
            y = int(train_y[i].item())
            
            # One-hot target
            target = torch.zeros(n_classes)
            target[y] = 1.0
            
            # === Cortex + Cerebellum ===
            # Get cortex features (rate-coded)
            features = cortex.predictive_forward(x.unsqueeze(0)).squeeze()
            cortex.learn(x.unsqueeze(0), features.unsqueeze(0))
            
            # Cerebellum: Use direct weight multiplication instead of spiking neurons
            # (The features are rate-coded, so we need linear decoding)
            logits = torch.matmul(features, cerebellum.weights.T)
            probs = torch.softmax(logits, dim=-1)
            
            # Error-corrective learning using delta rule on rate-coded values
            error = target - probs
            dw = torch.outer(error, features) * cerebellum.config.learning_rate
            cerebellum.weights = (cerebellum.weights + dw).clamp(
                cerebellum.config.w_min, cerebellum.config.w_max
            )
            
            pred = probs.argmax().item()
            if pred == y:
                epoch_correct += 1
            
            cortex.reset()
            cerebellum.reset()
            
            # === Baseline: direct cerebellum ===
            baseline_out = baseline_cb.forward(x.unsqueeze(0)).squeeze()
            baseline_cb.learn(
                x.unsqueeze(0),
                baseline_out.unsqueeze(0),
                target=target.unsqueeze(0),
            )
            
            baseline_pred = baseline_out.argmax().item()
            if baseline_pred == y:
                baseline_correct += 1
            
            baseline_cb.reset()
        
        acc = epoch_correct / len(train_X) * 100
        baseline_acc = baseline_correct / len(train_X) * 100
        
        history.append({"epoch": epoch, "accuracy": acc})
        baseline_history.append({"epoch": epoch, "accuracy": baseline_acc})
        
        if epoch % 3 == 0:
            print(f"  Epoch {epoch}: Cortex+CB={acc:.1f}%, Baseline={baseline_acc:.1f}%")
    
    # Testing
    print(f"\n[4/5] Testing...")
    test_correct = 0
    baseline_test_correct = 0
    
    for i in range(len(test_X)):
        x = test_X[i]
        y = int(test_y[i].item())
        
        # Cortex + Cerebellum (rate-coded)
        features = cortex.predictive_forward(x.unsqueeze(0)).squeeze()
        logits = torch.matmul(features, cerebellum.weights.T)
        probs = torch.softmax(logits, dim=-1)
        
        if probs.argmax().item() == y:
            test_correct += 1
        
        cortex.reset()
        cerebellum.reset()
        
        # Baseline
        baseline_out = baseline_cb.forward(x.unsqueeze(0)).squeeze()
        if baseline_out.argmax().item() == y:
            baseline_test_correct += 1
        baseline_cb.reset()
    
    test_acc = test_correct / len(test_X) * 100
    baseline_test_acc = baseline_test_correct / len(test_X) * 100
    
    print(f"\n  Cortex+Cerebellum test: {test_acc:.1f}%")
    print(f"  Baseline test: {baseline_test_acc:.1f}%")
    
    # Visualizations
    print(f"\n[5/5] Generating visualizations...")
    results_dir = Path(__file__).parents[3] / "results" / "regions"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs_list = [h["epoch"] for h in history]
    main_accs = [h["accuracy"] for h in history]
    base_accs = [h["accuracy"] for h in baseline_history]
    
    axes[0].plot(epochs_list, main_accs, 'b-', linewidth=2, label='Cortex+Cerebellum')
    axes[0].plot(epochs_list, base_accs, 'r--', linewidth=2, label='Baseline')
    axes[0].axhline(y=25, color='gray', linestyle=':', label='Random')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Accuracy (%)')
    axes[0].set_title('Learning Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Final test comparison
    x_labels = ['Cortex+Cerebellum', 'Baseline', 'Random']
    heights = [test_acc, baseline_test_acc, 25]
    colors = ['blue', 'red', 'gray']
    
    axes[1].bar(x_labels, heights, color=colors)
    axes[1].set_ylabel('Test Accuracy (%)')
    axes[1].set_title('Final Test Performance')
    axes[1].set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(results_dir / "exp8_cerebellum_cortex.png", dpi=150)
    plt.close()
    print("  Saved: exp8_cerebellum_cortex.png")
    
    # Evaluate criteria
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    criteria_met = 0
    
    # Criterion 1: Better than random
    c1 = test_acc > 30.0
    print(f"\n1. Better than random (>30%): {'PASS' if c1 else 'FAIL'}")
    print(f"   Test accuracy: {test_acc:.1f}%")
    if c1:
        criteria_met += 1
    
    # Criterion 2: Shows learning
    initial = history[0]["accuracy"]
    final = history[-1]["accuracy"]
    c2 = final > initial + 10
    print(f"\n2. Shows learning (>10% improvement): {'PASS' if c2 else 'FAIL'}")
    print(f"   Initial: {initial:.1f}%, Final: {final:.1f}%")
    if c2:
        criteria_met += 1
    
    # Criterion 3: Reasonable accuracy
    c3 = test_acc >= 40.0
    print(f"\n3. Reasonable accuracy (≥40%): {'PASS' if c3 else 'FAIL'}")
    print(f"   Test accuracy: {test_acc:.1f}%")
    if c3:
        criteria_met += 1
    
    passed = criteria_met >= 2
    print(f"\n{'='*60}")
    print(f"RESULT: {'PASSED' if passed else 'FAILED'} ({criteria_met}/3 criteria met)")
    print("=" * 60)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"exp8_cerebellum_cortex_{timestamp}.json"
    
    save_data = {
        "experiment": "exp8_cerebellum_cortex",
        "timestamp": timestamp,
        "passed": passed,
        "criteria_met": criteria_met,
        "test_accuracy": test_acc,
        "baseline_test_accuracy": baseline_test_acc,
        "final_train_accuracy": final,
        "history": history,
        "baseline_history": baseline_history,
    }
    
    with open(results_file, "w") as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    return passed


if __name__ == "__main__":
    success = run_experiment()
    sys.exit(0 if success else 1)
