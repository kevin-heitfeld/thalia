#!/usr/bin/env python3
"""Experiment 6: Cortex + Striatum - Category Learning with Reward.

This experiment tests the interaction between two brain regions:
- **Cortex**: Extracts features from visual input (Hebbian learning)
- **Striatum**: Learns action-reward associations (three-factor RL)

Task: Visual Category Classification
====================================
- Input: Visual patterns belonging to different categories
- All correct classifications are rewarded
- System must learn to classify correctly via reward signal

Biological Basis:
=================
- Ventral visual stream (cortex) → visual feature extraction
- Basal ganglia (striatum) → action selection based on expected reward
- Dopamine signals RPE to update striatal weights

Architecture:
=============
    Visual Input → Cortex → Features → Striatum → Action Selection
                                           ↑
                                      Dopamine (RPE)

Success Criteria:
=================
1. Test accuracy > 30% (better than 25% random)
2. Shows learning improvement over epochs
3. Achieves ≥40% accuracy
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
from thalia.regions.striatum import Striatum, StriatumConfig
from thalia.regions.base import LearningRule


# =============================================================================
# VISUAL PATTERN GENERATION
# =============================================================================

def create_visual_patterns(
    n_categories: int = 4,
    pattern_size: int = 64,
    n_patterns_per_category: int = 50,
    noise_level: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create visual patterns for each category.
    
    Each category has a distinct spatial pattern.
    """
    patterns = []
    labels = []
    
    for cat_idx in range(n_categories):
        for _ in range(n_patterns_per_category):
            pattern = torch.zeros(pattern_size)
            
            # Each category activates a different region
            region_size = pattern_size // n_categories
            start = cat_idx * region_size
            end = start + region_size
            
            # Strong signal in category region
            pattern[start:end] = 0.8 + 0.2 * torch.rand(region_size)
            
            # Add noise
            pattern = pattern + noise_level * torch.randn(pattern_size)
            pattern = pattern.clamp(0, 1)
            
            patterns.append(pattern)
            labels.append(cat_idx)
    
    # Shuffle
    indices = torch.randperm(len(patterns))
    patterns = torch.stack([patterns[i] for i in indices])
    labels = torch.tensor([labels[i] for i in indices])
    
    return patterns, labels


# =============================================================================
# SIMPLE LEARNER (direct feature-to-action weights)
# =============================================================================

class SimpleLearner:
    """Simple RL learner with weight-based action selection."""
    
    def __init__(self, n_input: int, n_actions: int, lr: float = 0.1):
        self.weights = torch.rand(n_actions, n_input) * 0.1
        self.lr = lr
        self.exploration = 0.3
        
    def select_action(self, features: torch.Tensor) -> int:
        """Select action using softmax on weight-based values."""
        if features.dim() == 2:
            features = features.squeeze(0)
        
        # Action values
        values = torch.matmul(self.weights, features)
        probs = torch.softmax(values / 0.5, dim=0)
        
        if np.random.rand() < self.exploration:
            return np.random.randint(self.weights.shape[0])
        return torch.multinomial(probs, 1).item()
    
    def learn(self, features: torch.Tensor, action: int, reward: float):
        """Three-factor learning: Δw = lr × eligibility × dopamine."""
        if features.dim() == 2:
            features = features.squeeze(0)
        
        # Eligibility for chosen action
        eligibility = torch.zeros_like(self.weights)
        eligibility[action, :] = features
        
        # Dopamine from reward (0.5 baseline)
        rpe = reward - 0.5
        da = rpe * 2
        
        # Update
        dw = self.lr * eligibility * da
        self.weights = (self.weights + dw).clamp(0, 1)
        
    def decay_exploration(self):
        self.exploration = max(0.1, self.exploration * 0.95)


# =============================================================================
# EXPERIMENT
# =============================================================================

def run_experiment() -> bool:
    """Run the Cortex + Striatum visual categorization experiment."""
    print("=" * 60)
    print("Experiment 6: Cortex + Striatum Visual Categorization")
    print("=" * 60)
    
    # Parameters
    n_categories = 4
    pattern_size = 64
    cortex_size = 32
    n_train = 400
    n_test = 100
    
    print(f"\n[1/5] Creating visual patterns...")
    print(f"  Categories: {n_categories}")
    print(f"  Pattern size: {pattern_size}")
    
    train_patterns, train_labels = create_visual_patterns(
        n_categories=n_categories,
        pattern_size=pattern_size,
        n_patterns_per_category=n_train // n_categories,
        noise_level=0.2,
    )
    
    test_patterns, test_labels = create_visual_patterns(
        n_categories=n_categories,
        pattern_size=pattern_size,
        n_patterns_per_category=n_test // n_categories,
        noise_level=0.2,
    )
    
    print(f"  Training patterns: {len(train_patterns)}")
    print(f"  Test patterns: {len(test_patterns)}")
    
    # Create Cortex for feature extraction (PREDICTIVE mode for rate-coded output)
    print(f"\n[2/5] Creating Cortex + Striatum system...")
    cortex = Cortex(CortexConfig(
        n_input=pattern_size,
        n_output=cortex_size,
        learning_rule=LearningRule.PREDICTIVE,  # Use predictive coding
        predictive_lr=0.05,
        predictive_inference_steps=10,
    ))
    
    # Striatum learner (from cortex features)
    striatum = SimpleLearner(n_input=cortex_size, n_actions=n_categories, lr=0.15)
    
    # Baseline learner (from raw input)
    baseline = SimpleLearner(n_input=pattern_size, n_actions=n_categories, lr=0.15)
    
    print(f"  Cortex: {pattern_size} → {cortex_size}")
    print(f"  Striatum: {cortex_size} → {n_categories} actions")
    
    # Training
    print(f"\n[3/5] Training...")
    n_epochs = 20
    history = []
    baseline_history = []
    
    for epoch in range(n_epochs):
        epoch_correct = 0
        baseline_correct = 0
        
        perm = torch.randperm(len(train_patterns))
        
        for i in perm:
            pattern = train_patterns[i]
            true_label = int(train_labels[i].item())
            
            # === Cortex + Striatum ===
            # Use predictive_forward for rate-coded features
            cortex_out = cortex.predictive_forward(pattern.unsqueeze(0))
            features = cortex_out.float().squeeze()
            
            # Cortex learning (predictive)
            cortex.learn(pattern.unsqueeze(0), cortex_out)
            
            # Striatum action selection
            action = striatum.select_action(features)
            
            # Reward for correct classification
            reward = 1.0 if action == true_label else 0.0
            
            # Striatum learning
            striatum.learn(features, action, reward)
            
            if action == true_label:
                epoch_correct += 1
            
            cortex.reset()
            
            # === Baseline (direct input) ===
            baseline_action = baseline.select_action(pattern)
            baseline_reward = 1.0 if baseline_action == true_label else 0.0
            baseline.learn(pattern, baseline_action, baseline_reward)
            
            if baseline_action == true_label:
                baseline_correct += 1
        
        accuracy = epoch_correct / len(train_patterns) * 100
        baseline_acc = baseline_correct / len(train_patterns) * 100
        
        history.append({"epoch": epoch, "accuracy": accuracy})
        baseline_history.append({"epoch": epoch, "accuracy": baseline_acc})
        
        # Decay exploration
        striatum.decay_exploration()
        baseline.decay_exploration()
        
        if epoch % 4 == 0:
            print(f"  Epoch {epoch}: Cortex+Striatum={accuracy:.1f}%, Baseline={baseline_acc:.1f}%")
    
    # Testing (low exploration)
    print(f"\n[4/5] Testing...")
    striatum.exploration = 0.05
    baseline.exploration = 0.05
    
    test_correct = 0
    baseline_test_correct = 0
    
    for i in range(len(test_patterns)):
        pattern = test_patterns[i]
        true_label = int(test_labels[i].item())
        
        # Cortex + Striatum
        cortex_out = cortex.predictive_forward(pattern.unsqueeze(0))
        features = cortex_out.float().squeeze()
        action = striatum.select_action(features)
        
        if action == true_label:
            test_correct += 1
        cortex.reset()
        
        # Baseline
        baseline_action = baseline.select_action(pattern)
        if baseline_action == true_label:
            baseline_test_correct += 1
    
    test_accuracy = test_correct / len(test_patterns) * 100
    baseline_test_accuracy = baseline_test_correct / len(test_patterns) * 100
    
    print(f"\n  Cortex+Striatum test accuracy: {test_accuracy:.1f}%")
    print(f"  Baseline test accuracy: {baseline_test_accuracy:.1f}%")
    
    # Visualizations
    print(f"\n[5/5] Generating visualizations...")
    results_dir = Path(__file__).parents[3] / "results" / "regions"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = [h["epoch"] for h in history]
    main_accs = [h["accuracy"] for h in history]
    base_accs = [h["accuracy"] for h in baseline_history]
    
    axes[0].plot(epochs, main_accs, 'b-', linewidth=2, label='Cortex+Striatum')
    axes[0].plot(epochs, base_accs, 'r--', linewidth=2, label='Baseline')
    axes[0].axhline(y=25, color='gray', linestyle=':', label='Random (25%)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Accuracy (%)')
    axes[0].set_title('Learning Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Final test comparison
    x = ['Cortex+Striatum', 'Baseline', 'Random']
    heights = [test_accuracy, baseline_test_accuracy, 25]
    colors = ['blue', 'red', 'gray']
    
    axes[1].bar(x, heights, color=colors)
    axes[1].set_ylabel('Test Accuracy (%)')
    axes[1].set_title('Final Test Performance')
    axes[1].set_ylim(0, 100)
    
    plt.tight_layout()
    plt.savefig(results_dir / "exp6_cortex_striatum.png", dpi=150)
    plt.close()
    print("  Saved: exp6_cortex_striatum.png")
    
    # Evaluate criteria
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    criteria_met = 0
    
    # Criterion 1: Better than random (>30%)
    c1 = test_accuracy > 30.0
    print(f"\n1. Better than random (>30%): {'PASS' if c1 else 'FAIL'}")
    print(f"   Test accuracy: {test_accuracy:.1f}%")
    if c1:
        criteria_met += 1
    
    # Criterion 2: Shows learning (improvement over epochs)
    initial_acc = history[0]["accuracy"]
    final_acc = history[-1]["accuracy"]
    c2 = final_acc > initial_acc + 5
    print(f"\n2. Shows learning (≥5% improvement): {'PASS' if c2 else 'FAIL'}")
    print(f"   Initial: {initial_acc:.1f}%, Final: {final_acc:.1f}%")
    if c2:
        criteria_met += 1
    
    # Criterion 3: Achieves ≥40% accuracy
    c3 = test_accuracy >= 40.0
    print(f"\n3. Achieves ≥40% accuracy: {'PASS' if c3 else 'FAIL'}")
    print(f"   Test accuracy: {test_accuracy:.1f}%")
    if c3:
        criteria_met += 1
    
    passed = criteria_met >= 2
    print(f"\n{'='*60}")
    print(f"RESULT: {'PASSED' if passed else 'FAILED'} ({criteria_met}/3 criteria met)")
    print("=" * 60)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"exp6_cortex_striatum_{timestamp}.json"
    
    save_data = {
        "experiment": "exp6_cortex_striatum",
        "timestamp": timestamp,
        "passed": passed,
        "criteria_met": criteria_met,
        "test_accuracy": test_accuracy,
        "baseline_test_accuracy": baseline_test_accuracy,
        "final_train_accuracy": final_acc,
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
