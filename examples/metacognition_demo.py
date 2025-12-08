"""
Metacognitive Calibration Demo.

This script demonstrates training a brain to calibrate its confidence
estimates to actual accuracy - a key metacognitive skill.

**Demonstrations**:
1. Generate calibration dataset with varying difficulties
2. Evaluate initial (uncalibrated) confidence
3. Train confidence estimation
4. Evaluate calibrated confidence
5. Generate reliability diagram and calibration report

**Usage**:
    python examples/metacognition_demo.py

**Expected Behavior**:
- Initial ECE (Expected Calibration Error) typically > 0.20 (poor calibration)
- After training, ECE should decrease to < 0.15 (acceptable calibration)
- Reliability diagram shows confidence-accuracy alignment

**Output**:
- Console output with calibration reports
- Optional: Reliability diagram plot (if matplotlib available)
"""

from __future__ import annotations

import torch

from thalia.training import MetacognitiveCalibrator
from thalia.training.metacognition import (
    CalibrationSample,
    create_simple_task_generator,
)


def create_mock_brain():
    """Create a simple mock brain for demonstration.
    
    In practice, you would use a real Brain instance with trained regions.
    """
    class MockBrain:
        def __init__(self, device='cpu'):
            self.device = torch.device(device)
            
            # Mock prefrontal region for confidence
            class MockPFC:
                def __init__(self):
                    self.state = type('State', (), {'spikes': torch.zeros(100)})()
            
            self.prefrontal = MockPFC()
        
        def forward(self, input_data):
            """Mock forward pass - returns random output."""
            # In practice, this would be real brain processing
            n_classes = 10
            output = torch.randn(n_classes, device=self.device)
            
            # Simulate confidence in PFC spikes
            # Start with random confidence (uncalibrated)
            confidence = torch.rand(100, device=self.device) * 0.5
            self.prefrontal.state.spikes = (confidence > 0.25).float()
            
            return output
    
    return MockBrain()


def demo_generate_calibration_dataset():
    """Demo 1: Generate calibration dataset."""
    print("\n" + "=" * 80)
    print("DEMO 1: Generate Calibration Dataset")
    print("=" * 80)
    print("\nScenario: Create tasks spanning difficulty spectrum\n")
    
    # Create task generator
    generator = create_simple_task_generator(
        n_classes=10,
        input_size=100,
    )
    
    # Create mock brain and calibrator
    brain = create_mock_brain()
    calibrator = MetacognitiveCalibrator(brain=brain, n_bins=10)
    
    # Generate dataset
    print("Generating 100 samples with difficulties from 0.3 to 0.9...")
    dataset = calibrator.generate_calibration_dataset(
        task_generator=generator,
        difficulty_range=(0.3, 0.9),
        n_samples=100,
        task_type='classification',
        stratified=True,
    )
    
    print(f"Generated {len(dataset)} samples")
    print(f"\nFirst 5 samples:")
    for i, sample in enumerate(dataset[:5]):
        print(f"  Sample {i+1}: difficulty={sample.difficulty:.2f}, "
              f"input_shape={sample.input_data.shape}, "
              f"target={sample.target.item()}")
    
    # Show difficulty distribution
    difficulties = [s.difficulty for s in dataset]
    print(f"\nDifficulty statistics:")
    print(f"  Min: {min(difficulties):.3f}")
    print(f"  Max: {max(difficulties):.3f}")
    print(f"  Mean: {sum(difficulties)/len(difficulties):.3f}")
    
    return brain, calibrator, dataset


def demo_evaluate_initial_calibration(brain, calibrator, dataset):
    """Demo 2: Evaluate initial (uncalibrated) confidence."""
    print("\n" + "=" * 80)
    print("DEMO 2: Evaluate Initial Calibration")
    print("=" * 80)
    print("\nScenario: Check baseline confidence calibration (before training)\n")
    
    # Evaluate
    print("Evaluating initial calibration on 100 samples...")
    metrics = calibrator.evaluate_calibration(dataset)
    
    print(f"\nInitial Calibration Metrics:")
    print(f"  ECE (Expected Calibration Error): {metrics.ece:.4f}")
    print(f"  MCE (Maximum Calibration Error): {metrics.mce:.4f}")
    print(f"  Accuracy: {metrics.accuracy:.4f}")
    print(f"  Average Confidence: {metrics.avg_confidence:.4f}")
    print(f"  Confidence-Accuracy Gap: {metrics.confidence_accuracy_gap:+.4f}")
    
    if metrics.ece > 0.15:
        print("\nStatus: POOR CALIBRATION (ECE > 0.15)")
        print("  The brain's confidence does not match its actual accuracy.")
        print("  Training needed to improve calibration.")
    
    return metrics


def demo_train_confidence_estimation(brain, calibrator, dataset):
    """Demo 3: Train confidence estimation."""
    print("\n" + "=" * 80)
    print("DEMO 3: Train Confidence Estimation")
    print("=" * 80)
    print("\nScenario: Train brain to calibrate confidence to accuracy\n")
    
    # Train
    history = calibrator.train_confidence_estimation(
        dataset=dataset,
        n_epochs=20,
        batch_size=32,
        learning_rate=0.001,
        log_interval=5,
        validation_split=0.2,
    )
    
    # Show training progress
    print("\nTraining complete!")
    print(f"\nFinal metrics:")
    print(f"  Train ECE: {history['train_ece'][-1]:.4f}")
    print(f"  Val ECE: {history['val_ece'][-1]:.4f}")
    print(f"  Train Accuracy: {history['train_accuracy'][-1]:.4f}")
    print(f"  Val Accuracy: {history['val_accuracy'][-1]:.4f}")
    
    # Show improvement
    initial_ece = history['val_ece'][0]
    final_ece = history['val_ece'][-1]
    improvement = ((initial_ece - final_ece) / initial_ece) * 100
    
    print(f"\nCalibration improvement:")
    print(f"  Initial ECE: {initial_ece:.4f}")
    print(f"  Final ECE: {final_ece:.4f}")
    print(f"  Improvement: {improvement:.1f}%")
    
    return history


def demo_evaluate_final_calibration(brain, calibrator, dataset):
    """Demo 4: Evaluate final (calibrated) confidence."""
    print("\n" + "=" * 80)
    print("DEMO 4: Evaluate Final Calibration")
    print("=" * 80)
    print("\nScenario: Check calibration after training\n")
    
    # Generate full report
    report = calibrator.generate_calibration_report(dataset)
    print(report)


def demo_calibration_history(calibrator):
    """Demo 5: Show calibration history."""
    print("\n" + "=" * 80)
    print("DEMO 5: Calibration History")
    print("=" * 80)
    print("\nScenario: Track calibration improvement over epochs\n")
    
    history = calibrator.get_calibration_history()
    
    if not history:
        print("No calibration history recorded.")
        return
    
    print(f"Recorded {len(history)} epochs\n")
    print(f"{'Epoch':>6} {'ECE':>8} {'Accuracy':>10} {'Confidence':>12} {'Gap':>8}")
    print("-" * 56)
    
    for epoch, metrics in history[::5]:  # Show every 5th epoch
        print(
            f"{epoch:>6} {metrics.ece:>8.4f} {metrics.accuracy:>10.4f} "
            f"{metrics.avg_confidence:>12.4f} {metrics.confidence_accuracy_gap:>8.4f}"
        )


def main():
    """Run all metacognitive calibration demonstrations."""
    print("\n" + "=" * 80)
    print("METACOGNITIVE CALIBRATION DEMONSTRATIONS")
    print("=" * 80)
    print("\nThis script demonstrates training a brain to calibrate its")
    print("confidence estimates to actual accuracy.\n")
    
    # Run demos
    brain, calibrator, dataset = demo_generate_calibration_dataset()
    
    initial_metrics = demo_evaluate_initial_calibration(brain, calibrator, dataset)
    
    history = demo_train_confidence_estimation(brain, calibrator, dataset)
    
    demo_evaluate_final_calibration(brain, calibrator, dataset)
    
    demo_calibration_history(calibrator)
    
    # Summary
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. Calibration measures confidence-accuracy alignment")
    print("  2. ECE (Expected Calibration Error) quantifies calibration quality")
    print("  3. Good calibration: ECE < 0.15")
    print("  4. Training improves confidence estimation over time")
    print("  5. Reliability diagrams visualize calibration")
    print("\nMetacognitive Skills Enabled:")
    print("  - Knowing when the brain is uncertain")
    print("  - Deciding when to ask for help or more information")
    print("  - Allocating cognitive resources effectively")
    print("  - Self-directed learning and error detection")
    print()


if __name__ == "__main__":
    main()
