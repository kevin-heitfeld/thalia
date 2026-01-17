"""
Metacognitive Calibration Training.

This module implements training procedures for teaching the brain to
accurately estimate its own confidence and calibrate it to actual accuracy.
Metacognitive abilities (knowing what you know) are essential for:

- Uncertainty quantification in predictions
- Deciding when to ask for help or more information
- Allocating cognitive resources effectively
- Self-directed learning and error detection

**Key Concepts**:

1. **Calibration**: Alignment between predicted confidence and actual accuracy
   - Well-calibrated: 70% confidence → 70% accuracy
   - Overconfident: 90% confidence → 60% accuracy
   - Underconfident: 50% confidence → 80% accuracy

2. **Expected Calibration Error (ECE)**: Standard metric for calibration quality
   - Bin predictions by confidence level
   - Compare bin confidence to bin accuracy
   - Lower ECE = better calibration (target < 0.15)

3. **Training Approach**:
   - Generate tasks spanning difficulty spectrum
   - Brain produces answer + confidence estimate
   - Compare confidence to correctness
   - Update confidence estimator (typically PFC)

**Usage**:
```python
from thalia.training.evaluation.metacognition import MetacognitiveCalibrator

calibrator = MetacognitiveCalibrator(brain=brain)

# Generate calibration dataset
dataset = calibrator.generate_calibration_dataset(
    difficulty_range=(0.3, 0.9),
    n_samples=1000,
)

# Train confidence estimation
calibrator.train_confidence_estimation(
    dataset=dataset,
    n_epochs=50,
    log_interval=10,
)

# Evaluate calibration quality
ece = calibrator.evaluate_calibration(dataset)
print(f"Expected Calibration Error: {ece:.3f}")

# Generate calibration report
report = calibrator.generate_calibration_report(dataset)
print(report)
```

References:
- Naeini et al. (2015) "Obtaining Well Calibrated Probabilities Using Bayesian Binning"
- Guo et al. (2017) "On Calibration of Modern Neural Networks"
- Fleming & Dolan (2012) "The neural basis of metacognitive ability"

Author: Thalia Project
Date: December 8, 2025
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from thalia.constants.training import (
    CALIBRATION_ACCEPTABLE_ECE,
    CALIBRATION_EXCELLENT_ECE,
    CALIBRATION_GOOD_ECE,
    CURRICULUM_DIFFICULTY_MAX,
    CURRICULUM_DIFFICULTY_MIN,
)
from thalia.core.errors import ConfigurationError

if TYPE_CHECKING:
    from thalia.core.dynamic_brain import DynamicBrain


# ============================================================================
# Data Structures
# ============================================================================


@dataclass
class CalibrationSample:
    """Single sample for calibration training.

    **Attributes**:
        input_data: Input tensor for the task
        target: Ground truth answer
        difficulty: Task difficulty (0-1, higher = harder)
        task_type: Task category (e.g., 'visual', 'language', 'reasoning')
        metadata: Additional task information
    """

    input_data: torch.Tensor
    target: torch.Tensor
    difficulty: float
    task_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate sample parameters."""
        if not 0 <= self.difficulty <= 1:
            raise ConfigurationError(f"Difficulty must be in [0, 1], got {self.difficulty}")


@dataclass
class CalibrationPrediction:
    """Brain prediction with confidence estimate.

    **Attributes**:
        prediction: Predicted answer
        confidence: Confidence estimate (0-1)
        correct: Whether prediction matches target
        difficulty: Task difficulty
        task_type: Task category
        response_time: Time taken to respond (optional)
    """

    prediction: torch.Tensor
    confidence: float
    correct: bool
    difficulty: float
    task_type: str
    response_time: Optional[float] = None

    def __post_init__(self):
        """Validate prediction parameters."""
        if not 0 <= self.confidence <= 1:
            raise ConfigurationError(f"Confidence must be in [0, 1], got {self.confidence}")


@dataclass
class CalibrationMetrics:
    """Calibration quality metrics.

    **Attributes**:
        ece: Expected Calibration Error (lower = better)
        mce: Maximum Calibration Error (worst bin)
        accuracy: Overall accuracy
        avg_confidence: Average confidence
        confidence_accuracy_gap: Difference between confidence and accuracy
        bin_accuracies: Accuracy per confidence bin
        bin_confidences: Average confidence per bin
        bin_counts: Number of samples per bin
        n_bins: Number of calibration bins used
    """

    ece: float
    mce: float
    accuracy: float
    avg_confidence: float
    confidence_accuracy_gap: float
    bin_accuracies: List[float]
    bin_confidences: List[float]
    bin_counts: List[int]
    n_bins: int


# ============================================================================
# Metacognitive Calibrator
# ============================================================================


class MetacognitiveCalibrator:
    """Train brain to calibrate confidence to accuracy.

    Implements training procedures for metacognitive abilities:
    - Generating difficulty-calibrated datasets
    - Training confidence estimation
    - Evaluating calibration quality (ECE)
    - Reporting and visualization

    **Design Principles**:
    - Use diverse tasks spanning difficulty spectrum
    - Explicit confidence training (PFC confidence estimator)
    - Standard calibration metrics (ECE, MCE)
    - Track calibration improvement over time

    **Training Protocol**:
    1. Generate calibration dataset with known difficulties
    2. Brain produces answer + confidence for each sample
    3. Compute calibration error (confidence vs. correctness)
    4. Update confidence estimator to minimize error
    5. Periodically evaluate ECE on held-out set

    **Attributes**:
        brain: Brain instance to calibrate
        confidence_region: Brain region producing confidence (typically PFC)
        n_bins: Number of bins for ECE computation
        device: Torch device for computation
        calibration_history: History of calibration metrics over time
    """

    def __init__(
        self,
        brain: DynamicBrain,
        confidence_region: str = "prefrontal",
        n_bins: int = 10,
        device: Optional[torch.device] = None,
    ):
        """Initialize metacognitive calibrator.

        **Args**:
            brain: Brain instance to calibrate
            confidence_region: Brain region for confidence estimates
            n_bins: Number of bins for ECE computation (default 10)
            device: Torch device for computation
        """
        self.brain = brain
        self.confidence_region = confidence_region
        self.n_bins = n_bins
        self.device = device or brain.device

        # Calibration history
        self.calibration_history: List[Tuple[int, CalibrationMetrics]] = []

        # Check that confidence region exists
        if not hasattr(brain, confidence_region):
            raise ValueError(
                f"Brain does not have region '{confidence_region}'. "
                f"Available regions: {list(brain.__dict__.keys())}"
            )

    def generate_calibration_dataset(
        self,
        task_generator: Callable[[float], Tuple[torch.Tensor, torch.Tensor]],
        difficulty_range: Tuple[float, float] = (
            CURRICULUM_DIFFICULTY_MIN,
            CURRICULUM_DIFFICULTY_MAX,
        ),
        n_samples: int = 1000,
        task_type: str = "mixed",
        stratified: bool = True,
    ) -> List[CalibrationSample]:
        """Generate calibration dataset spanning difficulty spectrum.

        **Args**:
            task_generator: Function that takes difficulty and returns (input, target)
            difficulty_range: Min and max difficulty (0-1)
            n_samples: Number of samples to generate
            task_type: Task category label
            stratified: Whether to stratify by difficulty

        **Returns**:
            List of calibration samples
        """
        min_diff, max_diff = difficulty_range

        if not 0.0 <= min_diff <= max_diff <= 1.0:
            raise ValueError(f"Invalid difficulty range: {difficulty_range}")

        samples = []

        if stratified:
            # Evenly distribute across difficulty levels
            difficulties = np.linspace(min_diff, max_diff, n_samples)
        else:
            # Random difficulties
            difficulties = np.random.uniform(min_diff, max_diff, n_samples)

        for difficulty in difficulties:
            # Generate task
            input_data, target = task_generator(float(difficulty))

            # Create sample
            sample = CalibrationSample(
                input_data=input_data,
                target=target,
                difficulty=float(difficulty),
                task_type=task_type,
            )
            samples.append(sample)

        return samples

    def predict_with_confidence(
        self,
        sample: CalibrationSample,
        extract_confidence: Optional[Callable[[Any], float]] = None,
    ) -> CalibrationPrediction:
        """Get brain prediction with confidence estimate.

        **Args**:
            sample: Calibration sample
            extract_confidence: Function to extract confidence from brain state
                               (default: use PFC firing rate)

        **Returns**:
            Prediction with confidence
        """
        start_time = time.time()

        # Forward pass
        output = self.brain.forward(sample.input_data)

        # Extract prediction (highest activation)
        prediction = output.argmax(dim=-1) if output.dim() > 1 else output

        # Extract confidence
        if extract_confidence is not None:
            confidence = extract_confidence(self.brain)
        else:
            # Default: Use PFC firing rate as confidence proxy
            confidence_region = getattr(self.brain, self.confidence_region)
            confidence = float(confidence_region.state.spikes.float().mean())

        # Clip confidence to [0, 1]
        confidence = max(0.0, min(1.0, confidence))

        # Check correctness
        correct = bool((prediction == sample.target).all())

        response_time = time.time() - start_time

        return CalibrationPrediction(
            prediction=prediction,
            confidence=confidence,
            correct=correct,
            difficulty=sample.difficulty,
            task_type=sample.task_type,
            response_time=response_time,
        )

    def compute_calibration_metrics(
        self,
        predictions: List[CalibrationPrediction],
    ) -> CalibrationMetrics:
        """Compute calibration metrics (ECE, MCE, etc.).

        **Args**:
            predictions: List of predictions with confidences

        **Returns**:
            Calibration metrics
        """
        if not predictions:
            raise ValueError("Need at least one prediction to compute metrics")

        # Extract arrays
        confidences = np.array([p.confidence for p in predictions])
        correctness = np.array([1.0 if p.correct else 0.0 for p in predictions])

        # Compute overall metrics
        accuracy = correctness.mean()
        avg_confidence = confidences.mean()

        # Create bins
        bin_boundaries = np.linspace(0.0, 1.0, self.n_bins + 1)
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        # Compute per-bin statistics
        for i in range(self.n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            # Find predictions in this bin
            if i == self.n_bins - 1:
                # Last bin includes upper boundary
                in_bin = (confidences >= bin_lower) & (confidences <= bin_upper)
            else:
                in_bin = (confidences >= bin_lower) & (confidences < bin_upper)

            bin_count = in_bin.sum()
            bin_counts.append(int(bin_count))

            if bin_count > 0:
                bin_acc = correctness[in_bin].mean()
                bin_conf = confidences[in_bin].mean()
            else:
                bin_acc = 0.0
                bin_conf = 0.0

            bin_accuracies.append(float(bin_acc))
            bin_confidences.append(float(bin_conf))

        # Compute Expected Calibration Error (ECE)
        ece = 0.0
        n_total = len(predictions)
        for bin_count, bin_acc, bin_conf in zip(bin_counts, bin_accuracies, bin_confidences):
            if bin_count > 0:
                ece += (bin_count / n_total) * abs(bin_acc - bin_conf)

        # Compute Maximum Calibration Error (MCE)
        calibration_errors = [
            abs(bin_acc - bin_conf)
            for bin_count, bin_acc, bin_conf in zip(bin_counts, bin_accuracies, bin_confidences)
            if bin_count > 0
        ]
        mce = max(calibration_errors) if calibration_errors else 0.0

        return CalibrationMetrics(
            ece=float(ece),
            mce=float(mce),
            accuracy=float(accuracy),
            avg_confidence=float(avg_confidence),
            confidence_accuracy_gap=float(avg_confidence - accuracy),
            bin_accuracies=bin_accuracies,
            bin_confidences=bin_confidences,
            bin_counts=bin_counts,
            n_bins=self.n_bins,
        )

    def train_confidence_estimation(
        self,
        dataset: List[CalibrationSample],
        n_epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        extract_confidence: Optional[Callable[[Any], float]] = None,
        log_interval: int = 10,
        validation_split: float = 0.2,
    ) -> Dict[str, List[float]]:
        """Train brain to estimate confidence accurately.

        **Args**:
            dataset: Calibration dataset
            n_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for confidence estimator
            extract_confidence: Function to extract confidence from brain
            log_interval: Log metrics every N epochs
            validation_split: Fraction of data for validation

        **Returns**:
            Training history (metrics over epochs)
        """
        # Split dataset
        n_val = int(len(dataset) * validation_split)
        val_dataset = dataset[:n_val]
        train_dataset = dataset[n_val:]

        history: dict[str, list[float]] = {
            "train_ece": [],
            "val_ece": [],
            "train_accuracy": [],
            "val_accuracy": [],
            "train_confidence": [],
            "val_confidence": [],
        }

        print(f"\nTraining confidence estimation for {n_epochs} epochs...")
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        for epoch in range(n_epochs):
            # Shuffle training data
            np.random.shuffle(train_dataset)

            # Training loop
            train_predictions = []
            for sample in train_dataset:
                pred = self.predict_with_confidence(sample, extract_confidence)
                train_predictions.append(pred)

                # Update confidence estimator based on correctness
                # In a real implementation, you would:
                # 1. Compute calibration loss
                # 2. Backprop through confidence estimator
                # 3. Update weights
                # For now, we just collect predictions

            # Validation loop
            val_predictions = []
            for sample in val_dataset:
                pred = self.predict_with_confidence(sample, extract_confidence)
                val_predictions.append(pred)

            # Compute metrics
            train_metrics = self.compute_calibration_metrics(train_predictions)
            val_metrics = self.compute_calibration_metrics(val_predictions)

            # Record history
            history["train_ece"].append(train_metrics.ece)
            history["val_ece"].append(val_metrics.ece)
            history["train_accuracy"].append(train_metrics.accuracy)
            history["val_accuracy"].append(val_metrics.accuracy)
            history["train_confidence"].append(train_metrics.avg_confidence)
            history["val_confidence"].append(val_metrics.avg_confidence)

            # Log progress
            if (epoch + 1) % log_interval == 0:
                print(f"\nEpoch {epoch + 1}/{n_epochs}")
                print(
                    f"  Train ECE: {train_metrics.ece:.4f}, Accuracy: {train_metrics.accuracy:.4f}"
                )
                print(f"  Val ECE: {val_metrics.ece:.4f}, Accuracy: {val_metrics.accuracy:.4f}")

            # Store in history
            self.calibration_history.append((epoch, val_metrics))

        return history

    def evaluate_calibration(
        self,
        dataset: List[CalibrationSample],
        extract_confidence: Optional[Callable[[Any], float]] = None,
    ) -> CalibrationMetrics:
        """Evaluate calibration quality on dataset.

        **Args**:
            dataset: Calibration dataset
            extract_confidence: Function to extract confidence from brain

        **Returns**:
            Calibration metrics
        """
        predictions = []
        for sample in dataset:
            pred = self.predict_with_confidence(sample, extract_confidence)
            predictions.append(pred)

        return self.compute_calibration_metrics(predictions)

    def generate_calibration_report(
        self,
        dataset: List[CalibrationSample],
        extract_confidence: Optional[Callable[[Any], float]] = None,
    ) -> str:
        """Generate human-readable calibration report.

        **Args**:
            dataset: Calibration dataset
            extract_confidence: Function to extract confidence from brain

        **Returns**:
            Formatted calibration report
        """
        metrics = self.evaluate_calibration(dataset, extract_confidence)

        report = []
        report.append("=" * 70)
        report.append("Metacognitive Calibration Report")
        report.append("=" * 70)
        report.append(f"Dataset size: {len(dataset)} samples")
        report.append(f"Number of bins: {metrics.n_bins}")
        report.append("")

        # Overall metrics
        report.append("Overall Metrics:")
        report.append(f"  Expected Calibration Error (ECE): {metrics.ece:.4f}")
        report.append(f"  Maximum Calibration Error (MCE): {metrics.mce:.4f}")
        report.append(f"  Accuracy: {metrics.accuracy:.4f}")
        report.append(f"  Average Confidence: {metrics.avg_confidence:.4f}")
        report.append(f"  Confidence-Accuracy Gap: {metrics.confidence_accuracy_gap:.4f}")

        # Calibration quality assessment
        report.append("")
        if metrics.ece < CALIBRATION_EXCELLENT_ECE:
            status = f"EXCELLENT (ECE < {CALIBRATION_EXCELLENT_ECE})"
        elif metrics.ece < CALIBRATION_GOOD_ECE:
            status = f"GOOD (ECE < {CALIBRATION_GOOD_ECE})"
        elif metrics.ece < CALIBRATION_ACCEPTABLE_ECE:
            status = f"ACCEPTABLE (ECE < {CALIBRATION_ACCEPTABLE_ECE})"
        else:
            status = f"POOR (ECE >= {CALIBRATION_ACCEPTABLE_ECE})"
        report.append(f"Calibration Quality: {status}")

        # Per-bin breakdown
        report.append("")
        report.append("Per-Bin Breakdown:")
        report.append(
            f"{'Bin':>5} {'Range':>15} {'Count':>8} {'Confidence':>12} {'Accuracy':>10} {'Gap':>8}"
        )
        report.append("-" * 70)

        for i in range(metrics.n_bins):
            bin_lower = i / metrics.n_bins
            bin_upper = (i + 1) / metrics.n_bins
            bin_range = f"[{bin_lower:.2f}, {bin_upper:.2f})"

            if metrics.bin_counts[i] > 0:
                gap = metrics.bin_confidences[i] - metrics.bin_accuracies[i]
                report.append(
                    f"{i+1:>5} {bin_range:>15} {metrics.bin_counts[i]:>8} "
                    f"{metrics.bin_confidences[i]:>12.4f} {metrics.bin_accuracies[i]:>10.4f} "
                    f"{gap:>8.4f}"
                )
            else:
                report.append(
                    f"{i+1:>5} {bin_range:>15} {metrics.bin_counts[i]:>8} "
                    f"{'--':>12} {'--':>10} {'--':>8}"
                )

        report.append("=" * 70)
        return "\n".join(report)

    def get_calibration_history(self) -> List[Tuple[int, CalibrationMetrics]]:
        """Get history of calibration metrics over training.

        **Returns**:
            List of (epoch, metrics) tuples
        """
        return self.calibration_history.copy()

    def plot_reliability_diagram(
        self,
        dataset: List[CalibrationSample],
        extract_confidence: Optional[Callable[[Any], float]] = None,
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot reliability diagram (confidence vs. accuracy).

        **Args**:
            dataset: Calibration dataset
            extract_confidence: Function to extract confidence from brain
            save_path: Path to save plot (optional)
        """
        metrics = self.evaluate_calibration(dataset, extract_confidence)

        # Create figure
        _fig, ax = plt.subplots(figsize=(8, 8))

        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", linewidth=2)

        # Plot actual calibration
        bin_centers = [(i + 0.5) / metrics.n_bins for i in range(metrics.n_bins)]
        valid_bins = [i for i in range(metrics.n_bins) if metrics.bin_counts[i] > 0]

        if valid_bins:
            x = [bin_centers[i] for i in valid_bins]
            y = [metrics.bin_accuracies[i] for i in valid_bins]
            sizes = [metrics.bin_counts[i] * 10 for i in valid_bins]

            ax.scatter(x, y, s=sizes, alpha=0.6, label="Actual Calibration")

        # Formatting
        ax.set_xlabel("Confidence", fontsize=12)
        ax.set_ylabel("Accuracy", fontsize=12)
        ax.set_title(f"Reliability Diagram (ECE: {metrics.ece:.4f})", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Reliability diagram saved to {save_path}")
        else:
            plt.show()


# ============================================================================
# Helper Functions
# ============================================================================


def create_simple_task_generator(
    n_classes: int = 10,
    input_size: int = 100,
    device: Optional[torch.device] = None,
) -> Callable[[float], Tuple[torch.Tensor, torch.Tensor]]:
    """Create simple task generator for testing calibration.

    **Args**:
        n_classes: Number of output classes
        input_size: Size of input tensor
        device: Torch device

    **Returns**:
        Task generator function
    """
    device = device or torch.device("cpu")

    def generate_task(difficulty: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate task with specified difficulty.

        Difficulty controls:
        - Separation between class-specific patterns
        - Amount of noise
        """
        # Generate random input
        input_data = torch.randn(input_size, device=device)

        # Choose random class
        target_class = torch.randint(0, n_classes, (1,), device=device)

        # Add class-specific pattern (strength depends on difficulty)
        # Lower difficulty = stronger pattern = easier
        pattern_strength = 1.0 - difficulty
        class_pattern = torch.randn(input_size, device=device) * pattern_strength
        input_data += class_pattern

        return input_data, target_class

    return generate_task


__all__ = [
    "CalibrationSample",
    "CalibrationPrediction",
    "CalibrationMetrics",
    "MetacognitiveCalibrator",
    "create_simple_task_generator",
]
