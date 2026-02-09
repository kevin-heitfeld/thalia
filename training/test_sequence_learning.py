"""
Temporal Sequence Learning Evaluation for Default Brain.

This script tests the default brain architecture on temporal sequence prediction,
evaluating:
- Next-step prediction accuracy (hippocampal memory)
- Pattern completion (given A-B, predict C)
- Violation detection (prediction error when patterns break)
- Generalization (novel symbol combinations)

Success Metrics:
- Basic: >50% accuracy on single pattern type
- Intermediate: >70% on multiple patterns
- Advanced: Detect violations, one-shot learning, noise robustness
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from thalia.brain import BrainConfig, BrainBuilder, DynamicBrain
from thalia.datasets import PatternType, SequenceConfig, TemporalSequenceDataset


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class TrainingConfig:
    """Configuration for temporal sequence training."""

    # Brain configuration
    thalamus_relay_size: int = 50  # Smaller for sequence symbols
    cortex_size: int = 200  # Reduced from default 1000
    pfc_executive_size: int = 150
    striatum_actions: int = 5  # One per symbol

    # Dataset configuration
    n_symbols: int = 5
    sequence_length: int = 10
    pattern_types: Optional[List[str]] = None  # ["ABC", "ABA", "AAB"]

    # Training configuration
    n_training_trials: int = 1000
    n_test_trials: int = 100
    n_violation_trials: int = 50

    # Timing (biological timesteps)
    timestep_ms: float = 1.0  # 1ms resolution
    trial_duration_ms: float = 100.0  # 100ms per trial
    inter_trial_interval_ms: float = 50.0  # 50ms between trials

    # Learning parameters
    use_dopamine_modulation: bool = True
    dopamine_baseline: float = 0.0
    dopamine_lr: float = 0.1  # How much prediction error affects dopamine
    reward_on_correct: float = 1.0
    penalty_on_error: float = -0.5

    # Spike encoding
    spike_rate_active: float = 0.8  # High rate for active symbol
    spike_rate_baseline: float = 0.05  # Low background rate

    # Evaluation
    min_spikes_for_prediction: int = 1  # Minimum spikes to count as prediction

    # Output
    output_dir: str = "data/results/sequence_learning"
    save_brain: bool = False
    verbose: bool = True

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        if self.pattern_types is None:
            self.pattern_types = ["ABC", "ABA", "AAB"]


# ============================================================================
# Helper Functions
# ============================================================================


def encode_symbol_to_spikes(
    symbol: int,
    n_symbols: int,
    spike_rate_active: float,
    spike_rate_baseline: float,
    device: torch.device,
    thalamus_size: Optional[int] = None,
) -> torch.Tensor:
    """
    Encode a symbol as spike rates (one-hot style).

    Args:
        symbol: Symbol index (0 to n_symbols-1)
        n_symbols: Total number of symbols
        spike_rate_active: Firing rate for active symbol
        spike_rate_baseline: Background firing rate
        device: Torch device
        thalamus_size: Target size for thalamic input (pad if larger than n_symbols)

    Returns:
        Spike pattern (n_symbols or thalamus_size,) with Bernoulli spikes
    """
    # Create rate pattern
    rates = torch.full((n_symbols,), spike_rate_baseline, device=device)
    rates[symbol] = spike_rate_active

    # Sample Bernoulli spikes
    spikes = torch.bernoulli(rates)

    # Pad to match thalamic relay size if specified
    # This simulates how real sensory systems (retina, cochlea) have many neurons
    # but the experiment uses fewer symbols
    if thalamus_size is not None and thalamus_size > n_symbols:
        # Pad with baseline noise
        padding_size = thalamus_size - n_symbols
        padding = torch.bernoulli(
            torch.full((padding_size,), spike_rate_baseline, device=device)
        )
        spikes = torch.cat([spikes, padding])

    return spikes


def decode_spikes_to_symbol(
    spike_counts: torch.Tensor,
    min_spikes: int = 1,
) -> Optional[int]:
    """
    Decode spike counts to most active symbol.

    Args:
        spike_counts: Accumulated spike counts per symbol (n_symbols,)
        min_spikes: Minimum total spikes to count as valid prediction

    Returns:
        Symbol index or None if below threshold
    """
    # Ensure float for comparison
    if spike_counts.dtype == torch.bool:
        spike_counts = spike_counts.float()

    total_spikes = spike_counts.sum()
    if total_spikes < min_spikes:
        return None

    # Return symbol with most spikes (ties go to first index)
    return int(torch.argmax(spike_counts).item())


def compute_prediction_accuracy(
    predictions: List[Optional[int]],
    targets: List[int],
) -> float:
    """
    Compute prediction accuracy.

    Args:
        predictions: List of predicted symbols (None = no prediction)
        targets: List of target symbols

    Returns:
        Accuracy (0.0 to 1.0)
    """
    if len(predictions) != len(targets):
        raise ValueError(f"Length mismatch: {len(predictions)} vs {len(targets)}")

    correct = sum(1 for pred, tgt in zip(predictions, targets) if pred == tgt)
    return correct / len(targets) if targets else 0.0


def compute_prediction_error(
    prediction_spikes: torch.Tensor,
    target_symbol: int,
    n_symbols: int,
) -> float:
    """
    Compute prediction error for dopamine signal.

    Uses a simple error: actual_rate - predicted_rate for target symbol.

    Args:
        prediction_spikes: Predicted spike pattern (n_symbols,)
        target_symbol: Actual next symbol
        n_symbols: Total number of symbols

    Returns:
        Error magnitude (-1.0 to 1.0)
    """
    # Create target one-hot
    target_onehot = torch.zeros(n_symbols, device=prediction_spikes.device)
    target_onehot[target_symbol] = 1.0

    # Normalize prediction spikes to rates
    pred_rates = prediction_spikes / (prediction_spikes.sum() + 1e-6)

    # Mean squared error
    error = F.mse_loss(pred_rates, target_onehot).item()

    return error


# ============================================================================
# Main Training Script
# ============================================================================


class SequenceLearningExperiment:
    """Runs temporal sequence learning experiment."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.brain: Optional[DynamicBrain] = None
        self.dataset: Optional[TemporalSequenceDataset] = None

        # Spike accumulation buffer (for temporal decoding)
        self.spike_buffer: Optional[torch.Tensor] = None
        self.buffer_size: int = 5  # Accumulate over 5 timesteps

        # Metrics
        self.metrics: Dict[str, List[float]] = {
            "train_accuracy": [],
            "train_dopamine": [],
            "test_accuracy": [],
            "violation_detection": [],
        }

    def setup(self):
        """Initialize brain and dataset."""
        if self.config.verbose:
            print("=" * 80)
            print("TEMPORAL SEQUENCE LEARNING EVALUATION")
            print("=" * 80)
            print(f"\nDevice: {self.device}")
            print(f"Output directory: {self.output_dir}")

        # Build brain with default architecture
        if self.config.verbose:
            print("\n[1/2] Building default brain architecture...")

        brain_config = BrainConfig(device=self.device, dt_ms=self.config.timestep_ms)
        self.brain = BrainBuilder.preset(
            name="default",
            brain_config=brain_config,
            thalamus_relay_size=self.config.thalamus_relay_size,
            cortex_size=self.config.cortex_size,
            pfc_executive_size=self.config.pfc_executive_size,
            striatum_actions=self.config.striatum_actions,
        )

        if self.config.verbose:
            print(f"   ✓ Brain created with {self._count_parameters()} parameters")
            print(f"   ✓ Regions: {list(self.brain.regions.keys())}")

        # Create dataset
        if self.config.verbose:
            print("\n[2/2] Creating temporal sequence dataset...")

        pattern_types = [PatternType(pt.lower()) for pt in self.config.pattern_types]
        seq_config = SequenceConfig(
            n_symbols=self.config.n_symbols,
            sequence_length=self.config.sequence_length,
            pattern_types=pattern_types,
            violation_probability=0.0,  # Controlled separately
            device=self.device,
        )
        self.dataset = TemporalSequenceDataset(seq_config)

        if self.config.verbose:
            print(f"   ✓ Dataset created: {self.config.n_symbols} symbols, "
                  f"{self.config.sequence_length} steps")
            print(f"   ✓ Pattern types: {self.config.pattern_types}")

    def _accumulate_spikes(self, spikes: torch.Tensor) -> torch.Tensor:
        """
        Accumulate spikes over a temporal window for rate-based decoding.

        Args:
            spikes: Current timestep spikes (n_symbols,) bool tensor

        Returns:
            Accumulated spike counts (n_symbols,) float tensor
        """
        if self.spike_buffer is None:
            self.spike_buffer = torch.zeros(
                self.buffer_size,
                self.config.n_symbols,
                device=self.device
            )

        # Shift buffer and add new spikes
        self.spike_buffer = torch.roll(self.spike_buffer, shifts=-1, dims=0)
        self.spike_buffer[-1] = spikes.float()

        # Return accumulated counts
        return self.spike_buffer.sum(dim=0)

    def _reset_spike_buffer(self):
        """Reset spike accumulation buffer between trials."""
        self.spike_buffer = None

    def train(self):
        """Run training phase."""
        if self.config.verbose:
            print("\n" + "=" * 80)
            print("TRAINING PHASE")
            print("=" * 80)

        train_predictions = []
        train_targets = []
        train_dopamine_values = []

        progress = tqdm(
            range(self.config.n_training_trials),
            desc="Training",
            disable=not self.config.verbose,
        )

        for trial_idx in progress:
            # Reset spike buffer at start of each trial
            self._reset_spike_buffer()

            # Generate sequence (brain maintains continuous learning state)
            sequence, targets, _pattern_type = self.dataset.generate_sequence()

            # Convert to symbol indices (assuming one-hot encoding)
            symbol_sequence = [torch.argmax(seq).item() for seq in sequence]
            target_sequence = [torch.argmax(tgt).item() for tgt in targets]

            # Present sequence timestep by timestep
            trial_predictions = []
            trial_targets = []
            trial_dopamine = []

            for t in range(len(symbol_sequence) - 1):
                # Current symbol
                current_symbol = symbol_sequence[t]
                next_symbol = target_sequence[t]  # What should come next

                # Encode to spikes (pad to match thalamic relay size)
                input_spikes = encode_symbol_to_spikes(
                    current_symbol,
                    self.config.n_symbols,
                    self.config.spike_rate_active,
                    self.config.spike_rate_baseline,
                    self.device,
                    thalamus_size=self.brain.thalamus.relay_size,
                )

                # Forward through brain
                # Thalamus receives external sensory input via ascending pathways
                # (e.g., retina→LGN, cochlea→MGN, mechanoreceptors→VPN)
                brain_input = {"thalamus": {"sensory": input_spikes}}
                brain_output = self.brain.forward(brain_input)

                # Read hippocampus CA1 output (memory prediction)
                # brain_output is Dict[region_name, Dict[port_name, Tensor]]
                # Get first available port from preferred region
                if "hippocampus" in brain_output and brain_output["hippocampus"]:
                    prediction_spikes = list(brain_output["hippocampus"].values())[0]
                elif "cortex" in brain_output and brain_output["cortex"]:
                    prediction_spikes = list(brain_output["cortex"].values())[0]
                elif "pfc" in brain_output and brain_output["pfc"]:
                    prediction_spikes = list(brain_output["pfc"].values())[0]
                else:
                    # Fallback: use any available output from any region
                    for region_output in brain_output.values():
                        if region_output:
                            prediction_spikes = list(region_output.values())[0]
                            break

                # Take first n_symbols neurons as prediction
                if prediction_spikes.shape[0] > self.config.n_symbols:
                    prediction_spikes = prediction_spikes[: self.config.n_symbols]
                elif prediction_spikes.shape[0] < self.config.n_symbols:
                    # Pad if needed
                    padding = torch.zeros(
                        self.config.n_symbols - prediction_spikes.shape[0],
                        device=self.device,
                        dtype=prediction_spikes.dtype,
                    )
                    prediction_spikes = torch.cat([prediction_spikes, padding])

                # Accumulate spikes over temporal window
                spike_counts = self._accumulate_spikes(prediction_spikes)

                # Decode prediction from accumulated counts
                predicted_symbol = decode_spikes_to_symbol(
                    spike_counts,
                    self.config.min_spikes_for_prediction,
                )

                trial_predictions.append(predicted_symbol)
                trial_targets.append(next_symbol)

                # Compute dopamine signal based on prediction error
                if self.config.use_dopamine_modulation:
                    error = compute_prediction_error(
                        spike_counts,  # Use accumulated counts for error
                        next_symbol,
                        self.config.n_symbols,
                    )

                    # Convert error to dopamine (negative error = positive dopamine)
                    dopamine = self.config.dopamine_baseline - (error * self.config.dopamine_lr)
                    dopamine = np.clip(dopamine, -1.0, 1.0)

                    self.brain.deliver_reward(external_reward=dopamine)

                    trial_dopamine.append(dopamine)

            # Accumulate metrics
            train_predictions.extend(trial_predictions)
            train_targets.extend(trial_targets)
            train_dopamine_values.extend(trial_dopamine)

            # Update progress bar every 100 trials
            if trial_idx % 100 == 0 and trial_idx > 0:
                recent_acc = compute_prediction_accuracy(
                    train_predictions[-100:],
                    train_targets[-100:],
                )
                progress.set_postfix({"accuracy": f"{recent_acc:.2%}"})

        # Compute final training accuracy
        train_accuracy = compute_prediction_accuracy(train_predictions, train_targets)
        avg_dopamine = np.mean(train_dopamine_values) if train_dopamine_values else 0.0

        self.metrics["train_accuracy"].append(train_accuracy)
        self.metrics["train_dopamine"].append(avg_dopamine)

        if self.config.verbose:
            print("\n✓ Training complete:")
            print(f"   Accuracy: {train_accuracy:.2%}")
            print(f"   Avg Dopamine: {avg_dopamine:.3f}")
            print(f"   Trials: {self.config.n_training_trials}")

    def test(self):
        """Run testing phase (novel sequences)."""
        if self.config.verbose:
            print("\n" + "=" * 80)
            print("TESTING PHASE")
            print("=" * 80)

        test_predictions = []
        test_targets = []

        progress = tqdm(
            range(self.config.n_test_trials),
            desc="Testing",
            disable=not self.config.verbose,
        )

        for _trial_idx in progress:
            # Reset spike buffer at start of each trial
            self._reset_spike_buffer()

            # Generate NEW sequence (not seen in training)
            sequence, targets, pattern_type = self.dataset.generate_sequence()
            symbol_sequence = [torch.argmax(seq).item() for seq in sequence]
            target_sequence = [torch.argmax(tgt).item() for tgt in targets]

            # Present sequence (no learning)
            for t in range(len(symbol_sequence) - 1):
                current_symbol = symbol_sequence[t]
                next_symbol = target_sequence[t]

                input_spikes = encode_symbol_to_spikes(
                    current_symbol,
                    self.config.n_symbols,
                    self.config.spike_rate_active,
                    self.config.spike_rate_baseline,
                    self.device,
                    thalamus_size=self.brain.thalamus.relay_size,
                )

                # Thalamus receives external sensory input via ascending pathways
                brain_input = {"thalamus": {"sensory": input_spikes}}
                brain_output = self.brain.forward(brain_input)

                # Get prediction (same logic as training)
                if "hippocampus" in brain_output and brain_output["hippocampus"]:
                    prediction_spikes = list(brain_output["hippocampus"].values())[0]
                elif "cortex" in brain_output and brain_output["cortex"]:
                    prediction_spikes = list(brain_output["cortex"].values())[0]
                else:
                    for region_output in brain_output.values():
                        if region_output:
                            prediction_spikes = list(region_output.values())[0]
                            break

                if prediction_spikes.shape[0] > self.config.n_symbols:
                    prediction_spikes = prediction_spikes[: self.config.n_symbols]
                elif prediction_spikes.shape[0] < self.config.n_symbols:
                    padding = torch.zeros(
                        self.config.n_symbols - prediction_spikes.shape[0],
                        device=self.device,
                        dtype=prediction_spikes.dtype,
                    )
                    prediction_spikes = torch.cat([prediction_spikes, padding])

                # Accumulate spikes over temporal window
                spike_counts = self._accumulate_spikes(prediction_spikes)

                predicted_symbol = decode_spikes_to_symbol(
                    spike_counts,
                    self.config.min_spikes_for_prediction,
                )

                test_predictions.append(predicted_symbol)
                test_targets.append(next_symbol)

        # Compute test accuracy
        test_accuracy = compute_prediction_accuracy(test_predictions, test_targets)
        self.metrics["test_accuracy"].append(test_accuracy)

        if self.config.verbose:
            print("\n✓ Testing complete:")
            print(f"   Accuracy: {test_accuracy:.2%}")
            print(f"   Trials: {self.config.n_test_trials}")
            print(f"   Random baseline: {1.0 / self.config.n_symbols:.2%}")

    def test_violation_detection(self):
        """Test if brain detects pattern violations."""
        if self.config.verbose:
            print("\n" + "=" * 80)
            print("VIOLATION DETECTION")
            print("=" * 80)

        # Test 1: Normal patterns
        normal_errors = []
        progress = tqdm(
            range(self.config.n_violation_trials),
            desc="Normal patterns",
            disable=not self.config.verbose,
        )

        for _trial_idx in progress:
            # Reset spike buffer at start of each trial
            self._reset_spike_buffer()

            sequence, targets, _ = self.dataset.generate_sequence(include_violation=False)

            # Measure prediction error on normal sequence
            error_sum = 0.0
            for t in range(len(sequence) - 1):
                current_symbol = torch.argmax(sequence[t]).item()
                next_symbol = torch.argmax(targets[t]).item()

                input_spikes = encode_symbol_to_spikes(
                    current_symbol,
                    self.config.n_symbols,
                    self.config.spike_rate_active,
                    self.config.spike_rate_baseline,
                    self.device,
                    thalamus_size=self.brain.thalamus.relay_size,
                )

                # Thalamus receives external sensory input via ascending pathways
                brain_input = {"thalamus": {"sensory": input_spikes}}
                brain_output = self.brain.forward(brain_input)

                if "hippocampus" in brain_output and brain_output["hippocampus"]:
                    prediction_spikes = list(brain_output["hippocampus"].values())[0]
                else:
                    for region_output in brain_output.values():
                        if region_output:
                            prediction_spikes = list(region_output.values())[0]
                            break

                if prediction_spikes.shape[0] > self.config.n_symbols:
                    prediction_spikes = prediction_spikes[: self.config.n_symbols]
                elif prediction_spikes.shape[0] < self.config.n_symbols:
                    padding = torch.zeros(
                        self.config.n_symbols - prediction_spikes.shape[0],
                        device=self.device,
                        dtype=prediction_spikes.dtype,
                    )
                    prediction_spikes = torch.cat([prediction_spikes, padding])

                spike_counts = self._accumulate_spikes(prediction_spikes)
                error = compute_prediction_error(
                    spike_counts,
                    next_symbol,
                    self.config.n_symbols,
                )
                error_sum += error

            normal_errors.append(error_sum / (len(sequence) - 1))

        # Test 2: Violation patterns
        violation_errors = []
        progress = tqdm(
            range(self.config.n_violation_trials),
            desc="Violation patterns",
            disable=not self.config.verbose,
        )

        for _trial_idx in progress:
            # Reset spike buffer at start of each trial
            self._reset_spike_buffer()

            sequence, targets, _ = self.dataset.generate_sequence(include_violation=True)

            error_sum = 0.0
            for t in range(len(sequence) - 1):
                current_symbol = torch.argmax(sequence[t]).item()
                next_symbol = torch.argmax(targets[t]).item()

                input_spikes = encode_symbol_to_spikes(
                    current_symbol,
                    self.config.n_symbols,
                    self.config.spike_rate_active,
                    self.config.spike_rate_baseline,
                    self.device,
                    thalamus_size=self.brain.thalamus.relay_size,
                )

                # Thalamus receives external sensory input via ascending pathways
                brain_input = {"thalamus": {"sensory": input_spikes}}
                brain_output = self.brain.forward(brain_input)

                if "hippocampus" in brain_output and brain_output["hippocampus"]:
                    prediction_spikes = list(brain_output["hippocampus"].values())[0]
                else:
                    for region_output in brain_output.values():
                        if region_output:
                            prediction_spikes = list(region_output.values())[0]
                            break

                if prediction_spikes.shape[0] > self.config.n_symbols:
                    prediction_spikes = prediction_spikes[: self.config.n_symbols]
                elif prediction_spikes.shape[0] < self.config.n_symbols:
                    padding = torch.zeros(
                        self.config.n_symbols - prediction_spikes.shape[0],
                        device=self.device,
                        dtype=prediction_spikes.dtype,
                    )
                    prediction_spikes = torch.cat([prediction_spikes, padding])

                spike_counts = self._accumulate_spikes(prediction_spikes)
                error = compute_prediction_error(
                    spike_counts,
                    next_symbol,
                    self.config.n_symbols,
                )
                error_sum += error

            violation_errors.append(error_sum / (len(sequence) - 1))

        # Compare errors
        normal_error = np.mean(normal_errors)
        violation_error = np.mean(violation_errors)
        detection_score = (violation_error - normal_error) / (normal_error + 1e-6)

        self.metrics["violation_detection"].append(detection_score)

        if self.config.verbose:
            print("\n✓ Violation detection complete:")
            print(f"   Normal error: {normal_error:.3f}")
            print(f"   Violation error: {violation_error:.3f}")
            print(f"   Detection score: {detection_score:.2%}")
            print("   (Higher = better violation detection)")

    def save_results(self):
        """Save metrics and configuration."""
        # Save configuration
        config_path = self.output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(asdict(self.config), f, indent=2)

        # Save metrics
        metrics_path = self.output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=2)

        # Save brain if requested
        if self.config.save_brain:
            brain_path = self.output_dir / "brain.pt"
            torch.save(self.brain.state_dict(), brain_path)
            if self.config.verbose:
                print(f"\n✓ Brain saved to {brain_path}")

        if self.config.verbose:
            print(f"\n✓ Results saved to {self.output_dir}")

    def print_summary(self):
        """Print final summary."""
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)

        train_acc = self.metrics["train_accuracy"][-1] if self.metrics["train_accuracy"] else 0.0
        test_acc = self.metrics["test_accuracy"][-1] if self.metrics["test_accuracy"] else 0.0
        violation_score = self.metrics["violation_detection"][-1] if self.metrics["violation_detection"] else 0.0

        random_baseline = 1.0 / self.config.n_symbols

        print(f"\n📊 Performance:")
        print(f"   Training accuracy:   {train_acc:.2%} (baseline: {random_baseline:.2%})")
        print(f"   Test accuracy:       {test_acc:.2%}")
        print(f"   Violation detection: {violation_score:.2%}")

        print(f"\n🎯 Goals:")
        print(f"   Basic (>50%):        {'✓' if test_acc > 0.50 else '✗'}")
        print(f"   Intermediate (>70%): {'✓' if test_acc > 0.70 else '✗'}")
        print(f"   Advanced (>85%):     {'✓' if test_acc > 0.85 else '✗'}")

        print(f"\n📁 Results saved to: {self.output_dir}")
        print("=" * 80)

    def _count_parameters(self) -> int:
        """Count total parameters in brain."""
        return sum(p.numel() for p in self.brain.parameters())

    def run(self):
        """Run complete experiment."""
        self.setup()
        self.train()
        self.test()
        self.test_violation_detection()
        self.save_results()
        self.print_summary()


# ============================================================================
# CLI Entry Point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Test default brain on temporal sequence learning"
    )

    # Training parameters
    parser.add_argument(
        "--n-training-trials",
        type=int,
        default=1000,
        help="Number of training trials",
    )
    parser.add_argument(
        "--n-test-trials",
        type=int,
        default=100,
        help="Number of test trials",
    )
    parser.add_argument(
        "--n-symbols",
        type=int,
        default=5,
        help="Number of distinct symbols",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=10,
        help="Length of each sequence",
    )

    # Architecture parameters
    parser.add_argument(
        "--cortex-size",
        type=int,
        default=200,
        help="Cortex size",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/results/sequence_learning",
        help="Output directory",
    )
    parser.add_argument(
        "--save-brain",
        action="store_true",
        help="Save trained brain state",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )

    args = parser.parse_args()

    # Create configuration
    config = TrainingConfig(
        n_training_trials=args.n_training_trials,
        n_test_trials=args.n_test_trials,
        n_symbols=args.n_symbols,
        sequence_length=args.sequence_length,
        cortex_size=args.cortex_size,
        output_dir=args.output_dir,
        save_brain=args.save_brain,
        verbose=not args.quiet,
        device=args.device,
    )

    # Run experiment
    experiment = SequenceLearningExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
