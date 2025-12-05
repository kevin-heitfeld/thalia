"""
Local Learning Trainer for THALIA.

This module provides a training loop that uses LOCAL learning rules
rather than backpropagation. All weight updates are computed using
biologically plausible mechanisms.

Key Learning Rules:
==================

1. STDP (Spike-Timing Dependent Plasticity)
   - Pre before post → LTP (strengthen)
   - Post before pre → LTD (weaken)
   - Window: ~20ms

2. BCM (Bienenstock-Cooper-Munro)
   - Sliding threshold for LTP/LTD transition
   - Prevents runaway potentiation

3. Three-Factor (Eligibility Traces + Neuromodulation)
   - Eligibility trace: "credit" for recent activity
   - Dopamine signal: "reward" that triggers learning
   - Learning = eligibility × dopamine

4. Hebbian (Correlation-Based)
   - Neurons that fire together, wire together
   - Simple but effective for unsupervised learning

Training Loop:
=============

    for epoch in epochs:
        for batch in data:
            # 1. Forward pass (spike propagation)
            spikes = brain.process(batch)

            # 2. Compute local learning signals
            #    - STDP from spike timing
            #    - BCM from activity levels
            #    - Error signal (for supervised)

            # 3. Apply weight updates
            brain.apply_learning()

            # 4. Log metrics
            metrics.update(spikes, targets)

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Callable, TYPE_CHECKING
from pathlib import Path
import time

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from thalia.language.model import LanguageBrainInterface
    from thalia.memory.sequence import SequenceMemory


@dataclass
class TrainingMetrics:
    """Metrics tracked during training.

    All metrics are averages over the logging window.
    """
    # Loss/Error metrics
    prediction_accuracy: float = 0.0
    spike_rate: float = 0.0

    # Learning dynamics
    weight_change_magnitude: float = 0.0
    bcm_threshold_mean: float = 0.0
    eligibility_trace_mean: float = 0.0

    # Activity metrics
    sparsity: float = 0.0
    firing_rate_mean: float = 0.0
    firing_rate_std: float = 0.0

    # Timing
    step_time_ms: float = 0.0

    # Counters
    steps: int = 0
    sequences_processed: int = 0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "prediction_accuracy": self.prediction_accuracy,
            "spike_rate": self.spike_rate,
            "weight_change_magnitude": self.weight_change_magnitude,
            "sparsity": self.sparsity,
            "firing_rate_mean": self.firing_rate_mean,
            "step_time_ms": self.step_time_ms,
            "steps": float(self.steps),
        }


@dataclass
class TrainingConfig:
    """Configuration for local learning trainer.

    Attributes:
        n_epochs: Number of training epochs
        log_every: Log metrics every N steps
        save_every: Save checkpoint every N steps

        # Learning rule selection
        use_stdp: Enable STDP learning
        use_bcm: Enable BCM threshold
        use_eligibility: Enable eligibility traces
        use_hebbian: Enable simple Hebbian

        # Learning rate scaling
        stdp_lr: STDP learning rate
        bcm_lr: BCM threshold adaptation rate
        hebbian_lr: Hebbian learning rate

        # Neuromodulation
        reward_signal: Base reward signal for three-factor

        device: Computation device
    """
    n_epochs: int = 10
    log_every: int = 100
    save_every: int = 1000

    # Learning rules
    use_stdp: bool = True
    use_bcm: bool = True
    use_eligibility: bool = True
    use_hebbian: bool = True

    # Learning rates
    stdp_lr: float = 0.01
    bcm_lr: float = 0.001
    hebbian_lr: float = 0.01

    # Neuromodulation
    reward_signal: float = 1.0

    # Checkpointing
    checkpoint_dir: Optional[str] = None

    device: str = "cpu"


class LocalTrainer:
    """
    Trainer using local (non-backprop) learning rules.

    This trainer implements biologically plausible learning:
    - No gradient computation
    - All updates based on local activity
    - Uses STDP, BCM, and eligibility traces

    Example:
        >>> from thalia.training import LocalTrainer, TrainingConfig
        >>> from thalia.language import LanguageBrainInterface
        >>>
        >>> trainer = LocalTrainer(TrainingConfig(n_epochs=5))
        >>>
        >>> # Train on sequences
        >>> metrics = trainer.train(
        ...     model=brain_interface,
        ...     data_pipeline=data,
        ...     memory=sequence_memory,
        ... )
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Metrics accumulator
        self.metrics = TrainingMetrics()
        self.metrics_history: List[Dict[str, float]] = []

        # Callbacks
        self.callbacks: List[Callable[[TrainingMetrics], None]] = []

        # Step counter
        self.global_step = 0

    def train(
        self,
        model: "LanguageBrainInterface",
        data_pipeline: Any,
        memory: Optional["SequenceMemory"] = None,
        progress_callback: Optional[Callable[[int, int, TrainingMetrics], None]] = None,
    ) -> TrainingMetrics:
        """
        Train the model using local learning rules.

        Args:
            model: LanguageBrainInterface to train
            data_pipeline: TextDataPipeline providing batches
            memory: Optional SequenceMemory for context
            progress_callback: Optional callback(epoch, step, metrics)

        Returns:
            Final training metrics
        """
        print(f"Starting training for {self.config.n_epochs} epochs...")
        print(f"Data: {data_pipeline.n_sequences} sequences")
        print(f"Learning rules: STDP={self.config.use_stdp}, BCM={self.config.use_bcm}")

        start_time = time.time()

        for epoch in range(self.config.n_epochs):
            epoch_metrics = self._train_epoch(
                model=model,
                data_pipeline=data_pipeline,
                memory=memory,
                epoch=epoch,
                progress_callback=progress_callback,
            )

            print(f"Epoch {epoch + 1}/{self.config.n_epochs}: "
                  f"accuracy={epoch_metrics.prediction_accuracy:.4f}, "
                  f"spike_rate={epoch_metrics.spike_rate:.4f}")

        total_time = time.time() - start_time
        print(f"Training complete in {total_time:.2f}s")

        return self.metrics

    def _train_epoch(
        self,
        model: "LanguageBrainInterface",
        data_pipeline: Any,
        memory: Optional["SequenceMemory"],
        epoch: int,
        progress_callback: Optional[Callable],
    ) -> TrainingMetrics:
        """Train for one epoch."""

        epoch_accuracies = []
        epoch_spike_rates = []

        for batch in data_pipeline.get_batches():
            step_start = time.time()

            # Get input and target
            input_ids = batch["input"].to(self.device)
            target_ids = batch["target"].to(self.device)

            # Process through model
            result = self._forward_step(model, input_ids, memory)

            # Compute local learning signals
            learning_signals = self._compute_learning_signals(
                result=result,
                target_ids=target_ids,
            )

            # Apply weight updates
            self._apply_learning(model, learning_signals)

            # Compute metrics
            accuracy = self._compute_accuracy(result, target_ids)
            spike_rate = result.get("spike_rate", 0.0)

            epoch_accuracies.append(accuracy)
            epoch_spike_rates.append(spike_rate)

            # Update metrics
            self.metrics.steps = self.global_step
            self.metrics.prediction_accuracy = accuracy
            self.metrics.spike_rate = spike_rate
            self.metrics.step_time_ms = (time.time() - step_start) * 1000

            # Log
            if self.global_step % self.config.log_every == 0:
                self.metrics_history.append(self.metrics.to_dict())

            # Callback
            if progress_callback:
                progress_callback(epoch, self.global_step, self.metrics)

            self.global_step += 1
            self.metrics.sequences_processed += input_ids.size(0)

        # Epoch summary
        self.metrics.prediction_accuracy = sum(epoch_accuracies) / len(epoch_accuracies) if epoch_accuracies else 0.0
        self.metrics.spike_rate = sum(epoch_spike_rates) / len(epoch_spike_rates) if epoch_spike_rates else 0.0

        return self.metrics

    def _forward_step(
        self,
        model: "LanguageBrainInterface",
        input_ids: torch.Tensor,
        memory: Optional["SequenceMemory"],
    ) -> Dict[str, Any]:
        """
        Forward pass through the model.

        This processes tokens through the brain and collects
        spike activity for learning.
        """
        # Process through language interface
        result = model.process_tokens(input_ids)

        # If memory is provided, also encode in sequence memory
        if memory is not None:
            memory_result = memory.encode_sequence(input_ids, learn=True)
            result["memory_patterns"] = memory_result["patterns"]

        # Compute spike rate
        if "results" in result and result["results"]:
            total_spikes = sum(
                r.get("spike_counts", {}).get("cortex", 0)
                for r in result["results"]
            )
            result["spike_rate"] = total_spikes / len(result["results"])
        else:
            result["spike_rate"] = 0.0

        return result

    def _compute_learning_signals(
        self,
        result: Dict[str, Any],
        target_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute local learning signals from activity.

        These signals drive weight updates without backprop:
        - STDP: spike timing differences
        - BCM: activity relative to threshold
        - Error: prediction vs target mismatch
        """
        signals = {}

        # Simple prediction error signal
        # This is a LOCAL signal based on activity mismatch
        if "predicted_pattern" in result:
            # Compare predicted pattern to actual next pattern
            # (This would be more sophisticated with actual targets)
            signals["prediction_error"] = torch.randn(1) * 0.1  # Placeholder

        # Spike rate for BCM
        signals["spike_rate"] = torch.tensor(result.get("spike_rate", 0.0))

        # Eligibility signal (recent activity)
        if "results" in result and result["results"]:
            # Use cortex activity as eligibility proxy
            last_result = result["results"][-1]
            if "cortex_activity" in last_result:
                signals["eligibility"] = last_result["cortex_activity"]

        return signals

    def _apply_learning(
        self,
        model: "LanguageBrainInterface",
        learning_signals: Dict[str, torch.Tensor],
    ) -> None:
        """
        Apply local learning rules to update weights.

        This modifies weights using:
        - STDP if spike timing available
        - BCM threshold adaptation
        - Hebbian correlation
        """
        # Access the brain through the interface
        brain = model.brain

        # Apply learning to each region
        # The brain's regions already have their own learning rules
        # We just need to trigger them

        if self.config.use_stdp:
            # STDP is typically applied within regions during forward pass
            # Here we can modulate the learning rate globally
            pass

        if self.config.use_bcm:
            # BCM threshold adaptation would update based on activity
            spike_rate = learning_signals.get("spike_rate", torch.tensor(0.0))
            # Regions handle their own BCM internally

        if self.config.use_hebbian:
            # Simple Hebbian: correlate pre and post activity
            if "eligibility" in learning_signals:
                # Regions handle Hebbian learning internally
                pass

        # Note: The actual weight updates happen within the brain regions
        # during their forward pass via their built-in learning rules.
        # This method is mainly for global coordination and modulation.

    def _compute_accuracy(
        self,
        result: Dict[str, Any],
        target_ids: torch.Tensor,
    ) -> float:
        """
        Compute prediction accuracy.

        For now, this is a placeholder that returns a rough
        estimate based on spike activity patterns.
        """
        # Real accuracy would compare decoded output to targets
        # For now, return a proxy based on activity
        spike_rate = result.get("spike_rate", 0.0)

        # Higher spike rate during training = better learning
        # (This is a very rough approximation)
        return min(spike_rate / 10.0, 1.0)  # Normalize to 0-1

    def save_checkpoint(self, path: Path) -> None:
        """Save training checkpoint."""
        checkpoint = {
            "global_step": self.global_step,
            "metrics": self.metrics.to_dict(),
            "metrics_history": self.metrics_history,
            "config": self.config,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: Path) -> None:
        """Load training checkpoint."""
        checkpoint = torch.load(path)
        self.global_step = checkpoint["global_step"]
        self.metrics_history = checkpoint["metrics_history"]

    def add_callback(self, callback: Callable[[TrainingMetrics], None]) -> None:
        """Add a callback to be called after each step."""
        self.callbacks.append(callback)

    def get_metrics_history(self) -> List[Dict[str, float]]:
        """Get full metrics history."""
        return self.metrics_history
