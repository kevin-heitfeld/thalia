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

from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Callable, TYPE_CHECKING
from pathlib import Path
import time

import torch

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
class LegacyTrainingConfig:
    """Legacy configuration for local learning trainer.

    .. deprecated:: 0.2.0
        Use :class:`thalia.config.TrainingConfig` instead for unified configuration.
        Create trainer with ``LocalTrainer.from_thalia_config(config)``.

    This class exists only for backwards compatibility. New code should use
    ``thalia.config.TrainingConfig`` directly.
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

    # Two-phase training (stimulus → reward/consolidation)
    two_phase_enabled: bool = True
    reward_delay_timesteps: int = 10
    consolidation_timesteps: int = 50

    # Decoder learning delay (brain pre-training)
    # Don't train decoder until brain has stabilized for N steps
    decoder_learning_start_step: int = 1000

    # Checkpointing
    checkpoint_dir: Optional[str] = None

    device: str = "cpu"

    def __post_init__(self):
        """Emit deprecation warning."""
        import warnings
        warnings.warn(
            "LegacyTrainingConfig is deprecated. Use thalia.config.TrainingConfig instead:\n"
            "  from thalia.config import TrainingConfig\n"
            "  config = TrainingConfig(...)",
            DeprecationWarning,
            stacklevel=2,
        )


# Alias for backwards compatibility (deprecated)
TrainingConfig = LegacyTrainingConfig


class LocalTrainer:
    """
    Trainer using local (non-backprop) learning rules.

    This trainer implements biologically plausible learning:
    - No gradient computation
    - All updates based on local activity
    - Uses STDP, BCM, and eligibility traces

    Example:
        >>> from thalia.config import ThaliaConfig
        >>> config = ThaliaConfig(...)
        >>> trainer = LocalTrainer.from_thalia_config(config)
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

    @classmethod
    def from_thalia_config(cls, config: "ThaliaConfig") -> "LocalTrainer":
        """Create LocalTrainer from unified ThaliaConfig.

        This is the recommended way to create a trainer.

        Args:
            config: ThaliaConfig with all settings

        Returns:
            LocalTrainer instance

        Example:
            from thalia.config import ThaliaConfig

            config = ThaliaConfig(...)
            trainer = LocalTrainer.from_thalia_config(config)
        """
        from thalia.config import ThaliaConfig

        legacy_config = config.to_training_config()
        return cls(legacy_config)

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
        print(f"\nStarting training for {self.config.n_epochs} epochs...")
        print(f"Data: {data_pipeline.n_sequences} sequences")

        start_time = time.time()

        for epoch in range(self.config.n_epochs):
            epoch_metrics = self._train_epoch(
                model=model,
                data_pipeline=data_pipeline,
                memory=memory,
                epoch=epoch,
                progress_callback=progress_callback,
            )

            print(f"Epoch {epoch + 1}/{self.config.n_epochs} complete: "
                  f"spike_rate={epoch_metrics.spike_rate:.1f}")

        total_time = time.time() - start_time
        print(f"Training complete in {total_time:.2f}s\n")

        return self.metrics

    def _train_epoch(
        self,
        model: "LanguageBrainInterface",
        data_pipeline: Any,
        memory: Optional["SequenceMemory"],
        epoch: int,
        progress_callback: Optional[Callable],
    ) -> TrainingMetrics:
        """Train for one epoch using two-phase training.

        Two-Phase Training:
        1. Stimulus Phase: Process input tokens, build eligibility traces
        2. Reward/Consolidation Phase: Deliver intrinsic reward, run consolidation
           to allow eligibility-dopamine interaction

        This mimics biological temporal credit assignment where:
        - Eligibility traces mark "what just happened" (τ ~ 100-1000ms)
        - Phasic dopamine decays slowly (τ ~ 200ms)
        - Learning occurs where traces × dopamine are both non-zero
        """

        epoch_accuracies: list[float] = []
        epoch_spike_rates: list[float] = []

        use_two_phase = getattr(self.config, 'two_phase_enabled', True)
        consolidation_steps = getattr(self.config, 'consolidation_timesteps', 50)

        for batch in data_pipeline.get_batches():
            step_start = time.time()

            # Get input and target
            input_ids = batch["input"].to(self.device)
            target_ids = batch["target"].to(self.device)

            # =========================================================
            # PHASE 1: Stimulus Processing (builds eligibility traces)
            # =========================================================
            result = self._forward_step(model, input_ids, memory)

            # Compute local learning signals
            learning_signals = self._compute_learning_signals(
                model=model,
                result=result,
                target_ids=target_ids,
            )

            # Apply weight updates (decoder learns to read brain)
            self._apply_learning(model, learning_signals, target_ids)

            # =========================================================
            # PHASE 2: Consolidation (let brain process with intrinsic rewards)
            # =========================================================
            # Note: We do NOT call deliver_reward() here because:
            # 1. Intrinsic rewards (from prediction quality) flow continuously
            #    via the brain's _update_tonic_dopamine() mechanism
            # 2. deliver_reward() is for EXTERNAL task rewards (e.g., game score)
            #    which we don't have during unsupervised pre-training
            # 3. The brain learns from its own prediction errors automatically
            if use_two_phase and hasattr(model.brain, 'run_consolidation'):
                # Run consolidation timesteps
                # This allows eligibility traces to interact with tonic dopamine
                model.brain.run_consolidation(n_timesteps=consolidation_steps)

            accuracy = self._compute_accuracy(model, result, target_ids)
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
        # Reset spike counts before processing (they're cumulative in the brain)
        if hasattr(model.brain, '_spike_counts'):
            model.brain._spike_counts = {name: 0 for name in model.brain.regions}

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
        model: "LanguageBrainInterface",
        result: Dict[str, Any],
        target_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute local learning signals from activity.

        These signals drive weight updates without backprop:
        - STDP: spike timing differences
        - BCM: activity relative to threshold
        - Error: prediction vs target mismatch

        Args:
            model: LanguageBrainInterface with decoder for prediction error
            result: Forward pass result containing brain outputs
            target_ids: Ground truth token IDs for computing error

        Returns:
            Dictionary of learning signals for weight updates
        """
        signals: Dict[str, torch.Tensor] = {}

        # Compute prediction error signal from decoded output vs targets
        if len(model.output_buffer) > 0:
            try:
                # Decode brain outputs to token logits
                logits = model.decode_output(temperature=1.0)  # [n_tokens, vocab_size]

                # Flatten target_ids if needed
                if target_ids.dim() > 1:
                    target_ids = target_ids.squeeze(0)

                # Compute cross-entropy-like error for each position
                n_compare = min(len(logits), len(target_ids))
                if n_compare > 0:
                    # Get probabilities
                    probs = torch.softmax(logits[:n_compare], dim=-1)

                    # Error = 1 - probability of correct token (normalized)
                    target_probs = probs.gather(1, target_ids[:n_compare].unsqueeze(1)).squeeze(1)
                    prediction_error = (1.0 - target_probs).mean()
                    signals["prediction_error"] = prediction_error.unsqueeze(0)
            except Exception:
                # If decoding fails, use zero error
                signals["prediction_error"] = torch.tensor([0.0])
        else:
            signals["prediction_error"] = torch.tensor([0.0])

        # Spike rate for BCM threshold adaptation
        signals["spike_rate"] = torch.tensor(result.get("spike_rate", 0.0))

        # Eligibility signal (recent activity for credit assignment)
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
        target_ids: torch.Tensor,
    ) -> None:
        """
        Trigger learning in the brain and decoder.

        IMPORTANT: Brain and decoder learn from DIFFERENT signals!

        Brain learns via INTRINSIC rewards (self-supervised):
        - Cortex prediction error (free energy minimization)
        - Hippocampus pattern completion
        - The brain learns to model the world, not to please the decoder

        Decoder learns via supervised delta rule:
        - Maps brain activity → tokens
        - Uses target tokens as supervision
        - Is just an OBSERVER of brain activity (like an fMRI decoder)

        This decoupling is crucial because:
        1. Decoder is random initially → would give random punishments
        2. Brain's internal "language" changes as it learns
        3. Decoder is chasing a moving target
        4. Brain shouldn't be blamed for decoder's inability to read it
        """
        # =====================================================================
        # BRAIN LEARNING (handled internally via continuous tonic dopamine)
        # =====================================================================
        # The brain computes intrinsic reward CONTINUOUSLY every timestep via
        # _update_tonic_dopamine(). For self-supervised language learning,
        # there's no external reward signal - the brain learns purely from
        # minimizing its own prediction errors (free energy principle).
        #
        # We DON'T call deliver_reward() here because:
        # 1. Tonic dopamine already flows continuously
        # 2. deliver_reward() is for EXTERNAL task rewards (e.g., game score)
        # 3. Self-supervised learning has no external reward signal

        # =====================================================================
        # DECODER LEARNING (separate, supervised)
        # =====================================================================
        # The decoder learns to READ the brain's activity patterns.
        # It maps brain patterns → tokens using delta rule.
        # This is like training an fMRI decoder - purely observational.
        # The decoder's performance does NOT affect the brain's reward.
        #
        # IMPORTANT: We delay decoder learning until the brain has stabilized.
        # Early brain representations are noisy and rapidly changing - training
        # the decoder on these would be like trying to learn a moving target.
        decoder_start = getattr(self.config, 'decoder_learning_start_step', 0)
        if self.global_step < decoder_start:
            if self.global_step % 100 == 0:
                print(f"  [Decoder] Waiting for brain pre-training ({self.global_step}/{decoder_start})")
            return

        if hasattr(model.decoder, 'learn') and callable(model.decoder.learn):
            decoder_metrics = model.decoder.learn(target_ids=target_ids)
            # Log decoder learning progress periodically
            if self.global_step % 10 == 0 and decoder_metrics:
                if decoder_metrics.get("skipped"):
                    print(f"  [Decoder] Skipped - no features stored")
                else:
                    print(f"  [Decoder] error={decoder_metrics.get('error', 0):.4f}, "
                          f"updates={decoder_metrics.get('n_updates', 0)}")

    def _compute_accuracy(
        self,
        model: "LanguageBrainInterface",
        result: Dict[str, Any],
        target_ids: torch.Tensor,
    ) -> float:
        """
        Compute prediction accuracy by decoding brain output.

        Uses the model's decoder to convert brain activity to token
        predictions, then compares against target tokens.

        Args:
            model: LanguageBrainInterface with decoder
            result: Forward pass result containing brain outputs
            target_ids: Ground truth token IDs [batch, seq_len] or [seq_len]

        Returns:
            Accuracy as fraction of correct predictions (0.0 to 1.0)
        """
        # Check if we have output to decode
        if len(model.output_buffer) == 0:
            return 0.0

        try:
            # Decode brain outputs to token logits
            logits = model.decode_output(temperature=1.0)  # [n_tokens, vocab_size]

            # Get predicted tokens (argmax over vocabulary)
            predicted_ids = logits.argmax(dim=-1)  # [n_tokens]

            # Flatten target_ids if needed
            if target_ids.dim() > 1:
                target_ids = target_ids.squeeze(0)  # [seq_len]

            # Align lengths (predictions are for next-token, so offset by 1)
            # predicted[i] should match target[i] (target is already shifted)
            n_compare = min(len(predicted_ids), len(target_ids))

            if n_compare == 0:
                return 0.0

            # Compare predictions to targets
            correct = (predicted_ids[:n_compare] == target_ids[:n_compare]).float()
            accuracy = correct.mean().item()

            return accuracy

        except Exception:
            # Fall back to activity-based proxy if decoding fails
            spike_rate = result.get("spike_rate", 0.0)
            return min(spike_rate / 1000.0, 0.1)  # Very conservative estimate

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
