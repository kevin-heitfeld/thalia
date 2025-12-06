"""
Training Configuration - Settings for local learning and training loops.

This module defines configuration for training without backpropagation,
using local learning rules (STDP, BCM, three-factor, etc.).

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .global_config import GlobalConfig


@dataclass
class LearningRatesConfig:
    """Unified learning rate configuration.

    Having all learning rates in one place makes it easy to:
    1. See the relative scaling of different rules
    2. Apply global learning rate multipliers
    3. Tune learning without hunting through configs
    """

    # STDP (spike-timing dependent plasticity)
    stdp: float = 0.01
    stdp_tau_plus: float = 20.0  # ms
    stdp_tau_minus: float = 20.0  # ms

    # BCM (sliding threshold)
    bcm: float = 0.001
    bcm_tau_theta: float = 5000.0  # ms

    # Three-factor (eligibility × dopamine)
    three_factor: float = 0.005
    eligibility_tau: float = 1000.0  # ms

    # Hebbian (simple correlation)
    hebbian: float = 0.01

    # Hippocampal (fast one-shot)
    hippocampal: float = 0.2

    # Global multiplier
    global_scale: float = 1.0

    def get_scaled(self, base_lr: float) -> float:
        """Apply global scale to a learning rate."""
        return base_lr * self.global_scale


@dataclass
class CheckpointConfig:
    """Configuration for model checkpointing."""

    enabled: bool = True
    save_every: int = 1000  # steps
    checkpoint_dir: Optional[str] = None  # If None, use default
    keep_last_n: int = 5  # Number of checkpoints to keep


@dataclass
class LoggingConfig:
    """Configuration for training logging."""

    log_every: int = 100  # steps
    log_weights: bool = False
    log_spikes: bool = True
    log_gradients: bool = False  # N/A for local learning, but kept for API
    tensorboard: bool = False
    wandb: bool = False


@dataclass
class TwoPhaseConfig:
    """Configuration for two-phase training (stimulus → reward/consolidation).
    
    Two-phase training separates:
    1. Stimulus Phase: Present input, process through brain, build eligibility traces
    2. Reward Phase: Deliver reward, run consolidation timesteps for eligibility-dopamine interaction
    
    This mimics biological temporal credit assignment where:
    - Eligibility traces mark "what just happened" (τ ~ 100-1000ms)
    - Phasic dopamine decays slowly (τ ~ 200ms) 
    - Learning occurs where traces and dopamine overlap
    """
    
    enabled: bool = True
    
    # Number of timesteps between action and reward delivery
    # Simulates delay between action and outcome
    reward_delay_timesteps: int = 10
    
    # Number of timesteps after reward to allow eligibility-dopamine interaction
    # This is the "consolidation window" where learning actually happens
    consolidation_timesteps: int = 50
    
    # Whether to clear eligibility traces at trial end
    # If False, traces persist across trials (longer temporal credit)
    clear_traces_at_trial_end: bool = True


@dataclass
class TrainingConfig:
    """Complete training configuration.

    This defines how training proceeds - epochs, batching,
    learning rules, and logging.
    """

    # Training duration
    n_epochs: int = 10
    steps_per_epoch: Optional[int] = None  # If None, full dataset

    # Learning rules to enable
    use_stdp: bool = True
    use_bcm: bool = True
    use_eligibility: bool = True
    use_hebbian: bool = True

    # Learning rates
    learning_rates: LearningRatesConfig = field(default_factory=LearningRatesConfig)

    # Neuromodulation
    base_dopamine: float = 0.0  # Tonic dopamine level
    reward_scale: float = 1.0  # Scale factor for reward signals

    # Two-phase training (stimulus → reward/consolidation)
    two_phase: TwoPhaseConfig = field(default_factory=TwoPhaseConfig)

    # Checkpointing and logging
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Early stopping
    early_stopping: bool = False
    patience: int = 10
    min_improvement: float = 0.001

    def summary(self) -> str:
        """Return formatted summary of training configuration."""
        lines = [
            "=== Training Configuration ===",
            f"  Epochs: {self.n_epochs}",
            f"  Steps/epoch: {self.steps_per_epoch or 'full dataset'}",
            "",
            "--- Learning Rules ---",
            f"  STDP: {self.use_stdp} (lr={self.learning_rates.stdp})",
            f"  BCM: {self.use_bcm} (lr={self.learning_rates.bcm})",
            f"  Eligibility: {self.use_eligibility} (lr={self.learning_rates.three_factor})",
            f"  Hebbian: {self.use_hebbian} (lr={self.learning_rates.hebbian})",
            f"  Global scale: {self.learning_rates.global_scale}x",
            "",
            "--- Logging ---",
            f"  Log every: {self.logging.log_every} steps",
            f"  Checkpoint every: {self.checkpoint.save_every} steps",
        ]
        return "\n".join(lines)
