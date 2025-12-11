"""
Training Configuration - Settings for training procedures.

This module defines configuration for training-specific parameters
that don't belong in brain or region configs (learning toggles, epochs, etc).

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from .base import BaseConfig


@dataclass
class TrainingConfig(BaseConfig):
    """Training-specific configuration parameters.

    These parameters control the training procedure, not the brain architecture.
    They affect how learning happens during training, but not the brain structure.

    Example:
        config = TrainingConfig(
            n_epochs=100,
            use_stdp=True,
            use_bcm=True,
            batch_size=32,
        )
    """

    # =========================================================================
    # LEARNING TOGGLES
    # =========================================================================
    use_stdp: bool = True
    """Enable Spike-Timing Dependent Plasticity across regions."""

    use_bcm: bool = True
    """Enable Bienenstock-Cooper-Munro learning (sliding threshold)."""

    use_hebbian: bool = True
    """Enable basic Hebbian learning where applicable."""

    use_homeostasis: bool = True
    """Enable homeostatic plasticity (intrinsic excitability, synaptic scaling)."""

    # =========================================================================
    # TRAINING PARAMETERS
    # =========================================================================
    n_epochs: int = 10
    """Number of training epochs."""

    batch_size: int = 1
    """Batch size for training. Note: Current implementation processes one at a time."""

    learning_rate_scale: float = 1.0
    """Global scaling factor for all learning rates.
    
    Useful for quick experiments:
    - 0.5 = half speed learning
    - 2.0 = double speed learning
    """

    # =========================================================================
    # VALIDATION & MONITORING
    # =========================================================================
    validate_every_n_epochs: int = 1
    """Run validation every N epochs."""

    log_diagnostics: bool = True
    """Enable diagnostic logging during training."""

    checkpoint_every_n_epochs: int = 10
    """Save checkpoint every N epochs. 0 = no checkpoints."""

    # =========================================================================
    # CURRICULUM LEARNING
    # =========================================================================
    use_curriculum: bool = False
    """Enable curriculum learning (start easy, increase difficulty)."""

    curriculum_start_difficulty: float = 0.3
    """Starting difficulty level for curriculum (0.0 = easiest, 1.0 = hardest)."""

    curriculum_end_difficulty: float = 1.0
    """Final difficulty level for curriculum."""

    def summary(self) -> str:
        """Return formatted summary of training configuration."""
        lines = [
            "=== Training Configuration ===",
            "--- Learning Rules ---",
            f"  STDP: {self.use_stdp}",
            f"  BCM: {self.use_bcm}",
            f"  Hebbian: {self.use_hebbian}",
            f"  Homeostasis: {self.use_homeostasis}",
            "",
            "--- Training Parameters ---",
            f"  Epochs: {self.n_epochs}",
            f"  Batch size: {self.batch_size}",
            f"  LR scale: {self.learning_rate_scale}",
            "",
            "--- Monitoring ---",
            f"  Validate every: {self.validate_every_n_epochs} epochs",
            f"  Log diagnostics: {self.log_diagnostics}",
            f"  Checkpoint every: {self.checkpoint_every_n_epochs} epochs",
            "",
            "--- Curriculum ---",
            f"  Use curriculum: {self.use_curriculum}",
        ]
        if self.use_curriculum:
            lines.extend([
                f"  Start difficulty: {self.curriculum_start_difficulty}",
                f"  End difficulty: {self.curriculum_end_difficulty}",
            ])
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "use_stdp": self.use_stdp,
            "use_bcm": self.use_bcm,
            "use_hebbian": self.use_hebbian,
            "use_homeostasis": self.use_homeostasis,
            "n_epochs": self.n_epochs,
            "batch_size": self.batch_size,
            "learning_rate_scale": self.learning_rate_scale,
            "validate_every_n_epochs": self.validate_every_n_epochs,
            "log_diagnostics": self.log_diagnostics,
            "checkpoint_every_n_epochs": self.checkpoint_every_n_epochs,
            "use_curriculum": self.use_curriculum,
            "curriculum_start_difficulty": self.curriculum_start_difficulty,
            "curriculum_end_difficulty": self.curriculum_end_difficulty,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        """Create from dictionary."""
        return cls(
            use_stdp=d.get("use_stdp", True),
            use_bcm=d.get("use_bcm", True),
            use_hebbian=d.get("use_hebbian", True),
            use_homeostasis=d.get("use_homeostasis", True),
            n_epochs=d.get("n_epochs", 10),
            batch_size=d.get("batch_size", 1),
            learning_rate_scale=d.get("learning_rate_scale", 1.0),
            validate_every_n_epochs=d.get("validate_every_n_epochs", 1),
            log_diagnostics=d.get("log_diagnostics", True),
            checkpoint_every_n_epochs=d.get("checkpoint_every_n_epochs", 10),
            use_curriculum=d.get("use_curriculum", False),
            curriculum_start_difficulty=d.get("curriculum_start_difficulty", 0.3),
            curriculum_end_difficulty=d.get("curriculum_end_difficulty", 1.0),
        )
