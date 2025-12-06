"""
ThaliaConfig - The unified configuration for the entire THALIA system.

This is the single source of truth for all configuration. It:
1. Contains GlobalConfig for universal parameters
2. Contains module-specific configs that inherit global values
3. Provides validation to catch inconsistencies
4. Offers summary() to show effective values
5. Can create resolved configs for legacy APIs

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List
import json
from pathlib import Path

from .global_config import GlobalConfig
from .brain_config import BrainConfig, RegionSizes
from .language_config import LanguageConfig
from .training_config import TrainingConfig


@dataclass
class ThaliaConfig:
    """Unified configuration for the entire THALIA system.

    This is the main configuration class that users should interact with.
    It contains all sub-configurations and provides methods for:
    - Validation
    - Summary display
    - Serialization
    - Creating legacy config objects

    Example:
        # Create with defaults
        config = ThaliaConfig()

        # Create with customizations
        config = ThaliaConfig(
            global_=GlobalConfig(device="cuda", vocab_size=10000),
            brain=BrainConfig(
                sizes=RegionSizes(cortex_size=256),
            ),
        )

        # Show effective configuration
        config.summary()

        # Validate consistency
        config.validate()

        # Create brain from config
        from thalia.core import EventDrivenBrain
        brain = EventDrivenBrain.from_thalia_config(config)
    """

    # Global parameters (inherited by all modules)
    global_: GlobalConfig = field(default_factory=GlobalConfig)

    # Module-specific configurations
    brain: BrainConfig = field(default_factory=BrainConfig)
    language: LanguageConfig = field(default_factory=LanguageConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        """Validate after initialization."""
        self.validate()

    def validate(self) -> List[str]:
        """Validate configuration consistency.

        Returns:
            List of warning/error messages (empty if valid)
        """
        issues: List[str] = []

        # Check that brain input matches what language encoder will produce
        expected_input = self.brain.sizes.input_size
        if self.language.encoding.embedding_dim != expected_input:
            # This is just informational - embedding_dim is intermediate
            pass

        # Check position encoding size ratio
        pos_neurons = int(self.brain.sizes.input_size * self.language.position.size_ratio)
        if pos_neurons < 16:
            issues.append(
                f"Position encoding only has {pos_neurons} neurons - may be too few"
            )

        # Check that sparsity is reasonable for network size
        if self.global_.default_sparsity * self.brain.sizes.cortex_size < 1:
            issues.append(
                f"With sparsity {self.global_.default_sparsity} and cortex_size "
                f"{self.brain.sizes.cortex_size}, average active neurons < 1"
            )

        # Check theta period vs timesteps
        theta_period_timesteps = self.global_.theta_period_ms / self.global_.dt_ms
        if self.brain.encoding_timesteps > theta_period_timesteps * 2:
            issues.append(
                f"Encoding timesteps ({self.brain.encoding_timesteps}) > 2 theta periods "
                f"({theta_period_timesteps:.1f}) - may have phase ambiguity"
            )

        return issues

    def summary(self, show_all: bool = False) -> str:
        """Return comprehensive summary of configuration.

        Args:
            show_all: If True, show all parameters. If False, show key parameters.

        Returns:
            Formatted string summary
        """
        lines = [
            "╔════════════════════════════════════════════════════════════╗",
            "║              THALIA Configuration Summary                  ║",
            "╚════════════════════════════════════════════════════════════╝",
            "",
            self.global_.summary(),
            "",
            self.brain.summary(),
            "",
            self.language.summary(),
            "",
            self.training.summary(),
        ]

        # Add validation warnings
        issues = self.validate()
        if issues:
            lines.extend([
                "",
                "⚠️  Validation Warnings:",
            ])
            for issue in issues:
                lines.append(f"   - {issue}")

        return "\n".join(lines)

    def print_summary(self, show_all: bool = False) -> None:
        """Print configuration summary."""
        print(self.summary(show_all))

    # =========================================================================
    # LEGACY CONFIG CREATION
    # =========================================================================

    def to_event_driven_brain_config(self) -> Any:
        """Create EventDrivenBrainConfig for legacy API.

        This creates the old-style config object for backward compatibility.
        Deprecation warning is suppressed since this is intentional internal use.
        """
        import warnings

        # Import here to avoid circular imports
        from thalia.core.brain import EventDrivenBrainConfig
        from thalia.core.diagnostics import DiagnosticLevel

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return EventDrivenBrainConfig(
                input_size=self.brain.sizes.input_size,
                cortex_size=self.brain.sizes.cortex_size,
                hippocampus_size=self.brain.sizes.hippocampus_size,
                pfc_size=self.brain.sizes.pfc_size,
                n_actions=self.brain.sizes.n_actions,
                cortex_type=self.brain.cortex_type,
                cortex_config=self.brain.cortex,  # Pass the full cortex config
                dt_ms=self.global_.dt_ms,
                theta_frequency_hz=self.global_.theta_frequency_hz,
                encoding_timesteps=self.brain.encoding_timesteps,
                delay_timesteps=self.brain.delay_timesteps,
                test_timesteps=self.brain.test_timesteps,
                neurons_per_action=self.brain.striatum.neurons_per_action,
                parallel=self.brain.parallel,
                diagnostic_level=DiagnosticLevel.SUMMARY,
                device=self.global_.device,
            )

    def to_language_interface_config(self) -> Any:
        """Create LanguageInterfaceConfig for legacy API."""
        import warnings
        from thalia.language.model import LanguageInterfaceConfig

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return LanguageInterfaceConfig(
                vocab_size=self.global_.vocab_size,
                n_timesteps=self.language.encoding.n_timesteps,
                sparsity=self.language.encoding.get_sparsity(self.global_),
                max_seq_len=self.language.position.max_positions,
                brain_input_size=self.brain.sizes.input_size,
                device=self.global_.device,
            )

    def to_sequence_memory_config(self) -> Any:
        """Create LegacySequenceMemoryConfig for legacy API."""
        import warnings
        from thalia.memory.sequence import LegacySequenceMemoryConfig

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return LegacySequenceMemoryConfig(
                vocab_size=self.global_.vocab_size,
                n_neurons=self.brain.sizes.hippocampus_size * 2,  # Larger for sequence storage
                context_length=self.language.sequence_memory.context_length,
                theta_frequency=self.global_.theta_frequency_hz,
                gamma_frequency=self.global_.gamma_frequency_hz,
                association_strength=self.language.sequence_memory.association_strength,
                retrieval_threshold=self.language.sequence_memory.retrieval_threshold,
                max_stored_contexts=self.language.sequence_memory.max_stored_contexts,
                learning_rate=self.language.sequence_memory.learning_rate,
                device=self.global_.device,
            )

    def to_training_config(self) -> Any:
        """Create LegacyTrainingConfig for legacy API."""
        import warnings
        from thalia.training.local_trainer import LegacyTrainingConfig

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return LegacyTrainingConfig(
                n_epochs=self.training.n_epochs,
                log_every=self.training.logging.log_every,
                save_every=self.training.checkpoint.save_every,
                use_stdp=self.training.use_stdp,
                use_bcm=self.training.use_bcm,
                use_eligibility=self.training.use_eligibility,
                use_hebbian=self.training.use_hebbian,
                stdp_lr=self.training.learning_rates.stdp,
                bcm_lr=self.training.learning_rates.bcm,
                hebbian_lr=self.training.learning_rates.hebbian,
                reward_signal=self.training.reward_scale,
                two_phase_enabled=self.training.two_phase.enabled,
                consolidation_timesteps=self.training.two_phase.consolidation_timesteps,
                decoder_learning_start_step=self.training.decoder_learning_start_step,
                checkpoint_dir=self.training.checkpoint.checkpoint_dir,
                device=self.global_.device,
            )

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "global": self.global_.to_dict(),
            "brain": {
                "sizes": {
                    "input_size": self.brain.sizes.input_size,
                    "cortex_size": self.brain.sizes.cortex_size,
                    "hippocampus_size": self.brain.sizes.hippocampus_size,
                    "pfc_size": self.brain.sizes.pfc_size,
                    "n_actions": self.brain.sizes.n_actions,
                },
                "encoding_timesteps": self.brain.encoding_timesteps,
                "delay_timesteps": self.brain.delay_timesteps,
                "test_timesteps": self.brain.test_timesteps,
                "parallel": self.brain.parallel,
            },
            "language": {
                "encoding": {
                    "n_timesteps": self.language.encoding.n_timesteps,
                    "embedding_dim": self.language.encoding.embedding_dim,
                },
                "position": {
                    "max_positions": self.language.position.max_positions,
                },
                "sequence_memory": {
                    "context_length": self.language.sequence_memory.context_length,
                },
            },
            "training": {
                "n_epochs": self.training.n_epochs,
                "use_stdp": self.training.use_stdp,
                "use_bcm": self.training.use_bcm,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.write_text(self.to_json())

    @classmethod
    def load(cls, path: str | Path) -> "ThaliaConfig":
        """Load configuration from JSON file."""
        path = Path(path)
        data = json.loads(path.read_text())
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ThaliaConfig":
        """Create from dictionary."""
        global_config = GlobalConfig.from_dict(d.get("global", {}))

        brain_data = d.get("brain", {})
        sizes_data = brain_data.get("sizes", {})
        sizes = RegionSizes(**sizes_data)

        brain = BrainConfig(
            sizes=sizes,
            encoding_timesteps=brain_data.get("encoding_timesteps", 15),
            delay_timesteps=brain_data.get("delay_timesteps", 10),
            test_timesteps=brain_data.get("test_timesteps", 15),
            parallel=brain_data.get("parallel", False),
        )

        return cls(
            global_=global_config,
            brain=brain,
        )

    # =========================================================================
    # CONVENIENCE FACTORY METHODS
    # =========================================================================

    @classmethod
    def for_language(
        cls,
        vocab_size: int = 10000,
        device: str = "cpu",
        cortex_size: int = 256,
        hippocampus_size: int = 128,
    ) -> "ThaliaConfig":
        """Create configuration optimized for language processing.

        Args:
            vocab_size: Token vocabulary size
            device: Computation device
            cortex_size: Size of cortex output
            hippocampus_size: Size of hippocampus output

        Returns:
            ThaliaConfig configured for language
        """
        return cls(
            global_=GlobalConfig(
                device=device,
                vocab_size=vocab_size,
                theta_frequency_hz=6.0,  # Slower theta for longer sequences
            ),
            brain=BrainConfig(
                sizes=RegionSizes(
                    input_size=cortex_size,  # Match cortex for language
                    cortex_size=cortex_size,
                    hippocampus_size=hippocampus_size,
                    pfc_size=64,
                    n_actions=vocab_size,  # For next-token prediction
                ),
            ),
        )

    @classmethod
    def for_classification(
        cls,
        n_classes: int,
        input_size: int = 256,
        device: str = "cpu",
    ) -> "ThaliaConfig":
        """Create configuration for classification tasks.

        Args:
            n_classes: Number of output classes
            input_size: Size of input (e.g., from image encoder)
            device: Computation device

        Returns:
            ThaliaConfig configured for classification
        """
        return cls(
            global_=GlobalConfig(
                device=device,
                theta_frequency_hz=8.0,
            ),
            brain=BrainConfig(
                sizes=RegionSizes(
                    input_size=input_size,
                    cortex_size=128,
                    hippocampus_size=64,
                    pfc_size=32,
                    n_actions=n_classes,
                ),
            ),
        )

    @classmethod
    def minimal(cls, device: str = "cpu") -> "ThaliaConfig":
        """Create minimal configuration for testing.

        Small network sizes for fast testing.
        """
        return cls(
            global_=GlobalConfig(device=device),
            brain=BrainConfig(
                sizes=RegionSizes(
                    input_size=64,
                    cortex_size=32,
                    hippocampus_size=16,
                    pfc_size=8,
                    n_actions=2,
                ),
                encoding_timesteps=5,
                delay_timesteps=3,
                test_timesteps=5,
            ),
            training=TrainingConfig(n_epochs=1),
        )
