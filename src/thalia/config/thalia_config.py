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
from typing import Dict, Any, List, Optional
import json
from pathlib import Path

from .global_config import GlobalConfig
from .brain_config import BrainConfig, RegionSizes
from .language_config import LanguageConfig
from .training_config import TrainingConfig
from .robustness_config import RobustnessConfig


def print_config(
    config: "ThaliaConfig",
    title: str = "CONFIGURATION",
    extra: Optional[Dict[str, Any]] = None,
    width: int = 70,
) -> None:
    """Print a comprehensive, formatted configuration summary.

    This is the main function for printing configuration in experiments.
    It shows all important parameters in a clear, organized format.

    Args:
        config: ThaliaConfig to print
        title: Title for the configuration block
        extra: Additional key-value pairs to print (experiment-specific)
        width: Width of the output box

    Example:
        from thalia.config import ThaliaConfig, print_config

        config = ThaliaConfig(...)
        print_config(config, title="EXPERIMENT CONFIG", extra={
            "experiment_name": "exp01",
            "run_id": 42,
        })
    """
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)

    # Extra experiment-specific info first
    if extra:
        print("\n--- EXPERIMENT ---")
        for key, value in extra.items():
            print(f"  {key}: {value}")

    # Global
    print("\n--- GLOBAL ---")
    print(f"  Device: {config.global_.device}")
    print(f"  Vocab size: {config.global_.vocab_size}")
    print(f"  Default sparsity: {config.global_.default_sparsity}")

    # Global learning toggles (from training config)
    print("\n--- TRAINING ---")
    print(f"  STDP enabled: {config.training.use_stdp}")
    print(f"  BCM enabled: {config.training.use_bcm}")
    print(f"  Hebbian enabled: {config.training.use_hebbian}")
    print(f"  N epochs: {config.training.n_epochs}")

    # Language encoding/decoding
    print("\n--- LANGUAGE ---")
    enc_cfg = config.language.encoding
    dec_cfg = config.language.decoding
    print(f"  Encoding type: {enc_cfg.encoding_type.value}")
    print(f"  Decoding type: {dec_cfg.decoding_type.value}")
    print(f"  Embedding dim: {enc_cfg.embedding_dim}")
    print(f"  Encoding timesteps: {enc_cfg.n_timesteps}")
    print(f"  Learnable embedding: {enc_cfg.learnable_embedding}")
    if enc_cfg.sparsity is not None:
        print(f"  SDR sparsity: {enc_cfg.sparsity}")
    print(f"  SDR overlap: {enc_cfg.sdr_overlap}")

    # Timing
    print("\n--- TIMING ---")
    print(f"  Brain encoding timesteps: {config.brain.encoding_timesteps}")
    print(f"  Delay timesteps: {config.brain.delay_timesteps}")
    print(f"  Test timesteps: {config.brain.test_timesteps}")
    print(f"  Theta frequency: {config.global_.theta_frequency_hz} Hz")
    print(f"  dt: {config.global_.dt_ms} ms")

    # Brain sizes
    print("\n--- BRAIN SIZES ---")
    sizes = config.brain.sizes
    print(f"  Input size: {sizes.input_size}")
    print(f"  Cortex size: {sizes.cortex_size}")
    print(f"  Hippocampus size: {sizes.hippocampus_size}")
    print(f"  PFC size: {sizes.pfc_size}")
    print(f"  N actions: {sizes.n_actions}")

    # Cortex config
    print("\n--- CORTEX ---")
    print(f"  Type: {config.brain.cortex_type.value}")
    cortex_cfg = config.brain.cortex
    print(f"  L4 sparsity: {cortex_cfg.l4_sparsity}")
    print(f"  L2/3 sparsity: {cortex_cfg.l23_sparsity}")
    print(f"  L5 sparsity: {cortex_cfg.l5_sparsity}")
    print(f"  Input -> L4 strength: {cortex_cfg.input_to_l4_strength}")
    print(f"  L4 -> L2/3 strength: {cortex_cfg.l4_to_l23_strength}")
    print(f"  L2/3 -> L5 strength: {cortex_cfg.l23_to_l5_strength}")
    print(f"  L2/3 recurrent strength: {cortex_cfg.l23_recurrent_strength}")
    print(f"  L2/3 recurrent decay: {cortex_cfg.l23_recurrent_decay}")
    print(f"  FFI enabled: {cortex_cfg.ffi_enabled}")
    print(f"  FFI strength: {cortex_cfg.ffi_strength}")

    # Weight bounds (cortex)
    print("\n--- CORTEX WEIGHT BOUNDS ---")
    print(f"  Feedforward: [{cortex_cfg.w_min}, {cortex_cfg.w_max}]")
    print(f"  L2/3 recurrent: [{cortex_cfg.l23_recurrent_w_min}, {cortex_cfg.l23_recurrent_w_max}]")

    # Cortex learning parameters
    print("\n--- CORTEX LEARNING ---")
    print(f"  STDP LR: {cortex_cfg.stdp_lr}")
    print(f"  STDP tau+: {cortex_cfg.stdp_tau_plus} ms")
    print(f"  STDP tau-: {cortex_cfg.stdp_tau_minus} ms")
    # BCM config if available
    if hasattr(cortex_cfg, 'bcm_config') and cortex_cfg.bcm_config is not None:
        bcm = cortex_cfg.bcm_config
        print(f"  BCM tau theta: {bcm.tau_theta} ms")
        print(f"  BCM theta init: {bcm.theta_init}")

    # Hippocampus
    hippo_cfg = config.brain.hippocampus
    print("\n--- HIPPOCAMPUS ---")
    print(f"  DG sparsity: {hippo_cfg.dg_sparsity}")
    print(f"  CA3 sparsity: {hippo_cfg.ca3_sparsity}")
    print(f"  CA3 recurrent strength: {hippo_cfg.ca3_recurrent_strength}")
    print(f"  CA1 sparsity: {hippo_cfg.ca1_sparsity}")
    print(f"  CA3 learning rate: {hippo_cfg.ca3_learning_rate}")
    print(f"  NMDA tau: {hippo_cfg.nmda_tau} ms")

    # Striatum
    striatum_cfg = config.brain.striatum
    print("\n--- STRIATUM ---")
    print(f"  Population coding: {striatum_cfg.population_coding}")
    print(f"  Neurons per action: {striatum_cfg.neurons_per_action}")
    print(f"  Learning rate: {striatum_cfg.learning_rate}")
    print(f"  Eligibility tau: {striatum_cfg.eligibility_tau_ms} ms")
    print(f"  STDP LR: {striatum_cfg.stdp_lr}")
    print(f"  Homeostatic: {striatum_cfg.homeostatic_enabled}")

    # PFC
    pfc_cfg = config.brain.pfc
    print("\n--- PFC ---")
    print(f"  WM decay tau: {pfc_cfg.wm_decay_tau_ms} ms")
    print(f"  Gate threshold: {pfc_cfg.gate_threshold}")
    print(f"  Gate strength: {pfc_cfg.gate_strength}")
    print(f"  Sparsity: {pfc_cfg.sparsity}")

    # Cerebellum
    cereb_cfg = config.brain.cerebellum
    print("\n--- CEREBELLUM ---")
    print(f"  LTP rate: {cereb_cfg.learning_rate_ltp}")
    print(f"  LTD rate: {cereb_cfg.learning_rate_ltd}")
    print(f"  STDP LR: {cereb_cfg.stdp_lr}")
    print(f"  Eligibility tau: {cereb_cfg.eligibility_tau_ms} ms")

    print("=" * width + "\n")


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
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)

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
            "",
            self.robustness.summary(),
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
