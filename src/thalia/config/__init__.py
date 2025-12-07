"""
THALIA Unified Configuration System.

This module provides a centralized configuration system that:
1. Eliminates parameter duplication across modules
2. Provides clear inheritance of global parameters
3. Validates configuration consistency
4. Shows effective values being used

Usage:
======

    from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig

    # Create configuration with customizations
    config = ThaliaConfig(
        global_=GlobalConfig(device="cuda", vocab_size=10000),
        brain=BrainConfig(cortex_size=256, hippocampus_size=128),
    )

    # Show what values are actually being used
    config.summary()

    # Create brain with resolved config
    brain = EventDrivenBrain.from_config(config)

Author: Thalia Project
Date: December 2025
"""

from .base import (
    BaseConfig,
    NeuralComponentConfig,
    LearningComponentConfig,
    RegionConfigBase,
)
from .global_config import GlobalConfig
from .brain_config import (
    BrainConfig,
    RegionSizes,
    CortexType,
    HippocampusConfig,
    StriatumConfig,
    PFCConfig,
    CerebellumConfig,
)
# Re-export LayeredCortexConfig as the canonical cortex config
from thalia.regions.cortex.config import LayeredCortexConfig
from .language_config import (
    LanguageConfig,
    EncodingConfig,
    DecodingConfig,
    PositionConfig,
    SequenceMemoryConfig,
)
from .training_config import (
    TrainingConfig,
    LearningRatesConfig,
    CheckpointConfig,
    LoggingConfig,
)
from .robustness_config import RobustnessConfig
from .thalia_config import ThaliaConfig, print_config

__all__ = [
    # Main config
    "ThaliaConfig",
    "print_config",
    # Base configs
    "BaseConfig",
    "NeuralComponentConfig",
    "LearningComponentConfig",
    "RegionConfigBase",
    # Global
    "GlobalConfig",
    # Brain
    "BrainConfig",
    "RegionSizes",
    "LayeredCortexConfig",  # Canonical cortex config (replaces CortexConfig)
    "CortexType",
    "HippocampusConfig",
    "StriatumConfig",
    "PFCConfig",
    "CerebellumConfig",
    # Language
    "LanguageConfig",
    "EncodingConfig",
    "DecodingConfig",
    "PositionConfig",
    "SequenceMemoryConfig",
    # Training
    "TrainingConfig",
    "LearningRatesConfig",
    "CheckpointConfig",
    "LoggingConfig",
    # Robustness
    "RobustnessConfig",
]
