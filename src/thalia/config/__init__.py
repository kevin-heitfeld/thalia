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
from .thalia_config import ThaliaConfig

__all__ = [
    # Main config
    "ThaliaConfig",
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
]
