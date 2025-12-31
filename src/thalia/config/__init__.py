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
    brain = DynamicBrain.from_config(config)

Author: Thalia Project
Date: December 2025
"""

# Re-export region configs from canonical locations
from thalia.regions.cortex.config import LayeredCortexConfig
from thalia.regions.cortex.predictive_cortex import PredictiveCortexConfig
from thalia.regions.cortex.robustness_config import RobustnessConfig
from thalia.regions.hippocampus.config import HippocampusConfig
from thalia.regions.striatum.config import StriatumConfig
from thalia.regions.prefrontal import PrefrontalConfig
from thalia.regions.cerebellum import CerebellumConfig

# Re-export component configs from core/
from thalia.core.base.component_config import (
    NeuralComponentConfig,
    LearningComponentConfig,
    PathwayConfig,
)

# Re-export region architecture constants from regulation module
from thalia.regulation.region_architecture_constants import (
    HIPPOCAMPUS_DG_EXPANSION_FACTOR,
    HIPPOCAMPUS_CA3_SIZE_RATIO,
    HIPPOCAMPUS_CA2_SIZE_RATIO,
    HIPPOCAMPUS_CA1_SIZE_RATIO,
    HIPPOCAMPUS_SPARSITY_TARGET,
    CORTEX_L4_RATIO,
    CORTEX_L23_RATIO,
    CORTEX_L5_RATIO,
    CORTEX_L6_RATIO,
    STRIATUM_NEURONS_PER_ACTION,
    STRIATUM_D1_D2_RATIO,
    THALAMUS_TRN_RATIO,
    MULTISENSORY_VISUAL_RATIO,
    MULTISENSORY_AUDITORY_RATIO,
    MULTISENSORY_LANGUAGE_RATIO,
    MULTISENSORY_INTEGRATION_RATIO,
    CEREBELLUM_GRANULE_EXPANSION,
    CEREBELLUM_PURKINJE_PER_DCN,
    PFC_WM_CAPACITY_RATIO,
    METACOG_ABSTENTION_STAGE1,
    METACOG_ABSTENTION_STAGE2,
    METACOG_ABSTENTION_STAGE3,
    METACOG_ABSTENTION_STAGE4,
    METACOG_CALIBRATION_LR,
)

from .base import BaseConfig

from .learning_config import (
    BaseLearningConfig,
    ModulatedLearningConfig,
    STDPLearningConfig,
    HebbianLearningConfig,
)
from .global_config import GlobalConfig
from .brain_config import (
    BrainConfig,
    RegionSizes,
    CortexType,
    NeuromodulationConfig,
)
from .training_config import TrainingConfig
from .language_config import (
    LanguageConfig,
    EncodingConfig,
    DecodingConfig,
    PositionConfig,
    SequenceMemoryConfig,
)
from .thalia_config import ThaliaConfig, print_config
from .validation import (
    validate_thalia_config,
    validate_brain_config,
    validate_global_consistency,
    validate_region_sizes,
    check_config_and_warn,
    ConfigValidationError,
    ValidatedConfig,
    ValidatorRegistry,
)
from .region_sizes import (
    # Hippocampus ratios
    DG_TO_EC_EXPANSION,
    CA3_TO_DG_RATIO,
    CA2_TO_DG_RATIO,
    CA1_TO_CA3_RATIO,
    # Cortex ratios
    L4_TO_INPUT_RATIO,
    L23_TO_L4_RATIO,
    L5_TO_L23_RATIO,
    # Default sizes
    DEFAULT_CORTEX_SIZE,
    DEFAULT_HIPPOCAMPUS_SIZE,
    DEFAULT_PFC_SIZE,
    DEFAULT_N_ACTIONS,
    # Utility functions
    compute_hippocampus_sizes,
    compute_cortex_layer_sizes,
    compute_striatum_sizes,
    compute_thalamus_sizes,
    compute_multisensory_sizes,
    compute_cerebellum_sizes,
)

__all__ = [
    # Main config
    "ThaliaConfig",
    "print_config",
    # Validation
    "validate_thalia_config",
    "validate_brain_config",
    "validate_global_consistency",
    "validate_region_sizes",
    "check_config_and_warn",
    "ConfigValidationError",
    "ValidatedConfig",
    "ValidatorRegistry",
    # Base configs
    "BaseConfig",
    "BaseLearningConfig",
    "ModulatedLearningConfig",
    "STDPLearningConfig",
    "HebbianLearningConfig",
    "NeuralComponentConfig",
    "LearningComponentConfig",
    "PathwayConfig",
    # Global
    "GlobalConfig",
    # Brain
    "BrainConfig",
    "RegionSizes",
    "CortexType",
    "NeuromodulationConfig",
    # Training
    "TrainingConfig",
    # Region configs (canonical locations)
    "LayeredCortexConfig",
    "PredictiveCortexConfig",
    "RobustnessConfig",
    "HippocampusConfig",
    "StriatumConfig",
    "PrefrontalConfig",
    "CerebellumConfig",
    # Language
    "LanguageConfig",
    "EncodingConfig",
    "DecodingConfig",
    "PositionConfig",
    "SequenceMemoryConfig",
    # Region size constants
    "DG_TO_EC_EXPANSION",
    "CA3_TO_DG_RATIO",
    "CA2_TO_DG_RATIO",
    "CA1_TO_CA3_RATIO",
    "L4_TO_INPUT_RATIO",
    "L23_TO_L4_RATIO",
    "L5_TO_L23_RATIO",
    "DEFAULT_CORTEX_SIZE",
    "DEFAULT_HIPPOCAMPUS_SIZE",
    "DEFAULT_PFC_SIZE",
    "DEFAULT_N_ACTIONS",
    "compute_hippocampus_sizes",
    "compute_cortex_layer_sizes",
    "compute_striatum_sizes",
    "compute_thalamus_sizes",
    "compute_multisensory_sizes",
    "compute_cerebellum_sizes",
    # Region architecture constants
    "HIPPOCAMPUS_DG_EXPANSION_FACTOR",
    "HIPPOCAMPUS_CA3_SIZE_RATIO",
    "HIPPOCAMPUS_CA2_SIZE_RATIO",
    "HIPPOCAMPUS_CA1_SIZE_RATIO",
    "HIPPOCAMPUS_SPARSITY_TARGET",
    "CORTEX_L4_RATIO",
    "CORTEX_L23_RATIO",
    "CORTEX_L5_RATIO",
    "CORTEX_L6_RATIO",
    "STRIATUM_NEURONS_PER_ACTION",
    "STRIATUM_D1_D2_RATIO",
    "THALAMUS_TRN_RATIO",
    "MULTISENSORY_VISUAL_RATIO",
    "MULTISENSORY_AUDITORY_RATIO",
    "MULTISENSORY_LANGUAGE_RATIO",
    "MULTISENSORY_INTEGRATION_RATIO",
    "CEREBELLUM_GRANULE_EXPANSION",
    "PFC_WM_CAPACITY_RATIO",
    "CEREBELLUM_GRANULE_EXPANSION",
    "CEREBELLUM_PURKINJE_PER_DCN",
    "METACOG_ABSTENTION_STAGE1",
    "METACOG_ABSTENTION_STAGE2",
    "METACOG_ABSTENTION_STAGE3",
    "METACOG_ABSTENTION_STAGE4",
    "METACOG_CALIBRATION_LR",
]
