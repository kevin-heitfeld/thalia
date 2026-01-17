"""
Learning Configuration Base Classes.

This module provides base configuration classes for learning-related parameters,
reducing duplication across region configs.

Author: Thalia Project
Date: December 22, 2025 (Tier 2.1 - Configuration consolidation)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from thalia.config.base import BaseConfig


@dataclass
class BaseLearningConfig(BaseConfig):
    """Base configuration for learning parameters.

    Provides common learning-related fields that appear across many regions:
    - learning_rate: Base learning rate
    - learning_enabled: Global learning enable/disable
    - weight_bounds: Min/max weight constraints

    Region configs can inherit from this to avoid duplicating these fields.

    Usage:
        @dataclass
        class MyRegionConfig(BaseLearningConfig):
            # Inherits learning_rate, learning_enabled, etc.
            n_input: int = 100
            n_output: int = 50
            # Add region-specific fields
    """

    learning_rate: float = 0.01
    """Base learning rate for synaptic updates"""

    learning_enabled: bool = True
    """Global flag to enable/disable learning"""

    weight_min: float = 0.0
    """Minimum weight value (default: 0.0 for excitatory-only)"""

    weight_max: float = 1.0
    """Maximum weight value (default: 1.0 normalized)"""

    use_weight_normalization: bool = False
    """Whether to normalize weights after updates"""


@dataclass
class ModulatedLearningConfig(BaseLearningConfig):
    """Configuration for neuromodulator-gated learning.

    Extends BaseLearningConfig with neuromodulator-specific parameters
    for regions that use dopamine/acetylcholine/norepinephrine gating.

    Used by: Striatum, Prefrontal, regions with three-factor learning

    Usage:
        @dataclass
        class StriatumConfig(NeuralComponentConfig):
            learning: ModulatedLearningConfig = field(
                default_factory=ModulatedLearningConfig
            )
    """

    modulator_threshold: float = 0.1
    """Minimum modulator level to enable learning (e.g., dopamine threshold)"""

    modulator_sensitivity: float = 1.0
    """Scaling factor for modulator influence on learning rate"""

    use_dopamine_gating: bool = True
    """Whether to gate learning by dopamine levels"""

    use_eligibility_traces: bool = False
    """Whether to use eligibility traces for credit assignment"""

    eligibility_tau_ms: Optional[float] = None
    """Time constant for eligibility trace decay (ms). None = use default"""


@dataclass
class STDPLearningConfig(BaseLearningConfig):
    """Configuration for STDP (Spike-Timing-Dependent Plasticity) learning.

    Provides STDP-specific parameters for regions using timing-based learning.

    Used by: Hippocampus, Cortex, regions with temporal learning

    Usage:
        @dataclass
        class CortexConfig(NeuralComponentConfig):
            learning: STDPLearningConfig = field(default_factory=STDPLearningConfig)
    """

    tau_plus_ms: float = 20.0
    """Time constant for pre-before-post potentiation (ms)"""

    tau_minus_ms: float = 20.0
    """Time constant for post-before-pre depression (ms)"""

    a_plus: float = 0.01
    """Maximum potentiation amplitude"""

    a_minus: float = 0.01
    """Maximum depression amplitude"""

    use_symmetric: bool = False
    """Whether to use symmetric STDP (equal potentiation/depression)"""


@dataclass
class HebbianLearningConfig(BaseLearningConfig):
    """Configuration for Hebbian learning ("fire together, wire together").

    Provides parameters for correlation-based learning where synaptic strength
    increases when pre- and postsynaptic neurons are coactive.

    Used by: Multisensory integration, simple associative regions

    Usage:
        @dataclass
        class MultisensoryConfig(HebbianLearningConfig):
            # Inherits Hebbian parameters
            n_visual: int = 256
            n_auditory: int = 128
    """

    hebbian_decay: float = 0.0
    """Weight decay rate per timestep (prevents runaway potentiation)"""

    use_normalized: bool = False
    """Whether to normalize synaptic weights after updates"""


@dataclass
class ErrorCorrectiveLearningConfig(BaseLearningConfig):
    """Configuration for error-corrective (supervised) learning.

    Provides parameters for supervised learning where synaptic changes are driven
    by error signals (target - actual). Supports bidirectional plasticity with
    separate rates for potentiation (LTP) and depression (LTD).

    Used by: Cerebellum (climbing fiber errors), error-driven regions

    Usage:
        @dataclass
        class CerebellumConfig(ErrorCorrectiveLearningConfig):
            # Inherits LTP/LTD rates, error parameters
            n_purkinje: int = 256
            granule_expansion_factor: float = 4.0
    """

    learning_rate_ltp: float = 0.01
    """Learning rate for long-term potentiation (strengthening)"""

    learning_rate_ltd: float = 0.01
    """Learning rate for long-term depression (weakening)"""

    error_threshold: float = 0.01
    """Minimum error magnitude to trigger learning"""

    use_eligibility_traces: bool = True
    """Whether to use eligibility traces for temporal credit assignment"""

    eligibility_tau_ms: float = 20.0
    """Time constant for eligibility trace decay (ms)"""


# =============================================================================
# CONFIGURATION INHERITANCE EXAMPLES
# =============================================================================
#
# Example 1: Simple region with basic learning
# ============================================
# @dataclass
# class ThalamicRelayConfig(BaseLearningConfig):
#     n_input: int = 100
#     n_output: int = 100
#     # Inherits: learning_rate, learning_enabled, weight_bounds
#
# Example 2: Region with neuromodulated learning
# ===============================================
# @dataclass
# class StriatumConfig(NeuralComponentConfig):
#     n_input: int = 50
#     n_output: int = 4
#     learning: ModulatedLearningConfig = field(default_factory=ModulatedLearningConfig)
#     # Nested learning config instead of flat fields
#
# =============================================================================


__all__ = [
    "BaseLearningConfig",
    "ModulatedLearningConfig",
    "STDPLearningConfig",
    "HebbianLearningConfig",
    "ErrorCorrectiveLearningConfig",
]
