"""
Brain Configuration - Settings for brain regions and architecture.

This module defines configuration for the EventDrivenBrain and its
constituent regions, with clear inheritance from GlobalConfig.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from .global_config import GlobalConfig


class RegionType(Enum):
    """Types of brain regions."""
    CORTEX = "cortex"
    HIPPOCAMPUS = "hippocampus"
    PFC = "pfc"
    STRIATUM = "striatum"
    CEREBELLUM = "cerebellum"


class CortexType(Enum):
    """Types of cortex implementation.
    
    LAYERED: Standard feedforward layered cortex (L4 → L2/3 → L5)
    PREDICTIVE: Layered cortex with predictive coding (local error signals)
    """
    LAYERED = "layered"
    PREDICTIVE = "predictive"


@dataclass
class RegionSizes:
    """Size configuration for brain regions.

    All sizes are in terms of number of neurons (or output size).
    Ratios and internal sizes are computed automatically.
    """

    input_size: int = 256
    """Size of input to the brain (e.g., from sensory encoding)."""

    cortex_size: int = 128
    """Output size of cortex. L2/3 and L5 layers will be sized relative to this."""

    hippocampus_size: int = 64
    """Output size of hippocampus (CA1 output)."""

    pfc_size: int = 32
    """Size of prefrontal cortex working memory."""

    n_actions: int = 2
    """Number of possible actions (for striatum output)."""

    # =========================================================================
    # DERIVED SIZES (computed properties)
    # =========================================================================

    @property
    def cortex_l4_size(self) -> int:
        """L4 (input layer) size - same as cortex output."""
        return self.cortex_size

    @property
    def cortex_l23_size(self) -> int:
        """L2/3 (processing layer) size - 1.5x cortex for recurrence."""
        return int(self.cortex_size * 1.5)

    @property
    def cortex_l5_size(self) -> int:
        """L5 (output layer) size - same as cortex output."""
        return self.cortex_size

    @property
    def hippocampus_dg_size(self) -> int:
        """Dentate gyrus size - 5x expansion for pattern separation."""
        # DG expands from cortex L2/3 input
        return int(self.cortex_l23_size * 5)

    @property
    def hippocampus_ca3_size(self) -> int:
        """CA3 size - 50% of DG for pattern completion."""
        return int(self.hippocampus_dg_size * 0.5)

    @property
    def hippocampus_ca1_size(self) -> int:
        """CA1 size - matches configured hippocampus output."""
        return self.hippocampus_size

    def summary(self) -> str:
        """Return formatted summary of region sizes."""
        lines = [
            "=== Region Sizes ===",
            f"  Input: {self.input_size}",
            f"  Cortex: {self.cortex_size}",
            f"    L4: {self.cortex_l4_size}",
            f"    L2/3: {self.cortex_l23_size}",
            f"    L5: {self.cortex_l5_size}",
            f"  Hippocampus: {self.hippocampus_size}",
            f"    DG: {self.hippocampus_dg_size}",
            f"    CA3: {self.hippocampus_ca3_size}",
            f"    CA1: {self.hippocampus_ca1_size}",
            f"  PFC: {self.pfc_size}",
            f"  Actions: {self.n_actions}",
        ]
        return "\n".join(lines)


@dataclass
class CortexConfig:
    """Configuration specific to layered cortex.

    These are cortex-specific parameters that don't apply to other regions.
    Global parameters (device, dt_ms, etc.) come from GlobalConfig.
    """

    # Layer sparsity (fraction of neurons active)
    l4_sparsity: float = 0.15
    l23_sparsity: float = 0.10
    l5_sparsity: float = 0.20

    # Recurrence in L2/3
    l23_recurrent_strength: float = 0.3
    l23_recurrent_decay: float = 0.9

    # Connection strengths
    input_to_l4_strength: float = 0.5
    l4_to_l23_strength: float = 0.4
    l23_to_l5_strength: float = 0.4
    l23_top_down_strength: float = 0.2

    # STDP learning parameters
    stdp_lr: float = 0.01
    stdp_tau_plus: float = 20.0
    stdp_tau_minus: float = 20.0

    # Feedforward inhibition
    ffi_enabled: bool = True
    ffi_threshold: float = 0.3
    ffi_strength: float = 0.8
    ffi_tau: float = 5.0

    # BCM sliding threshold
    bcm_enabled: bool = False
    bcm_tau_theta: float = 5000.0

    # Output configuration
    output_layer: str = "L5"
    dual_output: bool = True


@dataclass
class HippocampusConfig:
    """Configuration specific to trisynaptic hippocampus."""

    # DG sparsity (VERY sparse for pattern separation)
    dg_sparsity: float = 0.02
    dg_inhibition: float = 5.0

    # CA3 recurrent dynamics
    ca3_recurrent_strength: float = 0.4
    ca3_sparsity: float = 0.10
    ca3_learning_rate: float = 0.1

    # CA1 output
    ca1_sparsity: float = 0.15

    # NMDA coincidence detection
    nmda_tau: float = 50.0
    nmda_threshold: float = 0.4
    nmda_steepness: float = 12.0
    ampa_ratio: float = 0.05

    # Learning rates
    learning_rate: float = 0.2
    ec_ca1_learning_rate: float = 0.5
    stdp_tau_plus: float = 20.0
    stdp_tau_minus: float = 20.0

    # Feedforward inhibition
    ffi_threshold: float = 0.3
    ffi_strength: float = 0.8
    ffi_tau: float = 5.0


@dataclass
class StriatumConfig:
    """Configuration specific to striatum (action selection)."""

    # Population coding
    population_coding: bool = True
    neurons_per_action: int = 10

    # D1/D2 pathways
    d1_d2_enabled: bool = True
    d1_fraction: float = 0.5
    d1_lr_scale: float = 1.0
    d2_lr_scale: float = 1.0

    # Three-factor learning
    eligibility_tau_ms: float = 1000.0
    learning_rate: float = 0.005

    # RPE normalization
    normalize_rpe: bool = True
    rpe_avg_tau: float = 0.9
    rpe_clip: float = 2.0

    # Lateral inhibition
    lateral_inhibition: bool = True
    inhibition_strength: float = 2.0

    # Homeostasis
    homeostatic_enabled: bool = True
    homeostatic_rate: float = 0.1

    # Action selection
    softmax_temperature: float = 1.0


@dataclass
class PFCConfig:
    """Configuration specific to prefrontal cortex."""

    # Working memory
    wm_decay: float = 0.95
    wm_capacity: int = 7

    # Attention
    attention_gain: float = 2.0

    # Sparsity
    sparsity: float = 0.15


@dataclass
class CerebellumConfig:
    """Configuration specific to cerebellum."""

    # Granule cell expansion
    gc_expansion: float = 10.0
    gc_sparsity: float = 0.05

    # Purkinje cell learning
    purkinje_lr: float = 0.1

    # Error signal
    climbing_fiber_strength: float = 1.0


@dataclass
class BrainConfig:
    """Complete brain configuration.

    Combines region sizes with region-specific parameters.
    Global parameters come from GlobalConfig.
    """

    # Region sizes
    sizes: RegionSizes = field(default_factory=RegionSizes)

    # Region-specific configs
    cortex: CortexConfig = field(default_factory=CortexConfig)
    hippocampus: HippocampusConfig = field(default_factory=HippocampusConfig)
    striatum: StriatumConfig = field(default_factory=StriatumConfig)
    pfc: PFCConfig = field(default_factory=PFCConfig)
    cerebellum: CerebellumConfig = field(default_factory=CerebellumConfig)

    # Region type selection (allows swapping implementations)
    cortex_type: CortexType = CortexType.LAYERED
    """Which cortex implementation to use. PREDICTIVE enables local error learning."""

    # Timing (trial phases)
    encoding_timesteps: int = 15
    delay_timesteps: int = 10
    test_timesteps: int = 15

    # Execution mode
    parallel: bool = False

    def summary(self) -> str:
        """Return formatted summary of brain configuration."""
        lines = [
            "=== Brain Configuration ===",
            self.sizes.summary(),
            "",
            "--- Region Types ---",
            f"  Cortex: {self.cortex_type.value}",
            "",
            "--- Trial Timing ---",
            f"  Encoding: {self.encoding_timesteps} timesteps",
            f"  Delay: {self.delay_timesteps} timesteps",
            f"  Test: {self.test_timesteps} timesteps",
            f"  Parallel: {self.parallel}",
        ]
        return "\n".join(lines)

    @property
    def total_striatum_neurons(self) -> int:
        """Total neurons in striatum with population coding."""
        if self.striatum.population_coding:
            return self.sizes.n_actions * self.striatum.neurons_per_action
        return self.sizes.n_actions
