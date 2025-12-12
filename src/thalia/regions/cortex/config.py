"""
Layered Cortex Configuration and State.

Configuration and state dataclasses for the LayeredCortex brain region.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

import torch

from thalia.config.base import NeuralComponentConfig
from thalia.regions.base import NeuralComponentState
from thalia.learning.bcm import BCMConfig
from .robustness_config import RobustnessConfig


def calculate_layer_sizes(n_output: int, l4_ratio: float, l23_ratio: float, l5_ratio: float) -> tuple[int, int, int]:
    """Calculate layer sizes from output size and ratios.

    Args:
        n_output: Desired output size (typically n_output from config)
        l4_ratio: L4 size as ratio of n_output (typically 1.0)
        l23_ratio: L2/3 size as ratio of n_output (typically 1.5)
        l5_ratio: L5 size as ratio of n_output (typically 1.0)

    Returns:
        Tuple of (l4_size, l23_size, l5_size)

    Note:
        This is the canonical calculation used throughout the codebase.
        All layer size computations should use this function to ensure consistency.
    """
    l4_size = int(n_output * l4_ratio)
    l23_size = int(n_output * l23_ratio)
    l5_size = int(n_output * l5_ratio)
    return l4_size, l23_size, l5_size


class CorticalLayer(Enum):
    """Cortical layer identifiers."""

    L4 = "L4"  # Input layer
    L23 = "L2/3"  # Processing/cortico-cortical output
    L5 = "L5"  # Subcortical output


@dataclass
class LayeredCortexConfig(NeuralComponentConfig):
    """Configuration for layered cortical microcircuit.

    Layer Sizes:
        By default, layers are sized relative to the output size:
        - L4: Same as output (input processing)
        - L2/3: 1.5x output (processing, more neurons for recurrence)
        - L5: Same as output (subcortical output)
    """

    # Layer size ratios (relative to n_output)
    l4_ratio: float = 1.0  # Input layer
    l23_ratio: float = 1.5  # Processing layer (larger for recurrence)
    l5_ratio: float = 1.0  # Output layer

    # Layer sparsity (fraction of neurons active)
    l4_sparsity: float = 0.15  # Moderate sparsity
    l23_sparsity: float = 0.10  # Sparser (more selective)
    l5_sparsity: float = 0.20  # Less sparse (motor commands)

    # Recurrence in L2/3
    l23_recurrent_strength: float = 0.3  # Lateral connection strength
    l23_recurrent_decay: float = 0.9  # Recurrent activity decay

    # Feedforward connection strengths
    # These need to be strong enough that sparse activity can drive next layer above threshold.
    # Weight initialization uses abs(randn) * scale, where scale = 1/expected_active.
    # abs(randn) has mean ~0.8, so we need ~1.5-2.0x strength to compensate.
    # With ~10-15% sparsity and random weights, we need ~2.0x strength for input layer
    # and ~1.5x for subsequent layers to reliably activate postsynaptic neurons.
    input_to_l4_strength: float = 2.0  # External input → L4 (was 1.0, too weak for sparse input)
    l4_to_l23_strength: float = 1.5    # L4 → L2/3 (was 0.4, too weak)
    l23_to_l5_strength: float = 1.5    # L2/3 → L5 (was 0.4, too weak)

    # Top-down modulation (for attention pathway)
    l23_top_down_strength: float = 0.2  # Feedback to L2/3

    # Note: STDP parameters (stdp_lr, tau_plus_ms, tau_minus_ms, a_plus, a_minus)
    # are inherited from NeuralComponentConfig

    # Weight bounds for L2/3 recurrent connections (signed, compact E/I approximation)
    # Unlike feedforward connections, recurrent lateral connections use signed weights
    # to approximate the mixed excitatory/inhibitory microcircuit within a cortical layer.
    # Positive weights = local excitation, negative weights = lateral inhibition.
    l23_recurrent_w_min: float = -1.5  # Allows inhibitory-like connections
    l23_recurrent_w_max: float = 1.0   # Symmetric by default

    # =========================================================================
    # SPIKE-FREQUENCY ADAPTATION (SFA)
    # =========================================================================
    # Cortical pyramidal neurons show strong spike-frequency adaptation.
    # Inherited from base: adapt_increment=0.0, adapt_tau=100.0
    # Override for L2/3 strong adaptation:
    adapt_increment: float = 0.3  # L2/3 needs strong adaptation for decorrelation
    # adapt_tau: 100.0 (use base default)

    # =========================================================================
    # CORTEX-SPECIFIC DYNAMICS
    # =========================================================================
    # These parameters control cortical circuit mechanisms that are specific
    # to layered cortex architecture (not universal like homeostasis).
    #
    # Note: Intrinsic plasticity (threshold adaptation) is handled by
    # UnifiedHomeostasis via activity_target in the base NeuralComponentConfig.

    # Feedforward Inhibition (FFI) parameters
    # FFI detects stimulus changes and transiently suppresses recurrent activity
    # This is how the cortex naturally "clears" old representations when new input arrives
    # Always enabled (fundamental cortical mechanism)
    ffi_threshold: float = 0.3  # Input change threshold to trigger FFI
    ffi_strength: float = 0.8  # How much FFI suppresses L2/3 recurrent activity
    ffi_tau: float = 5.0  # FFI decay time constant (ms)

    # Gamma-based attention (spike-native phase gating for L2/3)
    # Always enabled for spike-native attention
    gamma_attention_width: float = 0.3     # Phase window width

    # =========================================================================
    # BCM SLIDING THRESHOLD (Metaplasticity)
    # =========================================================================
    # The BCM rule provides a sliding threshold for synaptic modification that
    # automatically adjusts based on postsynaptic activity history. This is
    # particularly important for cortical learning because:
    # 1. Prevents runaway potentiation in highly active neurons
    # 2. Maintains selectivity during feature learning
    # 3. Enables competitive dynamics between feature detectors
    #
    # In visual cortex, BCM explains orientation selectivity development:
    # neurons that respond strongly to one orientation have high thresholds,
    # making them less likely to respond to other orientations.
    bcm_enabled: bool = False
    bcm_config: Optional[BCMConfig] = None  # BCM configuration (if enabled)

    # =========================================================================
    # ROBUSTNESS MECHANISMS (Cortex-Specific)
    # =========================================================================
    # Optional configuration for mechanisms NOT already in UnifiedHomeostasis:
    # - E/I Balance: Maintains healthy excitation/inhibition ratio (critical for recurrence)
    # - Criticality: Maintains network near critical point (research/diagnostics)
    # - Metabolic: Energy-based regularization for sparsity (optional goal)
    #
    # Note: Activity regulation and threshold adaptation are handled by
    # UnifiedHomeostasis (in NeuralComponentConfig base class).
    robustness: Optional[RobustnessConfig] = field(default=None)


@dataclass
class LayeredCortexState(NeuralComponentState):
    """State for layered cortex."""

    # Input stored for continuous plasticity
    input_spikes: Optional[torch.Tensor] = None

    # Per-layer spike states
    l4_spikes: Optional[torch.Tensor] = None
    l23_spikes: Optional[torch.Tensor] = None
    l5_spikes: Optional[torch.Tensor] = None

    # L2/3 recurrent activity (accumulated over time)
    l23_recurrent_activity: Optional[torch.Tensor] = None

    # STDP traces per layer
    l4_trace: Optional[torch.Tensor] = None
    l23_trace: Optional[torch.Tensor] = None
    l5_trace: Optional[torch.Tensor] = None

    # Top-down modulation state
    top_down_modulation: Optional[torch.Tensor] = None

    # Feedforward inhibition strength (0-1, 1 = max suppression)
    ffi_strength: float = 0.0

    # Alpha oscillation suppression (0-1, 1 = no suppression, 0.5 = max suppression)
    alpha_suppression: float = 1.0

    # Gamma attention state (spike-native phase gating)
    gamma_attention_phase: Optional[float] = None  # Current gamma phase
    gamma_attention_gate: Optional[torch.Tensor] = None  # Per-neuron gating [l23_size]

    # Last plasticity delta (for monitoring continuous learning)
    last_plasticity_delta: float = 0.0
