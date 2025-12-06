"""
Layered Cortex Configuration and State.

Configuration and state dataclasses for the LayeredCortex brain region.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from enum import Enum

import torch

from thalia.regions.base import RegionConfig, RegionState
from thalia.learning.bcm import BCMConfig


class CorticalLayer(Enum):
    """Cortical layer identifiers."""

    L4 = "L4"  # Input layer
    L23 = "L2/3"  # Processing/cortico-cortical output
    L5 = "L5"  # Subcortical output


@dataclass
class LayeredCortexConfig(RegionConfig):
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

    # STDP learning parameters
    stdp_lr: float = 0.01
    stdp_tau_plus: float = 20.0
    stdp_tau_minus: float = 20.0

    # Weight bounds for feedforward connections (positive-only, Dale's law)
    w_max: float = 1.0
    w_min: float = 0.0

    # Weight bounds for L2/3 recurrent connections (signed, compact E/I approximation)
    # Unlike feedforward connections, recurrent lateral connections use signed weights
    # to approximate the mixed excitatory/inhibitory microcircuit within a cortical layer.
    # Positive weights = local excitation, negative weights = lateral inhibition.
    l23_recurrent_w_min: float = -1.5  # Allows inhibitory-like connections
    l23_recurrent_w_max: float = 1.0   # Symmetric by default

    # Which layer to use as output to next region
    output_layer: str = "L5"  # "L2/3" for cortical, "L5" for subcortical

    # Whether to output both layers (for different pathways)
    dual_output: bool = True  # Output both L2/3 and L5

    # Feedforward Inhibition (FFI) parameters
    # FFI detects stimulus changes and transiently suppresses recurrent activity
    # This is how the cortex naturally "clears" old representations when new input arrives
    ffi_enabled: bool = True  # Enable FFI mechanism
    ffi_threshold: float = 0.3  # Input change threshold to trigger FFI
    ffi_strength: float = 0.8  # How much FFI suppresses L2/3 recurrent activity
    ffi_tau: float = 5.0  # FFI decay time constant (ms)

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
    bcm_tau_theta: float = 5000.0  # Threshold adaptation time constant (ms)
    bcm_theta_init: float = 0.01  # Initial sliding threshold
    bcm_config: Optional[BCMConfig] = None  # Custom BCM configuration


@dataclass
class LayeredCortexState(RegionState):
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

    # Last plasticity delta (for monitoring continuous learning)
    last_plasticity_delta: float = 0.0
