"""
Layered Cortex Configuration and State.

Configuration and state dataclasses for the LayeredCortex brain region.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import torch

from thalia.core.base.component_config import NeuralComponentConfig
from thalia.core.region_state import BaseRegionState
from thalia.learning.rules.bcm import BCMConfig
from thalia.regulation.learning_constants import STDP_A_PLUS_CORTEX, STDP_A_MINUS_CORTEX
from thalia.neuromodulation.constants import DA_BASELINE_STANDARD, ACH_BASELINE, NE_BASELINE
from thalia.components.neurons.neuron_constants import ADAPT_INCREMENT_CORTEX_L23
from .robustness_config import RobustnessConfig


@dataclass
class LayeredCortexConfig(NeuralComponentConfig):
    """Configuration for layered cortical microcircuit.

    **BEHAVIORAL CONFIGURATION ONLY**

    This config contains ONLY behavioral parameters (learning rates, sparsity, etc.).
    Layer sizes are provided separately during instantiation via LayerSizeCalculator.

    Usage with BrainBuilder:
        >>> from thalia.config import LayerSizeCalculator
        >>> from thalia.core.brain_builder import BrainBuilder
        >>>
        >>> calc = LayerSizeCalculator()
        >>> cortex_sizes = calc.cortex_from_scale(scale_factor=128)
        >>>
        >>> builder = BrainBuilder(global_config)
        >>> builder.add_component("cortex", "cortex", **cortex_sizes)
        >>> brain = builder.build()

    Direct region instantiation (internal use by builder):
        >>> config = LayeredCortexConfig(stdp_lr=0.001, sparsity=0.1)
        >>> sizes = calc.cortex_from_scale(128)
        >>> cortex = LayeredCortex(config=config, sizes=sizes, device="cpu")

    OUTPUT COMPUTATION:
        output_size is computed as l23_size + l5_size (dual pathways:
        cortico-cortical via L2/3 and subcortical via L5).
    """

    # NO SIZE FIELDS - sizes passed separately to __init__

    # Layer sparsity (fraction of neurons active)
    l4_sparsity: float = 0.15  # Moderate sparsity
    l23_sparsity: float = 0.10  # Sparser (more selective)
    l5_sparsity: float = 0.20  # Less sparse (motor commands)
    l6a_sparsity: float = 0.12  # L6a → TRN (slightly more sparse than L2/3)
    l6b_sparsity: float = 0.15  # L6b → relay (moderate sparsity)

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
    l23_to_l6a_strength: float = 0.8   # L2/3 → L6a (reduced for low gamma 25-35Hz)
    l23_to_l6b_strength: float = 2.0   # L2/3 → L6b (higher for high gamma 60-80Hz)

    # Top-down modulation (for attention pathway)
    l23_top_down_strength: float = 0.2  # Feedback to L2/3

    # L6 corticothalamic feedback strengths (different pathways)
    l6a_to_trn_strength: float = 0.8   # L6a → TRN (inhibitory modulation, low gamma)
    l6b_to_relay_strength: float = 0.6 # L6b → relay (excitatory modulation, high gamma)

    # Spillover transmission (volume transmission)
    # Enable in cortex L2/3 and L5 where experimentally documented
    # (Agnati et al. 2010, Fuxe & Agnati 1991, Zoli et al. 1999)
    # Cortical spillover contributes to lateral excitation and feature binding
    enable_spillover: bool = True  # Override base config (disabled by default)
    spillover_mode: str = "connectivity"  # Use shared inputs for neighborhood
    spillover_strength: float = 0.15  # 15% of direct synaptic strength (biological range)

    # Gap junctions (L2/3 interneuron synchronization)
    # Basket cells and chandelier cells in L2/3 have dense gap junction networks
    # Critical for cortical gamma oscillations (30-80 Hz) and precise spike timing
    # ~70-80% of cortical gap junctions are interneuron-interneuron (Bennett 2004)
    gap_junctions_enabled: bool = True
    """Enable gap junction coupling in L2/3 interneurons."""

    gap_junction_strength: float = 0.12
    """Gap junction conductance for L2/3 interneurons (biological: 0.05-0.2)."""

    gap_junction_threshold: float = 0.25
    """Connectivity threshold for gap junction coupling (shared inputs)."""

    gap_junction_max_neighbors: int = 8
    """Maximum gap junction neighbors per interneuron (biological: 4-12)."""

    # Note: STDP parameters (stdp_lr, tau_plus_ms, tau_minus_ms, a_plus, a_minus)
    # are inherited from NeuralComponentConfig
    # Override with cortical values from constants:
    a_plus: float = STDP_A_PLUS_CORTEX      # LTP amplitude
    a_minus: float = STDP_A_MINUS_CORTEX    # LTD amplitude

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
    adapt_increment: float = ADAPT_INCREMENT_CORTEX_L23  # Very strong adaptation for decorrelation
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

    # =========================================================================
    # INTER-LAYER AXONAL DELAYS
    # =========================================================================
    # Biological signal propagation times within cortical laminae:
    # - L4→L2/3: ~2ms (short vertical projection)
    # - L2/3→L5: ~2ms (longer vertical projection)
    # - L2/3→L6: ~2-3ms (within column, vertical projection)
    # - L6→TRN: ~10ms (corticothalamic feedback, long-range)
    # Total laminar processing: ~4-6ms (much faster than processing timescales)
    #
    # Internal delays enable realistic temporal dynamics and support oscillation emergence:
    # - L6a with 2ms internal + 10ms feedback = 12ms loop → ~83 Hz (high gamma range)
    # - L6b with 3ms internal + 5ms feedback = 8ms loop → ~125 Hz (very high gamma)
    # - With neural refractory periods and integration, actual frequencies settle to
    #   low gamma (25-35 Hz) for L6a and high gamma (60-80 Hz) for L6b
    l4_to_l23_delay_ms: float = 2.0   # L4→L2/3 axonal delay (short vertical)
    l23_to_l5_delay_ms: float = 2.0   # L2/3→L5 axonal delay (longer vertical)
    l23_to_l6a_delay_ms: float = 2.0  # L2/3→L6a axonal delay (type I pathway, slow)
    l23_to_l6b_delay_ms: float = 3.0  # L2/3→L6b axonal delay (type II pathway, fast)

    # L6 feedback delays (key for gamma frequency tuning)
    l6a_to_trn_delay_ms: float = 10.0   # L6a→TRN feedback delay (~10ms biological, slow pathway)
    l6b_to_relay_delay_ms: float = 5.0  # L6b→relay feedback delay (~5ms biological, fast pathway)

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

    # =========================================================================
    # LAYER-SPECIFIC HETEROGENEITY (Phase 2A Enhancement)
    # =========================================================================
    # Biological reality: Each cortical layer has distinct cell types with
    # different electrophysiological properties:
    # - L4 spiny stellate: Fast, small tau_mem (~10ms), low threshold
    # - L2/3 pyramidal: Medium tau_mem (~20ms), moderate threshold
    # - L5 thick-tuft pyramidal: Slow tau_mem (~30ms), high threshold, burst-capable
    # - L6 corticothalamic: Variable tau_mem (~15-25ms), moderate threshold
    #
    # This heterogeneity enables:
    # - L4: Fast sensory processing and feature detection
    # - L2/3: Integration and association over longer timescales
    # - L5: Decision-making and sustained output generation
    # - L6: Feedback control with tuned dynamics
    #
    # References:
    # - Connors & Gutnick (1990): Intrinsic firing patterns of diverse neocortical neurons
    # - Markram et al. (2015): Reconstruction and simulation of neocortical microcircuitry
    # - Ramaswamy & Markram (2015): Anatomy and physiology of the thick-tufted layer 5 pyramidal neuron
    use_layer_heterogeneity: bool = False
    """Enable layer-specific neuron properties (Phase 2A).

    When True:
        - Each layer has distinct tau_mem, v_threshold, adaptation properties
        - Reflects biological diversity of cortical cell types
        - Improves layer-specific computational roles
        - Requires layer_properties config
    """

    # Layer-specific membrane time constants (ms)
    # These control integration timescales for each layer
    layer_tau_mem: Dict[str, float] = field(default_factory=lambda: {
        "l4": 10.0,   # Fast integration for sensory input
        "l23": 20.0,  # Moderate integration for association
        "l5": 30.0,   # Slow integration for output generation
        "l6a": 15.0,  # Fast for TRN feedback (low gamma)
        "l6b": 25.0,  # Moderate for relay feedback (high gamma)
    })
    """Membrane time constants per layer (Phase 2A).

    Biological ranges:
    - L4 spiny stellate: 8-12ms (fast sensory processing)
    - L2/3 pyramidal: 18-25ms (integration)
    - L5 pyramidal: 25-35ms (sustained output)
    - L6 pyramidal: 12-20ms (feedback control)
    """

    # Layer-specific voltage thresholds (mV)
    # Higher threshold = more selective, requires more input
    layer_v_threshold: Dict[str, float] = field(default_factory=lambda: {
        "l4": -52.0,  # Low threshold for sensitive input detection
        "l23": -55.0, # Moderate threshold for balanced processing
        "l5": -50.0,  # Lower threshold for reliable output (compensated by high tau)
        "l6a": -55.0, # Moderate for attention gating
        "l6b": -52.0, # Low for fast gain modulation
    })
    """Voltage thresholds per layer (Phase 2A).

    Biological values:
    - L4: -50 to -55mV (sensitive to input)
    - L2/3: -53 to -58mV (selective integration)
    - L5: -48 to -52mV (reliable output despite high tau)
    - L6: -50 to -55mV (varied for feedback roles)
    """

    # Layer-specific adaptation strengths
    # Controls spike-frequency adaptation per layer
    layer_adaptation: Dict[str, float] = field(default_factory=lambda: {
        "l4": 0.05,  # Minimal adaptation for faithful sensory relay
        "l23": 0.15, # Strong adaptation for decorrelation (inherited default)
        "l5": 0.10,  # Moderate adaptation for sustained output
        "l6a": 0.08, # Light adaptation for feedback
        "l6b": 0.12, # Moderate adaptation for gain control
    })
    """Adaptation increments per layer (Phase 2A).

    Biological justification:
    - L4: Minimal (faithful relay of sensory input)
    - L2/3: Strong (prevents runaway recurrence, decorrelates features)
    - L5: Moderate (allows burst patterns while preventing runaway)
    - L6: Light-moderate (supports feedback dynamics)
    """


@dataclass
class LayeredCortexState(BaseRegionState):
    """State for layered cortex with RegionState protocol compliance.

    Extends BaseRegionState with cortex-specific state:
    - 6-layer architecture (L4, L2/3, L5, L6a, L6b) with spikes and traces
    - L2/3 recurrent activity accumulation
    - Top-down modulation and attention gating
    - Feedforward inhibition and alpha suppression
    - Short-term plasticity (STP) state for L2/3 recurrent pathway

    Note: Neuromodulators (dopamine, acetylcholine, norepinephrine) are
    inherited from BaseRegionState.

    The 6-layer structure reflects canonical cortical microcircuit:
    - L4: Main input layer (thalamic recipient)
    - L2/3: Cortico-cortical output and recurrent processing
    - L5: Subcortical output (motor, striatum)
    - L6a: TRN feedback for attentional gating
    - L6b: Relay feedback for gain control
    """

    # Input stored for continuous plasticity
    input_spikes: Optional[torch.Tensor] = None

    # Per-layer spike states (6 layers)
    l4_spikes: Optional[torch.Tensor] = None
    l23_spikes: Optional[torch.Tensor] = None
    l5_spikes: Optional[torch.Tensor] = None
    l6a_spikes: Optional[torch.Tensor] = None  # L6a → TRN pathway
    l6b_spikes: Optional[torch.Tensor] = None  # L6b → relay pathway

    # L2/3 membrane potential (for gap junction coupling)
    l23_membrane: Optional[torch.Tensor] = None

    # L2/3 recurrent activity (accumulated over time)
    l23_recurrent_activity: Optional[torch.Tensor] = None

    # STDP traces per layer (5 layers)
    l4_trace: Optional[torch.Tensor] = None
    l23_trace: Optional[torch.Tensor] = None
    l5_trace: Optional[torch.Tensor] = None
    l6a_trace: Optional[torch.Tensor] = None  # L6a trace for TRN feedback plasticity
    l6b_trace: Optional[torch.Tensor] = None  # L6b trace for relay feedback plasticity

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

    # Short-term plasticity state for L2/3 recurrent pathway
    stp_l23_recurrent_state: Optional[Dict[str, torch.Tensor]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary for checkpointing.

        Returns:
            Dictionary with all state fields, including nested STP state for L2/3 recurrent.
        """
        return {
            # Base region state
            "spikes": self.spikes,
            "membrane": self.membrane,
            "dopamine": self.dopamine,
            "acetylcholine": self.acetylcholine,
            "norepinephrine": self.norepinephrine,
            # Input
            "input_spikes": self.input_spikes,
            # Layer spike states
            "l4_spikes": self.l4_spikes,
            "l23_spikes": self.l23_spikes,
            "l5_spikes": self.l5_spikes,
            "l6a_spikes": self.l6a_spikes,
            "l6b_spikes": self.l6b_spikes,
            # L2/3 membrane for gap junctions
            "l23_membrane": self.l23_membrane,
            # L2/3 recurrent activity
            "l23_recurrent_activity": self.l23_recurrent_activity,
            # STDP traces
            "l4_trace": self.l4_trace,
            "l23_trace": self.l23_trace,
            "l5_trace": self.l5_trace,
            "l6a_trace": self.l6a_trace,
            "l6b_trace": self.l6b_trace,
            # Modulation state
            "top_down_modulation": self.top_down_modulation,
            "ffi_strength": self.ffi_strength,
            "alpha_suppression": self.alpha_suppression,
            # Gamma attention
            "gamma_attention_phase": self.gamma_attention_phase,
            "gamma_attention_gate": self.gamma_attention_gate,
            # Plasticity monitoring
            "last_plasticity_delta": self.last_plasticity_delta,
            # STP state
            "stp_l23_recurrent_state": self.stp_l23_recurrent_state,
        }

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        device: str = "cpu",
    ) -> "LayeredCortexState":
        """Deserialize state from dictionary.

        Args:
            data: Dictionary with state fields
            device: Target device string (e.g., 'cpu', 'cuda', 'cuda:0')

        Returns:
            LayeredCortexState instance with restored state
        """
        def transfer_tensor(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if t is None:
                return t
            return t.to(device)

        def transfer_nested_dict(d: Optional[Dict[str, torch.Tensor]]) -> Optional[Dict[str, torch.Tensor]]:
            """Transfer nested dict of tensors to device."""
            if d is None:
                return None
            return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in d.items()}

        return cls(
            # Base region state
            spikes=transfer_tensor(data.get("spikes")),
            membrane=transfer_tensor(data.get("membrane")),
            dopamine=data.get("dopamine", DA_BASELINE_STANDARD),
            acetylcholine=data.get("acetylcholine", ACH_BASELINE),
            norepinephrine=data.get("norepinephrine", NE_BASELINE),
            # Input
            input_spikes=transfer_tensor(data.get("input_spikes")),
            # Layer spike states
            l4_spikes=transfer_tensor(data.get("l4_spikes")),
            l23_spikes=transfer_tensor(data.get("l23_spikes")),
            l5_spikes=transfer_tensor(data.get("l5_spikes")),
            l6a_spikes=transfer_tensor(data.get("l6a_spikes")),
            l6b_spikes=transfer_tensor(data.get("l6b_spikes")),
            # L2/3 recurrent activity
            l23_recurrent_activity=transfer_tensor(data.get("l23_recurrent_activity")),
            # STDP traces
            l4_trace=transfer_tensor(data.get("l4_trace")),
            l23_trace=transfer_tensor(data.get("l23_trace")),
            l5_trace=transfer_tensor(data.get("l5_trace")),
            l6a_trace=transfer_tensor(data.get("l6a_trace")),
            l6b_trace=transfer_tensor(data.get("l6b_trace")),
            # Modulation state
            top_down_modulation=transfer_tensor(data.get("top_down_modulation")),
            ffi_strength=data.get("ffi_strength", 0.0),
            alpha_suppression=data.get("alpha_suppression", 1.0),
            # Gamma attention
            gamma_attention_phase=data.get("gamma_attention_phase"),
            gamma_attention_gate=transfer_tensor(data.get("gamma_attention_gate")),
            # Plasticity monitoring
            last_plasticity_delta=data.get("last_plasticity_delta", 0.0),
            # STP state
            stp_l23_recurrent_state=transfer_nested_dict(data.get("stp_l23_recurrent_state")),
            # Gap junction state (added 2025-01, backward compatible)
            l23_membrane=transfer_tensor(data.get("l23_membrane")),
        )

    def reset(self) -> None:
        """Reset state to initial values (in-place mutation).

        Zeros all tensors and resets scalars to defaults.
        This is called when starting a new simulation or resetting the region.
        """
        # Reset base state (spikes, membrane, neuromodulators)
        super().reset()

        # Reset input spikes
        if self.input_spikes is not None:
            self.input_spikes.zero_()

        # Reset layer spikes
        if self.l4_spikes is not None:
            self.l4_spikes.zero_()
        if self.l23_spikes is not None:
            self.l23_spikes.zero_()
        if self.l5_spikes is not None:
            self.l5_spikes.zero_()
        if self.l6a_spikes is not None:
            self.l6a_spikes.zero_()
        if self.l6b_spikes is not None:
            self.l6b_spikes.zero_()

        # Reset L2/3 recurrent activity
        if self.l23_recurrent_activity is not None:
            self.l23_recurrent_activity.zero_()

        # Reset traces
        if self.l4_trace is not None:
            self.l4_trace.zero_()
        if self.l23_trace is not None:
            self.l23_trace.zero_()
        if self.l5_trace is not None:
            self.l5_trace.zero_()
        if self.l6a_trace is not None:
            self.l6a_trace.zero_()
        if self.l6b_trace is not None:
            self.l6b_trace.zero_()

        # Reset modulation state
        if self.top_down_modulation is not None:
            self.top_down_modulation.zero_()
        if self.gamma_attention_gate is not None:
            self.gamma_attention_gate.zero_()

        # Reset scalars (neuromodulators handled by BaseRegionState.reset())
        self.ffi_strength = 0.0
        self.alpha_suppression = 1.0  # Reset to no suppression
        self.gamma_attention_phase = None
        self.last_plasticity_delta = 0.0

        # Note: STP state is NOT reset here - it's managed by the STP module


def calculate_layer_sizes(
    n_output: int,
    l4_ratio: float = 1.0,
    l23_ratio: float = 1.5,
    l5_ratio: float = 1.0,
    l6a_ratio: float = 0.3,
    l6b_ratio: float = 0.2,
) -> dict[str, int]:
    """Calculate layer sizes from n_output using standard ratios.

    .. deprecated:: 0.3.0
        Use :class:`LayerSizeCalculator` instead. This function will be removed in v0.4.0.

        Replacement:
            from thalia.config import LayerSizeCalculator
            calc = LayerSizeCalculator()

            # This function calculates from output size:
            sizes = calc.cortex_from_output(target_output_size=n_output)

            # Or use custom ratios:
            from thalia.config import BiologicalRatios
            custom = BiologicalRatios(l4_to_input=l4_ratio, l23_to_l4=l23_ratio, ...)
            calc = LayerSizeCalculator(ratios=custom)

    Args:
        n_output: Desired output size (used as base for calculations)
        l4_ratio: L4 size as multiple of n_output (default: 1.0)
        l23_ratio: L2/3 size as multiple of n_output (default: 1.5)
        l5_ratio: L5 size as multiple of n_output (default: 1.0)
        l6a_ratio: L6a size as multiple of n_output (default: 0.3, 60% of old L6)
        l6b_ratio: L6b size as multiple of n_output (default: 0.2, 40% of old L6)

    Returns:
        Dictionary with keys: l4_size, l23_size, l5_size, l6a_size, l6b_size
    """
    import warnings
    warnings.warn(
        "calculate_layer_sizes() is deprecated. "
        "Use LayerSizeCalculator().cortex_from_output() or cortex_from_scale() instead. "
        "This function will be removed in v0.4.0.",
        DeprecationWarning,
        stacklevel=2,
    )

    l4_size = int(n_output * l4_ratio)
    l23_size = int(n_output * l23_ratio)
    l5_size = int(n_output * l5_ratio)
    l6a_size = int(n_output * l6a_ratio)
    l6b_size = int(n_output * l6b_ratio)

    return {
        "l4_size": l4_size,
        "l23_size": l23_size,
        "l5_size": l5_size,
        "l6a_size": l6a_size,
        "l6b_size": l6b_size,
    }
