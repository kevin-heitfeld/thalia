"""
Layered Cortex - Multi-layer cortical microcircuit.

This implements a biologically realistic cortical column with distinct layers:

Architecture (based on canonical cortical microcircuit):
=========================================================

         Feedback from higher areas
                    │
                    ▼
    ┌───────────────────────────────────┐
    │          LAYER 2/3                │ ← Superficial pyramidal cells
    │   (Cortico-cortical output)       │ → To other cortical areas
    │   - Receives from L4              │ → Gamma attention gating
    │   - Lateral recurrent connections │ → Gap junctions for sync
    │   - Top-down feedback target      │
    └───────────────┬───────────────────┘
                    │
    ┌───────────────┴───────────────────┐
    │          LAYER 4                  │ ← Spiny stellate cells
    │   (Feedforward input layer)       │ ← From thalamus/lower areas
    │   - Main sensory input recipient  │
    │   - No recurrent connections      │
    │   - Fast, feedforward processing  │
    └───────────────┬───────────────────┘
                    │
    ┌───────────────┴───────────────────┐
    │          LAYER 5                  │ ← Deep pyramidal cells
    │   (Subcortical output layer)      │ → To striatum, brainstem, etc.
    │   - Receives from L2/3            │ → Motor/action-related output
    │   - Different output pathway      │
    │   - Burst-capable neurons         │
    └───────────────┬───────────────────┘
                    │
    ┌───────────────┴───────────────────┐
    │          LAYER 6a                 │ ← CT Type I (low gamma)
    │   (Corticothalamic → TRN)         │ → Spatial attention via TRN
    │   - Receives from L2/3            │ → Inhibitory modulation
    │   - Projects to thalamic TRN      │
    │   - Sparse, low-gamma firing      │
    └───────────────────────────────────┘
                    │
    ┌───────────────┴───────────────────┐
    │          LAYER 6b                 │ ← CT Type II (high gamma)
    │   (Corticothalamic → Relay)       │ → Fast gain modulation
    │   - Receives from L2/3            │ → Direct relay excitation
    │   - Projects to thalamic relay    │
    │   - Dense, high-gamma firing      │
    └───────────────────────────────────┘

FILE ORGANIZATION (~2000 lines)
================================
Lines 1-150:     Module docstring, imports, class registration
Lines 151-350:   __init__() and layer initialization (L4/L2/3/L5/L6a/L6b)
Lines 351-500:   L4 forward pass (input processing)
Lines 501-700:   L2/3 forward pass (recurrent processing)
Lines 701-850:   L5 forward pass (output generation)
Lines 851-950:   L6a forward pass (corticothalamic type I → TRN)
Lines 951-1050:  L6b forward pass (corticothalamic type II → relay)
Lines 1051-1200: Learning (BCM + STDP for inter-layer connections)
Lines 1201-1350: Growth and homeostasis
Lines 1351-2000: Diagnostics and utility methods

QUICK NAVIGATION
================
VSCode shortcuts:
  • Ctrl+Shift+O (Cmd+Shift+O on Mac) - "Go to Symbol" for method jumping
  • Ctrl+K Ctrl+0 - Collapse all regions to see file outline
  • Ctrl+K Ctrl+J - Expand all regions
  • Ctrl+G - Go to specific line number
  • Ctrl+F - Search within file

Key methods to jump to:
  • __init__() - Layer initialization and weight setup
  • forward() - Main forward pass (L4→L2/3→L5 cascade)
  • _process_l4() - Layer 4 feedforward processing
  • _process_l23() - Layer 2/3 recurrent processing
  • _process_l5() - Layer 5 output generation
  • _apply_learning() - BCM + STDP learning
  • grow_output() / grow_input() - Layer growth
  • set_oscillator_phases() - Theta/gamma modulation
  • get_diagnostics() - Layer-wise health metrics

WHY THIS FILE IS LARGE
======================
The L4→L2/3→L5 cascade is a single biological computation within one timestep.
Splitting by layer would:
1. Require passing 15+ intermediate tensors (spikes, membrane, conductances)
2. Break the canonical microcircuit structure
3. Duplicate inter-layer connection management
4. Obscure the feedforward/feedback balance

Components ARE extracted where orthogonal:
- Learning strategies: BCM and STDP in learning/strategies.py
- StimulusGating: Stimulus-triggered inhibition (shared with hippocampus)
- LayerEIBalance: E/I balance (shared concern)

See: docs/decisions/adr-011-large-file-justification.md

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from thalia.components.coding import compute_firing_rate, compute_spike_count
from thalia.components.gap_junctions import GapJunctionConfig, GapJunctionCoupling
from thalia.components.neurons import create_cortical_layer_neurons
from thalia.components.synapses import (
    ShortTermPlasticity,
    STPConfig,
    STPType,
    WeightInitializer,
    update_trace,
)
from thalia.constants.architecture import (
    CORTEX_L4_DA_FRACTION,
    CORTEX_L5_DA_FRACTION,
    CORTEX_L6_DA_FRACTION,
    CORTEX_L23_DA_FRACTION,
)
from thalia.constants.learning import EMA_DECAY_FAST
from thalia.constants.oscillator import (
    L4_INPUT_ENCODING_SCALE,
    L23_RECURRENT_RETRIEVAL_SCALE,
)
from thalia.core.diagnostics_schema import (
    compute_activity_metrics,
    compute_health_metrics,
    compute_plasticity_metrics,
)
from thalia.core.neural_region import NeuralRegion
from thalia.learning import BCMStrategyConfig, STDPConfig, create_cortex_strategy
from thalia.learning.ei_balance import LayerEIBalance
from thalia.learning.homeostasis.synaptic_homeostasis import (
    UnifiedHomeostasis,
    UnifiedHomeostasisConfig,
)
from thalia.learning.rules.strategies import LearningStrategy
from thalia.managers.component_registry import register_region
from thalia.neuromodulation import compute_ne_gain
from thalia.regions.stimulus_gating import StimulusGating
from thalia.utils.core_utils import clamp_weights, ensure_1d, initialize_phase_preferences
from thalia.utils.input_routing import InputRouter
from thalia.utils.oscillator_utils import (
    compute_ach_recurrent_suppression,
    compute_theta_encoding_retrieval,
)

from .config import LayeredCortexConfig, LayeredCortexState


@register_region(
    "cortex",
    aliases=["layered_cortex"],
    description="Multi-layer cortical microcircuit with L4/L2/3/L5 structure",
    version="2.0",
    author="Thalia Project",
    config_class=LayeredCortexConfig,
)
class LayeredCortex(NeuralRegion):
    """Multi-layer cortical microcircuit with proper layer separation and routing.

    Implements a canonical cortical column with distinct computational layers:

    **Layer Architecture** (based on mammalian cortex):
    - **L4**: Input layer - receives thalamic/sensory input, feedforward processing
    - **L2/3**: Processing layer - recurrent computation, outputs to other cortex
    - **L5**: Output layer - projects to subcortical structures (striatum, etc.)
    - **L6a**: CT Type I - projects to thalamic TRN (spatial attention, low gamma)
    - **L6b**: CT Type II - projects to thalamic relay (gain modulation, high gamma)

    **Key Insight**:
    Output to next cortical area comes from a DIFFERENT layer (L2/3) than the
    one receiving recurrent feedback, solving the contamination problem in
    single-layer models. L5 provides separate subcortical output pathway.

    **Information Flow**:
    1. External input → L4 (feedforward processing)
    2. L4 → L2/3 (local integration with axonal delays)
    3. L2/3 → L2/3 (recurrent processing with STP and gap junctions)
    4. L2/3 → L5 (deep projection with axonal delays)
    5. L2/3 → L6a (corticothalamic type I with axonal delays)
    6. L2/3 → L6b (corticothalamic type II with axonal delays)
    7. L2/3 → Other cortex (cortico-cortical output)
    8. L5 → Subcortical (striatum, thalamus, brainstem)
    9. L6a → Thalamus TRN (spatial attention via inhibitory modulation)
    10. L6b → Thalamus Relay (fast gain modulation via excitatory drive)

    **Learning Mechanisms**:
    - **Intra-layer**: BCM rule for homeostatic plasticity
    - **Inter-layer**: STDP for connection refinement
    - **Modulation**: Dopamine gates learning rate, ACh modulates encoding mode

    **Output Format**:
    Concatenated [L2/3_spikes, L5_spikes] for routing:
    - First n_l23 neurons: Cortico-cortical pathway
    - Last n_l5 neurons: Subcortical pathway
    - L6a spikes: Available via port routing (source_port="l6a")
    - L6b spikes: Available via port routing (source_port="l6b")

    **Port-Based Routing**:
    Access specific layers via ports in BrainBuilder.connect():
    - source_port="l23" → Cortico-cortical connections
    - source_port="l5" → Cortico-subcortical connections
    - source_port="l6a" → Corticothalamic type I (TRN inhibitory modulation)
    - source_port="l6b" → Corticothalamic type II (relay excitatory modulation)

    **Usage Example**:

    .. code-block:: python

        config = LayeredCortexConfig(
            n_input=256,
            n_output=128,  # Total output size (L2/3 + L5)
            l4_size=64,    # Input layer
            l23_size=96,   # Processing/cortico-cortical output (1.5x)
            l5_size=32,    # Subcortical output
            l6a_size=16,   # CT type I → TRN (0.25x, sparse low-gamma)
            l6b_size=16,   # CT type II → relay (0.25x, dense high-gamma)
        )
        cortex = LayeredCortex(config)

        # Process input
        output = cortex(input_spikes)

        # Route output by layer
        l23_size = cortex.l23_size
        cortico_output = output[:l23_size]  # To other cortex
        subcortical_output = output[l23_size:]  # To striatum/thalamus

        # Access L6 outputs via port routing
        l6a_spikes = cortex.get_port_output("l6a")  # To TRN
        l6b_spikes = cortex.get_port_output("l6b")  # To relay

    **Mixins Provide**:

    From LearningStrategyMixin:
        - add_strategy(), apply_learning() - Pluggable learning rules

    From DiagnosticsMixin:
        - check_health(), get_firing_rate() - Health monitoring

    From GrowthMixin:
        - grow_output(), get_capacity_metrics() - Curriculum learning

    From NeuromodulatorMixin:
        - set_dopamine() - DA modulation

    **See Also**:
    - docs/patterns/mixins.md - Detailed mixin patterns
    - docs/decisions/adr-011-large-file-justification.md - Why single file
    - docs/patterns/state-management.md - LayeredCortexState usage
    """

    def __init__(
        self,
        config: LayeredCortexConfig,
        sizes: Dict[str, int],
        device: str,
    ):
        """Initialize layered cortex.

        Args:
            config: Behavioral configuration (learning rates, sparsity, etc.)
            sizes: Layer sizes from LayerSizeCalculator (l4_size, l23_size, l5_size, l6a_size, l6b_size, input_size)
            device: Device for tensors ("cpu" or "cuda")
        """
        # Store config and device
        self.config = config
        self.device = torch.device(device)

        # Read layer sizes from sizes dict (computed by LayerSizeCalculator)
        self.l4_size = sizes["l4_size"]
        self.l23_size = sizes["l23_size"]
        self.l5_size = sizes["l5_size"]
        self.l6a_size = sizes["l6a_size"]  # L6a → TRN pathway
        self.l6b_size = sizes["l6b_size"]  # L6b → relay pathway
        self.input_size = sizes.get("input_size", 0)  # May be 0 (inferred by builder)

        # Total neurons across all 5 layers
        total_neurons = self.l4_size + self.l23_size + self.l5_size + self.l6a_size + self.l6b_size

        # Initialize NeuralRegion with all cortical neurons
        super().__init__(
            n_neurons=total_neurons,
            device=device,
            dt_ms=config.dt_ms,
        )

        # Override n_output to match actual forward() return size
        # LayeredCortex returns concatenated L2/3 + L5 outputs
        # (L4, L6a, L6b are internal processing layers, not external outputs)
        self.n_output = self.l23_size + self.l5_size

        # Initialize layers
        self._init_layers()

        # Initialize inter-layer weights
        self._init_weights()

        # Initialize stimulus gating (transient inhibition) - always enabled
        self.stimulus_gating = StimulusGating(
            threshold=config.ffi_threshold,
            max_inhibition=config.ffi_strength * 10.0,
            decay_rate=1.0 - (1.0 / config.ffi_tau),
        )

        # STDP+BCM learning strategies for each layer
        if config.bcm_enabled:

            # Custom STDP config using biologically appropriate amplitudes from config
            stdp_cfg = STDPConfig(
                learning_rate=config.learning_rate,
                a_plus=config.a_plus,
                a_minus=config.a_minus,
                tau_plus=config.tau_plus_ms,
                tau_minus=config.tau_minus_ms,
                dt_ms=config.dt_ms,
                w_min=config.w_min,
                w_max=config.w_max,
            )

            # BCM config for homeostatic modulation
            bcm_cfg = config.bcm_config or BCMStrategyConfig(
                learning_rate=config.learning_rate,
                w_min=config.w_min,
                w_max=config.w_max,
                dt=config.dt_ms,
            )

            # Declare BCM learning strategies with Optional type
            self.bcm_l4: Optional[LearningStrategy] = None
            self.bcm_l23: Optional[LearningStrategy] = None
            self.bcm_l5: Optional[LearningStrategy] = None
            self.bcm_l6a: Optional[LearningStrategy] = None
            self.bcm_l6b: Optional[LearningStrategy] = None

            # Use create_cortex_strategy() factory helper for STDP+BCM composite
            self.bcm_l4 = create_cortex_strategy(
                use_stdp=True, stdp_config=stdp_cfg, bcm_config=bcm_cfg
            )
            self.bcm_l23 = create_cortex_strategy(
                use_stdp=True, stdp_config=stdp_cfg, bcm_config=bcm_cfg
            )
            self.bcm_l5 = create_cortex_strategy(
                use_stdp=True, stdp_config=stdp_cfg, bcm_config=bcm_cfg
            )
            self.bcm_l6a = create_cortex_strategy(
                use_stdp=True, stdp_config=stdp_cfg, bcm_config=bcm_cfg
            )
            self.bcm_l6b = create_cortex_strategy(
                use_stdp=True, stdp_config=stdp_cfg, bcm_config=bcm_cfg
            )

        # State
        self.state: LayeredCortexState = LayeredCortexState()  # type: ignore[assignment]

        # Theta phase for encoding/retrieval modulation
        self._theta_phase = 0.0

        # Cumulative spike counters (for diagnostics across timesteps)
        self._cumulative_l4_spikes = 0
        self._cumulative_l23_spikes = 0
        self._cumulative_l5_spikes = 0
        self._cumulative_l6a_spikes = 0
        self._cumulative_l6b_spikes = 0

        # Intrinsic plasticity tracking (initialized in _init_layers)
        self._l23_threshold_offset: Optional[torch.Tensor] = None
        self._l23_activity_history: Optional[torch.Tensor] = None

        # Homeostasis for synaptic scaling and intrinsic plasticity
        homeostasis_config = UnifiedHomeostasisConfig(
            weight_budget=config.weight_budget * self.input_size,
            w_min=config.w_min,
            w_max=config.w_max,
            soft_normalization=config.soft_normalization,
            normalization_rate=config.normalization_rate,
            activity_target=config.activity_target,
            device=str(self.device),
        )
        self.homeostasis = UnifiedHomeostasis(homeostasis_config)

        # =====================================================================
        # ROBUSTNESS MECHANISMS (from RobustnessConfig)
        # =====================================================================
        self._init_robustness_mechanisms()
        self._init_gamma_attention()

    def _initialize_weights(self) -> torch.Tensor:
        """Placeholder - real weights in _init_weights."""
        actual_out = cast(int, self._actual_output)
        return nn.Parameter(torch.zeros(actual_out, self.input_size))

    def _create_neurons(self):
        """Placeholder - neurons created in _init_layers."""
        return None

    def _init_layers(self) -> None:
        """Initialize conductance-based LIF neurons for each layer.

        Uses ConductanceLIF for biologically realistic gain control:
        - Separate excitatory and inhibitory conductances
        - Shunting inhibition (divisive effect)
        - Natural saturation at reversal potentials
        - No need for artificial divisive normalization

        **Layer-Specific Heterogeneity (Phase 2A)**:
        If config.use_layer_heterogeneity=True, applies distinct electrophysiological
        properties per layer (tau_mem, v_threshold, adaptation) reflecting biological
        diversity of cortical cell types.
        """
        cfg = self.config

        # =====================================================================
        # LAYER-SPECIFIC HETEROGENEITY (Phase 2A)
        # =====================================================================
        # Prepare layer-specific overrides if heterogeneity enabled
        l4_overrides = {}
        l23_overrides = {}
        l5_overrides = {}
        l6a_overrides = {}
        l6b_overrides = {}

        if cfg.use_layer_heterogeneity:
            # L4: Fast sensory processing
            l4_overrides = {
                "tau_mem": cfg.layer_tau_mem["l4"],
                "v_threshold": cfg.layer_v_threshold["l4"],
                "adapt_increment": cfg.layer_adaptation["l4"],
            }

            # L2/3: Integration and association
            l23_overrides = {
                "tau_mem": cfg.layer_tau_mem["l23"],
                "v_threshold": cfg.layer_v_threshold["l23"],
                "adapt_increment": cfg.layer_adaptation["l23"],
            }

            # L5: Output generation
            l5_overrides = {
                "tau_mem": cfg.layer_tau_mem["l5"],
                "v_threshold": cfg.layer_v_threshold["l5"],
                "adapt_increment": cfg.layer_adaptation["l5"],
            }

            # L6a: TRN feedback (low gamma)
            l6a_overrides = {
                "tau_mem": cfg.layer_tau_mem["l6a"],
                "v_threshold": cfg.layer_v_threshold["l6a"],
                "adapt_increment": cfg.layer_adaptation["l6a"],
            }

            # L6b: Relay feedback (high gamma)
            l6b_overrides = {
                "tau_mem": cfg.layer_tau_mem["l6b"],
                "v_threshold": cfg.layer_v_threshold["l6b"],
                "adapt_increment": cfg.layer_adaptation["l6b"],
            }

        # Create layer-specific neurons using factory functions with heterogeneous properties
        self.l4_neurons = create_cortical_layer_neurons(
            self.l4_size, "L4", self.device, **l4_overrides
        )
        self.l23_neurons = create_cortical_layer_neurons(
            self.l23_size,
            "L2/3",
            self.device,
            adapt_increment=(
                cfg.adapt_increment
                if not cfg.use_layer_heterogeneity
                else l23_overrides["adapt_increment"]
            ),
            tau_adapt=cfg.adapt_tau,
            **(
                {}
                if not cfg.use_layer_heterogeneity
                else {k: v for k, v in l23_overrides.items() if k != "adapt_increment"}
            ),
        )
        self.l5_neurons = create_cortical_layer_neurons(
            self.l5_size, "L5", self.device, **l5_overrides
        )

        # L6 split into two subtypes:
        # - L6a (corticothalamic type I): Projects to TRN (inhibitory modulation)
        # - L6b (corticothalamic type II): Projects to relay (excitatory modulation)
        self.l6a_neurons = create_cortical_layer_neurons(
            self.l6a_size, "L6a", self.device, **l6a_overrides
        )
        self.l6b_neurons = create_cortical_layer_neurons(
            self.l6b_size, "L6b", self.device, **l6b_overrides
        )

        # =====================================================================
        # SHORT-TERM PLASTICITY for L2/3 recurrent connections
        # =====================================================================
        # L2/3 recurrent connections show SHORT-TERM DEPRESSION, preventing
        # frozen attractors. Without STD, the same neurons fire every timestep.
        # Always enabled (critical for preventing frozen attractors)
        self.stp_l23_recurrent = ShortTermPlasticity(
            n_pre=self.l23_size,
            n_post=self.l23_size,
            config=STPConfig.from_type(STPType.DEPRESSING_FAST, dt=cfg.dt_ms),
            per_synapse=True,
        )
        self.stp_l23_recurrent.to(self.device)

        # =====================================================================
        # GAP JUNCTIONS for L2/3 interneuron synchronization
        # =====================================================================
        # Basket cells and chandelier cells have dense gap junction networks
        # Critical for cortical gamma oscillations (30-80 Hz) and precise timing
        # ~70-80% of cortical gap junctions are interneuron-interneuron (Bennett 2004)
        self.gap_junctions_l23: Optional[GapJunctionCoupling] = None
        if cfg.gap_junctions_enabled:
            gap_config = GapJunctionConfig(
                enabled=True,
                coupling_strength=cfg.gap_junction_strength,
                connectivity_threshold=cfg.gap_junction_threshold,
                max_neighbors=cfg.gap_junction_max_neighbors,
                interneuron_only=True,  # Only couple inhibitory neurons
            )

            # Use l23_inhib weights to define functional neighborhoods
            # Interneurons that inhibit similar pyramidal cells are anatomically close
            # Note: We'll initialize this after weights are created (see _init_weights)
            # For now, store config for later initialization
            self._gap_config_l23 = gap_config

        # =====================================================================
        # INTER-LAYER AXONAL DELAYS
        # =====================================================================
        # Create delay buffers for biological signal propagation within layers
        # L4→L2/3 delay: Short vertical projection (~2ms biologically)
        # L2/3→L5: Longer vertical projection (~2ms biologically)
        # L2/3→L6a/L6b: Within column (~2ms biologically)
        # Uses circular buffer mechanism from AxonalDelaysMixin
        self._l4_l23_delay_steps = int(cfg.l4_to_l23_delay_ms / cfg.dt_ms)
        self._l23_l5_delay_steps = int(cfg.l23_to_l5_delay_ms / cfg.dt_ms)
        self._l23_l6a_delay_steps = int(cfg.l23_to_l6a_delay_ms / cfg.dt_ms)
        self._l23_l6b_delay_steps = int(cfg.l23_to_l6b_delay_ms / cfg.dt_ms)

        # Initialize delay buffers (lazily initialized on first use)
        self._l4_l23_delay_buffer: Optional[torch.Tensor] = None
        self._l4_l23_delay_ptr: int = 0
        self._l23_l5_delay_buffer: Optional[torch.Tensor] = None
        self._l23_l5_delay_ptr: int = 0
        self._l23_l6a_delay_buffer: Optional[torch.Tensor] = None
        self._l23_l6a_delay_ptr: int = 0
        self._l23_l6b_delay_buffer: Optional[torch.Tensor] = None
        self._l23_l6b_delay_ptr: int = 0

    def _init_robustness_mechanisms(self) -> None:
        """Initialize robustness mechanisms from RobustnessConfig.

        These mechanisms are cortex-specific and NOT redundant with UnifiedHomeostasis:
        - E/I Balance: Critical for recurrent cortical stability

        Note: Activity regulation is handled by UnifiedHomeostasis.
        ConductanceLIF neurons provide natural gain control via shunting inhibition,
        so divisive normalization is not needed.
        """
        cfg = self.config
        rob = cfg.robustness
        device = self.device  # Use self.device, not cfg.device (doesn't exist)

        # Default: no robustness mechanisms
        self.ei_balance: Optional[LayerEIBalance] = None

        if rob is None:
            return

        # E/I Balance Regulator for L2/3 layer
        # Tracks excitatory (L2/3 pyramidal) vs inhibitory (lateral inhibition) activity
        if rob.enable_ei_balance:
            self.ei_balance = LayerEIBalance(
                n_exc=self.l23_size,
                n_inh=self.l23_size,  # Approximation: use L2/3 size for inhibition
                config=rob.ei_balance,
                device=device,
            )

    def _init_gamma_attention(self) -> None:
        """Initialize gamma-based attention (spike-native phase gating for L2/3).

        Uses centralized gamma oscillator from Brain (via set_oscillator_phases).
        Regions don't create their own oscillators - they receive phases from Brain.
        Always enabled for spike-native attention.
        """
        cfg = self.config
        device = self.device  # Use self.device, not cfg.device (doesn't exist)

        # Learnable phase preferences for each L2/3 neuron
        self.l23_phase_prefs = nn.Parameter(
            initialize_phase_preferences(self.l23_size, device=device)
        )
        self.gamma_attention_width = cfg.gamma_attention_width

    def _init_weights(self) -> None:
        """Initialize inter-layer weight matrices.

        Feedforward weights use positive initialization to ensure sparse
        presynaptic activity can drive postsynaptic neurons above threshold.
        With ~10-15% sparsity, we need weights scaled so that:
            sum(w_ij * spike_j) * strength ~ threshold

        Using uniform [0, max] with max scaled by fan-in and expected sparsity.
        """
        device = (
            self.device
        )  # Use self.device (torch.device), not self.config.device (doesn't exist)
        cfg = self.config

        # Expected number of active inputs given sparsity
        expected_active_l4 = max(1, int(self.l4_size * cfg.l4_sparsity))
        expected_active_l23 = max(1, int(self.l23_size * cfg.l23_sparsity))

        # Feedforward weights: positive, scaled so sparse input reaches threshold
        # With n_active inputs, threshold ~1.0, strength factor applied later:
        # target = threshold / (n_active * strength) ≈ 1.0 / (n_active * strength)
        # We initialize to mean ≈ target, with some variance for diversity

        # Input → L4: positive excitatory weights (EXTERNAL - moved to synaptic_weights)
        w_scale_input = 1.0 / max(1, int(self.input_size * 0.15))  # Assume 15% input sparsity
        input_weights = torch.abs(
            WeightInitializer.gaussian(
                n_output=self.l4_size,
                n_input=self.input_size,
                mean=0.0,
                std=w_scale_input,
                device=device,
            )
        )
        # Register external input source (NeuralRegion pattern)
        # Directly register in dicts without add_input_source to avoid device issues
        self.synaptic_weights["input"] = nn.Parameter(input_weights)
        self.input_sources["input"] = self.input_size

        # L4 → L2/3: positive excitatory weights (AT L2/3 DENDRITES)
        w_scale_l4_l23 = 1.0 / expected_active_l4
        l4_l23_weights = torch.abs(
            WeightInitializer.gaussian(
                n_output=self.l23_size,
                n_input=self.l4_size,
                mean=0.0,
                std=w_scale_l4_l23,
                device=device,
            )
        )
        self.synaptic_weights["l4_l23"] = nn.Parameter(l4_l23_weights)

        # L2/3 recurrent: SIGNED weights (compact E/I approximation) (AT L2/3 DENDRITES)
        # Unlike feedforward connections which are positive-only (Dale's law),
        # recurrent lateral connections use signed weights to approximate the
        # mixed excitatory/inhibitory microcircuit within a cortical layer.
        # Positive weights = local excitation, negative weights = lateral inhibition.
        # Uses dedicated bounds [l23_recurrent_w_min, l23_recurrent_w_max] during learning.
        l23_recurrent_weights = WeightInitializer.gaussian(
            n_output=self.l23_size, n_input=self.l23_size, mean=0.0, std=0.2, device=device
        )
        l23_recurrent_weights.fill_diagonal_(0.0)
        self.synaptic_weights["l23_recurrent"] = nn.Parameter(l23_recurrent_weights)

        # L2/3 → L5: positive excitatory weights (AT L5 DENDRITES)
        w_scale_l23_l5 = 1.0 / expected_active_l23
        l23_l5_weights = torch.abs(
            WeightInitializer.gaussian(
                n_output=self.l5_size,
                n_input=self.l23_size,
                mean=0.0,
                std=w_scale_l23_l5,
                device=device,
            )
        )
        self.synaptic_weights["l23_l5"] = nn.Parameter(l23_l5_weights)

        # L2/3 → L6a: positive excitatory weights (corticothalamic type I → TRN) (AT L6a DENDRITES)
        w_scale_l23_l6a = 1.0 / expected_active_l23
        l23_l6a_weights = torch.abs(
            WeightInitializer.gaussian(
                n_output=self.l6a_size,
                n_input=self.l23_size,
                mean=0.0,
                std=w_scale_l23_l6a,
                device=device,
            )
        )
        self.synaptic_weights["l23_l6a"] = nn.Parameter(l23_l6a_weights)

        # L2/3 → L6b: positive excitatory weights (corticothalamic type II → relay) (AT L6b DENDRITES)
        w_scale_l23_l6b = 1.0 / expected_active_l23
        l23_l6b_weights = torch.abs(
            WeightInitializer.gaussian(
                n_output=self.l6b_size,
                n_input=self.l23_size,
                mean=0.0,
                std=w_scale_l23_l6b,
                device=device,
            )
        )
        self.synaptic_weights["l23_l6b"] = nn.Parameter(l23_l6b_weights)

        # L2/3 inhibition: positive (inhibitory connections suppress) (AT L2/3 DENDRITES)
        l23_inhib_weights = (
            WeightInitializer.ones(n_output=self.l23_size, n_input=self.l23_size, device=device)
            * 0.3
        )
        l23_inhib_weights.fill_diagonal_(0.0)
        self.synaptic_weights["l23_inhib"] = nn.Parameter(l23_inhib_weights)

        # =====================================================================
        # GAP JUNCTIONS (L2/3 interneurons) - Initialize after weights created
        # =====================================================================
        # Now that l23_inhib weights exist, create gap junction network
        # Interneurons that share inhibitory targets are anatomically close
        if hasattr(self, "_gap_config_l23") and self._gap_config_l23.enabled:
            self.gap_junctions_l23 = GapJunctionCoupling(
                n_neurons=self.l23_size,
                afferent_weights=self.synaptic_weights["l23_inhib"],  # Shared targets → proximity
                config=self._gap_config_l23,
                interneuron_mask=None,  # All L2/3 lateral neurons treated as interneurons
                device=self.device,
            )

        # Note: L6 → TRN weights are not stored in cortex.
        # They will be created and managed by the thalamus component
        # because TRN is part of thalamic circuitry.
        # The brain's connection system will wire L6 spikes to thalamus input.

        # Main weights reference (for compatibility with base class)
        self.weights = self.synaptic_weights["input"]

        # Note: Learning strategies (STDP+BCM) are created in __init__ as
        # self.bcm_l4, self.bcm_l23, self.bcm_l5 composite strategies

    def _reset_subsystems(self, *names: str) -> None:
        """Reset state of named subsystems that have reset_state() method.

        Helper for backward compatibility with LearnableComponent pattern.
        """
        for name in names:
            if hasattr(self, name):
                subsystem = getattr(self, name)
                if subsystem is not None and hasattr(subsystem, "reset_state"):
                    subsystem.reset_state()

    def _reset_scalars(self, **scalar_values: Any) -> None:
        """Reset scalar attributes to specified values.

        Helper for backward compatibility with LearnableComponent pattern.
        """
        for name, value in scalar_values.items():
            setattr(self, name, value)

    def reset_state(self) -> None:
        """Reset all layer states.

        ADR-005: Uses 1D tensors (no batch dimension) for single-brain architecture.
        """
        dev = self.device

        # Reset neuron populations and STP using helpers
        self._reset_subsystems(
            "l4_neurons",
            "l23_neurons",
            "l5_neurons",
            "l6a_neurons",
            "l6b_neurons",
            "stp_l23_recurrent",
        )

        # Note: No local oscillators to reset - phases come from Brain

        # Preserve oscillator signals if they exist (set via set_oscillator_phases)
        existing_phases = getattr(self.state, "_oscillator_phases", {})
        existing_signals = getattr(self.state, "_oscillator_signals", {})

        self.state = LayeredCortexState(
            l4_spikes=torch.zeros(self.l4_size, device=dev),
            l23_spikes=torch.zeros(self.l23_size, device=dev),
            l5_spikes=torch.zeros(self.l5_size, device=dev),
            l6a_spikes=torch.zeros(self.l6a_size, device=dev),
            l6b_spikes=torch.zeros(self.l6b_size, device=dev),
            l23_recurrent_activity=torch.zeros(self.l23_size, device=dev),
            l4_trace=torch.zeros(self.l4_size, device=dev),
            l23_trace=torch.zeros(self.l23_size, device=dev),
            l5_trace=torch.zeros(self.l5_size, device=dev),
            l6a_trace=torch.zeros(self.l6a_size, device=dev),
            l6b_trace=torch.zeros(self.l6b_size, device=dev),
            top_down_modulation=None,
            ffi_strength=0.0,
            l23_membrane=torch.zeros(self.l23_size, device=dev),  # For gap junction coupling
        )

        # Restore/initialize oscillator signals (stored as internal attributes, not in state)
        # These are managed separately and don't need to be in the state dataclass

        # Reset cumulative spike counters using helper
        self._reset_scalars(
            _cumulative_l4_spikes=0,
            _cumulative_l23_spikes=0,
            _cumulative_l5_spikes=0,
            _cumulative_l6a_spikes=0,
            _cumulative_l6b_spikes=0,
        )

        # Initialize all delay buffers if delays are configured
        if self._l4_l23_delay_steps > 0:
            self._l4_l23_delay_buffer = torch.zeros(
                self._l4_l23_delay_steps, self.l4_size, device=dev, dtype=torch.bool
            )
            self._l4_l23_delay_ptr = 0

        if self._l23_l5_delay_steps > 0:
            self._l23_l5_delay_buffer = torch.zeros(
                self._l23_l5_delay_steps, self.l23_size, device=dev, dtype=torch.bool
            )
            self._l23_l5_delay_ptr = 0

        # L6a delay buffer (L2/3 → L6a)
        if self._l23_l6a_delay_steps > 0:
            self._l23_l6a_delay_buffer = torch.zeros(
                self._l23_l6a_delay_steps, self.l23_size, device=dev, dtype=torch.bool
            )
            self._l23_l6a_delay_ptr = 0

        # L6b delay buffer (L2/3 → L6b)
        if self._l23_l6b_delay_steps > 0:
            self._l23_l6b_delay_buffer = torch.zeros(
                self._l23_l6b_delay_steps, self.l23_size, device=dev, dtype=torch.bool
            )
            self._l23_l6b_delay_ptr = 0

        # Note: FFI state decays naturally, no hard reset needed

    def set_oscillator_phases(
        self,
        phases: Dict[str, float],
        signals: Optional[Dict[str, float]] = None,
        theta_slot: int = 0,
        coupled_amplitudes: Optional[Dict[str, float]] = None,
    ) -> None:
        """Set oscillator phases and effective amplitudes for attention gating.

        Stores oscillator state in LayeredCortexState for use during forward().
        Alpha signals gate input processing (early suppression).
        Gamma effective amplitude modulates learning (automatic multiplicative coupling).

        Args:
            phases: Dict mapping oscillator name ('alpha', 'theta', etc.) to phase [0, 2π)
            signals: Dict mapping oscillator name to signal magnitude [-1, 1]
            theta_slot: Current theta slot [0, n_slots-1] for sequence encoding
            coupled_amplitudes: Effective amplitudes per oscillator (pre-computed)
        """
        # Use base mixin implementation to store all oscillator data
        super().set_oscillator_phases(phases, signals, theta_slot, coupled_amplitudes)

    # region Growth and Neurogenesis

    def grow_output(
        self,
        n_new: int,
        initialization: str = "sparse_random",
        sparsity: float = 0.1,
    ) -> None:
        """Grow output dimension by expanding all layers proportionally.

        This expands L2/3 and L5 (output layers) by n_new neurons total,
        and grows internal layers (L4, L6a, L6b) proportionally to maintain
        current architecture ratios.

        All inter-layer weights are expanded to accommodate new neurons.

        Args:
            n_new: Number of neurons to add to output (L2/3 + L5)
            initialization: Weight initialization strategy
            sparsity: Sparsity for new connections
        """
        # Distribute n_new across output layers (L2/3 and L5) proportionally
        output_total = self.l23_size + self.l5_size
        l23_growth = int(n_new * self.l23_size / output_total)
        l5_growth = n_new - l23_growth  # Ensure exact n_new growth

        # Grow internal layers (L4, L6) proportionally to L2/3 growth
        # (they support the output layers)
        growth_ratio = l23_growth / self.l23_size if self.l23_size > 0 else 0
        l4_growth = int(self.l4_size * growth_ratio)
        l6a_growth = int(self.l6a_size * growth_ratio)
        l6b_growth = int(self.l6b_size * growth_ratio)

        old_l4_size = self.l4_size
        old_l23_size = self.l23_size
        old_l5_size = self.l5_size
        old_l6a_size = self.l6a_size
        old_l6b_size = self.l6b_size

        new_l4_size = old_l4_size + l4_growth
        new_l23_size = old_l23_size + l23_growth
        new_l5_size = old_l5_size + l5_growth
        new_l6a_size = old_l6a_size + l6a_growth
        new_l6b_size = old_l6b_size + l6b_growth

        # 1. Expand input→L4 weights [l4, input]
        # Add rows for new L4 neurons using helper
        expanded_input = self._grow_weight_matrix_rows(
            self.synaptic_weights["input"].data,
            l4_growth,
            initializer=initialization,
            sparsity=sparsity,
        )
        self.synaptic_weights["input"] = nn.Parameter(expanded_input)

        # 2. Expand L4→L2/3 weights [l23, l4]
        # Add rows for new L2/3 neurons, columns for new L4 neurons
        expanded_l23_rows = self._grow_weight_matrix_rows(
            self.synaptic_weights["l4_l23"].data,
            l23_growth,
            initializer=initialization,
            sparsity=sparsity,
        )
        self.synaptic_weights["l4_l23"] = nn.Parameter(
            self._grow_weight_matrix_cols(
                expanded_l23_rows, l4_growth, initializer=initialization, sparsity=sparsity
            )
        )

        # 3. Expand L2/3→L2/3 recurrent weights [l23, l23]
        # Add rows and columns for new L2/3 neurons
        expanded_recurrent_rows = self._grow_weight_matrix_rows(
            self.synaptic_weights["l23_recurrent"].data,
            l23_growth,
            initializer=initialization,
            sparsity=sparsity,
        )
        self.synaptic_weights["l23_recurrent"] = nn.Parameter(
            self._grow_weight_matrix_cols(
                expanded_recurrent_rows, l23_growth, initializer=initialization, sparsity=sparsity
            )
        )

        # 3b. Expand L2/3 inhibitory weights [l23, l23]
        # Same structure as recurrent, but negative weights for inhibition
        expanded_inhib_rows = -torch.abs(
            self._grow_weight_matrix_rows(
                self.synaptic_weights["l23_inhib"].data,
                l23_growth,
                initializer=initialization,
                sparsity=sparsity,
            )
        )
        self.synaptic_weights["l23_inhib"].data = -torch.abs(
            self._grow_weight_matrix_cols(
                expanded_inhib_rows, l23_growth, initializer=initialization, sparsity=sparsity
            )
        )
        # Zero out diagonal (no self-inhibition)
        self.synaptic_weights["l23_inhib"].data.fill_diagonal_(0.0)

        # 4. Expand L2/3→L5 weights [l5, l23]
        # Add rows for new L5 neurons, columns for new L2/3 neurons
        expanded_l5_rows = self._grow_weight_matrix_rows(
            self.synaptic_weights["l23_l5"].data,
            l5_growth,
            initializer=initialization,
            sparsity=sparsity,
        )
        self.synaptic_weights["l23_l5"] = nn.Parameter(
            self._grow_weight_matrix_cols(
                expanded_l5_rows, l23_growth, initializer=initialization, sparsity=sparsity
            )
        )

        # 4b. Expand L2/3→L6a weights [l6a, l23]
        # Add rows for new L6a neurons, columns for new L2/3 neurons
        expanded_l6a_rows = self._grow_weight_matrix_rows(
            self.synaptic_weights["l23_l6a"].data,
            l6a_growth,
            initializer=initialization,
            sparsity=sparsity,
        )
        self.synaptic_weights["l23_l6a"] = nn.Parameter(
            self._grow_weight_matrix_cols(
                expanded_l6a_rows, l23_growth, initializer=initialization, sparsity=sparsity
            )
        )

        # 4c. Expand L2/3→L6b weights [l6b, l23]
        # Add rows for new L6b neurons, columns for new L2/3 neurons
        expanded_l6b_rows = self._grow_weight_matrix_rows(
            self.synaptic_weights["l23_l6b"].data,
            l6b_growth,
            initializer=initialization,
            sparsity=sparsity,
        )
        self.synaptic_weights["l23_l6b"] = nn.Parameter(
            self._grow_weight_matrix_cols(
                expanded_l6b_rows, l23_growth, initializer=initialization, sparsity=sparsity
            )
        )

        # Update main weights reference (for base class compatibility)
        self.weights = self.synaptic_weights["l23_l5"]

        # 5. Expand neurons for all layers using factory functions
        # Apply layer-specific heterogeneity if enabled (Phase 2A)
        cfg = self.config
        l4_overrides = {}
        l23_overrides = {}
        l5_overrides = {}
        l6a_overrides = {}
        l6b_overrides = {}

        if cfg.use_layer_heterogeneity:
            l4_overrides = {
                "tau_mem": cfg.layer_tau_mem["l4"],
                "v_threshold": cfg.layer_v_threshold["l4"],
                "adapt_increment": cfg.layer_adaptation["l4"],
            }
            l23_overrides = {
                "tau_mem": cfg.layer_tau_mem["l23"],
                "v_threshold": cfg.layer_v_threshold["l23"],
                "adapt_increment": cfg.layer_adaptation["l23"],
            }
            l5_overrides = {
                "tau_mem": cfg.layer_tau_mem["l5"],
                "v_threshold": cfg.layer_v_threshold["l5"],
                "adapt_increment": cfg.layer_adaptation["l5"],
            }
            l6a_overrides = {
                "tau_mem": cfg.layer_tau_mem["l6a"],
                "v_threshold": cfg.layer_v_threshold["l6a"],
                "adapt_increment": cfg.layer_adaptation["l6a"],
            }
            l6b_overrides = {
                "tau_mem": cfg.layer_tau_mem["l6b"],
                "v_threshold": cfg.layer_v_threshold["l6b"],
                "adapt_increment": cfg.layer_adaptation["l6b"],
            }

        self.l4_size = new_l4_size
        self.l4_neurons = create_cortical_layer_neurons(
            self.l4_size, "L4", self.device, **l4_overrides
        )

        self.l23_size = new_l23_size
        self.l23_neurons = create_cortical_layer_neurons(
            self.l23_size,
            "L2/3",
            self.device,
            adapt_increment=(
                self.config.adapt_increment
                if not cfg.use_layer_heterogeneity
                else l23_overrides["adapt_increment"]
            ),
            tau_adapt=self.config.adapt_tau,
            **(
                {}
                if not cfg.use_layer_heterogeneity
                else {k: v for k, v in l23_overrides.items() if k != "adapt_increment"}
            ),
        )

        self.l5_size = new_l5_size
        self.l5_neurons = create_cortical_layer_neurons(
            self.l5_size, "L5", self.device, **l5_overrides
        )

        self.l6a_size = new_l6a_size
        self.l6a_neurons = create_cortical_layer_neurons(
            self.l6a_size, "L6a", self.device, **l6a_overrides
        )

        self.l6b_size = new_l6b_size
        self.l6b_neurons = create_cortical_layer_neurons(
            self.l6b_size, "L6b", self.device, **l6b_overrides
        )

        # 6. Update STP module to match L2/3 growth
        self.stp_l23_recurrent = ShortTermPlasticity(
            n_pre=new_l23_size,
            n_post=new_l23_size,
            config=STPConfig.from_type(STPType.DEPRESSING_FAST, dt=self.config.dt_ms),
            per_synapse=True,
        )
        self.stp_l23_recurrent.to(self.device)

        # 6b. Expand L2/3 phase preferences for gamma attention
        new_phase_prefs = initialize_phase_preferences(l23_growth, device=self.device)
        self.l23_phase_prefs.data = torch.cat([self.l23_phase_prefs.data, new_phase_prefs])

        # 6c. Recreate gap junctions if enabled (coupling matrix size must match L2/3)
        if self.gap_junctions_l23 is not None:
            self.gap_junctions_l23 = GapJunctionCoupling(
                n_neurons=new_l23_size,
                afferent_weights=self.synaptic_weights["l23_inhib"],  # Use updated weights
                config=GapJunctionConfig(
                    coupling_strength=self.config.gap_junction_strength,
                ),
                interneuron_mask=None,
                device=self.device,
            )

        # 7. Update output size (structural parameters - no longer in config)
        new_total_output = new_l23_size + new_l5_size
        old_total_output = old_l23_size + old_l5_size

        # Instance variables already updated above (l4_size, l23_size, etc.)
        # No config update needed - config contains only behavioral params

        # 8. Validate growth manually (standard validation doesn't apply to multi-layer growth)
        # Cortex grows L2/3 + L5 by exactly n_new, but L4/L6 grow proportionally
        # Verify: actual output growth matches expected
        actual_growth = new_total_output - old_total_output
        assert actual_growth == n_new, (
            f"Cortex growth mismatch: expected {n_new} output neurons, "
            f"but grew by {actual_growth} (old={old_total_output}, new={new_total_output})"
        )

    def grow_layer(
        self,
        layer_name: str,
        n_new: int,
        initialization: str = "sparse_random",
        sparsity: float = 0.1,
    ) -> None:
        """Grow specific cortical layer (SEMANTIC API).

        Args:
            layer_name: Layer to grow ('L4', 'L23', 'L5', 'L6A', 'L6B', or 'output')
            n_new: Number of neurons to add
            initialization: Weight init strategy
            sparsity: Connection sparsity

        Note:
            Use layer_name='output' to grow L2/3 and L5 proportionally (recommended).
            Direct growth of individual layers not yet supported.
        """
        if layer_name.lower() in ["output", "l23+l5"]:
            self.grow_output(n_new, initialization, sparsity)
        else:
            raise NotImplementedError(
                f"Direct growth of {layer_name} not yet supported. "
                f"Use grow_layer('output', n) to grow L2/3 + L5 proportionally."
            )

    def grow_input(
        self,
        n_new: int,
        initialization: str = "sparse_random",
        sparsity: float = 0.1,
    ) -> None:
        """Grow cortex input dimension when upstream region grows.

        When an upstream region (e.g., thalamus, sensory pathway) adds neurons,
        this method expands the cortex's input weights to accommodate the larger
        input dimension.

        This is the CRITICAL missing piece that enables full dynamic growth!

        Args:
            n_new: Number of input neurons to add
            initialization: Weight init strategy ('sparse_random', 'xavier', 'uniform')
            sparsity: Connection sparsity for new input neurons (if sparse_random)

        Example:
            >>> # Visual pathway grows from 784 → 804 neurons
            >>> visual_pathway.grow_output(20)
            >>> # Cortex must expand input dimension
            >>> cortex.grow_input(20)  # w_input_l4: [l4, 784] → [l4, 804]

        Implementation:
            Expands w_input_l4 weight matrix by adding COLUMNS:
            - Old: [l4_size, old_n_input]
            - New: [l4_size, old_n_input + n_new]
            - Preserves existing learned weights in left columns
            - Initializes new input weights small to avoid disruption
        """
        old_n_input = self.input_size
        new_n_input = old_n_input + n_new

        # Expand input→L4 weights [l4, input] → [l4, input+n_new]
        # Add COLUMNS for new input neurons
        self.synaptic_weights["input"] = nn.Parameter(
            self._grow_weight_matrix_cols(
                self.synaptic_weights["input"].data,
                n_new,
                initializer=initialization,
                sparsity=sparsity,
            )
        )

        # Update input size (structural parameter - no longer in config)
        self.input_size = new_n_input

        # Validate growth completed correctly
        self._validate_input_growth(old_n_input, n_new)

    # endregion

    # region Forward Pass (L4→L2/3→L5)

    def forward(
        self,
        inputs: Union[Dict[str, torch.Tensor], torch.Tensor],
        top_down: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Process input through layered cortical circuit with continuous plasticity.

        This method both processes spikes AND applies synaptic plasticity. Learning
        happens continuously at each timestep, modulated by neuromodulators (dopamine).
        This is how biological cortex works - plasticity is part of the dynamics,
        not a separate training phase.

        Args:
            inputs: Input spikes - dict {"input": tensor} or tensor [n_input] (1D per ADR-005)
            top_down: Optional top-down modulation [l23_size] (1D)

        Returns:
            Output spikes [l23_size + l5_size] - concatenated L2/3 and L5 outputs (1D)

        Note:
            Theta modulation and timestep (dt_ms) computed internally from config
        """
        # Concatenate all input sources in consistent order
        # Common sources: thalamus (feedforward), hippocampus (memory), cerebellum (predictions), pfc (top-down)
        # Supports zero-input execution for clock-driven architecture
        input_spikes = InputRouter.concatenate_sources(
            inputs,
            component_name="LayeredCortex",
            n_input=self.input_size,
            device=self.device,
        )

        # Get timestep from config for temporal dynamics
        dt = self.config.dt_ms

        # Compute theta modulation from oscillator phase (set by Brain)
        # encoding_mod: high at theta peak (0°), low at trough (180°)
        # retrieval_mod: low at theta peak, high at trough
        encoding_mod, retrieval_mod = compute_theta_encoding_retrieval(self._theta_phase)

        # ADR-005: Expect 1D tensors (no batch dimension)
        assert input_spikes.dim() == 1, (
            f"LayeredCortex.forward: Expected 1D input (ADR-005), got shape {input_spikes.shape}. "
            f"Thalia uses single-brain architecture with no batch dimension."
        )
        assert input_spikes.shape[0] == self.input_size, (
            f"LayeredCortex.forward: input_spikes has shape {input_spikes.shape} "
            f"but input_size={self.input_size}. Check that input matches cortex config."
        )

        if top_down is not None:
            assert (
                top_down.dim() == 1
            ), f"LayeredCortex.forward: Expected 1D top_down (ADR-005), got shape {top_down.shape}"
            assert top_down.shape[0] == self.l23_size, (
                f"LayeredCortex.forward: top_down has shape {top_down.shape} "
                f"but L2/3 size={self.l23_size}. Top-down must match L2/3 for modulation."
            )

        if self.state.l4_spikes is None:
            self.reset_state()

        cfg = self.config

        # =====================================================================
        # ALPHA-BASED ATTENTION GATING
        # =====================================================================
        # Alpha oscillations (8-13 Hz) suppress processing in cortical areas.
        # High alpha = attention directed elsewhere (suppress this region)
        # Low alpha = attention focused here (normal processing)
        #
        # Biological basis: Alpha power is inversely related to cortical
        # excitability. Regions with high alpha are "idling" or suppressed
        # to prevent interference with attended regions.
        alpha_suppression = 1.0  # Default: no suppression
        gamma_modulation = 1.0  # Default: no gamma modulation

        if (
            hasattr(self.state, "_oscillator_signals")
            and self.state._oscillator_signals is not None
        ):
            alpha_signal = self.state._oscillator_signals.get("alpha", 0.0)

            # Alpha signal ranges [-1, 1], convert to suppression [0, 0.5]
            # High positive alpha (near 1.0) → max suppression (50%)
            # Low/negative alpha → minimal suppression
            alpha_magnitude = max(0.0, alpha_signal)  # Only positive values
            alpha_suppression = 1.0 - (alpha_magnitude * 0.5)  # Scale to 50-100%

            # Automatic gamma modulation: ALL slower oscillators affect gamma
            # This gives emergent multi-oscillator coupling (e.g., theta-alpha-beta-gamma)
            gamma_modulation = self._gamma_amplitude_effective

            # Store for diagnostics (alpha_suppression only, gamma_modulation is internal)
            self.state.alpha_suppression = alpha_suppression

        # Apply alpha suppression to input (early gating)
        gated_input_spikes = input_spikes * alpha_suppression

        # L4: Input processing with conductance-based neurons
        # Compute excitatory conductance from input (using synaptic_weights)
        # Dense matmul for all inputs
        l4_g_exc = (
            torch.matmul(self.synaptic_weights["input"], gated_input_spikes.float())
            * cfg.input_to_l4_strength
        )
        l4_g_exc = l4_g_exc * (L4_INPUT_ENCODING_SCALE + L4_INPUT_ENCODING_SCALE * encoding_mod)

        # Inhibitory conductance: ~25% of excitatory (4:1 E/I ratio)
        # This provides feedback inhibition for gain control
        l4_g_inh = l4_g_exc * 0.25

        # =====================================================================
        # NOREPINEPHRINE GAIN MODULATION (Locus Coeruleus)
        # =====================================================================
        # High NE (arousal/uncertainty): Increase gain → more responsive
        # Low NE (baseline): Normal gain
        # Biological: β-adrenergic receptors increase neuronal excitability
        ne_level = self.state.norepinephrine
        # NE gain: 1.0 (baseline) to 1.5 (high arousal)
        ne_gain = compute_ne_gain(ne_level)
        l4_g_exc = l4_g_exc * ne_gain

        # ConductanceLIF automatically handles shunting inhibition
        l4_spikes, _ = self.l4_neurons(l4_g_exc, l4_g_inh)
        l4_spikes = self._apply_sparsity_1d(l4_spikes, cfg.l4_sparsity)
        self.state.l4_spikes = l4_spikes

        # Inter-layer shape check: L4 output
        assert l4_spikes.shape == (self.l4_size,), (
            f"LayeredCortex: L4 spikes have shape {l4_spikes.shape} "
            f"but expected ({self.l4_size},). "
            f"Check L4 sparsity or input→L4 weights shape."
        )

        # =====================================================================
        # APPLY L4→L2/3 AXONAL DELAY
        # =====================================================================
        # Apply biological transmission delay for L4→L2/3 vertical projection
        # If delay is 0, l4_spikes_delayed = l4_spikes (instant, backward compatible)
        if self._l4_l23_delay_steps > 0:
            # Initialize buffer on first use
            if self._l4_l23_delay_buffer is None:
                # Use exact delay size (no safety margin) for memory efficiency
                self._l4_l23_delay_buffer = torch.zeros(
                    self._l4_l23_delay_steps,
                    self.l4_size,
                    device=l4_spikes.device,
                    dtype=torch.bool,
                )
                self._l4_l23_delay_ptr = 0

            # Retrieve delayed spikes FIRST (before storing new ones)
            read_idx = (
                self._l4_l23_delay_ptr - self._l4_l23_delay_steps
            ) % self._l4_l23_delay_buffer.shape[0]
            l4_spikes_delayed = self._l4_l23_delay_buffer[read_idx]

            # Then store current spikes for future retrieval
            self._l4_l23_delay_buffer[self._l4_l23_delay_ptr] = l4_spikes

            # Advance pointer
            self._l4_l23_delay_ptr = (self._l4_l23_delay_ptr + 1) % self._l4_l23_delay_buffer.shape[
                0
            ]
        else:
            l4_spikes_delayed = l4_spikes

        # L2/3: Processing with recurrence
        # NOTE: Use delayed L4 spikes for biological accuracy
        l23_ff = (
            torch.matmul(l4_spikes_delayed.float(), self.synaptic_weights["l4_l23"].t())
            * cfg.l4_to_l23_strength
        )

        # Stimulus gating (transient inhibition - always enabled)
        ffi = self.stimulus_gating.compute(input_spikes, return_tensor=False)
        raw_ffi = ffi.item() if hasattr(ffi, "item") else float(ffi)
        self.state.ffi_strength = min(1.0, raw_ffi / self.stimulus_gating.max_inhibition)
        ffi_suppression = 1.0 - self.state.ffi_strength * cfg.ffi_strength

        # =====================================================================
        # RECURRENT L2/3 WITH STP (prevents frozen attractors)
        # =====================================================================
        # Without STP, recurrent connections cause the same neurons to fire
        # every timestep (frozen attractor). With DEPRESSING STP, frequently-
        # used synapses get temporarily weaker, allowing pattern transitions.

        # =====================================================================
        # ACETYLCHOLINE MODULATION OF HORIZONTAL CONNECTIONS (Hasselmo 1999)
        # =====================================================================
        # High ACh (encoding mode): Suppress horizontal/recurrent connections
        # to prevent contextual interference during new sensory encoding
        # Low ACh (retrieval mode): Enhance recurrent connections for
        # associative processing and pattern completion
        #
        # Biological mechanism: ACh from nucleus basalis suppresses horizontal
        # connections in cortex via presynaptic muscarinic receptors (M2/M4)
        ach_level = self.state.acetylcholine
        # ACh > 0.5 → encoding mode → suppress recurrence (down to 0.2x)
        # ACh < 0.5 → retrieval mode → full recurrence (1.0x)
        ach_recurrent_modulation = compute_ach_recurrent_suppression(ach_level)

        if self.state.l23_recurrent_activity is not None:
            recurrent_scale = (
                L23_RECURRENT_RETRIEVAL_SCALE + L23_RECURRENT_RETRIEVAL_SCALE * retrieval_mod
            )

            # Apply STP to recurrent connections (always enabled)
            stp_efficacy = self.stp_l23_recurrent(
                self.state.l23_recurrent_activity.float()
            )  # [l23_size, l23_size]

            effective_w_rec = self.synaptic_weights["l23_recurrent"] * stp_efficacy
            l23_rec = (
                torch.matmul(effective_w_rec, self.state.l23_recurrent_activity.float())
                * cfg.l23_recurrent_strength
                * recurrent_scale
                * ffi_suppression
                * ach_recurrent_modulation  # ACh suppression
            )
        else:
            l23_rec = torch.zeros_like(l23_ff)

        # Top-down modulation
        l23_td = top_down * cfg.l23_top_down_strength if top_down is not None else 0.0

        l23_input = l23_ff + l23_rec + l23_td

        # Lateral inhibition
        if self.state.l23_spikes is not None:
            l23_inhib = torch.matmul(
                self.synaptic_weights["l23_inhib"], self.state.l23_spikes.float()
            )

            # E/I Balance: Scale inhibition to maintain healthy E/I ratio
            if self.ei_balance is not None:
                # Track E/I balance using L2/3 excitation vs inhibition
                _ = self.ei_balance.update(
                    self.state.l23_spikes,  # Excitatory activity
                    self.state.l23_spikes,  # Proxy for inhibitory (scaled below)
                )
                # Scale inhibition to maintain target E/I ratio
                l23_inhib = self.ei_balance.scale_inhibition(l23_inhib)

            l23_input = l23_input - l23_inhib

        # INTRINSIC PLASTICITY: Apply per-neuron threshold offset (UnifiedHomeostasis)
        # Neurons that fire too much have higher thresholds (less excitable)
        cfg = self.config
        if cfg.homeostasis_enabled and self._l23_threshold_offset is not None:
            l23_input = l23_input - self._l23_threshold_offset

        # GAP JUNCTION COUPLING (L2/3 interneuron synchronization)
        # Ultra-fast electrical coupling for gamma synchronization
        # Apply gap junction current based on previous timestep's membrane potentials
        if self.gap_junctions_l23 is not None and self.state.l23_membrane is not None:
            # Get coupling current from neighboring interneurons
            gap_current = self.gap_junctions_l23(self.state.l23_membrane)
            # Add gap junction depolarization to input
            l23_input = l23_input + gap_current

        l23_spikes, l23_membrane = self.l23_neurons(F.relu(l23_input), F.relu(l23_input) * 0.25)
        l23_spikes = self._apply_sparsity_1d(l23_spikes, cfg.l23_sparsity)
        self.state.l23_spikes = l23_spikes
        self.state.l23_membrane = l23_membrane  # Store for next timestep gap junctions

        # Inter-layer shape check: L2/3 → L5
        assert l23_spikes.shape == (self.l23_size,), (
            f"LayeredCortex: L2/3 spikes have shape {l23_spikes.shape} "
            f"but expected ({self.l23_size},). "
            f"Check L2/3 sparsity or L4→L2/3 weights shape."
        )

        # Gamma-phase attention: Modulate L2/3 spikes by gamma phase from Brain
        # Always enabled for spike-native attention
        if hasattr(self.state, "_oscillator_phases"):
            gamma_phase = self.state._oscillator_phases.get("gamma", 0.0)

            # Compute phase-based gating for each L2/3 neuron
            phase_diff = torch.abs(self.l23_phase_prefs - gamma_phase)
            phase_diff = torch.min(phase_diff, 2 * math.pi - phase_diff)

            # Gaussian gating based on phase proximity
            gamma_gate = torch.exp(-(phase_diff**2) / (2 * self.gamma_attention_width**2))

            # Modulate L2/3 spikes (attention without Q/K/V projections!)
            l23_spikes = l23_spikes * gamma_gate

            # Store gating in state for diagnostics
            self.state.gamma_attention_phase = gamma_phase
            self.state.gamma_attention_gate = gamma_gate

        # Update recurrent activity trace
        if self.state.l23_recurrent_activity is not None:
            self.state.l23_recurrent_activity = (
                self.state.l23_recurrent_activity * cfg.l23_recurrent_decay + l23_spikes.float()
            )
        else:
            self.state.l23_recurrent_activity = l23_spikes.float()

        # =====================================================================
        # APPLY L2/3→L5 AXONAL DELAY
        # =====================================================================
        # Apply biological transmission delay for L2/3→L5 vertical projection
        # If delay is 0, l23_spikes_delayed = l23_spikes (instant, backward compatible)
        if self._l23_l5_delay_steps > 0:
            # Initialize buffer on first use
            if self._l23_l5_delay_buffer is None:
                # Use exact delay size (no safety margin) for memory efficiency
                self._l23_l5_delay_buffer = torch.zeros(
                    self._l23_l5_delay_steps,
                    self.l23_size,
                    device=l23_spikes.device,
                    dtype=torch.bool,
                )
                self._l23_l5_delay_ptr = 0

            # Retrieve delayed spikes FIRST (before storing new ones)
            read_idx = (
                self._l23_l5_delay_ptr - self._l23_l5_delay_steps
            ) % self._l23_l5_delay_buffer.shape[0]
            l23_spikes_delayed = self._l23_l5_delay_buffer[read_idx]

            # Then store current spikes for future retrieval
            self._l23_l5_delay_buffer[self._l23_l5_delay_ptr] = l23_spikes

            # Advance pointer
            self._l23_l5_delay_ptr = (self._l23_l5_delay_ptr + 1) % self._l23_l5_delay_buffer.shape[
                0
            ]
        else:
            l23_spikes_delayed = l23_spikes

        # L5: Subcortical output (conductance-based)
        # NOTE: Use delayed L2/3 spikes for biological accuracy
        l5_g_exc = (
            torch.matmul(self.synaptic_weights["l23_l5"], l23_spikes_delayed.float())
            * cfg.l23_to_l5_strength
        )

        # L5 inhibition: ~25% of excitation (4:1 E/I ratio)
        l5_g_inh = l5_g_exc * 0.25

        l5_spikes, _ = self.l5_neurons(l5_g_exc, l5_g_inh)
        l5_spikes = self._apply_sparsity_1d(l5_spikes, cfg.l5_sparsity)
        self.state.l5_spikes = l5_spikes

        # Inter-layer shape check: L5 output
        assert l5_spikes.shape == (self.l5_size,), (
            f"LayeredCortex: L5 spikes have shape {l5_spikes.shape} "
            f"but expected ({self.l5_size},). "
            f"Check L5 sparsity or L2/3→L5 weights shape."
        )

        # =====================================================================
        # LAYER 6: CORTICOTHALAMIC FEEDBACK (SPLIT INTO L6a AND L6b)
        # =====================================================================
        # L6a (corticothalamic type I): Projects to TRN (inhibitory modulation, low gamma)
        # L6b (corticothalamic type II): Projects to relay (excitatory modulation, high gamma)
        # This dual feedback implements both spatial attention (via TRN) and
        # fast gain modulation (direct relay excitation).

        # =====================================================================
        # L6a: Apply L2/3→L6a axonal delay
        # =====================================================================
        if self._l23_l6a_delay_steps > 0:
            # Initialize buffer on first use
            if self._l23_l6a_delay_buffer is None:
                self._l23_l6a_delay_buffer = torch.zeros(
                    self._l23_l6a_delay_steps,
                    self.l23_size,
                    device=l23_spikes.device,
                    dtype=torch.bool,
                )
                self._l23_l6a_delay_ptr = 0

            # Retrieve delayed spikes
            read_idx = (
                self._l23_l6a_delay_ptr - self._l23_l6a_delay_steps
            ) % self._l23_l6a_delay_buffer.shape[0]
            l23_spikes_for_l6a = self._l23_l6a_delay_buffer[read_idx]

            # Store current L2/3 spikes
            self._l23_l6a_delay_buffer[self._l23_l6a_delay_ptr] = l23_spikes
            self._l23_l6a_delay_ptr = (
                self._l23_l6a_delay_ptr + 1
            ) % self._l23_l6a_delay_buffer.shape[0]
        else:
            l23_spikes_for_l6a = l23_spikes

        # L6a forward pass (corticothalamic type I → TRN)
        l6a_g_exc = (
            torch.matmul(self.synaptic_weights["l23_l6a"], l23_spikes_for_l6a.float())
            * cfg.l23_to_l6a_strength
        )
        l6a_g_inh = l6a_g_exc * 0.8  # Strong local inhibition for sparse low-gamma firing

        l6a_spikes, _ = self.l6a_neurons(l6a_g_exc, l6a_g_inh)
        l6a_spikes = self._apply_sparsity_1d(l6a_spikes, cfg.l6a_sparsity)
        self.state.l6a_spikes = l6a_spikes

        # =====================================================================
        # L6b: Apply L2/3→L6b axonal delay
        # =====================================================================
        if self._l23_l6b_delay_steps > 0:
            # Initialize buffer on first use
            if self._l23_l6b_delay_buffer is None:
                self._l23_l6b_delay_buffer = torch.zeros(
                    self._l23_l6b_delay_steps,
                    self.l23_size,
                    device=l23_spikes.device,
                    dtype=torch.bool,
                )
                self._l23_l6b_delay_ptr = 0

            # Retrieve delayed spikes
            read_idx = (
                self._l23_l6b_delay_ptr - self._l23_l6b_delay_steps
            ) % self._l23_l6b_delay_buffer.shape[0]
            l23_spikes_for_l6b = self._l23_l6b_delay_buffer[read_idx]

            # Store current L2/3 spikes
            self._l23_l6b_delay_buffer[self._l23_l6b_delay_ptr] = l23_spikes
            self._l23_l6b_delay_ptr = (
                self._l23_l6b_delay_ptr + 1
            ) % self._l23_l6b_delay_buffer.shape[0]
        else:
            l23_spikes_for_l6b = l23_spikes

        # L6b forward pass (corticothalamic type II → relay)
        l6b_g_exc = (
            torch.matmul(self.synaptic_weights["l23_l6b"], l23_spikes_for_l6b.float())
            * cfg.l23_to_l6b_strength
        )
        l6b_g_inh = l6b_g_exc * 0.15  # Minimal local inhibition

        l6b_spikes, _ = self.l6b_neurons(l6b_g_exc, l6b_g_inh)
        l6b_spikes = self._apply_sparsity_1d(l6b_spikes, cfg.l6b_sparsity)
        self.state.l6b_spikes = l6b_spikes

        # Inter-layer shape checks
        assert l6a_spikes.shape == (self.l6a_size,), (
            f"LayeredCortex: L6a spikes have shape {l6a_spikes.shape} "
            f"but expected ({self.l6a_size},)."
        )
        assert l6b_spikes.shape == (self.l6b_size,), (
            f"LayeredCortex: L6b spikes have shape {l6b_spikes.shape} "
            f"but expected ({self.l6b_size},)."
        )

        # Update cumulative spike counters (for diagnostics)
        self._cumulative_l4_spikes += int(l4_spikes.sum().item())
        self._cumulative_l23_spikes += int(l23_spikes.sum().item())
        self._cumulative_l5_spikes += int(l5_spikes.sum().item())
        self._cumulative_l6a_spikes += int(l6a_spikes.sum().item())
        self._cumulative_l6b_spikes += int(l6b_spikes.sum().item())

        # Update STDP traces using utility function
        if self.state.l4_trace is not None:
            update_trace(self.state.l4_trace, l4_spikes, tau=cfg.tau_plus_ms, dt=dt)
        if self.state.l23_trace is not None:
            update_trace(self.state.l23_trace, l23_spikes, tau=cfg.tau_plus_ms, dt=dt)
        if self.state.l5_trace is not None:
            update_trace(self.state.l5_trace, l5_spikes, tau=cfg.tau_plus_ms, dt=dt)
        if self.state.l6a_trace is not None:
            update_trace(self.state.l6a_trace, l6a_spikes, tau=cfg.tau_plus_ms, dt=dt)
        if self.state.l6b_trace is not None:
            update_trace(self.state.l6b_trace, l6b_spikes, tau=cfg.tau_plus_ms, dt=dt)

        self.state.spikes = l5_spikes

        # Store input for plasticity
        self.state.input_spikes = input_spikes

        # Apply continuous plasticity (learning happens as part of forward dynamics)
        self._apply_plasticity()

        # Construct output: always concatenate L2/3 and L5 (biological cortex has both pathways)
        output = torch.cat([l23_spikes, l5_spikes], dim=-1)

        # Axonal delays are handled by AxonalProjection pathways, not within regions
        # ADR-005: Return 1D tensor as bool spikes
        return output.bool()

    def get_l6_spikes(self) -> Optional[torch.Tensor]:
        """Get L6 corticothalamic feedback spikes (combined L6a + L6b).

        This method is called by DynamicBrain to retrieve L6 spikes
        for routing to thalamus. Returns combined L6a + L6b for backward
        compatibility with existing brain configurations.

        For port-based routing, use get_output() with:
        - port="l6a" for L6a→TRN pathway only
        - port="l6b" for L6b→relay pathway only

        Returns:
            Combined L6a + L6b spikes [l6a_size + l6b_size] or None if not available

        Note:
            L6 spikes are NOT part of cortex output (forward() returns L2/3+L5).
            L6 is a dedicated feedback pathway that must be explicitly accessed.
        """
        if self.state.l6a_spikes is None or self.state.l6b_spikes is None:
            return None
        return torch.cat([self.state.l6a_spikes, self.state.l6b_spikes], dim=-1)

    def get_l6_feedback(self) -> Optional[torch.Tensor]:
        """Alias for get_l6_spikes for compatibility with DynamicBrain port extraction.

        Returns:
            Combined L6a + L6b feedback spikes [l6a_size + l6b_size] or None
        """
        return self.get_l6_spikes()

    def get_output(self, port: Optional[str] = None) -> torch.Tensor:
        """Get layer output with port-based routing support.

        Supports multiple output ports for flexible brain connectivity:
        - port="l6a": L6a→TRN pathway spikes (low gamma, inhibitory modulation)
        - port="l6b": L6b→relay pathway spikes (high gamma, excitatory modulation)
        - port="l23": L2/3 cortico-cortical pathway
        - port="l5": L5 subcortical pathway
        - port=None: Default cortex output (L2/3 + L5)

        Args:
            port: Output port name (None for default)

        Returns:
            Spikes from requested port [port_size], bool tensor

        Raises:
            ValueError: If port is not recognized
        """
        if port == "l6a":
            if self.state.l6a_spikes is not None:
                return self.state.l6a_spikes
            return torch.zeros(self.l6a_size, device=self.device, dtype=torch.bool)
        elif port == "l6b":
            if self.state.l6b_spikes is not None:
                return self.state.l6b_spikes
            return torch.zeros(self.l6b_size, device=self.device, dtype=torch.bool)
        elif port == "l23":
            if self.state.l23_spikes is not None:
                return self.state.l23_spikes
            return torch.zeros(self.l23_size, device=self.device, dtype=torch.bool)
        elif port == "l5":
            if self.state.l5_spikes is not None:
                return self.state.l5_spikes
            return torch.zeros(self.l5_size, device=self.device, dtype=torch.bool)
        elif port is None:
            # Default: L2/3 + L5 output
            if self.state.l23_spikes is None or self.state.l5_spikes is None:
                return torch.zeros(
                    self.l23_size + self.l5_size, device=self.device, dtype=torch.bool
                )
            return torch.cat([self.state.l23_spikes, self.state.l5_spikes], dim=-1)
        else:
            # Invalid port - raise error
            valid_ports = ["l6a", "l6b", "l23", "l5", None]
            raise ValueError(
                f"LayeredCortex.get_output: Invalid port '{port}'. " f"Valid ports: {valid_ports}"
            )

    def _apply_sparsity_1d(
        self,
        spikes: torch.Tensor,
        target_sparsity: float,
    ) -> torch.Tensor:
        """Apply winner-take-all sparsity to 1D spike tensor (ADR-005)."""
        assert spikes.dim() == 1, f"Expected 1D spikes, got shape {spikes.shape}"

        n_neurons = spikes.shape[0]
        k = max(1, int(n_neurons * target_sparsity))

        sparse_spikes = torch.zeros_like(spikes)
        active = spikes.nonzero(as_tuple=True)[0]

        if len(active) > k:
            keep_indices = active[torch.randperm(len(active))[:k]]
            sparse_spikes[keep_indices] = spikes[keep_indices]
        else:
            sparse_spikes = spikes

        return sparse_spikes

    def _apply_plasticity(self) -> None:
        """Apply continuous STDP learning with BCM modulation.

        This is called automatically at each forward() timestep.
        Learning rate is modulated by dopamine (via get_effective_learning_rate).

        In biological cortex, synaptic plasticity happens continuously based on
        pre/post spike timing. Dopamine doesn't trigger learning - it modulates
        how much weight change occurs from the spike-timing-based plasticity.

        Note:
            Timestep (dt_ms) is obtained from self.config for temporal dynamics
        """
        if self.state.l4_spikes is None or self.state.input_spikes is None:
            return

        cfg = self.config

        # =====================================================================
        # LAYER-SPECIFIC DOPAMINE MODULATION (Enhancement #1)
        # =====================================================================
        # Apply layer-specific dopamine scaling to learning rates.
        # Different layers have different DA receptor densities (relative sensitivity).
        base_dopamine = self.state.dopamine
        l4_dopamine = base_dopamine * CORTEX_L4_DA_FRACTION
        l23_dopamine = base_dopamine * CORTEX_L23_DA_FRACTION
        l5_dopamine = base_dopamine * CORTEX_L5_DA_FRACTION
        l6_dopamine = base_dopamine * CORTEX_L6_DA_FRACTION

        # Store for diagnostics and testing
        self._l4_dopamine = l4_dopamine
        self._l23_dopamine = l23_dopamine
        self._l5_dopamine = l5_dopamine
        self._l6_dopamine = l6_dopamine

        # Get base learning rate
        base_lr = cfg.stdp_lr

        # Early exit if base learning rate is too small
        if base_lr < 1e-8:
            self.state.last_plasticity_delta = 0.0
            return

        # NOTE: All neuromodulators (DA, ACh, NE) are now managed centrally by Brain.
        # VTA updates dopamine, LC updates NE, NB updates ACh.
        # Brain broadcasts to all regions every timestep via _update_neuromodulators().
        # No local decay needed.

        # Skip plasticity if BCM is not enabled
        if not self.config.bcm_enabled:
            return

        # Get 1D versions of spike tensors for torch.outer
        l4_spikes = ensure_1d(self.state.l4_spikes)
        l23_spikes = ensure_1d(self.state.l23_spikes) if self.state.l23_spikes is not None else None
        l5_spikes = ensure_1d(self.state.l5_spikes) if self.state.l5_spikes is not None else None
        input_spikes = ensure_1d(self.state.input_spikes)

        total_change = 0.0

        # Use STDP+BCM composite strategies for proper spike-timing-dependent learning
        # Input → L4 (using synaptic_weights) - L4-specific dopamine
        if self.bcm_l4 is not None:
            # L4 learning rate with layer-specific dopamine
            l4_lr = base_lr * (1.0 + l4_dopamine)
            updated_weights, _ = self.bcm_l4.compute_update(
                weights=self.synaptic_weights["input"].data,
                pre=input_spikes,
                post=l4_spikes,
                learning_rate=l4_lr,
            )
            dw = updated_weights - self.synaptic_weights["input"].data
            self.synaptic_weights["input"].data.copy_(updated_weights)
            clamp_weights(self.synaptic_weights["input"].data, cfg.w_min, cfg.w_max)
            total_change += dw.abs().mean().item()

        # L4 → L2/3 - L2/3-specific dopamine
        if l23_spikes is not None and self.bcm_l23 is not None:
            # L2/3 learning rate with layer-specific dopamine
            l23_lr = base_lr * (1.0 + l23_dopamine)
            updated_weights, _ = self.bcm_l23.compute_update(
                weights=self.synaptic_weights["l4_l23"].data,
                pre=l4_spikes,
                post=l23_spikes,
                learning_rate=l23_lr,
            )
            dw = updated_weights - self.synaptic_weights["l4_l23"].data
            self.synaptic_weights["l4_l23"].data.copy_(updated_weights)
            clamp_weights(self.synaptic_weights["l4_l23"].data, cfg.w_min, cfg.w_max)
            total_change += dw.abs().mean().item()

            # L2/3 recurrent (signed weights - compact E/I approximation)
            # Uses dedicated bounds [l23_recurrent_w_min, l23_recurrent_w_max] to allow
            # both excitatory and inhibitory-like lateral connections.
            # This is a simplification of explicit E/I interneuron populations.
            updated_weights, _ = self.bcm_l23.compute_update(
                weights=self.synaptic_weights["l23_recurrent"].data,
                pre=l23_spikes,
                post=l23_spikes,
                learning_rate=l23_lr * 0.5,  # Reduced for recurrent stability
            )
            dw = updated_weights - self.synaptic_weights["l23_recurrent"].data

            self.synaptic_weights["l23_recurrent"].data.copy_(updated_weights)
            self.synaptic_weights["l23_recurrent"].data.fill_diagonal_(0.0)
            clamp_weights(
                self.synaptic_weights["l23_recurrent"].data,
                cfg.l23_recurrent_w_min,
                cfg.l23_recurrent_w_max,
            )

            total_change += dw.abs().mean().item()

            # =================================================================
            # INTRINSIC PLASTICITY: Update per-neuron threshold offsets
            # =================================================================
            # Handled by UnifiedHomeostasis.compute_excitability_modulation()
            if cfg.homeostasis_enabled:
                l23_spikes_1d = l23_spikes.float()

                # Initialize if needed
                if self._l23_activity_history is None:
                    self._l23_activity_history = torch.zeros(
                        self.l23_size, device=l23_spikes_1d.device
                    )
                if self._l23_threshold_offset is None:
                    self._l23_threshold_offset = torch.zeros(
                        self.l23_size, device=l23_spikes_1d.device
                    )

                # Update activity history (exponential moving average)
                self._l23_activity_history = (
                    EMA_DECAY_FAST * self._l23_activity_history
                    + (1 - EMA_DECAY_FAST) * l23_spikes_1d
                )

                # Compute threshold modulation using UnifiedHomeostasis
                threshold_mod = self.homeostasis.compute_excitability_modulation(
                    activity_history=self._l23_activity_history, tau=100.0
                )
                self._l23_threshold_offset = threshold_mod.clamp(-0.5, 0.5)

            # L2/3 → L5 - L5-specific dopamine (highest modulation)
            if l5_spikes is not None and self.bcm_l5 is not None:
                # L5 learning rate with layer-specific dopamine (100% modulation)
                l5_lr = base_lr * (1.0 + l5_dopamine)
                updated_weights, _ = self.bcm_l5.compute_update(
                    weights=self.synaptic_weights["l23_l5"].data,
                    pre=l23_spikes,
                    post=l5_spikes,
                    learning_rate=l5_lr,
                )
                dw = updated_weights - self.synaptic_weights["l23_l5"].data
                self.synaptic_weights["l23_l5"].data.copy_(updated_weights)
                clamp_weights(self.synaptic_weights["l23_l5"].data, cfg.w_min, cfg.w_max)
                total_change += dw.abs().mean().item()

            # L2/3 → L6a (corticothalamic type I → TRN) - L6-specific dopamine
            l6a_spikes = self.state.l6a_spikes
            if l6a_spikes is not None and self.bcm_l6a is not None:
                # L6 learning rate with layer-specific dopamine (20% modulation - stable feedback)
                l6_lr = base_lr * (1.0 + l6_dopamine)
                updated_weights, _ = self.bcm_l6a.compute_update(
                    weights=self.synaptic_weights["l23_l6a"].data,
                    pre=l23_spikes,
                    post=l6a_spikes,
                    learning_rate=l6_lr,
                )
                dw = updated_weights - self.synaptic_weights["l23_l6a"].data
                self.synaptic_weights["l23_l6a"].data.copy_(updated_weights)
                clamp_weights(self.synaptic_weights["l23_l6a"].data, cfg.w_min, cfg.w_max)
                total_change += dw.abs().mean().item()

            # L2/3 → L6b (corticothalamic type II → relay) - L6-specific dopamine
            l6b_spikes = self.state.l6b_spikes
            if l6b_spikes is not None and self.bcm_l6b is not None:
                # L6 learning rate with layer-specific dopamine (20% modulation - stable feedback)
                updated_weights, _ = self.bcm_l6b.compute_update(
                    weights=self.synaptic_weights["l23_l6b"].data,
                    pre=l23_spikes,
                    post=l6b_spikes,
                    learning_rate=l6_lr,
                )
                dw = updated_weights - self.synaptic_weights["l23_l6b"].data
                self.synaptic_weights["l23_l6b"].data.copy_(updated_weights)
                clamp_weights(self.synaptic_weights["l23_l6b"].data, cfg.w_min, cfg.w_max)
                total_change += dw.abs().mean().item()

        # Store for monitoring
        self.state.last_plasticity_delta = total_change

    def get_layer_outputs(self) -> Dict[str, Optional[torch.Tensor]]:
        """Get outputs from all layers.

        Returns:
            Dict mapping layer names to spike tensors:
            - "L4": Input layer spikes
            - "L2/3": Processing/cortico-cortical layer spikes
            - "L5": Subcortical output layer spikes
            - "L6a": Corticothalamic type I feedback (to TRN)
            - "L6b": Corticothalamic type II feedback (to relay)

        Note:
            Layer outputs can be accessed via port-based routing:
            >>> builder.connect("cortex", "hippocampus", source_port="l23")
            >>> builder.connect("cortex", "striatum", source_port="l5")
            >>> builder.connect("cortex", "thalamus", source_port="l6a", target_port="l6a_feedback")
            >>> builder.connect("cortex", "thalamus", source_port="l6b", target_port="l6b_feedback")
        """
        return {
            "L4": self.state.l4_spikes,
            "L2/3": self.state.l23_spikes,
            "L5": self.state.l5_spikes,
            "L6a": self.state.l6a_spikes,
            "L6b": self.state.l6b_spikes,
        }

    def get_cortical_output(self) -> Optional[torch.Tensor]:
        """Get L2/3 output (for cortico-cortical pathways)."""
        return self.state.l23_spikes

    def get_subcortical_output(self) -> Optional[torch.Tensor]:
        """Get L5 output (for subcortical pathways)."""
        return self.state.l5_spikes

    # endregion

    # region Diagnostics and Health Monitoring

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics in standardized LayeredCortexDiagnostics format.

        Returns consolidated diagnostic information about:
        - Activity: L2/3 output spike statistics (primary cortical output)
        - Plasticity: Weight statistics for all inter-layer connections
        - Health: Layer sparsity, BCM thresholds, E/I balance, cumulative activity
        - Neuromodulators: Dopamine, norepinephrine, acetylcholine (if applicable)
        - Region-specific: Layer-specific activity (L4/L2/3/L5/L6), recurrent dynamics

        Note: Reports both instantaneous (l4_active_count) and cumulative
        (l4_cumulative_spikes) counts. During consolidation phases with
        zero input, instantaneous L4 will be 0 but cumulative shows
        total activity since last reset.

        This is the primary diagnostic interface for the Cortex.
        """
        cfg = self.config

        # Compute activity metrics from L2/3 (primary cortical output)
        activity = compute_activity_metrics(
            output_spikes=(
                self.state.l23_spikes
                if self.state.l23_spikes is not None
                else torch.zeros(self.l23_size, device=self.device)
            ),
            total_neurons=self.l23_size,
        )

        # Compute plasticity metrics from L2/3 recurrent (most dynamic)
        plasticity = None
        if cfg.learn:
            plasticity = compute_plasticity_metrics(
                weights=self.synaptic_weights["l23_recurrent"].data,
                learning_rate=cfg.learning_rate,
            )
            # Add other pathway statistics
            plasticity["input_l4_mean"] = float(self.synaptic_weights["input"].data.mean().item())  # type: ignore[typeddict-item]
            plasticity["l4_l23_mean"] = float(self.synaptic_weights["l4_l23"].data.mean().item())  # type: ignore[typeddict-item]
            plasticity["l23_l5_mean"] = float(self.synaptic_weights["l23_l5"].data.mean().item())  # type: ignore[typeddict-item]
            plasticity["l23_l6a_mean"] = float(self.synaptic_weights["l23_l6a"].data.mean().item())  # type: ignore[typeddict-item]
            plasticity["l23_l6b_mean"] = float(self.synaptic_weights["l23_l6b"].data.mean().item())  # type: ignore[typeddict-item]

        # Compute health metrics
        health_tensors = {
            "l4_spikes": (
                self.state.l4_spikes
                if self.state.l4_spikes is not None
                else torch.zeros(self.l4_size, device=self.device)
            ),
            "l23_spikes": (
                self.state.l23_spikes
                if self.state.l23_spikes is not None
                else torch.zeros(self.l23_size, device=self.device)
            ),
            "l5_spikes": (
                self.state.l5_spikes
                if self.state.l5_spikes is not None
                else torch.zeros(self.l5_size, device=self.device)
            ),
            "l6a_spikes": (
                self.state.l6a_spikes
                if self.state.l6a_spikes is not None
                else torch.zeros(self.l6a_size, device=self.device)
            ),
            "l6b_spikes": (
                self.state.l6b_spikes
                if self.state.l6b_spikes is not None
                else torch.zeros(self.l6b_size, device=self.device)
            ),
        }

        health = compute_health_metrics(
            state_tensors=health_tensors,
            firing_rate=activity.get("firing_rate", 0.0),
        )

        # Add cumulative activity tracking
        health["l4_cumulative_spikes"] = getattr(self, "_cumulative_l4_spikes", 0)  # type: ignore[typeddict-item]
        health["l23_cumulative_spikes"] = getattr(self, "_cumulative_l23_spikes", 0)  # type: ignore[typeddict-item]
        health["l5_cumulative_spikes"] = getattr(self, "_cumulative_l5_spikes", 0)  # type: ignore[typeddict-item]
        health["l6a_cumulative_spikes"] = getattr(self, "_cumulative_l6a_spikes", 0)  # type: ignore[typeddict-item]
        health["l6b_cumulative_spikes"] = getattr(self, "_cumulative_l6b_spikes", 0)  # type: ignore[typeddict-item]

        # Neuromodulator metrics
        neuromodulators = {}
        if hasattr(self, "neuromodulators"):
            neuromodulators = self.neuromodulators.copy()  # type: ignore[union-attr, operator]

        # Region-specific custom metrics
        region_specific = {
            "architecture": {
                "l4_size": self.l4_size,
                "l23_size": self.l23_size,
                "l5_size": self.l5_size,
                "l6a_size": self.l6a_size,
                "l6b_size": self.l6b_size,
            },
            "layer_activity": {},
            "recurrent_dynamics": {},
            "robustness": {},
        }

        # Layer-specific activity
        if self.state.l4_spikes is not None:
            region_specific["layer_activity"]["l4"] = {
                "active_count": int(compute_spike_count(self.state.l4_spikes)),  # type: ignore[assignment]
                "firing_rate": float(compute_firing_rate(self.state.l4_spikes)),  # type: ignore[assignment]
            }
        if self.state.l23_spikes is not None:
            region_specific["layer_activity"]["l23"] = {
                "active_count": int(compute_spike_count(self.state.l23_spikes)),  # type: ignore[assignment]
                "firing_rate": float(compute_firing_rate(self.state.l23_spikes)),  # type: ignore[assignment]
            }
        if self.state.l5_spikes is not None:
            region_specific["layer_activity"]["l5"] = {
                "active_count": int(compute_spike_count(self.state.l5_spikes)),  # type: ignore[assignment]
                "firing_rate": float(compute_firing_rate(self.state.l5_spikes)),  # type: ignore[assignment]
            }
        if self.state.l6a_spikes is not None:
            region_specific["layer_activity"]["l6a"] = {
                "active_count": int(compute_spike_count(self.state.l6a_spikes)),  # type: ignore[assignment]
                "firing_rate": float(compute_firing_rate(self.state.l6a_spikes)),  # type: ignore[assignment]
            }
        if self.state.l6b_spikes is not None:
            region_specific["layer_activity"]["l6b"] = {
                "active_count": int(compute_spike_count(self.state.l6b_spikes)),  # type: ignore[assignment]
                "firing_rate": float(compute_firing_rate(self.state.l6b_spikes)),  # type: ignore[assignment]
            }

        # Recurrent activity
        if self.state.l23_recurrent_activity is not None:
            region_specific["recurrent_dynamics"]["l23_recurrent_mean"] = float(
                self.state.l23_recurrent_activity.mean().item()
            )  # type: ignore[assignment]

        # Robustness mechanisms (E/I balance)
        if self.ei_balance is not None:
            ei_diag = self.ei_balance.get_diagnostics()
            region_specific["robustness"] = {
                "ei_ratio": ei_diag.get("current_ratio", 0.0),
                "ei_inh_scale": ei_diag.get("inh_scale", 1.0),
                "ei_status": ei_diag.get("status", "unknown"),
            }

        # Return in standardized format
        return {
            "activity": activity,
            "plasticity": plasticity,
            "health": health,
            "neuromodulators": neuromodulators,
            "region_specific": region_specific,
        }

    def get_full_state(self) -> Dict[str, Any]:
        """Get complete state for checkpointing.

        Returns state dictionary with keys:
        - weights: All inter-layer weight matrices
        - region_state: Current spikes, membrane potentials, traces
        - learning_state: BCM thresholds, STP state
        - neuromodulator_state: Current dopamine, norepinephrine, etc.
        - config: Configuration for validation
        """
        state_obj = self.get_state()
        state = state_obj.to_dict()

        # Add synaptic weights (required for checkpointing)
        state["synaptic_weights"] = {
            name: weights.detach().clone() for name, weights in self.synaptic_weights.items()
        }

        return state

    def load_full_state(self, state: Dict[str, Any]) -> None:
        """Load complete state from checkpoint.

        Args:
            state: State dictionary from get_full_state()
        """
        state_obj = LayeredCortexState.from_dict(state, device=str(self.device))
        self.load_state(state_obj)

        # Restore synaptic weights
        if "synaptic_weights" in state:
            for name, weights in state["synaptic_weights"].items():
                if name in self.synaptic_weights:
                    self.synaptic_weights[name].data = weights.to(self.device)

    def get_state(self) -> LayeredCortexState:
        """Get current state as LayeredCortexState (RegionState protocol).

        Returns LayeredCortexState with:
        - All layer spike states (L4, L2/3, L5, L6a, L6b)
        - All STDP traces per layer
        - L2/3 recurrent activity accumulation
        - Top-down modulation and attention gating
        - Feedforward inhibition and alpha suppression
        - STP state for L2/3 recurrent pathway
        - Neuromodulator levels

        This is the NEW API for RegionState protocol compliance.
        For backward compatibility, use get_full_state().

        Returns:
            LayeredCortexState with complete cortex state
        """
        # Get STP state
        stp_state = self.stp_l23_recurrent.get_state()

        return LayeredCortexState(
            # Base region state
            spikes=self.state.spikes.clone() if self.state.spikes is not None else None,
            membrane=self.state.membrane.clone() if self.state.membrane is not None else None,
            dopamine=self.state.dopamine,
            acetylcholine=self.state.acetylcholine,
            norepinephrine=self.state.norepinephrine,
            # Input
            input_spikes=(
                self.state.input_spikes.clone() if self.state.input_spikes is not None else None
            ),
            # Layer spike states
            l4_spikes=self.state.l4_spikes.clone() if self.state.l4_spikes is not None else None,
            l23_spikes=self.state.l23_spikes.clone() if self.state.l23_spikes is not None else None,
            l5_spikes=self.state.l5_spikes.clone() if self.state.l5_spikes is not None else None,
            l6a_spikes=self.state.l6a_spikes.clone() if self.state.l6a_spikes is not None else None,
            l6b_spikes=self.state.l6b_spikes.clone() if self.state.l6b_spikes is not None else None,
            # L2/3 recurrent activity
            l23_recurrent_activity=(
                self.state.l23_recurrent_activity.clone()
                if self.state.l23_recurrent_activity is not None
                else None
            ),
            # STDP traces
            l4_trace=self.state.l4_trace.clone() if self.state.l4_trace is not None else None,
            l23_trace=self.state.l23_trace.clone() if self.state.l23_trace is not None else None,
            l5_trace=self.state.l5_trace.clone() if self.state.l5_trace is not None else None,
            l6a_trace=self.state.l6a_trace.clone() if self.state.l6a_trace is not None else None,
            l6b_trace=self.state.l6b_trace.clone() if self.state.l6b_trace is not None else None,
            # Modulation state
            top_down_modulation=(
                self.state.top_down_modulation.clone()
                if self.state.top_down_modulation is not None
                else None
            ),
            ffi_strength=self.state.ffi_strength,
            alpha_suppression=self.state.alpha_suppression,
            # Gamma attention
            gamma_attention_phase=self.state.gamma_attention_phase,
            gamma_attention_gate=(
                self.state.gamma_attention_gate.clone()
                if self.state.gamma_attention_gate is not None
                else None
            ),
            # Plasticity monitoring
            last_plasticity_delta=self.state.last_plasticity_delta,
            # STP state
            stp_l23_recurrent_state=stp_state,  # type: ignore[arg-type]
            # Gap junction state
            l23_membrane=(
                self.state.l23_membrane.clone() if self.state.l23_membrane is not None else None
            ),
        )

    def load_state(self, state: LayeredCortexState) -> None:
        """Load state from LayeredCortexState (RegionState protocol).

        Restores:
        - All layer spike states (L4, L2/3, L5, L6a, L6b)
        - All STDP traces per layer
        - L2/3 recurrent activity accumulation
        - Top-down modulation and attention gating
        - Feedforward inhibition and alpha suppression
        - STP state for L2/3 recurrent pathway
        - Neuromodulator levels

        This is the NEW API for RegionState protocol compliance.
        For backward compatibility, use load_full_state().

        Args:
            state: LayeredCortexState to restore

        Note:
            Does NOT restore weights or BCM thresholds - those are learning state,
            not runtime state. Use load_full_state() for complete checkpoint loading.
        """
        # Restore base region state
        if state.spikes is not None:
            self.state.spikes = state.spikes.to(self.device)
        if state.membrane is not None:
            self.state.membrane = state.membrane.to(self.device)
        self.state.dopamine = state.dopamine
        self.state.acetylcholine = state.acetylcholine
        self.state.norepinephrine = state.norepinephrine

        # Restore input
        if state.input_spikes is not None:
            self.state.input_spikes = state.input_spikes.to(self.device)

        # Restore layer spike states
        if state.l4_spikes is not None:
            self.state.l4_spikes = state.l4_spikes.to(self.device)
        if state.l23_spikes is not None:
            self.state.l23_spikes = state.l23_spikes.to(self.device)
        if state.l5_spikes is not None:
            self.state.l5_spikes = state.l5_spikes.to(self.device)
        if state.l6a_spikes is not None:
            self.state.l6a_spikes = state.l6a_spikes.to(self.device)
        if state.l6b_spikes is not None:
            self.state.l6b_spikes = state.l6b_spikes.to(self.device)

        # Restore L2/3 recurrent activity
        if state.l23_recurrent_activity is not None:
            self.state.l23_recurrent_activity = state.l23_recurrent_activity.to(self.device)

        # Restore STDP traces
        if state.l4_trace is not None:
            self.state.l4_trace = state.l4_trace.to(self.device)
        if state.l23_trace is not None:
            self.state.l23_trace = state.l23_trace.to(self.device)
        if state.l5_trace is not None:
            self.state.l5_trace = state.l5_trace.to(self.device)
        if state.l6a_trace is not None:
            self.state.l6a_trace = state.l6a_trace.to(self.device)
        if state.l6b_trace is not None:
            self.state.l6b_trace = state.l6b_trace.to(self.device)

        # Restore modulation state
        if state.top_down_modulation is not None:
            self.state.top_down_modulation = state.top_down_modulation.to(self.device)
        self.state.ffi_strength = state.ffi_strength
        self.state.alpha_suppression = state.alpha_suppression

        # Restore gamma attention
        self.state.gamma_attention_phase = state.gamma_attention_phase
        if state.gamma_attention_gate is not None:
            self.state.gamma_attention_gate = state.gamma_attention_gate.to(self.device)

        # Restore plasticity monitoring
        self.state.last_plasticity_delta = state.last_plasticity_delta

        # Restore STP state
        if state.stp_l23_recurrent_state is not None:
            self.stp_l23_recurrent.load_state(state.stp_l23_recurrent_state)  # type: ignore[arg-type]

        # Restore gap junction state
        if state.l23_membrane is not None:
            self.state.l23_membrane = state.l23_membrane.to(self.device)

    # endregion
