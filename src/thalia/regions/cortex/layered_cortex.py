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
    │   - Receives from L4              │ → Attention pathway target
    │   - Lateral recurrent connections │
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
    └───────────────────────────────────┘

FILE ORGANIZATION (1295 lines)
===============================
Lines 1-150:     Module docstring, imports, class registration
Lines 151-350:   __init__() and layer initialization
Lines 351-500:   L4 forward pass (input processing)
Lines 501-700:   L2/3 forward pass (recurrent processing)
Lines 701-850:   L5 forward pass (output generation)
Lines 851-1000:  Learning (BCM + STDP for inter-layer connections)
Lines 1001-1150: Growth and homeostasis
Lines 1151-1295: Diagnostics and utility methods

NAVIGATION TIP: Use VSCode's "Go to Symbol" (Ctrl+Shift+O) or collapse
regions (Ctrl+K Ctrl+0) to navigate between layers.

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
- FeedforwardInhibition: Stimulus-triggered inhibition (shared with hippocampus)
- LayerEIBalance: E/I balance (shared concern)

See: docs/decisions/adr-011-large-file-justification.md

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from thalia.core.base.component_config import NeuralComponentConfig
from thalia.components.neurons.neuron_constants import NE_GAIN_RANGE
from thalia.core.errors import CheckpointError
from thalia.components.neurons import create_cortical_layer_neurons
from thalia.components.synapses.stp import ShortTermPlasticity, STPConfig, STPType
from thalia.components.synapses.weight_init import WeightInitializer
from thalia.managers.component_registry import register_region
from thalia.utils.core_utils import ensure_1d, clamp_weights
from thalia.components.synapses.traces import update_trace
from thalia.regions.base import NeuralComponent
from thalia.regions.cortex.config import calculate_layer_sizes
from thalia.regions.feedforward_inhibition import FeedforwardInhibition
from thalia.learning import BCMStrategyConfig, STDPConfig, create_cortex_strategy
from thalia.learning.ei_balance import LayerEIBalance
from thalia.learning.homeostasis.synaptic_homeostasis import UnifiedHomeostasis, UnifiedHomeostasisConfig

from .config import LayeredCortexConfig, LayeredCortexState


@register_region(
    "cortex",
    aliases=["layered_cortex"],
    description="Multi-layer cortical microcircuit with L4/L2/3/L5 structure",
    version="2.0",
    author="Thalia Project"
)
class LayeredCortex(NeuralComponent):
    """Multi-layer cortical microcircuit with proper layer separation and routing.

    Implements a canonical cortical column with distinct computational layers:

    **Layer Architecture** (based on mammalian cortex):
    - **L4**: Input layer - receives thalamic/sensory input, feedforward processing
    - **L2/3**: Processing layer - recurrent computation, outputs to other cortex
    - **L5**: Output layer - projects to subcortical structures (striatum, etc.)

    **Key Insight**:
    Output to next cortical area comes from a DIFFERENT layer (L2/3) than the
    one receiving recurrent feedback, solving the contamination problem in
    single-layer models. L5 provides separate subcortical output pathway.

    **Information Flow**:
    1. External input → L4 (feedforward processing)
    2. L4 → L2/3 (local integration)
    3. L2/3 → L2/3 (recurrent processing, lateral connections)
    4. L2/3 → L5 (deep projection)
    5. L2/3 → Other cortex (cortico-cortical output)
    6. L5 → Subcortical (striatum, thalamus, brainstem)

    **Learning Mechanisms**:
    - **Intra-layer**: BCM rule for homeostatic plasticity
    - **Inter-layer**: STDP for connection refinement
    - **Modulation**: Dopamine gates learning rate, ACh modulates encoding mode

    **Output Format**:
    Concatenated [L2/3_spikes, L5_spikes] for routing:
    - First n_l23 neurons: Cortico-cortical pathway
    - Last n_l5 neurons: Subcortical pathway

    **Usage Example**:

    .. code-block:: python

        config = LayeredCortexConfig(
            n_input=256,
            n_output=128,  # Total output size (L2/3 + L5)
            layer_ratio=(0.4, 0.4, 0.2),  # L4:L2/3:L5 proportions
        )
        cortex = LayeredCortex(config)

        # Process input
        output = cortex(input_spikes)

        # Route output by layer
        l23_size = cortex.l23_size
        cortico_output = output[:l23_size]  # To other cortex
        subcortical_output = output[l23_size:]  # To striatum/thalamus

    **Mixins Provide**:

    From LearningStrategyMixin:
        - add_strategy(), apply_learning() - Pluggable learning rules

    From DiagnosticsMixin:
        - check_health(), get_firing_rate() - Health monitoring

    From GrowthMixin:
        - add_neurons(), get_capacity_metrics() - Curriculum learning

    From NeuromodulatorMixin:
        - set_dopamine(), get_effective_learning_rate() - DA modulation

    **See Also**:
    - docs/patterns/mixins.md - Detailed mixin patterns
    - docs/decisions/adr-011-large-file-justification.md - Why single file
    - docs/patterns/state-management.md - LayeredCortexState usage
    """

    def __init__(self, config: LayeredCortexConfig):
        """Initialize layered cortex."""
        self.layer_config = config

        # Compute layer sizes
        self.l4_size, self.l23_size, self.l5_size = calculate_layer_sizes(
            config.n_output, config.l4_ratio, config.l23_ratio, config.l5_ratio
        )

        # Output is always both L2/3 and L5 (biological cortex has both pathways)
        actual_output = self.l23_size + self.l5_size

        # Create modified config for parent
        parent_config = NeuralComponentConfig(
            n_input=config.n_input,
            n_output=actual_output,
            dt_ms=config.dt_ms,
            axonal_delay_ms=config.axonal_delay_ms,  # Preserve axonal delay
            device=config.device,
        )

        # Store output size before parent init
        self._actual_output = actual_output

        # Call parent init
        super().__init__(parent_config)

        # Initialize layers
        self._init_layers()

        # Initialize inter-layer weights
        self._init_weights()

        # Initialize feedforward inhibition (FFI) - always enabled
        self.feedforward_inhibition = FeedforwardInhibition(
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

            # Use create_cortex_strategy() factory helper for STDP+BCM composite
            self.bcm_l4 = create_cortex_strategy(use_stdp=True, stdp_config=stdp_cfg, bcm_config=bcm_cfg)
            self.bcm_l23 = create_cortex_strategy(use_stdp=True, stdp_config=stdp_cfg, bcm_config=bcm_cfg)
            self.bcm_l5 = create_cortex_strategy(use_stdp=True, stdp_config=stdp_cfg, bcm_config=bcm_cfg)
        else:
            self.bcm_l4 = None
            self.bcm_l23 = None
            self.bcm_l5 = None

        # State
        self.state = LayeredCortexState()

        # Theta phase for encoding/retrieval modulation
        self._theta_phase = 0.0

        # Cumulative spike counters (for diagnostics across timesteps)
        self._cumulative_l4_spikes = 0
        self._cumulative_l23_spikes = 0
        self._cumulative_l5_spikes = 0

        # Intrinsic plasticity tracking (initialized in _init_layers)
        self._l23_threshold_offset: Optional[torch.Tensor] = None
        self._l23_activity_history: Optional[torch.Tensor] = None

        # Homeostasis for synaptic scaling and intrinsic plasticity
        homeostasis_config = UnifiedHomeostasisConfig(
            weight_budget=config.weight_budget * config.n_input,
            w_min=config.w_min,
            w_max=config.w_max,
            soft_normalization=config.soft_normalization,
            normalization_rate=config.normalization_rate,
            activity_target=config.activity_target,
            device=config.device,
        )
        self.homeostasis = UnifiedHomeostasis(homeostasis_config)

        # =====================================================================
        # ROBUSTNESS MECHANISMS (from RobustnessConfig)
        # =====================================================================
        self._init_robustness_mechanisms()
        self._init_gamma_attention()

    def _initialize_weights(self) -> torch.Tensor:
        """Placeholder - real weights in _init_weights."""
        return nn.Parameter(
            torch.zeros(self._actual_output, self.layer_config.n_input)
        )

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
        """
        cfg = self.layer_config

        # Create layer-specific neurons using factory functions
        self.l4_neurons = create_cortical_layer_neurons(self.l4_size, "L4", self.device)
        self.l23_neurons = create_cortical_layer_neurons(
            self.l23_size,
            "L2/3",
            self.device,
            adapt_increment=cfg.adapt_increment,  # SFA to prevent frozen attractors
            tau_adapt=cfg.adapt_tau,
        )
        self.l5_neurons = create_cortical_layer_neurons(self.l5_size, "L5", self.device)

        # =====================================================================
        # SHORT-TERM PLASTICITY for L2/3 recurrent connections
        # =====================================================================
        # L2/3 recurrent connections show SHORT-TERM DEPRESSION, preventing
        # frozen attractors. Without STD, the same neurons fire every timestep.
        # Always enabled (critical for preventing frozen attractors)
        device = torch.device(cfg.device)
        self.stp_l23_recurrent = ShortTermPlasticity(
            n_pre=self.l23_size,
            n_post=self.l23_size,
            config=STPConfig.from_type(STPType.DEPRESSING_FAST, dt=cfg.dt_ms),
            per_synapse=True,
        )
        self.stp_l23_recurrent.to(device)

        # =====================================================================
        # INTER-LAYER AXONAL DELAYS
        # =====================================================================
        # Create delay buffers for biological signal propagation within layers
        # L4→L2/3 delay: Short vertical projection (~2ms biologically)
        # L2/3→L5: Longer vertical projection (~2ms biologically)
        # Uses circular buffer mechanism from AxonalDelaysMixin
        self._l4_l23_delay_steps = int(cfg.l4_to_l23_delay_ms / cfg.dt_ms)
        self._l23_l5_delay_steps = int(cfg.l23_to_l5_delay_ms / cfg.dt_ms)

        # Initialize delay buffers (lazily initialized on first use)
        self._l4_l23_delay_buffer: Optional[torch.Tensor] = None
        self._l4_l23_delay_ptr: int = 0
        self._l23_l5_delay_buffer: Optional[torch.Tensor] = None
        self._l23_l5_delay_ptr: int = 0

    def _init_robustness_mechanisms(self) -> None:
        """Initialize robustness mechanisms from RobustnessConfig.

        These mechanisms are cortex-specific and NOT redundant with UnifiedHomeostasis:
        - E/I Balance: Critical for recurrent cortical stability

        Note: Activity regulation is handled by UnifiedHomeostasis.
        ConductanceLIF neurons provide natural gain control via shunting inhibition,
        so divisive normalization is not needed.
        """
        cfg = self.layer_config
        rob = cfg.robustness
        device = torch.device(cfg.device)

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
        cfg = self.layer_config
        device = torch.device(cfg.device)

        # Learnable phase preferences for each L2/3 neuron
        self.l23_phase_prefs = nn.Parameter(
            torch.rand(self.l23_size, device=device) * 2 * torch.pi
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
        device = torch.device(self.layer_config.device)
        cfg = self.layer_config

        # Expected number of active inputs given sparsity
        expected_active_l4 = max(1, int(self.l4_size * cfg.l4_sparsity))
        expected_active_l23 = max(1, int(self.l23_size * cfg.l23_sparsity))

        # Feedforward weights: positive, scaled so sparse input reaches threshold
        # With n_active inputs, threshold ~1.0, strength factor applied later:
        # target = threshold / (n_active * strength) ≈ 1.0 / (n_active * strength)
        # We initialize to mean ≈ target, with some variance for diversity

        # Input → L4: positive excitatory weights
        w_scale_input = 1.0 / max(1, int(cfg.n_input * 0.15))  # Assume 15% input sparsity
        self.w_input_l4 = nn.Parameter(
            torch.abs(
                WeightInitializer.gaussian(
                    n_output=self.l4_size,
                    n_input=cfg.n_input,
                    mean=0.0,
                    std=w_scale_input,
                    device=device
                )
            )
        )

        # L4 → L2/3: positive excitatory weights
        w_scale_l4_l23 = 1.0 / expected_active_l4
        self.w_l4_l23 = nn.Parameter(
            torch.abs(
                WeightInitializer.gaussian(
                    n_output=self.l23_size,
                    n_input=self.l4_size,
                    mean=0.0,
                    std=w_scale_l4_l23,
                    device=device
                )
            )
        )

        # L2/3 recurrent: SIGNED weights (compact E/I approximation)
        # Unlike feedforward connections which are positive-only (Dale's law),
        # recurrent lateral connections use signed weights to approximate the
        # mixed excitatory/inhibitory microcircuit within a cortical layer.
        # Positive weights = local excitation, negative weights = lateral inhibition.
        # Uses dedicated bounds [l23_recurrent_w_min, l23_recurrent_w_max] during learning.
        self.w_l23_recurrent = nn.Parameter(
            WeightInitializer.gaussian(
                n_output=self.l23_size,
                n_input=self.l23_size,
                mean=0.0,
                std=0.2,
                device=device
            )
        )
        with torch.no_grad():
            self.w_l23_recurrent.data.fill_diagonal_(0.0)

        # L2/3 → L5: positive excitatory weights
        w_scale_l23_l5 = 1.0 / expected_active_l23
        self.w_l23_l5 = nn.Parameter(
            torch.abs(
                WeightInitializer.gaussian(
                    n_output=self.l5_size,
                    n_input=self.l23_size,
                    mean=0.0,
                    std=w_scale_l23_l5,
                    device=device
                )
            )
        )

        # L2/3 inhibition: positive (inhibitory connections suppress)
        self.w_l23_inhib = nn.Parameter(
            WeightInitializer.ones(
                n_output=self.l23_size,
                n_input=self.l23_size,
                device=device
            ) * 0.3
        )
        with torch.no_grad():
            self.w_l23_inhib.data.fill_diagonal_(0.0)

        self.weights = self.w_input_l4

        # Note: Learning strategies (STDP+BCM) are created in __init__ as
        # self.bcm_l4, self.bcm_l23, self.bcm_l5 composite strategies

    def reset_state(self) -> None:
        """Reset all layer states.

        ADR-005: Uses 1D tensors (no batch dimension) for single-brain architecture.
        """
        dev = self.device

        # Reset neuron populations and STP using helpers
        self._reset_subsystems('l4_neurons', 'l23_neurons', 'l5_neurons', 'stp_l23_recurrent')

        # Note: No local oscillators to reset - phases come from Brain

        # Preserve oscillator signals if they exist (set via set_oscillator_phases)
        existing_phases = getattr(self.state, '_oscillator_phases', {})
        existing_signals = getattr(self.state, '_oscillator_signals', {})

        self.state = LayeredCortexState(
            l4_spikes=torch.zeros(self.l4_size, device=dev),
            l23_spikes=torch.zeros(self.l23_size, device=dev),
            l5_spikes=torch.zeros(self.l5_size, device=dev),
            l23_recurrent_activity=torch.zeros(self.l23_size, device=dev),
            l4_trace=torch.zeros(self.l4_size, device=dev),
            l23_trace=torch.zeros(self.l23_size, device=dev),
            l5_trace=torch.zeros(self.l5_size, device=dev),
            top_down_modulation=None,
            ffi_strength=0.0,
        )

        # Restore/initialize oscillator signals
        self.state._oscillator_phases = existing_phases
        self.state._oscillator_signals = existing_signals

        # Reset cumulative spike counters using helper
        self._reset_scalars(
            _cumulative_l4_spikes=0,
            _cumulative_l23_spikes=0,
            _cumulative_l5_spikes=0
        )

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
        # Store in state for forward() to access
        if not hasattr(self.state, '_oscillator_phases'):
            self.state._oscillator_phases = {}
            self.state._oscillator_signals = {}
        self.state._oscillator_phases = phases
        self.state._oscillator_signals = signals

        # Update theta phase for encoding/retrieval modulation
        self._theta_phase = phases.get('theta', 0.0)

        # Store effective gamma amplitude (pre-computed by OscillatorManager)
        # Automatic multiplicative coupling:
        # - Gamma modulated by ALL slower oscillators (delta, theta, alpha, beta)
        # OscillatorManager handles the multiplication, we just store the result.
        if coupled_amplitudes is not None:
            self.state._gamma_amplitude = coupled_amplitudes.get('gamma', 1.0)
        else:
            self.state._gamma_amplitude = 1.0

    # region Growth and Neurogenesis

    def add_neurons(
        self,
        n_new: int,
        initialization: str = 'sparse_random',
        sparsity: float = 0.1,
    ) -> None:
        """Add neurons to cortex, expanding all layers proportionally.

        This expands L4, L2/3, and L5 while maintaining layer ratios:
        - L4 expands by (l4_ratio * n_new)
        - L2/3 expands by (l23_ratio * n_new)
        - L5 expands by (l5_ratio * n_new)

        All inter-layer weights are expanded to accommodate new neurons.

        Args:
            n_new: Number of neurons to add to total cortex size
            initialization: Weight initialization strategy
            sparsity: Sparsity for new connections
        """
        # Calculate proportional growth for all layers
        l4_growth, l23_growth, l5_growth = calculate_layer_sizes(
            n_new, self.layer_config.l4_ratio, self.layer_config.l23_ratio, self.layer_config.l5_ratio
        )

        old_l4_size = self.l4_size
        old_l23_size = self.l23_size
        old_l5_size = self.l5_size

        new_l4_size = old_l4_size + l4_growth
        new_l23_size = old_l23_size + l23_growth
        new_l5_size = old_l5_size + l5_growth

        # Helper to create new weights
        def new_weights_for(n_out: int, n_in: int) -> torch.Tensor:
            if initialization == 'xavier':
                return WeightInitializer.xavier(n_out, n_in, device=self.device)
            elif initialization == 'sparse_random':
                return WeightInitializer.sparse_random(n_out, n_in, sparsity, device=self.device)
            else:
                return WeightInitializer.uniform(n_out, n_in, device=self.device)

        # 1. Expand input→L4 weights [l4, input]
        # Add rows for new L4 neurons
        new_input_l4 = new_weights_for(l4_growth, self.layer_config.n_input)
        self.w_input_l4 = nn.Parameter(
            torch.cat([self.w_input_l4.data, new_input_l4], dim=0)
        )

        # 2. Expand L4→L2/3 weights [l23, l4]
        # Add rows for new L2/3 neurons, columns for new L4 neurons
        new_l23_rows = new_weights_for(l23_growth, old_l4_size)
        expanded_l23_rows = torch.cat([self.w_l4_l23.data, new_l23_rows], dim=0)
        new_l4_cols = new_weights_for(new_l23_size, l4_growth)
        self.w_l4_l23 = nn.Parameter(
            torch.cat([expanded_l23_rows, new_l4_cols], dim=1)
        )

        # 3. Expand L2/3→L2/3 recurrent weights [l23, l23]
        # Add rows and columns for new L2/3 neurons
        new_l23_recurrent_rows = new_weights_for(l23_growth, old_l23_size)
        expanded_recurrent_rows = torch.cat([self.w_l23_recurrent.data, new_l23_recurrent_rows], dim=0)
        new_l23_recurrent_cols = new_weights_for(new_l23_size, l23_growth)
        self.w_l23_recurrent = nn.Parameter(
            torch.cat([expanded_recurrent_rows, new_l23_recurrent_cols], dim=1)
        )

        # 4. Expand L2/3→L5 weights [l5, l23]
        # Add rows for new L5 neurons, columns for new L2/3 neurons
        new_l5_rows = new_weights_for(l5_growth, old_l23_size)
        expanded_l5_rows = torch.cat([self.w_l23_l5.data, new_l5_rows], dim=0)
        new_l23_cols_to_l5 = new_weights_for(new_l5_size, l23_growth)
        self.w_l23_l5 = nn.Parameter(
            torch.cat([expanded_l5_rows, new_l23_cols_to_l5], dim=1)
        )

        # Update main weights reference (for base class compatibility)
        self.weights = self.w_l23_l5

        # 5. Expand neurons for all layers using factory functions
        self.l4_size = new_l4_size
        self.l4_neurons = create_cortical_layer_neurons(self.l4_size, "L4", self.device)

        self.l23_size = new_l23_size
        self.l23_neurons = create_cortical_layer_neurons(
            self.l23_size,
            "L2/3",
            self.device,
            adapt_increment=self.layer_config.adapt_increment,
            tau_adapt=self.layer_config.adapt_tau,
        )

        self.l5_size = new_l5_size
        self.l5_neurons = create_cortical_layer_neurons(self.l5_size, "L5", self.device)

        # 6. Update configs
        new_total_output = new_l23_size + new_l5_size

        self.config = replace(self.config, n_output=new_total_output)
        self.layer_config = replace(self.layer_config, n_output=new_total_output)

    # endregion

    # region Forward Pass (L4→L2/3→L5)

    def forward(
        self,
        input_spikes: torch.Tensor,
        top_down: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Process input through layered cortical circuit with continuous plasticity.

        This method both processes spikes AND applies synaptic plasticity. Learning
        happens continuously at each timestep, modulated by neuromodulators (dopamine).
        This is how biological cortex works - plasticity is part of the dynamics,
        not a separate training phase.

        Args:
            input_spikes: Input spike tensor [n_input] (1D per ADR-005)
            top_down: Optional top-down modulation [l23_size] (1D)

        Returns:
            Output spikes [l23_size + l5_size] - concatenated L2/3 and L5 outputs (1D)

        Note:
            Theta modulation and timestep (dt_ms) computed internally from config
        """
        # Get timestep from config for temporal dynamics
        dt = self.config.dt_ms

        # Compute theta modulation from oscillator phase (set by Brain)
        # encoding_mod: high at theta peak (0°), low at trough (180°)
        # retrieval_mod: low at theta peak, high at trough
        encoding_mod = 0.5 * (1.0 + math.cos(self._theta_phase))
        retrieval_mod = 0.5 * (1.0 - math.cos(self._theta_phase))

        # ADR-005: Expect 1D tensors (no batch dimension)
        assert input_spikes.dim() == 1, (
            f"LayeredCortex.forward: Expected 1D input (ADR-005), got shape {input_spikes.shape}. "
            f"Thalia uses single-brain architecture with no batch dimension."
        )
        assert input_spikes.shape[0] == self.layer_config.n_input, (
            f"LayeredCortex.forward: input_spikes has shape {input_spikes.shape} "
            f"but n_input={self.layer_config.n_input}. Check that input matches cortex config."
        )

        if top_down is not None:
            assert top_down.dim() == 1, (
                f"LayeredCortex.forward: Expected 1D top_down (ADR-005), got shape {top_down.shape}"
            )
            assert top_down.shape[0] == self.l23_size, (
                f"LayeredCortex.forward: top_down has shape {top_down.shape} "
                f"but L2/3 size={self.l23_size}. Top-down must match L2/3 for modulation."
            )

        if self.state.l4_spikes is None:
            self.reset_state()

        cfg = self.layer_config

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

        if hasattr(self.state, '_oscillator_signals') and self.state._oscillator_signals is not None:
            alpha_signal = self.state._oscillator_signals.get('alpha', 0.0)

            # Alpha signal ranges [-1, 1], convert to suppression [0, 0.5]
            # High positive alpha (near 1.0) → max suppression (50%)
            # Low/negative alpha → minimal suppression
            alpha_magnitude = max(0.0, alpha_signal)  # Only positive values
            alpha_suppression = 1.0 - (alpha_magnitude * 0.5)  # Scale to 50-100%

            # Automatic gamma modulation: ALL slower oscillators affect gamma
            # This gives emergent multi-oscillator coupling (e.g., theta-alpha-beta-gamma)
            if hasattr(self.state, '_gamma_amplitude'):
                gamma_modulation = self.state._gamma_amplitude

            # Store for diagnostics
            self.state.alpha_suppression = alpha_suppression
            self.state.gamma_modulation = gamma_modulation

        # Apply alpha suppression to input (early gating)
        gated_input_spikes = input_spikes * alpha_suppression

        # L4: Input processing with conductance-based neurons
        # Compute excitatory conductance from input
        l4_g_exc = (
            torch.matmul(self.w_input_l4, gated_input_spikes.float())
            * cfg.input_to_l4_strength
        )
        l4_g_exc = l4_g_exc * (0.5 + 0.5 * encoding_mod)

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
        ne_gain = 1.0 + NE_GAIN_RANGE * ne_level
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
                max_delay_steps = max(1, self._l4_l23_delay_steps * 2 + 1)
                self._l4_l23_delay_buffer = torch.zeros(
                    max_delay_steps, self.l4_size,
                    device=l4_spikes.device, dtype=torch.bool
                )
                self._l4_l23_delay_ptr = 0

            # Store current spikes in circular buffer
            self._l4_l23_delay_buffer[self._l4_l23_delay_ptr] = l4_spikes

            # Retrieve delayed spikes
            read_idx = (self._l4_l23_delay_ptr - self._l4_l23_delay_steps) % self._l4_l23_delay_buffer.shape[0]
            l4_spikes_delayed = self._l4_l23_delay_buffer[read_idx]

            # Advance pointer
            self._l4_l23_delay_ptr = (self._l4_l23_delay_ptr + 1) % self._l4_l23_delay_buffer.shape[0]
        else:
            l4_spikes_delayed = l4_spikes

        # L2/3: Processing with recurrence
        # NOTE: Use delayed L4 spikes for biological accuracy
        l23_ff = (
            torch.matmul(l4_spikes_delayed.float(), self.w_l4_l23.t())
            * cfg.l4_to_l23_strength
        )

        # Feedforward inhibition (always enabled)
        ffi = self.feedforward_inhibition.compute(input_spikes, return_tensor=False)
        raw_ffi = ffi.item() if hasattr(ffi, "item") else float(ffi)
        self.state.ffi_strength = min(
            1.0, raw_ffi / self.feedforward_inhibition.max_inhibition
        )
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
        ach_recurrent_modulation = 1.0 - 0.8 * max(0.0, ach_level - 0.5) / 0.5

        if self.state.l23_recurrent_activity is not None:
            recurrent_scale = 0.5 + 0.5 * retrieval_mod

            # Apply STP to recurrent connections (always enabled)
            stp_efficacy = self.stp_l23_recurrent(self.state.l23_recurrent_activity.float())  # [l23_size, l23_size]

            effective_w_rec = self.w_l23_recurrent * stp_efficacy
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
            l23_inhib = torch.matmul(self.w_l23_inhib, self.state.l23_spikes.float())

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
        cfg = self.layer_config
        if (cfg.homeostasis_enabled and
            self._l23_threshold_offset is not None):
            l23_input = l23_input - self._l23_threshold_offset

        l23_spikes, _ = self.l23_neurons(F.relu(l23_input), F.relu(l23_input) * 0.25)
        l23_spikes = self._apply_sparsity_1d(l23_spikes, cfg.l23_sparsity)
        self.state.l23_spikes = l23_spikes

        # Inter-layer shape check: L2/3 → L5
        assert l23_spikes.shape == (self.l23_size,), (
            f"LayeredCortex: L2/3 spikes have shape {l23_spikes.shape} "
            f"but expected ({self.l23_size},). "
            f"Check L2/3 sparsity or L4→L2/3 weights shape."
        )

        # Gamma-phase attention: Modulate L2/3 spikes by gamma phase from Brain
        # Always enabled for spike-native attention
        if hasattr(self.state, '_oscillator_phases'):
            gamma_phase = self.state._oscillator_phases.get('gamma', 0.0)

            # Compute phase-based gating for each L2/3 neuron
            phase_diff = torch.abs(self.l23_phase_prefs - gamma_phase)
            phase_diff = torch.min(phase_diff, 2 * math.pi - phase_diff)

            # Gaussian gating based on phase proximity
            gamma_gate = torch.exp(-phase_diff ** 2 / (2 * self.gamma_attention_width ** 2))

            # Modulate L2/3 spikes (attention without Q/K/V projections!)
            l23_spikes = l23_spikes * gamma_gate

            # Store gating in state for diagnostics
            self.state.gamma_attention_phase = gamma_phase
            self.state.gamma_attention_gate = gamma_gate

        # Update recurrent activity trace
        if self.state.l23_recurrent_activity is not None:
            self.state.l23_recurrent_activity = (
                self.state.l23_recurrent_activity * cfg.l23_recurrent_decay
                + l23_spikes.float()
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
                max_delay_steps = max(1, self._l23_l5_delay_steps * 2 + 1)
                self._l23_l5_delay_buffer = torch.zeros(
                    max_delay_steps, self.l23_size,
                    device=l23_spikes.device, dtype=torch.bool
                )
                self._l23_l5_delay_ptr = 0

            # Store current spikes in circular buffer
            self._l23_l5_delay_buffer[self._l23_l5_delay_ptr] = l23_spikes

            # Retrieve delayed spikes
            read_idx = (self._l23_l5_delay_ptr - self._l23_l5_delay_steps) % self._l23_l5_delay_buffer.shape[0]
            l23_spikes_delayed = self._l23_l5_delay_buffer[read_idx]

            # Advance pointer
            self._l23_l5_delay_ptr = (self._l23_l5_delay_ptr + 1) % self._l23_l5_delay_buffer.shape[0]
        else:
            l23_spikes_delayed = l23_spikes

        # L5: Subcortical output (conductance-based)
        # NOTE: Use delayed L2/3 spikes for biological accuracy
        l5_g_exc = (
            torch.matmul(self.w_l23_l5, l23_spikes_delayed.float())
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

        # Update cumulative spike counters (for diagnostics)
        self._cumulative_l4_spikes += int(l4_spikes.sum().item())
        self._cumulative_l23_spikes += int(l23_spikes.sum().item())
        self._cumulative_l5_spikes += int(l5_spikes.sum().item())

        # Update STDP traces using utility function
        if self.state.l4_trace is not None:
            update_trace(self.state.l4_trace, l4_spikes, tau=cfg.tau_plus_ms, dt=dt)
        if self.state.l23_trace is not None:
            update_trace(self.state.l23_trace, l23_spikes, tau=cfg.tau_plus_ms, dt=dt)
        if self.state.l5_trace is not None:
            update_trace(self.state.l5_trace, l5_spikes, tau=cfg.tau_plus_ms, dt=dt)

        self.state.spikes = l5_spikes

        # Store input for plasticity
        self.state.input_spikes = input_spikes

        # Apply continuous plasticity (learning happens as part of forward dynamics)
        self._apply_plasticity()

        # Construct output: always concatenate L2/3 and L5 (biological cortex has both pathways)
        output = torch.cat([l23_spikes, l5_spikes], dim=-1)

        # Apply axonal delay (biological reality: ALL neural connections have delays)
        delayed_output = self._apply_axonal_delay(output.bool(), dt)

        # ADR-005: Return 1D tensor as bool spikes
        return delayed_output

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

        cfg = self.layer_config

        # Get dopamine-modulated learning rate
        base_lr = cfg.stdp_lr
        effective_lr = self.get_effective_learning_rate(base_lr)

        # Early exit if learning rate is too small
        if effective_lr < 1e-8:
            self.state.last_plasticity_delta = 0.0
            return

        # NOTE: All neuromodulators (DA, ACh, NE) are now managed centrally by Brain.
        # VTA updates dopamine, LC updates NE, NB updates ACh.
        # Brain broadcasts to all regions every timestep via _update_neuromodulators().
        # No local decay needed.

        # Get 1D versions of spike tensors for torch.outer
        l4_spikes = ensure_1d(self.state.l4_spikes)
        l23_spikes = ensure_1d(self.state.l23_spikes) if self.state.l23_spikes is not None else None
        l5_spikes = ensure_1d(self.state.l5_spikes) if self.state.l5_spikes is not None else None
        input_spikes = ensure_1d(self.state.input_spikes)

        total_change = 0.0

        # Use STDP+BCM composite strategies for proper spike-timing-dependent learning
        # Input → L4
        if self.bcm_l4 is not None:
            updated_weights, _ = self.bcm_l4.compute_update(
                weights=self.w_input_l4.data,
                pre=input_spikes,
                post=l4_spikes,
                learning_rate=effective_lr,
            )
            dw = updated_weights - self.w_input_l4.data
            with torch.no_grad():
                self.w_input_l4.data.copy_(updated_weights)
                clamp_weights(self.w_input_l4.data, cfg.w_min, cfg.w_max)
            total_change += dw.abs().mean().item()

        # L4 → L2/3
        if l23_spikes is not None and self.bcm_l23 is not None:
            updated_weights, _ = self.bcm_l23.compute_update(
                weights=self.w_l4_l23.data,
                pre=l4_spikes,
                post=l23_spikes,
                learning_rate=effective_lr,
            )
            dw = updated_weights - self.w_l4_l23.data
            with torch.no_grad():
                self.w_l4_l23.data.copy_(updated_weights)
                clamp_weights(self.w_l4_l23.data, cfg.w_min, cfg.w_max)
            total_change += dw.abs().mean().item()

            # L2/3 recurrent (signed weights - compact E/I approximation)
            # Uses dedicated bounds [l23_recurrent_w_min, l23_recurrent_w_max] to allow
            # both excitatory and inhibitory-like lateral connections.
            # This is a simplification of explicit E/I interneuron populations.
            updated_weights, _ = self.bcm_l23.compute_update(
                weights=self.w_l23_recurrent.data,
                pre=l23_spikes,
                post=l23_spikes,
                learning_rate=effective_lr * 0.5,  # Reduced for recurrent stability
            )
            dw = updated_weights - self.w_l23_recurrent.data

            with torch.no_grad():
                self.w_l23_recurrent.data.copy_(updated_weights)
                self.w_l23_recurrent.data.fill_diagonal_(0.0)
                clamp_weights(
                    self.w_l23_recurrent.data,
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
                    self._l23_activity_history = torch.zeros(self.l23_size, device=l23_spikes_1d.device)
                if self._l23_threshold_offset is None:
                    self._l23_threshold_offset = torch.zeros(self.l23_size, device=l23_spikes_1d.device)

                # Update activity history (exponential moving average)
                from thalia.regulation.learning_constants import EMA_DECAY_FAST
                self._l23_activity_history = (
                    EMA_DECAY_FAST * self._l23_activity_history + (1 - EMA_DECAY_FAST) * l23_spikes_1d
                )

                # Compute threshold modulation using UnifiedHomeostasis
                threshold_mod = self.homeostasis.compute_excitability_modulation(
                    activity_history=self._l23_activity_history,
                    tau=100.0
                )
                self._l23_threshold_offset = threshold_mod.clamp(-0.5, 0.5)

            # L2/3 → L5
            if l5_spikes is not None and self.bcm_l5 is not None:
                updated_weights, _ = self.bcm_l5.compute_update(
                    weights=self.w_l23_l5.data,
                    pre=l23_spikes,
                    post=l5_spikes,
                    learning_rate=effective_lr,
                )
                dw = updated_weights - self.w_l23_l5.data
                with torch.no_grad():
                    self.w_l23_l5.data.copy_(updated_weights)
                    clamp_weights(self.w_l23_l5.data, cfg.w_min, cfg.w_max)
                total_change += dw.abs().mean().item()

        # Store for monitoring
        self.state.last_plasticity_delta = total_change

    def get_layer_outputs(self) -> Dict[str, Optional[torch.Tensor]]:
        """Get outputs from all layers."""
        return {
            "L4": self.state.l4_spikes,
            "L2/3": self.state.l23_spikes,
            "L5": self.state.l5_spikes,
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
        """Get layer-specific diagnostics using DiagnosticsMixin helpers.

        Note: Reports both instantaneous (l4_active_count) and cumulative
        (l4_cumulative_spikes) counts. During consolidation phases with
        zero input, instantaneous L4 will be 0 but cumulative shows
        total activity since last reset.
        """
        cfg = self.layer_config

        # Custom metrics specific to cortex
        custom = {
            "l4_size": self.l4_size,
            "l23_size": self.l23_size,
            "l5_size": self.l5_size,
            # Config weight bounds for reference
            "config_w_min": cfg.w_min,
            "config_w_max": cfg.w_max,
            "config_l23_rec_w_min": cfg.l23_recurrent_w_min,
            "config_l23_rec_w_max": cfg.l23_recurrent_w_max,
            # Cumulative spike counts (since last reset_state)
            "l4_cumulative_spikes": getattr(self, "_cumulative_l4_spikes", 0),
            "l23_cumulative_spikes": getattr(self, "_cumulative_l23_spikes", 0),
            "l5_cumulative_spikes": getattr(self, "_cumulative_l5_spikes", 0),
        }

        # Recurrent activity
        if self.state.l23_recurrent_activity is not None:
            custom["l23_recurrent_mean"] = self.state.l23_recurrent_activity.mean().item()

        # Robustness mechanisms (E/I balance only)
        if self.ei_balance is not None:
            ei_diag = self.ei_balance.get_diagnostics()
            custom["robustness_ei_ratio"] = ei_diag.get("current_ratio", 0.0)
            custom["robustness_ei_scale"] = ei_diag.get("inh_scale", 1.0)
            custom["robustness_ei_status"] = ei_diag.get("status", "unknown")

        # Use collect_standard_diagnostics for weight and spike statistics
        return self.collect_standard_diagnostics(
            region_name="cortex",
            weight_matrices={
                "input_l4": self.w_input_l4.data,
                "l4_l23": self.w_l4_l23.data,
                "l23_rec": self.w_l23_recurrent.data,
                "l23_l5": self.w_l23_l5.data,
            },
            spike_tensors={
                "l4": self.state.l4_spikes,
                "l23": self.state.l23_spikes,
                "l5": self.state.l5_spikes,
            },
            custom_metrics=custom,
        )

    def get_full_state(self) -> Dict[str, Any]:
        """Get complete state for checkpointing.

        Returns state dictionary with keys:
        - weights: All inter-layer weight matrices
        - region_state: Current spikes, membrane potentials, traces
        - learning_state: BCM thresholds, STP state
        - neuromodulator_state: Current dopamine, norepinephrine, etc.
        - config: Configuration for validation
        """
        state_dict = {
            "weights": {
                "w_input_l4": self.w_input_l4.data.clone(),
                "w_l4_l23": self.w_l4_l23.data.clone(),
                "w_l23_recurrent": self.w_l23_recurrent.data.clone(),
                "w_l23_l5": self.w_l23_l5.data.clone(),
                "w_l23_inhib": self.w_l23_inhib.data.clone(),
            },
            "region_state": {
                "l4_neurons": self.l4_neurons.get_state(),
                "l23_neurons": self.l23_neurons.get_state(),
                "l5_neurons": self.l5_neurons.get_state(),
                "l4_spikes": self.state.l4_spikes.clone() if self.state.l4_spikes is not None else None,
                "l23_spikes": self.state.l23_spikes.clone() if self.state.l23_spikes is not None else None,
                "l5_spikes": self.state.l5_spikes.clone() if self.state.l5_spikes is not None else None,
                "l4_trace": self.state.l4_trace.clone() if self.state.l4_trace is not None else None,
                "l23_trace": self.state.l23_trace.clone() if self.state.l23_trace is not None else None,
                "l5_trace": self.state.l5_trace.clone() if self.state.l5_trace is not None else None,
                "l23_recurrent_activity": self.state.l23_recurrent_activity.clone() if self.state.l23_recurrent_activity is not None else None,
                # Inter-layer axonal delay buffers
                "l4_l23_delay_buffer": self._l4_l23_delay_buffer.clone() if self._l4_l23_delay_buffer is not None else None,
                "l4_l23_delay_ptr": self._l4_l23_delay_ptr,
                "l23_l5_delay_buffer": self._l23_l5_delay_buffer.clone() if self._l23_l5_delay_buffer is not None else None,
                "l23_l5_delay_ptr": self._l23_l5_delay_ptr,
            },
            "learning_state": {},
            "neuromodulator_state": {
                "dopamine": self.state.dopamine,
                "norepinephrine": self.state.norepinephrine,
                "acetylcholine": self.state.acetylcholine,
            },
            "config": {
                "n_input": self.config.n_input,
                "n_output": self.config.n_output,
                "l4_size": self.l4_size,
                "l23_size": self.l23_size,
                "l5_size": self.l5_size,
            },
        }

        # BCM state (thresholds)
        if self.bcm_l4 is not None and hasattr(self.bcm_l4, 'theta') and self.bcm_l4.theta is not None:
            state_dict["learning_state"]["bcm_l4_theta"] = self.bcm_l4.theta.clone()
        if self.bcm_l23 is not None and hasattr(self.bcm_l23, 'theta') and self.bcm_l23.theta is not None:
            state_dict["learning_state"]["bcm_l23_theta"] = self.bcm_l23.theta.clone()
        if self.bcm_l5 is not None and hasattr(self.bcm_l5, 'theta') and self.bcm_l5.theta is not None:
            state_dict["learning_state"]["bcm_l5_theta"] = self.bcm_l5.theta.clone()

        # STP state (always present)
        state_dict["learning_state"]["stp_l23_recurrent"] = self.stp_l23_recurrent.get_state()

        return state_dict

    def load_full_state(self, state: Dict[str, Any]) -> None:
        """Load complete state from checkpoint.

        Args:
            state: State dictionary from get_full_state()

        Raises:
            ValueError: If config dimensions don't match
        """
        # Validate config compatibility
        config = state.get("config", {})
        if config.get("n_input") != self.config.n_input:
            raise CheckpointError(f"Config mismatch: n_input {config.get('n_input')} != {self.config.n_input}")
        if config.get("n_output") != self.config.n_output:
            raise CheckpointError(f"Config mismatch: n_output {config.get('n_output')} != {self.config.n_output}")
        if config.get("l4_size") != self.l4_size:
            raise CheckpointError(f"Config mismatch: l4_size {config.get('l4_size')} != {self.l4_size}")
        if config.get("l23_size") != self.l23_size:
            raise CheckpointError(f"Config mismatch: l23_size {config.get('l23_size')} != {self.l23_size}")
        if config.get("l5_size") != self.l5_size:
            raise CheckpointError(f"Config mismatch: l5_size {config.get('l5_size')} != {self.l5_size}")

        # Restore weights
        weights = state["weights"]
        self.w_input_l4.data.copy_(weights["w_input_l4"].to(self.device))
        self.w_l4_l23.data.copy_(weights["w_l4_l23"].to(self.device))
        self.w_l23_recurrent.data.copy_(weights["w_l23_recurrent"].to(self.device))
        self.w_l23_l5.data.copy_(weights["w_l23_l5"].to(self.device))
        self.w_l23_inhib.data.copy_(weights["w_l23_inhib"].to(self.device))

        # Restore neuron states
        region_state = state["region_state"]
        self.l4_neurons.load_state(region_state["l4_neurons"])
        self.l23_neurons.load_state(region_state["l23_neurons"])
        self.l5_neurons.load_state(region_state["l5_neurons"])

        # Restore region state
        if region_state["l4_spikes"] is not None:
            self.state.l4_spikes = region_state["l4_spikes"].to(self.device)
        if region_state["l23_spikes"] is not None:
            self.state.l23_spikes = region_state["l23_spikes"].to(self.device)
        if region_state["l5_spikes"] is not None:
            self.state.l5_spikes = region_state["l5_spikes"].to(self.device)
        if region_state["l4_trace"] is not None:
            self.state.l4_trace = region_state["l4_trace"].to(self.device)
        if region_state["l23_trace"] is not None:
            self.state.l23_trace = region_state["l23_trace"].to(self.device)
        if region_state["l5_trace"] is not None:
            self.state.l5_trace = region_state["l5_trace"].to(self.device)
        if region_state["l23_recurrent_activity"] is not None:
            self.state.l23_recurrent_activity = region_state["l23_recurrent_activity"].to(self.device)

        # Restore inter-layer delay buffers (if present in checkpoint)
        if "l4_l23_delay_buffer" in region_state and region_state["l4_l23_delay_buffer"] is not None:
            self._l4_l23_delay_buffer = region_state["l4_l23_delay_buffer"].to(self.device)
            self._l4_l23_delay_ptr = region_state["l4_l23_delay_ptr"]
        if "l23_l5_delay_buffer" in region_state and region_state["l23_l5_delay_buffer"] is not None:
            self._l23_l5_delay_buffer = region_state["l23_l5_delay_buffer"].to(self.device)
            self._l23_l5_delay_ptr = region_state["l23_l5_delay_ptr"]

        # Restore BCM thresholds
        learning_state = state["learning_state"]
        if "bcm_l4_theta" in learning_state and self.bcm_l4 is not None:
            self.bcm_l4.theta.copy_(learning_state["bcm_l4_theta"].to(self.device))
        if "bcm_l23_theta" in learning_state and self.bcm_l23 is not None:
            self.bcm_l23.theta.copy_(learning_state["bcm_l23_theta"].to(self.device))
        if "bcm_l5_theta" in learning_state and self.bcm_l5 is not None:
            self.bcm_l5.theta.copy_(learning_state["bcm_l5_theta"].to(self.device))

        # Restore STP state (always present)
        if "stp_l23_recurrent" in learning_state:
            self.stp_l23_recurrent.load_state(learning_state["stp_l23_recurrent"])

        # Restore neuromodulators
        neuromod = state["neuromodulator_state"]

        self.state.dopamine = neuromod["dopamine"]
        self.state.norepinephrine = neuromod["norepinephrine"]
        self.state.acetylcholine = neuromod["acetylcholine"]

    # endregion
