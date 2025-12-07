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

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from thalia.regions.base import BrainRegion, RegionConfig, LearningRule
from thalia.core.neuron import LIFNeuron, LIFConfig
from thalia.core.stp import ShortTermPlasticity, STPConfig, STPType
from thalia.core.weight_init import WeightInitializer
from thalia.regions.theta_dynamics import FeedforwardInhibition
from thalia.learning.bcm import BCMRule, BCMConfig
from thalia.learning import LearningStrategyMixin, STDPStrategy, STDPConfig
from thalia.core.utils import ensure_batch_dim, ensure_1d, clamp_weights, assert_single_instance
from thalia.core.traces import update_trace
from thalia.core.diagnostics_mixin import DiagnosticsMixin
from thalia.learning.ei_balance import LayerEIBalance
from thalia.core.normalization import DivisiveNormalization
from thalia.learning.intrinsic_plasticity import PopulationIntrinsicPlasticity

from .config import LayeredCortexConfig, LayeredCortexState


class LayeredCortex(LearningStrategyMixin, DiagnosticsMixin, BrainRegion):
    """
    Multi-layer cortical microcircuit with proper layer separation.

    This implements a canonical cortical column with:
    - L4: Input layer (receives external input)
    - L2/3: Processing layer (recurrent, outputs to other cortex)
    - L5: Output layer (outputs to subcortical structures)

    The key insight is that OUTPUT to next region comes from a different
    layer than the one receiving RECURRENT feedback, solving the
    contamination problem in single-layer models.

    Usage:
        config = LayeredCortexConfig(n_input=256, n_output=64)
        cortex = LayeredCortex(config)

        # Process input
        output = cortex.forward(input_spikes)

        # Output contains both L2/3 (cortico-cortical) and L5 (subcortical)
        l23_out = output[:, :cortex.l23_size]
        l5_out = output[:, cortex.l23_size:]
    
    Mixins Provide:
    ---------------
    From LearningStrategyMixin:
        - add_strategy(strategy) → None
        - apply_learning(pre, post, **kwargs) → Dict
        - Pluggable learning rules (Hebbian, STDP, BCM)
    
    From DiagnosticsMixin:
        - check_health() → HealthMetrics
        - get_firing_rate(spikes) → float
        - check_weight_health(weights, name) → WeightHealth
        - detect_runaway_excitation(spikes) → bool
    
    From BrainRegion (abstract base):
        - forward(input, **kwargs) → Tensor [must implement]
        - reset_state() → None
        - get_diagnostics() → Dict
        - Neuromodulator control methods
    
    See Also:
        docs/patterns/mixins.md for detailed mixin patterns
        docs/patterns/state-management.md for LayeredCortexState
    """

    def __init__(self, config: LayeredCortexConfig):
        """Initialize layered cortex."""
        self.layer_config = config

        # Compute layer sizes
        self.l4_size = int(config.n_output * config.l4_ratio)
        self.l23_size = int(config.n_output * config.l23_ratio)
        self.l5_size = int(config.n_output * config.l5_ratio)

        # Actual output size depends on dual_output setting
        if config.dual_output:
            actual_output = self.l23_size + self.l5_size
        elif config.output_layer == "L5":
            actual_output = self.l5_size
        else:
            actual_output = self.l23_size

        # Create modified config for parent
        parent_config = RegionConfig(
            n_input=config.n_input,
            n_output=actual_output,
            dt=config.dt,
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

        # Initialize feedforward inhibition (FFI)
        if config.ffi_enabled:
            self.feedforward_inhibition = FeedforwardInhibition(
                threshold=config.ffi_threshold,
                max_inhibition=config.ffi_strength * 10.0,
                decay_rate=1.0 - (1.0 / config.ffi_tau),
            )
        else:
            self.feedforward_inhibition = None

        # BCM for each layer
        if config.bcm_enabled:
            device = torch.device(config.device)
            bcm_cfg = config.bcm_config or BCMConfig(
                tau_theta=config.bcm_tau_theta,
                theta_init=config.bcm_theta_init,
            )
            self.bcm_l4 = BCMRule(n_post=self.l4_size, config=bcm_cfg)
            self.bcm_l4.to(device)
            self.bcm_l23 = BCMRule(n_post=self.l23_size, config=bcm_cfg)
            self.bcm_l23.to(device)
            self.bcm_l5 = BCMRule(n_post=self.l5_size, config=bcm_cfg)
            self.bcm_l5.to(device)
        else:
            self.bcm_l4 = None
            self.bcm_l23 = None
            self.bcm_l5 = None

        # State
        self.state = LayeredCortexState()

        # Cumulative spike counters (for diagnostics across timesteps)
        self._cumulative_l4_spikes = 0
        self._cumulative_l23_spikes = 0
        self._cumulative_l5_spikes = 0

        # Intrinsic plasticity tracking (initialized in _init_layers)
        self._l23_threshold_offset: Optional[torch.Tensor] = None
        self._l23_activity_history: Optional[torch.Tensor] = None

        # =====================================================================
        # ROBUSTNESS MECHANISMS (from RobustnessConfig)
        # =====================================================================
        self._init_robustness_mechanisms()

    def _get_learning_rule(self) -> LearningRule:
        """Cortex uses unsupervised Hebbian learning."""
        return LearningRule.HEBBIAN

    def _initialize_weights(self) -> torch.Tensor:
        """Placeholder - real weights in _init_weights."""
        return nn.Parameter(
            torch.zeros(self._actual_output, self.layer_config.n_input)
        )

    def _create_neurons(self):
        """Placeholder - neurons created in _init_layers."""
        return None

    def _init_layers(self) -> None:
        """Initialize LIF neurons for each layer."""
        cfg = self.layer_config

        l4_config = LIFConfig(tau_mem=15.0, v_threshold=1.0)
        # L2/3 gets spike-frequency adaptation to prevent frozen attractors
        l23_config = LIFConfig(
            tau_mem=20.0,
            v_threshold=1.0,
            adapt_increment=cfg.l23_adapt_increment,  # SFA enabled!
            tau_adapt=cfg.l23_adapt_tau,
        )
        l5_config = LIFConfig(tau_mem=20.0, v_threshold=0.9)

        self.l4_neurons = LIFNeuron(self.l4_size, l4_config)
        self.l23_neurons = LIFNeuron(self.l23_size, l23_config)
        self.l5_neurons = LIFNeuron(self.l5_size, l5_config)

        # =====================================================================
        # SHORT-TERM PLASTICITY for L2/3 recurrent connections
        # =====================================================================
        # L2/3 recurrent connections show SHORT-TERM DEPRESSION, preventing
        # frozen attractors. Without STD, the same neurons fire every timestep.
        if cfg.stp_l23_recurrent_enabled:
            device = torch.device(cfg.device)
            self.stp_l23_recurrent = ShortTermPlasticity(
                n_pre=self.l23_size,
                n_post=self.l23_size,
                config=STPConfig.from_type(STPType.DEPRESSING_FAST, dt=cfg.dt),
                per_synapse=True,
            )
            self.stp_l23_recurrent.to(device)
        else:
            self.stp_l23_recurrent = None

    def _init_robustness_mechanisms(self) -> None:
        """Initialize robustness mechanisms from RobustnessConfig.

        These mechanisms provide hyperparameter robustness similar to
        biological homeostatic regulation:
        - E/I Balance: Tracks excitation vs inhibition, scales inhibitory gain
        - Divisive Normalization: Automatic gain control on L4 inputs
        - Population Intrinsic Plasticity: Activity-dependent threshold adaptation
        """
        cfg = self.layer_config
        rob = cfg.robustness
        device = torch.device(cfg.device)

        # Default: no robustness mechanisms
        self.ei_balance: Optional[LayerEIBalance] = None
        self.divisive_norm_l4: Optional[DivisiveNormalization] = None
        self.pop_intrinsic_plasticity: Optional[PopulationIntrinsicPlasticity] = None

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

        # Divisive Normalization for L4 input processing
        # Provides automatic gain control regardless of input intensity
        if rob.enable_divisive_norm:
            self.divisive_norm_l4 = DivisiveNormalization(
                config=rob.divisive_norm,
                n_features=self.l4_size,
                device=device,
            )

        # Population Intrinsic Plasticity for L2/3
        # Global excitability modulation based on population activity
        if rob.enable_intrinsic_plasticity:
            self.pop_intrinsic_plasticity = PopulationIntrinsicPlasticity(
                config=rob.intrinsic_plasticity,
            )

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

        # Initialize learning strategy (STDP for cortical learning)
        # We use a single STDP strategy instance that we'll apply to different
        # weight matrices (input->L4, L4->L2/3, L2/3->L5, L2/3 recurrent)
        self.learning_strategy = STDPStrategy(
            STDPConfig(
                learning_rate=cfg.stdp_lr,
                a_plus=0.01,
                a_minus=0.012,
                tau_plus=20.0,
                tau_minus=20.0,
                dt=cfg.dt,
                w_min=cfg.w_min,
                w_max=cfg.w_max,
                soft_bounds=cfg.soft_bounds,
            )
        )

    def reset_state(self) -> None:
        """Reset all layer states.

        THALIA enforces batch_size=1 for single-instance architecture.
        For parallel evaluation, create multiple LayeredCortex instances.
        """
        dev = self.device
        batch_size = 1

        self.l4_neurons.reset_state()
        self.l23_neurons.reset_state()
        self.l5_neurons.reset_state()

        # Reset STP state for L2/3 recurrent
        if self.stp_l23_recurrent is not None:
            self.stp_l23_recurrent.reset_state()

        self.state = LayeredCortexState(
            l4_spikes=torch.zeros(batch_size, self.l4_size, device=dev),
            l23_spikes=torch.zeros(batch_size, self.l23_size, device=dev),
            l5_spikes=torch.zeros(batch_size, self.l5_size, device=dev),
            l23_recurrent_activity=torch.zeros(batch_size, self.l23_size, device=dev),
            l4_trace=torch.zeros(batch_size, self.l4_size, device=dev),
            l23_trace=torch.zeros(batch_size, self.l23_size, device=dev),
            l5_trace=torch.zeros(batch_size, self.l5_size, device=dev),
            top_down_modulation=None,
            ffi_strength=0.0,
        )

        # Reset cumulative spike counters (for diagnostics across timesteps)
        self._cumulative_l4_spikes = 0
        self._cumulative_l23_spikes = 0
        self._cumulative_l5_spikes = 0

        # Note: FFI state decays naturally, no hard reset needed

    def forward(
        self,
        input_spikes: torch.Tensor,
        dt: float = 1.0,
        encoding_mod: float = 1.0,
        retrieval_mod: float = 1.0,
        top_down: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Process input through layered cortical circuit with continuous plasticity.

        This method both processes spikes AND applies synaptic plasticity. Learning
        happens continuously at each timestep, modulated by neuromodulators (dopamine).
        This is how biological cortex works - plasticity is part of the dynamics,
        not a separate training phase.
        """
        input_spikes = ensure_batch_dim(input_spikes)

        batch_size = input_spikes.shape[0]

        # Enforce single-instance architecture
        assert_single_instance(batch_size, "LayeredCortex.forward")

        # =====================================================================
        # SHAPE ASSERTIONS - catch dimension mismatches early with clear messages
        # =====================================================================
        assert input_spikes.shape[-1] == self.layer_config.n_input, (
            f"LayeredCortex.forward: input_spikes has shape {input_spikes.shape} "
            f"but n_input={self.layer_config.n_input}. Check that input matches cortex config."
        )
        if top_down is not None:
            assert top_down.shape[-1] == self.l23_size, (
                f"LayeredCortex.forward: top_down has shape {top_down.shape} "
                f"but L2/3 size={self.l23_size}. Top-down must match L2/3 for modulation."
            )
            assert top_down.shape[0] == batch_size, (
                f"LayeredCortex.forward: top_down batch size {top_down.shape[0]} "
                f"doesn't match input_spikes batch size {batch_size}."
            )

        if self.state.l4_spikes is None:
            assert_single_instance(batch_size, "LayeredCortex")
            self.reset_state()

        cfg = self.layer_config

        # L4: Input processing
        l4_input = (
            torch.matmul(input_spikes.float(), self.w_input_l4.t())
            * cfg.input_to_l4_strength
        )
        l4_input = l4_input * (0.5 + 0.5 * encoding_mod)

        # Apply Divisive Normalization to L4 input (automatic gain control)
        if self.divisive_norm_l4 is not None:
            l4_input = self.divisive_norm_l4(l4_input)

        l4_spikes, _ = self.l4_neurons(l4_input)
        l4_spikes = self._apply_sparsity(l4_spikes, cfg.l4_sparsity)
        self.state.l4_spikes = l4_spikes

        # Inter-layer shape check: L4 → L2/3
        assert l4_spikes.shape == (batch_size, self.l4_size), (
            f"LayeredCortex: L4 spikes have shape {l4_spikes.shape} "
            f"but expected ({batch_size}, {self.l4_size}). "
            f"Check L4 sparsity or input→L4 weights shape."
        )

        # L2/3: Processing with recurrence
        l23_ff = (
            torch.matmul(l4_spikes.float(), self.w_l4_l23.t())
            * cfg.l4_to_l23_strength
        )

        # Feedforward inhibition
        ffi_suppression = 1.0
        if self.feedforward_inhibition is not None:
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
        if self.state.l23_recurrent_activity is not None:
            recurrent_scale = 0.5 + 0.5 * retrieval_mod

            if self.stp_l23_recurrent is not None:
                # Apply STP to recurrent connections
                # NOTE: STP with batch_size > 1 is not supported in THALIA's architecture
                # THALIA models a single continuous brain state, not parallel simulations
                stp_efficacy = self.stp_l23_recurrent(
                    self.state.l23_recurrent_activity.float()
                )  # (batch, l23_size, l23_size)

                # For batch_size=1, squeeze and apply directly
                assert_single_instance(stp_efficacy.shape[0], "STP efficacy in LayeredCortex")

                stp_efficacy = stp_efficacy.squeeze(0)  # (l23_size, l23_size)
                effective_w_rec = self.w_l23_recurrent * stp_efficacy
                l23_rec = (
                    torch.matmul(
                        self.state.l23_recurrent_activity,
                        effective_w_rec.t(),
                    )
                    * cfg.l23_recurrent_strength
                    * recurrent_scale
                    * ffi_suppression
                )
            else:
                l23_rec = (
                    torch.matmul(
                        self.state.l23_recurrent_activity,
                        self.w_l23_recurrent.t(),
                    )
                    * cfg.l23_recurrent_strength
                    * recurrent_scale
                    * ffi_suppression
                )
        else:
            l23_rec = torch.zeros_like(l23_ff)

        # Top-down modulation
        l23_td = top_down * cfg.l23_top_down_strength if top_down is not None else 0.0

        l23_input = l23_ff + l23_rec + l23_td

        # Lateral inhibition
        if self.state.l23_spikes is not None:
            l23_inhib = torch.matmul(
                self.state.l23_spikes.float(),
                self.w_l23_inhib.t(),
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

        # Population Intrinsic Plasticity: Modulate input based on population rate
        if self.pop_intrinsic_plasticity is not None and self.state.l23_spikes is not None:
            self.pop_intrinsic_plasticity.update(self.state.l23_spikes)
            l23_input = self.pop_intrinsic_plasticity.modulate_input(l23_input)

        # INTRINSIC PLASTICITY: Apply per-neuron threshold offset
        # Neurons that fire too much have higher thresholds (less excitable)
        cfg = self.layer_config
        if (cfg.intrinsic_plasticity_enabled and
            self._l23_threshold_offset is not None):
            l23_input = l23_input - self._l23_threshold_offset.unsqueeze(0)

        l23_spikes, _ = self.l23_neurons(F.relu(l23_input))
        l23_spikes = self._apply_sparsity(l23_spikes, cfg.l23_sparsity)
        self.state.l23_spikes = l23_spikes

        # Inter-layer shape check: L2/3 → L5
        assert l23_spikes.shape == (batch_size, self.l23_size), (
            f"LayeredCortex: L2/3 spikes have shape {l23_spikes.shape} "
            f"but expected ({batch_size}, {self.l23_size}). "
            f"Check L2/3 sparsity or L4→L2/3 weights shape."
        )

        # Update recurrent activity trace
        if self.state.l23_recurrent_activity is not None:
            self.state.l23_recurrent_activity = (
                self.state.l23_recurrent_activity * cfg.l23_recurrent_decay
                + l23_spikes.float()
            )
        else:
            self.state.l23_recurrent_activity = l23_spikes.float()

        # L5: Subcortical output
        l5_input = (
            torch.matmul(l23_spikes.float(), self.w_l23_l5.t())
            * cfg.l23_to_l5_strength
        )
        l5_spikes, _ = self.l5_neurons(l5_input)
        l5_spikes = self._apply_sparsity(l5_spikes, cfg.l5_sparsity)
        self.state.l5_spikes = l5_spikes

        # Inter-layer shape check: L5 output
        assert l5_spikes.shape == (batch_size, self.l5_size), (
            f"LayeredCortex: L5 spikes have shape {l5_spikes.shape} "
            f"but expected ({batch_size}, {self.l5_size}). "
            f"Check L5 sparsity or L2/3→L5 weights shape."
        )

        # Update cumulative spike counters (for diagnostics)
        self._cumulative_l4_spikes += int(l4_spikes.sum().item())
        self._cumulative_l23_spikes += int(l23_spikes.sum().item())
        self._cumulative_l5_spikes += int(l5_spikes.sum().item())

        # Update STDP traces using utility function
        if self.state.l4_trace is not None:
            update_trace(self.state.l4_trace, l4_spikes, tau=cfg.stdp_tau_plus, dt=dt)
        if self.state.l23_trace is not None:
            update_trace(self.state.l23_trace, l23_spikes, tau=cfg.stdp_tau_plus, dt=dt)
        if self.state.l5_trace is not None:
            update_trace(self.state.l5_trace, l5_spikes, tau=cfg.stdp_tau_plus, dt=dt)

        self.state.spikes = l5_spikes

        # Store input for plasticity
        self.state.input_spikes = input_spikes

        # Apply continuous plasticity (learning happens as part of forward dynamics)
        self._apply_plasticity(dt=dt)

        # Construct output
        if cfg.dual_output:
            output = torch.cat([l23_spikes, l5_spikes], dim=-1)
        elif cfg.output_layer == "L5":
            output = l5_spikes
        else:
            output = l23_spikes

        return output

    def _apply_sparsity(
        self,
        spikes: torch.Tensor,
        target_sparsity: float,
    ) -> torch.Tensor:
        """Apply winner-take-all sparsity."""
        batch_size, n_neurons = spikes.shape
        k = max(1, int(n_neurons * target_sparsity))

        sparse_spikes = torch.zeros_like(spikes)

        for b in range(batch_size):
            active = spikes[b].nonzero(as_tuple=True)[0]
            if len(active) > k:
                keep_indices = active[torch.randperm(len(active))[:k]]
                sparse_spikes[b, keep_indices] = spikes[b, keep_indices]
            else:
                sparse_spikes[b] = spikes[b]

        return sparse_spikes

    def _apply_plasticity(self, dt: float = 1.0) -> None:
        """Apply continuous STDP learning with BCM modulation.

        This is called automatically at each forward() timestep.
        Learning rate is modulated by dopamine (via get_effective_learning_rate).

        In biological cortex, synaptic plasticity happens continuously based on
        pre/post spike timing. Dopamine doesn't trigger learning - it modulates
        how much weight change occurs from the spike-timing-based plasticity.
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

        # Decay neuromodulators (ACh/NE decay locally, dopamine set by Brain)
        self.decay_neuromodulators(dt_ms=dt)

        # Get 1D versions of spike tensors for torch.outer
        l4_spikes = ensure_1d(self.state.l4_spikes)
        l23_spikes = ensure_1d(self.state.l23_spikes) if self.state.l23_spikes is not None else None
        l5_spikes = ensure_1d(self.state.l5_spikes) if self.state.l5_spikes is not None else None
        input_spikes = ensure_1d(self.state.input_spikes)

        total_change = 0.0

        # BCM modulation factors
        l4_bcm_mod = 1.0
        l23_bcm_mod = 1.0
        l5_bcm_mod = 1.0

        if self.bcm_l4 is not None:
            l4_activity = l4_spikes.mean()
            l4_bcm_mod = self.bcm_l4.compute_phi(l4_activity)
            self.bcm_l4.update_threshold(l4_activity)

        if self.bcm_l23 is not None and l23_spikes is not None:
            l23_activity = l23_spikes.mean()
            l23_bcm_mod = self.bcm_l23.compute_phi(l23_activity)
            self.bcm_l23.update_threshold(l23_activity)

        if self.bcm_l5 is not None and l5_spikes is not None:
            l5_activity = l5_spikes.mean()
            l5_bcm_mod = self.bcm_l5.compute_phi(l5_activity)
            self.bcm_l5.update_threshold(l5_activity)

        def bcm_to_scale(mod: Any) -> float:
            if isinstance(mod, torch.Tensor):
                return float(1.0 + 0.5 * torch.tanh(mod).item())
            return 1.0 + 0.5 * torch.tanh(torch.tensor(mod)).item()

        # Input → L4
        dw = effective_lr * bcm_to_scale(l4_bcm_mod) * torch.outer(
            l4_spikes.float(),
            input_spikes.float(),
        )
        with torch.no_grad():
            self.w_input_l4.data += dw
            clamp_weights(self.w_input_l4.data, cfg.w_min, cfg.w_max)
        total_change += dw.abs().mean().item()

        # L4 → L2/3
        if l23_spikes is not None:
            dw = effective_lr * bcm_to_scale(l23_bcm_mod) * torch.outer(
                l23_spikes.float(),
                l4_spikes.float(),
            )
            with torch.no_grad():
                self.w_l4_l23.data += dw
                clamp_weights(self.w_l4_l23.data, cfg.w_min, cfg.w_max)
            total_change += dw.abs().mean().item()

            # L2/3 recurrent (signed weights - compact E/I approximation)
            # Uses dedicated bounds [l23_recurrent_w_min, l23_recurrent_w_max] to allow
            # both excitatory and inhibitory-like lateral connections.
            # This is a simplification of explicit E/I interneuron populations.
            l23_activity = l23_spikes.float()
            dw = effective_lr * 0.5 * bcm_to_scale(l23_bcm_mod) * torch.outer(
                l23_activity, l23_activity
            )

            # =========================================================
            # HETEROSYNAPTIC PLASTICITY: Weaken inactive synapses
            # =========================================================
            # Synapses to inactive postsynaptic neurons get weakened when
            # nearby neurons fire strongly. This prevents winner-take-all
            # dynamics from permanently dominating.
            if cfg.heterosynaptic_ratio > 0:
                inactive_post = (l23_activity < 0.5).float()  # Inactive neurons
                active_pre = l23_activity  # Active neurons
                hetero_ltd = cfg.heterosynaptic_ratio * effective_lr
                hetero_dW = -hetero_ltd * torch.outer(active_pre, inactive_post)
                dw = dw + hetero_dW

            with torch.no_grad():
                self.w_l23_recurrent.data += dw
                self.w_l23_recurrent.data.fill_diagonal_(0.0)
                clamp_weights(
                    self.w_l23_recurrent.data,
                    cfg.l23_recurrent_w_min,
                    cfg.l23_recurrent_w_max,
                )

                # =============================================================
                # SYNAPTIC SCALING (Homeostatic)
                # =============================================================
                # Multiplicatively adjust weights towards target mean.
                if cfg.synaptic_scaling_enabled:
                    mean_weight = self.w_l23_recurrent.data.abs().mean()
                    scaling = 1.0 + cfg.synaptic_scaling_rate * (cfg.synaptic_scaling_target - mean_weight)
                    self.w_l23_recurrent.data *= scaling
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
            # This operates on LONGER timescales than SFA.
            if cfg.intrinsic_plasticity_enabled:
                l23_spikes_1d = l23_activity

                # Initialize if needed
                if self._l23_activity_history is None:
                    self._l23_activity_history = torch.zeros(self.l23_size, device=l23_spikes_1d.device)
                if self._l23_threshold_offset is None:
                    self._l23_threshold_offset = torch.zeros(self.l23_size, device=l23_spikes_1d.device)

                # Update activity history (exponential moving average)
                self._l23_activity_history = (
                    0.99 * self._l23_activity_history + 0.01 * l23_spikes_1d
                )

                # Adjust threshold: high activity → higher threshold (less excitable)
                rate_error = self._l23_activity_history - cfg.intrinsic_target_rate
                self._l23_threshold_offset = (
                    self._l23_threshold_offset + cfg.intrinsic_adaptation_rate * rate_error
                ).clamp(-0.5, 0.5)  # Limit threshold adjustment range

            # L2/3 → L5
            if l5_spikes is not None:
                dw = effective_lr * bcm_to_scale(l5_bcm_mod) * torch.outer(
                    l5_spikes.float(),
                    l23_spikes.float(),
                )
                with torch.no_grad():
                    self.w_l23_l5.data += dw
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

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get layer-specific diagnostics using DiagnosticsMixin helpers.

        Note: Reports both instantaneous (l4_active_count) and cumulative
        (l4_cumulative_spikes) counts. During consolidation phases with
        zero input, instantaneous L4 will be 0 but cumulative shows
        total activity since last reset.
        """
        cfg = self.layer_config
        diag: Dict[str, Any] = {
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

        # Spike diagnostics for each layer (instantaneous)
        if self.state.l4_spikes is not None:
            diag.update(self.spike_diagnostics(self.state.l4_spikes, "l4"))
        if self.state.l23_spikes is not None:
            diag.update(self.spike_diagnostics(self.state.l23_spikes, "l23"))
        if self.state.l5_spikes is not None:
            diag.update(self.spike_diagnostics(self.state.l5_spikes, "l5"))

        # Recurrent activity
        if self.state.l23_recurrent_activity is not None:
            diag["l23_recurrent_mean"] = self.state.l23_recurrent_activity.mean().item()

        # Weight diagnostics for inter-layer connections
        diag.update(self.weight_diagnostics(self.w_input_l4.data, "input_l4"))
        diag.update(self.weight_diagnostics(self.w_l4_l23.data, "l4_l23"))
        diag.update(self.weight_diagnostics(self.w_l23_recurrent.data, "l23_rec"))
        diag.update(self.weight_diagnostics(self.w_l23_l5.data, "l23_l5"))

        # Robustness mechanism diagnostics
        if self.ei_balance is not None:
            ei_diag = self.ei_balance.get_diagnostics()
            diag["robustness_ei_ratio"] = ei_diag.get("current_ratio", 0.0)
            diag["robustness_ei_scale"] = ei_diag.get("inh_scale", 1.0)
            diag["robustness_ei_status"] = ei_diag.get("status", "unknown")

        if self.divisive_norm_l4 is not None:
            diag["robustness_divisive_norm_enabled"] = True

        if self.pop_intrinsic_plasticity is not None:
            ip_diag = self.pop_intrinsic_plasticity.get_diagnostics()
            diag["robustness_ip_rate_avg"] = ip_diag.get("rate_avg", 0.0)
            diag["robustness_ip_excitability"] = ip_diag.get("excitability", 1.0)

        return diag
