"""
Thalamus - Sensory Relay, Gating, and Attentional Modulation.

The thalamus is the brain's "sensory switchboard" and attention controller:
- **Relays sensory information** to appropriate cortical areas
- **Gates sensory input** based on attention and arousal state
- **Switches between burst and tonic modes** for different processing needs
- **Modulates cortical excitability** via synchronized oscillations

**Key Features**:
=================
1. **SENSORY RELAY**:
   - All sensory modalities (except olfaction) pass through thalamus first
   - Selective routing to appropriate cortical areas (LGN→V1, MGN→A1, etc.)
   - Spatial filtering and preprocessing before cortical arrival
   - Maintains topographic organization (retinotopy, tonotopy)

2. **ATTENTIONAL GATING**:
   - Alpha oscillations (8-12 Hz) suppress IRRELEVANT inputs
   - Enhanced transmission for ATTENDED stimuli (reduced inhibition)
   - Norepinephrine modulates gain (arousal-dependent filtering)
   - Implements "spotlight" attention via TRN inhibition

3. **MODE SWITCHING**:
   - **Burst mode**: Low input, creates sharp transients → alerting, attention capture
   - **Tonic mode**: Steady input, faithful relay → normal processing
   - Mode controlled by membrane potential and oscillation phase
   - T-type Ca²⁺ channels enable burst firing when hyperpolarized

4. **THALAMIC RETICULAR NUCLEUS (TRN)**:
   - Inhibitory shell surrounding thalamus (GABAergic)
   - Implements "searchlight" attention mechanism
   - Coordinates coherent oscillations across thalamic nuclei
   - Winner-take-all competition between sensory streams

Biological Basis:
=================
- Lateral geniculate nucleus (LGN): Visual relay
- Medial geniculate nucleus (MGN): Auditory relay
- Ventral posterior nucleus (VPN): Somatosensory relay
- Pulvinar: Visual attention and salience
- Mediodorsal nucleus (MD): Prefrontal coordination

Architecture Pattern:
====================

    Sensory Input (spikes)
           │
           ▼
    ┌──────────────┐
    │   THALAMUS   │  Mode: burst vs tonic
    │              │  Gating: alpha suppression
    │  ┌────────┐  │  Gain: NE modulation
    │  │  TRN   │  │  (inhibitory shell)
    │  └────────┘  │
    └──────┬───────┘
           │ Gated spikes
           ▼
    ┌──────────────┐
    │    CORTEX    │
    │  (L4 input)  │
    └──────────────┘
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from thalia.components.coding import compute_firing_rate, compute_spike_count
from thalia.components.gap_junctions import GapJunctionConfig, GapJunctionCoupling
from thalia.components.neurons import create_relay_neurons, create_trn_neurons
from thalia.components.synapses import ShortTermPlasticity, STPConfig, WeightInitializer
from thalia.config.region_configs import ThalamicRelayConfig
from thalia.constants.regions import (
    THALAMUS_MODE_THRESHOLD,
    THALAMUS_NE_GAIN_SCALE,
    THALAMUS_RELAY_SCALE,
    THALAMUS_RELAY_SPARSITY,
    THALAMUS_SPATIAL_CENTER_SPARSITY,
    THALAMUS_SURROUND_WIDTH_RATIO,
    THALAMUS_TRN_FEEDBACK_SCALE,
    THALAMUS_TRN_FEEDBACK_SPARSITY,
    THALAMUS_TRN_FEEDFORWARD_SPARSITY,
)
from thalia.core.neural_region import NeuralRegion
from thalia.managers.component_registry import register_region
from thalia.utils.input_routing import InputRouter

from .state import ThalamicRelayState


@register_region(
    "thalamus",
    aliases=["thalamic_relay"],
    description="Sensory relay and gating with burst/tonic modes and attentional modulation",
    version="1.0",
    author="Thalia Project",
    config_class=ThalamicRelayConfig,
)
class ThalamicRelay(NeuralRegion):
    """Thalamic relay nucleus with burst/tonic modes and attentional gating.

    Provides:
    - Sensory relay with spatial filtering (center-surround)
    - Attentional gating via alpha oscillations
    - Burst vs tonic mode switching
    - TRN-mediated inhibitory coordination
    - Gain modulation via neuromodulators

    Architecture:

        Input → Relay Neurons ⇄ TRN → Output to Cortex
                     ↑              (inhibitory)
                     └── Recurrent inhibition

    From NeuralRegion (abstract base):
    ----------------------------------------
    - forward(input, **kwargs) → Tensor
    - reset_state() → None
    - get_diagnostics() → Dict
    - set_oscillator_phases(phases, signals) → None
    - Neuromodulator control methods
    """

    def __init__(
        self,
        config: ThalamicRelayConfig,
        sizes: Dict[str, int],
        device: str = "cpu",
    ):
        """Initialize thalamic relay.

        Args:
            config: Thalamic relay configuration (behavioral parameters only)
            sizes: Size dictionary from LayerSizeCalculator with keys:
                  - relay_size: Number of relay neurons
                  - trn_size: Number of TRN neurons
                  - input_size: Sensory input dimension (optional, for inference)
            device: Device to place tensors on ("cpu" or "cuda")
        """
        # Extract sizes from sizes dict
        self.relay_size = sizes["relay_size"]
        self.trn_size = sizes["trn_size"]
        self.input_size = sizes.get("input_size", self.relay_size)  # Default to relay_size

        # Computed properties
        self.n_output = self.relay_size
        self.total_neurons = self.relay_size + self.trn_size

        # Store device before calling super().__init__
        self.device = torch.device(device)

        # Initialize NeuralRegion with relay neurons as the primary population
        super().__init__(
            n_neurons=self.relay_size,
            device=device,
        )

        # Store config
        self.config = config

        # Standard interface attributes (used throughout thalamus implementation)
        self.n_relay = self.relay_size
        self.n_trn = self.trn_size

        # =====================================================================
        # RELAY NEURONS (Excitatory, glutamatergic)
        # =====================================================================
        self.relay_neurons = create_relay_neurons(self.relay_size, device)

        # =====================================================================
        # TRN NEURONS (Inhibitory, GABAergic)
        # =====================================================================
        self.trn_neurons = create_trn_neurons(self.trn_size, device)

        # =====================================================================
        # WEIGHTS
        # =====================================================================

        # Relay gain per neuron (learned modulation of filtered input)
        self.relay_gain = nn.Parameter(
            torch.ones(self.relay_size, device=self.device, requires_grad=False)
            * config.relay_strength
        )

        # Input → TRN (collateral activation)
        self.input_to_trn = nn.Parameter(
            WeightInitializer.sparse_random(
                n_output=self.trn_size,
                n_input=self.input_size,
                sparsity=THALAMUS_RELAY_SPARSITY,
                weight_scale=THALAMUS_RELAY_SCALE,
                device=self.device,
            ),
            requires_grad=False,
        )

        # Relay → TRN (collateral activation)
        self.relay_to_trn = nn.Parameter(
            WeightInitializer.sparse_random(
                n_output=self.trn_size,
                n_input=self.relay_size,
                sparsity=THALAMUS_TRN_FEEDBACK_SPARSITY,
                weight_scale=THALAMUS_TRN_FEEDBACK_SCALE,
                device=self.device,
            ),
            requires_grad=False,
        )

        # TRN → Relay (inhibitory feedback)
        self.trn_to_relay = nn.Parameter(
            WeightInitializer.sparse_random(
                n_output=self.relay_size,
                n_input=self.trn_size,
                sparsity=THALAMUS_TRN_FEEDFORWARD_SPARSITY,
                weight_scale=config.trn_inhibition_strength,
                device=self.device,
            ),
            requires_grad=False,
        )

        # TRN → TRN (recurrent inhibition for oscillations)
        self.trn_recurrent = nn.Parameter(
            WeightInitializer.sparse_random(
                n_output=self.trn_size,
                n_input=self.trn_size,
                sparsity=THALAMUS_SPATIAL_CENTER_SPARSITY,
                weight_scale=config.trn_recurrent_strength,
                device=self.device,
            ),
            requires_grad=False,
        )

        # =====================================================================
        # GAP JUNCTIONS (TRN interneuron synchronization)
        # =====================================================================
        # TRN neurons are densely coupled via gap junctions (Landisman et al. 2002)
        # This enables ultra-fast synchronization for coherent inhibitory volleys
        self.gap_junctions: Optional[GapJunctionCoupling] = None
        if config.gap_junctions_enabled:
            gap_config = GapJunctionConfig(
                enabled=True,
                coupling_strength=config.gap_junction_strength,
                connectivity_threshold=0.2,  # Liberal coupling (TRN is densely connected)
                max_neighbors=10,  # TRN has ~6-12 gap junction partners (Galarreta 1999)
                interneuron_only=True,  # All TRN neurons are inhibitory
            )

            # Use input_to_trn weights to define functional neighborhoods
            # TRN neurons sharing sensory inputs are anatomically close
            self.gap_junctions = GapJunctionCoupling(
                n_neurons=self.trn_size,
                afferent_weights=self.input_to_trn,  # Shared sensory inputs
                config=gap_config,
                interneuron_mask=None,  # All TRN neurons are interneurons
                device=self.device,
            )

        # =====================================================================
        # CENTER-SURROUND RECEPTIVE FIELDS
        # =====================================================================
        self._build_center_surround_filter()

        # =====================================================================
        # INTERNAL DELAY BUFFERS
        # =====================================================================
        # TRN → relay inhibition delay (GABAergic synaptic transmission ~3-5ms)
        self._trn_relay_delay_steps = int(config.trn_to_relay_delay_ms / config.dt_ms)
        self._trn_relay_delay_buffer: Optional[torch.Tensor] = None
        self._trn_relay_delay_ptr: int = 0

        # =====================================================================
        # SHORT-TERM PLASTICITY (STP) - HIGH PRIORITY for sensory gating
        # =====================================================================
        self.stp_sensory_relay: Optional[ShortTermPlasticity] = None
        self.stp_l6_feedback: Optional[ShortTermPlasticity] = None

        if config.stp_enabled:
            # Sensory input → relay depression (U=0.4, moderate depression)
            # Filters repetitive stimuli, responds to novelty
            # CRITICAL for attention capture and change detection
            self.stp_sensory_relay = ShortTermPlasticity(
                n_pre=self.input_size,
                n_post=self.relay_size,
                config=STPConfig.from_type(config.stp_sensory_relay_type),
                per_synapse=True,  # Per-synapse dynamics for maximum precision
            )

            # L6 cortical feedback → relay depression (U=0.7, strong depression)
            # Dynamic gain control: Sustained cortical feedback reduces thalamic transmission
            # Enables efficient filtering and sensory gating
            # NOTE: L6 size must match relay_size (validated at build time)
            self.stp_l6_feedback = ShortTermPlasticity(
                n_pre=self.relay_size,  # L6b must match relay size
                n_post=self.relay_size,
                config=STPConfig.from_type(config.stp_l6_feedback_type),
                per_synapse=True,
            )

        # =====================================================================
        # STATE
        # =====================================================================
        self.state: ThalamicRelayState = ThalamicRelayState()  # type: ignore[assignment]

        # Oscillator state (provided by brain)
        self._alpha_phase: float = 0.0
        self._alpha_amplitude: float = 1.0

        # =====================================================================
        # PHASE 2: REGISTER COMPONENTS FOR AUTO-GROWTH (opt-in)
        # =====================================================================
        # Phase 2 Registration: Opt-in auto-growth for STP modules
        # Register STP modules with their growth patterns:
        if self.stp_sensory_relay is not None:
            # Non-recurrent (n_input -> relay_size): grows in both contexts
            self._register_stp("stp_sensory_relay", direction="both")
        if self.stp_l6_feedback is not None:
            # Recurrent (relay_size -> relay_size): only grows during grow_output
            self._register_stp("stp_l6_feedback", direction="post", recurrent=True)

        # Note: TRN growth is manual (maintains current ratio, not fixed ratio)

        # =====================================================================
        # PORT-BASED ROUTING
        # =====================================================================
        # Register output ports for relay and TRN neurons
        # relay: Thalamic relay cells (sensory gating, cortical projection)
        # trn: Thalamic reticular nucleus (lateral inhibition, attention gating)
        self.register_output_port("relay", self.relay_size)
        self.register_output_port("trn", self.trn_size)

        # =====================================================================
        # MOVE TO DEVICE (must be last)
        # =====================================================================
        self.to(device)

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters when brain timestep changes.

        Propagates dt update to neurons and STP components.

        Args:
            dt_ms: New simulation timestep in milliseconds
        """
        # Update neurons
        if hasattr(self, "relay_neurons") and hasattr(
            self.relay_neurons, "update_temporal_parameters"
        ):
            self.relay_neurons.update_temporal_parameters(dt_ms)
        if hasattr(self, "trn_neurons") and hasattr(self.trn_neurons, "update_temporal_parameters"):
            self.trn_neurons.update_temporal_parameters(dt_ms)

        # Update STP components
        if hasattr(self, "stp_sensory_relay") and self.stp_sensory_relay is not None:
            self.stp_sensory_relay.update_temporal_parameters(dt_ms)
        if hasattr(self, "stp_l6_feedback") and self.stp_l6_feedback is not None:
            self.stp_l6_feedback.update_temporal_parameters(dt_ms)

    def _initialize_weights(self) -> Optional[torch.Tensor]:
        """Weights initialized in __init__, return None."""
        return None

    def _create_neurons(self) -> Optional[Any]:
        """Neurons created in __init__, return None."""
        return None

    def _build_center_surround_filter(self) -> None:
        """Build center-surround spatial filter for receptive fields.

        Creates Difference of Gaussians (DoG) filter:
        - Center: Narrow excitation
        - Surround: Wide inhibition

        This enhances edges and reduces redundancy (efficient coding).
        """
        # Simple 1D spatial filter (can be extended to 2D for vision)
        # For now, apply local connectivity pattern

        # Create distance matrix (neuron i to input j)
        relay_idx = torch.arange(self.n_relay, device=self.device).float()
        input_idx = torch.arange(self.input_size, device=self.device).float()

        # Scale to [0, 1]
        relay_norm = relay_idx / max(1, self.n_relay - 1)
        input_norm = input_idx / max(1, self.input_size - 1)

        # Distance matrix [n_relay, n_input]
        distances = torch.abs(relay_norm.unsqueeze(1) - input_norm.unsqueeze(0))

        # Center (narrow Gaussian)
        width_center = self.config.spatial_filter_width
        center = self.config.center_excitation * torch.exp(-(distances**2) / (2 * width_center**2))

        # Surround (wider Gaussian)
        width_surround = width_center * THALAMUS_SURROUND_WIDTH_RATIO
        surround = self.config.surround_inhibition * torch.exp(
            -(distances**2) / (2 * width_surround**2)
        )

        # DoG filter
        self.register_buffer("center_surround_filter", center - surround)

    def set_oscillator_phases(
        self,
        phases: Dict[str, float],
        signals: Optional[Dict[str, float]] = None,
        theta_slot: int = 0,
        coupled_amplitudes: Optional[Dict[str, float]] = None,
    ) -> None:
        """Set oscillator phases for this region.

        Thalamus uses alpha oscillations for attentional gating.

        Args:
            phases: Dict mapping oscillator name to phase [0, 2π)
            signals: Dict mapping oscillator name to signal value [-1, 1]
            theta_slot: Current theta slot (not used by thalamus)
            coupled_amplitudes: Pre-computed effective amplitudes (not used by thalamus)
        """
        # Use base mixin implementation to store all oscillator data
        super().set_oscillator_phases(phases, signals, theta_slot, coupled_amplitudes)

        # Store alpha amplitude from signals for gating
        if signals and "alpha" in signals:
            self._alpha_amplitude = signals["alpha"]

    def _compute_alpha_gate(self) -> torch.Tensor:
        """Compute attentional gating from alpha oscillation.

        Alpha oscillations (8-13 Hz) suppress unattended inputs:
        - Alpha peak (phase=π): Strong suppression
        - Alpha trough (phase=0): Weak suppression

        Returns:
            Gating factor [n_relay] (1D, ADR-005)
        """
        # Alpha gate: strong at trough (phase=0), weak at peak (phase=π)
        # gate = 1 - strength × (1 + cos(phase)) / 2
        # This gives: phase=0 → gate=1-strength, phase=π → gate=1.0

        # Normalize (1 + cos(phase)) from [0, 2] to [0, 1]
        alpha_modulation = 0.5 * (1.0 + math.cos(self._alpha_phase))
        gate = 1.0 - self.config.alpha_suppression_strength * alpha_modulation

        # Broadcast to all neurons (ADR-005: 1D)
        gate_tensor = torch.full(
            (self.n_relay,),
            gate,
            device=self.device,
            dtype=torch.float32,
        )

        return gate_tensor

    def _determine_mode(self, membrane: torch.Tensor) -> torch.Tensor:
        """Determine burst vs tonic mode based on membrane potential.

        Args:
            membrane: Current membrane potential [n_relay] (1D, ADR-005)

        Returns:
            Mode indicator [n_relay]: 0=burst, 1=tonic (1D, ADR-005)
        """
        # Burst mode: Hyperpolarized (membrane < burst_threshold)
        # Tonic mode: Depolarized (membrane > tonic_threshold)
        # Between: Maintain previous mode

        if self.state.current_mode is None:
            # Initialize to tonic
            self.state.current_mode = torch.ones_like(membrane)

        # Update mode based on thresholds
        burst_mask = membrane < self.config.burst_threshold
        tonic_mask = membrane > self.config.tonic_threshold

        self.state.current_mode = torch.where(
            burst_mask,
            torch.zeros_like(membrane),  # Burst mode
            torch.where(
                tonic_mask,
                torch.ones_like(membrane),  # Tonic mode
                self.state.current_mode,  # Maintain
            ),
        )

        return self.state.current_mode

    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> torch.Tensor:
        """Process sensory input through thalamic relay.

        Args:
            inputs: Multi-port input dict:
                   - {"sensory": tensor, "l6a_feedback": tensor, "l6b_feedback": tensor} (multi-port)
                   - {"input": tensor} (alias for "sensory")
            **kwargs: Additional arguments (unused)

        Returns:
            Relay output spikes [n_relay] (bool, ADR-004/005)

        Note:
            Multi-port architecture (biologically accurate):
            - "sensory" or "input" → relay neurons (thalamocortical projection)
            - "l6a_feedback" → TRN neurons (corticothalamic type I, inhibitory modulation)
            - "l6b_feedback" → relay neurons (corticothalamic type II, excitatory modulation)
            These are different post-synaptic targets, NOT concatenated!
        """
        # Route inputs to canonical port names
        routed = InputRouter.route(
            inputs,
            port_mapping={
                "sensory": ["sensory", "input", "default"],
                "l6a_feedback": ["l6a_feedback", "feedback"],
                "l6b_feedback": ["l6b_feedback"],
            },
            defaults={"sensory": None, "l6a_feedback": None, "l6b_feedback": None},
            required=[],  # Sensory input optional during consolidation (sleep)
            component_name="ThalamicRelay",
        )
        input_spikes = routed["sensory"]
        cortical_l6a_feedback = routed["l6a_feedback"]
        cortical_l6b_feedback = routed["l6b_feedback"]

        # If no sensory input, relay neurons are silent (e.g., during consolidation/sleep)
        # Biologically accurate: thalamus doesn't fire without sensory stimulation
        if input_spikes is None:
            return torch.zeros(self.n_output, dtype=torch.bool, device=self.device)

        # ADR-005: Expect 1D input
        assert (
            input_spikes.dim() == 1
        ), f"Thalamus.forward: Expected 1D input (ADR-005), got shape {input_spikes.shape}"
        assert (
            input_spikes.shape[0] == self.input_size
        ), f"Thalamus.forward: input has {input_spikes.shape[0]} neurons, expected {self.input_size}"

        # =====================================================================
        # 1. APPLY CENTER-SURROUND SPATIAL FILTER
        # ====================================================================="
        # Filter input spikes spatially before relay (ADR-005: 1D)
        # Convert bool to float for matmul (ADR-004)
        input_float = input_spikes.float()
        filtered_input = torch.mv(self.center_surround_filter, input_float).clamp(
            min=0
        )  # [n_relay]

        # Apply STP to sensory input → relay if enabled
        # CRITICAL for novelty detection and attention capture
        if self.stp_sensory_relay is not None:
            # Sensory depression: Sustained inputs fade, novel stimuli strong
            sensory_efficacy = self.stp_sensory_relay(input_spikes.float())  # [n_input, n_relay]
            # Transpose to match filter output shape [n_relay]
            # Apply efficacy as multiplicative modulation of filtered input
            filtered_input = filtered_input * sensory_efficacy.mean(
                dim=0
            )  # Average over input neurons

        # =====================================================================
        # 2. COMPUTE ALPHA ATTENTIONAL GATE
        # =====================================================================
        alpha_gate = self._compute_alpha_gate()  # [n_relay]
        self.state.alpha_gate = alpha_gate

        # =====================================================================
        # 3. RELAY NEURONS: Filtered input + L6b feedback → Relay
        # =====================================================================
        # Apply alpha gating to filtered input
        gated_input = filtered_input * alpha_gate

        # Apply learned per-neuron gain modulation
        relay_excitation = gated_input * self.relay_gain  # [n_relay]

        # Apply norepinephrine gain modulation (arousal)
        ne_gain = 1.0 + THALAMUS_NE_GAIN_SCALE * self.state.norepinephrine
        relay_excitation = relay_excitation * ne_gain

        # L6b corticothalamic feedback: Direct excitatory modulation of relay neurons
        # Type II corticothalamic pathway (Sherman & Guillery 2002)
        # Biological function: Cortex directly amplifies specific relay neurons for precision processing
        # BIOLOGICAL CONSTRAINT: L6b size must match n_relay (enforced at build time)
        if cortical_l6b_feedback is not None:
            l6b_excitation = cortical_l6b_feedback.float() * self.config.l6b_to_relay_strength

            # Validate size (should be guaranteed by builder)
            if l6b_excitation.shape[0] != self.n_relay:
                raise ValueError(
                    f"L6b feedback size mismatch: expected {self.n_relay}, got {l6b_excitation.shape[0]}. "
                    f"This indicates a brain building error - L6b must equal thalamus relay size."
                )

            # Apply STP to L6 feedback → relay if enabled
            # CRITICAL for dynamic gain control and sensory gating
            if self.stp_l6_feedback is not None:
                # L6 feedback depression: Sustained cortical feedback reduces thalamic transmission
                # Implements efficient filtering: Cortex can suppress irrelevant thalamic activity
                l6_efficacy = self.stp_l6_feedback(
                    cortical_l6b_feedback.float()
                )  # [n_relay, n_relay]
                # Apply per-postsynaptic efficacy (diagonal since L6b size == relay size)
                l6b_excitation = l6b_excitation * l6_efficacy.diag()

            # Add L6b excitation to relay (applied BEFORE TRN inhibition)
            relay_excitation = relay_excitation + l6b_excitation

        # TRN inhibition of relay (applied AFTER L6b excitation)
        # Apply axonal delay for TRN → relay (GABAergic transmission)
        if self.state.trn_spikes is not None:
            # Use delay buffer if configured
            if self._trn_relay_delay_steps > 0:
                # Initialize buffer if needed
                if self._trn_relay_delay_buffer is None:
                    self._trn_relay_delay_buffer = torch.zeros(
                        self._trn_relay_delay_steps,
                        self.n_trn,
                        dtype=torch.bool,
                        device=self.device,
                    )
                    self._trn_relay_delay_ptr = 0

                # Read delayed TRN spikes
                read_idx = (
                    self._trn_relay_delay_ptr - self._trn_relay_delay_steps
                ) % self._trn_relay_delay_buffer.shape[0]
                trn_spikes_delayed = self._trn_relay_delay_buffer[read_idx]

                # Write current TRN spikes to buffer
                self._trn_relay_delay_buffer[self._trn_relay_delay_ptr] = self.state.trn_spikes

                # Advance pointer (circular buffer)
                self._trn_relay_delay_ptr = (
                    self._trn_relay_delay_ptr + 1
                ) % self._trn_relay_delay_buffer.shape[0]

                # Compute inhibition from delayed spikes
                relay_inhibition = torch.mv(
                    self.trn_to_relay, trn_spikes_delayed.float()
                )  # [n_relay]
            else:
                # No delay (instantaneous inhibition)
                relay_inhibition = torch.mv(
                    self.trn_to_relay, self.state.trn_spikes.float()
                )  # [n_relay]
        else:
            relay_inhibition = torch.zeros(self.n_relay, device=self.device)

        # Update relay neurons (ADR-005: 1D tensors, no batch)
        relay_spikes, relay_membrane = self.relay_neurons(
            g_exc_input=relay_excitation,  # [n_relay]
            g_inh_input=relay_inhibition,  # [n_relay]
        )

        # =====================================================================
        # 4. MODE SWITCHING: Burst vs Tonic
        # =====================================================================
        current_mode = self._determine_mode(relay_membrane)  # [n_relay]

        # In burst mode, amplify spikes (convert bool to float temporarily)
        burst_mask = current_mode < THALAMUS_MODE_THRESHOLD  # Burst mode, [n_relay]
        burst_amplified = relay_spikes.float()  # [n_relay]

        if burst_mask.any():
            # Amplify burst spikes
            burst_amplified = torch.where(
                burst_mask, burst_amplified * self.config.burst_gain, burst_amplified
            )

        # Binarize and convert to bool (ADR-004)
        relay_output = burst_amplified > THALAMUS_MODE_THRESHOLD  # [n_relay], bool

        # =====================================================================
        # 5. TRN NEURONS: Input collaterals + Relay collaterals + L6a Feedback
        # =====================================================================
        # TRN receives:
        # - Input collaterals (sensory copy)
        # - Relay collaterals (relay activity)
        # - L6a corticothalamic feedback (attentional modulation, type I)
        # - Recurrent inhibition (TRN-TRN)

        # Convert bool to float for matmul (ADR-004)
        trn_excitation_input = torch.mv(self.input_to_trn, input_float)  # [n_trn]
        trn_excitation_relay = torch.mv(self.relay_to_trn, relay_output.float())  # [n_trn]

        # L6a corticothalamic feedback: Cortex modulates TRN to control attention
        # Type I corticothalamic pathway (Sherman & Guillery 2002)
        # This implements the feedback loop: Sensory → Thalamus → Cortex → L6a → TRN → Thalamus
        # Biological function: Cortex can amplify or suppress specific sensory channels
        # BIOLOGICAL CONSTRAINT: L6a size should match n_trn (enforced at build time)
        trn_excitation_l6a = torch.zeros(self.n_trn, device=self.device)
        if cortical_l6a_feedback is not None:
            # Scale L6a feedback appropriately (moderate strength)
            trn_excitation_l6a = cortical_l6a_feedback.float() * self.config.l6a_to_trn_strength

            # Validate size (should be guaranteed by builder)
            if trn_excitation_l6a.shape[0] != self.n_trn:
                raise ValueError(
                    f"L6a feedback size mismatch: expected {self.n_trn}, got {trn_excitation_l6a.shape[0]}. "
                    f"This indicates a brain building error - L6a must equal TRN size."
                )

        trn_excitation = trn_excitation_input + trn_excitation_relay + trn_excitation_l6a  # [n_trn]

        # TRN recurrent inhibition
        if self.state.trn_spikes is not None:
            trn_inhibition = torch.mv(self.trn_recurrent, self.state.trn_spikes.float())  # [n_trn]
        else:
            trn_inhibition = torch.zeros(self.n_trn, device=self.device)

        # Gap junction coupling (TRN synchronization)
        # Ultra-fast electrical coupling (<0.1ms) for coherent inhibitory volleys
        trn_gap_current = torch.zeros(self.n_trn, device=self.device)
        if self.gap_junctions is not None and self.state.trn_membrane is not None:
            # Apply voltage coupling based on previous timestep's membrane potentials
            trn_gap_current = self.gap_junctions(self.state.trn_membrane)
            # Gap junction current is excitatory (depolarizing current from neighbors)
            # Add to excitation rather than inhibition
            trn_excitation = trn_excitation + trn_gap_current

        # Update TRN neurons (ADR-005: 1D tensors)
        trn_spikes, trn_membrane = self.trn_neurons(
            g_exc_input=trn_excitation,  # [n_trn] (includes gap junction coupling)
            g_inh_input=trn_inhibition,  # [n_trn]
        )

        # =====================================================================
        # 6. UPDATE STATE
        # =====================================================================
        self.state.relay_spikes = relay_output  # [n_relay], bool
        self.state.relay_membrane = relay_membrane  # [n_relay]
        self.state.trn_spikes = trn_spikes  # [n_trn], bool
        self.state.trn_membrane = trn_membrane  # [n_trn]
        self.state.current_mode = current_mode  # [n_relay]

        # Store for component state (BrainComponent protocol)
        self.state.membrane = self.state.relay_membrane
        self.state.spikes = self.state.relay_spikes

        # =====================================================================
        # 7. PORT-BASED ROUTING OUTPUTS
        # =====================================================================
        # Set port outputs for biologically-specific routing
        self.clear_port_outputs()
        self.set_port_output("relay", relay_output)  # Thalamic relay → Cortex
        self.set_port_output("trn", trn_spikes)     # TRN → lateral inhibition

        return relay_output

    def get_state(self) -> ThalamicRelayState:
        """Capture complete thalamic relay state for checkpointing.

        Returns:
            ThalamicRelayState with all state fields including STP states
        """
        # Capture STP state if modules are enabled
        stp_sensory_state = None
        if self.stp_sensory_relay is not None:
            stp_sensory_state = {
                "u": (
                    self.stp_sensory_relay.u.clone()
                    if self.stp_sensory_relay.u is not None
                    else None
                ),
                "x": (
                    self.stp_sensory_relay.x.clone()
                    if self.stp_sensory_relay.x is not None
                    else None
                ),
            }

        stp_l6_state = None
        if self.stp_l6_feedback is not None:
            stp_l6_state = {
                "u": self.stp_l6_feedback.u.clone() if self.stp_l6_feedback.u is not None else None,
                "x": self.stp_l6_feedback.x.clone() if self.stp_l6_feedback.x is not None else None,
            }

        # Return new state with all fields
        return ThalamicRelayState(
            # Base region state
            spikes=self.state.spikes.clone() if self.state.spikes is not None else None,
            membrane=self.state.membrane.clone() if self.state.membrane is not None else None,
            dopamine=self.state.dopamine,
            acetylcholine=self.state.acetylcholine,
            norepinephrine=self.state.norepinephrine,
            # Relay neuron state
            relay_spikes=(
                self.state.relay_spikes.clone() if self.state.relay_spikes is not None else None
            ),
            relay_membrane=(
                self.state.relay_membrane.clone() if self.state.relay_membrane is not None else None
            ),
            # TRN state
            trn_spikes=self.state.trn_spikes.clone() if self.state.trn_spikes is not None else None,
            trn_membrane=(
                self.state.trn_membrane.clone() if self.state.trn_membrane is not None else None
            ),
            # Mode state
            current_mode=(
                self.state.current_mode.clone() if self.state.current_mode is not None else None
            ),
            burst_counter=(
                self.state.burst_counter.clone() if self.state.burst_counter is not None else None
            ),
            # Gating state
            alpha_gate=self.state.alpha_gate.clone() if self.state.alpha_gate is not None else None,
            # STP state
            stp_sensory_relay_state=stp_sensory_state,
            stp_l6_feedback_state=stp_l6_state,
        )

    def load_state(self, state: ThalamicRelayState) -> None:
        """Restore thalamic relay state from checkpoint.

        Args:
            state: ThalamicRelayState to restore
        """
        # Restore base region state
        self.state.spikes = state.spikes.clone() if state.spikes is not None else None
        self.state.membrane = state.membrane.clone() if state.membrane is not None else None
        self.state.dopamine = state.dopamine
        self.state.acetylcholine = state.acetylcholine
        self.state.norepinephrine = state.norepinephrine

        # Restore relay neuron state
        self.state.relay_spikes = (
            state.relay_spikes.clone() if state.relay_spikes is not None else None
        )
        self.state.relay_membrane = (
            state.relay_membrane.clone() if state.relay_membrane is not None else None
        )

        # Restore TRN state
        self.state.trn_spikes = state.trn_spikes.clone() if state.trn_spikes is not None else None
        self.state.trn_membrane = (
            state.trn_membrane.clone() if state.trn_membrane is not None else None
        )

        # Restore mode state
        self.state.current_mode = (
            state.current_mode.clone() if state.current_mode is not None else None
        )
        self.state.burst_counter = (
            state.burst_counter.clone() if state.burst_counter is not None else None
        )

        # Restore gating state
        self.state.alpha_gate = state.alpha_gate.clone() if state.alpha_gate is not None else None

        # Restore STP state if modules are enabled
        if self.stp_sensory_relay is not None and state.stp_sensory_relay_state is not None:
            if state.stp_sensory_relay_state.get("u") is not None:
                self.stp_sensory_relay.u = state.stp_sensory_relay_state["u"].clone()
            if state.stp_sensory_relay_state.get("x") is not None:
                self.stp_sensory_relay.x = state.stp_sensory_relay_state["x"].clone()

        if self.stp_l6_feedback is not None and state.stp_l6_feedback_state is not None:
            if state.stp_l6_feedback_state.get("u") is not None:
                self.stp_l6_feedback.u = state.stp_l6_feedback_state["u"].clone()
            if state.stp_l6_feedback_state.get("x") is not None:
                self.stp_l6_feedback.x = state.stp_l6_feedback_state["x"].clone()

    def reset_state(self) -> None:
        """Reset thalamic state."""
        super().reset_state()

        self.relay_neurons.reset_state()
        self.trn_neurons.reset_state()

        # Reset STP modules if enabled
        if self.stp_sensory_relay is not None:
            self.stp_sensory_relay.reset_state()
        if self.stp_l6_feedback is not None:
            self.stp_l6_feedback.reset_state()

        self.state.relay_spikes = None
        self.state.relay_membrane = None
        self.state.trn_spikes = None
        self.state.trn_membrane = None
        self.state.current_mode = None
        self.state.burst_counter = None
        self.state.alpha_gate = None
        self.state.stp_sensory_relay_state = None
        self.state.stp_l6_feedback_state = None

        self._alpha_phase = 0.0
        self._alpha_amplitude = 1.0

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics in standardized DiagnosticsDict format.

        Returns consolidated diagnostic information about:
        - Activity: Relay neuron spike statistics (primary output)
        - Plasticity: Weight statistics for all thalamic connections
        - Health: Membrane potentials, alpha gating, burst/tonic balance
        - Neuromodulators: Dopamine, norepinephrine, acetylcholine
        - Region-specific: TRN activity, mode distribution, oscillations

        This is the primary diagnostic interface for the Thalamus.
        """
        from thalia.core.diagnostics_schema import (
            compute_activity_metrics,
            compute_health_metrics,
            compute_plasticity_metrics,
        )

        # Compute activity metrics from relay neurons (primary output)
        activity = compute_activity_metrics(
            output_spikes=(
                self.state.relay_spikes
                if self.state.relay_spikes is not None
                else torch.zeros(self.n_relay, device=self.device)
            ),
            total_neurons=self.n_relay,
        )

        # Compute plasticity metrics from relay gain (primary modifiable weights)
        plasticity = None
        if self.config.learn:
            plasticity = compute_plasticity_metrics(
                weights=self.relay_gain.data,
                learning_rate=self.config.learning_rate,
            )
            # Add other connection statistics
            plasticity["input_to_trn_mean"] = float(self.input_to_trn.data.mean().item())  # type: ignore[typeddict-unknown-key]
            plasticity["relay_to_trn_mean"] = float(self.relay_to_trn.data.mean().item())  # type: ignore[typeddict-unknown-key]
            plasticity["trn_to_relay_mean"] = float(self.trn_to_relay.data.mean().item())  # type: ignore[typeddict-unknown-key]
            plasticity["trn_recurrent_mean"] = float(self.trn_recurrent.data.mean().item())  # type: ignore[typeddict-unknown-key]

        # Compute health metrics
        health_tensors = {
            "relay_spikes": (
                self.state.relay_spikes
                if self.state.relay_spikes is not None
                else torch.zeros(self.n_relay, device=self.device)
            ),
            "trn_spikes": (
                self.state.trn_spikes
                if self.state.trn_spikes is not None
                else torch.zeros(self.n_trn, device=self.device)
            ),
        }
        if self.state.relay_membrane is not None:
            health_tensors["relay_membrane"] = self.state.relay_membrane
        if self.state.trn_membrane is not None:
            health_tensors["trn_membrane"] = self.state.trn_membrane

        health = compute_health_metrics(
            state_tensors=health_tensors,
            firing_rate=activity.get("firing_rate", 0.0),
        )

        # Add thalamus-specific health metrics
        if self.state.current_mode is not None:
            burst_fraction = (
                (self.state.current_mode < THALAMUS_MODE_THRESHOLD).float().mean().item()
            )
            health["burst_mode_fraction"] = burst_fraction  # type: ignore[typeddict-unknown-key]
            health["tonic_mode_fraction"] = 1.0 - burst_fraction  # type: ignore[typeddict-unknown-key]

        if self.state.alpha_gate is not None:
            health["alpha_gate_mean"] = self.state.alpha_gate.mean().item()  # type: ignore[typeddict-unknown-key]
            health["alpha_gate_std"] = self.state.alpha_gate.std().item()  # type: ignore[typeddict-unknown-key]

        # Neuromodulator metrics
        neuromodulators = {
            "dopamine": self.state.dopamine,
            "norepinephrine": self.state.norepinephrine,
            "acetylcholine": self.state.acetylcholine,
        }

        # Region-specific custom metrics
        region_specific = {
            "architecture": {
                "n_relay": self.n_relay,
                "n_trn": self.n_trn,
            },
            "relay_activity": {},
            "trn_activity": {},
            "oscillations": {
                "alpha_phase": self._alpha_phase,
                "alpha_amplitude": self._alpha_amplitude,
            },
        }

        # Relay neuron details
        if self.state.relay_spikes is not None:
            region_specific["relay_activity"] = {
                "active_count": compute_spike_count(self.state.relay_spikes),
                "firing_rate": compute_firing_rate(self.state.relay_spikes),
            }

        # TRN details
        if self.state.trn_spikes is not None:
            region_specific["trn_activity"] = {
                "active_count": compute_spike_count(self.state.trn_spikes),
                "firing_rate": compute_firing_rate(self.state.trn_spikes),
            }

        # Return in standardized format
        return {
            "activity": activity,
            "plasticity": plasticity,
            "health": health,
            "neuromodulators": neuromodulators,
            "region_specific": region_specific,
        }

    def grow_output(
        self,
        n_new: int,
        initialization: str = "sparse_random",
        sparsity: float = 0.1,
    ) -> None:
        """Grow thalamus output dimension (relay neuron population).

        Expands relay and TRN neurons, and all associated weight matrices.

        Args:
            n_new: Number of relay neurons to add
            initialization: Weight init strategy ('sparse_random', 'xavier', 'uniform')
            sparsity: Connection sparsity for new neurons (if sparse_random)

        Example:
            >>> # Thalamus grows to send more neurons to cortex
            >>> thalamus.grow_output(30)
        """
        old_n_relay = self.n_relay
        old_n_trn = self.n_trn
        new_n_relay = old_n_relay + n_new
        # Maintain current TRN ratio (compute from current sizes, not config)
        new_n_trn = int(new_n_relay * self.n_trn / self.n_relay)
        n_trn_growth = new_n_trn - old_n_trn

        # Use GrowthMixin helpers (Architecture Review 2025-12-24, Tier 2.5)
        # 1. Expand relay_gain [n_relay] → [n_relay+n_new]
        new_relay_gains = torch.ones(n_new, device=self.device) * self.config.relay_strength
        self.relay_gain.data = torch.cat([self.relay_gain.data, new_relay_gains])

        # 2. Expand input_to_trn [n_trn, input] → [n_trn+growth, input]
        if n_trn_growth > 0:
            self.input_to_trn.data = self._grow_weight_matrix_rows(
                self.input_to_trn.data, n_trn_growth, initializer=initialization, sparsity=sparsity
            )

        # 3. Expand relay_to_trn [n_trn, n_relay] → [n_trn+growth, n_relay+n_new]
        if n_trn_growth > 0:
            # Add rows then columns for new TRN and relay neurons
            expanded_relay_trn = self._grow_weight_matrix_rows(
                self.relay_to_trn.data, n_trn_growth, initializer=initialization, sparsity=sparsity
            )
            self.relay_to_trn.data = self._grow_weight_matrix_cols(
                expanded_relay_trn, n_new, initializer=initialization, sparsity=sparsity
            )
        else:
            # Just add columns
            self.relay_to_trn.data = self._grow_weight_matrix_cols(
                self.relay_to_trn.data, n_new, initializer=initialization, sparsity=sparsity
            )

        # 4. Expand trn_to_relay [n_relay, n_trn] → [n_relay+n_new, n_trn+growth]
        # Add rows for new relay neurons
        expanded_trn_relay = self._grow_weight_matrix_rows(
            self.trn_to_relay.data, n_new, initializer=initialization, sparsity=sparsity
        )
        if n_trn_growth > 0:
            # Add columns for new TRN neurons
            self.trn_to_relay.data = self._grow_weight_matrix_cols(
                expanded_trn_relay, n_trn_growth, initializer=initialization, sparsity=sparsity
            )
        else:
            self.trn_to_relay.data = expanded_trn_relay

        # 5. Expand trn_recurrent [n_trn, n_trn] → [n_trn+growth, n_trn+growth]
        if n_trn_growth > 0:
            expanded_trn_recurrent = self._grow_weight_matrix_rows(
                self.trn_recurrent.data, n_trn_growth, initializer=initialization, sparsity=sparsity
            )
            self.trn_recurrent.data = self._grow_weight_matrix_cols(
                expanded_trn_recurrent, n_trn_growth, initializer=initialization, sparsity=sparsity
            )

        # 6. Expand neuron populations using efficient in-place growth (ConductanceLIF)
        self.n_relay = new_n_relay
        self.n_trn = new_n_trn

        # Grow relay neurons
        self.relay_neurons.grow_neurons(n_new)

        # Grow TRN neurons (manual - ratio-based scaling is complex)
        if n_trn_growth > 0:
            self.trn_neurons.grow_neurons(n_trn_growth)

        # Phase 2: Auto-grow registered STP modules
        self._auto_grow_registered_components("output", n_new)

        # 7. Update instance size tracking
        self.relay_size = new_n_relay
        self.trn_size = new_n_trn
        self.n_output = new_n_relay
        self.total_neurons = new_n_relay + new_n_trn

        # 8. Rebuild center-surround filter with new output size
        self._build_center_surround_filter()

    def grow_relay(
        self,
        n_new: int,
        initialization: str = "sparse_random",
        sparsity: float = 0.1,
    ) -> None:
        """Grow relay neuron population (SEMANTIC API).

        Args:
            n_new: Number of relay neurons to add
            initialization: Weight init strategy
            sparsity: Connection sparsity

        Note:
            Also grows TRN neurons proportionally to maintain current ratio.
        """
        self.grow_output(n_new, initialization, sparsity)

    def learn(
        self,
        input_spikes: torch.Tensor,
        output_spikes: torch.Tensor,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Thalamus is primarily relay - minimal learning.

        In biology, thalamocortical weights are mostly stable,
        but TRN can learn attention patterns. For now, keep fixed.
        """
        return {
            "thalamus_learning": 0.0,
        }

    def get_full_state(self) -> Dict[str, Any]:
        """Get full state for checkpointing."""
        state = self.get_state()
        return state.to_dict()

    def load_full_state(self, state: Dict[str, Any]) -> None:
        """Load full state from checkpoint."""
        state_obj = ThalamicRelayState.from_dict(state, device=str(self.device))
        self.load_state(state_obj)
