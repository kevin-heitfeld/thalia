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

FILE ORGANIZATION (680 lines)
==============================
Lines 1-95:    Module docstring, imports, ThalamicRelayConfig
Lines 96-175:  ThalamicRelay class __init__, weight initialization
Lines 176-320: Forward pass (relay neurons + TRN inhibition)
Lines 321-450: Mode switching (burst vs tonic)
Lines 451-550: Attentional gating (alpha oscillations)
Lines 551-620: Diagnostics and health monitoring
Lines 621-680: Utility methods (reset_state, get_full_state)

NAVIGATION TIP: Use VSCode's "Go to Symbol" (Ctrl+Shift+O) to jump between methods.

When to Use:
============
- Between sensory input and cortex
- For attentional modulation of sensory processing
- Sleep-wake state transitions (burst vs tonic)
- Implementing sensory gating and filtering

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

References:
- Sherman & Guillery (2002): The role of the thalamus in cortical function
- Jones (2007): The thalamus (2nd edition)
- Halassa & Kastner (2017): Thalamic functions in distributed cognitive control
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Dict, Any
import math

import torch
import torch.nn as nn

from thalia.components.neurons.neuron_factory import create_relay_neurons, create_trn_neurons
from thalia.regions.base import NeuralComponent, NeuralComponentConfig, NeuralComponentState
from thalia.managers.component_registry import register_region
from thalia.components.synapses.weight_init import WeightInitializer
from thalia.regulation.region_constants import (
    THALAMUS_BURST_THRESHOLD,
    THALAMUS_TONIC_THRESHOLD,
    THALAMUS_BURST_SPIKE_COUNT,
    THALAMUS_BURST_GAIN,
    THALAMUS_ALPHA_SUPPRESSION,
    THALAMUS_ALPHA_GATE_THRESHOLD,
    THALAMUS_TRN_RATIO,
    THALAMUS_TRN_INHIBITION,
    THALAMUS_TRN_RECURRENT,
    THALAMUS_SPATIAL_FILTER_WIDTH,
    THALAMUS_CENTER_EXCITATION,
    THALAMUS_SURROUND_INHIBITION,
    THALAMUS_RELAY_STRENGTH,
    THALAMUS_NE_GAIN_SCALE,
    THALAMUS_MODE_THRESHOLD,
    THALAMUS_RELAY_SPARSITY,
    THALAMUS_RELAY_SCALE,
    THALAMUS_TRN_FEEDBACK_SPARSITY,
    THALAMUS_TRN_FEEDBACK_SCALE,
    THALAMUS_TRN_FEEDFORWARD_SPARSITY,
    THALAMUS_SPATIAL_CENTER_SPARSITY,
)


@dataclass
class ThalamicRelayConfig(NeuralComponentConfig):
    """Configuration for thalamic relay nucleus.

    Thalamus sits between sensory input and cortex, providing:
    - Sensory gating (alpha-based suppression)
    - Mode switching (burst vs tonic)
    - Gain modulation (norepinephrine)
    - Spatial filtering
    """

    # Relay parameters
    relay_strength: float = THALAMUS_RELAY_STRENGTH
    """Base relay gain (thalamus amplifies weak inputs)."""

    # Mode switching
    burst_threshold: float = THALAMUS_BURST_THRESHOLD
    """Membrane potential threshold for burst mode (hyperpolarized)."""

    tonic_threshold: float = THALAMUS_TONIC_THRESHOLD
    """Membrane potential threshold for tonic mode (depolarized)."""

    burst_spike_count: int = THALAMUS_BURST_SPIKE_COUNT
    """Number of spikes in a burst (typically 2-5)."""

    burst_gain: float = THALAMUS_BURST_GAIN
    """Amplification factor for burst mode (alerting signal)."""

    # Attention gating (alpha oscillation)
    alpha_suppression_strength: float = THALAMUS_ALPHA_SUPPRESSION
    """How strongly alpha suppresses unattended inputs (0-1)."""

    alpha_gate_threshold: float = THALAMUS_ALPHA_GATE_THRESHOLD
    """Alpha phase threshold for suppression (0 = trough, π = peak)."""

    # TRN (inhibitory shell) - simplified
    trn_ratio: float = THALAMUS_TRN_RATIO
    """TRN neurons as fraction of relay neurons."""

    trn_inhibition_strength: float = THALAMUS_TRN_INHIBITION
    """Strength of TRN → relay inhibition."""

    trn_recurrent_strength: float = THALAMUS_TRN_RECURRENT
    """TRN recurrent inhibition (for oscillations)."""

    # Sensory filtering
    spatial_filter_width: float = THALAMUS_SPATIAL_FILTER_WIDTH
    """Gaussian filter width for center-surround (as fraction of input)."""

    center_excitation: float = THALAMUS_CENTER_EXCITATION
    """Center enhancement in receptive field."""

    surround_inhibition: float = THALAMUS_SURROUND_INHIBITION
    """Surround suppression in receptive field."""


@dataclass
class ThalamicRelayState(NeuralComponentState):
    """State for thalamic relay nucleus."""

    # Relay neuron state
    relay_spikes: Optional[torch.Tensor] = None
    relay_membrane: Optional[torch.Tensor] = None

    # TRN state
    trn_spikes: Optional[torch.Tensor] = None
    trn_membrane: Optional[torch.Tensor] = None

    # Mode state
    current_mode: Optional[torch.Tensor] = None  # 0=burst, 1=tonic
    burst_counter: Optional[torch.Tensor] = None  # Tracks spikes in burst

    # Gating state
    alpha_gate: Optional[torch.Tensor] = None  # Current gating factor [0, 1]


@register_region(
    "thalamus",
    aliases=["thalamic_relay"],
    description="Sensory relay and gating with burst/tonic modes and attentional modulation",
    version="1.0",
    author="Thalia Project",
    config_class=ThalamicRelayConfig,
)
class ThalamicRelay(NeuralComponent):
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

    From NeuralComponent (abstract base):
    ------------------------------------
    - forward(input, **kwargs) → Tensor
    - reset_state() → None
    - get_diagnostics() → Dict
    - set_oscillator_phases(phases, signals) → None
    - Neuromodulator control methods
    """

    def __init__(self, config: ThalamicRelayConfig):
        """Initialize thalamic relay.

        Args:
            config: Thalamic relay configuration
        """
        self.thalamus_config = config
        super().__init__(config)

        # Compute layer sizes
        self.n_relay = config.n_output  # Relay neurons = output size
        self.n_trn = int(config.n_output * config.trn_ratio)

        # =====================================================================
        # RELAY NEURONS (Excitatory, glutamatergic)
        # =====================================================================
        self.relay_neurons = create_relay_neurons(self.n_relay, self.device)
        self.relay_neurons.to(self.device)

        # =====================================================================
        # TRN NEURONS (Inhibitory, GABAergic)
        # =====================================================================
        self.trn_neurons = create_trn_neurons(self.n_trn, self.device)
        self.trn_neurons.to(self.device)

        # =====================================================================
        # WEIGHTS
        # =====================================================================

        # Relay gain per neuron (learned modulation of filtered input)
        self.relay_gain = nn.Parameter(
            torch.ones(self.n_relay, device=self.device) * config.relay_strength
        )

        # Input → TRN (collateral activation)
        self.input_to_trn = nn.Parameter(
            WeightInitializer.sparse_random(
                n_output=self.n_trn,
                n_input=config.n_input,
                sparsity=THALAMUS_RELAY_SPARSITY,
                scale=THALAMUS_RELAY_SCALE,
                device=self.device,
            )
        )

        # Relay → TRN (collateral activation)
        self.relay_to_trn = nn.Parameter(
            WeightInitializer.sparse_random(
                n_output=self.n_trn,
                n_input=self.n_relay,
                sparsity=THALAMUS_TRN_FEEDBACK_SPARSITY,
                scale=THALAMUS_TRN_FEEDBACK_SCALE,
                device=self.device,
            )
        )

        # TRN → Relay (inhibitory feedback)
        self.trn_to_relay = nn.Parameter(
            WeightInitializer.sparse_random(
                n_output=self.n_relay,
                n_input=self.n_trn,
                sparsity=THALAMUS_TRN_FEEDFORWARD_SPARSITY,
                scale=config.trn_inhibition_strength,
                device=self.device,
            )
        )

        # TRN → TRN (recurrent inhibition for oscillations)
        self.trn_recurrent = nn.Parameter(
            WeightInitializer.sparse_random(
                n_output=self.n_trn,
                n_input=self.n_trn,
                sparsity=THALAMUS_SPATIAL_CENTER_SPARSITY,
                scale=config.trn_recurrent_strength,
                device=self.device,
            )
        )

        # =====================================================================
        # CENTER-SURROUND RECEPTIVE FIELDS
        # =====================================================================
        self._build_center_surround_filter()

        # =====================================================================
        # STATE
        # =====================================================================
        self.state = ThalamicRelayState()

        # Oscillator state (provided by brain)
        self._alpha_phase: float = 0.0
        self._alpha_amplitude: float = 1.0

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
        input_idx = torch.arange(self.config.n_input, device=self.device).float()

        # Scale to [0, 1]
        relay_norm = relay_idx / max(1, self.n_relay - 1)
        input_norm = input_idx / max(1, self.config.n_input - 1)

        # Distance matrix [n_relay, n_input]
        distances = torch.abs(
            relay_norm.unsqueeze(1) - input_norm.unsqueeze(0)
        )

        # Center (narrow Gaussian)
        width_center = self.thalamus_config.spatial_filter_width
        center = self.thalamus_config.center_excitation * torch.exp(
            -distances**2 / (2 * width_center**2)
        )

        # Surround (wider Gaussian)
        width_surround = width_center * 3.0
        surround = self.thalamus_config.surround_inhibition * torch.exp(
            -distances**2 / (2 * width_surround**2)
        )

        # DoG filter
        self.register_buffer(
            'center_surround_filter',
            center - surround
        )

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
        # Store alpha phase for gating
        if 'alpha' in phases:
            self._alpha_phase = phases['alpha']

        if signals and 'alpha' in signals:
            self._alpha_amplitude = signals['alpha']

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

        alpha_modulation = THALAMUS_ALPHA_SUPPRESSION * (1.0 + math.cos(self._alpha_phase))
        gate = 1.0 - self.thalamus_config.alpha_suppression_strength * alpha_modulation

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
        burst_mask = membrane < self.thalamus_config.burst_threshold
        tonic_mask = membrane > self.thalamus_config.tonic_threshold

        self.state.current_mode = torch.where(
            burst_mask,
            torch.zeros_like(membrane),  # Burst mode
            torch.where(
                tonic_mask,
                torch.ones_like(membrane),  # Tonic mode
                self.state.current_mode  # Maintain
            )
        )

        return self.state.current_mode

    def forward(
        self,
        input_spikes: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Process sensory input through thalamic relay.

        Args:
            input_spikes: Sensory input spikes [n_input] (1D, ADR-005)
            **kwargs: Additional arguments (ignored)

        Returns:
            Relay output spikes [n_relay] (bool, ADR-004/005)
        """
        # ADR-005: Expect 1D input
        assert input_spikes.dim() == 1, (
            f"Thalamus.forward: Expected 1D input (ADR-005), got shape {input_spikes.shape}"
        )
        assert input_spikes.shape[0] == self.config.n_input, (
            f"Thalamus.forward: input has {input_spikes.shape[0]} neurons, expected {self.config.n_input}"
        )

        # =====================================================================
        # 1. APPLY CENTER-SURROUND SPATIAL FILTER
        # =====================================================================
        # Filter input spikes spatially before relay (ADR-005: 1D)
        # Convert bool to float for matmul (ADR-004)
        input_float = input_spikes.float()
        filtered_input = torch.mv(
            self.center_surround_filter,
            input_float
        ).clamp(min=0)  # [n_relay]

        # =====================================================================
        # 2. COMPUTE ALPHA ATTENTIONAL GATE
        # =====================================================================
        alpha_gate = self._compute_alpha_gate()  # [n_relay]
        self.state.alpha_gate = alpha_gate

        # =====================================================================
        # 3. RELAY NEURONS: Filtered input → Relay
        # =====================================================================
        # Apply alpha gating to filtered input
        gated_input = filtered_input * alpha_gate

        # Apply learned per-neuron gain modulation
        relay_excitation = gated_input * self.relay_gain  # [n_relay]

        # Apply norepinephrine gain modulation (arousal)
        ne_gain = 1.0 + THALAMUS_NE_GAIN_SCALE * self.state.norepinephrine
        relay_excitation = relay_excitation * ne_gain

        # TRN inhibition of relay
        if self.state.trn_spikes is not None:
            # Convert bool to float for matmul (ADR-004)
            relay_inhibition = torch.mv(
                self.trn_to_relay,
                self.state.trn_spikes.float()
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
                burst_mask,
                burst_amplified * self.thalamus_config.burst_gain,
                burst_amplified
            )

        # Binarize and convert to bool (ADR-004)
        relay_output = burst_amplified > THALAMUS_MODE_THRESHOLD  # [n_relay], bool

        # =====================================================================
        # 5. TRN NEURONS: Input collaterals + Relay collaterals
        # =====================================================================
        # TRN receives:
        # - Input collaterals (sensory copy)
        # - Relay collaterals (relay activity)
        # - Recurrent inhibition (TRN-TRN)

        # Convert bool to float for matmul (ADR-004)
        trn_excitation_input = torch.mv(self.input_to_trn, input_float)  # [n_trn]
        trn_excitation_relay = torch.mv(self.relay_to_trn, relay_output.float())  # [n_trn]
        trn_excitation = trn_excitation_input + trn_excitation_relay  # [n_trn]

        # TRN recurrent inhibition
        if self.state.trn_spikes is not None:
            trn_inhibition = torch.mv(
                self.trn_recurrent,
                self.state.trn_spikes.float()
            )  # [n_trn]
        else:
            trn_inhibition = torch.zeros(self.n_trn, device=self.device)

        # Update TRN neurons (ADR-005: 1D tensors)
        trn_spikes, trn_membrane = self.trn_neurons(
            g_exc_input=trn_excitation,  # [n_trn]
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

        # Store for component state (NeuralComponent protocol)
        self.state.membrane = self.state.relay_membrane
        self.state.spikes = self.state.relay_spikes

        return relay_output  # [n_relay], bool (ADR-004/005)

    def reset_state(self) -> None:
        """Reset thalamic state."""
        super().reset_state()

        self.relay_neurons.reset_state()
        self.trn_neurons.reset_state()

        self.state.relay_spikes = None
        self.state.relay_membrane = None
        self.state.trn_spikes = None
        self.state.trn_membrane = None
        self.state.current_mode = None
        self.state.burst_counter = None
        self.state.alpha_gate = None

        self._alpha_phase = 0.0
        self._alpha_amplitude = 1.0

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get thalamic diagnostic information using DiagnosticsMixin helpers."""
        # Custom metrics specific to thalamus
        custom = {
            "n_relay": self.n_relay,
            "n_trn": self.n_trn,
            "alpha_phase": self._alpha_phase,
            "alpha_amplitude": self._alpha_amplitude,
        }

        # Relay neuron stats using spike_diagnostics helper
        if self.state.relay_spikes is not None:
            custom.update(self.spike_diagnostics(self.state.relay_spikes, "relay"))

        # TRN stats using spike_diagnostics helper
        if self.state.trn_spikes is not None:
            custom.update(self.spike_diagnostics(self.state.trn_spikes, "trn"))

        # Relay membrane using membrane_diagnostics helper
        if self.state.relay_membrane is not None:
            custom.update(self.membrane_diagnostics(
                self.state.relay_membrane, threshold=1.0, prefix="relay"
            ))

        # TRN membrane using membrane_diagnostics helper
        if self.state.trn_membrane is not None:
            custom.update(self.membrane_diagnostics(
                self.state.trn_membrane, threshold=1.0, prefix="trn"
            ))

        # Mode distribution (thalamus-specific)
        if self.state.current_mode is not None:
            burst_fraction = (self.state.current_mode < THALAMUS_MODE_THRESHOLD).float().mean().item()
            custom["burst_mode_fraction"] = burst_fraction
            custom["tonic_mode_fraction"] = 1.0 - burst_fraction

        # Alpha gating (thalamus-specific)
        if self.state.alpha_gate is not None:
            custom["alpha_gate_mean"] = self.state.alpha_gate.mean().item()
            custom["alpha_gate_std"] = self.state.alpha_gate.std().item()

        # Neuromodulator levels
        custom["dopamine"] = self.state.dopamine
        custom["norepinephrine"] = self.state.norepinephrine
        custom["acetylcholine"] = self.state.acetylcholine

        # Use collect_standard_diagnostics for weights
        return self.collect_standard_diagnostics(
            region_name="thalamus",
            weight_matrices={
                "relay_gain": self.relay_gain.data,
                "input_to_trn": self.input_to_trn.data,
                "relay_to_trn": self.relay_to_trn.data,
                "trn_to_relay": self.trn_to_relay.data,
                "trn_recurrent": self.trn_recurrent.data,
            },
            custom_metrics=custom,
        )

    def grow_input(
        self,
        n_new: int,
        initialization: str = 'sparse_random',
        sparsity: float = 0.1,
    ) -> None:
        """Grow thalamus input dimension when upstream region grows.

        Expands input weight matrices by adding columns to accept larger input.

        Args:
            n_new: Number of input neurons to add
            initialization: Weight init strategy ('sparse_random', 'xavier', 'uniform')
            sparsity: Connection sparsity for new input neurons (if sparse_random)

        Example:
            >>> # Sensory pathway grows from 784 → 804 neurons
            >>> sensory_pathway.grow_output(20)
            >>> # Thalamus must expand input dimension
            >>> thalamus.grow_input(20)
        """
        old_n_input = self.config.n_input
        new_n_input = old_n_input + n_new

        # Helper to create new weights
        def new_weights_for(n_out: int, n_in: int) -> torch.Tensor:
            if initialization == 'xavier':
                return WeightInitializer.xavier(n_out, n_in, device=self.device)
            elif initialization == 'sparse_random':
                return WeightInitializer.sparse_random(n_out, n_in, sparsity, device=self.device)
            else:
                return WeightInitializer.uniform(n_out, n_in, device=self.device)

        # Expand input_to_trn [n_trn, input] → [n_trn, input+n_new]
        new_input_trn_cols = new_weights_for(self.n_trn, n_new)
        self.input_to_trn.data = torch.cat([self.input_to_trn.data, new_input_trn_cols], dim=1)

        # Rebuild center-surround filter with new input size
        self.config = replace(self.config, n_input=new_n_input)
        self._build_center_surround_filter()

        # Update config
        self.thalamus_config = replace(self.thalamus_config, n_input=new_n_input)

    def grow_output(
        self,
        n_new: int,
        initialization: str = 'sparse_random',
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
        new_n_trn = int(new_n_relay * self.thalamus_config.trn_ratio)
        n_trn_growth = new_n_trn - old_n_trn

        # Helper to create new weights
        def new_weights_for(n_out: int, n_in: int) -> torch.Tensor:
            if initialization == 'xavier':
                return WeightInitializer.xavier(n_out, n_in, device=self.device)
            elif initialization == 'sparse_random':
                return WeightInitializer.sparse_random(n_out, n_in, sparsity, device=self.device)
            else:
                return WeightInitializer.uniform(n_out, n_in, device=self.device)

        # 1. Expand relay_gain [n_relay] → [n_relay+n_new]
        new_relay_gains = torch.ones(n_new, device=self.device) * self.thalamus_config.relay_strength
        self.relay_gain.data = torch.cat([self.relay_gain.data, new_relay_gains])

        # 2. Expand input_to_trn [n_trn, input] → [n_trn+growth, input]
        if n_trn_growth > 0:
            new_input_trn_rows = new_weights_for(n_trn_growth, self.config.n_input)
            self.input_to_trn.data = torch.cat([self.input_to_trn.data, new_input_trn_rows], dim=0)

        # 3. Expand relay_to_trn [n_trn, n_relay] → [n_trn+growth, n_relay+n_new]
        if n_trn_growth > 0:
            # Add rows for new TRN neurons
            new_relay_trn_rows = new_weights_for(n_trn_growth, old_n_relay)
            expanded_relay_trn = torch.cat([self.relay_to_trn.data, new_relay_trn_rows], dim=0)
            # Add columns for new relay neurons
            new_relay_trn_cols = new_weights_for(new_n_trn, n_new)
            self.relay_to_trn.data = torch.cat([expanded_relay_trn, new_relay_trn_cols], dim=1)
        else:
            # Just add columns
            new_relay_trn_cols = new_weights_for(old_n_trn, n_new)
            self.relay_to_trn.data = torch.cat([self.relay_to_trn.data, new_relay_trn_cols], dim=1)

        # 4. Expand trn_to_relay [n_relay, n_trn] → [n_relay+n_new, n_trn+growth]
        # Add rows for new relay neurons
        new_trn_relay_rows = new_weights_for(n_new, old_n_trn)
        expanded_trn_relay = torch.cat([self.trn_to_relay.data, new_trn_relay_rows], dim=0)
        if n_trn_growth > 0:
            # Add columns for new TRN neurons
            new_trn_relay_cols = new_weights_for(new_n_relay, n_trn_growth)
            self.trn_to_relay.data = torch.cat([expanded_trn_relay, new_trn_relay_cols], dim=1)
        else:
            self.trn_to_relay.data = expanded_trn_relay

        # 5. Expand trn_recurrent [n_trn, n_trn] → [n_trn+growth, n_trn+growth]
        if n_trn_growth > 0:
            new_trn_recurrent_rows = new_weights_for(n_trn_growth, old_n_trn)
            expanded_trn_recurrent = torch.cat([self.trn_recurrent.data, new_trn_recurrent_rows], dim=0)
            new_trn_recurrent_cols = new_weights_for(new_n_trn, n_trn_growth)
            self.trn_recurrent.data = torch.cat([expanded_trn_recurrent, new_trn_recurrent_cols], dim=1)

        # 6. Expand neuron populations
        self.n_relay = new_n_relay
        self.n_trn = new_n_trn
        self.relay_neurons = create_relay_neurons(self.n_relay, self.device)
        self.relay_neurons.to(self.device)
        self.trn_neurons = create_trn_neurons(self.n_trn, self.device)
        self.trn_neurons.to(self.device)

        # 7. Rebuild center-surround filter with new output size
        self._build_center_surround_filter()

        # 8. Update configs
        self.config = replace(self.config, n_output=new_n_relay)
        self.thalamus_config = replace(self.thalamus_config, n_output=new_n_relay)

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
            'thalamus_learning': 0.0,
        }

    def get_full_state(self) -> Dict[str, Any]:
        """Get full state for checkpointing."""
        return {
            'relay_spikes': self.state.relay_spikes,
            'relay_membrane': self.state.relay_membrane,
            'trn_spikes': self.state.trn_spikes,
            'trn_membrane': self.state.trn_membrane,
            'current_mode': self.state.current_mode,
            'alpha_gate': self.state.alpha_gate,
            'alpha_phase': self._alpha_phase,
            'alpha_amplitude': self._alpha_amplitude,
        }

    def load_full_state(self, state: Dict[str, Any]) -> None:
        """Load full state from checkpoint."""
        self.state.relay_spikes = state.get('relay_spikes')
        self.state.relay_membrane = state.get('relay_membrane')
        self.state.trn_spikes = state.get('trn_spikes')
        self.state.trn_membrane = state.get('trn_membrane')
        self.state.current_mode = state.get('current_mode')
        self.state.alpha_gate = state.get('alpha_gate')
        self._alpha_phase = state.get('alpha_phase', 0.0)
        self._alpha_amplitude = state.get('alpha_amplitude', 1.0)


__all__ = [
    'ThalamicRelay',
    'ThalamicRelayConfig',
    'ThalamicRelayState',
]
