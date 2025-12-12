"""
Thalamus - Sensory Relay and Gating

The thalamus is the brain's "sensory switchboard" that:
- Relays sensory information to cortex
- Gates sensory input based on attention and arousal
- Switches between burst and tonic modes
- Modulates cortical excitability via oscillations

Key Features:
=============
1. SENSORY RELAY:
   - All sensory modalities (except olfaction) pass through thalamus
   - Selective routing to appropriate cortical areas
   - Spatial filtering and preprocessing

2. ATTENTIONAL GATING:
   - Alpha oscillations suppress irrelevant inputs
   - Enhanced transmission for attended stimuli
   - Norepinephrine modulates gain

3. MODE SWITCHING:
   - Burst mode: Low input, creates sharp transients (alerting, attention capture)
   - Tonic mode: Steady input, faithful relay (normal processing)
   - Mode controlled by membrane potential and oscillation phase

4. THALAMIC RETICULAR NUCLEUS (TRN):
   - Inhibitory shell around thalamus
   - Implements "searchlight" attention
   - Coordinates coherent oscillations

Biological Basis:
=================
- Lateral geniculate nucleus (LGN): Visual relay
- Medial geniculate nucleus (MGN): Auditory relay
- Ventral posterior nucleus (VPN): Somatosensory relay
- Pulvinar: Visual attention and salience
- Mediodorsal nucleus (MD): Prefrontal coordination

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

from dataclasses import dataclass
from typing import Optional, Dict, Any
import math

import torch
import torch.nn as nn

from thalia.regions.base import NeuralComponent, LearningRule, NeuralComponentConfig, NeuralComponentState
from thalia.core.component_registry import register_region
from thalia.core.neuron import ConductanceLIF, ConductanceLIFConfig
from thalia.core.neuron_constants import (
    V_THRESHOLD_STANDARD, V_RESET_STANDARD, E_LEAK,
    E_EXCITATORY, E_INHIBITORY, G_LEAK_STANDARD,
    TAU_MEM_STANDARD,
)
from thalia.core.weight_init import WeightInitializer


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
    relay_strength: float = 1.2
    """Base relay gain (thalamus amplifies weak inputs)."""

    # Mode switching
    burst_threshold: float = -0.2
    """Membrane potential threshold for burst mode (hyperpolarized)."""

    tonic_threshold: float = 0.3
    """Membrane potential threshold for tonic mode (depolarized)."""

    burst_spike_count: int = 3
    """Number of spikes in a burst (typically 2-5)."""

    burst_gain: float = 2.0
    """Amplification factor for burst mode (alerting signal)."""

    # Attention gating (alpha oscillation)
    alpha_suppression_strength: float = 0.5
    """How strongly alpha suppresses unattended inputs (0-1)."""

    alpha_gate_threshold: float = 0.0
    """Alpha phase threshold for suppression (0 = trough, π = peak)."""

    # TRN (inhibitory shell) - simplified
    trn_ratio: float = 0.2
    """TRN neurons as fraction of relay neurons."""

    trn_inhibition_strength: float = 0.3
    """Strength of TRN → relay inhibition."""

    trn_recurrent_strength: float = 0.4
    """TRN recurrent inhibition (for oscillations)."""

    # Sensory filtering
    spatial_filter_width: float = 0.15
    """Gaussian filter width for center-surround (as fraction of input)."""

    center_excitation: float = 1.5
    """Center enhancement in receptive field."""

    surround_inhibition: float = 0.5
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
    author="Thalia Project"
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
        relay_config = ConductanceLIFConfig(
            v_threshold=V_THRESHOLD_STANDARD,
            v_reset=V_RESET_STANDARD,
            E_L=E_LEAK,
            E_E=E_EXCITATORY,
            E_I=E_INHIBITORY,
            g_L=G_LEAK_STANDARD,
            tau_mem=TAU_MEM_STANDARD,
            tau_E=5.0,  # Fast excitatory
            tau_I=10.0,  # Slower inhibitory (from TRN)
        )
        self.relay_neurons = ConductanceLIF(n_neurons=self.n_relay, config=relay_config)
        self.relay_neurons.to(self.device)

        # =====================================================================
        # TRN NEURONS (Inhibitory, GABAergic)
        # =====================================================================
        trn_config = ConductanceLIFConfig(
            v_threshold=V_THRESHOLD_STANDARD,
            v_reset=V_RESET_STANDARD,
            E_L=E_LEAK,
            E_E=E_EXCITATORY,
            E_I=E_INHIBITORY,
            g_L=G_LEAK_STANDARD * 1.2,  # Slightly faster dynamics
            tau_mem=TAU_MEM_STANDARD * 0.8,
            tau_E=4.0,  # Very fast excitatory
            tau_I=8.0,  # Fast inhibitory
        )
        self.trn_neurons = ConductanceLIF(n_neurons=self.n_trn, config=trn_config)
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
                sparsity=0.3,
                scale=0.3,
                device=self.device,
            )
        )

        # Relay → TRN (collateral activation)
        self.relay_to_trn = nn.Parameter(
            WeightInitializer.sparse_random(
                n_output=self.n_trn,
                n_input=self.n_relay,
                sparsity=0.2,
                scale=0.4,
                device=self.device,
            )
        )

        # TRN → Relay (inhibitory feedback)
        self.trn_to_relay = nn.Parameter(
            WeightInitializer.sparse_random(
                n_output=self.n_relay,
                n_input=self.n_trn,
                sparsity=0.3,
                scale=config.trn_inhibition_strength,
                device=self.device,
            )
        )

        # TRN → TRN (recurrent inhibition for oscillations)
        self.trn_recurrent = nn.Parameter(
            WeightInitializer.sparse_random(
                n_output=self.n_trn,
                n_input=self.n_trn,
                sparsity=0.2,
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

    def _get_learning_rule(self) -> LearningRule:
        """Thalamus primarily relays - minimal learning."""
        return LearningRule.HEBBIAN  # Minimal plasticity

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

        alpha_modulation = 0.5 * (1.0 + math.cos(self._alpha_phase))
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
        ne_gain = 1.0 + 0.5 * self.state.norepinephrine  # [1.0, 1.5]
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
        burst_mask = (current_mode < 0.5)  # Burst mode, [n_relay]
        burst_amplified = relay_spikes.float()  # [n_relay]

        if burst_mask.any():
            # Amplify burst spikes
            burst_amplified = torch.where(
                burst_mask,
                burst_amplified * self.thalamus_config.burst_gain,
                burst_amplified
            )

        # Binarize and convert to bool (ADR-004)
        relay_output = (burst_amplified > 0.5)  # [n_relay], bool

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
        """Get thalamic diagnostic information."""
        diagnostics = super().get_diagnostics()

        # Relay neuron stats (ADR-004: convert bool to float for mean)
        if self.state.relay_spikes is not None:
            diagnostics['thalamus_relay_firing_rate'] = (
                self.state.relay_spikes.float().mean().item()
            )

        if self.state.relay_membrane is not None:
            diagnostics['thalamus_relay_membrane_mean'] = (
                self.state.relay_membrane.mean().item()
            )

        # TRN stats (ADR-004: convert bool to float for mean)
        if self.state.trn_spikes is not None:
            diagnostics['thalamus_trn_firing_rate'] = (
                self.state.trn_spikes.float().mean().item()
            )

        # Mode distribution
        if self.state.current_mode is not None:
            burst_fraction = (self.state.current_mode < 0.5).float().mean().item()
            diagnostics['thalamus_burst_mode_fraction'] = burst_fraction
            diagnostics['thalamus_tonic_mode_fraction'] = 1.0 - burst_fraction

        # Gating
        if self.state.alpha_gate is not None:
            diagnostics['thalamus_alpha_gate_mean'] = (
                self.state.alpha_gate.mean().item()
            )

        diagnostics['thalamus_alpha_phase'] = self._alpha_phase

        # Neuromodulator levels
        diagnostics['thalamus_dopamine'] = self.state.dopamine
        diagnostics['thalamus_norepinephrine'] = self.state.norepinephrine
        diagnostics['thalamus_acetylcholine'] = self.state.acetylcholine

        return diagnostics

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
