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

from thalia.brain.configs import ThalamusConfig
from thalia.components import (
    ShortTermPlasticity,
    STPConfig,
    GapJunctionConfig,
    GapJunctionCoupling,
    WeightInitializer,
    NeuronFactory,
)
from thalia.diagnostics import compute_plasticity_metrics
from thalia.learning import (
    STDPStrategy,
    STDPConfig,
)
from thalia.typing import (
    PopulationName,
    PopulationSizes,
    RegionSpikesDict,
    SpikesSourceKey,
)
from thalia.utils import clamp_weights

from ..neural_region import NeuralRegion
from ..region_registry import register_region


GAIN_MINIMUM_THALAMUS = 0.1  # Increased from 0.001 to allow recovery
GAIN_MAXIMUM_THALAMUS = 5.0  # Added upper bound to prevent explosion

THALAMUS_MODE_THRESHOLD = 0.5  # Threshold for burst/tonic mode detection (0=burst, 1=tonic)
THALAMUS_SURROUND_WIDTH_RATIO = 3.0  # Surround width as multiple of center width


@register_region(
    "thalamus",
    aliases=["thalamic_relay"],
    description="Sensory relay and gating with burst/tonic modes and attentional modulation",
    version="1.0",
    author="Thalia Project",
    config_class=ThalamusConfig,
)
class Thalamus(NeuralRegion[ThalamusConfig]):
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
    """

    OUTPUT_POPULATIONS: Dict[PopulationName, str] = {
        "relay": "relay_size",
        "trn": "trn_size",
    }

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, config: ThalamusConfig, population_sizes: PopulationSizes):
        """Initialize thalamic relay."""
        super().__init__(config=config, population_sizes=population_sizes)

        # =====================================================================
        # EXTRACT LAYER SIZES
        # =====================================================================
        self.relay_size = population_sizes["relay_size"]
        self.trn_size = population_sizes["trn_size"]

        # =====================================================================
        # NEURONS
        # =====================================================================
        # Relay neurons (Excitatory, glutamatergic)
        self.relay_neurons = NeuronFactory.create_relay_neurons(self.relay_size, self.device)

        # TRN neurons (Inhibitory, GABAergic)
        self.trn_neurons = NeuronFactory.create_trn_neurons(self.trn_size, self.device)

        # =====================================================================
        # INITIALIZE STATE VARIABLES
        # =====================================================================
        # TRN state
        self.trn_spikes: torch.Tensor = torch.zeros(self.trn_size, dtype=torch.bool, device=self.device)

        # Mode state (0=burst, 1=tonic)
        # Initialize to tonic
        self.current_mode: torch.Tensor = torch.ones(self.relay_size, device=self.device)

        # =====================================================================
        # WEIGHTS
        # =====================================================================
        # Relay gain per neuron (adaptive via homeostatic plasticity)
        self.relay_gain = nn.Parameter(
            torch.ones(self.relay_size, device=self.device, requires_grad=False)
            * config.relay_strength
        )

        # Homeostatic plasticity: track firing rates for gain adaptation
        self.register_buffer("relay_firing_rate", torch.zeros(self.relay_size, device=self.device))
        self._target_rate = config.target_firing_rate
        self._gain_lr = config.gain_learning_rate
        self._baseline_noise = config.baseline_noise_current

        # Adaptive threshold plasticity
        self._threshold_lr = config.threshold_learning_rate
        self._threshold_min = config.threshold_min
        self._threshold_max = config.threshold_max

        # Exponential moving average decay factor for firing rate
        # tau = 5000ms, dt = 1ms → alpha = 1 - exp(-dt/tau) ≈ 0.0002
        self._firing_rate_alpha = 1.0 - torch.exp(
            torch.tensor(-config.dt_ms / config.gain_tau_ms, device=self.device)
        )

        # Input → TRN (collateral activation)
        # Created lazily when first source is added (size unknown at init)
        self.input_to_trn: Optional[nn.Parameter] = None

        # Relay → TRN (collateral activation)
        self.relay_to_trn = nn.Parameter(
            WeightInitializer.sparse_random(
                n_output=self.trn_size,
                n_input=self.relay_size,
                sparsity=0.2,
                weight_scale=0.4,
                device=self.device,
            ),
            requires_grad=False,
        )

        # TRN → TRN (recurrent inhibition for oscillations)
        self.trn_recurrent = nn.Parameter(
            WeightInitializer.sparse_random(
                n_output=self.trn_size,
                n_input=self.trn_size,
                sparsity=0.2,
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
        # Will be created lazily after first source is added (needs input_to_trn weights)
        self.gap_junctions: Optional[GapJunctionCoupling] = None
        self._gap_junctions_config = GapJunctionConfig(
            coupling_strength=config.gap_junction_strength,
            connectivity_threshold=0.2,  # Liberal coupling (TRN is densely connected)
            max_neighbors=10,  # TRN has ~6-12 gap junction partners (Galarreta 1999)
            interneuron_only=True,  # All TRN neurons are inhibitory
        )

        # =====================================================================
        # CENTER-SURROUND RECEPTIVE FIELDS
        # =====================================================================
        self._build_center_surround_filter()

        # =====================================================================
        # SHORT-TERM PLASTICITY (STP) - HIGH PRIORITY for sensory gating
        # =====================================================================
        # Created with size=0, will grow when sources are added
        # Sensory input → relay depression (U=0.4, moderate depression)
        # Filters repetitive stimuli, responds to novelty
        # CRITICAL for attention capture and change detection
        self.stp_sensory_relay = ShortTermPlasticity(
            n_pre=0,  # Grow when sources added
            n_post=self.relay_size,
            config=STPConfig.from_type(config.stp_sensory_relay_type),
            per_synapse=True,  # Per-synapse dynamics for maximum precision
        )

        # L6 cortical feedback → relay depression (U=0.7, strong depression)
        # Dynamic gain control: Sustained cortical feedback reduces thalamic transmission
        # Enables efficient filtering and sensory gating
        # NOTE: L6 size must match relay_size (validated at build time)
        self.stp_l6_feedback = ShortTermPlasticity(
            n_pre=0,  # Grow when sources added
            n_post=self.relay_size,
            config=STPConfig.from_type(config.stp_l6_feedback_type),
            per_synapse=True,
        )

        # =====================================================================
        # REGISTER SENSORY INPUT SOURCE
        # =====================================================================
        # Register "sensory" as a valid input source for external sensory input
        # This represents ascending sensory pathways (retinogeniculate, etc.)
        # Size is initially set to relay_size (1:1 mapping), but can accept
        # any size due to center-surround filtering in _forward_internal()
        #
        # BIOLOGICAL JUSTIFICATION:
        # - Retina → LGN (Lateral Geniculate Nucleus) for vision
        # - Cochlea → MGN (Medial Geniculate Nucleus) for audition
        # - Mechanoreceptors → VPN (Ventral Posterior Nucleus) for touch
        # - These are direct synaptic connections, NOT routed through TRN
        # - TRN provides lateral inhibition but doesn't receive primary sensory input
        self.input_sources["sensory"] = self.relay_size

        # Initialize sensory input weights (relay_size x relay_size identity-like)
        # This creates a topographic mapping with center-surround filtering
        # The actual filtering is dynamic in _forward_internal() based on input size
        sensory_weights = WeightInitializer.sparse_random(
            n_output=self.relay_size,
            n_input=self.relay_size,
            sparsity=0.2,  # 20% sparsity for sparse sensory mapping
            weight_scale=1.0,  # Strong direct input
            device=self.device,
        )
        self.synaptic_weights["sensory"] = nn.Parameter(sensory_weights, requires_grad=False)

        # Initialize sensory collateral → TRN weights (for feedforward inhibition)
        # Sensory input branches to both relay and TRN simultaneously
        sensory_to_trn_weights = WeightInitializer.sparse_random(
            n_output=self.trn_size,
            n_input=self.relay_size,
            sparsity=0.3,  # Moderate connectivity
            weight_scale=0.4,  # Moderate strength
            device=self.device,
        )
        self.sensory_to_trn = nn.Parameter(sensory_to_trn_weights, requires_grad=False)

        # =====================================================================
        # STDP LEARNING STRATEGY
        # =====================================================================
        # Thalamocortical synapses show robust STDP
        # Critical for sensory learning, attention, and routing
        # Both ascending (sensory→relay) and descending (L6→relay) pathways learn
        stdp_cfg = STDPConfig(
            learning_rate=config.learning_rate,
            a_plus=0.005,  # Moderate LTP (thalamic synapses are conservative)
            a_minus=0.001,  # Weak LTD (5:1 LTP:LTD ratio)
            tau_plus=20.0,  # Standard STDP window
            tau_minus=20.0,
            w_min=config.w_min,
            w_max=config.w_max,
            device=str(self.device),
        )
        self.learning_strategy = STDPStrategy(stdp_cfg)

        # =====================================================================
        # POST-INITIALIZATION
        # =====================================================================
        self.__post_init__()

    def _build_center_surround_filter(self) -> None:
        """Build center-surround receptive field filters.

        Creates topographic mapping between input and relay neurons with
        Gaussian center-surround structure (Mexican hat function).
        """
        # Skip if no input yet
        if self.input_sources == 0:
            return

        self._build_center_surround_filter_for_size(self.n_input)

    def _build_center_surround_filter_for_size(self, sensory_size: int) -> None:
        """Build center-surround filter for specific sensory input size.

        Args:
            sensory_size: Size of sensory input dimension
        """
        if sensory_size == 0:
            return

        # Create topographic index mappings (normalized [0, 1])
        relay_idx = torch.arange(self.relay_size, device=self.device).float()
        input_idx = torch.arange(sensory_size, device=self.device).float()

        # Normalize indices
        relay_norm = relay_idx / max(1, self.relay_size - 1)
        input_norm = input_idx / max(1, sensory_size - 1)

        # Compute distances (shape: relay_size x sensory_size)
        distances = torch.abs(relay_norm.unsqueeze(1) - input_norm.unsqueeze(0))

        # Gaussian center
        width_center = self.config.spatial_filter_width
        center = self.config.center_excitation * torch.exp(
            -(distances**2) / (2 * width_center**2)
        )

        # Gaussian surround (inhibition)
        width_surround = width_center * THALAMUS_SURROUND_WIDTH_RATIO
        surround = self.config.surround_inhibition * torch.exp(
            -(distances**2) / (2 * width_surround**2)
        )

        # DoG filter (Difference of Gaussians)
        self.register_buffer("center_surround_filter", center - surround)

    def _ensure_input_to_trn_initialized(self) -> None:
        """Initialize input_to_trn weights if not already created.

        Called when first source is added. Uses current n_input size.
        """
        if self.input_to_trn is None and self.n_input > 0:
            self.input_to_trn = nn.Parameter(
                WeightInitializer.sparse_random(
                    n_output=self.trn_size,
                    n_input=self.n_input,
                    sparsity=0.3,
                    weight_scale=3.0,
                    device=self.device,
                ),
                requires_grad=False,
            )

            # Create gap junctions if enabled
            if self._gap_junctions_config is not None:
                self.gap_junctions = GapJunctionCoupling(
                    n_neurons=self.trn_size,
                    afferent_weights=self.input_to_trn,  # Shared sensory inputs
                    config=self._gap_junctions_config,
                    interneuron_mask=None,  # All TRN neurons are interneurons
                    device=self.device,
                )

    # =========================================================================
    # SYNAPTIC WEIGHT MANAGEMENT
    # =========================================================================

    def add_input_source(
        self,
        source_name: SpikesSourceKey,
        target_population: PopulationName,
        n_input: int,
        sparsity: float = 0.2,
        weight_scale: float = 0.3,
    ) -> None:
        """Add input source and grow STP modules.

        Overrides base to grow STP when sources are added.
        """
        # Track old size for growth
        old_n_input = self.n_input

        # Call parent to register source
        super().add_input_source(
            source_name=source_name,
            target_population=target_population,
            n_input=n_input,
            sparsity=sparsity,
            weight_scale=weight_scale,
        )

        # Grow STP modules
        n_new = self.n_input - old_n_input
        if n_new > 0:
            self.stp_sensory_relay.grow(n_new, target="pre")
            self.stp_l6_feedback.grow(n_new, target="pre")

        # Initialize input_to_trn if this is first source
        self._ensure_input_to_trn_initialized()

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    def forward(self, region_inputs: RegionSpikesDict) -> RegionSpikesDict:
        """Process sensory input through thalamic relay."""
        self._pre_forward(region_inputs)

        # =====================================================================
        # MULTI-SOURCE SYNAPTIC INTEGRATION
        # =====================================================================
        # Accumulate synaptic currents from all registered sources
        # - "sensory": External sensory input (retina→LGN, etc.)
        # - "cortex:l6a": Corticothalamic feedback to TRN
        # - "cortex:l6b": Corticothalamic feedback to relay
        # - "thalamus:trn": TRN→relay inhibition (recurrent)
        relay_current = torch.zeros(self.relay_size, device=self.device)
        trn_current = torch.zeros(self.trn_size, device=self.device)

        for source_name, source_spikes in region_inputs.items():
            # Convert to float for matrix multiplication
            source_spikes_float = (
                source_spikes.float() if source_spikes.dtype == torch.bool else source_spikes
            )

            # Route to appropriate target neurons
            if source_name == "sensory":
                # External sensory input → relay neurons
                # Apply synaptic weights (includes topographic mapping)
                if "sensory" in self.synaptic_weights:
                    relay_current += self.synaptic_weights["sensory"] @ source_spikes_float

                # Sensory collateral → TRN (for feedforward inhibition)
                trn_current += self.sensory_to_trn @ source_spikes_float

            elif source_name == "cortex:l6a":
                # L6a feedback → TRN (attentional modulation)
                if "cortex:l6a" in self.synaptic_weights:
                    trn_current += self.synaptic_weights["cortex:l6a"] @ source_spikes_float

            elif source_name == "cortex:l6b":
                # L6b feedback → relay (precision modulation)
                if "cortex:l6b" in self.synaptic_weights:
                    relay_current += self.synaptic_weights["cortex:l6b"] @ source_spikes_float

            elif source_name == "thalamus:trn":
                # TRN → relay inhibition (recurrent gating)
                if "thalamus:trn" in self.synaptic_weights:
                    # TRN is inhibitory, apply negative weights
                    relay_current += self.synaptic_weights["thalamus:trn"] @ source_spikes_float

        # =====================================================================
        # COMPUTE ALPHA ATTENTIONAL GATE
        # =====================================================================
        alpha_gate = self._compute_alpha_gate()  # [relay_size]

        # =====================================================================
        # RELAY NEURONS: Synaptic currents → Relay
        # =====================================================================
        # Apply alpha gating to relay current
        gated_current = relay_current * alpha_gate

        # Apply learned per-neuron gain modulation
        relay_excitation = gated_current * self.relay_gain  # [relay_size]

        # Apply norepinephrine gain modulation (arousal)
        ne_level = self._ne_concentration.mean().item() if hasattr(self, '_ne_concentration') else 0.5
        ne_gain = 1.0 + 0.5 * ne_level
        relay_excitation = relay_excitation * ne_gain

        # =====================================================================
        # HOMEOSTATIC INTRINSIC PLASTICITY: Add Baseline Noise (Bootstrap)
        # =====================================================================
        # Add baseline noise to overcome silent network problem
        noise = torch.randn(self.relay_size, device=self.device) * self._baseline_noise
        relay_excitation = relay_excitation + noise

        # =====================================================================
        # TRN→RELAY INHIBITION (now EXTERNAL via population routing)
        # =====================================================================
        # TRN inhibition now arrives via external AxonalProjection
        # with proper 1ms axonal delay (fast local GABAergic connections)
        # Input via "trn_inhibition" population

        trn_inhibition_input = region_inputs.get("trn_inhibition", None)
        if trn_inhibition_input is not None:
            relay_inhibition = trn_inhibition_input.float()  # Already weighted and delayed
        else:
            relay_inhibition = torch.zeros(self.relay_size, device=self.device, dtype=torch.float32)

        # Update relay neurons (ADR-005: 1D tensors, no batch)
        relay_spikes, relay_membrane = self.relay_neurons(
            g_exc_input=relay_excitation,  # [relay_size]
            g_inh_input=relay_inhibition,  # [relay_size]
        )

        # =====================================================================
        # HOMEOSTATIC INTRINSIC PLASTICITY: Adaptive Gain Control
        # =====================================================================
        # Relay neurons adapt their gains to maintain target firing rate
        # This solves the cold-start problem: neurons increase gain when underactive
        # Biologically accurate: thalamic relay neurons exhibit homeostatic regulation
        # Compute gain update: increase gain when underactive, decrease when overactive
        rate_error = self._target_rate - self.relay_firing_rate  # [relay_size]
        gain_update = self._gain_lr * rate_error  # [relay_size]

        # Apply update with minimum floor to prevent negative gains
        # Negative gains would invert excitation (making excitatory input inhibitory)
        self.relay_gain.data.add_(gain_update).clamp_(min=GAIN_MINIMUM_THALAMUS, max=GAIN_MAXIMUM_THALAMUS)

        # Adaptive threshold: lower threshold when underactive
        self.relay_neurons.v_threshold.data = torch.clamp(
            self.relay_neurons.v_threshold - self._threshold_lr * rate_error,
            min=self._threshold_min,
            max=self._threshold_max,
        )

        # =====================================================================
        # MODE SWITCHING: Burst vs Tonic
        # =====================================================================
        current_mode = self._determine_mode(relay_membrane)  # [relay_size]

        # In burst mode, amplify spikes (convert bool to float temporarily)
        burst_mask = current_mode < THALAMUS_MODE_THRESHOLD  # Burst mode, [relay_size]
        burst_amplified = relay_spikes.float()  # [relay_size]

        if burst_mask.any():
            # Amplify burst spikes
            burst_amplified = torch.where(
                burst_mask, burst_amplified * self.config.burst_gain, burst_amplified
            )

        # Binarize and convert to bool (ADR-004)
        relay_output = burst_amplified > THALAMUS_MODE_THRESHOLD  # [relay_size], bool

        # =====================================================================
        # TRN NEURONS: Synaptic currents → TRN
        # =====================================================================
        # TRN excitation was already accumulated from region_inputs above:
        # - Sensory collateral (via input_to_trn)
        # - L6a feedback (via cortex:l6a routing)
        #
        # Now add relay collateral excitation
        trn_excitation_relay = torch.mv(self.relay_to_trn, relay_output.float())  # [trn_size]
        trn_excitation = trn_current + trn_excitation_relay

        # TRN recurrent inhibition
        trn_inhibition = torch.mv(self.trn_recurrent, self.trn_spikes.float())  # [trn_size]

        # Gap junction coupling (TRN synchronization)
        # Ultra-fast electrical coupling (<0.1ms) for coherent inhibitory volleys
        trn_gap_current = torch.zeros(self.trn_size, device=self.device)
        if self.gap_junctions is not None and self.trn_neurons.membrane is not None:
            # Apply voltage coupling based on previous timestep's membrane potentials
            trn_gap_current = self.gap_junctions(self.trn_neurons.membrane)
            # Gap junction current is excitatory (depolarizing current from neighbors)
            # Add to excitation rather than inhibition
            trn_excitation = trn_excitation + trn_gap_current

        # Update TRN neurons (ADR-005: 1D tensors)
        trn_spikes, _trn_membrane = self.trn_neurons(
            g_exc_input=trn_excitation,  # [trn_size] (includes gap junction coupling)
            g_inh_input=trn_inhibition,  # [trn_size]
        )
        self.trn_spikes = trn_spikes  # [trn_size], bool

        # =====================================================================
        # HOMEOSTATIC FEEDBACK: Update firing rate based on OUTPUT
        # =====================================================================
        # CRITICAL: Track OUTPUT spikes (what cortex receives), not internal relay spikes
        # This ensures gains respond to actual transmission, not internally generated
        # activity that gets blocked by TRN or burst/tonic modulation
        output_rate = relay_output.float()  # [relay_size], actual output (0 or 1)
        self.relay_firing_rate.mul_(1.0 - self._firing_rate_alpha).add_(output_rate * self._firing_rate_alpha)

        # =====================================================================
        # THALAMOCORTICAL SYNAPTIC PLASTICITY
        # =====================================================================
        # Apply STDP learning to relay synapses
        # Thalamocortical plasticity is critical for:
        # - Sensory learning (what stimuli are relevant)
        # - Attentional routing (which inputs to amplify)
        # - Adaptive filtering (noise suppression, signal enhancement)
        #
        # Biology: Both ascending (sensory→relay) and descending (L6→relay)
        # pathways show robust STDP that shapes sensory representations

        if self.config.learning_rate > 0 and self.learning_strategy is not None:
            # Learn sensory→relay pathway (ascending plasticity)
            # This allows thalamus to learn which sensory features are informative
            sensory_input = region_inputs.get("sensory")
            if sensory_input is not None and "sensory" in self.synaptic_weights:
                updated_weights, _ = self.learning_strategy.compute_update(
                    weights=self.synaptic_weights["sensory"].data,
                    pre_spikes=sensory_input,
                    post_spikes=relay_output,
                    learning_rate=self.config.learning_rate,
                )
                self.synaptic_weights["sensory"].data.copy_(updated_weights)
                clamp_weights(
                    self.synaptic_weights["sensory"].data,
                    self.config.w_min,
                    self.config.w_max,
                )

            # Learn L6b→relay pathway (corticothalamic precision modulation)
            # This implements top-down attention: cortex learns to modulate relay gain
            l6b_feedback = region_inputs.get("cortex:l6b")
            if l6b_feedback is not None and "cortex:l6b" in self.synaptic_weights:
                updated_weights, _ = self.learning_strategy.compute_update(
                    weights=self.synaptic_weights["cortex:l6b"].data,
                    pre_spikes=l6b_feedback,
                    post_spikes=relay_output,
                    learning_rate=self.config.learning_rate * 0.5,  # Slower for feedback
                )
                self.synaptic_weights["cortex:l6b"].data.copy_(updated_weights)
                clamp_weights(
                    self.synaptic_weights["cortex:l6b"].data,
                    self.config.w_min,
                    self.config.w_max,
                )

            # Learn L6a→TRN pathway (corticothalamic attention control)
            # This allows cortex to learn attentional gating patterns
            l6a_feedback = region_inputs.get("cortex:l6a")
            if l6a_feedback is not None and "cortex:l6a" in self.synaptic_weights:
                updated_weights, _ = self.learning_strategy.compute_update(
                    weights=self.synaptic_weights["cortex:l6a"].data,
                    pre_spikes=l6a_feedback,
                    post_spikes=trn_spikes,  # TRN is postsynaptic target
                    learning_rate=self.config.learning_rate * 0.3,  # Even slower for TRN
                )
                self.synaptic_weights["cortex:l6a"].data.copy_(updated_weights)
                clamp_weights(
                    self.synaptic_weights["cortex:l6a"].data,
                    self.config.w_min,
                    self.config.w_max,
                )

        region_outputs: RegionSpikesDict = {
            "relay": relay_output,
            "trn": trn_spikes,
        }

        return self._post_forward(region_outputs)

    def _determine_mode(self, membrane: torch.Tensor) -> torch.Tensor:
        """Determine burst vs tonic mode based on membrane potential.

        Args:
            membrane: Current membrane potential [relay_size] (1D, ADR-005)

        Returns:
            Mode indicator [relay_size]: 0=burst, 1=tonic (1D, ADR-005)
        """
        # Burst mode: Hyperpolarized (membrane < burst_threshold)
        # Tonic mode: Depolarized (membrane > tonic_threshold)
        # Between: Maintain previous mode

        # Update mode based on thresholds
        burst_mask = membrane < self.config.burst_threshold
        tonic_mask = membrane > self.config.tonic_threshold

        self.current_mode = torch.where(
            burst_mask,
            torch.zeros_like(membrane),  # Burst mode
            torch.where(
                tonic_mask,
                torch.ones_like(membrane),  # Tonic mode
                self.current_mode,  # Maintain
            ),
        )

        return self.current_mode

    # =========================================================================
    # TEMPORAL PARAMETER MANAGEMENT
    # =========================================================================

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters when brain timestep changes.

        Propagates dt update to neurons and STP components.

        Args:
            dt_ms: New simulation timestep in milliseconds
        """
        super().update_temporal_parameters(dt_ms)

        # Update neurons
        if hasattr(self, "relay_neurons") and self.relay_neurons is not None:
            self.relay_neurons.update_temporal_parameters(dt_ms)
        if hasattr(self, "trn_neurons") and self.trn_neurons is not None:
            self.trn_neurons.update_temporal_parameters(dt_ms)

        # Update STP components
        if hasattr(self, "stp_sensory_relay") and self.stp_sensory_relay is not None:
            self.stp_sensory_relay.update_temporal_parameters(dt_ms)
        if hasattr(self, "stp_l6_feedback") and self.stp_l6_feedback is not None:
            self.stp_l6_feedback.update_temporal_parameters(dt_ms)

        # Update homeostatic plasticity time constant
        self._firing_rate_alpha = 1.0 - torch.exp(
            torch.tensor(-dt_ms / self.config.gain_tau_ms, device=self.device)
        )

    def _compute_alpha_gate(self) -> torch.Tensor:
        """Compute attentional gating from alpha oscillation.

        Alpha oscillations (8-13 Hz) suppress unattended inputs:
        - Alpha peak (phase=π): Strong suppression
        - Alpha trough (phase=0): Weak suppression

        Returns:
            Gating factor [relay_size] (1D, ADR-005)
        """
        # Alpha gate: strong at trough (phase=0), weak at peak (phase=π)
        # gate = 1 - strength × (1 + cos(phase)) / 2
        # This gives: phase=0 → gate=1-strength, phase=π → gate=1.0

        # Normalize (1 + cos(phase)) from [0, 2] to [0, 1]
        alpha_modulation = 0.5 * (1.0 + math.cos(self._alpha_phase))
        gate = 1.0 - self.config.alpha_suppression_strength * alpha_modulation

        # Broadcast to all neurons (ADR-005: 1D)
        gate_tensor = torch.full(
            (self.relay_size,),
            gate,
            device=self.device,
            dtype=torch.float32,
        )

        return gate_tensor

    # =========================================================================
    # DIAGNOSTICS
    # =========================================================================

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic information for this region."""
        # Compute plasticity metrics from relay gain (primary modifiable weights)
        plasticity = compute_plasticity_metrics(
            weights=self.relay_gain.data,
            learning_rate=self.config.learning_rate,
        )

        # Add learned pathway statistics
        if "sensory" in self.synaptic_weights:
            plasticity["sensory_relay_mean"] = float(self.synaptic_weights["sensory"].data.mean().item())
        if "cortex:l6b" in self.synaptic_weights:
            plasticity["l6b_relay_mean"] = float(self.synaptic_weights["cortex:l6b"].data.mean().item())
        if "cortex:l6a" in self.synaptic_weights:
            plasticity["l6a_trn_mean"] = float(self.synaptic_weights["cortex:l6a"].data.mean().item())

        return {
            "plasticity": plasticity,
            "architecture": {
                "relay_size": self.relay_size,
                "trn_size": self.trn_size,
            },
        }
