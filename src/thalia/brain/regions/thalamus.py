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

from typing import Any, ClassVar, Dict, List, Optional

import torch

from thalia.brain.configs import ThalamusConfig
from thalia.components import (
    GapJunctionConfig,
    GapJunctionCoupling,
    NeuromodulatorReceptor,
    NeuronFactory,
    STPConfig,
    STPType,
    WeightInitializer,
)
from thalia.learning import (
    STDPStrategy,
    STDPConfig,
)
from thalia.typing import (
    ConductanceTensor,
    GapJunctionReversal,
    NeuromodulatorInput,
    PopulationPolarity,
    PopulationSizes,
    ReceptorType,
    RegionName,
    RegionOutput,
    SynapseId,
    SynapticInput,
)
from thalia.utils import (
    CircularDelayBuffer,
    compute_ach_recurrent_suppression,
    split_excitatory_conductance,
)

from .neural_region import NeuralRegion
from .population_names import ThalamusPopulation
from .region_registry import register_region


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
    """

    # ACh from nucleus basalis gates thalamic relay gain (nicotinic receptors on relay neurons).
    # Modulates burst vs. tonic mode threshold and alpha oscillation depth.
    neuromodulator_subscriptions: ClassVar[List[str]] = ['ach']

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, config: ThalamusConfig, population_sizes: PopulationSizes, region_name: RegionName):
        """Initialize thalamic relay."""
        super().__init__(config, population_sizes, region_name)

        # =====================================================================
        # EXTRACT LAYER SIZES
        # =====================================================================
        self.relay_size = population_sizes[ThalamusPopulation.RELAY]
        self.trn_size = population_sizes[ThalamusPopulation.TRN]

        # =====================================================================
        # NEURONS
        # =====================================================================
        # Relay neurons (Excitatory, glutamatergic)
        self.relay_neurons = NeuronFactory.create_relay_neurons(
            region_name=self.region_name,
            population_name=ThalamusPopulation.RELAY,
            n_neurons=self.relay_size,
            device=self.device,
        )

        # TRN neurons (Inhibitory, GABAergic)
        self.trn_neurons = NeuronFactory.create_trn_neurons(
            region_name=self.region_name,
            population_name=ThalamusPopulation.TRN,
            n_neurons=self.trn_size,
            device=self.device,
        )

        # =====================================================================
        # INITIALIZE STATE VARIABLES
        # =====================================================================
        # Mode state (0=burst, 1=tonic)
        # Initialize to tonic
        self.current_mode: torch.Tensor
        self.register_buffer("current_mode", torch.ones(self.relay_size, device=self.device))

        # Homeostatic plasticity: track firing rates for gain adaptation
        # Neuron registered below; looked up lazily in _update_homeostasis().
        self._register_homeostasis(ThalamusPopulation.RELAY, self.relay_size)

        # =====================================================================
        # SYNAPTIC WEIGHTS
        # =====================================================================
        # Relay → TRN (collateral activation)
        # CONDUCTANCE-BASED: Strong conductance to drive TRN firing
        self._add_internal_connection(
            source_population=ThalamusPopulation.RELAY,
            target_population=ThalamusPopulation.TRN,
            weights=WeightInitializer.sparse_random(
                n_input=self.relay_size,
                n_output=self.trn_size,
                connectivity=0.2,
                weight_scale=0.005,
                device=self.device,
            ),
            stp_config=None,
            receptor_type=ReceptorType.AMPA,
        )

        # =====================================================================
        # TRN RECURRENT INHIBITION
        # =====================================================================
        # TRN → TRN (recurrent inhibition for oscillations)
        self._add_internal_connection(
            source_population=ThalamusPopulation.TRN,
            target_population=ThalamusPopulation.TRN,
            weights=WeightInitializer.sparse_random_no_autapses(
                n_input=self.trn_size,
                n_output=self.trn_size,
                connectivity=0.01,
                weight_scale=0.0003,
                device=self.device,
            ),
            stp_config=None,
            receptor_type=ReceptorType.GABA_A,
        )

        # TRN recurrent delay buffer (prevents instant feedback oscillations)
        trn_recurrent_delay_steps = int(config.trn_recurrent_delay_ms / config.dt_ms)
        self._trn_recurrent_buffer = CircularDelayBuffer(
            max_delay=trn_recurrent_delay_steps,
            size=self.trn_size,
            device=str(self.device),
            dtype=torch.bool,
        )

        # Relay→TRN delay buffer (1-step; relay axon collaterals reach TRN with ~1ms delay)
        # Biology: relay→TRN is a collateral of the thalamocortical axon, not zero-latency
        self._relay_spike_buffer = CircularDelayBuffer(
            max_delay=1,
            size=self.relay_size,
            device=str(self.device),
            dtype=torch.bool,
        )

        # =====================================================================
        # ACETYLCHOLINE RECEPTOR (NB projection for sleep/wake modulation)
        # =====================================================================
        # Thalamus receives ACh from nucleus basalis (cortical arousal)
        # and brainstem cholinergic nuclei (ascending arousal system)
        # ACh controls thalamic oscillations and sensory gating:
        # - High ACh (wakefulness): Suppress TRN oscillations → tonic mode (relay)
        # - Low ACh (sleep): Enable TRN spindles → burst mode (disconnection)
        self.ach_receptor = NeuromodulatorReceptor(
            n_receptors=self.trn_size,
            tau_rise_ms=10.0,  # Moderate rise
            tau_decay_ms=100.0,  # Slower decay for sustained state control
            spike_amplitude=0.15,
            device=self.device,
        )
        # ACh concentration state (updated each timestep from NB/brainstem spikes)
        self._ach_concentration_trn = torch.zeros(self.trn_size, device=self.device)

        # =====================================================================
        # STDP LEARNING STRATEGY
        # =====================================================================
        # Thalamocortical synapses show robust STDP
        # Critical for sensory learning, attention, and routing
        # Both ascending (sensory→relay) and descending (L6→relay) pathways learn
        self._stdp_strategy = STDPStrategy(STDPConfig(
            learning_rate=config.learning_rate,
            a_plus=0.005,  # Moderate LTP (thalamic synapses are conservative)
            a_minus=0.001,  # Weak LTD (5:1 LTP:LTD ratio)
            tau_plus=20.0,  # Standard STDP window
            tau_minus=20.0,
            w_min=config.w_min,
            w_max=config.w_max,
            device=str(self.device),
        ))

        # =====================================================================
        # GAP JUNCTIONS (TRN interneuron synchronization)
        # =====================================================================
        # TRN neurons are densely coupled via gap junctions
        # This enables ultra-fast synchronization for coherent inhibitory volleys
        # Gap junctions will be created in finalize_initialization() after all input
        # sources have been added
        #
        # Physics: I_gap = g × (E_neighbor_avg - V), where E_neighbor_avg is
        # the weighted average of neighbor voltages (dynamic reversal potential).
        self.gap_junctions: Optional[GapJunctionCoupling] = None

        # =====================================================================
        # REGISTER NEURON POPULATIONS
        # =====================================================================
        self._register_neuron_population(ThalamusPopulation.RELAY, self.relay_neurons, polarity=PopulationPolarity.EXCITATORY)
        self._register_neuron_population(ThalamusPopulation.TRN, self.trn_neurons, polarity=PopulationPolarity.INHIBITORY)

        # Ensure all tensors are on the correct device
        self.to(self.device)

    def finalize_initialization(self) -> None:
        """Finalize initialization after all input sources have been added.

        This method should be called by brain builder after all connections
        have been established. It builds gap junctions and internal recurrent
        connections.

        Must be called after all add_input_source() calls complete.
        """
        # Create gap junctions using TRN→TRN recurrent weights as proximity proxy.
        # We use the TRN-internal recurrent synapse weights to estimate which
        # TRN neurons are functionally adjacent (share inhibitory connectivity).
        # This replaces the old lazy input_to_trn approach which bypassed SynapseId.
        if self.gap_junctions is None:
            trn_recurrent_synapse = SynapseId(
                source_region=self.region_name,
                source_population=ThalamusPopulation.TRN,
                target_region=self.region_name,
                target_population=ThalamusPopulation.TRN,
                receptor_type=ReceptorType.GABA_A,
            )
            trn_trn_weights = self.get_synaptic_weights(trn_recurrent_synapse)
            self.gap_junctions = GapJunctionCoupling(
                n_neurons=self.trn_size,
                afferent_weights=trn_trn_weights,
                config=GapJunctionConfig(
                    coupling_strength=0.1,
                    connectivity_threshold=0.3,
                    max_neighbors=6,
                    interneuron_only=True,  # All TRN neurons are inhibitory
                ),
                interneuron_mask=None,  # All TRN neurons are interneurons
                device=self.device,
            )

        # Create internal TRN→relay synaptic weights (local recurrent connection)
        # Biology: TRN provides feedforward and lateral inhibition to relay neurons
        # Distance: ~50-200μm (local), unmyelinated → 1ms delay (handled in forward pass)
        # Implements searchlight attention mechanism
        # Goal: Hyperpolarize relay below V_half_h_T (-0.3) → h_T de-inactivates → rebound burst on release
        # Need balance: enough inhibition for hyperpolarization, not so much it clamps neurons
        self._add_internal_connection(
            source_population=ThalamusPopulation.TRN,
            target_population=ThalamusPopulation.RELAY,
            weights=WeightInitializer.sparse_gaussian(
                n_input=self.trn_size,
                n_output=self.relay_size,
                connectivity=0.6,
                mean=0.003,
                std=0.001,
                device=self.device,
            ),
            stp_config=None,
            receptor_type=ReceptorType.GABA_A,
        )

    # =========================================================================
    # SYNAPTIC WEIGHT MANAGEMENT
    # =========================================================================

    def add_input_source(
        self,
        synapse_id: SynapseId,
        n_input: int,
        connectivity: float,
        weight_scale: float,
        *,
        stp_config: Optional[STPConfig] = None,
    ) -> None:
        """Add input source and TRN projections.

        Overrides base to:
        1. Create routing keys for relay/TRN targeting
        2. Grow STP modules for relay neurons
        3. For sensory inputs: add a separate sensory→TRN SynapseId for feedforward inhibition
        """

        is_sensory_input = synapse_id.is_external_sensory_input()

        if stp_config is None:
            if synapse_id.target_population == ThalamusPopulation.RELAY:
                if is_sensory_input:
                    # Sensory input → relay depression
                    # Filters repetitive stimuli, responds to novelty
                    # CRITICAL for attention capture and change detection
                    stp_config = STPConfig.from_type(STPType.DEPRESSING_MODERATE)

            elif synapse_id.target_population == ThalamusPopulation.TRN:
                # All inputs → TRN facilitation (including sensory collateral)
                # Ensures TRN can be recruited by weak inputs for effective gating
                stp_config = STPConfig.from_type(STPType.FACILITATING_STRONG)

        super().add_input_source(synapse_id, n_input, connectivity, weight_scale, stp_config=stp_config)

        # === Handle sensory-specific projection ===
        # Sensory input needs special handling for center-surround filtering
        if is_sensory_input:
            sensory_trn_synapse = SynapseId(
                source_region=synapse_id.source_region,
                source_population=synapse_id.source_population,
                target_region=self.region_name,
                target_population=ThalamusPopulation.TRN,
                receptor_type=ReceptorType.AMPA,
            )
            if sensory_trn_synapse not in self.synaptic_weights:
                self._add_synaptic_weights(
                    sensory_trn_synapse,
                    WeightInitializer.sparse_random(
                        n_input=n_input,
                        n_output=self.trn_size,
                        connectivity=0.3,
                        weight_scale=weight_scale,  # Match sensory→relay scale
                        device=self.device,
                    )
                )

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    @torch.no_grad()
    def forward(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Process sensory input through thalamic relay.

        Args:
            synaptic_inputs: Point-to-point synaptic connections from sensory inputs and cortex
            neuromodulator_inputs: Broadcast neuromodulatory signals (DA, NE, ACh)
        """
        self._pre_forward(synaptic_inputs, neuromodulator_inputs)

        cfg = self.config

        # =====================================================================
        # ACETYLCHOLINE RECEPTOR PROCESSING (from NB)
        # =====================================================================
        # Process NB acetylcholine spikes → sleep/wake mode control
        # High ACh → suppress TRN oscillations (wakefulness, sensory relay)
        # Low ACh → enable TRN spindles (sleep, sensory disconnection)
        nb_ach_spikes = neuromodulator_inputs.get('ach', None)
        if nb_ach_spikes is not None:
            self._ach_concentration_trn = self.ach_receptor.update(nb_ach_spikes)
        else:
            self._ach_concentration_trn = self.ach_receptor.update(None)

        # =====================================================================
        # MULTI-SOURCE SYNAPTIC INTEGRATION
        # =====================================================================
        # Accumulate synaptic conductances from all registered external sources
        # Internal connections (TRN→relay, relay→TRN, TRN→TRN) are integrated below
        # using _integrate_synaptic_inputs_at_dendrites with their respective delay buffers.
        relay_conductance = torch.zeros(self.relay_size, device=self.device)
        trn_conductance = torch.zeros(self.trn_size, device=self.device)

        for synapse_id, source_spikes in synaptic_inputs.items():
            # Convert to float for matrix multiplication
            source_spikes_float = source_spikes.float()

            # Route to appropriate target neurons
            if synapse_id.target_population == ThalamusPopulation.TRN:
                trn_conductance += self.get_synaptic_weights(synapse_id) @ source_spikes_float
            elif synapse_id.target_population == ThalamusPopulation.RELAY:
                relay_conductance += self.get_synaptic_weights(synapse_id) @ source_spikes_float

            # Sensory inputs also project to TRN for feedforward inhibition (gating mechanism)
            if synapse_id.is_external_sensory_input():
                assert synapse_id.target_population != ThalamusPopulation.TRN, (
                    f"External sensory input should not directly target TRN population. "
                    f"Found target_population={synapse_id.target_population} for synapse_id={synapse_id}"
                )

                # Sensory collateral → TRN (for feedforward inhibition)
                sensory_trn_synapse = SynapseId(
                    source_region=synapse_id.source_region,
                    source_population=synapse_id.source_population,
                    target_region=self.region_name,
                    target_population=ThalamusPopulation.TRN,
                    receptor_type=ReceptorType.AMPA,
                )
                trn_conductance += self.get_synaptic_weights(sensory_trn_synapse) @ source_spikes_float

        # =====================================================================
        # INTERNAL TRN→RELAY INHIBITION (1-step delayed, via _trn_recurrent_buffer)
        # =====================================================================
        # TRN spikes from the PREVIOUS timestep inhibit relay neurons.
        # _trn_recurrent_buffer already holds TRN spikes; reading at lag=1 reuses
        # it for both TRN→relay and TRN→TRN recurrent paths without a separate tensor.
        trn_relay_synapse = SynapseId(
            source_region=self.region_name,
            source_population=ThalamusPopulation.TRN,
            target_region=self.region_name,
            target_population=ThalamusPopulation.RELAY,
            receptor_type=ReceptorType.GABA_A,
        )
        relay_inhibition = self._integrate_synaptic_inputs_at_dendrites(
            {trn_relay_synapse: self._trn_recurrent_buffer.read(1)},
            n_neurons=self.relay_size,
        ).g_gaba_a

        # =====================================================================
        # RELAY NEURONS: Conductances → Relay
        # =====================================================================
        relay_excitation = relay_conductance

        # =====================================================================
        # HOMEOSTATIC INTRINSIC PLASTICITY: Add Minimal Noise
        # =====================================================================
        # Add tiny baseline noise to overcome silent network problem
        # Reduced from 0.01 to 0.001 for conductance-based model
        noise = torch.randn(self.relay_size, device=self.device) * 0.001
        relay_excitation = relay_excitation + noise

        # =====================================================================
        # UPDATE RELAY NEURONS
        # =====================================================================
        # Update relay neurons (ADR-005: 1D tensors, no batch)
        # relay_excitation and relay_inhibition are conductances (weights @ spikes)
        # Split excitatory conductance into AMPA (fast) and NMDA (slow)
        relay_g_ampa, relay_g_nmda = split_excitatory_conductance(relay_excitation, nmda_ratio=0.2)

        relay_spikes, relay_membrane = self.relay_neurons.forward(
            g_ampa_input=ConductanceTensor(relay_g_ampa),  # [relay_size]
            g_nmda_input=ConductanceTensor(relay_g_nmda),  # [relay_size]
            g_gaba_a_input=ConductanceTensor(relay_inhibition),  # [relay_size]
            g_gaba_b_input=None,
        )

        # =====================================================================
        # HOMEOSTATIC INTRINSIC PLASTICITY: Modulate Leak Conductance
        # =====================================================================
        self._update_homeostasis(ThalamusPopulation.RELAY, relay_spikes)

        # =====================================================================
        # MODE SWITCHING: Burst vs Tonic
        # =====================================================================
        current_mode = self._determine_mode(relay_membrane)  # [relay_size]

        # In burst mode, amplify spikes (convert bool to float temporarily)
        burst_mask = current_mode < cfg.mode_threshold  # Burst mode, [relay_size]
        burst_amplified = relay_spikes.float()  # [relay_size]

        if burst_mask.any():
            # Amplify burst spikes
            burst_amplified = torch.where(burst_mask, burst_amplified * cfg.burst_gain, burst_amplified)

        # Binarize and convert to bool (ADR-004)
        relay_output = burst_amplified > cfg.mode_threshold  # [relay_size], bool

        # =====================================================================
        # TRN NEURONS: Synaptic conductances → TRN
        # =====================================================================
        # TRN excitation was already accumulated from synaptic_inputs above:
        # - Sensory collateral (via separate sensory→TRN SynapseId registered in add_input_source)
        # - L6a feedback (via cortex:l6a→thalamus:trn routing)

        # Now add relay collateral excitation (1-step delayed via _relay_spike_buffer)
        # Biology: relay→TRN is a collateral of the thalamocortical axon; ~1ms synaptic delay.
        # Using the previous step's relay output preserves causality within the timestep.
        relay_trn_synapse = SynapseId(
            source_region=self.region_name,
            source_population=ThalamusPopulation.RELAY,
            target_region=self.region_name,
            target_population=ThalamusPopulation.TRN,
            receptor_type=ReceptorType.AMPA,
        )
        trn_excitation = trn_conductance + self._integrate_synaptic_inputs_at_dendrites(
            {relay_trn_synapse: self._relay_spike_buffer.read(1)},
            n_neurons=self.trn_size,
        ).g_ampa

        # TRN recurrent inhibition with configurable delay (prevents instant feedback oscillations)
        # ACh modulation of TRN recurrent inhibition (McCormick & Prince 1986):
        # High ACh (wakefulness): Suppress TRN oscillations → tonic relay mode
        # Low ACh (sleep): Enable TRN spindles → burst mode, sensory disconnection
        ach_level = self._ach_concentration_trn.mean().item()
        ach_recurrent_modulation = compute_ach_recurrent_suppression(ach_level)

        trn_recurrent_synapse = SynapseId(
            source_region=self.region_name,
            source_population=ThalamusPopulation.TRN,
            target_region=self.region_name,
            target_population=ThalamusPopulation.TRN,
            receptor_type=ReceptorType.GABA_A,
        )
        trn_inhibition = self._integrate_synaptic_inputs_at_dendrites(
            {trn_recurrent_synapse: self._trn_recurrent_buffer.read(self._trn_recurrent_buffer.max_delay)},
            n_neurons=self.trn_size,
        ).g_gaba_a * ach_recurrent_modulation

        # Gap junction coupling (TRN synchronization)
        # Ultra-fast electrical coupling (<0.1ms) for coherent inhibitory volleys
        # FIXED: Gap junctions now return (conductance, dynamic_reversal) instead of current
        # This correctly models I_gap = g × (E_neighbor_avg - V) in conductance-based framework
        trn_gap_conductance = torch.zeros(self.trn_size, device=self.device)
        trn_gap_reversal = torch.zeros(self.trn_size, device=self.device)
        if self.gap_junctions is not None and self.trn_neurons.membrane is not None:
            # Gap junctions return (g_total, E_effective) where:
            # - g_total: Sum of gap junction conductances for each neuron
            # - E_effective: Weighted average of neighbor voltages (dynamic reversal)
            # Physics: I_gap = g × (E_eff - V), where E_eff = weighted_avg(neighbor_voltages)
            trn_gap_conductance, trn_gap_reversal = self.gap_junctions.forward(self.trn_neurons.membrane)

        # Split excitatory conductance into AMPA (fast) and NMDA (slow)
        trn_g_ampa, trn_g_nmda = split_excitatory_conductance(trn_excitation, nmda_ratio=0.2)

        # Pass gap junction conductances EXPLICITLY as forward() parameters
        # This is cleaner than monkey-patching _get_additional_conductances()
        apply_gap = trn_gap_conductance.abs().sum() > 1e-6  # Only apply if conductance is non-negligible
        trn_spikes, _trn_membrane = self.trn_neurons.forward(
            g_ampa_input=ConductanceTensor(trn_g_ampa),  # [trn_size]
            g_nmda_input=ConductanceTensor(trn_g_nmda),  # [trn_size]
            g_gaba_a_input=ConductanceTensor(trn_inhibition),  # [trn_size]
            g_gaba_b_input=None,
            g_gap_input=ConductanceTensor(trn_gap_conductance) if apply_gap else None,
            E_gap_reversal=GapJunctionReversal(trn_gap_reversal) if apply_gap else None,
        )

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

        region_outputs: RegionOutput = {
            ThalamusPopulation.RELAY: relay_output,
            ThalamusPopulation.TRN: trn_spikes,
        }

        # Ensure strategies are registered before dispatch
        for synapse_id in list(synaptic_inputs.keys()):
            if self.get_learning_strategy(synapse_id) is None:
                self._add_learning_strategy(synapse_id, self._stdp_strategy)

        # Write current outputs to delay buffers for next timestep (causality)
        self._relay_spike_buffer.write_and_advance(relay_output)
        self._trn_recurrent_buffer.write_and_advance(trn_spikes)

        return self._post_forward(region_outputs)

    def _get_learning_kwargs(self, synapse_id: SynapseId) -> Dict[str, Any]:
        # RELAY: slower for L6b feedback; TRN: even slower for L6a
        if synapse_id.target_population == ThalamusPopulation.TRN:
            return {"learning_rate": self.config.learning_rate * 0.3}
        return {"learning_rate": self.config.learning_rate * 0.5}

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
        self.relay_neurons.update_temporal_parameters(dt_ms)
        self.trn_neurons.update_temporal_parameters(dt_ms)
