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

from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Union

import torch

from thalia import GlobalConfig
from thalia.brain.configs import ThalamusConfig
from thalia.brain.gap_junctions import (
    GapJunctionConfig,
    GapJunctionCoupling,
)
from thalia.brain.neurons import (
    ConductanceLIFConfig,
    heterogeneous_adapt_increment,
    heterogeneous_dendrite_coupling,
    heterogeneous_g_L,
    heterogeneous_noise_std,
    heterogeneous_tau_adapt,
    heterogeneous_tau_mem,
    heterogeneous_v_threshold,
    split_excitatory_conductance,
)
from thalia.brain.synapses import (
    ConductanceScaledSpec,
    STPConfig,
    WeightInitializer,
)
from thalia.learning import (
    InhibitorySTDPConfig,
    InhibitorySTDPStrategy,
    LearningStrategy,
    STDPStrategy,
    STDPConfig,
)
from thalia.typing import (
    ConductanceTensor,
    GapJunctionReversal,
    NeuromodulatorChannel,
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
)

from .neural_region import NeuralRegion
from .population_names import ThalamusPopulation
from .region_registry import register_region

if TYPE_CHECKING:
    from thalia.brain.neurons import ConductanceLIF


@register_region(
    "thalamus",
    aliases=["thalamic_relay"],
    description="Sensory relay and gating with burst/tonic modes and attentional modulation",
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
    neuromodulator_subscriptions: ClassVar[List[NeuromodulatorChannel]] = [
        NeuromodulatorChannel.ACH,
        NeuromodulatorChannel.DA_MESOCORTICAL,
        NeuromodulatorChannel.NE,
    ]

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        config: ThalamusConfig,
        population_sizes: PopulationSizes,
        region_name: RegionName,
        *,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ):
        """Initialize thalamic relay."""
        super().__init__(config, population_sizes, region_name, device=device)

        # =====================================================================
        # EXTRACT LAYER SIZES
        # =====================================================================
        self.relay_size = population_sizes[ThalamusPopulation.RELAY]
        self.trn_size = population_sizes[ThalamusPopulation.TRN]

        # =====================================================================
        # NEURONS
        # =====================================================================
        # Relay neurons (Excitatory, glutamatergic)
        self.relay_neurons: ConductanceLIF
        self.relay_neurons = self._create_and_register_neuron_population(
            population_name=ThalamusPopulation.RELAY,
            n_neurons=self.relay_size,
            polarity=PopulationPolarity.EXCITATORY,
            config=ConductanceLIFConfig(
                tau_mem_ms=heterogeneous_tau_mem(18.0, self.relay_size, device),
                v_reset=0.0,
                v_threshold=heterogeneous_v_threshold(0.8, self.relay_size, device),
                tau_ref=4.0,
                g_L=heterogeneous_g_L(0.08, self.relay_size, device),
                E_E=3.0,
                E_I=-0.5,
                tau_E=5.0,
                tau_I=10.0,
                tau_nmda=100.0,
                E_nmda=3.0,
                g_nmda_max=0.5,  # Thalamocortical burst mode risks NMDA accumulation
                tau_GABA_B=400.0,
                E_GABA_B=-0.8,
                noise_std=heterogeneous_noise_std(0.08, self.relay_size, device=self.device),
                noise_tau_ms=3.0,
                # Adaptation calibrated for tonic awake mode (10-30 Hz target):
                # Raised 0.08→0.15: at 25 Hz, g_adapt_ss = 0.15×25×0.10 = 0.375
                # (4.7× g_L=0.08) — strong negative feedback to self-limit relay firing.
                # Previous 0.08 was insufficient against excitatory drive saturation.
                tau_adapt_ms=heterogeneous_tau_adapt(100.0, self.relay_size, device),
                adapt_increment=heterogeneous_adapt_increment(0.15, self.relay_size, device),
                enable_t_channels=True,
                g_T=0.001,  # Reduced 0.008→0.003→0.001: 0.003 still caused 36% short-ISI bursting;
                            # at 0.001 the T-channel LTS is too weak to generate burst doublets in awake tonic mode.
                            # T-channels are primarily relevant during sleep/drowsy states (low ACh).
                E_Ca=4.0,
                tau_h_T_ms=50.0,
                V_half_h_T=-0.75,  # Shifted -0.65→-0.75: T-channel deinactivation requires deeper hyperpolarization,
                                    # preventing burst mode during normal tonic firing (V ~ -0.2 to 0.0).
                k_h_T=0.12,
                enable_ih=True,
                g_h_max=0.02,
                E_h=-0.3,
                V_half_h=-0.3,
                k_h=0.10,
                tau_h_ms=100.0,
                dendrite_coupling_scale=heterogeneous_dendrite_coupling(0.2, self.relay_size, device, cv=0.25),
            ),
        )

        # TRN neurons (Inhibitory, GABAergic)
        self.trn_neurons: ConductanceLIF
        self.trn_neurons = self._create_and_register_neuron_population(
            population_name=ThalamusPopulation.TRN,
            n_neurons=self.trn_size,
            polarity=PopulationPolarity.INHIBITORY,
            config=ConductanceLIFConfig(
                tau_mem_ms=heterogeneous_tau_mem(12.0, self.trn_size, device, cv=0.10),
                v_reset=0.0,
                v_threshold=heterogeneous_v_threshold(1.35, self.trn_size, device, cv=0.06),
                tau_ref=5.0,
                g_L=heterogeneous_g_L(0.10, self.trn_size, device, cv=0.08),
                E_E=3.0,
                E_I=-0.5,
                tau_E=4.0,
                tau_I=6.0,
                tau_nmda=100.0,
                E_nmda=3.0,
                tau_GABA_B=400.0,
                E_GABA_B=-0.8,
                noise_std=heterogeneous_noise_std(0.08, self.trn_size, device=self.device),
                noise_tau_ms=3.0,
                tau_adapt_ms=heterogeneous_tau_adapt(80.0, self.trn_size, device),
                adapt_increment=heterogeneous_adapt_increment(0.25, self.trn_size, device),
                enable_t_channels=True,
                g_T=0.08,
                E_Ca=4.0,
                tau_h_T_ms=50.0,
                V_half_h_T=-0.3,
                k_h_T=0.15,
                dendrite_coupling_scale=heterogeneous_dendrite_coupling(0.2, self.trn_size, device, cv=0.20),
            ),
        )

        # =====================================================================
        # INITIALIZE STATE VARIABLES
        # =====================================================================
        # Brainstem ascending arousal: tonic per-step AMPA conductance for relay neurons.
        # Represents LC-NE, PPN-ACh, and raphe-5HT excitatory drive to thalamus during wakefulness.
        self._relay_baseline = torch.full(
            (self.relay_size,), config.relay_baseline_drive, device=device
        )

        # Brainstem ascending arousal: tonic per-step AMPA conductance for TRN neurons.
        # Biology: TRN receives direct cholinergic (PPN) and noradrenergic (LC) excitation
        # independent of relay collaterals.  Without this, TRN is purely relay-driven and
        # fires too slowly to provide adequate inhibition for physiological E/I ratios.
        self._trn_baseline = torch.full(
            (self.trn_size,), config.trn_baseline_drive, device=device
        )

        # =====================================================================
        # LEARNING STRATEGIES
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
            w_min=config.synaptic_scaling.w_min,
            w_max=config.synaptic_scaling.w_max,
        ))

        # Inhibitory STDP (Vogels et al. 2011) for TRN inhibitory synapses.
        # TRN→TRN lateral inhibition and TRN→relay feedforward gating need
        # homeostatic plasticity to maintain stable E/I balance as excitatory
        # thalamocortical weights change via STDP.
        self._istdp_strategy = InhibitorySTDPStrategy(InhibitorySTDPConfig(
            learning_rate=config.learning_rate * 0.3,  # Conservative: thalamic inhibition is stability-critical
            tau_istdp=20.0,
            alpha=0.12,  # Target ~12% post-synaptic firing rate
            w_min=config.synaptic_scaling.w_min,
            w_max=config.synaptic_scaling.w_max,
        ))

        # =====================================================================
        # SYNAPTIC WEIGHTS
        # =====================================================================
        # Relay → TRN (collateral activation)
        self._relay_trn_synapse = self._add_internal_connection(
            source_population=ThalamusPopulation.RELAY,
            target_population=ThalamusPopulation.TRN,
            weights=WeightInitializer.sparse_random(
                n_input=self.relay_size,
                n_output=self.trn_size,
                connectivity=0.2,
                weight_scale=0.012,
                device=device,
            ),
            receptor_type=ReceptorType.AMPA,
            # Relay→TRN is a facilitating synapse (thalamocortical collaterals).
            # U=0.05: low initial release probability → builds up during sustained
            # relay firing.  tau_f=300ms > tau_d=150ms ensures net facilitation.
            stp_config=STPConfig(U=0.05, tau_d=150.0, tau_f=300.0),
            # STDP: relay collateral plasticity shapes which relay neurons
            # recruit TRN — attention routing via selective excitation.
            learning_strategy=self._stdp_strategy,
        )

        # =====================================================================
        # TRN LATERAL INHIBITION (winner-take-all across sensory streams)
        # =====================================================================
        # TRN → TRN (lateral inhibition — the "searchlight spotlight" mechanism)
        #
        # Biology:
        # Each TRN sector receives from one thalamic nucleus and projects lateral
        # GABA_A inhibition onto TRN neurons representing *other* nuclei (and also
        # locally). When one sensory stream (e.g. visual) strongly drives relay
        # neurons → those relay collaterals excite the visual TRN sector → that
        # TRN sector laterally inhibits the auditory TRN sector → auditory relay
        # is disinhibited proportionally less → winner-take-all across modalities.
        self._trn_trn_gaba_a_synapse = self._add_internal_connection(
            source_population=ThalamusPopulation.TRN,
            target_population=ThalamusPopulation.TRN,
            weights=WeightInitializer.sparse_gaussian_no_autapses(
                n_input=self.trn_size,
                n_output=self.trn_size,
                connectivity=0.15,
                mean=0.012,
                std=0.004,
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=STPConfig(U=0.25, tau_d=300.0, tau_f=50.0),
            # iSTDP: homeostatic tuning of lateral inhibition strength.
            # As excitatory drive changes, TRN lateral inhibition adapts
            # to maintain competitive dynamics across sensory streams.
            learning_strategy=self._istdp_strategy,
        )

        # TRN → TRN: slow GABA_B component
        # TRN→TRN lateral inhibition has a prominent GABA_B component (τ~150-300ms)
        # that shapes spindle refractory period and inter-spindle interval.
        # GABA_B-mediated slow hyperpolarization de-inactivates T-type Ca²⁺ channels
        # in TRN neurons, enabling rebound bursts that sustain spindle oscillations.
        # Weight ~25% of GABA_A (slower kinetics → smaller instantaneous conductance).
        self._trn_trn_gaba_b_synapse = self._add_internal_connection(
            source_population=ThalamusPopulation.TRN,
            target_population=ThalamusPopulation.TRN,
            weights=WeightInitializer.sparse_gaussian_no_autapses(
                n_input=self.trn_size,
                n_output=self.trn_size,
                connectivity=0.12,
                mean=0.003,
                std=0.001,
                device=device,
            ),
            receptor_type=ReceptorType.GABA_B,
            stp_config=STPConfig(U=0.20, tau_d=400.0, tau_f=40.0),
        )

        # TRN recurrent delay buffer (prevents instant feedback oscillations)
        trn_recurrent_delay_steps = int(config.trn_recurrent_delay_ms / config.dt_ms)
        self._trn_recurrent_buffer = CircularDelayBuffer(
            max_delay=trn_recurrent_delay_steps,
            size=self.trn_size,
            device=device,
            dtype=torch.bool,
        )

        # =====================================================================
        # ACETYLCHOLINE RECEPTOR (NB projection for sleep/wake modulation)
        # =====================================================================
        # Thalamus receives ACh from nucleus basalis (cortical arousal)
        # and brainstem cholinergic nuclei (ascending arousal system)
        # ACh M1 on TRN (sleep/wake mode), DA D1 on relay (reward gain), NE α₁ on relay (arousal gain)
        self._init_receptors_from_config(device)

        # =====================================================================
        # GAP JUNCTIONS (TRN interneuron synchronization)
        # =====================================================================
        # TRN neurons are densely coupled via gap junctions
        # This enables ultra-fast synchronization for coherent inhibitory volleys
        #
        # Physics: I_gap = g × (E_neighbor_avg - V), where E_neighbor_avg is
        # the weighted average of neighbor voltages (dynamic reversal potential).
        #
        # Create gap junctions using TRN→TRN lateral inhibition weights as proximity proxy.
        # TRN neurons that share lateral inhibitory connectivity are the same anatomical
        # neighbors that are coupled by gap junctions — so the lateral weight matrix is
        # the correct adjacency structure for electrical coupling (stronger lateral GABA_A
        # connectivity → higher probability of gap junction coupling).
        trn_trn_weights = self.get_synaptic_weights(self._trn_trn_gaba_a_synapse)
        self.gap_junctions = GapJunctionCoupling(
            n_neurons=self.trn_size,
            afferent_weights=trn_trn_weights,
            config=GapJunctionConfig(
                coupling_strength=config.gap_junctions.coupling_strength,
                connectivity_threshold=config.gap_junctions.connectivity_threshold,
                max_neighbors=config.gap_junctions.max_neighbors,
                interneuron_only=True,  # All TRN neurons are inhibitory
            ),
            interneuron_mask=None,  # All TRN neurons are interneurons
            device=device,
        )

        # =====================================================================
        # TRN→RELAY INHIBITION (internal recurrent connection)
        # =====================================================================
        # Biology: TRN provides feedforward and lateral inhibition to relay neurons
        # Distance: ~50-200μm (local), unmyelinated → 1ms delay (handled in forward pass)
        # Implements searchlight attention mechanism
        # Goal: Hyperpolarize relay below V_half_h_T (-0.3) → h_T de-inactivates → rebound burst on release
        # Need balance: enough inhibition for hyperpolarization, not so much it clamps neurons.
        self._trn_relay_synapse = self._add_internal_connection(
            source_population=ThalamusPopulation.TRN,
            target_population=ThalamusPopulation.RELAY,
            weights=WeightInitializer.sparse_gaussian(
                n_input=self.trn_size,
                n_output=self.relay_size,
                connectivity=0.6,
                mean=config.trn_relay_gaba_a_mean,
                std=config.trn_relay_gaba_a_mean / 3.0,  # Moderate variability in inhibitory strength across TRN→relay synapses
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=STPConfig(U=0.20, tau_d=250.0, tau_f=30.0),  # U 0.25→0.20: less depletion at TRN ~6 Hz
            # iSTDP: homeostatic feedforward gating balance — as relay excitatory
            # weights change via STDP, inhibition adjusts to maintain E/I ratio.
            learning_strategy=self._istdp_strategy,
        )

        # TRN → Relay: slow GABA_B component
        # GABA_B-mediated slow IPSPs (~100-300ms) at TRN→relay synapses are critical
        # for burst mode: the slow hyperpolarization (E_GABA_B = -0.8) pushes relay
        # membrane below V_half_h_T, de-inactivating T-type Ca²⁺ channels. On GABA_B
        # decay, the resulting rebound burst generates high-frequency spike bursts
        # (2-7 spikes at 200-400 Hz) that serve as "wake-up" signals.
        # Weight ~25% of GABA_A (slower kinetics = smaller peak g needed).
        self._trn_relay_gaba_b_synapse = self._add_internal_connection(
            source_population=ThalamusPopulation.TRN,
            target_population=ThalamusPopulation.RELAY,
            weights=WeightInitializer.sparse_gaussian(
                n_input=self.trn_size,
                n_output=self.relay_size,
                connectivity=0.5,
                mean=0.0003,  # Was 0.0008: GABA_B slow IPSPs are primary burst trigger;
                std=0.0001,   # reduce to limit T-channel deinactivation in waking state
                device=device,
            ),
            receptor_type=ReceptorType.GABA_B,
            stp_config=STPConfig(U=0.20, tau_d=400.0, tau_f=40.0),
            # iSTDP: slow GABA_B inhibition homeostatically tunes burst
            # mode threshold — as excitatory drive changes, GABA_B strength
            # adapts to maintain appropriate T-channel de-inactivation depth.
            learning_strategy=self._istdp_strategy,
        )

        # Ensure all tensors are on the correct device
        self.to(device)

    # =========================================================================
    # SYNAPTIC INPUT MANAGEMENT
    # =========================================================================

    def add_input_source(
        self,
        synapse_id: SynapseId,
        n_input: int,
        connectivity: float,
        weight_scale: Union[float, ConductanceScaledSpec],
        *,
        stp_config: Optional[STPConfig],
        learning_strategy: Optional[LearningStrategy],
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ) -> None:
        """Add an external input source to the thalamus.

        This method is overridden to handle the special case of external sensory inputs,
        which require additional routing to the TRN for feedforward inhibition (attentional gating).
        """
        super().add_input_source(
            synapse_id,
            n_input,
            connectivity,
            weight_scale,
            stp_config=stp_config,
            learning_strategy=learning_strategy,
            device=device,
        )

        # === Handle sensory-specific projection ===
        # Sensory input needs special handling for center-surround filtering
        if synapse_id.is_external_sensory_input():
            sensory_trn_synapse = SynapseId(
                source_region=synapse_id.source_region,
                source_population=synapse_id.source_population,
                target_region=self.region_name,
                target_population=ThalamusPopulation.TRN,
                receptor_type=ReceptorType.AMPA,
            )
            if sensory_trn_synapse not in self.synaptic_weights:
                if isinstance(weight_scale, ConductanceScaledSpec):
                    # TRN params: g_L=0.10, tau_E=4ms, v_threshold=1.6.
                    # Use source_rate_hz from relay spec; drive TRN strongly
                    # (feedforward inhibition must fire reliably on sensory burst).
                    trn_weights = WeightInitializer.conductance_scaled(
                        n_input=n_input,
                        n_output=self.trn_size,
                        connectivity=0.3,
                        source_rate_hz=weight_scale.source_rate_hz,
                        target_g_L=0.10,
                        target_tau_E_ms=4.0,
                        target_v_inf=1.65,
                        fraction_of_drive=0.80,
                        device=device,
                    )
                else:
                    trn_weights = WeightInitializer.sparse_random(
                        n_input=n_input,
                        n_output=self.trn_size,
                        connectivity=0.3,
                        weight_scale=weight_scale,  # Match sensory→relay scale
                        device=device,
                    )
                self.add_synapse(
                    synapse_id=sensory_trn_synapse,
                    weights=trn_weights,
                    stp_config=STPConfig(U=0.25, tau_d=350.0, tau_f=30.0),
                    # STDP: sensory→TRN drives attentional gating; plasticity
                    # shapes which stimuli recruit TRN for surround suppression.
                    learning_strategy=self._stdp_strategy,
                )

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    def _step(
        self,
        synaptic_inputs: SynapticInput,
        neuromodulator_inputs: NeuromodulatorInput,
    ) -> RegionOutput:
        """Process sensory input through thalamic relay."""
        device = self.device

        # =====================================================================
        # NEUROMODULATOR RECEPTOR UPDATES
        # =====================================================================
        self._update_receptors(neuromodulator_inputs)

        # =====================================================================
        # MULTI-SOURCE SYNAPTIC INTEGRATION
        # =====================================================================
        # Accumulate synaptic conductances from all registered external sources
        # Internal connections (TRN→relay, relay→TRN, TRN→TRN) are integrated below
        # using _integrate_synaptic_inputs_at_dendrites with their respective delay buffers.
        relay_conductance = torch.zeros(self.relay_size, device=device)
        relay_inhibition_external = torch.zeros(self.relay_size, device=device)  # GPi, SNr → relay GABA_A
        trn_conductance = torch.zeros(self.trn_size, device=device)

        for synapse_id, source_spikes in synaptic_inputs.items():
            source_spikes = source_spikes.float()
            # Route to appropriate target neurons
            if synapse_id.target_population == ThalamusPopulation.TRN:
                trn_conductance += self.get_synaptic_weights(synapse_id) @ source_spikes
            elif synapse_id.target_population == ThalamusPopulation.RELAY:
                # IMPORTANT: route by receptor type — GABA_A/B from GPi/SNr must go to
                # the inhibitory accumulator, not excitatory (bug fix: previously all
                # external relay inputs were incorrectly lumped into relay_conductance
                # which was then passed as AMPA excitation, causing GPi to excite relay).
                if synapse_id.receptor_type in (ReceptorType.GABA_A, ReceptorType.GABA_B):
                    relay_inhibition_external += self.get_synaptic_weights(synapse_id) @ source_spikes
                else:
                    relay_conductance += self.get_synaptic_weights(synapse_id) @ source_spikes

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
                trn_conductance += self.get_synaptic_weights(sensory_trn_synapse) @ source_spikes

        # =====================================================================
        # INTERNAL TRN→RELAY INHIBITION
        # =====================================================================
        prev_trn_spikes = self._prev_spikes(ThalamusPopulation.TRN)
        trn_relay_dendrite = self._integrate_synaptic_inputs_at_dendrites(
            {
                self._trn_relay_synapse: prev_trn_spikes,
                self._trn_relay_gaba_b_synapse: prev_trn_spikes,
            },
            n_neurons=self.relay_size,
        )
        relay_inhibition = trn_relay_dendrite.g_gaba_a + relay_inhibition_external
        relay_inhibition_gaba_b = trn_relay_dendrite.g_gaba_b

        # =====================================================================
        # RELAY NEURONS: Conductances → Relay
        # =====================================================================
        relay_excitation = relay_conductance

        # =====================================================================
        # HOMEOSTATIC INTRINSIC PLASTICITY: Add Minimal Noise
        # =====================================================================
        # Add baseline noise + brainstem tonic drive.
        # noise: zero-mean stochastic fluctuation representing unmodelled inputs.
        # _relay_baseline: deterministic ascending arousal drive (LC/PPN/raphe).
        # With g_L=0.08, E/I is already high (12.7); reduce noise to lower E/I
        # while maintaining stochastic depolarization above V_half_h_T (-0.65).
        noise = torch.randn(self.relay_size, device=device) * 0.013  # 0.015→0.008→0.013: 0.008 removed
        # desynchronizing noise, contributing to relay ρ=0.77 and burst mode 56%. 0.013 is a compromise.
        relay_excitation = relay_excitation + noise + self._relay_baseline

        # DA D1 + NE α₁ gain modulation on relay neurons
        # DA: reward-relevant stimuli get enhanced relay (+15% at max DA)
        # NE: arousal sharpens sensory throughput (+20% at max NE)
        da_relay_gain = 1.0 + 0.15 * self._da_concentration_relay.mean().item()
        ne_relay_gain = 1.0 + 0.2 * self._ne_concentration_relay.mean().item()
        relay_excitation = relay_excitation * da_relay_gain * ne_relay_gain

        # =====================================================================
        # UPDATE RELAY NEURONS
        # =====================================================================
        # Update relay neurons (ADR-005: 1D tensors, no batch)
        # relay_excitation and relay_inhibition are conductances (weights @ spikes)
        # Split excitatory conductance into AMPA (fast) and NMDA (slow)
        relay_g_ampa, relay_g_nmda = split_excitatory_conductance(relay_excitation, nmda_ratio=0.2)

        relay_spikes, _relay_membrane = self.relay_neurons.forward(
            g_ampa_input=ConductanceTensor(relay_g_ampa),  # [relay_size]
            g_nmda_input=ConductanceTensor(relay_g_nmda),  # [relay_size]
            g_gaba_a_input=ConductanceTensor(relay_inhibition),  # [relay_size]
            g_gaba_b_input=ConductanceTensor(relay_inhibition_gaba_b),  # [relay_size] TRN→relay slow IPSP
        )

        # =====================================================================
        # HOMEOSTATIC INTRINSIC PLASTICITY + SYNAPTIC SCALING
        # =====================================================================
        self._apply_all_population_homeostasis({ThalamusPopulation.RELAY: relay_spikes})

        # =====================================================================
        # TRN NEURONS: Synaptic conductances → TRN
        # =====================================================================
        # TRN excitation was already accumulated from synaptic_inputs above:
        # - Sensory collateral (via separate sensory→TRN SynapseId registered in add_input_source)
        # - L6a feedback (via cortex:l6a→thalamus:trn routing)

        # Now add relay collateral excitation
        # Biology: relay→TRN is a collateral of the thalamocortical axon; ~1ms synaptic delay.
        # Using the previous step's relay output preserves causality within the timestep.
        prev_relay = self._prev_spikes(ThalamusPopulation.RELAY)
        trn_excitation = trn_conductance + self._integrate_single_synaptic_input(self._relay_trn_synapse, prev_relay).g_ampa + self._trn_baseline

        # TRN lateral inhibition (winner-take-all across sensory streams)
        # ACh suppresses TRN lateral inhibition (McCormick & Prince 1986):
        # High ACh (wakefulness): Reduces TRN→TRN strength → softer competition →
        #   all streams pass (broadened attentional spotlight)
        # Low ACh (sleep/drowsy): Full TRN lateral inhibition → sharp winner-take-all →
        #   one sensory stream dominates (narrow spatial attention / spindle grouping)
        ach_level = self._ach_concentration_trn.mean().item()
        ach_recurrent_modulation = compute_ach_recurrent_suppression(ach_level)

        # Read delayed TRN spikes (1-step causally delayed via _trn_recurrent_buffer).
        # Each TRN neuron's spikes at t-delay inhibit its lateral peers at t,
        # implementing competition across sensory sectors.
        trn_recurrent_spikes = self._trn_recurrent_buffer.read(self._trn_recurrent_buffer.max_delay)
        trn_recurrent_dendrite = self._integrate_synaptic_inputs_at_dendrites(
            {
                self._trn_trn_gaba_a_synapse: trn_recurrent_spikes,
                self._trn_trn_gaba_b_synapse: trn_recurrent_spikes,
            },
            n_neurons=self.trn_size,
        )
        trn_inhibition_gaba_a = trn_recurrent_dendrite.g_gaba_a * ach_recurrent_modulation
        trn_inhibition_gaba_b = trn_recurrent_dendrite.g_gaba_b * ach_recurrent_modulation

        # Gap junction coupling (TRN synchronization)
        # Ultra-fast electrical coupling (<0.1ms) for coherent inhibitory volleys
        # Gap junctions return (g_total, E_effective) where:
        # - g_total: Sum of gap junction conductances for each neuron
        # - E_effective: Weighted average of neighbor voltages (dynamic reversal)
        # Physics: I_gap = g × (E_eff - V), where E_eff = weighted_avg(neighbor_voltages)
        trn_gap_conductance, trn_gap_reversal = self.gap_junctions.forward(self.trn_neurons.V_soma)

        # Split excitatory conductance into AMPA (fast) and NMDA (slow)
        trn_g_ampa, trn_g_nmda = split_excitatory_conductance(trn_excitation, nmda_ratio=0.2)

        # Pass gap junction conductances EXPLICITLY as forward() parameters
        # This is cleaner than monkey-patching _get_additional_conductances()
        apply_gap = trn_gap_conductance.abs().sum() > 1e-6  # Only apply if conductance is non-negligible
        trn_spikes, _trn_membrane = self.trn_neurons.forward(
            g_ampa_input=ConductanceTensor(trn_g_ampa),  # [trn_size]
            g_nmda_input=ConductanceTensor(trn_g_nmda),  # [trn_size]
            g_gaba_a_input=ConductanceTensor(trn_inhibition_gaba_a),  # [trn_size]
            g_gaba_b_input=ConductanceTensor(trn_inhibition_gaba_b),  # [trn_size] TRN→TRN slow IPSP
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
            ThalamusPopulation.RELAY: relay_spikes,
            ThalamusPopulation.TRN: trn_spikes,
        }

        # Ensure strategies are registered before dispatch.
        # Excitatory afferents (AMPA/NMDA) get Hebbian STDP; inhibitory
        # afferents (GABA_A/GABA_B from GPi, SNr, etc.) get homeostatic
        # iSTDP (Vogels et al. 2011) — excitatory STDP is biologically
        # inappropriate for GABAergic synapses.
        for synapse_id in list(synaptic_inputs.keys()):
            if self.get_learning_strategy(synapse_id) is None:
                if synapse_id.receptor_type.is_inhibitory:
                    self._add_learning_strategy(synapse_id, self._istdp_strategy, device=device)
                else:
                    self._add_learning_strategy(synapse_id, self._stdp_strategy, device=device)

        # =====================================================================
        # INTERNAL SYNAPTIC PLASTICITY
        # =====================================================================
        # Relay→TRN: STDP shapes which relay neurons recruit TRN (attention routing)
        self._apply_learning(self._relay_trn_synapse, prev_relay, trn_spikes,
                             learning_rate=self.config.learning_rate * 0.3)
        # TRN→TRN GABA_A: iSTDP homeostatically tunes lateral inhibition
        self._apply_learning(self._trn_trn_gaba_a_synapse, trn_recurrent_spikes, trn_spikes)
        # TRN→RELAY GABA_A: iSTDP homeostatically tunes feedforward gating
        self._apply_learning(self._trn_relay_synapse, prev_trn_spikes, relay_spikes)

        # Write TRN spikes to delay buffer for next timestep (multi-step recurrent delay)
        self._trn_recurrent_buffer.write_and_advance(trn_spikes)

        return region_outputs

    def _get_learning_kwargs(self, synapse_id: SynapseId) -> Dict[str, Any]:
        # RELAY: slower for L6b feedback; TRN: even slower for L6a
        if synapse_id.target_population == ThalamusPopulation.TRN:
            return {"learning_rate": self.config.learning_rate * 0.3}
        return {"learning_rate": self.config.learning_rate * 0.5}
