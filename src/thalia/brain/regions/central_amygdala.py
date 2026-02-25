"""Central Amygdala (CeA) - Fear Output and Autonomic Response Coordination.

The CeA is the primary output nucleus of the amygdala, translating BLA fear signals
into coordinated autonomic, behavioural, and endocrine fear responses.

Biological Background:
======================
**Anatomy:**
- Location: Medial temporal lobe (medial to BLA)
- Two main subdivisions:
    - CeL (lateral): Integrative; receives BLA, contains ON/OFF cells
    - CeM (medial): Output; projects to hypothalamus, brainstem, LC, LHb
- ~90% GABAergic (inhibitory) neurons
- Dense lateral inhibitory connectivity (CeL → CeM disinhibition dynamics)

**CeL Dynamics (On/Off cells; Haubensak et al. 2010):**
- PKCδ+ cells (fear-OFF): active during baseline, inhibited by fear CS
- PKCδ- cells (fear-ON): suppressed at baseline, activated by CS
- ON and OFF cells mutually inhibit each other: winner-take-all for fear gate

**CeM Output:**
- Hypothalamus (PVN): HPA axis activation, corticosterone release
- Brainstem (PAG): Freezing behaviour
- LC: NE-mediated arousal and cardiac acceleration
- LHb → RMTg → VTA: Aversive RPE, dopamine pause

**Key Inputs:**
- BLA principal neurons: CS-conditioned fear signal
- PFC (prelimbic): Fear potentiation
- PFC (infralimbic): Extinction (via BLA interneurons)
- LC (NE): Arousal state modulation

**Key Outputs:**
- LC: Fear-driven NE arousal
- LHb: Aversive signal → DA pause (negative RPE from unexpected US)
- (Future) Hypothalamus, PAG (brainstem fear responses - not modelled here)

**Implementation Notes:**
=========================
- CeL models the integrative lateral nucleus with moderate lateral inhibition
- CeM models the output medial nucleus, gated by CeL
- Both are primarily GABAergic (inhibitory to downstream targets)
- CeL → CeM creates ON/OFF dynamics (disinhibition when CeL is suppressed)

References:
-----------
- Ciocchi et al. (2010) Encoding of conditioned fear in central amygdala inhibitory
  circuits. Nature.
- Haubensak et al. (2010) Genetic dissection of an amygdala microcircuit that gates
  conditioned fear. Nature.
- LeDoux (2000) Emotion circuits in the brain. Annu Rev Neurosci.
"""

from __future__ import annotations

from typing import ClassVar, List

import torch

from thalia.brain.configs import CentralAmygdalaConfig
from thalia.components import (
    ConductanceLIF,
    ConductanceLIFConfig,
    WeightInitializer,
)
from thalia.components.synapses.stp import STPConfig, STPType
from thalia.typing import (
    ConductanceTensor,
    NeuromodulatorInput,
    PopulationPolarity,
    PopulationSizes,
    ReceptorType,
    RegionName,
    RegionOutput,
    SynapseId,
    SynapticInput,
)
from thalia.utils import split_excitatory_conductance

from .neural_region import NeuralRegion
from .population_names import CeAPopulation
from .region_registry import register_region


@register_region(
    "central_amygdala",
    aliases=["cea", "amygdala_cea"],
    description="Central amygdala — fear output nucleus (CeL integrator + CeM effector)",
    version="1.0",
    author="Thalia Project",
    config_class=CentralAmygdalaConfig,
)
class CentralAmygdala(NeuralRegion[CentralAmygdalaConfig]):
    """Central Amygdala: fear output and autonomic response nucleus.

    Contains:
    - CeL (lateral): Integrates BLA input, gating via ON/OFF mutual inhibition
    - CeM (medial): Projects fear output to LC, LHb, and (future) brainstem

    Architecture:
        BLA Principal ──→ [CeL] ─── (inhibits CeM) ─── [CeM] ──→ LC (NE arousal)
                            ↑                                  └──→ LHb (aversive RPE)
                         (BLA also)
                         direct to CeM (bypass CeL)
    """

    # NE from LC modulates CeM arousal output gain (alpha-1 adrenoceptors on CeM neurons).
    neuromodulator_subscriptions: ClassVar[List[str]] = ['ne']

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, config: CentralAmygdalaConfig, population_sizes: PopulationSizes, region_name: RegionName):
        """Initialize CeA populations and internal connectivity."""
        super().__init__(config, population_sizes, region_name)

        self.lateral_size = population_sizes[CeAPopulation.LATERAL]
        self.medial_size = population_sizes[CeAPopulation.MEDIAL]

        # =====================================================================
        # NEURON POPULATIONS
        # =====================================================================

        # CeL neurons (lateral CeA): integrative, GABAergic
        # Moderate activity; receives BLA principal input directly
        self.lateral_neurons = ConductanceLIF(
            n_neurons=self.lateral_size,
            config=ConductanceLIFConfig(
                region_name=self.region_name,
                population_name=CeAPopulation.LATERAL,
                tau_mem=config.tau_mem,
                v_threshold=config.v_threshold,
                v_reset=0.0,
                tau_ref=config.tau_ref,
                g_L=0.06,
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                tau_E=6.0,
                tau_I=12.0,
                adapt_increment=0.06,
                tau_adapt=120.0,
                noise_std=0.03,
            ),
            device=self.device,
        )

        # CeM neurons (medial CeA): output, projects to LC and LHb
        # More excitable than CeL; gated by CeL inhibition
        self.medial_neurons = ConductanceLIF(
            n_neurons=self.medial_size,
            config=ConductanceLIFConfig(
                region_name=self.region_name,
                population_name=CeAPopulation.MEDIAL,
                tau_mem=config.tau_mem,
                v_threshold=config.v_threshold * 0.9,  # Slightly easier to activate
                v_reset=0.0,
                tau_ref=config.tau_ref,
                g_L=0.05,
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                tau_E=6.0,
                tau_I=12.0,
                adapt_increment=0.05,
                tau_adapt=150.0,
                noise_std=0.03,
            ),
            device=self.device,
        )

        # =====================================================================
        # INTERNAL CONNECTIVITY
        # =====================================================================

        # CeL → CeM: lateral inhibition (the fear gate)
        # When CeL is suppressed (by extinction or CS absence), CeM is disinhibited
        # When CeL is active (during CS), it partially gates CeM output
        # Net effect: ON cells (PKCδ-) release CeM, OFF cells (PKCδ+) suppress it
        self._add_internal_connection(
            source_population=CeAPopulation.LATERAL,
            target_population=CeAPopulation.MEDIAL,
            weights=WeightInitializer.sparse_random(
                n_input=self.lateral_size,
                n_output=self.medial_size,
                connectivity=0.4,
                weight_scale=0.002,
                device=self.device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=STPConfig.from_type(STPType.DEPRESSING),
        )

        # CeL self-inhibition (lateral mutual inhibition: ON/OFF dynamics)
        self._add_internal_connection(
            source_population=CeAPopulation.LATERAL,
            target_population=CeAPopulation.LATERAL,
            weights=WeightInitializer.sparse_random(
                n_input=self.lateral_size,
                n_output=self.lateral_size,
                connectivity=0.2,
                weight_scale=0.0015,
                device=self.device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=STPConfig.from_type(STPType.DEPRESSING),
        )

        # Baseline drives
        self.baseline_drive_lateral = torch.full(
            (self.lateral_size,), config.baseline_drive_lateral, device=self.device
        )
        self.baseline_drive_medial = torch.full(
            (self.medial_size,), config.baseline_drive_medial, device=self.device
        )

        # =====================================================================
        # REGISTER POPULATIONS
        # =====================================================================
        self._register_neuron_population(CeAPopulation.LATERAL, self.lateral_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(CeAPopulation.MEDIAL, self.medial_neurons, polarity=PopulationPolarity.INHIBITORY)

        # Ensure all tensors are on the correct device
        self.to(self.device)

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    @torch.no_grad()
    def forward(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Compute CeA activity for one timestep.

        Args:
            synaptic_inputs: BLA principal and other inputs
            neuromodulator_inputs: NE from LC (arousal modulation)
        """
        self._pre_forward(synaptic_inputs, neuromodulator_inputs)

        cfg = self.config

        # =====================================================================
        # NE MODULATION (LC → CeA arousal state)
        # =====================================================================
        # NE increases CeA sensitivity during high arousal / stress
        ne_signal = neuromodulator_inputs.get('ne', None)
        ne_boost = 0.0
        if ne_signal is not None:
            ne_rate = ne_signal.float().mean().item()
            ne_boost = ne_rate * 0.3  # Mild NE-driven excitability increase

        # =====================================================================
        # CeL NEURONS (lateral CeA)
        # =====================================================================
        dendrite_lateral = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.lateral_size,
            filter_by_target_population=CeAPopulation.LATERAL,
        )

        # CeL self-inhibition (previous step)
        cel_cel_synapse = SynapseId(
            source_region=self.region_name,
            source_population=CeAPopulation.LATERAL,
            target_region=self.region_name,
            target_population=CeAPopulation.LATERAL,
            receptor_type=ReceptorType.GABA_A,
        )
        int_cel_inh = self._integrate_synaptic_inputs_at_dendrites(
            {cel_cel_synapse: self._lateral_spikes_prev}
            if hasattr(self, '_lateral_spikes_prev')
            else {},
            n_neurons=self.lateral_size,
        )

        g_exc_lat = self.baseline_drive_lateral.clone() + dendrite_lateral.g_ampa + ne_boost
        g_inh_lat = dendrite_lateral.g_gaba_a + int_cel_inh.g_gaba_a

        g_ampa_lat, g_nmda_lat = split_excitatory_conductance(g_exc_lat, nmda_ratio=0.20)

        lateral_spikes, _ = self.lateral_neurons.forward(
            g_ampa_input=ConductanceTensor(g_ampa_lat),
            g_nmda_input=ConductanceTensor(g_nmda_lat),
            g_gaba_a_input=ConductanceTensor(g_inh_lat),
            g_gaba_b_input=None,
        )

        # =====================================================================
        # CeM NEURONS (medial CeA — fear output)
        # =====================================================================
        dendrite_medial = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.medial_size,
            filter_by_target_population=CeAPopulation.MEDIAL,
        )

        # CeL → CeM inhibition (lateral CeA gates medial output)
        cel_cem_synapse = SynapseId(
            source_region=self.region_name,
            source_population=CeAPopulation.LATERAL,
            target_region=self.region_name,
            target_population=CeAPopulation.MEDIAL,
            receptor_type=ReceptorType.GABA_A,
        )
        int_cel_cem = self._integrate_synaptic_inputs_at_dendrites(
            {cel_cem_synapse: lateral_spikes},   # Current-step CeL spikes gate CeM
            n_neurons=self.medial_size,
        )

        g_exc_med = self.baseline_drive_medial.clone() + dendrite_medial.g_ampa + ne_boost
        g_inh_med = dendrite_medial.g_gaba_a + int_cel_cem.g_gaba_a

        g_ampa_med, g_nmda_med = split_excitatory_conductance(g_exc_med, nmda_ratio=0.20)

        medial_spikes, _ = self.medial_neurons.forward(
            g_ampa_input=ConductanceTensor(g_ampa_med),
            g_nmda_input=ConductanceTensor(g_nmda_med),
            g_gaba_a_input=ConductanceTensor(g_inh_med),
            g_gaba_b_input=None,
        )

        # Cache for next timestep
        self._lateral_spikes_prev = lateral_spikes

        region_outputs: RegionOutput = {
            CeAPopulation.LATERAL: lateral_spikes,
            CeAPopulation.MEDIAL: medial_spikes,
        }

        return self._post_forward(region_outputs)

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Propagate temporal parameter update to neuron populations."""
        super().update_temporal_parameters(dt_ms)
        self.lateral_neurons.update_temporal_parameters(dt_ms)
        self.medial_neurons.update_temporal_parameters(dt_ms)
