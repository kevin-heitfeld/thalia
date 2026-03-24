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
"""

from __future__ import annotations

from typing import ClassVar, List, Optional, Union

import torch

from thalia import GlobalConfig
from thalia.brain.configs import CentralAmygdalaConfig
from thalia.brain.neurons import (
    ConductanceLIF,
    build_conductance_lif_config,
    split_excitatory_conductance,
)
from thalia.brain.synapses import (
    STPConfig,
    WeightInitializer,
)
from thalia.typing import (
    ConductanceTensor,
    NeuromodulatorChannel,
    NeuromodulatorInput,
    PopulationSizes,
    ReceptorType,
    RegionName,
    RegionOutput,
    SynapticInput,
)

from .neural_region import NeuralRegion
from .population_names import CeAPopulation
from .region_registry import register_region


@register_region(
    "central_amygdala",
    aliases=["cea", "amygdala_cea"],
    description="Central amygdala — fear output nucleus (CeL integrator + CeM effector)",
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
    neuromodulator_subscriptions: ClassVar[List[NeuromodulatorChannel]] = [
        NeuromodulatorChannel.NE,
        NeuromodulatorChannel.SHT,
    ]

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        config: CentralAmygdalaConfig,
        population_sizes: PopulationSizes,
        region_name: RegionName,
        *,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ):
        """Initialize CeA populations and internal connectivity."""
        super().__init__(config, population_sizes, region_name, device=device)

        self.lateral_size = population_sizes[CeAPopulation.LATERAL]
        self.medial_size = population_sizes[CeAPopulation.MEDIAL]

        lateral_cfg = config.population_overrides[CeAPopulation.LATERAL]
        medial_cfg = config.population_overrides[CeAPopulation.MEDIAL]

        # =====================================================================
        # NEURON POPULATIONS
        # =====================================================================

        # CeL neurons (lateral CeA): integrative, GABAergic
        # Moderate activity; receives BLA principal input directly
        self.lateral_neurons: ConductanceLIF
        self.lateral_neurons = self._create_and_register_neuron_population(
            population_name=CeAPopulation.LATERAL,
            n_neurons=self.lateral_size,
            polarity=lateral_cfg.polarity,
            config=build_conductance_lif_config(
                lateral_cfg, self.lateral_size, device,
                tau_ref=self.config.tau_ref, g_L=0.06, tau_E=6.0, tau_I=12.0,
            ),
        )

        # CeM neurons (medial CeA): output, projects to LC and LHb
        # More excitable than CeL; gated by CeL inhibition
        self.medial_neurons: ConductanceLIF
        self.medial_neurons = self._create_and_register_neuron_population(
            population_name=CeAPopulation.MEDIAL,
            n_neurons=self.medial_size,
            polarity=medial_cfg.polarity,
            config=build_conductance_lif_config(
                medial_cfg, self.medial_size, device,
                tau_ref=self.config.tau_ref, tau_E=6.0, tau_I=12.0,
            ),
        )

        # =====================================================================
        # INTERNAL CONNECTIVITY
        # =====================================================================

        # CeL → CeM: lateral inhibition (the fear gate)
        # When CeL is suppressed (by extinction or CS absence), CeM is disinhibited
        # When CeL is active (during CS), it partially gates CeM output
        # Net effect: ON cells (PKCδ-) release CeM, OFF cells (PKCδ+) suppress it
        self._cel_cem_synapse = self._add_internal_connection(
            source_population=CeAPopulation.LATERAL,
            target_population=CeAPopulation.MEDIAL,
            weights=WeightInitializer.sparse_random(
                n_input=self.lateral_size,
                n_output=self.medial_size,
                connectivity=0.4,
                weight_scale=0.002,
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=STPConfig(U=0.30, tau_d=350.0, tau_f=20.0),
        )

        # CeL self-inhibition (lateral mutual inhibition: ON/OFF dynamics)
        self._cel_cel_synapse = self._add_internal_connection(
            source_population=CeAPopulation.LATERAL,
            target_population=CeAPopulation.LATERAL,
            weights=WeightInitializer.sparse_random(
                n_input=self.lateral_size,
                n_output=self.lateral_size,
                connectivity=0.2,
                weight_scale=0.0015,
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=STPConfig(U=0.30, tau_d=350.0, tau_f=20.0),
        )

        # CeM self-inhibition (local lateral inhibition regulates fear output)
        # Biology: CeM GABAergic neurons have local inhibitory connections
        # that prevent runaway synchronous firing when CeL disinhibits CeM.
        # Creates sparse competitive output: only the most strongly driven
        # CeM neurons fire, proportional to threat level.
        # Refs: Cassell et al. 1999; Ciocchi et al. 2010.
        self._cem_cem_synapse = self._add_internal_connection(
            source_population=CeAPopulation.MEDIAL,
            target_population=CeAPopulation.MEDIAL,
            weights=WeightInitializer.sparse_random(
                n_input=self.medial_size,
                n_output=self.medial_size,
                connectivity=0.25,
                weight_scale=0.002,
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=STPConfig(U=0.30, tau_d=350.0, tau_f=20.0),
        )

        # Baseline drives
        self.baseline_drive_lateral = torch.full(
            (self.lateral_size,), lateral_cfg.baseline_drive, device=device
        )
        self.baseline_drive_medial = torch.full(
            (self.medial_size,), medial_cfg.baseline_drive, device=device
        )

        # =====================================================================
        # SEROTONIN RECEPTOR (DRN → CeA, 5-HT1A inhibitory on CeM)
        # 5-HT1A (Gi → GIRK) on CeM: suppresses fear output
        self._init_receptors_from_config(device)

        # Ensure all tensors are on the correct device
        self.to(device)

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    def _step(
        self,
        synaptic_inputs: SynapticInput,
        neuromodulator_inputs: NeuromodulatorInput,
    ) -> RegionOutput:
        """Compute CeA activity for one timestep."""
        # =====================================================================
        # NE MODULATION (LC → CeA arousal state)
        # =====================================================================
        # NE increases CeA sensitivity during high arousal / stress
        ne_signal = self._extract_neuromodulator(neuromodulator_inputs, NeuromodulatorChannel.NE)
        ne_boost = 0.0
        if ne_signal is not None:
            ne_rate = ne_signal.float().mean().item()
            ne_boost = ne_rate * 0.3  # Mild NE-driven excitability increase

        # =====================================================================
        # 5-HT MODULATION (DRN → CeA anxiolysis)
        # =====================================================================
        # 5-HT1A on CeM: high serotonin suppresses fear output (-30% at max)
        self._update_receptors(neuromodulator_inputs)
        sht_level = self._sht_concentration.mean().item()
        sht_cem_suppression = max(0.0, 1.0 - 0.3 * sht_level)

        # =====================================================================
        # CeL NEURONS (lateral CeA)
        # =====================================================================
        dendrite_lateral = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.lateral_size,
            filter_by_target_population=CeAPopulation.LATERAL,
        )

        # CeL self-inhibition (previous step)
        int_cel_inh = self._integrate_single_synaptic_input(self._cel_cel_synapse, self._prev_spikes(CeAPopulation.LATERAL))

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
        int_cel_cem = self._integrate_single_synaptic_input(self._cel_cem_synapse, lateral_spikes)

        # CeM → CeM self-inhibition (previous step: local lateral inhibition)
        int_cem_inh = self._integrate_single_synaptic_input(self._cem_cem_synapse, self._prev_spikes(CeAPopulation.MEDIAL))

        g_exc_med = self.baseline_drive_medial.clone() + dendrite_medial.g_ampa + ne_boost
        g_exc_med = g_exc_med * sht_cem_suppression  # 5-HT1A anxiolytic suppression
        g_inh_med = dendrite_medial.g_gaba_a + int_cel_cem.g_gaba_a + int_cem_inh.g_gaba_a

        g_ampa_med, g_nmda_med = split_excitatory_conductance(g_exc_med, nmda_ratio=0.20)

        medial_spikes, _ = self.medial_neurons.forward(
            g_ampa_input=ConductanceTensor(g_ampa_med),
            g_nmda_input=ConductanceTensor(g_nmda_med),
            g_gaba_a_input=ConductanceTensor(g_inh_med),
            g_gaba_b_input=None,
        )

        region_outputs: RegionOutput = {
            CeAPopulation.LATERAL: lateral_spikes,
            CeAPopulation.MEDIAL: medial_spikes,
        }

        return region_outputs
