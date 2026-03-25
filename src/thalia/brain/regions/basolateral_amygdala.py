"""Basolateral Amygdala (BLA) - Fear Conditioning and Extinction.

The BLA is the principal site of fear learning. It receives convergent CS
(conditioned stimulus) and US (unconditioned stimulus) inputs and forms
hebbian CS–US associations via STDP-like plasticity at principal neuron synapses.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Union

import torch

from thalia import GlobalConfig
from thalia.brain.configs import BasolateralAmygdalaConfig
from thalia.brain.neurons import (
    build_conductance_lif_config,
    build_two_compartment_config,
    split_excitatory_conductance,
)
from thalia.brain.synapses import STPConfig
from thalia.learning import (
    InhibitorySTDPConfig,
    InhibitorySTDPStrategy,
    MetaplasticityConfig,
    MetaplasticityStrategy,
    STDPConfig,
    STDPStrategy,
    TagAndCaptureConfig,
    TagAndCaptureStrategy,
    ThreeFactorConfig,
    ThreeFactorStrategy,
)
from thalia.typing import (
    ConductanceTensor,
    NeuromodulatorChannel,
    NeuromodulatorInput,
    PopulationSizes,
    ReceptorType,
    RegionName,
    RegionOutput,
    SynapseId,
    SynapticInput,
)

from .neural_region import InternalConnectionSpec, NeuralRegion
from .population_names import BLAPopulation
from .region_registry import register_region


@register_region(
    "basolateral_amygdala",
    aliases=["bla", "amygdala_bla"],
    description="Basolateral amygdala — fear conditioning and extinction via CS-US association",
)
class BasolateralAmygdala(NeuralRegion[BasolateralAmygdalaConfig]):
    """Basolateral Amygdala: fear conditioning and extinction nucleus.

    Contains:
    - Principal neurons (glutamatergic): form CS–US associations via STDP
    - PV interneurons: fast feedforward inhibition (fear discrimination)
    - SOM interneurons: slow dendritic inhibition (extinction gating)
    """

    # Mesolimbic DA (VTA → BLA) modulates fear conditioning plasticity.
    # NE (LC → BLA) gates fear memory consolidation via β-adrenergic receptors.
    # 5-HT (DRN → BLA) suppresses principal excitability during extinction (5-HT1A).
    neuromodulator_subscriptions: ClassVar[List[NeuromodulatorChannel]] = [
        NeuromodulatorChannel.DA_MESOLIMBIC,
        NeuromodulatorChannel.NE,
        NeuromodulatorChannel.ACH,
        NeuromodulatorChannel.SHT,
    ]

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(
        self,
        config: BasolateralAmygdalaConfig,
        population_sizes: PopulationSizes,
        region_name: RegionName,
        *,
        device: Union[str, torch.device] = GlobalConfig.DEFAULT_DEVICE,
    ):
        """Initialize BLA populations and internal connectivity."""
        super().__init__(config, population_sizes, region_name, device=device)

        self.principal_size = population_sizes[BLAPopulation.PRINCIPAL]
        self.pv_size = population_sizes[BLAPopulation.PV]
        self.som_size = population_sizes[BLAPopulation.SOM]

        principal_cfg = config.population_overrides[BLAPopulation.PRINCIPAL]
        pv_cfg = config.population_overrides[BLAPopulation.PV]
        som_cfg = config.population_overrides[BLAPopulation.SOM]

        # =====================================================================
        # NEURON POPULATIONS
        # =====================================================================

        # Principal neurons: slow, STDP-plastic, CS-US association
        # Two-compartment: basal (proximal) receives thalamic/cortical CS input;
        # apical (distal) receives SOM dendritic inhibition for extinction gating.
        self.principal_neurons = self._create_and_register_neuron_population(
            population_name=BLAPopulation.PRINCIPAL,
            n_neurons=self.principal_size,
            polarity=principal_cfg.polarity,
            config=build_two_compartment_config(
                principal_cfg, self.principal_size, device,
                tau_ref=config.tau_ref, tau_E=7.0, tau_I=12.0,
            ),
        )

        # PV interneurons: fast-spiking, feedforward fear gating
        self.pv_neurons = self._create_and_register_neuron_population(
            population_name=BLAPopulation.PV,
            n_neurons=self.pv_size,
            polarity=pv_cfg.polarity,
            config=build_conductance_lif_config(
                pv_cfg, self.pv_size, device,
                tau_ref=2.5, g_L=0.10, tau_E=3.0, tau_I=3.0,
                tau_mem_cv=0.10, v_threshold_cv=0.06, g_L_cv=0.08,
            ),
        )

        # SOM interneurons: slower, dendritic inhibition, extinction gating
        # Modelled as standard neurons with longer time constants.
        self.som_neurons = self._create_and_register_neuron_population(
            population_name=BLAPopulation.SOM,
            n_neurons=self.som_size,
            polarity=som_cfg.polarity,
            config=build_conductance_lif_config(
                som_cfg, self.som_size, device,
                g_L=0.06, tau_E=8.0, tau_I=15.0,
            ),
        )

        # =====================================================================
        # INTERNAL CONNECTIVITY
        # =====================================================================

        # Inhibitory STDP for I→E synapses (Vogels et al. 2011)
        istdp_cfg = InhibitorySTDPConfig(
            learning_rate=config.istdp_learning_rate,
            tau_istdp=config.istdp_tau_ms,
            alpha=config.istdp_alpha,
            w_min=config.synaptic_scaling.w_min,
            w_max=config.synaptic_scaling.w_max,
        )
        self.istdp_pv:  InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)
        self.istdp_som: InhibitorySTDPStrategy = InhibitorySTDPStrategy(istdp_cfg)

        # Hebbian E→I STDP for Principal→PV and Principal→SOM.
        # PV/SOM interneurons must track evolving principal cell assemblies
        # so they are recruited to gate the correct fear engrams (Wolff et al.
        # 2014; Krabbe et al. 2019).  Conservative rate (15% of E→E) to avoid
        # destabilising the inhibitory feedback loop.
        ei_stdp_cfg = STDPConfig(
            learning_rate=config.learning_rate * 0.15,
            a_plus=0.005,
            a_minus=0.003,
            tau_plus=20.0,
            tau_minus=20.0,
            w_min=config.synaptic_scaling.w_min,
            w_max=config.synaptic_scaling.w_max,
        )
        self.ei_stdp_pv:  STDPStrategy = STDPStrategy(ei_stdp_cfg)
        self.ei_stdp_som: STDPStrategy = STDPStrategy(ei_stdp_cfg)

        # Principal ↔ PV (feedforward excitation + fast feedback inhibition)
        # Biology: Principal axon collaterals excite local PV interneurons;
        # PV cells hyperpolarise principal soma, sharpening temporal tuning.
        # BLA:pv is quiescent at rest (~0.5–5 Hz) — only recruited during CS/US events.
        # At baseline with ~2 Hz principal drive, weight_scale=0.0005 yields ~0.9 Hz PV,
        # appropriate for resting state (Woodruff & Sah 2007; Bienvenu et al. 2012).
        self._principal_pv_synapse, self._pv_principal_synapse = self._add_feedforward_inhibition(
            exc_pop=BLAPopulation.PRINCIPAL,
            inh_pop=BLAPopulation.PV,
            e_to_i=InternalConnectionSpec(
                connectivity=0.4, weight_scale=0.0005,
                receptor_type=ReceptorType.AMPA,
                stp_config=STPConfig(U=0.5, tau_d=800.0, tau_f=20.0),
                learning_strategy=self.ei_stdp_pv,
            ),
            i_to_e=InternalConnectionSpec(
                connectivity=0.5,
                weight_scale=0.016,  # Doubled (0.008→0.016): feedback inhibition must overcome runaway
                receptor_type=ReceptorType.GABA_A,
                stp_config=STPConfig(U=0.25, tau_d=250.0, tau_f=20.0),
                learning_strategy=self.istdp_pv,
            ),
            device=device,
        )

        # Principal ↔ SOM (slower excitation + dendritic inhibition: extinction gating)
        # Biology: Activity-dependent recruitment of SOM cells during sustained input;
        # SOM targets principal cell dendrites, reducing excitatory integration.
        self._principal_som_synapse, self._som_principal_synapse = self._add_feedforward_inhibition(
            exc_pop=BLAPopulation.PRINCIPAL,
            inh_pop=BLAPopulation.SOM,
            e_to_i=InternalConnectionSpec(
                connectivity=0.3, weight_scale=0.0015,
                receptor_type=ReceptorType.AMPA,
                stp_config=STPConfig(U=0.1, tau_d=300.0, tau_f=300.0),
                learning_strategy=self.ei_stdp_som,
            ),
            i_to_e=InternalConnectionSpec(
                connectivity=0.4, weight_scale=0.002,
                receptor_type=ReceptorType.GABA_A,
                stp_config=STPConfig(U=0.25, tau_d=400.0, tau_f=20.0),
                learning_strategy=self.istdp_som,
            ),
            device=device,
        )

        # =====================================================================
        # THREE-FACTOR LEARNING (CS–US association at principal neurons)
        # =====================================================================
        # Fear conditioning requires temporal credit assignment: CS precedes US
        # by hundreds of milliseconds.  A simple two-factor STDP window (±25 ms)
        # cannot bridge this gap.  Three-factor learning solves this:
        #   Factor 1: Pre-post spike coincidence → eligibility trace
        #   Factor 2: Neuromodulator (NE β-adrenergic, DA D1) → gate
        #   Factor 3: Weight update = eligibility × modulator
        # The eligibility trace (tau ~500 ms) persists until the US-driven NE/DA
        # burst arrives, enabling CS–US association across realistic delays.
        # Refs: Izhikevich 2007; Frémaux & Gerstner 2016; Johansen et al. 2014.
        self._three_factor_strategy = ThreeFactorStrategy(ThreeFactorConfig(
            learning_rate=config.learning_rate,
            eligibility_tau=500.0,  # Eligibility trace persists ~500 ms
            modulator_tau=50.0,     # Modulator (NE/DA) integration window
            w_min=config.synaptic_scaling.w_min,
            w_max=config.synaptic_scaling.w_max,
        ))

        # Synaptic tagging and capture: short-lived tags created by initial
        # plasticity are consolidated into long-term changes only when a
        # neuromodulator signal (NE/DA) confirms behavioural significance.
        # This implements the Frey–Morris (1997) synaptic tagging hypothesis
        # for emotional memory persistence (Moncada & Bhagya 2007).
        self._tag_and_capture_config = TagAndCaptureConfig(
            tag_decay=0.999,                # ~1000-step tag lifetime (slow decay for fear)
            tag_threshold=0.0,              # Any activity can set a tag
            consolidation_lr_scale=0.5,     # 50% of base LR during capture
            consolidation_da_threshold=0.1, # Moderate DA needed for consolidation
        )

        # Metaplasticity config for principal-neuron synapses.
        # Emotional memories are persistent (McGaugh 2004); consolidated fear
        # associations resist extinction → strong consolidation sensitivity.
        self._meta_config = MetaplasticityConfig(
            tau_recovery_ms=5000.0,
            depression_strength=5.0,
            tau_consolidation_ms=300000.0,
            consolidation_sensitivity=0.1,
            rate_min=0.1,
        )

        # =====================================================================
        # SOM AFFERENT PLASTICITY (extinction learning)
        # =====================================================================
        # PFC→BLA:SOM plasticity is critical for fear extinction.  During
        # extinction, infralimbic PFC strengthens projections to BLA SOM
        # interneurons, increasing dendritic inhibition of principal cells
        # and suppressing conditioned fear responses.
        # Refs: Likhtik et al. 2005; Cho et al. 2013; Duvarci & Bhagya 2014.
        # Conservative learning rate (20% of principal) — SOM recruitment
        # should shift gradually across extinction trials.
        self._som_afferent_stdp = STDPStrategy(STDPConfig(
            learning_rate=config.learning_rate * 0.20,
            a_plus=0.005,
            a_minus=0.003,
            tau_plus=20.0,
            tau_minus=20.0,
            w_min=config.synaptic_scaling.w_min,
            w_max=config.synaptic_scaling.w_max,
        ))

        # Baseline drive tensor (tonic low-level activity)
        self.baseline_drive = torch.full((self.principal_size,), config.baseline_drive, device=device)

        # =====================================================================
        # NEUROMODULATOR RECEPTORS
        # =====================================================================
        # DA D1 (VTA → BLA): fear acquisition / excitability boost (Bissière 2003)
        # NE β-adrenergic (LC → BLA): fear memory consolidation (McGaugh 2004)
        # 5-HT1A (DRN → BLA): extinction gating (Gross 2002; Bhagya 2015)
        # ACh M1 (NB → BLA): attention-enhanced encoding (Power 2003)
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
        """Compute BLA activity for one timestep."""
        device = self.device

        # =====================================================================
        # NEUROMODULATOR RECEPTOR UPDATES
        # =====================================================================
        self._update_receptors(neuromodulator_inputs)

        da_level = self._da_concentration.mean().item()
        ne_level = self._ne_concentration.mean().item()
        sht_level = self._sht_concentration.mean().item()
        ach_level = self._ach_concentration.mean().item()

        # =====================================================================
        # PRINCIPAL NEURONS
        # =====================================================================
        dendrite_principal = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.principal_size,
            filter_by_target_population=BLAPopulation.PRINCIPAL,
        )

        # PV → PRINCIPAL: perisomatic (basal) inhibition — fast, soma-targeted
        # Biology: PV basket cells form synapses on soma and proximal dendrites
        #
        # SOM → PRINCIPAL: dendritic (apical) inhibition — slow, extinction gating
        # Biology: SOM/Martinotti cells target distal apical dendrites, reducing
        # excitatory integration of top-down and associative inputs during extinction
        #
        # Separate PV and SOM inhibition so they map to the correct compartments
        pv_int_inh = self._integrate_single_synaptic_input(self._pv_principal_synapse, self._prev_spikes(BLAPopulation.PV))
        som_int_inh = self._integrate_single_synaptic_input(self._som_principal_synapse, self._prev_spikes(BLAPopulation.SOM))

        # DA D1: mild excitability boost via cAMP/PKA (+20% at max DA)
        # NE β-adrenergic: mild excitability boost during arousal (+30% at max NE)
        # 5-HT1A (Gi-coupled): hyperpolarising gate, reduces excitability (-35% at max 5-HT)
        # ACh M1: attention-enhanced encoding boost (+15% at max ACh)
        da_excitability = 1.0 + 0.2 * da_level
        ne_excitability = 1.0 + 0.3 * ne_level
        sht_excitability = max(0.0, 1.0 - 0.35 * sht_level)
        ach_excitability = 1.0 + 0.15 * ach_level
        g_exc = (self.baseline_drive.clone() + dendrite_principal.g_ampa) * da_excitability * ne_excitability * sht_excitability * ach_excitability
        g_ampa, g_nmda = split_excitatory_conductance(g_exc, nmda_ratio=0.25)

        # Basal conductances: external GABAergic inputs + PV perisomatic inhibition
        g_gaba_a_basal = dendrite_principal.g_gaba_a + pv_int_inh.g_gaba_a
        # Apical conductances: SOM dendritic inhibition (extinction gating)
        g_gaba_a_apical = som_int_inh.g_gaba_a

        principal_spikes, _v_soma, _v_dend = self.principal_neurons.forward(
            g_ampa_basal=ConductanceTensor(g_ampa),
            g_nmda_basal=ConductanceTensor(g_nmda),
            g_gaba_a_basal=ConductanceTensor(g_gaba_a_basal),
            g_gaba_b_basal=None,
            g_gaba_a_apical=ConductanceTensor(g_gaba_a_apical),
        )

        # =====================================================================
        # PV INTERNEURONS (fast feedforward inhibition)
        # =====================================================================
        dendrite_pv = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.pv_size,
            filter_by_target_population=BLAPopulation.PV,
        )

        # Principal → PV internal excitation
        int_principal_pv = self._integrate_single_synaptic_input(self._principal_pv_synapse, principal_spikes)

        g_exc_pv = dendrite_pv.g_ampa + int_principal_pv.g_ampa
        g_inh_pv = dendrite_pv.g_gaba_a

        g_ampa_pv, g_nmda_pv = split_excitatory_conductance(g_exc_pv, nmda_ratio=0.05)  # Fast, mostly AMPA

        pv_spikes, _ = self.pv_neurons.forward(
            g_ampa_input=ConductanceTensor(g_ampa_pv),
            g_nmda_input=ConductanceTensor(g_nmda_pv),
            g_gaba_a_input=ConductanceTensor(g_inh_pv),
            g_gaba_b_input=None,
        )

        # =====================================================================
        # SOM INTERNEURONS (slow dendritic inhibition)
        # =====================================================================
        dendrite_som = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.som_size,
            filter_by_target_population=BLAPopulation.SOM,
        )

        int_principal_som = self._integrate_single_synaptic_input(self._principal_som_synapse, principal_spikes)

        g_exc_som = dendrite_som.g_ampa + int_principal_som.g_ampa
        g_inh_som = dendrite_som.g_gaba_a

        g_ampa_som, g_nmda_som = split_excitatory_conductance(g_exc_som, nmda_ratio=0.15)

        som_spikes, _ = self.som_neurons.forward(
            g_ampa_input=ConductanceTensor(g_ampa_som),
            g_nmda_input=ConductanceTensor(g_nmda_som),
            g_gaba_a_input=ConductanceTensor(g_inh_som),
            g_gaba_b_input=None,
        )

        region_outputs: RegionOutput = {
            BLAPopulation.PRINCIPAL: principal_spikes,
            BLAPopulation.PV: pv_spikes,
            BLAPopulation.SOM: som_spikes,
        }

        # =====================================================================
        # STDP LEARNING (CS–US association at principal neurons)
        # =====================================================================
        if not self.config.learning_disabled:
            # Register three-factor learning for any external synapse targeting PRINCIPAL.
            # Each synapse gets its own MetaplasticityStrategy wrapper so that
            # per-synapse consolidation/rate buffers are correctly shaped.
            # Stack: ThreeFactor → TagAndCapture → Metaplasticity
            #
            # External excitatory→SOM synapses get STDP for extinction learning:
            # PFC projections to BLA SOM interneurons strengthen during extinction,
            # increasing dendritic inhibition of principal cells (Likhtik et al. 2005).
            # PV afferents remain fixed — PV handles fast feedforward gating, not
            # slow extinction-related plasticity.
            for synapse_id in synaptic_inputs:
                if self.get_learning_strategy(synapse_id) is not None:
                    continue
                if synapse_id.target_population == BLAPopulation.PRINCIPAL:
                    tagged = TagAndCaptureStrategy(
                        base_strategy=self._three_factor_strategy,
                        config=self._tag_and_capture_config,
                    )
                    strategy = MetaplasticityStrategy(
                        base_strategy=tagged,
                        config=self._meta_config,
                    )
                    self._add_learning_strategy(synapse_id, strategy, device=device)
                elif (
                    synapse_id.target_population == BLAPopulation.SOM
                    and synapse_id.receptor_type.is_excitatory
                ):
                    self._add_learning_strategy(
                        synapse_id, self._som_afferent_stdp, device=device,
                    )

        # =====================================================================
        # HOMEOSTASIS: Rate tracking + synaptic scaling for principal cells
        # =====================================================================
        self._apply_all_population_homeostasis(region_outputs)

        return region_outputs

    def _get_learning_kwargs(self, synapse_id: SynapseId) -> Dict[str, Any]:
        da_level = self._da_concentration.mean().item()
        ne_level = self._ne_concentration.mean().item()
        sht_level = self._sht_concentration.mean().item()
        ach_level = self._ach_concentration.mean().item()
        # NE β-adrenergic: enhances learning magnitude during emotional arousal
        # Range: LR × [1.0, 2.5] (high NE during shock strongly boosts fear learning)
        # Ref: McGaugh 2004; Roozendaal et al. 2009
        # ACh M1: enhances LTP during attentional encoding (+40% at max ACh)
        effective_lr = self.config.learning_rate * (1.0 + 1.5 * ne_level) * (1.0 + 0.4 * ach_level)
        # Three-factor modulator: combined NE + DA signal gates eligibility → weight.
        # NE is the primary US signal in fear conditioning (Johansen et al. 2014);
        # DA provides additional salience gating via D1 receptors (Bissière et al. 2003).
        modulator = ne_level + 0.5 * da_level
        # 5-HT passed separately for asymmetric LTP/LTD modulation:
        # suppresses LTP (anti-impulsive gating) and enhances LTD (extinction).
        return {
            "learning_rate": max(effective_lr, 0.0),
            "modulator": modulator,
            "dopamine": da_level,
            "serotonin": sht_level,
            "norepinephrine": ne_level,
            "acetylcholine": ach_level,
        }
