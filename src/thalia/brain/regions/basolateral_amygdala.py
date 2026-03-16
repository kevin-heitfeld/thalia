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
    ConductanceLIFConfig,
    TwoCompartmentLIFConfig,
    heterogeneous_tau_mem,
    heterogeneous_v_threshold,
    heterogeneous_adapt_increment,
    heterogeneous_g_L,
    split_excitatory_conductance,
)
from thalia.brain.synapses import (
    NMReceptorType,
    make_neuromodulator_receptor,
    STPConfig,
    WeightInitializer,
)
from thalia.learning import STDPConfig, STDPStrategy
from thalia.typing import (
    ConductanceTensor,
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

from .neural_region import NeuralRegion
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

        # =====================================================================
        # NEURON POPULATIONS
        # =====================================================================

        # Principal neurons: slow, STDP-plastic, CS-US association
        # Two-compartment: basal (proximal) receives thalamic/cortical CS input;
        # apical (distal) receives SOM dendritic inhibition for extinction gating.
        self.principal_neurons = self._create_and_register_neuron_population(
            population_name=BLAPopulation.PRINCIPAL,
            n_neurons=self.principal_size,
            polarity=PopulationPolarity.EXCITATORY,
            config=TwoCompartmentLIFConfig(
                tau_mem_ms=heterogeneous_tau_mem(config.tau_mem_principal, self.principal_size, device),
                v_threshold=heterogeneous_v_threshold(config.v_threshold_principal, self.principal_size, device),
                v_reset=-0.10,
                tau_ref=config.tau_ref,
                g_L=heterogeneous_g_L(0.05, self.principal_size, device),
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                tau_E=7.0,         # Moderate AMPA (slightly slower for integration)
                tau_I=12.0,        # GABA_A for PV/SOM inhibition
                adapt_increment=heterogeneous_adapt_increment(0.30, self.principal_size, device),
                tau_adapt=200.0,   # Slower decay to maintain adaptation during burst
                noise_std=0.02,    # Reduced noise; high noise was driving extra spikes
                # Two-compartment dendritic parameters
                g_c=0.05,          # Somato-dendritic coupling conductance
                C_d=0.5,           # Dendritic membrane capacitance
                g_L_d=0.03,        # Dendritic leak conductance
                bap_amplitude=0.3, # Back-propagating AP strength (enables STDP coincidence)
                theta_Ca=2.0,      # Ca2+ spike threshold (dendritic burst)
                g_Ca_spike=0.30,   # Ca2+ spike conductance
                tau_Ca_ms=20.0,    # Ca2+ decay time (synaptogenesis window)
            ),
        )

        # PV interneurons: fast-spiking, feedforward fear gating
        self.pv_neurons = self._create_and_register_neuron_population(
            population_name=BLAPopulation.PV,
            n_neurons=self.pv_size,
            polarity=PopulationPolarity.INHIBITORY,
            config=ConductanceLIFConfig(
                v_threshold=heterogeneous_v_threshold(1.0, self.pv_size, device, cv=0.06),
                v_reset=0.0,
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                tau_E=3.0,
                tau_I=3.0,
                tau_ref=2.5,
                g_L=heterogeneous_g_L(0.10, self.pv_size, device, cv=0.08),
                tau_mem_ms=heterogeneous_tau_mem(8.0, self.pv_size, device=device, cv=0.10),
            ),
        )

        # SOM interneurons: slower, dendritic inhibition, extinction gating
        # Modelled as standard neurons with longer time constants.
        self.som_neurons = self._create_and_register_neuron_population(
            population_name=BLAPopulation.SOM,
            n_neurons=self.som_size,
            polarity=PopulationPolarity.INHIBITORY,
            config=ConductanceLIFConfig(
                tau_mem_ms=heterogeneous_tau_mem(config.tau_mem_som, self.som_size, self.device),
                v_threshold=heterogeneous_v_threshold(1.1, self.som_size, self.device),
                v_reset=0.0,
                tau_ref=5.0,
                g_L=heterogeneous_g_L(0.06, self.som_size, self.device),
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                tau_E=8.0,
                tau_I=15.0,
                adapt_increment=heterogeneous_adapt_increment(0.1, self.som_size, self.device),
                tau_adapt=200.0,
                E_adapt=-0.5,
                noise_std=0.03,
            ),
        )

        # =====================================================================
        # INTERNAL CONNECTIVITY
        # =====================================================================

        # Principal → PV (feedforward excitation of PV → rapid feedback inhibition)
        # Biology: Principal axon collaterals excite local PV interneurons.
        # BLA:pv is quiescent at rest (~0.5–5 Hz) — only recruited during CS/US events.
        # At baseline with ~2 Hz principal drive, weight_scale=0.0005 yields ~0.9 Hz PV, which
        # is appropriate for resting state (Woodruff & Sah 2007; Bienvenu et al. 2012).
        self._add_internal_connection(
            source_population=BLAPopulation.PRINCIPAL,
            target_population=BLAPopulation.PV,
            weights=WeightInitializer.sparse_random(
                n_input=self.principal_size,
                n_output=self.pv_size,
                connectivity=0.4,
                weight_scale=0.0005,
                device=device,
            ),
            receptor_type=ReceptorType.AMPA,
            stp_config=STPConfig(U=0.5, tau_d=800.0, tau_f=20.0),
        )

        # PV → Principal (fast feedforward inhibition)
        # Biology: PV cells hyperpolarise principal soma, sharpening temporal tuning
        self._add_internal_connection(
            source_population=BLAPopulation.PV,
            target_population=BLAPopulation.PRINCIPAL,
            weights=WeightInitializer.sparse_random(
                n_input=self.pv_size,
                n_output=self.principal_size,
                connectivity=0.5,
                weight_scale=0.016,  # Doubled (0.008→0.016): feedback inhibition must overcome runaway
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=STPConfig(U=0.25, tau_d=250.0, tau_f=20.0),
        )

        # Principal → SOM (slower recurrent excitation of SOM interneurons)
        # Biology: Activity-dependent recruitment of SOM cells during sustained input
        self._add_internal_connection(
            source_population=BLAPopulation.PRINCIPAL,
            target_population=BLAPopulation.SOM,
            weights=WeightInitializer.sparse_random(
                n_input=self.principal_size,
                n_output=self.som_size,
                connectivity=0.3,
                weight_scale=0.0015,
                device=device,
            ),
            receptor_type=ReceptorType.AMPA,
            stp_config=STPConfig(U=0.1, tau_d=300.0, tau_f=300.0),
        )

        # SOM → Principal (dendritic inhibition: extinction gating)
        # Biology: SOM targets principal cell dendrites, reducing excitatory integration
        self._add_internal_connection(
            source_population=BLAPopulation.SOM,
            target_population=BLAPopulation.PRINCIPAL,
            weights=WeightInitializer.sparse_random(
                n_input=self.som_size,
                n_output=self.principal_size,
                connectivity=0.4,
                weight_scale=0.002,
                device=device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=STPConfig(U=0.25, tau_d=400.0, tau_f=20.0),
        )

        # =====================================================================
        # STDP LEARNING STRATEGY (CS–US association at principal neurons)
        # =====================================================================
        # Standard STDP for fear conditioning: Hebbian + anti-Hebbian window
        # Additional modulation by DA (three-factor rule) handled implicitly:
        # DA from VTA increases baseline excitability → more co-activation
        # (Full three-factor STDP would require eligibility trace — future work)
        self._stdp_strategy = STDPStrategy(STDPConfig(
            learning_rate=config.learning_rate,
            a_plus=0.012,   # LTP (fear acquisition is fast)
            a_minus=0.006,  # LTD (slower forgetting)
            tau_plus=25.0,  # ±25ms window (BLA temporal resolution)
            tau_minus=25.0,
            w_min=config.w_min,
            w_max=config.w_max,
        ))

        # Baseline drive tensor (tonic low-level activity)
        self.baseline_drive = torch.full((self.principal_size,), config.baseline_drive, device=device)

        # =====================================================================
        # NOREPINEPHRINE RECEPTOR (LC → BLA, β-adrenergic)
        # =====================================================================
        # NE from LC gates fear memory consolidation. High NE during emotional
        # arousal (stress/shock) enhances STDP magnitude via β-adrenergic signalling,
        # and mildly boosts principal excitability (signal-to-noise gain).
        # Ref: McGaugh 2004; Roozendaal et al. 2009 (stress-NE synergy in BLA)
        # β-adrenergic (Gs → cAMP → PKA): gates fear memory consolidation.
        # τ_rise=80 ms, τ_decay=1000 ms (Woodward 1991; McGaugh 2004).
        self.ne_receptor = make_neuromodulator_receptor(
            NMReceptorType.NE_BETA, n_receptors=self.principal_size, dt_ms=self.config.dt_ms, device=device
        )
        self._ne_concentration: torch.Tensor
        self.register_buffer("_ne_concentration", torch.zeros(self.principal_size, device=device))

        # =====================================================================
        # SEROTONIN RECEPTOR (DRN → BLA, 5-HT1A inhibitory)
        # =====================================================================
        # 5-HT1A (Gi-coupled) on principal neurons reduces excitability:
        # opens K+ channels → mild hyperpolarisation → gates fear extinction.
        # Ref: Gross et al. 2002 (5-HT1A KO → impaired extinction);
        #      Bhagya et al. 2015; Denny et al. 2014 (DRN→BLA extinction circuit)
        # 5-HT1A (Gi → GIRK): gates fear extinction, τ_decay=500 ms
        # (Gross et al. 2002; Bhagya et al. 2015).
        self.sht_receptor = make_neuromodulator_receptor(
            NMReceptorType.SHT_1A, n_receptors=self.principal_size, dt_ms=self.config.dt_ms, device=device
        )
        self._sht_concentration: torch.Tensor
        self.register_buffer("_sht_concentration", torch.zeros(self.principal_size, device=device))

        # Homeostasis for principal cells
        self._register_homeostasis(BLAPopulation.PRINCIPAL, self.principal_size, target_firing_rate=0.003, device=device)

        self._pv_spikes_prev = None
        self._som_spikes_prev = None

        # Ensure all tensors are on the correct device
        self.to(device)

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    def _step(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Compute BLA activity for one timestep.

        Args:
            synaptic_inputs: CS inputs (cortex, thalamus, hippocampus, PFC)
            neuromodulator_inputs: DA from VTA (US signal, gates plasticity)
        """
        device = self.device

        # =====================================================================
        # DA NEUROMODULATION (US/RPE signal)
        # =====================================================================
        # DA from VTA/SNc gates fear acquisition by boosting principal excitability
        da_signal = self._extract_neuromodulator(neuromodulator_inputs, NeuromodulatorChannel.DA_MESOLIMBIC)
        da_boost = 0.0
        if da_signal is not None:
            da_rate = da_signal.float().mean().item()
            da_boost = da_rate * 0.1  # Reduced DA gain (0.5→0.1): 20 Hz VTA was adding +0.01 continuous boost

        # NE (β-adrenergic): fear consolidation gain + STDP enhancement
        ne_spikes = self._extract_neuromodulator(neuromodulator_inputs, NeuromodulatorChannel.NE)
        self._ne_concentration = self.ne_receptor.update(ne_spikes)

        # 5-HT (5-HT1A inhibitory): extinction gating via excitability suppression
        sht_spikes = self._extract_neuromodulator(neuromodulator_inputs, NeuromodulatorChannel.SHT)
        self._sht_concentration = self.sht_receptor.update(sht_spikes)

        ne_level = self._ne_concentration.mean().item()
        sht_level = self._sht_concentration.mean().item()

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
        pv_principal_synapse = SynapseId(
            source_region=self.region_name,
            source_population=BLAPopulation.PV,
            target_region=self.region_name,
            target_population=BLAPopulation.PRINCIPAL,
            receptor_type=ReceptorType.GABA_A,
        )
        # SOM → PRINCIPAL: dendritic (apical) inhibition — slow, extinction gating
        # Biology: SOM/Martinotti cells target distal apical dendrites, reducing
        # excitatory integration of top-down and associative inputs during extinction
        som_principal_synapse = SynapseId(
            source_region=self.region_name,
            source_population=BLAPopulation.SOM,
            target_region=self.region_name,
            target_population=BLAPopulation.PRINCIPAL,
            receptor_type=ReceptorType.GABA_A,
        )

        # Separate PV and SOM inhibition so they map to the correct compartments
        pv_int_inh = self._integrate_synaptic_inputs_at_dendrites(
            {pv_principal_synapse: self._pv_spikes_prev} if self._pv_spikes_prev is not None else {},
            n_neurons=self.principal_size,
        )
        som_int_inh = self._integrate_synaptic_inputs_at_dendrites(
            {som_principal_synapse: self._som_spikes_prev} if self._som_spikes_prev is not None else {},
            n_neurons=self.principal_size,
        )

        # NE β-adrenergic: mild excitability boost during arousal (+30% at max NE)
        # 5-HT1A (Gi-coupled): hyperpolarising gate, reduces excitability (-35% at max 5-HT)
        ne_excitability = 1.0 + 0.3 * ne_level
        sht_excitability = max(0.0, 1.0 - 0.35 * sht_level)
        g_exc = (self.baseline_drive.clone() + dendrite_principal.g_ampa + da_boost) * ne_excitability * sht_excitability
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
        principal_pv_synapse = SynapseId(
            source_region=self.region_name,
            source_population=BLAPopulation.PRINCIPAL,
            target_region=self.region_name,
            target_population=BLAPopulation.PV,
            receptor_type=ReceptorType.AMPA,
        )
        int_principal_pv = self._integrate_synaptic_inputs_at_dendrites(
            {principal_pv_synapse: principal_spikes},
            n_neurons=self.pv_size,
        )

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

        principal_som_synapse = SynapseId(
            source_region=self.region_name,
            source_population=BLAPopulation.PRINCIPAL,
            target_region=self.region_name,
            target_population=BLAPopulation.SOM,
            receptor_type=ReceptorType.AMPA,
        )
        int_principal_som = self._integrate_synaptic_inputs_at_dendrites(
            {principal_som_synapse: principal_spikes},
            n_neurons=self.som_size,
        )

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
        if not GlobalConfig.LEARNING_DISABLED:
            # Register STDP strategy for any external synapse targeting PRINCIPAL
            # (SOM/PV targets are interneurons; their weights are fixed by design)
            for synapse_id in synaptic_inputs:
                if synapse_id.target_population == BLAPopulation.PRINCIPAL:
                    if self.get_learning_strategy(synapse_id) is None:
                        self._add_learning_strategy(synapse_id, self._stdp_strategy, device=device)

        # Cache inhibitory spike trains for next-step integration
        self._pv_spikes_prev = pv_spikes
        self._som_spikes_prev = som_spikes

        # =====================================================================
        # HOMEOSTASIS: Rate tracking + synaptic scaling for principal cells
        # =====================================================================
        self._apply_all_population_homeostasis(region_outputs)

        return region_outputs

    def _get_learning_kwargs(self, synapse_id: SynapseId) -> Dict[str, Any]:
        ne_level = self._ne_concentration.mean().item()
        sht_level = self._sht_concentration.mean().item()
        # NE β-adrenergic: enhances STDP magnitude during emotional arousal
        # Range: LR × [1.0, 2.5] (high NE during shock strongly boosts fear learning)
        # Ref: McGaugh 2004; Roozendaal et al. 2009
        # 5-HT1A: attenuates plasticity during extinction / safety learning
        # Range: × [0.5, 1.0] (makes extinction slower, consistent with biology)
        # Ref: Bhagya et al. 2015; Denny et al. 2014
        effective_lr = self.config.learning_rate * (1.0 + 1.5 * ne_level) * (1.0 - 0.5 * sht_level)
        return {"learning_rate": max(effective_lr, 0.0)}
