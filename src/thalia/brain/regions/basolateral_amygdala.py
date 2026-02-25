"""Basolateral Amygdala (BLA) - Fear Conditioning and Extinction.

The BLA is the principal site of fear learning. It receives convergent CS
(conditioned stimulus) and US (unconditioned stimulus) inputs and forms
hebbian CS–US associations via STDP-like plasticity at principal neuron synapses.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List

import torch

from thalia import GlobalConfig
from thalia.brain.configs import BasolateralAmygdalaConfig
from thalia.components import (
    ConductanceLIF,
    ConductanceLIFConfig,
    NeuronFactory,
    NeuronType,
    TwoCompartmentLIF,
    TwoCompartmentLIFConfig,
    WeightInitializer,
)
from thalia.components.synapses.stp import STPConfig, STPType
from thalia.learning import STDPConfig, STDPStrategy
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
from .population_names import BLAPopulation
from .region_registry import register_region


@register_region(
    "basolateral_amygdala",
    aliases=["bla", "amygdala_bla"],
    description="Basolateral amygdala — fear conditioning and extinction via CS-US association",
    version="1.0",
    author="Thalia Project",
    config_class=BasolateralAmygdalaConfig,
)
class BasolateralAmygdala(NeuralRegion[BasolateralAmygdalaConfig]):
    """Basolateral Amygdala: fear conditioning and extinction nucleus.

    Contains:
    - Principal neurons (glutamatergic): form CS–US associations via STDP
    - PV interneurons: fast feedforward inhibition (fear discrimination)
    - SOM interneurons: slow dendritic inhibition (extinction gating)
    """

    # Mesolimbic DA (VTA → BLA) modulates fear conditioning plasticity.
    # Higher DA during CS-US pairing enhances STDP (three-factor gating, Phase 2).
    neuromodulator_subscriptions: ClassVar[List[str]] = ['da_mesolimbic']

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, config: BasolateralAmygdalaConfig, population_sizes: PopulationSizes, region_name: RegionName):
        """Initialize BLA populations and internal connectivity."""
        super().__init__(config, population_sizes, region_name)

        self.principal_size = population_sizes[BLAPopulation.PRINCIPAL]
        self.pv_size = population_sizes[BLAPopulation.PV]
        self.som_size = population_sizes[BLAPopulation.SOM]

        # =====================================================================
        # NEURON POPULATIONS
        # =====================================================================

        # Principal neurons: slow, STDP-plastic, CS-US association
        # Two-compartment: basal (proximal) receives thalamic/cortical CS input;
        # apical (distal) receives SOM dendritic inhibition for extinction gating.
        self.principal_neurons = TwoCompartmentLIF(
            n_neurons=self.principal_size,
            config=TwoCompartmentLIFConfig(
                region_name=self.region_name,
                population_name=BLAPopulation.PRINCIPAL,
                tau_mem=config.tau_mem_principal,
                v_threshold=config.v_threshold_principal,
                v_reset=0.0,
                tau_ref=config.tau_ref,
                g_L=0.05,
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                tau_E=7.0,      # Moderate AMPA (slightly slower for integration)
                tau_I=12.0,     # GABA_A for PV/SOM inhibition
                adapt_increment=0.30,  # Strong adaptation — prevents runaway at 54 Hz; equivalent to Ca²⁺-activated K⁺
                tau_adapt=200.0,       # Slower decay (200 ms) to maintain adaptation during burst
                noise_std=0.02,        # Reduced noise; high noise was driving extra spikes
                # Two-compartment dendritic parameters
                g_c=0.05,          # Somato-dendritic coupling conductance
                C_d=0.5,           # Dendritic membrane capacitance
                g_L_d=0.03,        # Dendritic leak conductance
                bap_amplitude=0.3, # Back-propagating AP strength (enables STDP coincidence)
                theta_Ca=2.0,      # Ca2+ spike threshold (dendritic burst)
                g_Ca_spike=0.30,   # Ca2+ spike conductance
                tau_Ca_ms=20.0,    # Ca2+ decay time (synaptogenesis window)
            ),
            device=self.device,
        )

        # PV interneurons: fast-spiking, feedforward fear gating
        self.pv_neurons = NeuronFactory.create(
            region_name=self.region_name,
            population_name=BLAPopulation.PV,
            neuron_type=NeuronType.FAST_SPIKING,
            n_neurons=self.pv_size,
            device=self.device,
        )

        # SOM interneurons: slower, dendritic inhibition, extinction gating
        # Modelled as standard neurons with longer time constants
        self.som_neurons = ConductanceLIF(
            n_neurons=self.som_size,
            config=ConductanceLIFConfig(
                region_name=self.region_name,
                population_name=BLAPopulation.SOM,
                tau_mem=config.tau_mem_som,
                v_threshold=1.1,   # Slightly harder to recruit
                v_reset=0.0,
                tau_ref=5.0,
                g_L=0.06,
                E_L=0.0,
                E_E=3.0,
                E_I=-0.5,
                tau_E=8.0,
                tau_I=15.0,
                adapt_increment=0.1,
                tau_adapt=200.0,
                noise_std=0.03,
            ),
            device=self.device,
        )

        # =====================================================================
        # INTERNAL CONNECTIVITY
        # =====================================================================

        # Principal → PV (feedforward excitation of PV → rapid feedback inhibition)
        # Biology: Principal axon collaterals excite local PV interneurons
        # weight_scale reduced 4× (0.002→0.0005): principal at 2.77% was driving PV to 18%
        self._add_internal_connection(
            source_population=BLAPopulation.PRINCIPAL,
            target_population=BLAPopulation.PV,
            weights=WeightInitializer.sparse_random(
                n_input=self.principal_size,
                n_output=self.pv_size,
                connectivity=0.4,
                weight_scale=0.0005,
                device=self.device,
            ),
            receptor_type=ReceptorType.AMPA,
            stp_config=STPConfig.from_type(STPType.DEPRESSING),
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
                device=self.device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=STPConfig.from_type(STPType.DEPRESSING),
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
                device=self.device,
            ),
            receptor_type=ReceptorType.AMPA,
            stp_config=STPConfig.from_type(STPType.FACILITATING_MODERATE),
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
                device=self.device,
            ),
            receptor_type=ReceptorType.GABA_A,
            stp_config=STPConfig.from_type(STPType.DEPRESSING),
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
            device=str(self.device),
        ))

        # Baseline drive tensor (tonic low-level activity)
        self.baseline_drive = torch.full(
            (self.principal_size,), config.baseline_drive, device=self.device
        )

        # =====================================================================
        # REGISTER POPULATIONS
        # =====================================================================
        self._register_neuron_population(BLAPopulation.PRINCIPAL, self.principal_neurons, polarity=PopulationPolarity.EXCITATORY)
        self._register_neuron_population(BLAPopulation.PV, self.pv_neurons, polarity=PopulationPolarity.INHIBITORY)
        self._register_neuron_population(BLAPopulation.SOM, self.som_neurons, polarity=PopulationPolarity.INHIBITORY)

        # Homeostasis for principal cells: diagnostics showed g_L_scale stuck at 1.000
        # (never adapted) while principal fired at 27 Hz despite target 1-5 Hz.
        # The missing registration means the homeostatic controller was never created;
        # all other pyramidal populations have this call (see cortical_column.py).
        self._register_homeostasis(BLAPopulation.PRINCIPAL, self.principal_size, target_firing_rate=0.003)

        # Ensure all tensors are on the correct device
        self.to(self.device)

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    @torch.no_grad()
    def forward(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Compute BLA activity for one timestep.

        Args:
            synaptic_inputs: CS inputs (cortex, thalamus, hippocampus, PFC)
            neuromodulator_inputs: DA from VTA (US signal, gates plasticity)
        """
        self._pre_forward(synaptic_inputs, neuromodulator_inputs)

        # =====================================================================
        # DA NEUROMODULATION (US/RPE signal)
        # =====================================================================
        # DA from VTA/SNc gates fear acquisition by boosting principal excitability
        da_signal = self._extract_neuromodulator(neuromodulator_inputs, 'da_mesolimbic')
        da_boost = 0.0
        if da_signal is not None:
            da_rate = da_signal.float().mean().item()
            da_boost = da_rate * 0.1  # Reduced DA gain (0.5→0.1): 20 Hz VTA was adding +0.01 continuous boost

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
            {pv_principal_synapse: self._pv_spikes_prev}
            if hasattr(self, '_pv_spikes_prev')
            else {},
            n_neurons=self.principal_size,
        )
        som_int_inh = self._integrate_synaptic_inputs_at_dendrites(
            {som_principal_synapse: self._som_spikes_prev}
            if hasattr(self, '_som_spikes_prev')
            else {},
            n_neurons=self.principal_size,
        )

        g_exc = self.baseline_drive.clone() + dendrite_principal.g_ampa + da_boost
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
                        self._add_learning_strategy(synapse_id, self._stdp_strategy)

        # Cache inhibitory spike trains for next-step integration
        self._pv_spikes_prev = pv_spikes
        self._som_spikes_prev = som_spikes

        return self._post_forward(region_outputs)

    def _get_learning_kwargs(self, synapse_id: SynapseId) -> Dict[str, Any]:
        return {"learning_rate": self.config.learning_rate}

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Propagate temporal parameter update to neuron populations."""
        super().update_temporal_parameters(dt_ms)
        self.principal_neurons.update_temporal_parameters(dt_ms)
        self.pv_neurons.update_temporal_parameters(dt_ms)
        self.som_neurons.update_temporal_parameters(dt_ms)
