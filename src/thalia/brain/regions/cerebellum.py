"""
Cerebellum - Supervised Error-Corrective Learning for Precise Motor Control.

The cerebellum learns through supervised error signals from climbing fibers,
enabling fast, precise learning of input-output mappings without trial-and-error.
"""

from __future__ import annotations

from typing import Optional

import torch

from thalia.brain.configs import CerebellumConfig
from thalia.components import (
    ConductanceLIFConfig,
    ConductanceLIF,
    GapJunctionConfig,
    GapJunctionCoupling,
    NeuronFactory,
    NeuronType,
    STPConfig,
    STPType,
    WeightInitializer,
)
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
from thalia.utils import clamp_weights, split_excitatory_conductance

from .cerebellum_purkinje_layer import VectorizedPurkinjeLayer
from .neural_region import NeuralRegion
from .population_names import CerebellumPopulation
from .region_registry import register_region


@register_region(
    "cerebellum",
    description="Supervised error-corrective learning via climbing fiber error signals",
    version="1.0",
    author="Thalia Project",
    config_class=CerebellumConfig,
)
class Cerebellum(NeuralRegion[CerebellumConfig]):
    """Cerebellar region with supervised error-corrective learning.

    Architecture:
    - Granule cells: Internal only (parallel fibers → Purkinje dendrites)
    - Purkinje cells: Internal only (axons → DCN, inhibitory GABAergic)
    - DCN (deep cerebellar nuclei): Output neurons (axons → thalamus/motor)
    """

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def __init__(self, config: CerebellumConfig, population_sizes: PopulationSizes, region_name: RegionName):
        """Initialize cerebellum."""
        super().__init__(config, population_sizes, region_name)

        # =====================================================================
        # EXTRACT LAYER SIZES
        # =====================================================================
        self.dcn_size = population_sizes[CerebellumPopulation.DCN]
        self.granule_size = population_sizes[CerebellumPopulation.GRANULE]
        self.purkinje_size = population_sizes[CerebellumPopulation.PURKINJE]

        # =====================================================================
        # MOSSY FIBER LAYER (Pontine Nuclei equivalent)
        # =====================================================================
        # Biology: Cortex projects to pontine nuclei → mossy fibers → cerebellum
        # We model this as an intermediate representation layer
        # Size: Typically ~10% of granule cells (biological ratio)
        self.n_mossy = max(int(self.granule_size * 0.1), 50)  # At least 50 mossy fibers

        # =====================================================================
        # ENHANCED MICROCIRCUIT (granule-Purkinje-DCN)
        # =====================================================================
        # --------------------------------------------------------------------
        # Granule cell layer (sparse expansion from mossy fibers)
        # --------------------------------------------------------------------
        # Granule cells are small, fast-spiking neurons
        granule_config = ConductanceLIFConfig(
            region_name=self.region_name,
            population_name=CerebellumPopulation.GRANULE,
            device=self.device,
            v_threshold=0.85,  # Normalized: more excitable than pyramidal (1.0)
            v_reset=0.0,       # Normalized rest potential
            tau_mem=5.0,  # ms, faster than pyramidal (5ms vs 10-30ms)
            tau_E=2.5,  # ms, fast AMPA-like (biological minimum ~2-3ms)
            tau_I=6.0,  # ms, fast GABA_A (biological range 5-10ms)
        )
        self.granule_neurons = ConductanceLIF(
            n_neurons=self.granule_size,
            config=granule_config,
        )

        # --------------------------------------------------------------------
        # Purkinje cells (vectorized layer for efficiency)
        # --------------------------------------------------------------------
        # PERFORMANCE: Vectorized implementation ~50x faster than ModuleList
        self.purkinje_layer = VectorizedPurkinjeLayer(
            n_purkinje=self.purkinje_size,
            n_parallel_fibers=self.granule_size,
            n_dendrites=self.config.purkinje_n_dendrites,
            dt_ms=self.config.dt_ms,
            device=self.device,
        )

        # --------------------------------------------------------------------
        # Deep cerebellar nuclei (final output)
        # --------------------------------------------------------------------
        # Receives both Purkinje inhibition and mossy fiber collaterals

        # Purkinje → DCN (inhibitory, convergent)
        # Many Purkinje cells converge onto each DCN neuron
        # CONDUCTANCE-BASED: Strong inhibition to sculpt DCN output
        self._add_internal_connection(
            source_population=CerebellumPopulation.PURKINJE,
            target_population=CerebellumPopulation.DCN,
            weights=WeightInitializer.sparse_random(
                n_input=self.purkinje_size,
                n_output=self.dcn_size,
                connectivity=0.2,
                weight_scale=0.0015,
                device=self.device,
            ),
            # Biology: Strong Purkinje inhibition shows depression, preventing runaway inhibition
            stp_config=STPConfig.from_type(STPType.DEPRESSING),
            receptor_type=ReceptorType.GABA_A,
        )

        # Mossy fiber → DCN (excitatory collaterals)
        # Mossy fibers send collaterals to DCN before reaching granule cells
        # CONDUCTANCE-BASED: Moderate excitation (Purkinje sculpts final output)
        # Routing key: dcn:cerebellum:mossy (target:region:internal_source)
        self._add_internal_connection(
            source_population=CerebellumPopulation.MOSSY,
            target_population=CerebellumPopulation.DCN,
            weights=WeightInitializer.sparse_random(
                n_input=self.n_mossy,
                n_output=self.dcn_size,
                connectivity=0.1,
                weight_scale=0.0005,
                device=self.device,
            ),
            # Biology: Mossy fiber collaterals show facilitation, reinforcing sustained input
            stp_config=STPConfig.from_type(STPType.FACILITATING),
            receptor_type=ReceptorType.AMPA,
        )

        # DCN neurons (tonic firing, modulated by inhibition)
        # DCN neurons are spontaneously active (tonic firing ~40-60 Hz)
        # Use NORMALIZED units (threshold=1.0 scale) NOT absolute millivolts
        dcn_config = ConductanceLIFConfig(
            region_name=self.region_name,
            population_name=CerebellumPopulation.DCN,
            device=self.device,
            v_threshold=1.0,  # Standard normalized threshold
            v_reset=0.0,  # Reset to rest
            v_rest=0.0,  # Resting potential (normalized)
            E_L=0.0,  # Leak reversal (normalized)
            E_E=3.0,  # Excitatory reversal (normalized, above threshold)
            E_I=-0.5,  # Inhibitory reversal (normalized, hyperpolarizing)
            g_L=0.10,  # Moderate leak conductance
            tau_mem=20.0,  # ms, moderate integration
            tau_E=4.0,  # ms, AMPA kinetics (biological range 2-5ms)
            tau_I=10.0,  # ms, GABA_A kinetics (biological range 5-10ms)
            tau_ref=12.0,  # ms, refractory period (max ~83 Hz ceiling, allows biological 40-60 Hz range)
            noise_std=0.007 if config.baseline_noise_conductance_enabled else 0.0,  # Membrane voltage noise
        )
        self.dcn_neurons = ConductanceLIF(n_neurons=self.dcn_size, config=dcn_config)

        # Initialize DCN neuron membrane potentials with heterogeneous values
        # This prevents all neurons starting at same potential and synchronizing
        dcn_v_init = torch.normal(
            mean=0.0,  # Around rest (normalized units)
            std=0.1,  # Moderate spread
            size=(self.dcn_size,),
            device=self.device,
        ).clamp(min=-0.2, max=0.5)  # Subthreshold range (normalized units)
        self.dcn_neurons.membrane = dcn_v_init.clone()

        # IO membrane potential for gap junction coupling
        self._io_membrane: Optional[torch.Tensor] = None

        # =====================================================================
        # HOMEOSTATIC INTRINSIC PLASTICITY (Adaptive Gain)
        # =====================================================================
        # EMA tracking of Purkinje firing rates (reserved for future use).
        self._register_homeostasis(CerebellumPopulation.PURKINJE, self.purkinje_size)

        # Adaptive gains (per neuron)
        self.gain = torch.ones(self.purkinje_size, device=self.device)

        # =====================================================================
        # INFERIOR OLIVE NEURONS (Error Signal Generation)
        # =====================================================================
        # One IO neuron per Purkinje cell (1:1 mapping)
        # IO neurons generate climbing fiber signals (complex spikes) that teach
        # Purkinje cells. Dense gap junction coupling synchronizes IO activity
        # across related cerebellar modules.
        self.io_neurons = NeuronFactory.create(
            region_name=self.region_name,
            population_name=CerebellumPopulation.INFERIOR_OLIVE,
            neuron_type=NeuronType.RELAY,  # Relay neurons have tonic firing suitable for IO baseline activity
            n_neurons=self.purkinje_size,
            device=self.device,
        )

        # =====================================================================
        # GAP JUNCTIONS (Inferior Olive Synchronization)
        # =====================================================================
        # IO neurons are densely coupled via gap junctions, creating synchronized
        # complex spikes across multiple Purkinje cells. This is critical for
        # coordinated motor learning across cerebellar modules.
        #
        # Biology: IO forms one of the densest gap junction networks in the brain
        # - Strong coupling (0.18) for population synchronization
        # - Dense connectivity (~80% of nearby neurons)
        # - Synchronization time: <1ms

        gap_junctions_config = GapJunctionConfig(
            coupling_strength=config.gap_junction_strength,
            connectivity_threshold=config.gap_junction_threshold,
            max_neighbors=config.gap_junction_max_neighbors,
            interneuron_only=False,  # IO neurons are projection neurons, not interneurons
        )

        # Initialize gap junctions using Purkinje dendritic weights as connectivity pattern
        # IO neurons corresponding to Purkinje cells with similar parallel fiber inputs
        # are anatomically close and should be electrically coupled
        self.gap_junctions_io = GapJunctionCoupling(
            n_neurons=self.purkinje_size,
            afferent_weights=self.purkinje_layer.synaptic_weights,  # Use parallel fiber connectivity
            config=gap_junctions_config,
            interneuron_mask=None,  # All IO neurons can be coupled
            device=self.device,
        )

        # =====================================================================
        # REGISTER NEURON POPULATIONS
        # =====================================================================
        self._register_neuron_population(CerebellumPopulation.DCN, self.dcn_neurons, polarity=PopulationPolarity.EXCITATORY)
        self._register_neuron_population(CerebellumPopulation.GRANULE, self.granule_neurons, polarity=PopulationPolarity.EXCITATORY)
        self._register_neuron_population(CerebellumPopulation.INFERIOR_OLIVE, self.io_neurons, polarity=PopulationPolarity.EXCITATORY)
        self._register_neuron_population(CerebellumPopulation.PURKINJE, self.purkinje_layer.soma_neurons, polarity=PopulationPolarity.INHIBITORY)

        # Ensure all tensors are on the correct device
        self.to(self.device)

    # =========================================================================
    # FORWARD PASS
    # =========================================================================

    @torch.no_grad()
    def forward(self, synaptic_inputs: SynapticInput, neuromodulator_inputs: NeuromodulatorInput) -> RegionOutput:
        """Process input through cerebellar circuit.

        Note: neuromodulator_inputs is not used - cerebellum is a neuromodulator source region.
        """
        self._pre_forward(synaptic_inputs, neuromodulator_inputs)

        cfg = self.config

        # =====================================================================
        # MULTI-SOURCE SYNAPTIC INTEGRATION
        # =====================================================================
        # Biology: Different sources (cortex, spinal, brainstem) project to pontine
        # nuclei, which give rise to mossy fibers. We model this as a integration stage.
        # NOTE: Weights project directly to granule layer (granule_size), not to mossy
        # fiber intermediates. The granule layer handles internal expansion/sparsification.
        mossy_fiber_conductances = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.granule_size,
            filter_by_target_population=CerebellumPopulation.GRANULE,
        ).g_ampa

        # Add baseline noise for spontaneous activity (biology: granule cells have ~5Hz baseline)
        # Noise represents stochastic miniature EPSPs from spontaneous vesicle release (conductance, not current)
        if cfg.baseline_noise_conductance_enabled:
            noise = torch.randn(self.granule_size, device=self.device) * 0.007
            mossy_fiber_conductances = mossy_fiber_conductances + noise

        g_ampa, g_nmda = split_excitatory_conductance(mossy_fiber_conductances, nmda_ratio=0.3)

        granule_spikes, _ = self.granule_neurons.forward(
            g_ampa_input=ConductanceTensor(g_ampa),
            g_nmda_input=ConductanceTensor(g_nmda),
            g_gaba_a_input=None,  # TODO: No inhibition for now (future: Golgi cells)
            g_gaba_b_input=None,
        )

        # Enforce sparsity (top-k activation) - cerebellar granule cells are VERY sparse (~3%)
        k = int(self.granule_size * cfg.granule_connectivity)  # Target number of active neurons
        n_spiking = granule_spikes.sum().item()
        if n_spiking > k:
            # More neurons spiked than target, select top-k by excitation
            # Only consider neurons that actually spiked
            spiking_mask = granule_spikes.bool()
            g_exc_spiking = mossy_fiber_conductances.clone()
            g_exc_spiking[~spiking_mask] = -float("inf")  # Exclude non-spiking
            _, top_k_idx = torch.topk(g_exc_spiking, k)
            sparse_spikes = torch.zeros_like(granule_spikes, dtype=torch.bool)
            sparse_spikes[top_k_idx] = True
            granule_spikes = sparse_spikes

        # Create mossy fiber approximation by downsampling granule spikes to mossy fiber dimensionality
        # TODO: In the full implementation, mossy fiber collaterals would provide direct input to DCN,
        # separate from granule layer. For now, we use a downsampled version of granule spikes as a proxy
        # for mossy fiber state.
        step = self.granule_size // self.n_mossy
        mossy_spikes = granule_spikes[::step][:self.n_mossy]  # [n_mossy]

        # =====================================================================
        # STAGE 2: INFERIOR OLIVE → CLIMBING FIBERS (Error Signal)
        # =====================================================================
        # IO neurons generate climbing fiber spikes, synchronized via gap junctions
        # Error signal drives IO activity (high error → depolarization → spike)
        # For now, use spontaneous activity (~1 Hz baseline)
        # TODO: Add error-driven input based on motor prediction errors

        # Apply gap junction coupling to IO neurons (if they have been initialized)
        if self.io_neurons.membrane is not None:
            g_gap_io, E_gap_io = self.gap_junctions_io.forward(self.io_neurons.membrane)

            # Define additional conductances hook for IO neurons
            def io_get_additional_conductances():
                return [(g_gap_io, E_gap_io)]

            self.io_neurons._get_additional_conductances = io_get_additional_conductances

        # Update IO neurons (spontaneous activity + gap junction synchronization)
        # Integrate any external error-signal inputs projecting to INFERIOR_OLIVE,
        # then add a small baseline depolarizing conductance for ~1 Hz spontaneous firing.
        io_external = self._integrate_synaptic_inputs_at_dendrites(
            synaptic_inputs,
            n_neurons=self.purkinje_size,
            filter_by_target_population=CerebellumPopulation.INFERIOR_OLIVE,
        )
        io_exc = io_external.g_ampa  # [purkinje_size] — external error drive
        if cfg.baseline_noise_conductance_enabled:
            baseline_drive = 0.004  # Small baseline conductance for ~1 Hz spontaneous firing
            io_g_ampa = ConductanceTensor(io_exc + baseline_drive)
        else:
            io_g_ampa = ConductanceTensor(io_exc) if io_exc.any() else None

        climbing_fiber_spikes, _ = self.io_neurons.forward(
            g_ampa_input=io_g_ampa,
            g_nmda_input=None,
            g_gaba_a_input=None,
            g_gaba_b_input=None,
        )

        # =====================================================================
        # STAGE 3: GRANULE CELLS → PURKINJE CELLS (Parallel Fibers)
        # =====================================================================
        # Process all Purkinje cells in parallel (vectorized)
        # Climbing fiber spikes from IO neurons trigger complex spikes
        purkinje_simple_spikes, _purkinje_complex_spikes = self.purkinje_layer.forward(
            parallel_fiber_input=granule_spikes,
            climbing_fiber_active=climbing_fiber_spikes,
        )  # [purkinje_size]

        # =====================================================================
        # STAGE 4: PURKINJE + MOSSY COLLATERALS → DCN (Final Output)
        # =====================================================================
        # Biology: DCN receives both Purkinje inhibition and mossy fiber collaterals

        purkinje_synapse = SynapseId(
            source_region=self.region_name,
            source_population=CerebellumPopulation.PURKINJE,
            target_region=self.region_name,
            target_population=CerebellumPopulation.DCN,
            receptor_type=ReceptorType.GABA_A,
        )
        purkinje_conductance = self._integrate_synaptic_inputs_at_dendrites({purkinje_synapse: purkinje_simple_spikes}, n_neurons=self.dcn_size).g_gaba_a

        mossy_synapse = SynapseId(
            source_region=self.region_name,
            source_population=CerebellumPopulation.MOSSY,
            target_region=self.region_name,
            target_population=CerebellumPopulation.DCN,
            receptor_type=ReceptorType.AMPA,
        )
        dcn_total_excitation = self._integrate_synaptic_inputs_at_dendrites({mossy_synapse: mossy_spikes}, n_neurons=self.dcn_size).g_ampa

        # DCN spiking: Excitation (mossy + tonic) vs Inhibition (Purkinje)
        # Purkinje provides GABAergic shunting inhibition
        dcn_total_g_ampa, dcn_total_g_nmda = split_excitatory_conductance(dcn_total_excitation, nmda_ratio=0.3)

        dcn_spikes, _ = self.dcn_neurons.forward(
            g_ampa_input=ConductanceTensor(dcn_total_g_ampa),
            g_nmda_input=ConductanceTensor(dcn_total_g_nmda),
            g_gaba_a_input=ConductanceTensor(purkinje_conductance),
            g_gaba_b_input=None,
        )

        # ======================================================================
        # APPLY CEREBELLAR LEARNING: Marr-Albus-Ito Rule
        # ======================================================================
        # Biology (Ito 1989):
        #   - Climbing fiber (CF) fires → LTD at co-active parallel fiber (PF) synapses
        #   - PF active, CF silent  → slow normalizing LTP
        #
        # Computation:
        #   dw = ltp_rate × outer(1−cf, pf) − ltd_rate × outer(cf, pf)
        # Shapes: weights [n_purkinje, n_granule]
        pre_f = granule_spikes.float()  # [n_granule]
        cf_f = climbing_fiber_spikes.float()  # [n_purkinje]

        ltd_dw = cfg.mai_ltd_rate * torch.outer(cf_f, pre_f)
        ltp_dw = cfg.mai_ltp_rate * torch.outer((1.0 - cf_f).clamp(min=0.0), pre_f)
        mai_delta = ltp_dw - ltd_dw  # [n_purkinje, n_granule]

        self.purkinje_layer.synaptic_weights.data = clamp_weights(
            weights=self.purkinje_layer.synaptic_weights.data + mai_delta,
            w_min=cfg.w_min,
            w_max=cfg.w_max,
            inplace=False,
        )

        region_outputs: RegionOutput = {
            CerebellumPopulation.DCN: dcn_spikes,
            CerebellumPopulation.GRANULE: granule_spikes,
            CerebellumPopulation.INFERIOR_OLIVE: climbing_fiber_spikes,
            CerebellumPopulation.PURKINJE: purkinje_simple_spikes,
        }

        return self._post_forward(region_outputs)

    # =========================================================================
    # TEMPORAL PARAMETER MANAGEMENT
    # =========================================================================

    def update_temporal_parameters(self, dt_ms: float) -> None:
        """Update temporal parameters when brain timestep changes.

        Propagates dt update to neurons, STP components, and learning strategies.

        Args:
            dt_ms: New simulation timestep in milliseconds
        """
        super().update_temporal_parameters(dt_ms)
        self.granule_neurons.update_temporal_parameters(dt_ms)
        self.purkinje_layer.update_temporal_parameters(dt_ms)
        self.dcn_neurons.update_temporal_parameters(dt_ms)
