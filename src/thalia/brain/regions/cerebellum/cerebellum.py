"""
Cerebellum - Supervised Error-Corrective Learning for Precise Motor Control.

The cerebellum learns through supervised error signals from climbing fibers,
enabling fast, precise learning of input-output mappings without trial-and-error.
"""

from __future__ import annotations

from typing import Optional

import torch

from thalia.brain.configs import CerebellumConfig
from thalia.brain.regions.population_names import CerebellumPopulation
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
from thalia.learning import EligibilityTraceManager, STDPConfig
from thalia.typing import (
    NeuromodulatorInput,
    PopulationSizes,
    RegionName,
    RegionOutput,
    SynapseId,
    SynapticInput,
)
from thalia.units import ConductanceTensor
from thalia.utils import clamp_weights

from .purkinje_layer import VectorizedPurkinjeLayer

from ..neural_region import NeuralRegion
from ..region_registry import register_region


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
        self.granule_size = population_sizes[CerebellumPopulation.GRANULE.value]
        self.purkinje_size = population_sizes[CerebellumPopulation.PURKINJE.value]
        self.dcn_size = population_sizes[CerebellumPopulation.DCN.value]

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
            population_name=CerebellumPopulation.GRANULE.value,
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
            source_population=CerebellumPopulation.PURKINJE.value,
            target_population=CerebellumPopulation.DCN.value,
            weights=WeightInitializer.sparse_random(
                n_input=self.purkinje_size,
                n_output=self.dcn_size,
                connectivity=0.2,
                weight_scale=0.0015,
                device=self.device,
            ),
            # Biology: Strong Purkinje inhibition shows depression, preventing runaway inhibition
            stp_config=STPConfig.from_type(STPType.DEPRESSING),
            is_inhibitory=True,
        )

        # Mossy fiber → DCN (excitatory collaterals)
        # Mossy fibers send collaterals to DCN before reaching granule cells
        # CONDUCTANCE-BASED: Moderate excitation (Purkinje sculpts final output)
        # Routing key: dcn:cerebellum:mossy (target:region:internal_source)
        self._add_internal_connection(
            source_population=CerebellumPopulation.MOSSY.value,
            target_population=CerebellumPopulation.DCN.value,
            weights=WeightInitializer.sparse_random(
                n_input=self.n_mossy,
                n_output=self.dcn_size,
                connectivity=0.1,
                weight_scale=0.0005,
                device=self.device,
            ),
            # Biology: Mossy fiber collaterals show facilitation, reinforcing sustained input
            stp_config=STPConfig.from_type(STPType.FACILITATING),
            is_inhibitory=False,
        )

        # DCN neurons (tonic firing, modulated by inhibition)
        # DCN neurons are spontaneously active (tonic firing ~40-60 Hz)
        # Use NORMALIZED units (threshold=1.0 scale) NOT absolute millivolts
        dcn_config = ConductanceLIFConfig(
            region_name=self.region_name,
            population_name=CerebellumPopulation.DCN.value,
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

        # =====================================================================
        # ELIGIBILITY TRACE MANAGER for STDP
        # =====================================================================
        self._trace_manager = EligibilityTraceManager(
            n_input=self.granule_size,
            n_output=self.purkinje_size,
            config=STDPConfig(
                learning_rate=self.config.learning_rate,
                w_min=config.w_min,
                w_max=config.w_max,
                a_plus=1.0,
                a_minus=self.config.heterosynaptic_ratio,
                tau_plus=self.config.tau_plus_ms,
                tau_minus=self.config.tau_plus_ms,  # Use same tau for simplicity
                eligibility_tau_ms=self.config.eligibility_tau_ms,
                heterosynaptic_ratio=self.config.heterosynaptic_ratio,
            ),
            device=self.device,
        )

        # IO membrane potential for gap junction coupling
        self._io_membrane: Optional[torch.Tensor] = None

        # =====================================================================
        # HOMEOSTATIC INTRINSIC PLASTICITY (Adaptive Gain)
        # =====================================================================
        # EMA tracking of firing rates
        self.register_buffer("firing_rate", torch.zeros(self.purkinje_size, device=self.device))

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
            population_name=CerebellumPopulation.INFERIOR_OLIVE.value,
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
        self._register_neuron_population(CerebellumPopulation.INFERIOR_OLIVE.value, self.io_neurons)
        self._register_neuron_population(CerebellumPopulation.GRANULE.value, self.granule_neurons)
        self._register_neuron_population(CerebellumPopulation.PURKINJE.value, self.purkinje_layer.soma_neurons)
        self._register_neuron_population(CerebellumPopulation.DCN.value, self.dcn_neurons)

        # =====================================================================
        # POST-INITIALIZATION
        # =====================================================================
        self.__post_init__()

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
        mossy_fiber_conductances = self._integrate_synaptic_inputs_at_dendrites(synaptic_inputs, n_neurons=self.granule_size).g_exc

        # Add baseline noise for spontaneous activity (biology: granule cells have ~5Hz baseline)
        # Noise represents stochastic miniature EPSPs from spontaneous vesicle release (conductance, not current)
        if cfg.baseline_noise_conductance_enabled:
            noise = torch.randn(self.granule_size, device=self.device) * 0.007
            mossy_fiber_conductances = mossy_fiber_conductances + noise

        # Pass conductances directly to granule cell neurons
        # Split excitatory conductance: 70% AMPA (fast), 30% NMDA (slow)
        g_ampa = mossy_fiber_conductances * 0.7
        g_nmda = mossy_fiber_conductances * 0.3

        granule_spikes, _ = self.granule_neurons.forward(
            g_ampa_input=ConductanceTensor(g_ampa),
            g_gaba_a_input=None,  # TODO: No inhibition for now (future: Golgi cells)
            g_nmda_input=ConductanceTensor(g_nmda),
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
        # Add baseline depolarizing conductance to maintain ~1 Hz spontaneous firing
        if cfg.baseline_noise_conductance_enabled:
            baseline_conductance = ConductanceTensor(torch.full(
                (self.purkinje_size,), 0.004, device=self.device  # Small baseline conductance for ~1 Hz
            ))
        else:
            baseline_conductance = None

        climbing_fiber_spikes, _ = self.io_neurons.forward(
            g_ampa_input=baseline_conductance,  # Baseline depolarization for spontaneous activity (or None)
            g_gaba_a_input=None,
            g_nmda_input=None,
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
        # Apply STP and compute conductances for internal connections

        # Apply STP to Purkinje → DCN (depressing)
        # Pattern: efficacy = stp.forward(spikes), effective_weights = weights * efficacy.T
        purkinje_synapse = SynapseId(
            source_region=self.region_name,
            source_population=CerebellumPopulation.PURKINJE.value,
            target_region=self.region_name,
            target_population=CerebellumPopulation.DCN.value,
            is_inhibitory=True,  # Purkinje cells are GABAergic (inhibitory)
        )
        purkinje_conductance = self._integrate_synaptic_inputs_at_dendrites({purkinje_synapse: purkinje_simple_spikes}, n_neurons=self.dcn_size).g_inh

        # Apply STP to Mossy → DCN (facilitating)
        mossy_synapse = SynapseId(
            source_region=self.region_name,
            source_population=CerebellumPopulation.MOSSY.value,
            target_region=self.region_name,
            target_population=CerebellumPopulation.DCN.value,
        )
        dcn_total_excitation = self._integrate_synaptic_inputs_at_dendrites({mossy_synapse: mossy_spikes}, n_neurons=self.dcn_size).g_exc

        # DCN spiking: Excitation (mossy + tonic) vs Inhibition (Purkinje)
        # Purkinje provides GABAergic shunting inhibition
        # Split excitatory conductance: 70% AMPA (fast), 30% NMDA (slow)
        dcn_total_g_ampa = dcn_total_excitation * 0.7
        dcn_total_g_nmda = dcn_total_excitation * 0.3

        dcn_spikes, _ = self.dcn_neurons.forward(
            g_ampa_input=ConductanceTensor(dcn_total_g_ampa),
            g_gaba_a_input=ConductanceTensor(purkinje_conductance),  # Purkinje inhibition
            g_nmda_input=ConductanceTensor(dcn_total_g_nmda),
        )

        # ======================================================================
        # Update STDP eligibility using trace manager
        # ======================================================================
        # For learning: use granule spikes as effective input
        effective_input = granule_spikes

        # Use trace manager for consolidated STDP computation
        self._trace_manager.update_traces(
            input_spikes=effective_input,  # Use granule spikes if enhanced
            output_spikes=dcn_spikes,
            dt_ms=self.dt_ms,
        )

        # Compute STDP weight change direction (raw LTP/LTD without combining)
        ltp, ltd = self._trace_manager.compute_ltp_ltd_separate(
            input_spikes=effective_input,
            output_spikes=dcn_spikes,
        )

        # Combine LTP and LTD with learning rate and heterosynaptic ratio
        stdp_dw = cfg.learning_rate * (ltp - cfg.heterosynaptic_ratio * ltd)

        # Accumulate into eligibility trace (with decay)
        if isinstance(stdp_dw, torch.Tensor):
            self._trace_manager.accumulate_eligibility(stdp_dw, dt_ms=self.dt_ms)

        # ======================================================================
        # APPLY CEREBELLAR LEARNING (Parallel Fiber → Purkinje)
        # ======================================================================
        # Biology: Climbing fiber error signals gate the application of parallel
        # fiber eligibility traces. When climbing fiber fires (error detected),
        # active parallel fibers undergo LTD. When no climbing fiber (correct),
        # active parallel fibers undergo LTP.
        #
        # Implementation: We use the accumulated eligibility traces and apply them
        # to the Purkinje layer's dendritic weights (vectorized).
        # TODO: In the full implementation, climbing fiber error would gate
        # these updates (error × eligibility).
        # For now, we apply the eligibility-based STDP learning.

        # Get eligibility from trace manager
        eligibility = self._trace_manager.eligibility  # [n_purkinje, n_granule]

        # Apply weight updates to vectorized Purkinje layer
        # eligibility shape: [n_purkinje, n_granule]
        # synaptic_weights shape: [n_purkinje, n_parallel_fibers]
        # Since n_parallel_fibers == n_granule, they should match directly

        # Apply update with weight bounds
        # Biology: Parallel fiber synapses have limited dynamic range
        self.purkinje_layer.synaptic_weights.data = clamp_weights(
            weights=self.purkinje_layer.synaptic_weights.data + eligibility,
            w_min=cfg.w_min,
            w_max=cfg.w_max,
            inplace=False,
        )

        region_outputs: RegionOutput = {
            CerebellumPopulation.DCN.value: dcn_spikes,
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
