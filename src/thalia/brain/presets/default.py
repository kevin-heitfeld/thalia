"""Default biologically realistic brain preset.

Implements the full biologically-grounded default brain architecture decomposed
into one helper function per anatomical circuit.  ``build`` orchestrates
these helpers and is registered as the ``"default"`` preset in ``brain_builder.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from thalia.brain.configs import (
    BasolateralAmygdalaConfig,
    CentralAmygdalaConfig,
    CerebellumConfig,
    CorticalColumnConfig,
    CorticalPopulationConfig,
    DopaminePacemakerConfig,
    DorsalRapheNucleusConfig,
    HomeostaticGainConfig,
    HomeostaticThresholdConfig,
    LayerFractions,
    LocusCoeruleusConfig,
    NMReceptorConfig,
    NeuralRegionConfig,
    ThalamusConfig,
    VTAConfig,
)
from thalia.brain.presets.amygdala import connect_amygdala
from thalia.brain.presets.basal_ganglia import add_basal_ganglia_regions, add_basal_ganglia_circuit
from thalia.brain.presets.cerebellum import connect_cerebellum
from thalia.brain.presets.cortical import connect_corticocortical, connect_prefrontal, connect_thalamocortical
from thalia.brain.presets.medial_temporal_lobe import add_medial_temporal_lobe_regions, add_medial_temporal_lobe_circuit
from thalia.brain.presets.neuromodulators import connect_neuromodulators
from thalia.brain.regions.population_names import (
    BLAPopulation,
    CeAPopulation,
    CerebellumPopulation,
    CortexPopulation,
    DRNPopulation,
    EntorhinalCortexPopulation,
    ExternalPopulation,
    GPePopulation,
    GPiPopulation,
    HippocampusPopulation,
    LHbPopulation,
    LocusCoeruleusPopulation,
    MedialSeptumPopulation,
    NucleusBasalisPopulation,
    RMTgPopulation,
    SNcPopulation,
    STNPopulation,
    StriatumPopulation,
    SubiculumPopulation,
    SubstantiaNigraPopulation,
    ThalamusPopulation,
    VTAPopulation,
)
from thalia.brain.synapses import ConductanceScaledSpec, NMReceptorType, STPConfig
from thalia.typing import NeuromodulatorChannel, RegionSizes, ReceptorType, SynapseId

if TYPE_CHECKING:
    from thalia.brain.brain_builder import BrainBuilder


# =============================================================================
# Population-size defaults
# =============================================================================

def _default_sizes() -> RegionSizes:
    """Return default population sizes based on rodent brain estimates (scaled for tractability)."""
    return {
        "basolateral_amygdala": {
            BLAPopulation.PRINCIPAL: 2000,  # ~60% of BLA — glutamatergic fear/extinction engrams
            BLAPopulation.PV: 500,          # ~20% — fast-spiking, feedforward inhibition
            BLAPopulation.SOM: 300,         # ~10% — slow, dendritic inhibition (extinction)
        },
        "central_amygdala": {
            CeAPopulation.LATERAL: 750,  # CeL — integrative (ON/OFF cell division)
            CeAPopulation.MEDIAL: 500,   # CeM — output nucleus (→ LC, LHb)
        },
        "cerebellum": {
            CerebellumPopulation.GRANULE: 10000,  # Granule:Purkinje = 100:1 (biology: 1000:1)
            CerebellumPopulation.PURKINJE: 100,   # Sole output of cerebellar cortex
            CerebellumPopulation.DCN: 100,        # Glutamatergic output (→ thalamus)
            CerebellumPopulation.DCN_GABA: 30,    # GABAergic output (→ IO, nucleo-olivary inhibition)
            CerebellumPopulation.BASKET: 100,     # Molecular layer: soma inhibition, 1:1 with Purkinje
            CerebellumPopulation.STELLATE: 300,   # Molecular layer: dendritic inhibition, 3:1 with Purkinje
        },
        "cortex_association": {
            # Association cortex: thinner L4, thicker L2/3 vs sensory
            CortexPopulation.L23_PYR: 900,
            CortexPopulation.L4_PYR: 150,
            CortexPopulation.L5_PYR: 375,
            CortexPopulation.L6A_PYR: 120,
            CortexPopulation.L6B_PYR: 750,
        },
        "cortex_sensory": {
            # Primary sensory cortex: thick L4 is the hallmark of granular cortex
            CortexPopulation.L23_PYR: 1000,
            CortexPopulation.L4_PYR: 800,
            CortexPopulation.L5_PYR: 500,
            CortexPopulation.L6A_PYR: 150,
            CortexPopulation.L6B_PYR: 600,
        },
        "dorsal_raphe": {
            DRNPopulation.SEROTONIN: 5000,  # Serotonergic projection neurons
            DRNPopulation.GABA: 500,        # Local GABAergic interneurons
        },
        "entorhinal_cortex": {
            EntorhinalCortexPopulation.EC_II: 400,          # Layer II stellate cells: grid/place → DG, CA3
            EntorhinalCortexPopulation.EC_III: 300,         # Layer III pyramidal time cells → CA1
            EntorhinalCortexPopulation.EC_V: 200,           # Layer V output back-projection ← CA1 → neocortex
            EntorhinalCortexPopulation.EC_INHIBITORY: 250,  # PV basket cells (~22% of EC; fixes E/I=7.5 imbalance)
        },
        "globus_pallidus_externa": {
            GPePopulation.ARKYPALLIDAL: 700,  # ~25%, project back to striatum
            GPePopulation.PROTOTYPIC: 2000,   # ~75%, project to STN + SNr
        },
        "globus_pallidus_interna": {
            GPiPopulation.PRINCIPAL: 1500,    # ~75%, project to thalamus VA/VL/MD
            GPiPopulation.BORDER_CELLS: 500,  # ~25%, value-coding; pause on reward
        },
        "hippocampus": {
            HippocampusPopulation.DG: 500,
            HippocampusPopulation.CA3: 250,
            HippocampusPopulation.CA2: 75,
            HippocampusPopulation.CA1: 375,
        },
        "subiculum": {
            SubiculumPopulation.PRINCIPAL: 400,  # Pyramidal relay (burst→regular)
            SubiculumPopulation.PV: 60,          # PV basket-cell interneurons (~15%)
        },
        "lateral_habenula": {
            LHbPopulation.PRINCIPAL: 500,  # Glutamatergic, bad-outcome signal
        },
        "locus_coeruleus": {
            LocusCoeruleusPopulation.NE: 1600,
            LocusCoeruleusPopulation.GABA: 300,
        },
        "medial_septum": {
            MedialSeptumPopulation.ACH: 200,
            MedialSeptumPopulation.GABA: 200,
        },
        "nucleus_basalis": {
            NucleusBasalisPopulation.ACH: 3000,
            NucleusBasalisPopulation.GABA: 500,
        },
        "prefrontal_cortex": {
            # Agranular frontal: thick L2/3 (WM), thin L4, thick L5 (output)
            CortexPopulation.L23_PYR: 360,
            CortexPopulation.L4_PYR: 80,
            CortexPopulation.L5_PYR: 240,
            CortexPopulation.L6A_PYR: 60,
            CortexPopulation.L6B_PYR: 60,
        },
        "rostromedial_tegmentum": {
            RMTgPopulation.GABA: 1000,  # GABAergic, inhibit VTA DA
        },
        "striatum": {
            StriatumPopulation.D1: 200,
            StriatumPopulation.D2: 200,
        },
        "substantia_nigra": {
            SubstantiaNigraPopulation.VTA_FEEDBACK: 1000,
        },
        "substantia_nigra_compacta": {
            SNcPopulation.DA: 1500,
            SNcPopulation.GABA: 500,
        },
        "subthalamic_nucleus": {
            STNPopulation.STN: 500,  # Glutamatergic pacemakers (~20 Hz autonomous)
        },
        # Thalamus — subdivided into three nuclei (sensory, association, mediodorsal).
        # Biology: first-order VPL/VPM relays sensory streams; pulvinar mediates
        # cortico-cortical communication; mediodorsal (MD) supports PFC working memory.
        "thalamus_sensory": {
            # VPL/VPM/LGN/MGN — first-order sensory relay, largest nucleus.
            ThalamusPopulation.RELAY: 250,
            ThalamusPopulation.TRN: 75,      # 3.3:1 relay:TRN ratio preserved.
        },
        "thalamus_association": {
            # Pulvinar — higher-order relay for cortico-cortical communication.
            ThalamusPopulation.RELAY: 80,
            ThalamusPopulation.TRN: 24,
        },
        "thalamus_md": {
            # Mediodorsal nucleus — mnemonic relay for PFC ↔ MD working memory loop.
            ThalamusPopulation.RELAY: 100,
            ThalamusPopulation.TRN: 30,
        },
        "vta": {
            VTAPopulation.DA_MESOLIMBIC: 1375,   # 55% — reward/motivation
            VTAPopulation.DA_MESOCORTICAL: 875,  # 35% — executive/arousal
            VTAPopulation.GABA: 1000,            # 40% of DA population
        },
    }


def _resolve_sizes(
    overrides: Dict[str, Any],
) -> Tuple[RegionSizes, int, int]:
    """Merge user overrides into default sizes.

    Returns:
        ``(sizes, external_sensory_size, external_reward_size)``
    """
    defaults = _default_sizes()
    sizes_overrides: RegionSizes = overrides.get("population_sizes", {})

    sizes: RegionSizes = {
        region_name: {**population_sizes, **sizes_overrides.get(region_name, {})}
        for region_name, population_sizes in defaults.items()
    }

    external_reward_size: int = (
        sizes_overrides
        .get(SynapseId._EXTERNAL_REGION_NAME, {})
        .get(ExternalPopulation.REWARD, 100)
    )
    external_sensory_size_override: Optional[int] = (
        sizes_overrides
        .get(SynapseId._EXTERNAL_REGION_NAME, {})
        .get(ExternalPopulation.SENSORY, None)
    )
    external_sensory_size: int = (
        external_sensory_size_override
        if external_sensory_size_override is not None
        else sizes["thalamus_sensory"][ThalamusPopulation.RELAY]
    )

    return sizes, external_sensory_size, external_reward_size


# =============================================================================
# Region registration
# =============================================================================

def _add_regions(builder: BrainBuilder, sizes: RegionSizes) -> None:
    """Register all brain regions with the builder."""
    builder.add_region(
        "basolateral_amygdala", "basolateral_amygdala",
        population_sizes=sizes["basolateral_amygdala"],
        config=BasolateralAmygdalaConfig(),
    )
    builder.add_region(
        "central_amygdala", "central_amygdala",
        population_sizes=sizes["central_amygdala"],
        config=CentralAmygdalaConfig(),
    )
    builder.add_region(
        "cerebellum", "cerebellum",
        population_sizes=sizes["cerebellum"],
        config=CerebellumConfig(),
    )
    builder.add_region(
        "cortex_association", "cortical_column",
        population_sizes=sizes["cortex_association"],
        config=CorticalColumnConfig(),
    )
    sensory_cortex_config = CorticalColumnConfig()
    # PV: adapt_increment=0.0 (non-adapting fast-spiking; Kv3 channels enable sustained high-freq firing)
    # SST overrides: slightly lower threshold than default (0.80 → 0.75) for sensory cortex
    sensory_cortex_config.population_overrides[CortexPopulation.L23_INHIBITORY_SST] = CorticalPopulationConfig(tau_mem_ms=15.0, v_threshold=0.75, v_reset=0.0, adapt_increment=0.10, tau_adapt_ms= 90.0, noise_std=0.08)
    sensory_cortex_config.population_overrides[CortexPopulation.L4_INHIBITORY_SST]  = CorticalPopulationConfig(tau_mem_ms=15.0, v_threshold=0.75, v_reset=0.0, adapt_increment=0.10, tau_adapt_ms= 90.0, noise_std=0.08)
    sensory_cortex_config.population_overrides[CortexPopulation.L5_INHIBITORY_SST]  = CorticalPopulationConfig(tau_mem_ms=15.0, v_threshold=0.75, v_reset=0.0, adapt_increment=0.10, tau_adapt_ms= 90.0, noise_std=0.08)
    sensory_cortex_config.population_overrides[CortexPopulation.L6A_INHIBITORY_SST] = CorticalPopulationConfig(tau_mem_ms=15.0, v_threshold=0.75, v_reset=0.0, adapt_increment=0.10, tau_adapt_ms= 90.0, noise_std=0.08)
    sensory_cortex_config.population_overrides[CortexPopulation.L6B_INHIBITORY_SST] = CorticalPopulationConfig(tau_mem_ms=15.0, v_threshold=0.75, v_reset=0.0, adapt_increment=0.10, tau_adapt_ms= 90.0, noise_std=0.08)
    builder.add_region(
        "cortex_sensory", "cortical_column",
        population_sizes=sizes["cortex_sensory"],
        config=sensory_cortex_config,
    )
    builder.add_region(
        "dorsal_raphe", "dorsal_raphe",
        population_sizes=sizes["dorsal_raphe"],
        config=DorsalRapheNucleusConfig(),
    )
    builder.add_region(
        "locus_coeruleus", "locus_coeruleus",
        population_sizes=sizes["locus_coeruleus"],
        config=LocusCoeruleusConfig(),
    )
    builder.add_region(
        "nucleus_basalis", "nucleus_basalis",
        population_sizes=sizes["nucleus_basalis"],
        config=NeuralRegionConfig(),
    )
    pfc_config = CorticalColumnConfig(
        # PFC requires slower homeostasis to preserve WM patterns across delays.
        homeostatic_gain=HomeostaticGainConfig(
            lr_per_ms=0.001,        # Very slow (prevents WM collapse)
            tau_ms=5000.0,          # 5 s averaging window
        ),
        homeostatic_threshold=HomeostaticThresholdConfig(
            lr_per_ms=0.001,  # Reduced 0.02→0.001: effective tau must be ≥ 1000 ms
            threshold_min=0.05,     # Lower floor for under-firing regions
            threshold_max=1.5,
        ),

        # Dense mesocortical DA to L2/3: WM gating via D1 receptors.
        # Standard cortex: 7.5%.  PFC L2/3: 30% (matches L5 primary innervation).
        # Biology: Goldman-Rakic et al. 1992; mesocortical DA densely innervates
        # deep L3 / L5 of dlPFC which routes through our L2/3 WM population.
        da_fractions=LayerFractions(
            l23=0.30,    # PFC: dense DA to L2/3 for WM gating
            l4=0.03,
            l5=0.30,
            l6a=0.135,
            l6b=0.135,
        ),

        # Dense L2/3 recurrence for WM attractor dynamics.
        l23_recurrent_connectivity=0.3,
        l23_recurrent_weight_scale=0.0003,

        # NMDA plateau potentials: sustained dendritic depolarization (100-300 ms)
        # supports PFC WM persistent activity via NMDA-dependent plateau potentials
        # in L2/3 pyramidal apical dendrites (Major et al. 2013).
        l23_enable_nmda_plateau=True,
    )
    pfc_config.population_overrides[CortexPopulation.L23_PYR] = CorticalPopulationConfig(
        tau_mem_ms=200.0,       # Very long integration for WM persistence
        v_threshold=2.5,        # Raised 1.8→2.5: PFC L23 at 8.25 Hz (target ≤3); higher selectivity for WM gating
        v_reset=-0.15,          # AHP: enables SFA in PFC WM neurons
        adapt_increment=1.0,    # Raised 0.65→1.0: stronger homeostatic suppression of recurrent drive
        tau_adapt_ms=500.0,     # Raised 120→500: slow AHP range; enables SFA visibility (>1250ms quarter)
        noise_std=0.20,
    )
    pfc_config.population_overrides[CortexPopulation.L4_PYR] = CorticalPopulationConfig(
        tau_mem_ms=10.0,        # Fast — same as default
        v_threshold=0.65,       # Low threshold — same as default
        v_reset=-0.10,          # AHP: mild for fast relay
        adapt_increment=0.06,   # Reduced 0.20→0.06: compensate tau 80→400ms; g_adapt_ss ≈ 0.06×6.8×0.4 = 0.16
        tau_adapt_ms=400.0,     # Raised 80→400: slow AHP for SFA visibility
        noise_std=0.08,
    )
    pfc_config.population_overrides[CortexPopulation.L5_PYR] = CorticalPopulationConfig(
        tau_mem_ms=150.0,       # Long — planning / output integration
        v_threshold=1.5,        # Raised 1.2→1.5: PFC L5 at 17.54 Hz (target 2-15 Hz), 41.3% epileptiform windows.
                                # Higher threshold requires more converging input to fire — reduces baseline overexcitability
                                # and pathological burst recruitment from global alpha oscillation.
        v_reset=-0.12,          # AHP: moderate for output neurons
        adapt_increment=0.15,   # Raised 0.05→0.15: stronger per-spike AHP provides tonic rate suppression.
        tau_adapt_ms=1500.0,    # Raised 300→1500: adaptation must still be BUILDING at the SFA measurement window
                                # start (t=2000ms). At tau=1500ms, 73% of SS reached by t=2s, 95% by t=4.5s.
                                # First-quarter mean ~79% SS, last-quarter ~95% SS → visible SFA>1.3 with strong adapt.
        noise_std=0.12,
    )
    pfc_config.population_overrides[CortexPopulation.L6A_PYR] = CorticalPopulationConfig(
        tau_mem_ms=15.0,
        v_threshold=1.4,
        v_reset=-0.10,          # AHP: mild for feedback neurons
        adapt_increment=0.25,   # Raised 0.18→0.25: match base cortical column L6A increase
        tau_adapt_ms=100.0,
        noise_std=0.12,
    )
    pfc_config.population_overrides[CortexPopulation.L6B_PYR] = CorticalPopulationConfig(
        tau_mem_ms=25.0,
        v_threshold=1.1,
        v_reset=-0.10,          # AHP: mild for relay modulation
        adapt_increment=0.06,   # Reduced 0.22→0.06: compensate tau 100→400ms; g_adapt_ss ≈ 0.06×5.6×0.4 = 0.13
        tau_adapt_ms=400.0,     # Raised 100→400: slow AHP for SFA visibility
        noise_std=0.10,
    )
    builder.add_region(
        "prefrontal_cortex", "prefrontal_cortex",
        population_sizes=sizes["prefrontal_cortex"],
        config=pfc_config,
    )
    builder.add_region(
        "substantia_nigra_compacta", "substantia_nigra_compacta",
        population_sizes=sizes["substantia_nigra_compacta"],
        config=DopaminePacemakerConfig(
            baseline_drive=0.008,
            neuromodulator_receptors=[
                NMReceptorConfig(NMReceptorType.SHT_1A, NeuromodulatorChannel.SHT, "_sht_concentration", (SNcPopulation.DA,)),
            ],
        ),
    )
    # Three thalamic nuclei: sensory (VPL/VPM), association (pulvinar), mediodorsal (MD).
    builder.add_region(
        "thalamus_sensory", "thalamus",
        population_sizes=sizes["thalamus_sensory"],
        config=ThalamusConfig(),
    )
    builder.add_region(
        "thalamus_association", "thalamus",
        population_sizes=sizes["thalamus_association"],
        config=ThalamusConfig(
            trn_baseline_drive=0.012,  # TRN=10.14 Hz, healthy.
            trn_relay_gaba_a_mean=0.105,  # E/I=9.9 Hz, need -20%.
        ),
    )
    builder.add_region(
        "thalamus_md", "thalamus",
        population_sizes=sizes["thalamus_md"],
        config=ThalamusConfig(
            trn_baseline_drive=0.012,  # TRN=10.03 Hz, healthy.
            trn_relay_gaba_a_mean=0.085,  # E/I=8.5 Hz, need -6%.
            homeostatic_target_rates={ThalamusPopulation.RELAY: 0.015},
        ),
    )
    builder.add_region(
        "vta", "vta",
        population_sizes=sizes["vta"],
        config=VTAConfig(),
    )

    add_basal_ganglia_regions(
        builder,
        sizes,
        str_name="striatum",
        gpe_name="globus_pallidus_externa",
        gpi_name="globus_pallidus_interna",
        stn_name="subthalamic_nucleus",
        snr_name="substantia_nigra",
        lhb_name="lateral_habenula",
        rmtg_name="rostromedial_tegmentum",
    )

    add_medial_temporal_lobe_regions(
        builder,
        sizes,
        ms_name="medial_septum",
        ec_name="entorhinal_cortex",
        hpc_name="hippocampus",
        sub_name="subiculum",
    )


# =============================================================================
# External inputs
# =============================================================================

def _connect_external_inputs(
    builder: BrainBuilder,
    external_sensory_size: int,
    external_reward_size: int,
) -> None:
    """Wire external (training-loop) spike vectors into the brain.

    Two sources:
    * Sensory input → thalamic relay (ascending sensory pathway).
    * Reward signal → VTA DA mesolimbic (population-coded RPE signal).
    """
    # External Sensory Input → Thalamus RELAY
    # Represents retinogeniculate / cochlear / somatosensory ascending pathway.
    # Very fast, heavily myelinated; delay implicit in simulation timestep.
    builder.add_external_input_source(
        synapse_id=SynapseId.external_sensory_to_thalamus_relay("thalamus_sensory"),
        n_input=external_sensory_size,
        connectivity=0.25,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=20.0,
            target_g_L=0.08,
            target_tau_E_ms=5.0,
            target_v_inf=0.60,  # Lowered 0.85→0.60: with homeostatic threshold_min=0.55,
            # v_inf=0.85 was far above adapted threshold → sustained 38 Hz.
            # v_inf=0.60 keeps relay in fluctuation-driven regime near threshold.
            fraction_of_drive=0.15,  # Reduced 0.30→0.15: 15% external + 50% L6B = 65% total.
        ),
        stp_config=STPConfig(U=0.30, tau_d=400.0, tau_f=30.0),
    )

    # External reward → VTA DA MESOLIMBIC
    # Population-coded spikes generated by Brain.deliver_reward().
    builder.add_external_input_source(
        synapse_id=SynapseId.external_reward_to_vta_da("vta"),
        n_input=external_reward_size,
        connectivity=0.7,
        weight_scale=0.0008,
        stp_config=None,
    )

    # External novelty → VTA DA MESOLIMBIC (Hippocampal-VTA loop, Lisman & Grace 2005)
    # CA1 mismatch signal (prediction error: EC input > CA3 recall) auto-injected by
    # Brain.forward() with one-step causal delay.  Represents the subiculum → ventral
    # striatum → VTA disinhibition pathway that gates exploratory DA bursts on novel stimuli.
    builder.add_external_input_source(
        synapse_id=SynapseId.external_novelty_to_vta_da("vta"),
        n_input=external_reward_size,
        connectivity=0.5,
        weight_scale=0.0005,
        stp_config=None,
    )


# =============================================================================
# Cortex → Entorhinal Cortex → Hippocampus circuit
# =============================================================================

def _connect_cortex_ec_hippocampus(builder: BrainBuilder) -> None:
    """Wire the cortex → EC → hippocampus → EC → cortex memory loop.

    Also includes:
    * Thalamus → DG (fast subcortical sensory encoding bypass).
    * Medial Septum ↔ Hippocampus (septal theta pacemaker loop).
    """
    # Sensory cortex L5 → EC_II: spatial/sensory context → perforant path
    # Moderately myelinated; distance ~3-5 cm → 5-7ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_sensory",
            source_population=CortexPopulation.L5_PYR,
            target_region="entorhinal_cortex",
            target_population=EntorhinalCortexPopulation.EC_II,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=6.0,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=10.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=1.05,
            fraction_of_drive=0.40,
        ),
        stp_config=STPConfig(U=0.50, tau_d=700.0, tau_f=20.0),
    )

    # Association cortex L2/3 → EC_II: semantic / multi-modal context → perforant path
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_association",
            source_population=CortexPopulation.L23_PYR,
            target_region="entorhinal_cortex",
            target_population=EntorhinalCortexPopulation.EC_II,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=5.0,
        connectivity=0.35,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=3.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=1.05,
            fraction_of_drive=0.35,
        ),
        stp_config=STPConfig(U=0.50, tau_d=600.0, tau_f=25.0),
    )

    # Association cortex L2/3 → EC_III: temporal / semantic context → temporoammonic path
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_association",
            source_population=CortexPopulation.L23_PYR,
            target_region="entorhinal_cortex",
            target_population=EntorhinalCortexPopulation.EC_III,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=5.0,
        connectivity=0.30,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=3.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=1.05,
            fraction_of_drive=0.55,
        ),
        stp_config=STPConfig(U=0.50, tau_d=600.0, tau_f=25.0),
    )

    # Internal MTL wiring: EC ↔ HPC afferents, CA1 → Sub → EC_V back-projection,
    # and septal theta pacemaker loop (MS ↔ HPC) are all wired by the preset.
    add_medial_temporal_lobe_circuit(builder, add_regions=False, include_subiculum=True)

    # EC_V → Association cortex L2/3: memory indexing output → cortical consolidation
    # EC layer V broadcasts the compressed hippocampal memory index back to neocortex.
    builder.connect(
        synapse_id=SynapseId(
            source_region="entorhinal_cortex",
            source_population=EntorhinalCortexPopulation.EC_V,
            target_region="cortex_association",
            target_population=CortexPopulation.L23_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=6.0,
        connectivity=0.25,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=3.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=1.05,
            fraction_of_drive=0.10,
        ),
        # Moderately depressing long-range excitatory (entorhinal→neocortex).
        stp_config=STPConfig(U=0.30, tau_d=400.0, tau_f=20.0),
    )

    # Thalamus sensory → Hippocampus DG: direct sensory-to-memory pathway (bypass cortex)
    # Nucleus reuniens provides direct thalamic input; fast subcortical encoding.
    # Distance: ~4-6cm → 6-10ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="thalamus_sensory",
            source_population=ThalamusPopulation.RELAY,
            target_region="hippocampus",
            target_population=HippocampusPopulation.DG,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=8.0,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=30.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=1.60,  # Raised 0.90→1.60: DG granule v_threshold=1.80; v_inf must be
            # near threshold for any firing. 1.60 = 89% of threshold = sparse fluctuation-driven.
            fraction_of_drive=0.30,  # 0.55→0.30: with relay recovering from GPi fix (~10-15 Hz
            # vs 2.2 Hz), thalamus→DG share must decrease to avoid re-overdrive.
        ),
        stp_config=STPConfig(U=0.20, tau_d=250.0, tau_f=20.0),  # U 0.30→0.20, tau_d 400→250:
        # Less depression preserves drive under tonic relay firing.
    )


# =============================================================================
# Striatal inputs: corticostriatal + hippocampostriatal + thalamostriatal
# =============================================================================

def _connect_striatal_inputs(builder: BrainBuilder) -> None:
    """Wire all multi-source inputs to the striatum.

    Each call to ``connect_to_striatum`` creates **four** connections —
    one per sub-population (D1, D2, FSI, TAN) — with biologically motivated
    connectivity and weight differences.

    Sources and their axonal delays (Gerfen & Surmeier 2011):
    * Sensory cortex L5  → striatum: 3-5ms
    * Hippocampus CA1    → striatum: 7-10ms
    * PFC                → striatum: 12-18ms
    * Thalamus RELAY     → striatum: 4-7ms
    * Association cortex → striatum: slightly longer than sensory
    """
    # Sensory cortex L5 → Striatum: primary corticostriatal drive
    builder.connect_to_striatum(
        source_region="cortex_sensory",
        source_population=CortexPopulation.L5_PYR,
        axonal_delay_ms=4.0,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=10.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=0.95,
            fraction_of_drive=0.45,
        ),
        stp_config=STPConfig(U=0.4, tau_d=250.0, tau_f=150.0),
    )

    # Subiculum → Striatum: hippocampostriatal pathway
    # Subiculum is the primary hippocampal output to dorsal striatum for
    # context-dependent action selection (Groenewegen et al. 1987).
    builder.connect_to_striatum(
        source_region="subiculum",
        source_population=SubiculumPopulation.PRINCIPAL,
        axonal_delay_ms=8.5,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=0.95,
            fraction_of_drive=0.35,
        ),
        stp_config=STPConfig(U=0.5, tau_d=700.0, tau_f=400.0),
    )

    # PFC → Striatum: prefrontal corticostriatal pathway
    builder.connect_to_striatum(
        source_region="prefrontal_cortex",
        source_population=CortexPopulation.L5_PYR,
        axonal_delay_ms=15.0,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=0.95,
            fraction_of_drive=0.35,
        ),
        stp_config=STPConfig(U=0.4, tau_d=250.0, tau_f=150.0),
    )

    # Thalamus sensory → Striatum: thalamostriatal pathway for habitual responses
    # Direct sensory-action pathway bypassing cortex.
    builder.connect_to_striatum(
        source_region="thalamus_sensory",
        source_population=ThalamusPopulation.RELAY,
        axonal_delay_ms=5.0,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=30.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=0.95,
            fraction_of_drive=0.40,
        ),
        # Thalamostriatal synapses are facilitating (Raju et al. 2006; Ellender et al. 2011).
        # U=0.05: low initial release probability → builds with sustained relay bursts.
        # tau_f=500ms > tau_d=200ms ensures net facilitation across the burst window.
        stp_config=STPConfig(U=0.05, tau_d=200.0, tau_f=500.0),
    )

    # Association cortex L5 → Striatum: goal-directed corticostriatal projection
    # Slightly longer delay than sensory→striatum (additional cortical processing).
    builder.connect_to_striatum(
        source_region="cortex_association",
        source_population=CortexPopulation.L5_PYR,
        axonal_delay_ms=6.0,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=10.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=0.95,
            fraction_of_drive=0.40,
        ),
        stp_config=STPConfig(U=0.4, tau_d=250.0, tau_f=150.0),
    )


# =============================================================================
# Basal ganglia: direct / indirect / hyperdirect pathways
# =============================================================================

def _connect_basal_ganglia(builder: BrainBuilder) -> None:
    """Wire the complete basal ganglia circuit.

    Direct pathway:   D1 → SNr (GABA, strong inhibition → disinhibit thalamus).
    Indirect pathway: D2 → GPe → STN → SNr (net excitation of SNr → suppress thalamus).
    Hyperdirect:      Cortex L5 → STN (fastest action-suppression route).
    Also:             GPe → SNr (pallido-nigral inhibitory gate).
    """
    # All direct/indirect pathways, GPi → thalamus, and the anti-reward cascade
    # (SNr → LHb → RMTg → VTA) are delegated to the BG preset.
    # Regions are already registered by _add_regions; only connections are wired here.
    add_basal_ganglia_circuit(
        builder,
        thalamus_name="thalamus_sensory",
        vta_name="vta",
        add_regions=False,
    )

    # Sensory cortex L5 → STN: HYPERDIRECT pathway (fastest cortex→BG route)
    # Cortex sends a 'hold' signal to STN before striatal decision propagates.
    # Arrives at SNr before striatal signals, enabling rapid action suppression.
    # Distance: ~3-8cm, heavily myelinated corticospinal-type axons → 3-8ms.
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_sensory",
            source_population=CortexPopulation.L5_PYR,
            target_region="subthalamic_nucleus",
            target_population=STNPopulation.STN,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=5.0,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=10.0,
            target_g_L=0.08,
            target_tau_E_ms=5.0,
            target_v_inf=1.05,
            fraction_of_drive=0.30,
        ),
        stp_config=STPConfig(U=0.50, tau_d=600.0, tau_f=25.0),
    )

    # PFC L5 → STN: PREFRONTAL HYPERDIRECT pathway (Aron & Poldrack 2006)
    # PFC is the primary source of hyperdirect input for response inhibition
    # and conflict-driven decisional pause (Frank 2006). Without this, PFC
    # cannot globally suppress BG output during approach-avoidance conflict.
    # Longer delay than sensory (more synapses, longer axonal distance).
    # Weaker drive fraction — PFC conflict signal modulates STN, not dominates.
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal_cortex",
            source_population=CortexPopulation.L5_PYR,
            target_region="subthalamic_nucleus",
            target_population=STNPopulation.STN,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=7.0,
        connectivity=0.25,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.08,
            target_tau_E_ms=5.0,
            target_v_inf=1.05,
            fraction_of_drive=0.20,
        ),
        stp_config=STPConfig(U=0.50, tau_d=600.0, tau_f=25.0),
    )

    # SNc DA → Striatum: nigrostriatal pathway (glutamate co-release)
    # SNc DA neurons are the primary dopaminergic input to dorsal striatum.
    # Besides volume-transmission DA (handled by DA_NIGROSTRIATAL neuromodulator
    # channel), SNc DA neurons co-release glutamate at striatal synapses
    # (Tritsch et al. 2012; Hnasko et al. 2010).  This provides a fast
    # excitatory signal coincident with DA release.
    # Distance: ~3-4cm (midbrain → dorsal striatum), myelinated → 4-6ms delay.
    # DA neurons fire tonically at ~4-6 Hz; facilitating STP amplifies phasic bursts.
    builder.connect_to_striatum(
        source_region="substantia_nigra_compacta",
        source_population=SNcPopulation.DA,
        axonal_delay_ms=5.0,
        connectivity=0.20,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=0.95,
            fraction_of_drive=0.10,  # Modest: glutamate co-release is secondary to DA modulation
        ),
        stp_config=STPConfig(U=0.15, tau_d=200.0, tau_f=600.0),
    )


# =============================================================================
# Top-level preset function
# =============================================================================

def build(builder: BrainBuilder, **overrides: Any) -> None:
    """Default biologically realistic brain architecture.

    Orchestrates all circuit helpers.  Each helper encapsulates one anatomically
    coherent sub-circuit; see individual docstrings for biology references.
    """
    sizes, ext_sensory_size, ext_reward_size = _resolve_sizes(overrides)

    _add_regions(builder, sizes)
    _connect_external_inputs(builder, ext_sensory_size, ext_reward_size)
    connect_thalamocortical(builder)
    _connect_cortex_ec_hippocampus(builder)
    connect_prefrontal(builder)
    connect_cerebellum(builder)
    _connect_striatal_inputs(builder)
    _connect_basal_ganglia(builder)
    connect_corticocortical(builder)
    connect_neuromodulators(builder)
    connect_amygdala(builder)
