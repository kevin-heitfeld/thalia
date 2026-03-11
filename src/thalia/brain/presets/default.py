"""Default biologically realistic brain preset.

Implements the full biologically-grounded default brain architecture decomposed
into one helper function per anatomical circuit.  ``build`` orchestrates
these helpers and is registered as the ``"default"`` preset in ``brain_builder.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from thalia.brain.presets.basal_ganglia import add_bg_circuit
from thalia.brain.presets.medial_temporal_lobe import add_mtl_circuit
from thalia.brain.regions.population_names import (
    BLAPopulation,
    CeAPopulation,
    CerebellumPopulation,
    CortexPopulation,
    DRNPopulation,
    ECPopulation,
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
    SubstantiaNigraPopulation,
    ThalamusPopulation,
    VTAPopulation,
)
from thalia.brain.synapses import ConductanceScaledSpec, STPConfig
from thalia.typing import RegionSizes, ReceptorType, SynapseId

if TYPE_CHECKING:
    from thalia.brain.brain_builder import BrainBuilder


# =============================================================================
# Population-size defaults
# =============================================================================

def _default_sizes() -> RegionSizes:
    """Return default population sizes based on rodent brain estimates (scaled for tractability)."""
    return {
        "basolateral_amygdala": {
            BLAPopulation.PRINCIPAL: 2000,   # ~60% of BLA — glutamatergic fear/extinction engrams
            BLAPopulation.PV: 500,           # ~20% — fast-spiking, feedforward inhibition
            BLAPopulation.SOM: 300,          # ~10% — slow, dendritic inhibition (extinction)
        },
        "central_amygdala": {
            CeAPopulation.LATERAL: 750,      # CeL — integrative (ON/OFF cell division)
            CeAPopulation.MEDIAL: 500,       # CeM — output nucleus (→ LC, LHb)
        },
        "cerebellum": {
            CerebellumPopulation.GRANULE: 10000,  # Granule:Purkinje = 100:1 (biology: 1000:1)
            CerebellumPopulation.PURKINJE: 100,   # Sole output of cerebellar cortex
            CerebellumPopulation.DCN: 100,        # Sole cerebellar output neurons
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
            DRNPopulation.SEROTONIN: 5000,   # Serotonergic projection neurons
            DRNPopulation.GABA: 500,         # Local GABAergic interneurons
        },
        "entorhinal_cortex": {
            ECPopulation.EC_II: 400,    # Layer II stellate cells: grid/place → DG, CA3
            ECPopulation.EC_III: 300,   # Layer III pyramidal time cells → CA1
            ECPopulation.EC_V: 200,     # Layer V output back-projection ← CA1 → neocortex
        },
        "globus_pallidus_externa": {
            GPePopulation.ARKYPALLIDAL: 700,   # ~25%, project back to striatum
            GPePopulation.PROTOTYPIC: 2000,    # ~75%, project to STN + SNr
        },
        "globus_pallidus_interna": {
            GPiPopulation.PRINCIPAL: 1500,     # ~75%, project to thalamus VA/VL/MD
            GPiPopulation.BORDER_CELLS: 500,   # ~25%, value-coding; pause on reward
        },
        "hippocampus": {
            HippocampusPopulation.DG: 500,
            HippocampusPopulation.CA3: 250,
            HippocampusPopulation.CA2: 75,
            HippocampusPopulation.CA1: 375,
        },
        "lateral_habenula": {
            LHbPopulation.PRINCIPAL: 500,      # Glutamatergic, bad-outcome signal
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
            RMTgPopulation.GABA: 1000,         # GABAergic, inhibit VTA DA
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
            STNPopulation.STN: 500,            # Glutamatergic pacemakers (~20 Hz autonomous)
        },
        "thalamus": {
            ThalamusPopulation.RELAY: 400,
            ThalamusPopulation.TRN: 40,        # 10:1 relay:TRN ratio
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
        region: {**defaults[region], **sizes_overrides.get(region, {})}
        for region in defaults
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
        else sizes["thalamus"][ThalamusPopulation.RELAY]
    )

    return sizes, external_sensory_size, external_reward_size


# =============================================================================
# Region registration
# =============================================================================

def _add_regions(builder: BrainBuilder, sizes: RegionSizes) -> None:
    """Register all brain regions with the builder."""
    builder.add_region("basolateral_amygdala", "basolateral_amygdala", population_sizes=sizes["basolateral_amygdala"])
    builder.add_region("central_amygdala", "central_amygdala", population_sizes=sizes["central_amygdala"])
    builder.add_region("cerebellum", "cerebellum", population_sizes=sizes["cerebellum"])
    builder.add_region("cortex_association", "cortical_column", population_sizes=sizes["cortex_association"])
    builder.add_region("cortex_sensory", "cortical_column", population_sizes=sizes["cortex_sensory"])
    builder.add_region("dorsal_raphe", "dorsal_raphe", population_sizes=sizes["dorsal_raphe"])
    builder.add_region("entorhinal_cortex", "entorhinal_cortex", population_sizes=sizes["entorhinal_cortex"])
    builder.add_region("globus_pallidus_externa", "globus_pallidus_externa", population_sizes=sizes["globus_pallidus_externa"])
    builder.add_region("globus_pallidus_interna", "globus_pallidus_interna", population_sizes=sizes["globus_pallidus_interna"])
    builder.add_region("hippocampus", "hippocampus", population_sizes=sizes["hippocampus"])
    builder.add_region("lateral_habenula", "lateral_habenula", population_sizes=sizes["lateral_habenula"])
    builder.add_region("locus_coeruleus", "locus_coeruleus", population_sizes=sizes["locus_coeruleus"])
    builder.add_region("medial_septum", "medial_septum", population_sizes=sizes["medial_septum"])
    builder.add_region("nucleus_basalis", "nucleus_basalis", population_sizes=sizes["nucleus_basalis"])
    builder.add_region("prefrontal_cortex", "prefrontal_cortex", population_sizes=sizes["prefrontal_cortex"])
    builder.add_region("rostromedial_tegmentum", "rostromedial_tegmentum", population_sizes=sizes["rostromedial_tegmentum"])
    builder.add_region("striatum", "striatum", population_sizes=sizes["striatum"])
    builder.add_region("substantia_nigra", "substantia_nigra", population_sizes=sizes["substantia_nigra"])
    builder.add_region("substantia_nigra_compacta", "substantia_nigra_compacta", population_sizes=sizes["substantia_nigra_compacta"])
    builder.add_region("subthalamic_nucleus", "subthalamic_nucleus", population_sizes=sizes["subthalamic_nucleus"])
    builder.add_region("thalamus", "thalamus", population_sizes=sizes["thalamus"])
    builder.add_region("vta", "vta", population_sizes=sizes["vta"])


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
    # ConductanceScaledSpec: external sensory (20 Hz) drives relay to V_inf=0.85
    # (just above threshold=0.8). 80% of relay drive; L6b corticothalamic handles 20%.
    builder.add_external_input_source(
        synapse_id=SynapseId.external_sensory_to_thalamus_relay("thalamus"),
        n_input=external_sensory_size,
        connectivity=0.25,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=20.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=0.85,
            fraction_of_drive=0.80,
        ),
    )

    # External reward → VTA DA MESOLIMBIC
    # Population-coded spikes generated by Brain.deliver_reward().
    builder.add_external_input_source(
        synapse_id=SynapseId.external_reward_to_vta_da("vta"),
        n_input=external_reward_size,
        connectivity=0.7,
        weight_scale=0.0008,
    )

    # External novelty → VTA DA MESOLIMBIC (Hippocampal-VTA loop, Lisman & Grace 2005)
    # CA1 mismatch signal (prediction error: EC input > CA3 recall) auto-injected by
    # Brain.forward() with one-step causal delay.  Represents the subiculum → ventral
    # striatum → VTA disinhibition pathway that gates exploratory DA bursts on novel stimuli.
    # Weight ~60% of reward weight: novelty drive is weaker than explicit reward.
    builder.add_external_input_source(
        synapse_id=SynapseId.external_novelty_to_vta_da("vta"),
        n_input=external_reward_size,
        connectivity=0.5,
        weight_scale=0.0005,
    )


# =============================================================================
# Thalamocortical circuit: thalamus ↔ cortex_sensory
# =============================================================================

def _connect_thalamocortical(builder: BrainBuilder) -> None:
    """Wire the bidirectional thalamus ↔ sensory cortex loop.

    * Thalamus RELAY → L4 PYR: Main thalamocortical drive.
    * Thalamus RELAY → L4 PV: Feedforward inhibition (arrives ~1 ms after PYR).
    * Sensory L6A → Thalamus TRN: Type-I slow corticothalamic feedback.
    * Sensory L6B → Thalamus RELAY: Type-II fast corticothalamic feedback.
    """
    # Thalamus → L4 Pyramidal: Main thalamocortical drive
    # Distance: ~2-3cm, conduction velocity: ~10-20 m/s → 2-3ms delay
    builder.connect(
        synapse_id=SynapseId(
            source_region="thalamus",
            source_population=ThalamusPopulation.RELAY,
            target_region="cortex_sensory",
            target_population=CortexPopulation.L4_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=2.5,
        axonal_delay_std_ms=0.75,
        connectivity=0.7,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=30.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=0.70,
            fraction_of_drive=0.85,
        ),
        stp_config=STPConfig(U=0.45, tau_d=700.0, tau_f=20.0),
    )

    # Thalamus → L4 PV: Feedforward inhibition drive
    # PV cells have lower thresholds; thalamus provides 35% of PV drive.
    # Reduced from 0.80: 217 Hz PV at 0.80 silenced L4_pyr completely.
    builder.connect(
        synapse_id=SynapseId(
            source_region="thalamus",
            source_population=ThalamusPopulation.RELAY,
            target_region="cortex_sensory",
            target_population=CortexPopulation.L4_INHIBITORY_PV,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=2.5,
        axonal_delay_std_ms=0.75,
        connectivity=0.7,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=20.0,
            target_g_L=0.10,
            target_tau_E_ms=3.0,
            target_v_inf=0.95,
            fraction_of_drive=0.35,
        ),
        stp_config=STPConfig(U=0.20, tau_d=200.0, tau_f=20.0),
    )

    # CorticalColumn L6a → Thalamus TRN: Inhibitory attention modulation (type-I, slow)
    # L6a→TRN: ~10ms (selectively gates thalamic relay for selective attention).
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_sensory",
            source_population=CortexPopulation.L6A_PYR,
            target_region="thalamus",
            target_population=ThalamusPopulation.TRN,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=10.0,
        axonal_delay_std_ms=3.0,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.10,
            target_tau_E_ms=4.0,
            target_v_inf=1.0,
            fraction_of_drive=0.40,
        ),
    )

    # CorticalColumn L6b → Thalamus RELAY: Excitatory precision feedback (type-II, fast)
    # L6b→Relay: ~5ms; precision-enhancing corticothalamic feedback.
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_sensory",
            source_population=CortexPopulation.L6B_PYR,
            target_region="thalamus",
            target_population=ThalamusPopulation.RELAY,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=5.0,
        axonal_delay_std_ms=1.5,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=0.85,
            fraction_of_drive=0.20,
        ),
        stp_config=STPConfig(U=0.08, tau_d=150.0, tau_f=800.0),
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
            target_population=ECPopulation.EC_II,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=6.0,
        axonal_delay_std_ms=1.8,
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
            target_population=ECPopulation.EC_II,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=5.0,
        axonal_delay_std_ms=1.5,
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
            target_population=ECPopulation.EC_III,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=5.0,
        axonal_delay_std_ms=1.5,
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

    # Internal MTL wiring: EC ↔ HPC afferents, HPC → EC back-projection,
    # and septal theta pacemaker loop (MS ↔ HPC) are all wired by the preset.
    add_mtl_circuit(builder, add_regions=False)

    # EC_V → Association cortex L2/3: memory indexing output → cortical consolidation
    # EC layer V broadcasts the compressed hippocampal memory index back to neocortex.
    builder.connect(
        synapse_id=SynapseId(
            source_region="entorhinal_cortex",
            source_population=ECPopulation.EC_V,
            target_region="cortex_association",
            target_population=CortexPopulation.L23_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=6.0,
        axonal_delay_std_ms=1.8,
        connectivity=0.25,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=3.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=1.05,
            fraction_of_drive=0.10,
        ),
    )

    # Thalamus → Hippocampus DG: direct sensory-to-memory pathway (bypass cortex)
    # Nucleus reuniens provides direct thalamic input; fast subcortical encoding.
    # Distance: ~4-6cm → 6-10ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="thalamus",
            source_population=ThalamusPopulation.RELAY,
            target_region="hippocampus",
            target_population=HippocampusPopulation.DG,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=8.0,
        axonal_delay_std_ms=2.4,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=30.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=0.90,
            fraction_of_drive=0.20,
        ),
        stp_config=STPConfig(U=0.45, tau_d=700.0, tau_f=20.0),
    )


# =============================================================================
# Prefrontal cortex circuit: PFC ↔ hippocampus, PFC → cortex, striatum → PFC
# =============================================================================

def _connect_prefrontal(builder: BrainBuilder) -> None:
    """Wire prefrontal cortex into the rest of the brain.

    * Association cortex → PFC: multi-modal input to executive control.
    * Striatum D1 → PFC: basal ganglia gating of working memory (via thalamus).
    * PFC ↔ Hippocampus: memory-guided decision making.
    * PFC → Sensory cortex L2/3: top-down attentional modulation.
    * CA1 → PFC L5 apical: hippocampal context to PFC output-layer apical tufts.
    * PFC L23 → Sensory L5 apical: top-down deep-layer FB to sensory output layer.
    """
    # Association → PFC: higher-level representations drive executive control
    # Distance: ~5-10cm → 10-15ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_association",
            source_population=CortexPopulation.L23_PYR,
            target_region="prefrontal_cortex",
            target_population=CortexPopulation.L4_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=12.5,
        axonal_delay_std_ms=3.75,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=3.0,
            target_g_L=0.02,
            target_tau_E_ms=10.0,
            target_v_inf=1.05,
            fraction_of_drive=0.50,
        ),
        # Corticofrontal feedforward: moderate depression (CORTICAL_FF).
        stp_config=STPConfig(U=0.50, tau_d=600.0, tau_f=25.0),
    )

    # Striatum D1 → PFC: BG gating of working memory (via MD/VA thalamic relay)
    # Total delay: striatum→thalamus→PFC relay → 15-20ms.
    builder.connect(
        synapse_id=SynapseId(
            source_region="striatum",
            source_population=StriatumPopulation.D1,
            target_region="prefrontal_cortex",
            target_population=CortexPopulation.L4_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=17.5,
        axonal_delay_std_ms=5.25,
        connectivity=0.6,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=2.0,
            target_g_L=0.02,
            target_tau_E_ms=10.0,
            target_v_inf=1.05,
            fraction_of_drive=0.20,
        ),
        # D1-MSN disinhibitory gate: encoded as AMPA but represents the net
        # suppressive effect transmitted via the direct pathway.  Striatopallidal-
        # like depression (moderate) prevents sustained BG gating.
        stp_config=STPConfig(U=0.5, tau_d=800.0, tau_f=20.0),
    )

    # PFC → Hippocampus CA1: top-down memory retrieval and schema application
    # Distance: ~5-7cm → 12-18ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal_cortex",
            source_population=CortexPopulation.L23_PYR,
            target_region="hippocampus",
            target_population=HippocampusPopulation.CA1,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=15.0,
        axonal_delay_std_ms=4.5,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=0.95,
            fraction_of_drive=0.15,
        ),
        # PFC→CA1: corticofugal depressing top-down
        stp_config=STPConfig(U=0.50, tau_d=600.0, tau_f=25.0),
    )

    # Hippocampus CA1 → PFC: memory-guided decision making
    # Distance: ~5-7cm → 10-15ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="hippocampus",
            source_population=HippocampusPopulation.CA1,
            target_region="prefrontal_cortex",
            target_population=CortexPopulation.L23_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=12.0,
        axonal_delay_std_ms=3.6,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=3.0,
            target_g_L=0.02,
            target_tau_E_ms=10.0,
            target_v_inf=1.05,
            fraction_of_drive=0.15,
        ),
        stp_config=STPConfig(U=0.5, tau_d=700.0, tau_f=400.0),
    )

    # Hippocampus CA1 → PFC L5: hippocampal context to PFC output-layer apical tufts
    # CA1 in biology projects to BOTH L2/3 and L5/6 of PFC via the fornix
    # (Thierry et al. 2000; Jay et al. 1992).  The L5 target provides apical dendritic
    # input, gating subcortical output via coincidence detection in the two-compartment
    # model.  PFC is the apex of the cortical hierarchy — hippocampus is the only
    # major source of top-down context for PFC L5 apical compartment.
    builder.connect(
        synapse_id=SynapseId(
            source_region="hippocampus",
            source_population=HippocampusPopulation.CA1,
            target_region="prefrontal_cortex",
            target_population=CortexPopulation.L5_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=12.0,
        axonal_delay_std_ms=3.6,
        connectivity=0.25,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=3.0,
            target_g_L=0.02,
            target_tau_E_ms=10.0,
            target_v_inf=1.05,
            fraction_of_drive=0.10,
        ),
        # CA1→PFC L5 apical: same Schaffer-collateral-like dynamics as L23 path.
        stp_config=STPConfig(U=0.5, tau_d=700.0, tau_f=400.0),
    )

    # PFC → Sensory cortex L2/3: top-down attention and cognitive control
    # Corticocortical feedback targets superficial layers (L2/3), bypassing thalamic input.
    # Distance: ~5-8cm → 10-15ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal_cortex",
            source_population=CortexPopulation.L23_PYR,
            target_region="cortex_sensory",
            target_population=CortexPopulation.L23_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=12.0,
        axonal_delay_std_ms=3.6,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=1.05,
            fraction_of_drive=0.20,
        ),
        # PFC top-down to sensory L2/3: facilitating (canonical hierarchical FB
        # targets superficial layers; Wang et al. 2006 — UF-type).
        stp_config=STPConfig(U=0.1, tau_d=300.0, tau_f=300.0),
    )

    # PFC → Sensory cortex L5 apical: deep-layer top-down context to sensory output cells
    # Canonical predictive-coding FB targets both L2/3 (suppression of prediction error)
    # and L5 apical tufts (gain modulation of output layer).  L5 is the main
    # subcortical projection layer; PFC gating here modulates what sensory signals
    # reach thalamus, striatum and cerebellum.  (Larkum 2013; Bastos et al. 2012)
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal_cortex",
            source_population=CortexPopulation.L23_PYR,
            target_region="cortex_sensory",
            target_population=CortexPopulation.L5_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=12.0,
        axonal_delay_std_ms=3.6,
        connectivity=0.25,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=1.05,
            fraction_of_drive=0.10,
        ),
        # PFC→sensory L5 apical: facilitating FB (same type as PFC→L23 pathway).
        stp_config=STPConfig(U=0.1, tau_d=300.0, tau_f=300.0),
    )


# =============================================================================
# Cerebellum: motor / cognitive forward models
# =============================================================================

def _connect_cerebellum(builder: BrainBuilder) -> None:
    """Wire the cerebellum forward-model circuit.

    Input: sensory L5 and PFC drive granule cells (via pons).
    Output: DCN predictions fan back to sensory cortex L4 (via thalamus VL/VA).
    """
    # Sensory cortex L5 → Cerebellum GRANULE: corticopontocerebellar pathway
    # Via pontine nuclei (Schmahmann 1996); distance ~10-15cm → 20-30ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_sensory",
            source_population=CortexPopulation.L5_PYR,
            target_region="cerebellum",
            target_population=CerebellumPopulation.GRANULE,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=25.0,
        axonal_delay_std_ms=7.5,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=10.0,
            target_g_L=0.05,
            target_tau_E_ms=2.5,
            target_v_inf=0.90,
            fraction_of_drive=0.10,
        ),
        stp_config=STPConfig(U=0.10, tau_d=100.0, tau_f=500.0),
    )

    # PFC → Cerebellum GRANULE: goal / context input (similar pathway length)
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal_cortex",
            source_population=CortexPopulation.L5_PYR,
            target_region="cerebellum",
            target_population=CerebellumPopulation.GRANULE,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=25.0,
        axonal_delay_std_ms=7.5,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.05,
            target_tau_E_ms=2.5,
            target_v_inf=0.90,
            fraction_of_drive=0.08,  # Reduced 0.25→0.08: PFC→granule weight was 4× sensory weight (dominant driver); reduce to match sensory
        ),
        stp_config=STPConfig(U=0.10, tau_d=100.0, tau_f=500.0),
    )

    # Cerebellum DCN → Sensory cortex L4 PYR: motor predictions (via VL/VA thalamus)
    # Distance: ~8-12cm → 15-20ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="cerebellum",
            source_population=CerebellumPopulation.DCN,
            target_region="cortex_sensory",
            target_population=CortexPopulation.L4_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=17.5,
        axonal_delay_std_ms=5.25,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=50.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=1.05,
            fraction_of_drive=0.10,
        ),
        # DCN→cortex output: facilitating at baseline DCN rates
        # (Pugh & Raman 2006: DCN→VL thalamus facilitates at low firing rates).
        stp_config=STPConfig(U=0.1, tau_d=300.0, tau_f=300.0),
    )

    # Cerebellum DCN → Sensory cortex L4 PV: feedforward inhibition for motor predictions
    builder.connect(
        synapse_id=SynapseId(
            source_region="cerebellum",
            source_population=CerebellumPopulation.DCN,
            target_region="cortex_sensory",
            target_population=CortexPopulation.L4_INHIBITORY_PV,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=17.5,
        axonal_delay_std_ms=5.25,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=50.0,
            target_g_L=0.10,
            target_tau_E_ms=3.0,
            target_v_inf=0.95,
            fraction_of_drive=0.05,
        ),
        # Same DCN facilitating STP for PV feedforward.
        stp_config=STPConfig(U=0.1, tau_d=300.0, tau_f=300.0),
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
        axonal_delay_std_ms=1.2,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=10.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=0.95,
            fraction_of_drive=0.65,  # Increased 0.25→0.65: MSN D1/D2 at 0.5 Hz (target 8 Hz); source fires ~3 Hz vs 10 Hz design
        ),
        stp_config=STPConfig(U=0.4, tau_d=250.0, tau_f=150.0),
    )

    # Hippocampus CA1 → Striatum: hippocampostriatal pathway
    builder.connect_to_striatum(
        source_region="hippocampus",
        source_population=HippocampusPopulation.CA1,
        axonal_delay_ms=8.5,
        axonal_delay_std_ms=2.55,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=3.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=0.95,
            fraction_of_drive=0.35,  # Increased 0.13→0.35: hippocampal CA1 also underfires; 2.7× boost
        ),
        stp_config=STPConfig(U=0.5, tau_d=700.0, tau_f=400.0),
    )

    # PFC → Striatum: prefrontal corticostriatal pathway
    builder.connect_to_striatum(
        source_region="prefrontal_cortex",
        source_population=CortexPopulation.L5_PYR,
        axonal_delay_ms=15.0,
        axonal_delay_std_ms=4.5,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=0.95,
            fraction_of_drive=0.55,  # Increased 0.21→0.55: PFC L5 underfires; 2.6× boost
        ),
        stp_config=STPConfig(U=0.4, tau_d=250.0, tau_f=150.0),
    )

    # Thalamus → Striatum: thalamostriatal pathway for habitual responses
    # Direct sensory-action pathway bypassing cortex (Smith et al. 2004, 2009, 2014).
    # Strong THALAMO_STRIATAL depression (U=0.5) motivates the elevated base weight;
    # auto-correction compensates by using the actual steady-state u_eff.
    builder.connect_to_striatum(
        source_region="thalamus",
        source_population=ThalamusPopulation.RELAY,
        axonal_delay_ms=5.0,
        axonal_delay_std_ms=1.5,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=30.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=0.95,
            fraction_of_drive=0.70,
        ),
        stp_config=STPConfig(U=0.12, tau_d=200.0, tau_f=500.0),
    )

    # Association cortex L5 → Striatum: goal-directed corticostriatal projection
    # Slightly longer delay than sensory→striatum (additional cortical processing).
    builder.connect_to_striatum(
        source_region="cortex_association",
        source_population=CortexPopulation.L5_PYR,
        axonal_delay_ms=6.0,
        axonal_delay_std_ms=1.8,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=10.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=0.95,
            fraction_of_drive=0.55,  # Increased 0.21→0.55: assoc cortex underfires; 2.6× boost
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
    add_bg_circuit(
        builder,
        thalamus_name="thalamus",
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
        axonal_delay_std_ms=1.5,
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


# =============================================================================
# Corticocortical connections: two-column hierarchy + association column inputs
# =============================================================================

def _connect_corticocortical(builder: BrainBuilder) -> None:
    """Wire the inter-column corticocortical hierarchy.

    Implements the canonical predictive-coding FF/FB architecture
    (Felleman & Van Essen 1991; Bastos et al. 2012):

    * Sensory L2/3 → Association L4: feedforward (FF) percept transfer.
    * Association L6B → Sensory L2/3: feedback (FB) prediction.
    * Hippocampus CA1 → Association L2/3: episodic content to context.
    * PFC → Association L2/3: top-down executive modulation.
    * PFC → Association L5 apical: deep-layer top-down to assoc output neurons.
    * Association L6A → Thalamus TRN: corticothalamic attention gating.
    """
    # Sensory L2/3 → Association L4: feedforward percept transfer
    # Supragranular pyramidals project to granular layer of the next-higher area.
    # Distance: ~2-3cm, well-myelinated → 5ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_sensory",
            source_population=CortexPopulation.L23_PYR,
            target_region="cortex_association",
            target_population=CortexPopulation.L4_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=5.0,
        axonal_delay_std_ms=1.5,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=2.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=0.85,
            fraction_of_drive=0.30,
        ),
        stp_config=STPConfig(U=0.50, tau_d=600.0, tau_f=25.0),
    )

    # Association L6B → Sensory L2/3: top-down prediction feedback
    # Deep-layer → superficial-layer of lower area (canonical FB pathway).
    # Carries predictions; suppresses expected patterns (predictive coding).
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_association",
            source_population=CortexPopulation.L6B_PYR,
            target_region="cortex_sensory",
            target_population=CortexPopulation.L23_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=8.0,
        axonal_delay_std_ms=2.4,
        connectivity=0.2,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=1.05,
            fraction_of_drive=0.10,
        ),
        # Association L6B→Sensory L2/3: canonical top-down FB from deep layer
        # to superficial layer.  L6B origin is type-II CT-like (facilitating);
        # Markov et al. (2014) show FB connections are broadly facilitating.
        stp_config=STPConfig(U=0.08, tau_d=150.0, tau_f=800.0),
    )

    # Hippocampus CA1 → Association L2/3: retrieved episodic content to context
    # Retrieval arrives at association for integration with ongoing percepts.
    builder.connect(
        synapse_id=SynapseId(
            source_region="hippocampus",
            source_population=HippocampusPopulation.CA1,
            target_region="cortex_association",
            target_population=CortexPopulation.L23_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=6.5,
        axonal_delay_std_ms=1.95,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=3.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=1.05,
            fraction_of_drive=0.10,
        ),
        # CA1→Association L2/3: hippocampal output moderate depression.
        stp_config=STPConfig(U=0.5, tau_d=700.0, tau_f=400.0),
    )

    # PFC → Association L2/3: top-down executive modulation of higher representations
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal_cortex",
            source_population=CortexPopulation.L23_PYR,
            target_region="cortex_association",
            target_population=CortexPopulation.L23_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=12.0,
        axonal_delay_std_ms=3.6,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=1.05,
            fraction_of_drive=0.20,
        ),
        # PFC top-down to association L2/3: facilitating deep-layer FB.
        stp_config=STPConfig(U=0.1, tau_d=300.0, tau_f=300.0),
    )

    # PFC → Association L5 apical: deep-layer top-down context to assoc output neurons
    # Complements the L2/3 target: while PFC→L2/3 suppresses prediction errors,
    # PFC→L5 apical gates what association-cortex outputs reach subcortical targets
    # (striatum, thalamus).  L5 apical coincidence detection (Larkum 2013) requires
    # both L2/3 basal drive AND apical top-down input to produce burst firing.
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal_cortex",
            source_population=CortexPopulation.L23_PYR,
            target_region="cortex_association",
            target_population=CortexPopulation.L5_PYR,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=12.0,
        axonal_delay_std_ms=3.6,
        connectivity=0.25,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=1.05,
            fraction_of_drive=0.12,
        ),
        # PFC→assoc L5 apical: facilitating FB projection (type-II; Markov et al. 2014).
        stp_config=STPConfig(U=0.1, tau_d=300.0, tau_f=300.0),
    )

    # Association L6A → Thalamus TRN: corticothalamic attention control
    # Association cortex gates thalamic relay to shape L4 input in sensory column.
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_association",
            source_population=CortexPopulation.L6A_PYR,
            target_region="thalamus",
            target_population=ThalamusPopulation.TRN,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=12.0,
        axonal_delay_std_ms=3.6,
        connectivity=0.2,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.10,
            target_tau_E_ms=4.0,
            target_v_inf=1.0,
            fraction_of_drive=0.40,
        ),
    )


# =============================================================================
# Neuromodulator systems: NE (LC), ACh (NB), 5-HT (DRN)
# =============================================================================
# Note: the SNr → LHb → RMTg → VTA anti-reward cascade is now wired by
# add_bg_circuit() inside _connect_basal_ganglia() above.

def _connect_neuromodulators(builder: BrainBuilder) -> None:
    """Wire inputs into the three spike-based neuromodulator systems.

    * Locus Coeruleus (NE): PFC uncertainty + hippocampal novelty + fear.
    * Nucleus Basalis (ACh): PFC prediction error + BLA emotional salience.
    * Dorsal Raphe (5-HT): LHb punishment signal suppresses serotonin.
    """
    # --- Locus Coeruleus -------------------------------------------------------

    # PFC → LC NE: prefrontal variance signals uncertainty
    # High PFC activity variance → high LC firing → NE release.
    # Distance: ~3-5cm → 5-8ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal_cortex",
            source_population=CortexPopulation.L5_PYR,
            target_region="locus_coeruleus",
            target_population=LocusCoeruleusPopulation.NE,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=6.5,
        axonal_delay_std_ms=1.95,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.056,
            target_tau_E_ms=5.0,
            target_v_inf=0.95,
            fraction_of_drive=0.20,
        ),
        stp_config=STPConfig(U=0.50, tau_d=600.0, tau_f=25.0),
    )

    # Hippocampus CA1 → LC NE: novelty detection drives arousal
    # CA1 output variance indicates contextual novelty.
    # Distance: ~4-6cm → 8-12ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="hippocampus",
            source_population=HippocampusPopulation.CA1,
            target_region="locus_coeruleus",
            target_population=LocusCoeruleusPopulation.NE,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=10.0,
        axonal_delay_std_ms=3.0,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=3.0,
            target_g_L=0.056,
            target_tau_E_ms=5.0,
            target_v_inf=0.95,
            fraction_of_drive=0.10,
        ),
        stp_config=STPConfig(U=0.5, tau_d=700.0, tau_f=400.0),
    )

    # --- Nucleus Basalis -------------------------------------------------------

    # PFC → NB ACH: prefrontal activity changes signal prediction errors
    # Unexpected events drive encoding mode in cortex/hippocampus.
    # Distance: ~3-5cm → 5-8ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal_cortex",
            source_population=CortexPopulation.L5_PYR,
            target_region="nucleus_basalis",
            target_population=NucleusBasalisPopulation.ACH,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=6.5,
        axonal_delay_std_ms=1.95,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.083,
            target_tau_E_ms=4.0,
            target_v_inf=0.95,
            fraction_of_drive=0.25,
        ),
        stp_config=STPConfig(U=0.50, tau_d=600.0, tau_f=25.0),
    )

    # BLA PRINCIPAL → NB ACH: emotional salience drives ACh encoding-mode bursts
    # BLA principal neurons respond to unexpected / aversive stimuli (US, threat).
    # High BLA activity → strong prediction error → NB bursts ACh.
    # (Zaborszky et al. 2015; Holland & Gallagher 1999)
    # Distance: ~2-4cm → 5-8ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="basolateral_amygdala",
            source_population=BLAPopulation.PRINCIPAL,
            target_region="nucleus_basalis",
            target_population=NucleusBasalisPopulation.ACH,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=6.0,
        axonal_delay_std_ms=1.8,
        connectivity=0.2,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=3.0,
            target_g_L=0.083,
            target_tau_E_ms=4.0,
            target_v_inf=0.95,
            fraction_of_drive=0.28,
        ),
        stp_config=STPConfig(U=0.50, tau_d=700.0, tau_f=20.0),
    )

    # --- Dorsal Raphe Nucleus --------------------------------------------------

    # LHb → DRN SEROTONIN: punishment / negative RPE → 5-HT pause
    # LHb principal (glutamatergic) projects heavily to DRN.
    # High LHb activity → 5-HT pause (via local GABA interneurons in DRN).
    # (Vertes & Linley 2008; Stern et al. 2017)
    # Distance: ~1-2cm (adjacent midbrain) → 2-4ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="lateral_habenula",
            source_population=LHbPopulation.PRINCIPAL,
            target_region="dorsal_raphe",
            target_population=DRNPopulation.SEROTONIN,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=3.0,
        axonal_delay_std_ms=0.9,
        connectivity=0.5,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.067,
            target_tau_E_ms=5.0,
            target_v_inf=0.95,
            fraction_of_drive=0.40,
        ),
        stp_config=STPConfig(U=0.50, tau_d=700.0, tau_f=20.0),
    )


# =============================================================================
# Amygdala: BLA (fear/extinction) + CeA (fear output)
# =============================================================================

def _connect_amygdala(builder: BrainBuilder) -> None:
    """Wire the amygdala circuits.

    BLA inputs: sensory cortex (CS slow), thalamus (CS fast), hippocampus (context),
                PFC (top-down extinction regulation).
    BLA → CeA: transmit conditioned fear signal to output nucleus.
    CeA outputs: LC (arousal) and LHb (aversive RPE).
    """
    # Sensory cortex L5 → BLA PRINCIPAL: CS representation (slow, detailed pathway)
    # Auditory/somatosensory cortex provides the conditioned stimulus (CS) signal.
    # Distance: ~3-5cm → 6-10ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="cortex_sensory",
            source_population=CortexPopulation.L5_PYR,
            target_region="basolateral_amygdala",
            target_population=BLAPopulation.PRINCIPAL,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=8.0,
        axonal_delay_std_ms=2.4,
        connectivity=0.15,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=6.0,
            target_g_L=0.05,
            target_tau_E_ms=7.0,
            target_v_inf=1.05,
            fraction_of_drive=0.35,
        ),
        stp_config=None,
    )

    # Thalamus RELAY → BLA PRINCIPAL: fast CS pathway (thalamo-amygdalar shortcut)
    # Direct thalamic relay bypasses cortex (~12ms faster than cortical path).
    # Enables rapid fear conditioning before full cortical elaboration of CS.
    # Distance: ~2-3cm → ~5ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="thalamus",
            source_population=ThalamusPopulation.RELAY,
            target_region="basolateral_amygdala",
            target_population=BLAPopulation.PRINCIPAL,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=5.0,
        axonal_delay_std_ms=1.5,
        connectivity=0.2,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=30.0,
            target_g_L=0.05,
            target_tau_E_ms=7.0,
            target_v_inf=1.05,
            fraction_of_drive=0.40,
        ),
        stp_config=None,
    )

    # Hippocampus CA1 → BLA PRINCIPAL: contextual fear / extinction renewal
    # CA1 encodes spatial/temporal context; gates fear recall based on place-memory.
    # Distance: ~1-2cm (directly adjacent structures) → 3-5ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="hippocampus",
            source_population=HippocampusPopulation.CA1,
            target_region="basolateral_amygdala",
            target_population=BLAPopulation.PRINCIPAL,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=4.0,
        axonal_delay_std_ms=1.2,
        connectivity=0.2,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=2.0,
            target_g_L=0.05,
            target_tau_E_ms=7.0,
            target_v_inf=1.05,
            fraction_of_drive=0.15,
        ),
        stp_config=None,
    )

    # PFC → BLA SOM: top-down extinction regulation
    # Infralimbic PFC → BLA SOM interneurons inhibit principal neurons → extinction.
    # Distance: ~4-6cm → 6-10ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="prefrontal_cortex",
            source_population=CortexPopulation.L5_PYR,
            target_region="basolateral_amygdala",
            target_population=BLAPopulation.SOM,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=8.0,
        axonal_delay_std_ms=2.4,
        connectivity=0.2,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.06,
            target_tau_E_ms=8.0,
            target_v_inf=1.05,
            fraction_of_drive=1.0,
        ),
        stp_config=STPConfig(U=0.1, tau_d=300.0, tau_f=300.0),
    )

    # BLA PRINCIPAL → CeA LATERAL: core fear signal transmission
    # LA/BA principal neurons project to CeL, driving fear-ON cells.
    # Distance: ~0.5-1cm (within amygdaloid complex) → 2-3ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="basolateral_amygdala",
            source_population=BLAPopulation.PRINCIPAL,
            target_region="central_amygdala",
            target_population=CeAPopulation.LATERAL,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=2.5,
        axonal_delay_std_ms=0.75,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=3.0,
            target_g_L=0.06,
            target_tau_E_ms=6.0,
            target_v_inf=1.05,
            fraction_of_drive=0.60,
        ),
        stp_config=STPConfig(U=0.50, tau_d=700.0, tau_f=20.0),
    )

    # BLA PRINCIPAL → CeA MEDIAL: direct projection (bypasses CeL gating)
    # Some BLA principal neurons project directly to CeM for rapid fear output.
    # Distance: ~0.5-1cm → 2-3ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="basolateral_amygdala",
            source_population=BLAPopulation.PRINCIPAL,
            target_region="central_amygdala",
            target_population=CeAPopulation.MEDIAL,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=3.0,
        axonal_delay_std_ms=0.9,
        connectivity=0.2,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=3.0,
            target_g_L=0.05,
            target_tau_E_ms=6.0,
            target_v_inf=0.90,
            fraction_of_drive=0.45,
        ),
        stp_config=STPConfig(U=0.50, tau_d=700.0, tau_f=20.0),
    )

    # CeA MEDIAL → LC NE: fear-driven norepinephrine arousal
    # CeM activates LC during fear, driving NE release and sympathetic arousal.
    # Distance: ~3-5cm (amygdala → pons) → 5-10ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="central_amygdala",
            source_population=CeAPopulation.MEDIAL,
            target_region="locus_coeruleus",
            target_population=LocusCoeruleusPopulation.NE,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=7.0,
        axonal_delay_std_ms=2.1,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=3.0,
            target_g_L=0.056,
            target_tau_E_ms=5.0,
            target_v_inf=0.95,
            fraction_of_drive=0.20,
        ),
        stp_config=STPConfig(U=0.50, tau_d=700.0, tau_f=20.0),
    )

    # CeA MEDIAL → LHb: aversive prediction error signal
    # CeM output encodes expected punishment; drives LHb for negative RPE.
    # LHb will then activate RMTg → DA pause in VTA.
    # Distance: ~3-4cm → 5-8ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region="central_amygdala",
            source_population=CeAPopulation.MEDIAL,
            target_region="lateral_habenula",
            target_population=LHbPopulation.PRINCIPAL,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=6.0,
        axonal_delay_std_ms=1.8,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=3.0,
            target_g_L=0.08,
            target_tau_E_ms=5.0,
            target_v_inf=1.05,
            fraction_of_drive=0.25,
        ),
        stp_config=STPConfig(U=0.50, tau_d=700.0, tau_f=20.0),
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
    _connect_thalamocortical(builder)
    _connect_cortex_ec_hippocampus(builder)
    _connect_prefrontal(builder)
    _connect_cerebellum(builder)
    _connect_striatal_inputs(builder)
    _connect_basal_ganglia(builder)
    _connect_corticocortical(builder)
    _connect_neuromodulators(builder)
    _connect_amygdala(builder)
