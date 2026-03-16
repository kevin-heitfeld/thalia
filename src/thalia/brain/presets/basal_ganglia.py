"""Basal Ganglia (BG) circuit preset.

Wires the complete cortico-basal ganglia loop in a single call, including all
three output pathways and the anti-reward habenular circuit:

    - **Striatum**: D1/D2 MSNs + FSI + TAN (cholinergic interneurons).
    - **GPe** (globus pallidus externa): prototypic + arkypallidal sub-types.
    - **GPi** (globus pallidus interna): principal + border cells.
    - **STN** (subthalamic nucleus): glutamatergic pacemaker.
    - **SNr** (substantia nigra reticulata): GABAergic BG output nucleus.
    - **LHb** (lateral habenula): bad-outcome / negative-RPE signal.
    - **RMTg** (rostromedial tegmentum): GABAergic DA-pause mediator.

Internal pathways
-----------------
Direct:     D1 → SNr (GABA_A) — disinhibit thalamus / promote action.
Indirect:   D2 → GPe → STN → SNr — suppress thalamus / cancel action.
Hyperdirect input from cortex is an *external* connection; connect cortex L5 →
STN to enable it (not wired here because cortex is external to BG).
Pallido-pallidal: GPe → GPi, GPe → STN (inhibitory pacing loops).
Anti-reward: SNr → LHb → RMTg (negative RPE / DA-pause cascade).

Optional output connections
---------------------------
Pass ``thalamus_name`` and/or ``vta_name`` to extend the preset with the key
BG-output projections without having to manually replicate the calibrated
weight parameters:

  * GPi PRINCIPAL → Thalamus RELAY  (motor gating)
  * SNr VTA_FEEDBACK → Thalamus RELAY  (SNr tonically suppresses thalamus too)
  * SNr VTA_FEEDBACK → VTA DA_MESOLIMBIC  (value-feedback path for TD error)
  * RMTg GABA → VTA DA_MESOLIMBIC / DA_MESOCORTICAL  (GABAergic DA pause)

Usage
-----
Standalone preset brain::

    from thalia.brain import BrainBuilder
    brain = BrainBuilder.preset("basal_ganglia")

Embedded in a larger builder (inject-and-wire pattern)::

    from thalia.brain.presets.bg_preset import add_basal_ganglia_circuit
    add_basal_ganglia_circuit(builder, thalamus_name="thalamus", vta_name="vta")
    # then wire cortex → STN (hyperdirect) externally

Name-overriding::

    add_basal_ganglia_circuit(
        builder,
        striatum_name="striatum_dorsal",
        snr_name="snr_left",
        thalamus_name="thalamus",
        vta_name="vta",
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from thalia.brain.configs.basal_ganglia import (
    StriatumConfig,
    TonicPacemakerConfig,
)
from thalia.brain.regions.population_names import (
    GPePopulation,
    GPiPopulation,
    LHbPopulation,
    RMTgPopulation,
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
# Default population sizes
# =============================================================================

DEFAULT_BASAL_GANGLIA_SIZES: RegionSizes = {
    "striatum": {
        StriatumPopulation.D1: 200,   # Direct-pathway MSNs  (Go)
        StriatumPopulation.D2: 200,   # Indirect-pathway MSNs (NoGo)
        # FSI and TAN sizes are derived internally by Striatum if not specified.
    },
    "globus_pallidus_externa": {
        GPePopulation.PROTOTYPIC: 2000,   # ~75%: project to STN + SNr + GPi
        GPePopulation.ARKYPALLIDAL: 700,  # ~25%: project back to striatum
    },
    "globus_pallidus_interna": {
        GPiPopulation.PRINCIPAL: 1500,    # ~75%: thalamic gate (VA/VL/MD)
        GPiPopulation.BORDER_CELLS: 500,  # ~25%: value-coding
    },
    "subthalamic_nucleus": {
        STNPopulation.STN: 500,           # Glutamatergic pacemakers (~20 Hz)
    },
    "substantia_nigra": {
        SubstantiaNigraPopulation.VTA_FEEDBACK: 1000,  # SNr GABAergic output
    },
    "lateral_habenula": {
        LHbPopulation.PRINCIPAL: 500,     # Glutamatergic bad-outcome signal
    },
    "rostromedial_tegmentum": {
        RMTgPopulation.GABA: 1000,        # GABAergic DA-pause mediator
    },
}


def resolve_basal_ganglia_sizes(
    overrides: Dict[str, Any],
    str_name: str,
    gpe_name: str,
    gpi_name: str,
    stn_name: str,
    snr_name: str,
    lhb_name: str,
    rmtg_name: str,
) -> RegionSizes:
    size_overrides: RegionSizes = overrides.get("population_sizes", {})

    canonical_map = {
        str_name:  "striatum",
        gpe_name:  "globus_pallidus_externa",
        gpi_name:  "globus_pallidus_interna",
        stn_name:  "subthalamic_nucleus",
        snr_name:  "substantia_nigra",
        lhb_name:  "lateral_habenula",
        rmtg_name: "rostromedial_tegmentum",
    }

    result: RegionSizes = {}
    for instance, canonical in canonical_map.items():
        defaults = dict(DEFAULT_BASAL_GANGLIA_SIZES[canonical])
        user = size_overrides.get(instance, size_overrides.get(canonical, {}))
        result[instance] = {**defaults, **user}
    return result


# =============================================================================
# Add regions
# =============================================================================

def add_basal_ganglia_regions(
    builder: BrainBuilder,
    sizes: RegionSizes,
    str_name: str,
    gpe_name: str,
    gpi_name: str,
    stn_name: str,
    snr_name: str,
    lhb_name: str,
    rmtg_name: str,
) -> None:
    builder.add_region(
        gpe_name, "globus_pallidus_externa",
        population_sizes=sizes[gpe_name],
        config=TonicPacemakerConfig(baseline_drive=0.011),
    )
    builder.add_region(
        gpi_name, "globus_pallidus_interna",
        population_sizes=sizes[gpi_name],
        config=TonicPacemakerConfig(baseline_drive=0.012),
    )
    builder.add_region(
        lhb_name, "lateral_habenula",
        population_sizes=sizes[lhb_name],
        config=TonicPacemakerConfig(baseline_drive=0.007, tau_mem_ms=20.0),
    )
    builder.add_region(
        rmtg_name, "rostromedial_tegmentum",
        population_sizes=sizes[rmtg_name],
        config=TonicPacemakerConfig(baseline_drive=0.004),
    )
    builder.add_region(
        str_name, "striatum",
        population_sizes=sizes[str_name],
        config=StriatumConfig(),
    )
    builder.add_region(
        snr_name, "substantia_nigra",
        population_sizes=sizes[snr_name],
        config=TonicPacemakerConfig(baseline_drive=0.015, v_threshold=1.25),
    )
    builder.add_region(
        stn_name, "subthalamic_nucleus",
        population_sizes=sizes[stn_name],
        config=TonicPacemakerConfig(baseline_drive=0.007, tau_mem_ms=18.0, i_h_conductance=0.0006),
    )


# =============================================================================
# Internal BG connections
# =============================================================================

def _connect_direct_pathway(
    builder: BrainBuilder,
    str_name: str,
    snr_name: str,
    gpi_name: str,
) -> None:
    """D1 MSNs → SNr and D1 MSNs → GPi (Go signal)."""
    # Striatum D1 → SNr: direct pathway
    # Monosynaptic GABAergic inhibition; dis-inhibits thalamus on Go.
    # Distance: ~1-2 cm → 2.5 ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region=str_name,
            source_population=StriatumPopulation.D1,
            target_region=snr_name,
            target_population=SubstantiaNigraPopulation.VTA_FEEDBACK,
            receptor_type=ReceptorType.GABA_A,
        ),
        axonal_delay_ms=2.5,
        axonal_delay_std_ms=0.75,
        connectivity=0.6,
        weight_scale=0.0008,
        stp_config=STPConfig(U=0.45, tau_d=500.0, tau_f=25.0),
    )

    # Striatum D1 → GPi PRINCIPAL: parallel direct pathway
    # Same Go logic as D1→SNr; gates motor (VA/VL) thalamus.
    builder.connect(
        synapse_id=SynapseId(
            source_region=str_name,
            source_population=StriatumPopulation.D1,
            target_region=gpi_name,
            target_population=GPiPopulation.PRINCIPAL,
            receptor_type=ReceptorType.GABA_A,
        ),
        axonal_delay_ms=3.0,
        axonal_delay_std_ms=0.9,
        connectivity=0.6,
        weight_scale=0.0008,
        stp_config=STPConfig(U=0.45, tau_d=500.0, tau_f=25.0),
    )


def _connect_indirect_pathway(
    builder: BrainBuilder,
    str_name: str,
    gpe_name: str,
    stn_name: str,
    snr_name: str,
    gpi_name: str,
) -> None:
    """D2 MSNs → GPe → STN → SNr/GPi (NoGo / action suppression)."""
    # Striatum D2 → GPe PROTOTYPIC
    builder.connect(
        synapse_id=SynapseId(
            source_region=str_name,
            source_population=StriatumPopulation.D2,
            target_region=gpe_name,
            target_population=GPePopulation.PROTOTYPIC,
            receptor_type=ReceptorType.GABA_A,
        ),
        axonal_delay_ms=3.0,
        axonal_delay_std_ms=0.9,
        connectivity=0.5,
        weight_scale=0.002,
        stp_config=STPConfig(U=0.45, tau_d=500.0, tau_f=25.0),
    )

    # Striatum D2 → GPe ARKYPALLIDAL (Mallet et al. 2012)
    builder.connect(
        synapse_id=SynapseId(
            source_region=str_name,
            source_population=StriatumPopulation.D2,
            target_region=gpe_name,
            target_population=GPePopulation.ARKYPALLIDAL,
            receptor_type=ReceptorType.GABA_A,
        ),
        axonal_delay_ms=3.0,
        axonal_delay_std_ms=0.9,
        connectivity=0.4,
        weight_scale=0.002,
        stp_config=STPConfig(U=0.45, tau_d=500.0, tau_f=25.0),
    )

    # GPe PROTOTYPIC → STN: inhibitory pacing (GABA_A)
    builder.connect(
        synapse_id=SynapseId(
            source_region=gpe_name,
            source_population=GPePopulation.PROTOTYPIC,
            target_region=stn_name,
            target_population=STNPopulation.STN,
            receptor_type=ReceptorType.GABA_A,
        ),
        axonal_delay_ms=4.0,
        axonal_delay_std_ms=1.2,
        connectivity=0.5,
        weight_scale=0.00017,
        stp_config=STPConfig(U=0.25, tau_d=200.0, tau_f=20.0),
    )

    # GPe PROTOTYPIC → STN: slow GABA_B component
    # Late-onset, long-duration STN suppression; critical for beta-band power.
    # Weight calibrated to ~30% of GABA-A component (0.00017 × 0.29 ≈ 0.000050).
    # Previous value 0.0000013 was 130× too small — diagnostics confirmed μ ≈ 0.
    builder.connect(
        synapse_id=SynapseId(
            source_region=gpe_name,
            source_population=GPePopulation.PROTOTYPIC,
            target_region=stn_name,
            target_population=STNPopulation.STN,
            receptor_type=ReceptorType.GABA_B,
        ),
        axonal_delay_ms=4.0,
        axonal_delay_std_ms=1.2,
        connectivity=0.4,
        weight_scale=0.000050,
        stp_config=STPConfig(U=0.25, tau_d=350.0, tau_f=30.0),
    )

    # STN → GPe PROTOTYPIC: excitatory feedback (closes GPe-STN oscillatory loop)
    builder.connect(
        synapse_id=SynapseId(
            source_region=stn_name,
            source_population=STNPopulation.STN,
            target_region=gpe_name,
            target_population=GPePopulation.PROTOTYPIC,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=4.0,
        axonal_delay_std_ms=1.2,
        connectivity=0.5,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=20.0,
            target_g_L=0.10,
            target_tau_E_ms=5.0,
            target_v_inf=1.05,
            fraction_of_drive=0.13,
        ),
        stp_config=STPConfig(U=0.30, tau_d=400.0, tau_f=20.0),
    )

    # STN → GPe ARKYPALLIDAL: closes STN-arky loop
    builder.connect(
        synapse_id=SynapseId(
            source_region=stn_name,
            source_population=STNPopulation.STN,
            target_region=gpe_name,
            target_population=GPePopulation.ARKYPALLIDAL,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=4.0,
        axonal_delay_std_ms=1.2,
        connectivity=0.3,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=20.0,
            target_g_L=0.10,
            target_tau_E_ms=5.0,
            target_v_inf=1.05,
            fraction_of_drive=0.08,
        ),
        stp_config=STPConfig(U=0.30, tau_d=400.0, tau_f=20.0),
    )

    # STN → SNr: excitatory indirect output
    builder.connect(
        synapse_id=SynapseId(
            source_region=stn_name,
            source_population=STNPopulation.STN,
            target_region=snr_name,
            target_population=SubstantiaNigraPopulation.VTA_FEEDBACK,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=5.0,
        axonal_delay_std_ms=1.5,
        connectivity=0.5,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=20.0,
            target_g_L=0.10,
            target_tau_E_ms=5.0,
            target_v_inf=1.10,
            fraction_of_drive=0.30,
        ),
        stp_config=STPConfig(U=0.30, tau_d=400.0, tau_f=20.0),
    )

    # STN → GPi PRINCIPAL: hyperdirect path reaches GPi too
    builder.connect(
        synapse_id=SynapseId(
            source_region=stn_name,
            source_population=STNPopulation.STN,
            target_region=gpi_name,
            target_population=GPiPopulation.PRINCIPAL,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=5.0,
        axonal_delay_std_ms=1.5,
        connectivity=0.5,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=20.0,
            target_g_L=0.10,
            target_tau_E_ms=5.0,
            target_v_inf=1.10,
            fraction_of_drive=0.25,
        ),
        stp_config=STPConfig(U=0.30, tau_d=400.0, tau_f=20.0),
    )

    # STN → GPi BORDER_CELLS: value-coding border cells require excitatory drive.
    # No excitatory input was wired here; without it they sit at V_inf ≈ 0.85
    # (sub-threshold) from baseline drive alone and fire at 0 Hz.
    # fraction_of_drive=0.10: conservative because STN currently bursts at ~78%
    # (due to absent D1/D2 inhibition cascade). When BG is healthy this gives
    # V_inf ≈ 1.0 at 20 Hz STN → border_cells fire tonically at design 30-80 Hz.
    builder.connect(
        synapse_id=SynapseId(
            source_region=stn_name,
            source_population=STNPopulation.STN,
            target_region=gpi_name,
            target_population=GPiPopulation.BORDER_CELLS,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=5.0,
        axonal_delay_std_ms=1.5,
        connectivity=0.4,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=20.0,
            target_g_L=0.10,
            target_tau_E_ms=5.0,
            target_v_inf=1.10,
            fraction_of_drive=0.10,
        ),
        stp_config=STPConfig(U=0.30, tau_d=400.0, tau_f=20.0),
    )

    # GPe PROTOTYPIC → SNr: pallido-nigral inhibitory bypass
    builder.connect(
        synapse_id=SynapseId(
            source_region=gpe_name,
            source_population=GPePopulation.PROTOTYPIC,
            target_region=snr_name,
            target_population=SubstantiaNigraPopulation.VTA_FEEDBACK,
            receptor_type=ReceptorType.GABA_A,
        ),
        axonal_delay_ms=4.0,
        axonal_delay_std_ms=1.2,
        connectivity=0.4,
        weight_scale=0.00024,
        stp_config=STPConfig(U=0.25, tau_d=200.0, tau_f=25.0),
    )

    # GPe PROTOTYPIC → GPi PRINCIPAL: pallido-pallidal inhibitory pacing
    builder.connect(
        synapse_id=SynapseId(
            source_region=gpe_name,
            source_population=GPePopulation.PROTOTYPIC,
            target_region=gpi_name,
            target_population=GPiPopulation.PRINCIPAL,
            receptor_type=ReceptorType.GABA_A,
        ),
        axonal_delay_ms=3.0,
        axonal_delay_std_ms=0.9,
        connectivity=0.4,
        weight_scale=0.00020,
        stp_config=STPConfig(U=0.25, tau_d=200.0, tau_f=20.0),
    )


def _connect_anti_reward_pathway(
    builder: BrainBuilder,
    snr_name: str,
    lhb_name: str,
    rmtg_name: str,
) -> None:
    """Wire the negative RPE cascade: SNr → LHb → RMTg.

    Completes the anti-reward pathway within the BG circuit.  Output
    connections from RMTg → VTA are wired separately when vta_name is provided.
    (Hong et al. 2011 Nature; Matsumoto & Hikosaka 2007)
    """
    # SNr → LHb: high SNr activity = bad outcome → LHb excited
    # source_rate_hz=70 Hz matches SNr tonic rate for correct weight calibration.
    # No STP: depressing STP at 70 Hz depletes to <5%, nullifying drive.
    # fraction_of_drive raised 0.03→0.25: previous value computed w≈0.000014 per
    # synapse (diagnostics: μ≈0, max=0.00001) — LHb received no meaningful drive.
    builder.connect(
        synapse_id=SynapseId(
            source_region=snr_name,
            source_population=SubstantiaNigraPopulation.VTA_FEEDBACK,
            target_region=lhb_name,
            target_population=LHbPopulation.PRINCIPAL,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=3.0,
        axonal_delay_std_ms=0.9,
        connectivity=0.5,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=70.0,
            target_g_L=0.08,
            target_tau_E_ms=5.0,
            target_v_inf=1.05,
            fraction_of_drive=0.25,
        ),
        # Low U + fast τ_d sustains ~64% efficacy at 70 Hz tonic firing.
        stp_config=STPConfig(U=0.08, tau_d=100.0, tau_f=20.0),
    )

    # LHb → RMTg: activates GABAergic DA-pause mediator
    # Facilitating burst-driven synapse (Hong et al. 2011).
    builder.connect(
        synapse_id=SynapseId(
            source_region=lhb_name,
            source_population=LHbPopulation.PRINCIPAL,
            target_region=rmtg_name,
            target_population=RMTgPopulation.GABA,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=2.0,
        axonal_delay_std_ms=0.6,
        connectivity=0.6,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=20.0,
            target_g_L=0.10,
            target_tau_E_ms=5.0,
            target_v_inf=0.95,
            fraction_of_drive=0.40,
        ),
        stp_config=STPConfig(U=0.15, tau_d=200.0, tau_f=250.0),
    )


def _connect_bg_output_to_thalamus(
    builder: BrainBuilder,
    gpi_name: str,
    snr_name: str,
    thalamus_name: str,
) -> None:
    """Wire GPi → Thalamus and SNr → Thalamus inhibitory gating.

    GPi tonically suppresses thalamic relay cells (VA/VL and MD nuclei).
    Disinhibited by D1 Go signal; re-suppressed by indirect / hyperdirect paths.
    """
    # GPi PRINCIPAL → Thalamus RELAY
    builder.connect(
        synapse_id=SynapseId(
            source_region=gpi_name,
            source_population=GPiPopulation.PRINCIPAL,
            target_region=thalamus_name,
            target_population=ThalamusPopulation.RELAY,
            receptor_type=ReceptorType.GABA_A,
        ),
        axonal_delay_ms=3.0,
        axonal_delay_std_ms=0.9,
        connectivity=0.5,
        weight_scale=0.0010,
        stp_config=STPConfig(U=0.20, tau_d=200.0, tau_f=50.0),
    )


def _connect_bg_output_to_vta(
    builder: BrainBuilder,
    snr_name: str,
    rmtg_name: str,
    vta_name: str,
) -> None:
    """Wire SNr → VTA value-feedback and RMTg → VTA DA-pause output.

    SNr → VTA provides the V(s) signal for the full TD error δ = r + γV(s') − V(s).
    RMTg → VTA provides the GABAergic DA pause for negative RPE.
    """
    # SNr → VTA DA_MESOLIMBIC: tonic GABAergic suppression of DA cells.
    # Weight raised 0.00001→0.0005: previous value was 50× too small (diagnostics:
    # μ≈0, max=0.00001) — SNr provided no inhibitory gating of VTA DA cells.
    builder.connect(
        synapse_id=SynapseId(
            source_region=snr_name,
            source_population=SubstantiaNigraPopulation.VTA_FEEDBACK,
            target_region=vta_name,
            target_population=VTAPopulation.DA_MESOLIMBIC,
            receptor_type=ReceptorType.GABA_A,
        ),
        axonal_delay_ms=1.5,
        axonal_delay_std_ms=0.5,
        connectivity=0.4,
        weight_scale=0.0005,
        # Low U + fast τ_d sustains ~64% efficacy at 70 Hz tonic firing.
        stp_config=STPConfig(U=0.08, tau_d=100.0, tau_f=20.0),
    )

    # RMTg → VTA DA_MESOLIMBIC: GABAergic pause
    builder.connect(
        synapse_id=SynapseId(
            source_region=rmtg_name,
            source_population=RMTgPopulation.GABA,
            target_region=vta_name,
            target_population=VTAPopulation.DA_MESOLIMBIC,
            receptor_type=ReceptorType.GABA_A,
        ),
        axonal_delay_ms=1.5,
        axonal_delay_std_ms=0.5,
        connectivity=0.7,
        weight_scale=0.0005,
        stp_config=STPConfig(U=0.30, tau_d=350.0, tau_f=20.0),
    )

    # RMTg → VTA DA_MESOCORTICAL: same pause, mesocortical sub-population
    builder.connect(
        synapse_id=SynapseId(
            source_region=rmtg_name,
            source_population=RMTgPopulation.GABA,
            target_region=vta_name,
            target_population=VTAPopulation.DA_MESOCORTICAL,
            receptor_type=ReceptorType.GABA_A,
        ),
        axonal_delay_ms=1.5,
        axonal_delay_std_ms=0.5,
        connectivity=0.7,
        weight_scale=0.0005,
        stp_config=STPConfig(U=0.30, tau_d=350.0, tau_f=20.0),
    )


# =============================================================================
# Public entry points
# =============================================================================

def add_basal_ganglia_circuit(
    builder: BrainBuilder,
    *,
    striatum_name: str = "striatum",
    gpe_name: str = "globus_pallidus_externa",
    gpi_name: str = "globus_pallidus_interna",
    stn_name: str = "subthalamic_nucleus",
    snr_name: str = "substantia_nigra",
    lhb_name: str = "lateral_habenula",
    rmtg_name: str = "rostromedial_tegmentum",
    thalamus_name: Optional[str] = None,
    vta_name: Optional[str] = None,
    population_sizes: Optional[RegionSizes] = None,
    add_regions: bool = True,
) -> None:
    """Add the BG circuit to an existing *builder* and wire all internal connections.

    All seven core BG regions are added (striatum, GPe, GPi, STN, SNr, LHb, RMTg)
    and their internal connections wired.  Output connections to thalamus and VTA
    are only added when the corresponding name arguments are supplied — the caller
    is responsible for having already added those regions to the builder.

    Args:
        builder: :class:`~thalia.brain.brain_builder.BrainBuilder` to modify.
        striatum_name / gpe_name / … : Instance names for BG regions.
        thalamus_name: If provided, adds GPi → thalamus inhibitory gating.
            The named region must already be registered in *builder*.
        vta_name: If provided, adds SNr → VTA value-feedback and
            RMTg → VTA DA-pause connections.  The named region must already
            be registered in *builder*.
        population_sizes: Optional ``RegionSizes`` dict overriding default
            population sizes for BG regions.
        add_regions: When ``False``, skip adding BG regions to *builder* and
            only wire the internal connections.  Use this when the regions
            have already been registered (e.g. inside the ``"default"`` preset).
    """
    if add_regions:
        overrides: Dict[str, Any] = {}
        if population_sizes is not None:
            overrides["population_sizes"] = population_sizes
        sizes = resolve_basal_ganglia_sizes(
            overrides,
            striatum_name, gpe_name, gpi_name, stn_name, snr_name, lhb_name, rmtg_name,
        )
        add_basal_ganglia_regions(
            builder, sizes,
            striatum_name, gpe_name, gpi_name, stn_name, snr_name, lhb_name, rmtg_name,
        )

    _connect_direct_pathway(builder, striatum_name, snr_name, gpi_name)
    _connect_indirect_pathway(builder, striatum_name, gpe_name, stn_name, snr_name, gpi_name)
    _connect_anti_reward_pathway(builder, snr_name, lhb_name, rmtg_name)

    if thalamus_name is not None:
        _connect_bg_output_to_thalamus(builder, gpi_name, snr_name, thalamus_name)

    if vta_name is not None:
        _connect_bg_output_to_vta(builder, snr_name, rmtg_name, vta_name)


def build(builder: BrainBuilder, **overrides: Any) -> None:
    """Standalone BG preset: striatum + GPe + GPi + STN + SNr + LHb + RMTg.

    Registered as the ``"basal_ganglia"`` preset in
    :class:`~thalia.brain.brain_builder.BrainBuilder`.  Does **not** include
    thalamus or VTA; internal BG circuit only.

    Example::

        brain = BrainBuilder.preset("basal_ganglia")

    To also wire outputs (thalamus and VTA must be added manually)::

        builder = BrainBuilder.preset_builder("basal_ganglia")
        builder.add_region("thalamus", "thalamus", ...)
        builder.add_region("vta", "vta", ...)
        add_basal_ganglia_circuit(builder, thalamus_name="thalamus", vta_name="vta")
        brain = builder.build()
    """
    add_basal_ganglia_circuit(
        builder,
        population_sizes=overrides.get("population_sizes"),
    )
