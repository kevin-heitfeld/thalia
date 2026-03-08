"""Medial Temporal Lobe (MTL) circuit preset.

Wires the trisynaptic hippocampal loop together with its septal theta pacemaker
and entorhinal gateway in a single call:

    - **Medial Septum (MS)**: Cholinergic + GABAergic theta pacemaker.
    - **Entorhinal Cortex (EC)**: Grid/time-cell input gateway (EC_II → DG/CA3,
      EC_III → CA1) and memory-index output (EC_V ← CA1).
    - **Hippocampus (HPC)**: Full trisynaptic circuit (DG → CA3 → CA2 → CA1).
    - **Subiculum (Sub)** *(optional)*: Hippocampal output relay (CA1 → Sub → EC_V).

Internal connections
--------------------
MS ↔ HPC (septal theta loop):
  * MS GABA → HPC CA3 (phase-locking GABAergic; Freund & Antal 1988)
  * HPC CA1 → MS GABA (feedback inhibition; septo-hippocampal loop closure)

EC → HPC (afferent input):
  * EC_II → HPC DG  (perforant path, outer molecular layer; depressing STP)
  * EC_II → HPC CA3 (direct EC→CA3; stratum lacunosum-moleculare)
  * EC_III → HPC CA1 (temporoammonic direct path; depressing STP)

HPC → EC (back-projection):
  * HPC CA1 → EC_V  (or Sub PRINCIPAL → EC_V when subiculum is included)

Optional Subiculum stage (enable with ``include_subiculum=True``):
  * HPC CA1 → Sub PRINCIPAL (burst-to-regular conversion relay)
  * Sub PRINCIPAL → EC_V   (CA1-EC relay; replaces direct CA1→EC_V)

Usage
-----
Standalone preset brain::

    from thalia.brain import BrainBuilder
    brain = BrainBuilder.preset("mtl")

Embedded in a larger builder (inject-and-wire pattern)::

    from thalia.brain.presets.mtl_preset import add_mtl_circuit
    add_mtl_circuit(builder)
    # now add external cortex→EC and EC→cortex connections as needed

Name-overriding (for multi-hemisphere models)::

    add_mtl_circuit(
        builder,
        medial_septum_name="ms_left",
        entorhinal_cortex_name="ec_left",
        hippocampus_name="hpc_left",
        subiculum_name="sub_left",
        include_subiculum=True,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from thalia.brain.regions.population_names import (
    ECPopulation,
    HippocampusPopulation,
    MedialSeptumPopulation,
    SubiculumPopulation,
)
from thalia.brain.synapses import ConductanceScaledSpec
from thalia.brain.synapses.stp import (
    PERFORANT_PATH_PRESET,
    PV_BASKET_PRESET,
    SCHAFFER_COLLATERAL_PRESET,
    TEMPOROAMMONIC_PRESET,
)
from thalia.typing import PopulationSizes, RegionSizes, ReceptorType, SynapseId

if TYPE_CHECKING:
    from thalia.brain.brain_builder import BrainBuilder


# =============================================================================
# Default population sizes
# =============================================================================

_DEFAULT_MTL_SIZES: RegionSizes = {
    "medial_septum": {
        MedialSeptumPopulation.ACH: 200,   # Cholinergic pacemaker neurons
        MedialSeptumPopulation.GABA: 200,  # GABAergic phase-locking neurons
    },
    "entorhinal_cortex": {
        ECPopulation.EC_II: 400,   # Stellate cells: grid/place → DG, CA3
        ECPopulation.EC_III: 300,  # Pyramidal time cells → CA1
        ECPopulation.EC_V: 200,    # Output back-projection ← CA1 → neocortex
    },
    "hippocampus": {
        HippocampusPopulation.DG: 500,   # Dentate gyrus (pattern separation)
        HippocampusPopulation.CA3: 250,  # Cornu ammonis 3 (pattern completion)
        HippocampusPopulation.CA2: 75,   # Social memory sub-field
        HippocampusPopulation.CA1: 375,  # Comparison / output stage
    },
    "subiculum": {
        SubiculumPopulation.PRINCIPAL: 400,  # Pyramidal relay (burst→regular)
    },
}


def _resolve_mtl_sizes(
    overrides: Dict[str, Any],
    ms_name: str,
    ec_name: str,
    hpc_name: str,
    sub_name: str,
) -> RegionSizes:
    """Merge caller population-size overrides into MTL defaults.

    Returns a dict keyed by the *instance* names provided by the caller, so
    the result can be passed directly to ``builder.add_region()``.
    """
    size_overrides: RegionSizes = overrides.get("population_sizes", {})

    def _merge(canonical: str, instance: str) -> PopulationSizes:
        defaults = dict(_DEFAULT_MTL_SIZES[canonical])
        user = size_overrides.get(instance, size_overrides.get(canonical, {}))
        return {**defaults, **user}

    return {
        ms_name:  _merge("medial_septum",    ms_name),
        ec_name:  _merge("entorhinal_cortex", ec_name),
        hpc_name: _merge("hippocampus",       hpc_name),
        sub_name: _merge("subiculum",         sub_name),
    }


# =============================================================================
# Add regions
# =============================================================================

def _add_mtl_regions(
    builder: BrainBuilder,
    sizes: RegionSizes,
    ms_name: str,
    ec_name: str,
    hpc_name: str,
    sub_name: str,
    include_subiculum: bool,
) -> None:
    builder.add_region(ms_name,  "medial_septum",    population_sizes=sizes[ms_name])
    builder.add_region(ec_name,  "entorhinal_cortex", population_sizes=sizes[ec_name])
    builder.add_region(hpc_name, "hippocampus",       population_sizes=sizes[hpc_name])
    if include_subiculum:
        builder.add_region(sub_name, "subiculum",     population_sizes=sizes[sub_name])


# =============================================================================
# Internal connections
# =============================================================================

def _connect_septal_theta_loop(
    builder: BrainBuilder,
    ms_name: str,
    hpc_name: str,
) -> None:
    """Wire the septo-hippocampal theta pacemaker loop.

    MS GABAergic neurons fire at theta-rhythm (~8 Hz), phase-locking
    hippocampal OLM interneurons and producing the 8 Hz LFP theta oscillation.
    CA1 feedback closes the loop and prevents runaway septal drive.
    """
    # MS GABA → HPC CA3: septal theta drive
    # GABAergic pacemaker phase-locks hippocampal OLM interneurons.
    # Well-myelinated; distance ~1-2 cm → 2 ms delay.
    # STP: strong depressing PV basket-like (Freund & Antal 1988; Varga et al. 2008).
    builder.connect(
        synapse_id=SynapseId(
            source_region=ms_name,
            source_population=MedialSeptumPopulation.GABA,
            target_region=hpc_name,
            target_population=HippocampusPopulation.CA3,
            receptor_type=ReceptorType.GABA_A,
        ),
        axonal_delay_ms=2.0,
        axonal_delay_std_ms=3.0,
        connectivity=0.15,
        weight_scale=0.0009,
        stp_config=PV_BASKET_PRESET.configure(),
    )

    # HPC CA1 → MS GABA: hippocampal feedback closure
    # Suppresses septal drive when hippocampus is hyperactive.
    # Distance ~1-2 cm → 2 ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region=hpc_name,
            source_population=HippocampusPopulation.CA1,
            target_region=ms_name,
            target_population=MedialSeptumPopulation.GABA,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=2.0,
        axonal_delay_std_ms=3.0,
        connectivity=0.20,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=3.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=1.05,
            fraction_of_drive=0.25,
        ),
        stp_config=SCHAFFER_COLLATERAL_PRESET.configure(),
    )


def _connect_ec_to_hpc(
    builder: BrainBuilder,
    ec_name: str,
    hpc_name: str,
) -> None:
    """Wire the entorhinal–hippocampal afferent input pathways.

    Three parallel routes carry information into the hippocampal formation:
    1. Perforant path (EC_II → DG, EC_II → CA3): sparse encoding input.
    2. Temporoammonic path (EC_III → CA1): direct novelty/mismatch input.
    """
    # EC_II → HPC DG: perforant path — outer molecular layer
    # Sparse 15-20% connectivity; depressing STP privileges encoding onset.
    # (McNaughton 1980; Bortolotto et al. 2003)
    builder.connect(
        synapse_id=SynapseId(
            source_region=ec_name,
            source_population=ECPopulation.EC_II,
            target_region=hpc_name,
            target_population=HippocampusPopulation.DG,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=3.0,
        axonal_delay_std_ms=4.5,
        connectivity=0.25,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=0.90,
            fraction_of_drive=0.75,
        ),
        stp_config=PERFORANT_PATH_PRESET.configure(),
    )

    # EC_II → HPC CA3: direct perforant path (stratum lacunosum-moleculare)
    # Higher target_v_inf (1.20) so EC_II alone can drive CA3 to threshold
    # when DG mossy fibre drive is near-zero at tonic firing rates.
    builder.connect(
        synapse_id=SynapseId(
            source_region=ec_name,
            source_population=ECPopulation.EC_II,
            target_region=hpc_name,
            target_population=HippocampusPopulation.CA3,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=3.5,
        axonal_delay_std_ms=5.0,
        connectivity=0.20,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=1.20,
            fraction_of_drive=0.65,
        ),
        stp_config=PERFORANT_PATH_PRESET.configure(),
    )

    # EC_III → HPC CA1: temporoammonic direct path (distal apical dendrites)
    # Stronger depression than perforant path; emphasises novelty detection.
    # (Otmakhova et al. 2002)
    builder.connect(
        synapse_id=SynapseId(
            source_region=ec_name,
            source_population=ECPopulation.EC_III,
            target_region=hpc_name,
            target_population=HippocampusPopulation.CA1,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=4.0,
        axonal_delay_std_ms=6.0,
        connectivity=0.25,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=1.15,
            fraction_of_drive=0.55,
        ),
        stp_config=TEMPOROAMMONIC_PRESET.configure(),
    )


def _connect_hpc_to_ec_direct(
    builder: BrainBuilder,
    hpc_name: str,
    ec_name: str,
) -> None:
    """Wire CA1 → EC_V back-projection (without subiculum relay)."""
    builder.connect(
        synapse_id=SynapseId(
            source_region=hpc_name,
            source_population=HippocampusPopulation.CA1,
            target_region=ec_name,
            target_population=ECPopulation.EC_V,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=3.0,
        axonal_delay_std_ms=4.5,
        connectivity=0.30,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=3.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=1.05,
            fraction_of_drive=0.80,
        ),
        # CA1 → EC_V: moderate depression (Jones 1993; Bhattacharyya 2009).
        stp_config=SCHAFFER_COLLATERAL_PRESET.configure(),
    )


def _connect_hpc_via_subiculum(
    builder: BrainBuilder,
    hpc_name: str,
    sub_name: str,
    ec_name: str,
) -> None:
    """Wire CA1 → Subiculum → EC_V back-projection (with subiculum relay).

    The subiculum converts CA1 complex-spike bursts to regular spiking before
    broadcasting to EC_V.  Biologically more accurate than the direct CA1→EC_V
    shortcut; adds ~2 ms latency.
    """
    # HPC CA1 → Subiculum PRINCIPAL: burst-to-regular conversion
    # Strong excitatory drive; subiculum threshold neurons add adaptation.
    # Distance ~0.5 cm within hippocampal formation → 1 ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region=hpc_name,
            source_population=HippocampusPopulation.CA1,
            target_region=sub_name,
            target_population=SubiculumPopulation.PRINCIPAL,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=1.5,
        axonal_delay_std_ms=2.0,
        connectivity=0.40,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=3.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=1.10,
            fraction_of_drive=0.85,
        ),
        # CA1→Sub: Schaffer-collateral-like moderate depression.
        stp_config=SCHAFFER_COLLATERAL_PRESET.configure(),
    )

    # Subiculum PRINCIPAL → EC_V: back-projection from hippocampal gateway
    # Subicular axons ascend to entorhinal layer V; ~2-3 ms delay.
    builder.connect(
        synapse_id=SynapseId(
            source_region=sub_name,
            source_population=SubiculumPopulation.PRINCIPAL,
            target_region=ec_name,
            target_population=ECPopulation.EC_V,
            receptor_type=ReceptorType.AMPA,
        ),
        axonal_delay_ms=2.5,
        axonal_delay_std_ms=3.5,
        connectivity=0.35,
        weight_scale=ConductanceScaledSpec(
            source_rate_hz=5.0,
            target_g_L=0.05,
            target_tau_E_ms=5.0,
            target_v_inf=1.05,
            fraction_of_drive=0.80,
        ),
        # Sub→EC_V: same moderate depression as CA1→EC_V.
        stp_config=SCHAFFER_COLLATERAL_PRESET.configure(),
    )


# =============================================================================
# Public entry points
# =============================================================================

def add_mtl_circuit(
    builder: BrainBuilder,
    *,
    medial_septum_name: str = "medial_septum",
    entorhinal_cortex_name: str = "entorhinal_cortex",
    hippocampus_name: str = "hippocampus",
    subiculum_name: str = "subiculum",
    include_subiculum: bool = False,
    population_sizes: Optional[RegionSizes] = None,
    add_regions: bool = True,
) -> None:
    """Add the MTL circuit to an existing *builder* and wire all internal connections.

    Regions are added with default population sizes unless overridden via
    ``population_sizes`` (keyed by the *instance* names supplied to this function).

    Args:
        builder: :class:`~thalia.brain.brain_builder.BrainBuilder` to modify.
        medial_septum_name: Instance name of the medial septum region.
        entorhinal_cortex_name: Instance name of the entorhinal cortex region.
        hippocampus_name: Instance name of the hippocampus region.
        subiculum_name: Instance name of the subiculum region.
        include_subiculum: When ``True``, adds the subiculum relay and wires
            CA1 → Sub → EC_V instead of direct CA1 → EC_V.
        population_sizes: Optional ``RegionSizes`` dict overriding default
            population sizes.  Keys should match the *instance* names used
            for the regions (e.g. ``{"hippocampus": {HippocampusPopulation.CA1: 500}}``).
        add_regions: When ``False``, skip adding regions to *builder* and only
            wire the internal connections.  Use this when the regions have
            already been registered (e.g. inside the ``"default"`` preset).
    """
    overrides: Dict[str, Any] = {}
    if population_sizes is not None:
        overrides["population_sizes"] = population_sizes

    ms_name  = medial_septum_name
    ec_name  = entorhinal_cortex_name
    hpc_name = hippocampus_name
    sub_name = subiculum_name

    sizes = _resolve_mtl_sizes(overrides, ms_name, ec_name, hpc_name, sub_name)

    if add_regions:
        _add_mtl_regions(builder, sizes, ms_name, ec_name, hpc_name, sub_name, include_subiculum)

    _connect_septal_theta_loop(builder, ms_name, hpc_name)
    _connect_ec_to_hpc(builder, ec_name, hpc_name)

    if include_subiculum:
        _connect_hpc_via_subiculum(builder, hpc_name, sub_name, ec_name)
    else:
        _connect_hpc_to_ec_direct(builder, hpc_name, ec_name)


def build(builder: BrainBuilder, **overrides: Any) -> None:
    """Standalone MTL preset: medial septum + entorhinal cortex + hippocampus.

    Registered as the ``"mtl"`` preset in :class:`~thalia.brain.brain_builder.BrainBuilder`.
    Uses default region names and population sizes; accepts ``population_sizes``
    override dict and ``include_subiculum`` flag.

    Example::

        brain = BrainBuilder.preset("mtl")

    To build with subiculum relay::

        brain = BrainBuilder.preset("mtl", include_subiculum=True)
    """
    add_mtl_circuit(
        builder,
        include_subiculum=bool(overrides.get("include_subiculum", False)),
        population_sizes=overrides.get("population_sizes"),
    )
