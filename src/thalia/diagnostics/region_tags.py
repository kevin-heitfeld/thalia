"""Shared region-name tag constants for biological health checks.

All checkers that filter or identify brain regions by name should import from
here rather than scattering bare string literals across modules.  Centralising
the tags ensures that renaming a region in the brain registry requires updating
only this file, rather than hunting silent mismatches across every health check.

The module covers three levels of identification:

1. **Region tags** — matched against region names (e.g. ``"cortex_sensory"``).
2. **Population tags** — matched against population names (e.g. ``"pv_interneuron"``).
3. **Neuromodulator tags** — matched against neuromodulator/receptor names.

All matching is case-insensitive substring matching via :func:`matches_any`.
"""

from __future__ import annotations

# ── Region tags ───────────────────────────────────────────────────────────────

# Cortical regions (neocortex + prefrontal).
CORTICAL_TAGS: frozenset[str] = frozenset({"cortex", "prefrontal", "entorhinal"})
PREFRONTAL_TAGS: frozenset[str] = frozenset({"prefrontal"})
SENSORY_CORTEX_TAGS: frozenset[str] = frozenset({"cortex_sensory"})
MOTOR_CORTEX_TAGS: frozenset[str] = frozenset({"cortex_motor"})

# Hippocampal formation.
HIPPOCAMPAL_TAGS: frozenset[str] = frozenset({"hippocampus"})

# Medial septum (theta pacemaker).
SEPTUM_TAGS: frozenset[str] = frozenset({"septum"})

# Striatum / basal ganglia input nucleus.
STRIATAL_TAGS: frozenset[str] = frozenset({"striatum"})

# Basal ganglia nuclei.
GPE_TAGS: frozenset[str] = frozenset({"globus_pallidus", "gpe"})
STN_TAGS: frozenset[str] = frozenset({"subthalamic", "stn"})
SNR_TAGS: frozenset[str] = frozenset({"substantia_nigra_reticulata", "snr"})
GPI_TAGS: frozenset[str] = frozenset({"globus_pallidus_internal", "globus_pallidus_interna", "gpi"})

# Thalamic relay and TRN regions (all nuclei).
THALAMIC_TAGS: frozenset[str] = frozenset({"thalamus_sensory", "thalamus_association", "thalamus_md", "thalamus"})

# Cerebellar cortex and nuclei.
CEREBELLAR_TAGS: frozenset[str] = frozenset({"cerebellum"})

# Dopaminergic source nuclei (VTA + SNc).
DA_SOURCE_TAGS: frozenset[str] = frozenset({"vta", "substantia_nigra_compacta"})

# Regions expected to exhibit up/down state dynamics.
UPDOWN_STATE_TAGS: frozenset[str] = frozenset({
    "prefrontal", "striatum", "cortex_motor", "cortex_sensory",
    "hippocampus", "entorhinal",
})

# Regions where NMDA/AMPA fraction checks are biologically meaningful.
NMDA_REGION_TAGS: frozenset[str] = frozenset({
    "cortex", "hippocampus", "entorhinal", "prefrontal", "basolateral_amygdala",
})

# ── Population tags ───────────────────────────────────────────────────────────

# Interneuron subtypes.
PV_TAGS: frozenset[str] = frozenset({"pv", "parvalbumin"})
SST_TAGS: frozenset[str] = frozenset({"sst", "somatostatin", "som"})
VIP_TAGS: frozenset[str] = frozenset({"vip"})

# Cortical layer populations (matched against population names).
L4_TAGS: frozenset[str] = frozenset({"l4"})
L23_TAGS: frozenset[str] = frozenset({"l23", "l2_3"})
L5_TAGS: frozenset[str] = frozenset({"l5"})
L6_TAGS: frozenset[str] = frozenset({"l6"})

# Excitatory principal cells.
PYRAMIDAL_TAGS: frozenset[str] = frozenset({"pyr", "pyramidal"})

# Cerebellar cell types.
PURKINJE_TAGS: frozenset[str] = frozenset({"purkinje"})
GRANULE_TAGS: frozenset[str] = frozenset({"granule"})

# Striatal medium spiny neurons (MSN) — matched against population names.
D1_MSN_TAGS: frozenset[str] = frozenset({"d1"})
D2_MSN_TAGS: frozenset[str] = frozenset({"d2"})

# GPe arkypallidal subtype.
ARKYPALLIDAL_TAGS: frozenset[str] = frozenset({"arkypallidal"})

# Dopaminergic neuron populations (matched against population names within
# DA source regions, e.g. VTA "da" population).
DA_NEURON_TAGS: frozenset[str] = frozenset({"da"})

# Thalamic population subtypes.
TRN_TAGS: frozenset[str] = frozenset({"trn"})
RELAY_TAGS: frozenset[str] = frozenset({"relay"})

# ── Neuromodulator tags ───────────────────────────────────────────────────────

DOPAMINE_NM_TAGS: frozenset[str] = frozenset({"dopamine"})
ACH_NM_TAGS: frozenset[str] = frozenset({"acetylcholine", "ach"})
SEROTONIN_NM_TAGS: frozenset[str] = frozenset({"serotonin", "5-ht", "5ht"})


def matches_any(name: str, tags: frozenset[str]) -> bool:
    """Return ``True`` if any tag is a case-insensitive substring of *name*."""
    nl = name.lower()
    return any(t in nl for t in tags)


# ── D1/D2 MSN population lookup ──────────────────────────────────────────────


def find_d1_d2_fr(
    rec_pop_keys: list[tuple[str, str]],
    region_pop_indices: list[int],
    pop_stats: dict[tuple[str, str], object],
    rn: str,
) -> tuple[list[float], list[float]]:
    """Return ``(d1_fr_vals, d2_fr_vals)`` for a striatal region.

    Collects the mean firing rates of D1-MSN and D2-MSN populations within
    *rn* from *pop_stats*.  Shared by every health check that needs the
    D1/D2 FR split (neuromodulators, striatum, oscillations).
    """
    import numpy as _np  # local to avoid circular import at module level

    region_pops = [rec_pop_keys[i][1] for i in region_pop_indices]
    d1_fr_vals: list[float] = [
        pop_stats[(rn, pn)].mean_fr_hz  # type: ignore[union-attr]
        for pn in region_pops
        if matches_any(pn, D1_MSN_TAGS)
        and (rn, pn) in pop_stats
        and not _np.isnan(pop_stats[(rn, pn)].mean_fr_hz)  # type: ignore[union-attr]
    ]
    d2_fr_vals: list[float] = [
        pop_stats[(rn, pn)].mean_fr_hz  # type: ignore[union-attr]
        for pn in region_pops
        if matches_any(pn, D2_MSN_TAGS)
        and (rn, pn) in pop_stats
        and not _np.isnan(pop_stats[(rn, pn)].mean_fr_hz)  # type: ignore[union-attr]
    ]
    return d1_fr_vals, d2_fr_vals


def find_d1_d2_pop_indices(
    rec_pop_keys: list[tuple[str, str]],
    rec_pop_index: dict[tuple[str, str], int],
    region_pop_indices: list[int],
    rn: str,
) -> tuple[list[int], list[int]]:
    """Return ``(d1_indices, d2_indices)`` — pop-index lists for D1/D2 MSNs in *rn*."""
    d1: list[int] = []
    d2: list[int] = []
    for i in region_pop_indices:
        _, pn = rec_pop_keys[i]
        key = (rn, pn)
        if key not in rec_pop_index:
            continue
        idx = rec_pop_index[key]
        if matches_any(pn, D1_MSN_TAGS):
            d1.append(idx)
        elif matches_any(pn, D2_MSN_TAGS):
            d2.append(idx)
    return d1, d2
