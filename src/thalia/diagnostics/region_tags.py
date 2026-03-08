"""Shared region-name tag constants for biological health checks.

All checkers that filter or identify brain regions by name should import from
here rather than scattering bare string literals across modules.  Centralising
the tags ensures that renaming a region in the brain registry requires updating
only this file, rather than hunting silent mismatches across every health check.
"""

from __future__ import annotations

# Cortical regions (neocortex + prefrontal).
CORTICAL_TAGS: frozenset[str] = frozenset({"cortex", "prefrontal", "entorhinal"})

# Prefrontal cortex specifically (used for PFC-specific message variants).
PREFRONTAL_TAGS: frozenset[str] = frozenset({"prefrontal"})

# Hippocampal formation.
HIPPOCAMPAL_TAGS: frozenset[str] = frozenset({"hippocampus"})

# Striatum / basal ganglia input nucleus.
STRIATAL_TAGS: frozenset[str] = frozenset({"striatum"})

# Thalamic relay and TRN regions.
THALAMIC_TAGS: frozenset[str] = frozenset({"thalamus"})

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


def matches_any(name: str, tags: frozenset[str]) -> bool:
    """Return ``True`` if any tag is a case-insensitive substring of *name*."""
    nl = name.lower()
    return any(t in nl for t in tags)
