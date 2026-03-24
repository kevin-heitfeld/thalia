"""Cross-validation between region_tags.py tag constants and bio_firing_ranges.py RegionSpec entries.

Catches naming drift: if a region or population is renamed in only one of the
two subsystems, these tests will fail immediately.
"""

from __future__ import annotations

import pytest

from thalia.diagnostics.bio_firing_ranges import REGION_SPECS
from thalia.diagnostics.region_tags import (
    ARKYPALLIDAL_TAGS,
    CEREBELLAR_TAGS,
    CORTICAL_TAGS,
    D1_MSN_TAGS,
    D2_MSN_TAGS,
    DA_NEURON_TAGS,
    DA_SOURCE_TAGS,
    GPE_TAGS,
    GPI_TAGS,
    GRANULE_TAGS,
    HIPPOCAMPAL_TAGS,
    L23_TAGS,
    L4_TAGS,
    L5_TAGS,
    L6_TAGS,
    MOTOR_CORTEX_TAGS,
    NMDA_REGION_TAGS,
    PREFRONTAL_TAGS,
    PURKINJE_TAGS,
    PV_TAGS,
    PYRAMIDAL_TAGS,
    RELAY_TAGS,
    SENSORY_CORTEX_TAGS,
    SEPTUM_TAGS,
    SNR_TAGS,
    SST_TAGS,
    STN_TAGS,
    STRIATAL_TAGS,
    THALAMIC_TAGS,
    TRN_TAGS,
    UPDOWN_STATE_TAGS,
    VIP_TAGS,
)

# All non-empty RegionSpec.region values (lowercased).
_SPEC_REGIONS = {s.region.lower() for s in REGION_SPECS if s.region}
# All non-empty RegionSpec.population values (lowercased).
_SPEC_POPULATIONS = {s.population.lower() for s in REGION_SPECS if s.population}


def _region_tag_covered(tag: str) -> bool:
    """True if *tag* is a substring of some RegionSpec.region or vice versa."""
    t = tag.lower()
    return any(t in r or r in t for r in _SPEC_REGIONS)


def _population_tag_covered(tag: str) -> bool:
    """True if *tag* is a substring of some RegionSpec.population or vice versa."""
    t = tag.lower()
    return any(t in p or p in t for p in _SPEC_POPULATIONS)


# ── Region tag sets ──────────────────────────────────────────────────────────

# Short aliases (e.g. "gpe", "stn") intentionally match hypothetical short
# region names that the brain builder might use; they won't appear in the
# longer canonical RegionSpec.region strings.  We expect only the canonical
# member of each set to cross-reference into REGION_SPECS.
_SHORT_REGION_ALIASES: frozenset[str] = frozenset({
    "gpe", "stn", "snr", "gpi",
    "thalamus_sensory", "thalamus_association", "thalamus_md",
    "substantia_nigra_reticulata",
    "globus_pallidus_internal",
})

# Region-level tag frozensets (name → tags).
_REGION_TAG_SETS: dict[str, frozenset[str]] = {
    "CORTICAL_TAGS": CORTICAL_TAGS,
    "PREFRONTAL_TAGS": PREFRONTAL_TAGS,
    "SENSORY_CORTEX_TAGS": SENSORY_CORTEX_TAGS,
    "MOTOR_CORTEX_TAGS": MOTOR_CORTEX_TAGS,
    "HIPPOCAMPAL_TAGS": HIPPOCAMPAL_TAGS,
    "SEPTUM_TAGS": SEPTUM_TAGS,
    "STRIATAL_TAGS": STRIATAL_TAGS,
    "GPE_TAGS": GPE_TAGS,
    "STN_TAGS": STN_TAGS,
    "SNR_TAGS": SNR_TAGS,
    "GPI_TAGS": GPI_TAGS,
    "THALAMIC_TAGS": THALAMIC_TAGS,
    "CEREBELLAR_TAGS": CEREBELLAR_TAGS,
    "DA_SOURCE_TAGS": DA_SOURCE_TAGS,
    "UPDOWN_STATE_TAGS": UPDOWN_STATE_TAGS,
    "NMDA_REGION_TAGS": NMDA_REGION_TAGS,
}


@pytest.mark.parametrize(
    "set_name,tags",
    [(n, t) for n, t in _REGION_TAG_SETS.items()],
    ids=[n for n in _REGION_TAG_SETS],
)
def test_region_tags_have_spec_coverage(set_name: str, tags: frozenset[str]) -> None:
    """Every region tag (excluding known short aliases) must cross-reference a RegionSpec."""
    uncovered = {
        tag for tag in tags
        if tag not in _SHORT_REGION_ALIASES and not _region_tag_covered(tag)
    }
    assert not uncovered, (
        f"{set_name} contains region tags with no RegionSpec coverage: {uncovered}"
    )


# ── Population tag sets ──────────────────────────────────────────────────────

# Short population aliases that are intentionally broader than any
# RegionSpec.population substring (e.g. "parvalbumin" won't appear as a
# RegionSpec population pattern because "_pv" is used instead).
_SHORT_POP_ALIASES: frozenset[str] = frozenset({
    "parvalbumin", "somatostatin", "som", "pyramidal", "l2_3",
})

_POP_TAG_SETS: dict[str, frozenset[str]] = {
    "PV_TAGS": PV_TAGS,
    "SST_TAGS": SST_TAGS,
    "VIP_TAGS": VIP_TAGS,
    "L4_TAGS": L4_TAGS,
    "L23_TAGS": L23_TAGS,
    "L5_TAGS": L5_TAGS,
    "L6_TAGS": L6_TAGS,
    "PYRAMIDAL_TAGS": PYRAMIDAL_TAGS,
    "PURKINJE_TAGS": PURKINJE_TAGS,
    "GRANULE_TAGS": GRANULE_TAGS,
    "D1_MSN_TAGS": D1_MSN_TAGS,
    "D2_MSN_TAGS": D2_MSN_TAGS,
    "ARKYPALLIDAL_TAGS": ARKYPALLIDAL_TAGS,
    "DA_NEURON_TAGS": DA_NEURON_TAGS,
    "TRN_TAGS": TRN_TAGS,
    "RELAY_TAGS": RELAY_TAGS,
}


@pytest.mark.parametrize(
    "set_name,tags",
    [(n, t) for n, t in _POP_TAG_SETS.items()],
    ids=[n for n in _POP_TAG_SETS],
)
def test_population_tags_have_spec_coverage(set_name: str, tags: frozenset[str]) -> None:
    """Every population tag (excluding known short aliases) must cross-reference a RegionSpec."""
    uncovered = {
        tag for tag in tags
        if tag not in _SHORT_POP_ALIASES and not _population_tag_covered(tag)
    }
    assert not uncovered, (
        f"{set_name} contains population tags with no RegionSpec coverage: {uncovered}"
    )


# ── Structural invariants ────────────────────────────────────────────────────

def test_region_specs_sorted_by_specificity() -> None:
    """REGION_SPECS must be sorted longest-first so lookup functions work correctly."""
    keys = [(len(s.region), len(s.population)) for s in REGION_SPECS]
    assert keys == sorted(keys, reverse=True), "REGION_SPECS is not sorted by specificity"


def test_no_duplicate_region_spec_entries() -> None:
    """No two RegionSpec entries may share the same (region, population) key."""
    seen: set[tuple[str, str]] = set()
    for s in REGION_SPECS:
        key = (s.region.lower(), s.population.lower())
        assert key not in seen, f"Duplicate RegionSpec entry: {key}"
        seen.add(key)
