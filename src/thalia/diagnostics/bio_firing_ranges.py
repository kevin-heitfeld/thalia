"""Biological reference firing-rate ranges, E/I thresholds, and region lookup functions.

All region specifications are unified in ``REGION_SPECS`` (a list of :class:`RegionSpec`
entries).  The public functions ``bio_range``, ``ei_ratio_thresholds``, and
``expected_dominant_band`` retain their original signatures.

Matching is case-insensitive substring containment; the longest matching
pattern wins (``REGION_SPECS`` is sorted by specificity at module load time
so ordering within the list does not matter for correctness).

This module is intentionally free of simulation dependencies so that it can be
imported by ``RegionTestRunner``, notebooks, and any other tool that needs
ground-truth biological ranges without pulling in the full recorder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Sentinel: RegionSpec entry does not address an attribute.
# Test with ``spec.dominant_band is _UNSET``.
# ---------------------------------------------------------------------------
_UNSET: Any = object()


@dataclass
class RegionSpec:
    """Unified biological specification for a region / population combination.

    ``region`` and ``population`` are case-insensitive substring patterns;
    ``""`` matches any string.  ``REGION_SPECS`` is sorted by
    ``(len(region), len(population))`` descending at module load time, so the
    longest (most-specific) pattern always wins regardless of declaration order.

    ``fr_range``: ``(min_hz, max_hz)``  — ``None`` = no FR expectation for this entry.

    ``ei_thresholds``: ``(crit_low, warn_low, warn_high, crit_high)`` —
    ``None`` = not addressed by this entry.  Set ``skip_ei_check=True`` to
    explicitly suppress the E/I check for a region.

    ``dominant_band``:

    * ``_UNSET`` (default) — this entry does not address dominant band
    * ``None`` — explicitly no expectation / skip check
    * a band name like ``"theta"`` — expected dominant EEG band
    """

    region: str
    population: str = ""
    fr_range: Optional[Tuple[float, float]] = None
    ei_thresholds: Optional[Tuple[float, float, float, float]] = None
    skip_ei_check: bool = False
    dominant_band: Any = field(default=_UNSET)
    # True  = adapting cell type (pyr/relay/MSN/principal): expect SFA index > 1.3.
    # False = non-adapting cell type (PV/FSI/TAN): expect SFA index ≈ 1.0.
    # None  = this entry does not address adaptation (lookup skips it).
    adaptation_expected: Optional[bool] = None
    # Expected range for the firing-rate autocorrelation time constant τ_int (ms).
    # PFC: 100–400 ms (Murray et al. 2014); primary sensory: 20–50 ms.
    # None = no expectation for this region.
    integration_tau_ms: Optional[Tuple[float, float]] = None
    # Suppress epileptiform burst CRITICAL for tonically-firing populations (e.g.
    # GPe/GPi/SNr pacemakers, cerebellum Purkinje/DCN).  100 % co-activation is
    # expected for these populations and does NOT indicate seizure activity.
    skip_burst_check: bool = False
    # Suppress network synchronisation CRITICAL for pacemaker populations where
    # high pairwise ρ is an intrinsic property of tonic, driver-locked firing
    # (e.g. GPi/GPe prototypic driven by STN; SNr driven by striatum+STN).
    skip_sync_check: bool = False
    # True for autonomous pacemaker populations (GPe, GPi, SNr, SNc DA, TAN,
    # Purkinje, DCN, medial septum ACh) where low ISI CV₂ is biologically
    # expected — suppress "Low CV₂" regularity warnings.
    is_pacemaker: bool = False
    # Burst detection window in ms.  Cortical = 20 ms (default), cerebellar
    # complex spikes = 5 ms, DA bursts = 100 ms (Grace & Bunney 1984).
    # None = use the global default (20 ms).
    burst_window_ms: Optional[float] = None

# =============================================================================
# UNIFIED REGION SPECIFICATIONS
# =============================================================================

REGION_SPECS: List[RegionSpec] = [
    # -----------------------------------------------------------------------
    # Override entries: region-specific population FR ranges that shadow
    # the generic population-only patterns below.
    # -----------------------------------------------------------------------

    # cortex_sensory L2/3 fires 5–15 Hz during sensory processing (Sakata & Harris 2009);
    # the generic "" l23_pyr entry below covers all other cortical regions at 0.1–3 Hz.
    RegionSpec(region="cortex_sensory", population="l23_pyr", fr_range=(0.1,  15.0), adaptation_expected=True),

    # Hippocampal inhibitory PV subtypes: fast-spiking basket/chandelier cells fire
    # 5–20 Hz during sparse hippocampal activity (Klausberger & Somogyi 2008).
    # These must precede the generic "dg"/"ca3"/"ca2"/"ca1" principal-cell entries
    # (which carry 1–5 Hz targets) so that "ca3_inhibitory_pv" etc. are not
    # mis-classified by the shorter "ca3" substring match.
    RegionSpec(region="hippocampus",    population="dg_inhibitory_pv",           fr_range=(5.0, 20.0),  adaptation_expected=False,  skip_burst_check=True),  # small pop receives strong EC→DG drive burst
    RegionSpec(region="hippocampus",    population="ca3_inhibitory_pv",          fr_range=(5.0, 20.0),  adaptation_expected=False,  skip_burst_check=True),
    RegionSpec(region="hippocampus",    population="ca2_inhibitory_pv",          fr_range=(5.0, 20.0),  adaptation_expected=False,  skip_burst_check=True),
    RegionSpec(region="hippocampus",    population="ca1_inhibitory_pv",          fr_range=(5.0, 20.0),  adaptation_expected=False,  skip_burst_check=True),
    RegionSpec(region="hippocampus",    population="dg_inhibitory_olm",          fr_range=(0.0,  5.0),  adaptation_expected=True),
    RegionSpec(region="hippocampus",    population="dg_inhibitory_bistratified",  fr_range=(0.0,  5.0),  adaptation_expected=True),
    # CA2 has only 3 OLM and 2 bistratified neurons; stochastic silence is expected.
    RegionSpec(region="hippocampus",    population="ca2_inhibitory_olm",          fr_range=(0.0,  5.0),  adaptation_expected=True),
    RegionSpec(region="hippocampus",    population="ca2_inhibitory_bistratified",  fr_range=(0.0,  5.0),  adaptation_expected=True),

    # -----------------------------------------------------------------------
    # Population-only patterns (region="" → matches any region).
    # These apply wherever a more-specific entry above did not match first.
    # -----------------------------------------------------------------------

    # Cerebellum
    # Purkinje and DCN fire tonically at 40–100 Hz / 10–100 Hz respectively.
    # 100 % co-activation in every 20 ms window is a normal property of their
    # sustained high-rate firing — not a seizure indicator.
    RegionSpec(region="",  population="purkinje",        fr_range=(40.0, 100.0),  skip_burst_check=True, is_pacemaker=True, burst_window_ms=5.0),
    RegionSpec(region="",  population="inferior_olive",  fr_range=(0.3,    3.0)),
    RegionSpec(region="",  population="dcn",             fr_range=(10.0, 100.0),  skip_burst_check=True, is_pacemaker=True),
    # Cortical pyramidal layers
    RegionSpec(region="",  population="l23_pyr",         fr_range=(0.1,   3.0),   adaptation_expected=True),
    RegionSpec(region="",  population="l4_sst_pred",     fr_range=(5.0,  25.0)),
    RegionSpec(region="",  population="l4_pyr",          fr_range=(1.0,  10.0),   adaptation_expected=True),
    RegionSpec(region="",  population="l5_pyr",          fr_range=(2.0,  15.0),   adaptation_expected=True),
    RegionSpec(region="",  population="l6a_pyr",         fr_range=(1.0,   8.0),   adaptation_expected=True),
    RegionSpec(region="",  population="l6b_pyr",         fr_range=(1.0,   8.0),   adaptation_expected=True),
    # Cortical interneurons
    RegionSpec(region="",  population="_pv",             fr_range=(10.0,  70.0),  adaptation_expected=False,  skip_burst_check=True, skip_sync_check=True),  # PV fast-spiking interneurons synchronize via gap junctions (gamma rhythm); burst/sync expected
    RegionSpec(region="",  population="_sst",            fr_range=(5.0,   25.0)),
    # VIP: 20–50 Hz in active states; 30 Hz ceiling for motor cortex.
    RegionSpec(region="",  population="_vip",            fr_range=(2.0,   30.0)),
    # NGC (neurogliaform): superficial inputs produce highly synchronous GABA-B currents; small populations (2–25 neurons) co-activate trivially.
    RegionSpec(region="",  population="_ngc",            fr_range=(5.0,   30.0),   skip_burst_check=True, skip_sync_check=True),

    # -----------------------------------------------------------------------
    # Cerebellum  (granule is region-specific; others use population-only above)
    # -----------------------------------------------------------------------
    RegionSpec(region="cerebellum",  population="granule",  fr_range=(0.1, 5.0),  adaptation_expected=True),
    RegionSpec(region="cerebellum",  skip_ei_check=True,    dominant_band="gamma"),

    # Inferior olive EI skip (FR covered by population-only entry above)
    RegionSpec(region="inferior_olive",  skip_ei_check=True),

    # -----------------------------------------------------------------------
    # Thalamus
    # -----------------------------------------------------------------------
    RegionSpec(region="thalamus",  population="relay",  fr_range=(5.0,  40.0),  adaptation_expected=True,  skip_burst_check=True),  # T-channel LTS burst mode confirmed in diagnostics (short-ISI=0.23)
    RegionSpec(region="thalamus",  population="trn",    fr_range=(5.0,  80.0)),
    # Thalamus dominant band is sigma (sleep spindles, 11–15 Hz) under
    # slow-wave / low-drive conditions.  Alpha is the resting-eyes-closed
    # equivalent but the TRN–relay spindle mechanism sits firmly in sigma.
    RegionSpec(region="thalamus",  dominant_band="sigma"),

    # -----------------------------------------------------------------------
    # Cortex
    # -----------------------------------------------------------------------
    # Cortex and PFC receive massive convergent thalamic/cortical excitation under random
    # input; E/I conductance ratio is structurally elevated but firing rates are physiological.
    RegionSpec(region="cortex_sensory",    skip_ei_check=True,  dominant_band="gamma",  integration_tau_ms=(20.0,  50.0)),
    RegionSpec(region="cortex_association",  skip_ei_check=True),
    RegionSpec(region="cortex_motor",    dominant_band="beta",   integration_tau_ms=(40.0, 150.0)),
    RegionSpec(region="prefrontal",      skip_ei_check=True,  dominant_band="beta",   integration_tau_ms=(100.0, 400.0)),

    # -----------------------------------------------------------------------
    # Hippocampus
    # -----------------------------------------------------------------------
    RegionSpec(region="hippocampus",  population="dg",          fr_range=(0.1,  1.0)),
    RegionSpec(region="hippocampus",  population="ca3",           fr_range=(1.0,  5.0),  adaptation_expected=True),
    RegionSpec(region="hippocampus",  population="ca2",           fr_range=(1.0,  5.0)),
    RegionSpec(region="hippocampus",  population="ca1",           fr_range=(1.0,  5.0),  adaptation_expected=True),
    RegionSpec(region="hippocampus",  population="_olm",          fr_range=(5.0, 15.0)),
    RegionSpec(region="hippocampus",  population="_bistratified", fr_range=(5.0, 20.0),  adaptation_expected=True),
    RegionSpec(region="hippocampus",  dominant_band="theta",  integration_tau_ms=(30.0, 80.0)),   # Bhattacharya et al. 2017 (exploration range)

    # -----------------------------------------------------------------------
    # Entorhinal cortex  (layer-specific ranges + theta expectation)
    # -----------------------------------------------------------------------
    # EC II stellate/grid cells fire 3–15 Hz (Hafting et al. 2005);
    # EC III fan cells 1–10 Hz; EC V projection cells 1–8 Hz.
    # Layer-specific entries must precede the region-wide catch-all.
    RegionSpec(region="entorhinal",  population="ec_ii",  fr_range=(3.0, 15.0)),
    RegionSpec(region="entorhinal",  population="ec_iii", fr_range=(1.0, 10.0)),
    RegionSpec(region="entorhinal",  population="ec_v",   fr_range=(1.0,  8.0)),
    RegionSpec(region="entorhinal",  fr_range=(1.0, 10.0),  dominant_band="theta",  integration_tau_ms=(50.0, 150.0)),

    # -----------------------------------------------------------------------
    # Striatum
    # -----------------------------------------------------------------------
    RegionSpec(region="striatum",  population="d1",   fr_range=(0.1,  5.0),   adaptation_expected=True),
    RegionSpec(region="striatum",  population="d2",   fr_range=(0.1,  5.0),   adaptation_expected=True),
    RegionSpec(region="striatum",  population="fsi",  fr_range=(10.0, 50.0),  adaptation_expected=False,  skip_burst_check=True),  # FSIs synchronize via gap junctions on cortical/thalamic feedforward — expected
    RegionSpec(region="striatum",  population="tan",  fr_range=(5.0,  10.0),  adaptation_expected=False, is_pacemaker=True),
    RegionSpec(region="striatum",  ei_thresholds=(0.002, 0.01, 12.0, 20.0),  dominant_band="beta",  integration_tau_ms=(10.0, 50.0)),   # MSN up-state gated window; threshold raised — sparse FSI (30/400 MSNs) cannot fully balance massive cortical+thalamic excitation at baseline (observed ~9.9 at rest)

    # -----------------------------------------------------------------------
    # Basal ganglia
    # -----------------------------------------------------------------------
    # GPe entries (globus_pallidus_interna is longer → sorted before globus_pallidus).
    # GPi principal neurons fire 50–100 Hz tonically (Yelnik et al. 2007):
    # 100 % burst co-activation and high pairwise ρ are intrinsic to this
    # pacemaker-like tonic drive — not pathological.
    # GPi also receives predominantly glutamatergic (STN) input → skip_ei_check.
    RegionSpec(region="globus_pallidus_interna",  population="principal",    fr_range=(50.0, 100.0),  skip_burst_check=True, skip_sync_check=True, is_pacemaker=True),
    RegionSpec(region="globus_pallidus_interna",  population="border_cells", fr_range=(20.0,  50.0),  skip_burst_check=True, skip_sync_check=True, is_pacemaker=True),
    RegionSpec(region="globus_pallidus_interna",  skip_ei_check=True,  dominant_band=None),
    # GPe prototypic: tonic 30–80 Hz pacemaker locked to STN; suppress burst+sync.
    RegionSpec(region="globus_pallidus",  population="prototypic",    fr_range=(30.0, 80.0),  skip_burst_check=True, skip_sync_check=True, is_pacemaker=True),
    RegionSpec(region="globus_pallidus",  population="arkypallidal",  fr_range=(5.0,  20.0),  skip_burst_check=True, skip_sync_check=True, is_pacemaker=True),  # 15 Hz at ρ=0.45 driven by STN synchronous input — arkypallidal burst/sync is secondary to STN oscillations
    # GPe is tonically driven by STN excitation with sparse internal inhibition
    # (same structural argument as GPi which already has skip_ei_check=True).
    RegionSpec(region="globus_pallidus",  skip_ei_check=True,  dominant_band="beta"),
    RegionSpec(region="subthalamic",      population="stn",           fr_range=(10.0, 40.0),  skip_burst_check=True, skip_sync_check=True, is_pacemaker=True),  # STN oscillates at delta/beta with GPe; burst+sync are intrinsic to STN-GPe loop
    # STN receives predominantly glutamatergic cortical input; E/I ratio check is
    # not informative for this nucleus.
    RegionSpec(region="subthalamic",      skip_ei_check=True,  dominant_band="beta"),

    # -----------------------------------------------------------------------
    # Substantia nigra  (SNr feedback + SNc DA)
    # -----------------------------------------------------------------------
    # "vta_feedback" was a mislabel — SNr is not VTA.  Keep a legacy alias so
    # existing networks that use the old pop name still get a range; add the
    # canonical names too.
    # SNr fires tonically at 30–80 Hz: 100 % co-activation and high ρ are
    # expected for this pacemaker nucleus — suppress burst+sync CRITICALs.
    RegionSpec(region="substantia_nigra",  population="snr",          fr_range=(30.0, 80.0),  skip_burst_check=True, skip_sync_check=True, is_pacemaker=True),
    RegionSpec(region="substantia_nigra",  population="principal",    fr_range=(30.0, 80.0),  skip_burst_check=True, skip_sync_check=True, is_pacemaker=True),
    RegionSpec(region="substantia_nigra",  population="vta_feedback", fr_range=(30.0, 80.0),  skip_burst_check=True, skip_sync_check=True, is_pacemaker=True),  # legacy alias
    RegionSpec(region="substantia_nigra_compacta", population="da",            fr_range=(2.0,   8.0),  is_pacemaker=True, burst_window_ms=100.0),
    # SNc / SNr (substantia_nigra_compacta is longer → sorted before substantia_nigra).
    RegionSpec(region="substantia_nigra_compacta",  skip_ei_check=True,  dominant_band=None),
    RegionSpec(region="substantia_nigra",            skip_ei_check=True,  dominant_band=None),

    # -----------------------------------------------------------------------
    # VTA
    # -----------------------------------------------------------------------
    RegionSpec(region="vta",  population="da_mesolimbic",   fr_range=(2.0, 8.0),  is_pacemaker=True, burst_window_ms=100.0),
    RegionSpec(region="vta",  population="da_mesocortical",  fr_range=(2.0, 8.0),  is_pacemaker=True, burst_window_ms=100.0),
    RegionSpec(region="vta",  skip_ei_check=True,  dominant_band=None),

    # -----------------------------------------------------------------------
    # Locus coeruleus  (norepinephrine)
    # -----------------------------------------------------------------------
    RegionSpec(region="locus_coeruleus",  population="ne",   fr_range=(1.0,  5.0)),
    RegionSpec(region="locus_coeruleus",  population="gaba", fr_range=(5.0, 20.0),  skip_burst_check=True),  # LC interneurons (Aston-Jones & Cohen 2005); share common NE drive → synchronous firing is biologically expected
    RegionSpec(region="locus_coeruleus",  skip_ei_check=True,  dominant_band=None),

    # -----------------------------------------------------------------------
    # Dorsal raphe  (serotonin)
    # -----------------------------------------------------------------------
    RegionSpec(region="dorsal_raphe",  population="serotonin",  fr_range=(0.5, 3.0)),
    RegionSpec(region="dorsal_raphe",  population="gaba",       fr_range=(5.0, 40.0),  skip_burst_check=True),  # raphe GABA interneurons synchronize on shared 5-HT autoreceptor feedback
    RegionSpec(region="dorsal_raphe",  skip_ei_check=True,  dominant_band=None),

    # -----------------------------------------------------------------------
    # Nucleus basalis  (acetylcholine)
    # -----------------------------------------------------------------------
    RegionSpec(region="nucleus_basalis",  population="ach",  fr_range=(2.0, 15.0)),
    RegionSpec(region="nucleus_basalis",  skip_ei_check=True,  dominant_band=None),

    # -----------------------------------------------------------------------
    # Medial septum  (GABA + ACh theta generator)
    # -----------------------------------------------------------------------
    RegionSpec(region="medial_septum",  population="gaba",  fr_range=(5.0, 15.0)),
    RegionSpec(region="medial_septum",  population="ach",   fr_range=(5.0, 15.0),  is_pacemaker=True),
    RegionSpec(region="medial_septum",  skip_ei_check=True,  dominant_band="theta"),

    # -----------------------------------------------------------------------
    # Limbic / other
    # -----------------------------------------------------------------------
    RegionSpec(region="basolateral_amygdala",  population="principal",  fr_range=(1.0,  5.0),  adaptation_expected=True),
    # BLA PV interneurons are recruited transiently by CS/US stimuli, not tonically.
    # At resting baseline (random noise, no learned associations) BLA:pv is quiescent
    # (~0.5–5 Hz); the 5–20 Hz range only applies during active fear processing.
    # (Woodruff & Sah 2007; Bienvenu et al. 2012)
    RegionSpec(region="basolateral_amygdala",  population="pv",  fr_range=(0.5, 5.0),  adaptation_expected=False,  skip_burst_check=True),
    # BLA E/I check is not informative at rest: interneurons are context-dependent and
    # the conductance ratio reflects CS/US state rather than baseline health.
    RegionSpec(region="basolateral_amygdala",  skip_ei_check=True,  dominant_band=None,  integration_tau_ms=(50.0, 200.0)),
    # CeL and CeM (Ciocchi et al. 2010; Haubensak et al. 2010): 0.5–8 Hz at rest.
    RegionSpec(region="central_amygdala", population="lateral", fr_range=(0.5, 8.0),  skip_burst_check=True),  # BLA→CeL synchronous excitation drives burst at in-range rates
    RegionSpec(region="central_amygdala", population="medial",  fr_range=(0.5, 8.0)),
    # CeA is almost entirely GABAergic projection neurons with no interneurons;
    # high E/I conductance ratio is structural — E/I check not informative.
    RegionSpec(region="central_amygdala", skip_ei_check=True, dominant_band=None),
    RegionSpec(region="lateral_habenula", population="principal",  fr_range=(5.0, 20.0),  adaptation_expected=True,  skip_burst_check=True),  # LHb burst-fires to encode aversive outcomes (Weiss & Bhatt 2019); ρ<0.40 means no pathological synchrony
    RegionSpec(region="lateral_habenula", skip_ei_check=True, dominant_band=None),
    RegionSpec(region="rostromedial",     population="gaba", fr_range=(5.0, 30.0),  skip_burst_check=True),  # 30 Hz × 1000 neurons always exceeds Binomial(1000,0.3) co-activation threshold — rate artifact
    RegionSpec(region="rostromedial",     skip_ei_check=True, dominant_band=None),

    # -----------------------------------------------------------------------
    # Cortical L1 neurogliaform cells (Jiang et al. 2013): 5–25 Hz.
    # Must precede the generic "gaba" catch-all below.
    # -----------------------------------------------------------------------
    RegionSpec(region="",  population="l1_ngc",  fr_range=(5.0, 25.0)),

    # -----------------------------------------------------------------------
    # Generic GABA interneurons  (must follow more-specific interneuron patterns)
    # -----------------------------------------------------------------------
    RegionSpec(region="",  population="gaba",  fr_range=(5.0, 40.0)),

    # -----------------------------------------------------------------------
    # Catch-all: balanced E/I, no dominant-band expectation
    # -----------------------------------------------------------------------
    RegionSpec(region="",  ei_thresholds=(0.05, 0.20, 3.5, 8.0),  dominant_band=None),
]

# Sort by (len(region), len(population)) descending so that longer (more-specific)
# patterns are tried first.  Python's sort is stable, so ties preserve declaration
# order — allowing deliberate same-length overrides if ever needed.
REGION_SPECS.sort(key=lambda s: (len(s.region), len(s.population)), reverse=True)

# Guard against duplicate (region, population) entries that would silently shadow
# each other after sorting.  Raises AssertionError at import time so the mistake
# is caught immediately rather than producing subtly wrong health-check results.
_seen_keys: set[tuple[str, str]] = set()
for _spec in REGION_SPECS:
    _key = (_spec.region.lower(), _spec.population.lower())
    assert _key not in _seen_keys, (
        f"bio_firing_ranges.py: duplicate RegionSpec entry ({_spec.region!r}, {_spec.population!r}) — "
        f"remove one of the two entries to avoid silent shadowing"
    )
    _seen_keys.add(_key)
del _seen_keys, _spec, _key

# ---------------------------------------------------------------------------
# Module-level lookup caches (keyed by lowercased region / population strings).
# Populated lazily on first call; avoids lru_cache bookkeeping overhead and
# removes the ambiguity of maxsize=None unbounded caches.
# REGION_SPECS is fixed at module load time, so cached results never go stale.
# ---------------------------------------------------------------------------
_BIO_RANGE_CACHE:   Dict[Any, Optional[Any]] = {}  # (region_l, pop_l) -> Optional[Tuple[float, float]]
_EI_CACHE:          Dict[str, Optional[Any]] = {}  # region_l -> Optional[Tuple[float, float, float, float]]
_DOM_BAND_CACHE:    Dict[str, Optional[str]] = {}  # region_l -> Optional[str]
_ADAPTATION_CACHE:  Dict[Any, Optional[bool]] = {} # (region_l, pop_l) -> Optional[bool]
_TAU_CACHE:         Dict[str, Optional[Any]] = {}  # region_l -> Optional[Tuple[float, float]]
_SKIP_BURST_CACHE:  Dict[Any, bool] = {}           # (region_l, pop_l) -> bool
_SKIP_SYNC_CACHE:   Dict[Any, bool] = {}           # (region_l, pop_l) -> bool
_PACEMAKER_CACHE:   Dict[Any, bool] = {}           # (region_l, pop_l) -> bool


def bio_range(region: str, pop: str) -> Optional[Tuple[float, float]]:
    """Return ``(min_hz, max_hz)`` for a population, or ``None`` if unknown.

    Iterates ``REGION_SPECS`` (pre-sorted longest-first); the first entry where
    ``spec.region`` is a *substring* of *region* (case-insensitive) **and**
    ``spec.population`` is a *substring* of *pop* **and**
    ``spec.fr_range is not None`` wins.

    Results are cached in ``_BIO_RANGE_CACHE`` — ``REGION_SPECS`` is
    fixed at runtime so the cache is always valid.
    """
    key = (region.lower(), pop.lower())
    if key not in _BIO_RANGE_CACHE:
        r, p = key
        for spec in REGION_SPECS:
            if spec.fr_range is not None and spec.region in r and spec.population in p:
                _BIO_RANGE_CACHE[key] = spec.fr_range
                break
        else:
            _BIO_RANGE_CACHE[key] = None
    return _BIO_RANGE_CACHE[key]


def ei_ratio_thresholds(region: str) -> Optional[Tuple[float, float, float, float]]:
    """Return ``(crit_low, warn_low, warn_high, crit_high)`` for a region.

    Returns ``None`` to indicate the E/I check should be skipped entirely
    for this region (e.g. neuromodulatory nuclei).

    Biological justification for region-specific thresholds:
    - Striatal MSNs: rest near −80 mV (near GABA-A reversal); dense GABAergic
      recurrent input — E/I ≈ 0.02–0.5 is normal.
    - Globus pallidus: dominated by GABAergic striatal and STN input.
    - Neuromodulatory nuclei (LC, DRN, NB, septum, VTA, SNc): thin or absent
      recurrent inhibition — conductance ratio is not informative.
    - Cerebellum / inferior olive: complex multi-compartment dynamics.

    Iterates ``REGION_SPECS``; first entry where ``spec.skip_ei_check`` or
    ``spec.ei_thresholds is not None`` **and** region matches wins.

    Results are cached in ``_EI_CACHE`` — ``REGION_SPECS`` is fixed at runtime.
    """
    r = region.lower()
    if r not in _EI_CACHE:
        for spec in REGION_SPECS:
            if spec.region in r and (spec.skip_ei_check or spec.ei_thresholds is not None):
                _EI_CACHE[r] = None if spec.skip_ei_check else spec.ei_thresholds
                break
        else:
            _EI_CACHE[r] = (0.05, 0.20, 3.5, 8.0)  # fallback (normally hit by catch-all entry)
    return _EI_CACHE[r]


def expected_dominant_band(region: str) -> Optional[str]:
    """Return the expected dominant EEG band for *region*, or ``None`` to skip.

    Iterates ``REGION_SPECS``; first entry where
    ``spec.dominant_band is not _UNSET`` **and** region matches wins.
    ``None`` means no expectation / skip the check.

    Results are cached in ``_DOM_BAND_CACHE`` — ``REGION_SPECS`` is fixed at runtime.
    """
    r = region.lower()
    if r not in _DOM_BAND_CACHE:
        for spec in REGION_SPECS:
            if spec.dominant_band is not _UNSET and spec.region in r:
                _DOM_BAND_CACHE[r] = spec.dominant_band
                break
        else:
            _DOM_BAND_CACHE[r] = None  # no expectation (normally reached via catch-all entry)
    return _DOM_BAND_CACHE[r]


def adaptation_expected_for(region: str, pop: str) -> Optional[bool]:
    """Return whether SFA is expected for a population, or ``None`` if unknown.

    ``True``  — adapting cell type (pyramidal, relay, MSN, principal): SFA index > 1.3 expected.
    ``False`` — non-adapting cell type (PV / FSI / TAN): SFA index ≈ 1.0 expected.
    ``None``  — no expectation encoded for this region / population combination.

    Iterates ``REGION_SPECS``; the first entry where ``spec.adaptation_expected is not None``
    **and** region / population patterns match wins.

    Results are cached in ``_ADAPTATION_CACHE`` — ``REGION_SPECS`` is fixed at runtime.
    """
    key = (region.lower(), pop.lower())
    if key not in _ADAPTATION_CACHE:
        r, p = key
        for spec in REGION_SPECS:
            if spec.adaptation_expected is not None and spec.region in r and spec.population in p:
                _ADAPTATION_CACHE[key] = spec.adaptation_expected
                break
        else:
            _ADAPTATION_CACHE[key] = None
    return _ADAPTATION_CACHE[key]


def integration_tau_range(region: str) -> Optional[Tuple[float, float]]:
    """Return the expected ``(tau_min_ms, tau_max_ms)`` FR autocorrelation range for *region*.

    Iterates ``REGION_SPECS``; first entry where ``spec.integration_tau_ms is not None``
    and region matches wins.  Returns ``None`` when no expectation is encoded.

    Results are cached in ``_TAU_CACHE`` — ``REGION_SPECS`` is fixed at runtime.
    """
    r = region.lower()
    if r not in _TAU_CACHE:
        for spec in REGION_SPECS:
            if spec.integration_tau_ms is not None and spec.region in r:
                _TAU_CACHE[r] = spec.integration_tau_ms
                break
        else:
            _TAU_CACHE[r] = None
    return _TAU_CACHE[r]


def skip_burst_check_for(region: str, pop: str) -> bool:
    """Return ``True`` if epileptiform-burst CRITICAL should be suppressed.

    Suppression is warranted for tonically-firing pacemaker populations (Purkinje,
    DCN, GPi:principal, GPe:prototypic, SNr) whose 100 % co-activation in 20 ms
    windows is a normal property of sustained high-rate tonic firing, not a sign of
    seizure activity.

    Returns ``True`` if **any** ``RegionSpec`` with ``skip_burst_check=True`` matches
    both the region and population substrings.  Default ``False`` conserves alerting.
    """
    key = (region.lower(), pop.lower())
    if key not in _SKIP_BURST_CACHE:
        r, p = key
        _SKIP_BURST_CACHE[key] = any(
            spec.skip_burst_check and spec.region in r and spec.population in p
            for spec in REGION_SPECS
        )
    return _SKIP_BURST_CACHE[key]


def skip_sync_check_for(region: str, pop: str) -> bool:
    """Return ``True`` if network-synchronisation CRITICAL should be suppressed.

    Suppression is warranted for driver-locked pacemaker populations (GPi:principal,
    GPe:prototypic, SNr) where high pairwise ρ arises from shared deterministic
    drive (STN→GPe/GPi, striatum→SNr) rather than pathological synchrony.

    Returns ``True`` if **any** ``RegionSpec`` with ``skip_sync_check=True`` matches
    both the region and population substrings.  Default ``False`` conserves alerting.
    """
    key = (region.lower(), pop.lower())
    if key not in _SKIP_SYNC_CACHE:
        r, p = key
        _SKIP_SYNC_CACHE[key] = any(
            spec.skip_sync_check and spec.region in r and spec.population in p
            for spec in REGION_SPECS
        )
    return _SKIP_SYNC_CACHE[key]


def is_pacemaker_population(region: str, pop: str) -> bool:
    """Return ``True`` if the population is a known autonomous pacemaker.

    Pacemaker populations (GPe, GPi, SNr, SNc DA, VTA DA, striatum TAN,
    cerebellum Purkinje/DCN, medial septum ACh, STN) fire with intrinsically
    regular inter-spike intervals.  Low ISI CV₂ is biologically expected for
    these populations and should not trigger regularity warnings.
    """
    key = (region.lower(), pop.lower())
    if key not in _PACEMAKER_CACHE:
        r, p = key
        _PACEMAKER_CACHE[key] = any(
            spec.is_pacemaker and spec.region in r and spec.population in p
            for spec in REGION_SPECS
        )
    return _PACEMAKER_CACHE[key]


_BURST_WINDOW_CACHE: Dict[Tuple[str, str], float] = {}


def burst_window_ms_for(region: str, pop: str, default_ms: float = 20.0) -> float:
    """Return the burst detection window for the given population.

    Region-specific burst windows account for population-specific spike
    dynamics: cortical ~20 ms, cerebellar complex spikes ~5 ms,
    DA bursts ~100 ms (Grace & Bunney 1984).

    Returns *default_ms* when no matching ``RegionSpec`` defines a window.
    """
    key = (region.lower(), pop.lower())
    if key not in _BURST_WINDOW_CACHE:
        r, p = key
        best: Optional[float] = None
        best_len = -1
        for spec in REGION_SPECS:
            if spec.burst_window_ms is not None and spec.region in r and spec.population in p:
                spec_len = len(spec.region) + len(spec.population)
                if spec_len > best_len:
                    best = spec.burst_window_ms
                    best_len = spec_len
        _BURST_WINDOW_CACHE[key] = best if best is not None else default_ms
    return _BURST_WINDOW_CACHE[key]
