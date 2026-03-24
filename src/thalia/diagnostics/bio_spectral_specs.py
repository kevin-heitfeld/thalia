"""EEG band definitions, coherence specs, and phase–amplitude coupling specs.

Spectral reference data used by oscillation analysis and health checks.
Split from ``bio_ranges.py`` for module cohesion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


# =============================================================================
# EEG SPECTRAL BANDS
# =============================================================================

EEG_BANDS: Dict[str, Tuple[float, float]] = {
    "delta": (0.5,   4.0),
    "theta": (4.0,   8.0),
    # Sigma (sleep spindles, 11–15 Hz, Steriade et al. 1993) is placed
    # BEFORE alpha so the dominant-band classifier (first-match) correctly labels
    # a spindle peak at e.g. 12 Hz as "sigma" rather than "alpha".
    # Alpha power is still computed over the full 8–13 Hz biological range;
    # the band ordering here only affects the peak-frequency → band-name mapping.
    "sigma": (11.0, 15.0),
    "alpha": (8.0,  13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 100.0),
}


def freq_to_band(freq_hz: float) -> str:
    """Return the :data:`EEG_BANDS` name whose half-open interval ``[f1, f2)``
    contains *freq_hz*.  Falls back to ``"gamma"`` for frequencies above all
    defined bands (mirrors the default assumed by the dominant-band classifier).

    The ``sigma``-before-``alpha`` ordering in :data:`EEG_BANDS` is respected
    automatically because this function iterates the dict in insertion order.
    """
    for band, (f1, f2) in EEG_BANDS.items():
        if f1 <= freq_hz < f2:
            return band
    return "gamma"


# =============================================================================
# CROSS-REGIONAL COHERENCE HEALTH THRESHOLDS
# =============================================================================


@dataclass
class CoherenceSpec:
    """Expected minimum cross-regional coherence for a specific frequency band.

    ``region_a`` and ``region_b`` are case-insensitive *substring* patterns
    matched against region names in ``OscillatoryStats.region_order``.
    """

    region_a: str        # substring pattern for the first region
    region_b: str        # substring pattern for the second region
    band: str            # ``"theta"``, ``"beta"``, or ``"gamma"``
    expected_min: float  # warn when computed coherence < expected_min
    note: str            # context printed in the health-issue message


COHERENCE_SPECS: List[CoherenceSpec] = [
    # Theta (4–8 Hz): medial septum paces hippocampal theta via the
    # septohippocampal pathway.
    CoherenceSpec(
        "septum", "hippocampus", "theta", 0.40,
        "medial septum drives hippocampal theta; low coherence indicates a broken "
        "septohippocampal pathway",
    ),
    # Theta (4–8 Hz): hippocampal→entorhinal theta coupling is required for
    # grid-cell phase precession and spatial memory encoding.
    CoherenceSpec(
        "hippocampus", "entorhinal", "theta", 0.20,
        "hippocampus → entorhinal theta coupling; required for grid-cell phase "
        "precession and spatial memory encoding",
    ),
    # Theta (4–8 Hz): hippocampus → prefrontal theta synchrony underlies spatial
    # navigation and episodic memory retrieval.
    CoherenceSpec(
        "hippocampus", "prefrontal", "theta", 0.15,
        "hippocampus → prefrontal theta synchrony; required for spatial working "
        "memory and episodic retrieval",
    ),
    # Theta (4–8 Hz): thalamus → cortex theta coherence reflects sensory-gating
    # and attentional routing during active waking.
    CoherenceSpec(
        "thalamus", "cortex", "theta", 0.20,
        "thalamus → cortex theta coherence; required for attentional routing and "
        "sensory gating during active waking",
    ),
    # Beta (13–30 Hz): prefrontal–striatal synchrony underlies working-memory
    # gating and action selection.
    CoherenceSpec(
        "prefrontal", "striatum", "beta", 0.15,
        "prefrontal → striatum beta synchrony; required for working-memory gating "
        "and action selection",
    ),
    # Gamma (30–100 Hz): thalamocortical relay → sensory cortex gamma coherence
    # is a signature of active sensory transmission and feature binding.
    # Only relevant under driven conditions.
    CoherenceSpec(
        "thalamus", "cortex_sensory", "gamma", 0.30,
        "thalamus relay → sensory cortex gamma coherence; required for sensory "
        "transmission and binding under driven conditions",
    ),
]


# =============================================================================
# CROSS-FREQUENCY COUPLING SPECS
# =============================================================================


@dataclass
class CFCSpec:
    """Cross-frequency coupling specification for a region class, band pair, and coupling type.

    ``region`` is a case-insensitive *substring* pattern matched against region
    names.  ``phase_band`` and ``amp_band`` must be keys of :data:`EEG_BANDS`.

    ``coupling_type`` selects the metric:

    * ``"pac"`` — Phase–amplitude coupling (MVL; Canolty et al. 2006).
    * ``"aac"`` — Amplitude–amplitude coupling (Pearson envelope correlation;
      Bruns et al. 2000).
    * ``"pfc"`` — Phase–frequency coupling (circular–linear correlation;
      Tort et al. 2008).

    ``expected_min`` is the lower threshold used by health checks; a measured
    value below this triggers a warning.
    """

    region: str          # substring pattern for the target region
    coupling_type: str   # "pac", "aac", or "pfc"
    phase_band: str      # key into EEG_BANDS for the phase / low-freq signal
    amp_band: str        # key into EEG_BANDS for the amplitude / high-freq signal
    expected_min: float  # minimum expected value for health-check warning
    note: str            # context printed in the health-issue message


CFC_SPECS: List[CFCSpec] = [
    # ── PAC specs (phase–amplitude coupling) ────────────────────────────
    # Hippocampal theta–gamma PAC: encodes spatial information and episodic
    # memory via multiplexed phase coding.
    CFCSpec("hippocampus", "pac", "theta", "gamma", 0.015,
            "theta–gamma PAC encodes spatial information via multiplexed phase "
            "coding (Canolty et al. 2006; Lisman & Jensen 2013)"),
    # Motor-cortex beta–gamma PAC: encodes movement initiation and motor
    # preparation.
    CFCSpec("cortex_motor", "pac", "beta", "gamma", 0.008,
            "beta–gamma PAC encodes movement initiation and motor preparation "
            "(Yanovsky et al. 2012)"),
    # Striatal beta–gamma PAC: encodes reinforcement context and action
    # selection.
    CFCSpec("striatum", "pac", "beta", "gamma", 0.006,
            "beta–gamma PAC encodes reinforcement context and action selection "
            "(Crone et al. 2006)"),
    # ── AAC specs (amplitude–amplitude coupling) ────────────────────────
    # Hippocampal theta–gamma AAC: amplitude comodulation during active
    # memory encoding reflects engagement of local interneuron networks.
    CFCSpec("hippocampus", "aac", "theta", "gamma", 0.10,
            "theta–gamma amplitude comodulation during memory encoding "
            "(Canolty & Knight 2010)"),
    # Prefrontal theta–gamma AAC: cross-frequency amplitude comodulation
    # supports working-memory maintenance.
    CFCSpec("prefrontal", "aac", "theta", "gamma", 0.10,
            "prefrontal theta–gamma amplitude comodulation for working-memory "
            "maintenance (Canolty & Knight 2010)"),
]
