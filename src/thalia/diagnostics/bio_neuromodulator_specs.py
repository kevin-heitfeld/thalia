"""Neuromodulator tonic concentration ranges.

Reference ranges for healthy tonic (non-phasic) neuromodulator receptor
activation levels.
"""

from __future__ import annotations

from typing import List, Tuple


# =============================================================================
# NEUROMODULATOR TONIC CONCENTRATION RANGES  [normalized 0–1 activation]
# =============================================================================
# NeuromodulatorReceptor.concentration is bounded [0, 1]; these thresholds
# describe healthy tonic (non-phasic) mean activation.  Values above warn_high
# indicate near-saturation / receptor desensitisation; values below warn_low
# indicate chronically inadequate neuromodulator release or clearance failure.
_NM_TONIC_RANGES: List[Tuple[str, float, float]] = [
    # (mod_name_substring, warn_low, warn_high)
    ("dopamine",        0.005, 0.65),
    ("acetylcholine",   0.005, 0.75),
    ("serotonin",       0.002, 0.60),
    ("norepinephrine",  0.005, 0.65),
]
_NM_TONIC_DEFAULT: Tuple[float, float] = (0.005, 0.70)


def nm_tonic_range(mod_name: str) -> Tuple[float, float]:
    """Return ``(warn_low, warn_high)`` tonic concentration bounds.

    Uses case-insensitive substring matching on *mod_name* (the dotted module
    path of the :class:`NeuromodulatorReceptor`).  Returns the default bounds
    ``(0.005, 0.70)`` when no specific entry matches.
    """
    mn = mod_name.lower()
    for key, low, high in _NM_TONIC_RANGES:
        if key in mn:
            return (low, high)
    return _NM_TONIC_DEFAULT
