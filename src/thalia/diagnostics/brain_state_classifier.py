"""Automatic brain-state classifier from EEG band-power ratios.

Infers physiological state from the spectral decomposition of simulated
neural activity, without requiring manual annotations or sensory-pattern
labels.

Classification uses a decision-tree over global relative band power,
grounded in established clinical EEG criteria:

* **Wake**: desynchronised EEG — low delta, elevated beta+gamma or
  theta/delta > 1 (Niedermeyer & da Silva 2005).
* **NREM**: delta-dominant (>40 %) with sigma (sleep spindles) and
  suppressed gamma (Steriade et al. 1993).
* **REM**: theta-dominant, low delta, low alpha+beta — resembles quiet
  waking but without fast-frequency desynchronisation (Siegel 2005).
* **Anesthesia**: extreme delta (>60 %), near-zero gamma, steep 1/f
  exponent (Purdon et al. 2013).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .diagnostics_metrics import OscillatoryStats

BrainState = Literal["wake", "nrem", "rem", "anesthesia", "unknown"]


def classify_brain_state(oscillations: OscillatoryStats) -> BrainState:
    """Classify physiological brain state from global band-power ratios.

    Uses ``oscillations.global_band_power`` (relative, sums to ~1) and,
    when available, the mean aperiodic exponent across regions.

    Returns ``"unknown"`` when spectral data is absent or ambiguous.
    """
    bp = oscillations.global_band_power
    if not bp:
        return "unknown"

    delta = bp.get("delta", 0.0)
    theta = bp.get("theta", 0.0)
    alpha = bp.get("alpha", 0.0)
    sigma = bp.get("sigma", 0.0)
    beta = bp.get("beta", 0.0)
    gamma = bp.get("gamma", 0.0)

    total = delta + theta + alpha + sigma + beta + gamma
    if total < 1e-12:
        return "unknown"

    # Relative power (already ~normalised, but guard against rounding).
    dr = delta / total
    tr = theta / total
    ar = alpha / total
    sr = sigma / total
    br = beta / total
    gr = gamma / total

    # Mean aperiodic exponent (optional refinement).
    chi_vals = [
        v for v in oscillations.region_aperiodic_exponent.values()
        if not np.isnan(v)
    ]
    mean_chi = float(np.mean(chi_vals)) if chi_vals else float("nan")

    # ── 1. Anesthesia ────────────────────────────────────────────────────
    # Burst-suppression pattern: extreme delta, virtually no gamma.
    # Purdon et al. 2013 — EEG signatures of general anesthesia.
    if dr > 0.60 and gr < 0.05:
        if not np.isnan(mean_chi) and mean_chi > 2.5:
            return "anesthesia"
        if dr > 0.75:
            return "anesthesia"

    # ── 2. NREM ──────────────────────────────────────────────────────────
    # Slow-wave sleep: delta-dominant, sigma spindles, suppressed gamma.
    # Steriade et al. 1993 — thalamocortical oscillations in sleep.
    if dr > 0.40 and gr < 0.10:
        if sr > 0.05 or dr > 0.50:
            return "nrem"

    # ── 3. REM ───────────────────────────────────────────────────────────
    # Theta-dominant, low delta, low alpha+beta — "activated" sleep.
    # Siegel 2005 — clues to the functions of mammalian sleep.
    if tr > 0.30 and dr < 0.30 and (ar + br) < 0.20:
        return "rem"

    # ── 4. Wake ──────────────────────────────────────────────────────────
    # Desynchronised cortex: theta/delta > 1, or substantial fast power,
    # or simply not delta-dominant.
    # Niedermeyer & da Silva 2005 — EEG: basic principles.
    theta_delta = tr / max(dr, 1e-9)
    fast_power = br + gr
    if theta_delta > 1.0 or fast_power > 0.20 or dr < 0.30:
        return "wake"

    return "unknown"
