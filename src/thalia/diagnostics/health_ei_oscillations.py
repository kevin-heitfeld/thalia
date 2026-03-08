"""E/I balance, oscillatory band, and integration-tau health checks."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from .bio_ranges import (
    COHERENCE_SPECS,
    ei_ratio_thresholds,
    expected_dominant_band,
    integration_tau_range,
)
from .diagnostics_types import (
    HealthCategory,
    HealthIssue,
    OscillatoryStats,
    RegionStats,
)
from .region_tags import NMDA_REGION_TAGS, PREFRONTAL_TAGS, matches_any


def check_ei_balance(
    region_stats: Dict[str, RegionStats],
    issues: List[HealthIssue],
) -> None:
    """E1: Per-region excitatory/inhibitory conductance balance check.

    Runs three independent sub-checks per region:
    - E/I ratio (AMPA+NMDA numerator, GABA-A+GABA-B denominator)
    - NMDA/(AMPA+NMDA) fraction for cortical/hippocampal regions (S1.1)
    - GABA-B/GABA-A ratio where both are sampled (S1.2)
    """
    for rn, rs in region_stats.items():
        # ── E/I ratio check ────────────────────────────────────────────────────
        ratio = rs.ei_ratio  # AMPA + NMDA in numerator, GABA-A + GABA-B in denominator
        if not np.isnan(ratio):
            thresholds = ei_ratio_thresholds(rn)
            if thresholds is not None:
                crit_low, warn_low, warn_high, crit_high = thresholds
                if ratio > crit_high:
                    issues.append(HealthIssue(severity="critical", category=HealthCategory.EI_BALANCE, region=rn,
                        message=f"HYPEREXCITABLE E/I ratio: {rn}  ei_ratio = {ratio:.1f}"
                                f"  (threshold > {crit_high})"))
                elif ratio > warn_high:
                    issues.append(HealthIssue(severity="warning", category=HealthCategory.EI_BALANCE, region=rn,
                        message=f"E/I imbalance: {rn}  ei_ratio = {ratio:.1f} (excitation dominant)"
                                f"  (warn threshold > {warn_high})"))
                elif ratio < crit_low:
                    issues.append(HealthIssue(severity="critical", category=HealthCategory.EI_BALANCE, region=rn,
                        message=f"OVER-INHIBITED E/I ratio: {rn}  ei_ratio = {ratio:.4f}"
                                f"  (threshold < {crit_low})"))
                elif ratio < warn_low:
                    issues.append(HealthIssue(severity="warning", category=HealthCategory.EI_BALANCE, region=rn,
                        message=f"E/I imbalance: {rn}  ei_ratio = {ratio:.3f} (inhibition dominant)"
                                f"  (warn threshold < {warn_low})"))

        # ── NMDA/AMPA fraction check ─────────────────────────────────────
        # Only for cortical and hippocampal regions where NMDA channels underlie
        # plasticity-relevant depolarisation.  Healthy NMDA fraction of total
        # excitatory drive: 30–70 % (Jahr & Stevens 1990; Myme et al. 2003).
        # Near-zero → channels not activating (Mg²⁺ block, requires V > −40 mV).
        # Near-100 % → AMPA drive absent or grossly mis-scaled.
        _nmda_region = matches_any(rn, NMDA_REGION_TAGS)
        if _nmda_region and not np.isnan(rs.mean_g_nmda) and not np.isnan(rs.mean_g_exc):
            exc_total = rs.mean_g_exc + rs.mean_g_nmda
            if exc_total > 0:
                nmda_frac = rs.mean_g_nmda / exc_total
                if nmda_frac < 0.10:
                    issues.append(HealthIssue(severity="warning", category=HealthCategory.EI_BALANCE, region=rn,
                        message=(
                            f"Low NMDA fraction: {rn}  "
                            f"NMDA/(AMPA+NMDA) = {nmda_frac:.2f}  (expected 0.30–0.70) — "
                            f"NMDA channels may not be activating (Mg\u00b2\u207a block requires V > \u221240 mV)"
                        )))
                elif nmda_frac > 0.80:
                    issues.append(HealthIssue(severity="warning", category=HealthCategory.EI_BALANCE, region=rn,
                        message=(
                            f"High NMDA fraction: {rn}  "
                            f"NMDA/(AMPA+NMDA) = {nmda_frac:.2f}  (expected 0.30–0.70) — "
                            f"AMPA drive absent or grossly mis-scaled"
                        )))

        # ── GABA-B/GABA-A ratio check ────────────────────────────────────
        # GABA-B activates somatic K⁺ channels (τ ≈ 100–200 ms); GABA-A opens Cl⁻
        # channels (τ ≈ 10–20 ms).  Summing them in the E/I denominator conflates
        # distinct functional roles.  Healthy range: GABA-B/GABA-A ≈ 0.05–1.0
        # (Connors 1992 for cortex).  Both fields are NaN unless GABA-B was
        # sampled in full mode.
        if (
            not np.isnan(rs.mean_g_gaba_a)
            and not np.isnan(rs.mean_g_gaba_b)
            and rs.mean_g_gaba_a > 0
        ):
            gaba_b_ratio = rs.mean_g_gaba_b / rs.mean_g_gaba_a
            if gaba_b_ratio > 2.0:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.EI_BALANCE, region=rn,
                    message=(
                        f"GABA-B dominant: {rn}  "
                        f"GABA-B/GABA-A = {gaba_b_ratio:.2f}  (expected \u2264 1.0) — "
                        f"slow K\u207a inhibition overwhelming fast Cl\u207b inhibition; "
                        f"check inhibitory synaptic weight calibration"
                    )))
            elif gaba_b_ratio < 0.02 and rs.mean_g_gaba_b > 1e-9:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.EI_BALANCE, region=rn,
                    message=(
                        f"Negligible GABA-B: {rn}  "
                        f"GABA-B/GABA-A = {gaba_b_ratio:.3f}  (expected \u2265 0.05) — "
                        f"slow inhibitory component absent or disconnected"
                    )))


def check_oscillatory_bands(
    region_stats: Dict[str, RegionStats],
    oscillations: OscillatoryStats,
    issues: List[HealthIssue],
) -> None:
    """Dominant oscillatory band + D1/D2 MSN competition index checks."""
    # Per-region dominant oscillatory band health check
    for rn, rs in region_stats.items():
        if not rs.is_active:
            continue
        dom_band = oscillations.region_dominant_band.get(rn, "none")
        if dom_band == "none":
            continue
        exp_band = expected_dominant_band(rn)
        if exp_band is not None and dom_band != exp_band:
            dom_freq = oscillations.region_dominant_freq.get(rn, 0.0)
            issues.append(HealthIssue(severity="warning", category=HealthCategory.OSCILLATIONS, region=rn,
                message=f"Unexpected dominant band: {rn}  "
                        f"expected={exp_band}  observed={dom_band}  "
                        f"(peak={dom_freq:.1f} Hz) — possible pathological synchrony"))

    # Branching ratio σ criticality check (Beggs & Plenz 2003).
    # σ > 1.05 → supercritical: recurrent excitation dominates, seizure risk.
    # σ < 0.5  → strongly subcritical: very low spike propagation (disconnection or
    #             over-inhibition); informational but not necessarily pathological.
    sigma = oscillations.avalanche_branching_ratio
    if not np.isnan(sigma):
        if sigma > 1.05:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.OSCILLATIONS,
                message=(
                    f"Supercritical network: branching ratio σ={sigma:.3f} > 1.05 — "
                    f"runaway spike propagation; reduce recurrent excitation or strengthen inhibition. "
                    f"(Exponent={oscillations.avalanche_exponent:.2f}, R²={oscillations.avalanche_r2:.2f})"
                )))
        elif sigma < 0.5:
            issues.append(HealthIssue(
                severity="info",
                category=HealthCategory.OSCILLATIONS,
                message=(
                    f"Strongly subcritical network: branching ratio σ={sigma:.3f} < 0.5 — "
                    f"very low spike propagation; check for disconnected regions or excessive inhibition. "
                    f"(Exponent={oscillations.avalanche_exponent:.2f}, R²={oscillations.avalanche_r2:.2f})"
                )))

    # D1/D2 MSN competition index (striatal regions).
    # Checked at two timescales: 200 ms (action-selection epoch, Mink 1996) and
    # 50 ms (striatal mutual-inhibition timescale).  The 50 ms check catches
    # sub-window alternation that the 200 ms bins blend to near-zero correlation.
    for rn_d, rs_d in region_stats.items():
        ci_200 = rs_d.d1_d2_competition_index_200ms
        ci_50 = rs_d.d1_d2_competition_index_50ms
        if not np.isnan(ci_200) and ci_200 > 0.3:
            issues.append(HealthIssue(severity="warning", category=HealthCategory.OSCILLATIONS, region=rn_d,
                message=f"D1/D2 co-activation (200 ms bins): {rn_d}  CI={ci_200:.2f} "
                        f"(>0.3) \u2014 D1 and D2 MSNs firing together; "
                        f"mutual inhibition insufficient for action-selection"))
        # Emit fine-grained warning only when the 200 ms check did not already
        # catch co-activation, to avoid duplicate warnings.
        if not np.isnan(ci_50) and ci_50 > 0.3 and (np.isnan(ci_200) or ci_200 <= 0.3):
            issues.append(HealthIssue(severity="warning", category=HealthCategory.OSCILLATIONS, region=rn_d,
                message=f"D1/D2 co-activation (50 ms bins): {rn_d}  CI={ci_50:.2f} "
                        f"(>0.3) \u2014 sub-window co-activation missed by 200 ms bins; "
                        f"mutual inhibition insufficient at the competition timescale"))
        # Info: 50 ms bins show healthy competition that 200 ms bins obscure.
        if (not np.isnan(ci_50) and ci_50 < -0.3
                and not np.isnan(ci_200) and ci_200 > -0.1):
            issues.append(HealthIssue(severity="info", category=HealthCategory.OSCILLATIONS, region=rn_d,
                message=f"D1/D2 sub-window competition: {rn_d}  "
                        f"CI_50ms={ci_50:.2f}  CI_200ms={ci_200:.2f} \u2014 "
                        f"correct mutual inhibition at striatal timescale; "
                        f"200 ms bins obscure the signal"))

    # Beta burst duration check (BG and motor cortex regions).
    # Healthy motor cortex: mean ≈ 100–200 ms, max < 400 ms (Tinkhauser et al. 2017;
    # Sherman et al. 2016).  Prolonged bursts (> 400 ms) indicate pathological
    # STN–GPe synchronisation typical of Parkinson's disease.
    for rn_bb, bb in oscillations.beta_burst_stats.items():
        n_b = bb.get("n_bursts", 0.0)
        max_dur = bb.get("max_duration_ms", float("nan"))
        mean_dur = bb.get("mean_duration_ms", float("nan"))
        mean_ibi = bb.get("mean_ibi_ms", float("nan"))
        if n_b < 3 or np.isnan(max_dur):
            continue  # Too few bursts for reliable statistics
        if max_dur > 400.0:
            ibi_str = f"  IBI={mean_ibi:.0f} ms" if not np.isnan(mean_ibi) else ""
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.OSCILLATIONS,
                region=rn_bb,
                message=(
                    f"Pathological beta burst: {rn_bb}  max={max_dur:.0f} ms  "
                    f"mean={mean_dur:.0f} ms  n={n_b:.0f}{ibi_str} \u2014 "
                    f"prolonged beta synchrony (>400 ms) suggests STN\u2013GPe driven "
                    f"pathological synchronisation (Tinkhauser et al. 2017)"
                )))

    # Cross-regional coherence health thresholds (COHERENCE_SPECS).
    # Matrices are NaN-initialised; uncomputed pairs stay NaN and are skipped.
    _band_matrix = {
        "theta": oscillations.coherence_theta,
        "beta":  oscillations.coherence_beta,
        "gamma": oscillations.coherence_gamma,
    }
    ro = oscillations.region_order
    for cspec in COHERENCE_SPECS:
        matrix = _band_matrix.get(cspec.band)
        if matrix is None:
            continue
        idxs_a = [i for i, rn in enumerate(ro) if cspec.region_a in rn.lower()]
        idxs_b = [i for i, rn in enumerate(ro) if cspec.region_b in rn.lower()]
        for ia in idxs_a:
            for ib in idxs_b:
                if ia == ib:
                    continue
                val = float(matrix[ia, ib])
                if np.isnan(val):
                    continue  # pair not computed (sparse sampling)
                if val < cspec.expected_min:
                    issues.append(HealthIssue(
                        severity="warning",
                        category=HealthCategory.OSCILLATIONS,
                        region=ro[ia],
                        message=(
                            f"Low {cspec.band} coherence: {ro[ia]} \u2194 {ro[ib]}  "
                            f"{cspec.band}={val:.3f} "
                            f"(expected \u2265 {cspec.expected_min:.2f})  "
                            f"\u2014 {cspec.note}"
                        ),
                    ))

    # Global gamma hypercoherence check.
    # Any region-pair gamma coherence > 0.7 is pathological (Uhlhaas & Singer 2006):
    # it reflects epileptiform synchronisation, not healthy feature-binding.
    gamma_matrix = oscillations.coherence_gamma
    if gamma_matrix is not None:
        n_reg = len(ro)
        for ia in range(n_reg):
            for ib in range(ia + 1, n_reg):
                val = float(gamma_matrix[ia, ib])
                if np.isnan(val):
                    continue
                if val > 0.70:
                    issues.append(HealthIssue(
                        severity="critical",
                        category=HealthCategory.OSCILLATIONS,
                        region=ro[ia],
                        message=(
                            f"GAMMA HYPERCOHERENCE: {ro[ia]} \u2194 {ro[ib]}  "
                            f"gamma={val:.3f} (>0.70) \u2014 pathological synchrony; "
                            f"check for runaway E/I imbalance or epileptiform activity "
                            f"(Uhlhaas & Singer 2006)"
                        ),
                    ))


def check_integration_tau(
    oscillations: OscillatoryStats,
    issues: List[HealthIssue],
) -> None:
    """Check firing-rate autocorrelation time constants against biological references.

    The temporal integration time constant τ_int reflects intrinsic neural memory
    and varies systematically across the cortical hierarchy (Murray et al. 2014):

    - Prefrontal cortex: τ ≈ 100–400 ms
    - Motor cortex:      τ ≈  40–150 ms
    - Sensory cortex:    τ ≈  20– 50 ms

    Deviations indicate:
    - τ too short for PFC: persistent activity absent — working-memory substrate broken.
    - τ too long for sensory cortex: sensory cortex stuck in reverberant activity;
      possible runaway recurrent excitation.
    """
    for rn, tau_ms in oscillations.region_integration_tau_ms.items():
        if np.isnan(tau_ms):
            continue
        ref = integration_tau_range(rn)
        if ref is None:
            continue
        tau_min, tau_max = ref
        if tau_ms < tau_min:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.OSCILLATIONS,
                region=rn,
                message=(
                    f"Short integration τ: {rn}  τ={tau_ms:.0f} ms "
                    f"(expected {tau_min:.0f}–{tau_max:.0f} ms; "
                    f"Murray et al. 2014) — "
                    + (
                        "working-memory persistent activity absent (PFC)"
                        if matches_any(rn, PREFRONTAL_TAGS)
                        else "intrinsic temporal integration weaker than expected"
                    )
                ),
            ))
        elif tau_ms > tau_max:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.OSCILLATIONS,
                region=rn,
                message=(
                    f"Long integration τ: {rn}  τ={tau_ms:.0f} ms "
                    f"(expected {tau_min:.0f}–{tau_max:.0f} ms; "
                    f"Murray et al. 2014) — "
                    + (
                        "sensory cortex showing reverberant integration; "
                        "check recurrent excitation strength"
                        if "cortex_sensory" in rn.lower()
                        else "integration time constant exceeds expected range"
                    )
                ),
            ))


def check_tau_hierarchy(
    oscillations: OscillatoryStats,
    issues: List[HealthIssue],
) -> None:
    """Check that the cortical τ_int gradient is preserved across the hierarchy.

    Murray et al. 2014 (*Nature Neuroscience*) showed that the population-rate
    autocorrelation time constant increases monotonically from sensory to
    prefrontal cortex:

        cortex_sensory  τ ≈  20– 50 ms
        cortex_motor    τ ≈  40–150 ms
        prefrontal      τ ≈ 100–400 ms

    Each region is checked individually by :func:`check_integration_tau`.  This
    function performs the *cross-tier* check: even when every region’s τ is
    within its own reference range, the ordering can still be violated (e.g.
    PFC τ = 105 ms < motor cortex τ = 148 ms, both within range but inverted).

    The check requires that the *mean* τ over all regions in each tier satisfies
    the ordering.  Filtering to mean (rather than min/max) is robust to brains
    with multiple cortical sub-regions in the same tier.  The check is skipped
    when fewer than two tiers have valid (non-NaN) tau values.
    """
    tau_map = oscillations.region_integration_tau_ms

    sensory_taus = {
        rn: tau for rn, tau in tau_map.items()
        if not np.isnan(tau) and "cortex_sensory" in rn.lower()
    }
    motor_taus = {
        rn: tau for rn, tau in tau_map.items()
        if not np.isnan(tau) and "cortex_motor" in rn.lower()
    }
    pfc_taus = {
        rn: tau for rn, tau in tau_map.items()
        if not np.isnan(tau) and matches_any(rn, PREFRONTAL_TAGS)
    }

    # Skip if fewer than 2 tiers present (can’t assess a gradient).
    if sum(bool(t) for t in [sensory_taus, motor_taus, pfc_taus]) < 2:
        return

    def _mean(d: Dict[str, float]) -> float:
        return sum(d.values()) / len(d)

    s_mean = _mean(sensory_taus) if sensory_taus else None
    m_mean = _mean(motor_taus)   if motor_taus   else None
    p_mean = _mean(pfc_taus)     if pfc_taus     else None

    pairs: List[tuple[str, str]] = []
    if s_mean is not None and m_mean is not None and s_mean > m_mean:
        pairs.append((f"sensory τ={s_mean:.0f} ms", f"motor τ={m_mean:.0f} ms"))
    if m_mean is not None and p_mean is not None and m_mean > p_mean:
        pairs.append((f"motor τ={m_mean:.0f} ms", f"PFC τ={p_mean:.0f} ms"))
    if s_mean is not None and p_mean is not None and m_mean is None and s_mean > p_mean:
        pairs.append((f"sensory τ={s_mean:.0f} ms", f"PFC τ={p_mean:.0f} ms"))

    for higher, lower in pairs:
        issues.append(HealthIssue(
            severity="warning",
            category=HealthCategory.OSCILLATIONS,
            message=(
                f"Inverted cortical τ_int hierarchy: {higher} > {lower} — "
                f"expected sensory ≤ motor ≤ PFC "
                f"(Murray et al. 2014 Nature Neurosci)"
            ),
        ))


def check_pac(
    oscillations: OscillatoryStats,
    issues: List[HealthIssue],
) -> None:
    """Check phase-amplitude coupling (PAC) indices against biological expectations.

    Uses the Mean Vector Length (MVL) modulation index stored in
    :attr:`~.OscillatoryStats.pac_modulation_index`, computed by
    :func:`~.analysis_oscillations.compute_pac_per_region` for regions
    matching :data:`~.bio_ranges.PAC_SPECS` entries.

    Expected coupling under driven (waking-state) conditions:

    * Hippocampus theta–gamma: MVL ≈ 0.01–0.20 (active encoding).
      MVL < 0.005 → theta or local gamma generator absent
      (Canolty et al. 2006; Lisman & Jensen 2013).
    * Motor cortex / striatum beta–gamma: MVL < 0.003 → beta drive or
      gamma generator absent (Yanovsky et al. 2012; Crone et al. 2006).

    NaN values (simulation too short, or fs too low for gamma) are silently
    skipped — the analysis module documents when PAC cannot be computed.
    """
    # Minimum expected MVL per region substring.
    # Values are deliberately conservative (lower than active-encoding peaks)
    # to avoid false positives under background / spontaneous conditions.
    _PAC_MIN: dict[str, float] = {
        "hippocampus":  0.005,
        "cortex_motor": 0.003,
        "striatum":     0.003,
    }

    for rn, mi in oscillations.pac_modulation_index.items():
        if np.isnan(mi):
            continue
        threshold = next(
            (v for k, v in _PAC_MIN.items() if k in rn.lower()),
            0.003,
        )
        if mi < threshold:
            # Identify the expected phase and amplitude bands from PAC_SPECS so
            # the message names them explicitly rather than hard-coding them here.
            from .bio_ranges import PAC_SPECS
            spec = next(
                (s for s in PAC_SPECS if s.region.lower() in rn.lower()), None
            )
            band_info = (
                f"{spec.phase_band}–{spec.amp_band} PAC"
                if spec is not None
                else "PAC"
            )
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.OSCILLATIONS,
                region=rn,
                message=(
                    f"Weak {band_info}: {rn}  MVL={mi:.4f} < {threshold:.3f} — "
                    f"phase-amplitude coupling absent; "
                    f"check {'theta' if 'hippocampus' in rn.lower() else 'beta'} "
                    f"rhythm generation and local gamma drive "
                    f"(Canolty et al. 2006; Lisman & Jensen 2013)"
                ),
            ))
