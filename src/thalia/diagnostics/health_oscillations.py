"""Oscillatory band, coherence, PAC, and criticality health checks."""

from __future__ import annotations

import numpy as np

from .bio_ranges import CFC_SPECS, COHERENCE_SPECS, expected_dominant_band
from .diagnostics_report import HealthCategory, HealthIssue
from .health_context import HealthCheckContext


def check_oscillatory_bands(ctx: HealthCheckContext) -> None:
    """Dominant oscillatory band + D1/D2 MSN competition index checks."""
    region_stats, oscillations, issues = ctx.region_stats, ctx.oscillations, ctx.issues
    config = ctx.thresholds.oscillations
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
    sigma = oscillations.avalanche.branching_ratio
    if not np.isnan(sigma):
        if sigma > config.branching_ratio_supercritical:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.OSCILLATIONS,
                message=(
                    f"Supercritical network: branching ratio σ={sigma:.3f} > 1.05 — "
                    f"runaway spike propagation; reduce recurrent excitation or strengthen inhibition. "
                    f"(Exponent={oscillations.avalanche.exponent:.2f}, R²={oscillations.avalanche.r2:.2f})"
                )))
        elif sigma < config.branching_ratio_subcritical:
            issues.append(HealthIssue(
                severity="info",
                category=HealthCategory.OSCILLATIONS,
                message=(
                    f"Strongly subcritical network: branching ratio σ={sigma:.3f} < 0.5 — "
                    f"very low spike propagation; check for disconnected regions or excessive inhibition. "
                    f"(Exponent={oscillations.avalanche.exponent:.2f}, R²={oscillations.avalanche.r2:.2f})"
                )))

    # D1/D2 MSN competition index (striatal regions).
    # Checked at two timescales: 200 ms (action-selection epoch, Mink 1996) and
    # 50 ms (striatal mutual-inhibition timescale).  The 50 ms check catches
    # sub-window alternation that the 200 ms bins blend to near-zero correlation.
    for rn_d, rs_d in region_stats.items():
        ci_200 = rs_d.d1_d2_competition_index_200ms
        ci_50 = rs_d.d1_d2_competition_index_50ms
        if not np.isnan(ci_200) and ci_200 > config.d1_d2_coactivation:
            issues.append(HealthIssue(severity="warning", category=HealthCategory.OSCILLATIONS, region=rn_d,
                message=f"D1/D2 co-activation (200 ms bins): {rn_d}  CI={ci_200:.2f} "
                        f"(>{config.d1_d2_coactivation}) \u2014 D1 and D2 MSNs firing together; "
                        f"mutual inhibition insufficient for action-selection"))
        # Emit fine-grained warning only when the 200 ms check did not already
        # catch co-activation, to avoid duplicate warnings.
        if not np.isnan(ci_50) and ci_50 > config.d1_d2_coactivation and (np.isnan(ci_200) or ci_200 <= config.d1_d2_coactivation):
            issues.append(HealthIssue(severity="warning", category=HealthCategory.OSCILLATIONS, region=rn_d,
                message=f"D1/D2 co-activation (50 ms bins): {rn_d}  CI={ci_50:.2f} "
                        f"(>{config.d1_d2_coactivation}) \u2014 sub-window co-activation missed by 200 ms bins; "
                        f"mutual inhibition insufficient at the competition timescale"))
        # Info: 50 ms bins show healthy competition that 200 ms bins obscure.
        if (not np.isnan(ci_50) and ci_50 < -config.d1_d2_coactivation
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
        n_b = bb.n_bursts
        max_dur = bb.max_duration_ms
        mean_dur = bb.mean_duration_ms
        mean_ibi = bb.mean_ibi_ms
        if n_b < 3 or np.isnan(max_dur):
            continue  # Too few bursts for reliable statistics
        if max_dur > config.beta_burst_max_ms:
            ibi_str = f"  IBI={mean_ibi:.0f} ms" if not np.isnan(mean_ibi) else ""
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.OSCILLATIONS,
                region=rn_bb,
                message=(
                    f"Pathological beta burst: {rn_bb}  max={max_dur:.0f} ms  "
                    f"mean={mean_dur:.0f} ms  n={n_b:.0f}{ibi_str} \u2014 "
                    f"prolonged beta synchrony (>{config.beta_burst_max_ms:.0f} ms) suggests STN\u2013GPe driven "
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
    # Literature threshold from MEG/LFP is ~0.70 (Uhlhaas & Singer 2006), but
    # spike-based coherence in simulation lacks volume conduction noise,
    # electrode misalignment, and recording incompleteness — all of which
    # reduce real-brain coherence by 10–30 %.  We raise to 0.85 to avoid
    # false positives in the noise-free simulated setting.
    gamma_matrix = oscillations.coherence_gamma
    if gamma_matrix is not None:
        n_reg = len(ro)
        for ia in range(n_reg):
            for ib in range(ia + 1, n_reg):
                val = float(gamma_matrix[ia, ib])
                if np.isnan(val):
                    continue
                if val > config.gamma_hypercoherence:
                    issues.append(HealthIssue(
                        severity="critical",
                        category=HealthCategory.OSCILLATIONS,
                        region=ro[ia],
                        message=(
                            f"GAMMA HYPERCOHERENCE: {ro[ia]} \u2194 {ro[ib]}  "
                            f"gamma={val:.3f} (>{config.gamma_hypercoherence}) \u2014 pathological synchrony; "
                            f"check for runaway E/I imbalance or epileptiform activity "
                            f"(Uhlhaas & Singer 2006; threshold raised for spike-based data)"
                        ),
                    ))


def check_cfc(ctx: HealthCheckContext) -> None:
    """Check cross-frequency coupling results against biological expectations.

    Uses the :class:`~.diagnostics_metrics.CFCResult` entries stored in
    :attr:`~.OscillatoryStats.cfc_results`, computed by
    :func:`~.coupling.cross_frequency.compute_cfc_per_region` for regions
    matching :data:`~.bio_spectral_specs.CFC_SPECS` entries.

    Each spec carries an ``expected_min`` threshold.  When a measured value
    falls below this threshold (and is not NaN), a warning is issued.

    NaN values (simulation too short or fs too low) are silently skipped.
    """
    oscillations, issues = ctx.oscillations, ctx.issues

    _TYPE_LABELS = {"pac": "PAC", "aac": "AAC", "pfc": "PFC"}

    for result in oscillations.cfc_results:
        if np.isnan(result.value):
            continue
        # Find the matching spec to get the threshold.
        spec = next(
            (s for s in CFC_SPECS
             if s.region.lower() in result.region.lower()
             and s.coupling_type == result.coupling_type
             and s.phase_band == result.phase_band
             and s.amp_band == result.amp_band),
            None,
        )
        if spec is None:
            continue
        if result.value < spec.expected_min:
            type_label = _TYPE_LABELS.get(result.coupling_type, result.coupling_type.upper())
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.OSCILLATIONS,
                region=result.region,
                message=(
                    f"Weak {result.phase_band}–{result.amp_band} {type_label}: "
                    f"{result.region}  value={result.value:.4f} < {spec.expected_min:.3f} — "
                    f"{spec.note}"
                ),
            ))


def check_aperiodic_exponent(ctx: HealthCheckContext) -> None:
    """Check the aperiodic (1/f) spectral exponent per region.

    The aperiodic component of the PSD follows PSD(f) ∝ 1/f^χ.
    The exponent χ reflects the excitation/inhibition balance and
    temporal correlation structure of neural activity:

    * χ ≈ 1.0–2.0: healthy cortical activity (He 2014; Donoghue et al. 2020).
    * χ < threshold_low (default 0.5): Flattened spectrum — reduced temporal
      correlations, consistent with epileptiform activity or noise-dominated
      dynamics (Voytek et al. 2015).
    * χ > threshold_high (default 3.0): Overly steep — activity dominated
      by slow fluctuations, consistent with over-inhibition or disconnected
      low-frequency drift.
    """
    oscillations, issues = ctx.oscillations, ctx.issues
    config = ctx.thresholds.oscillations

    # In NREM sleep, the aperiodic exponent is naturally steeper due to
    # slow-wave dominance (He 2014).  Relax the upper bound.
    chi_high = config.aperiodic_exponent_high
    if ctx.inferred_brain_state == "nrem":
        chi_high = max(chi_high, 4.0)

    for rn, chi in oscillations.region_aperiodic_exponent.items():
        if np.isnan(chi):
            continue
        if chi < config.aperiodic_exponent_low:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.OSCILLATIONS,
                region=rn,
                message=(
                    f"Flattened aperiodic spectrum: {rn}  χ={chi:.2f} "
                    f"(< {config.aperiodic_exponent_low}) — reduced temporal "
                    f"correlations; possible epileptiform or noise-dominated "
                    f"activity (He 2014; Donoghue et al. 2020)"
                ),
            ))
        elif chi > chi_high:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.OSCILLATIONS,
                region=rn,
                message=(
                    f"Steep aperiodic spectrum: {rn}  χ={chi:.2f} "
                    f"(> {chi_high}) — excessive slow-"
                    f"frequency dominance; possible over-inhibition or "
                    f"disconnected activity (He 2014; Donoghue et al. 2020)"
                ),
            ))
