"""E/I balance health checks.

Per-region excitatory/inhibitory conductance ratio, NMDA fraction,
and GABA-B/GABA-A ratio.
"""

from __future__ import annotations

import numpy as np

from ._helpers import compute_nmda_fraction
from .bio_ranges import ei_ratio_thresholds
from .diagnostics_report import HealthCategory, HealthIssue
from .health_context import HealthCheckContext
from .region_tags import NMDA_REGION_TAGS, matches_any


def check_ei_balance(ctx: HealthCheckContext) -> None:
    """E1: Per-region excitatory/inhibitory conductance balance check.

    Runs three independent sub-checks per region:
    - E/I ratio (AMPA+NMDA numerator, GABA-A+GABA-B denominator)
    - NMDA/(AMPA+NMDA) fraction for cortical/hippocampal regions (S1.1)
    - GABA-B/GABA-A ratio where both are sampled (S1.2)
    """
    region_stats, issues = ctx.region_stats, ctx.issues
    config = ctx.thresholds.connectivity
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
        # plasticity-relevant depolarisation.  We measure the NMDA *conductance*
        # fraction, not synaptic input ratios.  At sub-threshold voltages NMDA
        # is largely blocked by Mg²⁺ (Jahr & Stevens 1990), so the measured
        # conductance fraction is naturally lower than the 30–70 % input ratio
        # cited in voltage-clamp studies.  Healthy conductance-based range:
        # 5–70 % (Myme et al. 2003; Jahr & Stevens 1990).
        # Near-zero → channels not activating; near-100 % → AMPA absent.
        _nmda_region = matches_any(rn, NMDA_REGION_TAGS)
        if _nmda_region and not np.isnan(rs.mean_g_nmda) and not np.isnan(rs.mean_g_exc):
            nmda_frac = compute_nmda_fraction(rs.mean_g_exc, rs.mean_g_nmda)
            if not np.isnan(nmda_frac):
                if nmda_frac < config.nmda_fraction_low:
                    issues.append(HealthIssue(severity="warning", category=HealthCategory.EI_BALANCE, region=rn,
                        message=(
                            f"Low NMDA fraction: {rn}  "
                            f"NMDA/(AMPA+NMDA) = {nmda_frac:.2f}  (expected {config.nmda_fraction_low}–{config.nmda_fraction_high}) — "
                            f"NMDA channels may not be activating (Mg\u00b2\u207a block requires V > \u221240 mV)"
                        )))
                elif nmda_frac > config.nmda_fraction_high:
                    issues.append(HealthIssue(severity="warning", category=HealthCategory.EI_BALANCE, region=rn,
                        message=(
                            f"High NMDA fraction: {rn}  "
                            f"NMDA/(AMPA+NMDA) = {nmda_frac:.2f}  (expected {config.nmda_fraction_low}–{config.nmda_fraction_high} conductance fraction) — "
                            f"AMPA drive may be absent or grossly mis-scaled"
                        )))

        # ── GABA-B/GABA-A ratio check ────────────────────────────────────
        # GABA-B activates somatic K⁺ channels (τ ≈ 100–200 ms); GABA-A opens Cl⁻
        # channels (τ ≈ 10–20 ms).  Summing them in the E/I denominator conflates
        # distinct functional roles.  Healthy range: GABA-B/GABA-A ≈ 0.05–1.0.
        if (
            not np.isnan(rs.mean_g_gaba_a)
            and not np.isnan(rs.mean_g_gaba_b)
            and rs.mean_g_gaba_a > 0
        ):
            gaba_b_ratio = rs.mean_g_gaba_b / rs.mean_g_gaba_a
            if gaba_b_ratio > config.gaba_ba_ratio_critical:
                issues.append(HealthIssue(severity="critical", category=HealthCategory.EI_BALANCE, region=rn,
                    message=(
                        f"GABA-B dominant: {rn}  "
                        f"GABA-B/GABA-A = {gaba_b_ratio:.2f}  (expected \u2264 1.0) — "
                        f"slow K\u207a inhibition overwhelming fast Cl\u207b inhibition; "
                        f"somatic K\u207a channel saturation territory (Krabbe et al. 2019)"
                    )))
            elif gaba_b_ratio > config.gaba_ba_ratio_warn:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.EI_BALANCE, region=rn,
                    message=(
                        f"GABA-B elevated: {rn}  "
                        f"GABA-B/GABA-A = {gaba_b_ratio:.2f}  (expected \u2264 1.0) — "
                        f"slow K\u207a inhibition exceeding fast Cl\u207b inhibition; "
                        f"check inhibitory synaptic weight calibration"
                    )))
            elif gaba_b_ratio < config.gaba_ba_ratio_low and rs.mean_g_gaba_b > 1e-9:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.EI_BALANCE, region=rn,
                    message=(
                        f"Negligible GABA-B: {rn}  "
                        f"GABA-B/GABA-A = {gaba_b_ratio:.3f}  (expected \u2265 0.05) — "
                        f"slow inhibitory component absent or disconnected"
                    )))
