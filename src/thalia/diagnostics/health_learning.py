"""Health checks for learning / training diagnostics.

Detects pathological learning states: weight collapse, weight explosion,
weight homogenization, dead plasticity, eligibility trace saturation,
BCM threshold saturation, homeostatic interference with learning, and
representational instability.
"""

from __future__ import annotations

from typing import List

import numpy as np

from .diagnostics_config import LearningThresholds
from .diagnostics_metrics import LearningStats
from .diagnostics_report import HealthCategory, HealthIssue
from .diagnostics_snapshot import RecorderSnapshot
from .health_context import HealthCheckContext

_CAT = HealthCategory.LEARNING


def check_weight_distribution(learning: LearningStats, issues: List[HealthIssue], config: LearningThresholds) -> None:
    """Check for weight collapse, explosion, and homogenization."""
    for key, summary in learning.synapse_summaries.items():
        w_end = summary.weight_end

        # Weight collapse: all weights near w_min
        if w_end.sparsity > config.weight_sparsity_critical:
            issues.append(HealthIssue(
                severity="critical",
                category=_CAT,
                message=(
                    f"WEIGHT COLLAPSE: {key} — {w_end.sparsity:.0%} of weights "
                    f"are near-zero (mean={w_end.mean:.4f}). Learning is dead."
                ),
                region=key.split(":")[0],
            ))
        elif w_end.sparsity > config.weight_sparsity_warn:
            issues.append(HealthIssue(
                severity="warning",
                category=_CAT,
                message=(
                    f"Weight sparsification: {key} — {w_end.sparsity:.0%} of weights "
                    f"near-zero (mean={w_end.mean:.4f}). Risk of collapse."
                ),
                region=key.split(":")[0],
            ))

        # Weight explosion: weights at w_max
        if w_end.max_val > 0 and w_end.min_val / max(w_end.max_val, 1e-10) > 0.95:
            # All weights near the same max value → saturated
            if w_end.std < config.weight_saturation_cv * max(abs(w_end.mean), 1e-6):
                issues.append(HealthIssue(
                    severity="critical",
                    category=_CAT,
                    message=(
                        f"WEIGHT SATURATION: {key} — all weights converged to "
                        f"{w_end.mean:.4f} (std={w_end.std:.6f}). Selectivity lost."
                    ),
                    region=key.split(":")[0],
                ))

        # Weight homogenization: std → 0 relative to mean
        if abs(w_end.mean) > 1e-6 and w_end.std / abs(w_end.mean) < config.weight_homogenization_cv and w_end.sparsity < 0.5:
            issues.append(HealthIssue(
                severity="warning",
                category=_CAT,
                message=(
                    f"Weight homogenization: {key} — "
                    f"CV={w_end.std / abs(w_end.mean):.4f} (<{config.weight_homogenization_cv}). "
                    f"All synapses encode the same strength; selectivity is lost."
                ),
                region=key.split(":")[0],
            ))

        # Large drift
        if abs(summary.weight_drift) > config.weight_drift_warn:
            severity = "critical" if abs(summary.weight_drift) > config.weight_drift_critical else "warning"
            direction = "increased" if summary.weight_drift > 0 else "decreased"
            issues.append(HealthIssue(
                severity=severity,
                category=_CAT,
                message=(
                    f"Weight drift: {key} — mean weight {direction} by "
                    f"{abs(summary.weight_drift):.0%} over the run "
                    f"({summary.weight_start.mean:.4f} → {summary.weight_end.mean:.4f})."
                ),
                region=key.split(":")[0],
            ))


def check_weight_update_magnitude(learning: LearningStats, issues: List[HealthIssue], config: LearningThresholds) -> None:
    """Check for dead plasticity (no updates) or unstable learning (huge updates)."""
    for key, summary in learning.synapse_summaries.items():
        if summary.mean_update_magnitude < config.dead_plasticity:
            issues.append(HealthIssue(
                severity="warning",
                category=_CAT,
                message=(
                    f"Dead plasticity: {key} — mean |ΔW|/|W| = "
                    f"{summary.mean_update_magnitude:.2e}. Weights are not changing."
                ),
                region=key.split(":")[0],
            ))
        elif summary.mean_update_magnitude > config.unstable_learning_warn:
            severity = "critical" if summary.mean_update_magnitude > config.unstable_learning_critical else "warning"
            issues.append(HealthIssue(
                severity=severity,
                category=_CAT,
                message=(
                    f"Unstable learning: {key} — mean |ΔW|/|W| = "
                    f"{summary.mean_update_magnitude:.4f} per sample interval. "
                    f"Weights are oscillating or diverging."
                ),
                region=key.split(":")[0],
            ))


def check_eligibility_traces(learning: LearningStats, issues: List[HealthIssue], config: LearningThresholds) -> None:
    """Check eligibility trace health: zero traces, chronic imbalance."""
    for key, summary in learning.synapse_summaries.items():
        if np.isnan(summary.mean_eligibility):
            continue  # Strategy doesn't use eligibility traces

        if summary.mean_eligibility < 1e-8:
            issues.append(HealthIssue(
                severity="warning",
                category=_CAT,
                message=(
                    f"Zero eligibility: {key} — mean |eligibility| = "
                    f"{summary.mean_eligibility:.2e}. Reward signals will have "
                    f"nothing to gate. Check that pre/post spikes are reaching "
                    f"this synapse."
                ),
                region=key.split(":")[0],
            ))

        # LTP/LTD imbalance
        if not np.isnan(summary.ltp_ltd_ratio):
            if summary.ltp_ltd_ratio > config.ltp_ltd_ratio_high:
                issues.append(HealthIssue(
                    severity="warning",
                    category=_CAT,
                    message=(
                        f"LTP-dominant eligibility: {key} — LTP/LTD ratio = "
                        f"{summary.ltp_ltd_ratio:.1f}. Chronic potentiation bias "
                        f"may lead to weight explosion."
                    ),
                    region=key.split(":")[0],
                ))
            elif summary.ltp_ltd_ratio < config.ltp_ltd_ratio_low:
                issues.append(HealthIssue(
                    severity="warning",
                    category=_CAT,
                    message=(
                        f"LTD-dominant eligibility: {key} — LTP/LTD ratio = "
                        f"{summary.ltp_ltd_ratio:.2f}. Chronic depression bias "
                        f"may lead to weight collapse."
                    ),
                    region=key.split(":")[0],
                ))


def check_bcm_thresholds(learning: LearningStats, issues: List[HealthIssue], config: LearningThresholds) -> None:
    """Check BCM sliding threshold dynamics."""
    for key, summary in learning.synapse_summaries.items():
        if np.isnan(summary.bcm_theta_start):
            continue  # Not a BCM strategy

        # Theta stuck at extremes
        if summary.bcm_theta_end < config.bcm_theta_collapsed:
            issues.append(HealthIssue(
                severity="critical",
                category=_CAT,
                message=(
                    f"BCM theta collapsed: {key} — θ = {summary.bcm_theta_end:.6f} "
                    f"(near θ_min). All activity produces LTP — runaway potentiation."
                ),
                region=key.split(":")[0],
            ))

        if summary.bcm_theta_end > config.bcm_theta_saturated:
            issues.append(HealthIssue(
                severity="warning",
                category=_CAT,
                message=(
                    f"BCM theta saturated: {key} — θ = {summary.bcm_theta_end:.4f} "
                    f"(near θ_max=0.5). Depression dominates — weight collapse risk."
                ),
                region=key.split(":")[0],
            ))


def check_da_eligibility_alignment(learning: LearningStats, issues: List[HealthIssue], config: LearningThresholds) -> None:
    """Check DA–eligibility temporal alignment for three-factor learning."""
    for key, alignment in learning.da_eligibility_alignment.items():
        if alignment < config.da_eligibility_min:
            issues.append(HealthIssue(
                severity="warning",
                category=_CAT,
                message=(
                    f"DA–eligibility misaligned: {key} — only {alignment:.0%} of "
                    f"eligibility episodes overlap with phasic DA. Reward signal "
                    f"is not reaching synapses when traces are non-zero."
                ),
                region=key.split(":")[0],
            ))


def check_homeostatic_plasticity_interaction(learning: LearningStats, rec: RecorderSnapshot, issues: List[HealthIssue], config: LearningThresholds) -> None:
    """Detect homeostasis undoing learning by excessive correction rate."""
    n_steps = rec._gain_sample_step
    if n_steps < 10:
        return

    for pop_key, corr_rate in learning.homeostatic_correction_rate.items():
        valid = corr_rate[~np.isnan(corr_rate)]
        if len(valid) < 10:
            continue

        # Split into first and second half
        half = len(valid) // 2
        first_half_rate = float(np.mean(valid[:half]))
        second_half_rate = float(np.mean(valid[half:]))

        # If correction rate is increasing during training, homeostasis is
        # fighting harder — likely undoing learning
        if (
            first_half_rate > 1e-6
            and second_half_rate > config.homeostatic_correction_doubling * first_half_rate
            and second_half_rate > 1e-4
        ):
            issues.append(HealthIssue(
                severity="warning",
                category=_CAT,
                message=(
                    f"Homeostatic interference: {pop_key} — correction rate "
                    f"doubled from {first_half_rate:.6f} to {second_half_rate:.6f}. "
                    f"Homeostasis may be undoing learning-driven weight changes."
                ),
                population=pop_key,
                region=pop_key.split(":")[0],
            ))


def check_representational_stability(learning: LearningStats, issues: List[HealthIssue], config: LearningThresholds) -> None:
    """Check population vector stability across the run."""
    if np.isnan(learning.popvec_stability):
        return

    if learning.popvec_stability < config.popvec_stability_min:
        issues.append(HealthIssue(
            severity="warning",
            category=_CAT,
            message=(
                f"Representational instability: population vector correlation = "
                f"{learning.popvec_stability:.3f}. Neural representations are "
                f"drifting rapidly, which may prevent stable learning."
            ),
        ))


def check_stdp_timing_balance(learning: LearningStats, issues: List[HealthIssue], config: LearningThresholds) -> None:
    """Check STDP timing distributions for extreme LTP/LTD imbalance."""
    for key, timing in learning.stdp_timing.items():
        region = key.split(":")[0]
        if timing.ltp_fraction > config.stdp_ltp_fraction_high:
            issues.append(HealthIssue(
                severity="warning",
                category=_CAT,
                message=(
                    f"LTP-dominant spike timing: {key} — {timing.ltp_fraction:.0%} "
                    f"of {timing.n_pairs} spike pairs have pre-before-post timing "
                    f"(mean \u0394t = {timing.mean_delta_ms:+.1f} ms). "
                    f"Chronic potentiation bias may cause weight explosion."
                ),
                region=region,
            ))
        elif timing.ltp_fraction < config.stdp_ltp_fraction_low:
            issues.append(HealthIssue(
                severity="warning",
                category=_CAT,
                message=(
                    f"LTD-dominant spike timing: {key} — only {timing.ltp_fraction:.0%} "
                    f"of {timing.n_pairs} spike pairs have pre-before-post timing "
                    f"(mean \u0394t = {timing.mean_delta_ms:+.1f} ms). "
                    f"Chronic depression bias may cause weight collapse."
                ),
                region=region,
            ))


def check_learning_health(ctx: HealthCheckContext) -> None:
    """Run all learning health checks. No-op when learning is not active."""
    learning, rec, issues = ctx.learning, ctx.rec, ctx.issues
    if learning is None:
        return

    config = ctx.thresholds.learning
    check_weight_distribution(learning, issues, config)
    check_weight_update_magnitude(learning, issues, config)
    check_eligibility_traces(learning, issues, config)
    check_bcm_thresholds(learning, issues, config)
    check_da_eligibility_alignment(learning, issues, config)
    check_homeostatic_plasticity_interaction(learning, rec, issues, config)
    check_representational_stability(learning, issues, config)
    check_stdp_timing_balance(learning, issues, config)
