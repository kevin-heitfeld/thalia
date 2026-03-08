"""Homeostatic and STP health checks — gain convergence, STP state and directionality."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

from .diagnostics_types import (
    HealthCategory,
    HealthIssue,
    HomeostaticStats,
    PopulationStats,
)

if TYPE_CHECKING:
    from .diagnostics_recorder import DiagnosticsRecorder


def check_homeostasis(
    rec: "DiagnosticsRecorder",
    issues: List[HealthIssue],
    pop_stats: Dict[Tuple[str, str], PopulationStats] | None = None,
) -> None:
    """Check homeostatic gain collapse/convergence and STP efficacy convergence."""
    n_steps = rec._gain_sample_step

    # Check homeostatic gain collapse
    for idx, (rn, pn) in enumerate(rec._pop_keys):
        if n_steps > 0:
            final_gain = rec._g_L_scale_history[n_steps - 1, idx]
            if not np.isnan(final_gain) and final_gain < rec.config.thresholds.gain_collapse_threshold:
                issues.append(HealthIssue(severity="critical", category=HealthCategory.HOMEOSTASIS,
                    population=f"{rn}:{pn}", region=rn,
                    message=f"GAIN COLLAPSED: {rn}:{pn} g_L_scale = {final_gain:.3f}"))

    # Check homeostatic gain convergence
    # Compare first 10% vs last 10% of each trajectory; warn if drifting >15%.
    # Requires at least 20 sample points to have meaningful 10% segments.
    if n_steps >= 20:
        seg = max(1, n_steps // 10)
        for idx, (rn, pn) in enumerate(rec._pop_keys):
            traj = rec._g_L_scale_history[:n_steps, idx]
            valid = traj[~np.isnan(traj)]
            if len(valid) < 20:
                continue
            first_mean = float(np.mean(valid[:seg]))
            last_mean  = float(np.mean(valid[-seg:]))
            if abs(first_mean) < 1e-6:
                continue
            drift_pct = abs(last_mean - first_mean) / abs(first_mean) * 100.0
            if drift_pct > rec.config.thresholds.gain_drift_pct:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.HOMEOSTASIS,
                    population=f"{rn}:{pn}", region=rn,
                    message=f"Gain not converged: {rn}:{pn}  "
                            f"drift={drift_pct:.1f}%  "
                            f"({first_mean:.3f} \u2192 {last_mean:.3f})"))

    # Check homeostatic gain slope (monotonic drift at recording end).
    # Fit a linear regression to the LAST 50% of each gain trajectory.
    # If the implied total change over that half-window exceeds gain_slope_pct %
    # of the mean gain, the control loop has not levelled off — target rate may
    # be unreachable or η too small (issued in addition to, not instead of, the
    # first/last drift check above).
    if n_steps >= 40:  # need at least 20 points in the second half
        half = n_steps // 2
        xs = np.arange(half, dtype=np.float64)
        for idx, (rn, pn) in enumerate(rec._pop_keys):
            traj = rec._g_L_scale_history[n_steps - half:n_steps, idx]
            valid_mask = ~np.isnan(traj)
            if valid_mask.sum() < 20:
                continue
            y = traj[valid_mask]
            x = xs[valid_mask]
            mean_gain = float(np.mean(y))
            if abs(mean_gain) < 1e-6:
                continue
            slope = float(np.polyfit(x, y, 1)[0])
            # implied total change over the half-window as % of mean
            implied_pct = abs(slope) * (len(x) - 1) / abs(mean_gain) * 100.0
            if implied_pct > rec.config.thresholds.gain_slope_pct:
                direction = "rising" if slope > 0 else "falling"
                issues.append(HealthIssue(
                    severity="warning",
                    category=HealthCategory.HOMEOSTASIS,
                    population=f"{rn}:{pn}",
                    region=rn,
                    message=(
                        f"Homeostatic gain not converged: {rn}:{pn}  "
                        f"slope={slope:+.4f}/sample ({direction})  "
                        f"implied drift={implied_pct:.1f}% over last 50% of recording  "
                        f"\u2014 target rate may be unreachable or \u03b7 too small"
                    ),
                ))

    # Cross-check: gain converged but mean FR still outside target range.
    # Uses the PopulationHomeostasisState.target_firing_rate (spikes/ms) registered
    # in the brain at build time as the biological target.  Only fires when the gain
    # trajectory has settled (slope < threshold AND enough samples), ensuring that the
    # warning reflects a genuine equilibrium rather than a still-adapting transient.
    if pop_stats is not None and n_steps >= 40:
        half = n_steps // 2
        xs = np.arange(half, dtype=np.float64)
        for idx, (rn, pn) in enumerate(rec._pop_keys):
            # Require gain to have converged before flagging FR mismatch
            traj = rec._g_L_scale_history[n_steps - half:n_steps, idx]
            valid_mask = ~np.isnan(traj)
            if valid_mask.sum() < 20:
                continue
            y = traj[valid_mask]
            x = xs[valid_mask]
            mean_gain = float(np.mean(y))
            if abs(mean_gain) < 1e-6:
                continue
            slope = float(np.polyfit(x, y, 1)[0])
            implied_pct = abs(slope) * (len(x) - 1) / abs(mean_gain) * 100.0
            # Only proceed when gain has converged (slope implies <5% drift)
            if implied_pct > rec.config.thresholds.gain_slope_pct:
                continue  # still adapting — FR mismatch may resolve

            # Fetch the registered target rate from the brain object
            region = rec.brain.regions.get(rn)
            if region is None:
                continue
            homeostasis_state = getattr(region, "_homeostasis", {}).get(pn)
            if homeostasis_state is None:
                continue
            target_spikes_per_ms: float = float(homeostasis_state.target_firing_rate)
            if target_spikes_per_ms <= 0.0:
                continue
            target_hz = target_spikes_per_ms * 1000.0

            ps = pop_stats.get((rn, pn))
            if ps is None or ps.total_spikes < 10:
                continue
            mean_fr = ps.mean_fr_hz
            # Warn when mean FR deviates from target by more than ±50 %
            ratio = mean_fr / target_hz if target_hz > 0.0 else float("nan")
            if ratio > 2.0:
                issues.append(HealthIssue(
                    severity="warning",
                    category=HealthCategory.HOMEOSTASIS,
                    population=f"{rn}:{pn}",
                    region=rn,
                    message=(
                        f"Homeostatic target mismatch (gain converged): {rn}:{pn}  "
                        f"FR={mean_fr:.1f} Hz  target={target_hz:.1f} Hz  "
                        f"ratio={ratio:.1f}× — gain plateau reached but neuron fires "
                        f"above biological target; check target_firing_rate or input drive"
                    ),
                ))
            elif ratio < 0.5:
                issues.append(HealthIssue(
                    severity="warning",
                    category=HealthCategory.HOMEOSTASIS,
                    population=f"{rn}:{pn}",
                    region=rn,
                    message=(
                        f"Homeostatic target mismatch (gain converged): {rn}:{pn}  "
                        f"FR={mean_fr:.1f} Hz  target={target_hz:.1f} Hz  "
                        f"ratio={ratio:.2f}× — gain plateau reached but neuron fires "
                        f"below biological target; check input connectivity or weight scale"
                    ),
                ))

    # Check STP efficacy convergence
    if n_steps >= 20:
        seg_stp = max(1, n_steps // 10)
        for stp_idx, (rn, syn_id) in enumerate(rec._stp_keys):
            traj_stp = rec._stp_efficacy_history[:n_steps, stp_idx]
            valid_stp = traj_stp[~np.isnan(traj_stp)]
            if len(valid_stp) < 20:
                continue
            first_stp = float(np.mean(valid_stp[:seg_stp]))
            last_stp  = float(np.mean(valid_stp[-seg_stp:]))
            if abs(first_stp) < 1e-6:
                continue
            drift_stp = abs(last_stp - first_stp) / abs(first_stp) * 100.0
            if drift_stp > rec.config.thresholds.gain_drift_pct:
                issues.append(HealthIssue(severity="warning", category=HealthCategory.HOMEOSTASIS, region=rn,
                    message=f"STP not converged: {rn} [{syn_id}]  "
                            f"x\u00b7u drift={drift_stp:.1f}%  "
                            f"({first_stp:.3f} \u2192 {last_stp:.3f})  "
                            f"— firing-rate metrics may reflect a transient; use --warmup"))


def check_stp_directionality(
    rec: "DiagnosticsRecorder",
    issues: List[HealthIssue],
) -> None:
    """Check that STP x·u direction under drive matches the expected biology.

    Classifies each STP synapse as facilitating or depressing based on its
    :class:`~thalia.brain.synapses.stp.STPConfig`:

    * **Facilitating**: ``tau_f ≥ 0.5 × tau_d`` **and** ``U ≤ 0.25``
      — facilitation time constant is substantial; low baseline release probability.
    * **Depressing**:   ``tau_d ≥ 2 × tau_f`` **and** ``U ≥ 0.35``
      — depression recovery dominates; high baseline release probability.

    Theoretical quiescent x·u = U (when x=1, u=U at rest).  Under sustained
    drive:

    * Facilitating → u rises above U, x·u should exceed U (at least modestly).
    * Depressing   → x falls below 1, x·u should fall below U.

    A reversed direction (e.g. depressing STP on a thalamocortical projection
    intended to be facilitating) silently eliminates transmission during
    sustained sensory processing without triggering any convergence warning.

    Requires ≥ 20 gain-sample steps and at least some drive (quiescent runs
    will show x·u ≈ U regardless and produce no false alarms).

    References: Markram et al. 1998; Zucker & Regehr 2002 *Annu Rev Physiol*.
    """
    n_steps = rec._gain_sample_step
    if n_steps < 20:
        return

    seg = max(1, n_steps // 10)

    for stp_idx, (rn, syn_id) in enumerate(rec._stp_keys):
        stp_mod = rec.brain.regions[rn].stp_modules[syn_id]
        if not hasattr(stp_mod, "config"):
            continue
        cfg = stp_mod.config
        U_val   = float(cfg.U)
        tau_d   = float(cfg.tau_d)
        tau_f   = float(cfg.tau_f)

        is_facilitating = tau_f >= 0.5 * tau_d and U_val <= 0.25
        is_depressing   = tau_d >= 2.0 * tau_f  and U_val >= 0.35
        if not is_facilitating and not is_depressing:
            continue  # Mixed / pseudolinear — no single expected direction

        traj = rec._stp_efficacy_history[:n_steps, stp_idx]
        valid = traj[~np.isnan(traj)]
        if len(valid) < 20:
            continue

        last_mean = float(np.mean(valid[-seg:]))
        quiescent = U_val  # x=1, u=U at rest → x·u = U

        if is_facilitating and last_mean < quiescent * 0.95:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.HOMEOSTASIS,
                region=rn,
                message=(
                    f"STP direction reversed (facilitating expected): {rn} [{syn_id}]  "
                    f"x\u00b7u={last_mean:.3f} < U={quiescent:.3f}  "
                    f"(\u03c4f={tau_f:.0f}ms, \u03c4d={tau_d:.0f}ms \u2014 synapse should "
                    f"facilitate under drive but is depressing; check STPConfig)"
                ),
            ))
        elif is_depressing and last_mean > quiescent * 1.05:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.HOMEOSTASIS,
                region=rn,
                message=(
                    f"STP direction reversed (depressing expected): {rn} [{syn_id}]  "
                    f"x\u00b7u={last_mean:.3f} > U={quiescent:.3f}  "
                    f"(\u03c4d={tau_d:.0f}ms, \u03c4f={tau_f:.0f}ms \u2014 synapse should "
                    f"depress under drive but is facilitating; check STPConfig)"
                ),
            ))


def check_stp_final_state(
    rec: "DiagnosticsRecorder",
    homeostasis: HomeostaticStats,
    issues: List[HealthIssue],
) -> None:
    """Check final absolute STP state for chronic depletion and facilitation ceiling.

    Inspects the end-of-run x·u efficacy and mean_u values from
    :attr:`HomeostaticStats.stp_final_state`:

    * **Chronic depletion** (``efficacy < 0.05``): The vesicle pool (x) is
      nearly empty and/or release probability (u) is near zero.  The synapse
      contributes essentially nothing; upstream spike trains are effectively
      silenced before reaching the postsynaptic population.  This can arise from
      excessively high baseline firing rates that continuously deplete the
      available-resource variable (Markram et al. 1998 *J Neurophysiol*).

    * **Facilitation ceiling** (``mean_u > 0.95``): The utilisation variable u
      is saturated near 1.0.  In this regime facilitation is exhausted and the
      synapse behaves as a constant-release device regardless of presynaptic
      activity, losing the dynamic-gain modulation intended by the STP model
      (Tsodyks & Markram 1997 *PNAS*).
    """
    syn_to_region = {str(syn_id): rn for rn, syn_id in rec._stp_keys}

    for syn_key, state in homeostasis.stp_final_state.items():
        rn       = syn_to_region.get(syn_key, "")
        efficacy = state.get("efficacy", float("nan"))
        mean_u   = state.get("mean_u",   float("nan"))

        if not np.isnan(efficacy) and efficacy < 0.05:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.HOMEOSTASIS,
                region=rn,
                message=(
                    f"STP chronically depleted: [{syn_key}]  "
                    f"x\u00b7u={efficacy:.3f} < 0.05  "
                    f"\u2014 synapse is effectively silent; baseline drive may be "
                    f"saturating the release machinery (Markram et al. 1998)"
                ),
            ))

        if not np.isnan(mean_u) and mean_u > 0.95:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.HOMEOSTASIS,
                region=rn,
                message=(
                    f"STP facilitation variable at ceiling: [{syn_key}]  "
                    f"mean_u={mean_u:.3f} > 0.95  "
                    f"\u2014 u is saturated; facilitation gain is exhausted; "
                    f"check U parameter or calcium dynamics (Tsodyks & Markram 1997)"
                ),
            ))
