"""Neuromodulator health checks — receptor levels, phasic dynamics, downstream effects."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

from .bio_ranges import nm_tonic_range
from .diagnostics_types import (
    HealthCategory,
    HealthIssue,
    OscillatoryStats,
    PopulationStats,
)
from .region_tags import CORTICAL_TAGS, DA_SOURCE_TAGS, PREFRONTAL_TAGS, STRIATAL_TAGS, matches_any

if TYPE_CHECKING:
    from .diagnostics_recorder import DiagnosticsRecorder


def check_neuromodulators(
    rec: "DiagnosticsRecorder",
    pop_stats: Dict[Tuple[str, str], PopulationStats],
    issues: List[HealthIssue],
) -> None:
    """Check neuromodulator receptor concentrations stuck at zero.

    Only fires if at least one neuromodulator source population was active —
    prevents false positives when source regions are not connected/activated.
    """
    n_nm_steps = rec._gain_sample_step
    if n_nm_steps > 0 and rec._n_nm_receptors > 0 and rec._nm_source_pop_keys:
        any_source_firing = any(
            pop_stats.get((rn, pn), None) is not None
            and pop_stats[(rn, pn)].mean_fr_hz > 0.5
            for (rn, pn) in rec._nm_source_pop_keys
        )
        if any_source_firing:
            for nm_idx, (rn, mod_name) in enumerate(rec._nm_receptor_keys):
                vals = rec._nm_concentration_history[:n_nm_steps, nm_idx]
                valid = vals[~np.isnan(vals)]
                if len(valid) > 0 and float(np.max(valid)) < 1e-6:
                    issues.append(HealthIssue(severity="warning", category=HealthCategory.NEUROMODULATORS, region=rn,
                        message=f"Neuromodulator receptor silent: {rn}/{mod_name} "
                                f"(peak concentration = {float(np.max(valid)):.2e} "
                                f"over {len(valid)} samples)"))


def check_neuromodulator_levels(
    rec: "DiagnosticsRecorder",
    issues: List[HealthIssue],
) -> None:
    """Check tonic neuromodulator concentrations are within physiological range.

    Complements :func:`check_neuromodulators` (which only flags completely silent
    receptors).  This check detects chronically elevated (near-saturation) or
    persistently very-low (but non-zero) tonic concentrations that indicate
    pathological release, reuptake failure, or disconnected source populations.

    Concentrations are the normalised ``[0, 1]`` receptor activation values from
    :class:`NeuromodulatorReceptor`.  Thresholds come from
    :func:`~.bio_ranges.nm_tonic_range`.

    References: Grace 1991 (DA); Descarries et al. 1997 (ACh); Mlinar et al.
    2016 (5-HT); Berridge 2008 (NE).
    """
    n_nm_steps = rec._gain_sample_step
    if n_nm_steps < 10 or rec._n_nm_receptors == 0:
        return
    for nm_idx, (rn, mod_name) in enumerate(rec._nm_receptor_keys):
        vals = rec._nm_concentration_history[:n_nm_steps, nm_idx]
        valid = vals[~np.isnan(vals)]
        if len(valid) < 5:
            continue
        mean_c = float(np.mean(valid))
        max_c = float(np.max(valid))
        warn_low, warn_high = nm_tonic_range(mod_name)
        # Chronically elevated — near saturation / receptor desensitisation.
        if mean_c > warn_high:
            severity = "critical" if mean_c > 0.90 else "warning"
            label = "SATURATED" if severity == "critical" else "Elevated"
            issues.append(HealthIssue(
                severity=severity,
                category=HealthCategory.NEUROMODULATORS,
                region=rn,
                message=(
                    f"{label} neuromodulator: {rn}/{mod_name}  "
                    f"mean={mean_c:.3f}  max={max_c:.3f}  "
                    f"(tonic warn threshold {warn_high:.2f}; "
                    f"receptor desensitisation expected above 0.9)"
                ),
            ))
        # Chronically near-zero but not completely silent.
        # (max < 1e-6 is handled by check_neuromodulators; this catches
        # max > 1e-6 but very low mean → intermittent or insufficient input.)
        elif max_c > 1e-6 and mean_c < warn_low:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.NEUROMODULATORS,
                region=rn,
                message=(
                    f"Low tonic neuromodulator: {rn}/{mod_name}  "
                    f"mean={mean_c:.4f}  max={max_c:.4f}  "
                    f"(expected mean ≥ {warn_low:.3f}; "
                    f"check source-population firing rate and connectivity)"
                ),
            ))


_NM_PHASIC_SPECS: List[Tuple[str, float, float, str]] = [
    # (mod_name_substring, mean_thresh, min_phasic_ratio, clinical_note)
    # DA: phasic burst should be > 3× tonic (Schultz 1998).  Flag when the
    # mean is already in the elevated range (≥ 0.50) yet no transients appear
    # (p90/p10 < 2.0) — the hallmark of tonic hyperdopaminergia / receptor
    # desensitisation.
    (
        "dopamine",
        0.50,
        2.0,
        "chronically elevated with no phasic bursts "
        "— receptor desensitisation territory; DA p90/p10 should be ≥ 2.0 "
        "(Schultz 1998; Grace 1991)",
    ),
    # NE: locus-coeruleus output should be transient; sustained elevation
    # shifts the inverted-U gain curve past the optimum, impairing cognitive
    # flexibility (Arnsten 2011).  Mean ≥ 0.30 with little variation (ratio
    # < 1.5) indicates tonically high rather than episodic LC bursting.
    (
        "norepinephrine",
        0.30,
        1.5,
        "tonically elevated — inverted-U gain curve predicts impaired cognitive "
        "flexibility at sustained high concentrations; bursts should be transient "
        "(Arnsten 2011; Berridge 2008)",
    ),
]


def check_neuromodulator_phasic(
    rec: "DiagnosticsRecorder",
    issues: List[HealthIssue],
) -> None:
    """Check that phasic neuromodulators are not chronically elevated without transients.

    Uses the 10th-/90th-percentile ratio of the concentration trajectory as a
    proxy for phasic variability::

        phasic_ratio = p90(c) / max(p10(c), 1e-6)

    A healthy phasic neuromodulator (e.g. DA on reward, NE on arousal) will
    spend most of its time near baseline (low p10) and exhibit sharp upward
    excursions (high p90), giving a ratio ≥ 2–3.  A chronically elevated,
    un-varying trajectory (ratio < threshold) suggests:

    - **Dopamine**: receptor desensitisation / tonic hyperdopaminergia (Grace
      1991; Schultz 1998).
    - **Norepinephrine**: locus-coeruleus in sustained high-gain mode — shifts
      the inverted-U past the optimum and impairs task-switching (Arnsten 2011;
      Berridge 2008).

    Checked against :data:`_NM_PHASIC_SPECS`.  Only fires when the mean
    concentration exceeds the module-specific threshold AND phasic_ratio falls
    below the expected minimum.

    Requires ≥ 50 valid samples for a meaningful percentile estimate.
    """
    n_nm_steps = rec._gain_sample_step
    if n_nm_steps < 50 or rec._n_nm_receptors == 0:
        return
    for nm_idx, (rn, mod_name) in enumerate(rec._nm_receptor_keys):
        vals = rec._nm_concentration_history[:n_nm_steps, nm_idx]
        valid = vals[~np.isnan(vals)]
        if len(valid) < 50:
            continue
        mean_c  = float(np.mean(valid))
        p10_c   = float(np.percentile(valid, 10))
        p90_c   = float(np.percentile(valid, 90))
        phasic_ratio = p90_c / max(p10_c, 1e-6)
        mn = mod_name.lower()
        for key, mean_thresh, ratio_thresh, note in _NM_PHASIC_SPECS:
            if key not in mn:
                continue
            if mean_c >= mean_thresh and phasic_ratio < ratio_thresh:
                issues.append(HealthIssue(
                    severity="warning",
                    category=HealthCategory.NEUROMODULATORS,
                    region=rn,
                    message=(
                        f"Neuromodulator tonic without phasic bursts: "
                        f"{rn}/{mod_name}  "
                        f"mean={mean_c:.3f}  "
                        f"phasic_ratio(p90/p10)={phasic_ratio:.2f} "
                        f"(threshold {ratio_thresh:.1f}×)  "
                        f"— {note}"
                    ),
                ))
            break  # only the first matching spec per modulator


def check_nm_downstream_effects(
    rec: "DiagnosticsRecorder",
    pop_stats: Dict[Tuple[str, str], PopulationStats],
    issues: List[HealthIssue],
) -> None:
    """Check that DA burst firing produces the expected D1/D2 MSN downstream effects.

    When any VTA or SNc DA population is actively burst-firing
    (``da_burst_events_per_s > 0``), the net dopaminergic effect on striatal MSNs
    should be:

    - **D1-MSN (direct pathway):** D1-receptor activation is excitatory →
      up-state facilitation → D1 FR ≥ D2 FR under tonic or phasic DA drive.
    - **D2-MSN (indirect pathway):** D2-receptor activation is inhibitory →
      down-state suppression → D2 FR ≤ D1 FR.

    Two checks are performed for each striatal region containing both D1 and D2
    MSN populations:

    1. **Static ratio**: D2 mean FR should not exceed D1 mean FR by more than 2×
       across the whole recording.  A D2/D1 FR ratio > 2 with active DA indicates
       the direct pathway is not being driven (missing D1-receptor activation,
       broken corticostriatal input, or DA→D1 pathway disconnect).

    2. **Temporal half-split**: When both populations were active in the first half
       (≥ 0.5 Hz), D1 should not strongly decline while D2 rises in the second
       half.  This catches the pattern of progressive D1 silencing concurrent with
       D2 disinhibition — the reverse of the expected DA effect.

    Requires ≥ 500 recorded steps for the half-split to be meaningful.

    References: Gerfen & Surmeier 2011; Schultz 1998; Frank 2005.
    """
    T = rec._n_recorded or rec.config.n_timesteps
    if T < 500:
        return

    # Only fire when DA source populations are actively burst-firing.
    da_burst_active = any(
        matches_any(rn, DA_SOURCE_TAGS)
        and "da" in pn.lower()
        and not np.isnan(ps.da_burst_events_per_s)
        and ps.da_burst_events_per_s > 0.0
        for (rn, pn), ps in pop_stats.items()
    )
    if not da_burst_active:
        return

    dt_s = rec.dt_ms / 1000.0
    half = T // 2
    spike_counts = rec._pop_spike_counts  # shape [T, P]

    for rn, region in rec.brain.regions.items():
        if not matches_any(rn, STRIATAL_TAGS):
            continue

        # Identify D1-MSN and D2-MSN population indices for this region.
        d1_indices = [
            rec._pop_index[(rn, pn)]
            for pn in region.neuron_populations.keys()
            if "d1" in pn.lower() and (rn, pn) in rec._pop_index
        ]
        d2_indices = [
            rec._pop_index[(rn, pn)]
            for pn in region.neuron_populations.keys()
            if "d2" in pn.lower() and (rn, pn) in rec._pop_index
        ]
        if not d1_indices or not d2_indices:
            continue

        # Whole-recording mean FRs from pop_stats.
        d1_fr_vals = [
            pop_stats[(rn, pn)].mean_fr_hz
            for pn in region.neuron_populations.keys()
            if "d1" in pn.lower() and (rn, pn) in pop_stats
            and not np.isnan(pop_stats[(rn, pn)].mean_fr_hz)
        ]
        d2_fr_vals = [
            pop_stats[(rn, pn)].mean_fr_hz
            for pn in region.neuron_populations.keys()
            if "d2" in pn.lower() and (rn, pn) in pop_stats
            and not np.isnan(pop_stats[(rn, pn)].mean_fr_hz)
        ]
        d1_fr = float(np.mean(d1_fr_vals)) if d1_fr_vals else 0.0
        d2_fr = float(np.mean(d2_fr_vals)) if d2_fr_vals else 0.0

        if d1_fr < 0.5 and d2_fr < 0.5:
            continue  # Both essentially silent — DA effect not testable.

        # Static check: D2 should not dominate D1 more than 2:1 under DA drive.
        if d2_fr > d1_fr * 2.0:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.NEUROMODULATORS,
                region=rn,
                message=(
                    f"DA active but D2-MSN dominates D1-MSN: {rn}  "
                    f"D1_FR={d1_fr:.1f} Hz  D2_FR={d2_fr:.1f} Hz  "
                    f"(with VTA/SNc burst firing, D1-receptor activation should "
                    f"facilitate the direct pathway above D2-MSN baseline; "
                    f"check corticostriatal D1 connectivity or DA→D1 modulation; "
                    f"Gerfen & Surmeier 2011)"
                ),
            ))

        # Temporal half-split check: D1 should not collapse while D2 rises.
        d1_sizes = np.array([rec._pop_sizes[i] for i in d1_indices], dtype=np.float64)
        d2_sizes = np.array([rec._pop_sizes[i] for i in d2_indices], dtype=np.float64)
        d1_n = float(d1_sizes.sum())
        d2_n = float(d2_sizes.sum())
        if d1_n == 0 or d2_n == 0:
            continue

        d1_first  = float(spike_counts[:half,  d1_indices].sum()) / (half       * dt_s * d1_n)
        d1_second = float(spike_counts[half:T, d1_indices].sum()) / ((T - half) * dt_s * d1_n)
        d2_first  = float(spike_counts[:half,  d2_indices].sum()) / (half       * dt_s * d2_n)
        d2_second = float(spike_counts[half:T, d2_indices].sum()) / ((T - half) * dt_s * d2_n)

        if d1_first >= 0.5 and d2_first >= 0.5:
            d1_ratio = d1_second / d1_first
            d2_ratio = d2_second / d2_first
            if d1_ratio < 0.6 and d2_ratio > 1.4:
                issues.append(HealthIssue(
                    severity="warning",
                    category=HealthCategory.NEUROMODULATORS,
                    region=rn,
                    message=(
                        f"DA downstream effect inverted over recording: {rn}  "
                        f"D1 2nd/1st={d1_ratio:.2f}  D2 2nd/1st={d2_ratio:.2f}  "
                        f"(D1 declining while D2 rising with persistent DA bursting — "
                        f"expected D1 facilitated and D2 suppressed; "
                        f"Gerfen & Surmeier 2011)"
                    ),
                ))


def check_nm_oscillation_gating(
    rec: "DiagnosticsRecorder",
    oscillations: OscillatoryStats,
    issues: List[HealthIssue],
) -> None:
    """Check that neuromodulator levels produce the expected oscillatory gating effects.

    Elevated ACh (nucleus-basalis output) should suppress slow delta/sigma
    oscillations and engage theta/gamma rhythms — the hallmark of the waking,
    attentive cortical state (Metherate et al. 1992).  Elevated DA in striatum
    or PFC should promote beta-band synchrony by facilitating tonic D1/D2
    receptor activation in the cortico-striato-thalamic loop (Seamans & Yang
    2004; Murthy & Fetz 1996).  When these cross-signatures are absent it
    indicates the NM pathways are driving receptor activation without producing
    their canonical downstream oscillatory effects.

    Checks performed per NM receptor instance:

    * **ACh + high cortical delta**: mean ACh > 0.35 AND delta fraction > 0.40
      in the same cortical region → warning: ACh should suppress slow rhythms.
    * **DA + absent beta**: mean DA > 0.30 AND beta fraction < 0.10 in the
      same striatal or prefrontal region → warning: DA should promote beta.

    Requires ≥ 10 NM samples and non-empty band power data for the region.

    References: Metherate et al. 1992 *Science*; Seamans & Yang 2004 *Nat Rev
    Neurosci*; Murthy & Fetz 1996 *J Neurophysiol*.
    """
    n_nm_steps = rec._gain_sample_step
    if n_nm_steps < 10 or rec._n_nm_receptors == 0:
        return

    for nm_idx, (rn, mod_name) in enumerate(rec._nm_receptor_keys):
        vals = rec._nm_concentration_history[:n_nm_steps, nm_idx]
        valid = vals[~np.isnan(vals)]
        if len(valid) < 10:
            continue
        mean_c = float(np.mean(valid))
        mn = mod_name.lower()

        band_power_map = oscillations.region_band_power.get(rn, {})
        if not band_power_map:
            continue

        # ACh: should suppress delta in cortex
        if ("acetylcholine" in mn or "ach" in mn) and matches_any(rn, CORTICAL_TAGS):
            if mean_c > 0.35:
                delta_frac = band_power_map.get("delta", float("nan"))
                if not np.isnan(delta_frac) and delta_frac > 0.40:
                    issues.append(HealthIssue(
                        severity="warning",
                        category=HealthCategory.NEUROMODULATORS,
                        region=rn,
                        message=(
                            f"ACh high but delta not suppressed: {rn}/{mod_name}  "
                            f"mean_ACh={mean_c:.3f} (> 0.35)  "
                            f"delta_frac={delta_frac:.2f} (> 0.40)  "
                            f"\u2014 elevated ACh (nucleus basalis \u2192 cortex) should "
                            f"reduce delta power and engage theta/gamma; "
                            f"check NB\u2192cortex ACh pathway (Metherate et al. 1992)"
                        ),
                    ))

        # DA: should promote beta in striatum / PFC
        if "dopamine" in mn and (
            matches_any(rn, STRIATAL_TAGS) or matches_any(rn, PREFRONTAL_TAGS)
        ):
            if mean_c > 0.30:
                beta_frac = band_power_map.get("beta", float("nan"))
                if not np.isnan(beta_frac) and beta_frac < 0.10:
                    issues.append(HealthIssue(
                        severity="warning",
                        category=HealthCategory.NEUROMODULATORS,
                        region=rn,
                        message=(
                            f"DA elevated but beta absent: {rn}/{mod_name}  "
                            f"mean_DA={mean_c:.3f} (> 0.30)  "
                            f"beta_frac={beta_frac:.2f} (< 0.10)  "
                            f"\u2014 DA should promote beta synchrony in striatum/PFC "
                            f"via D1/D2 activation in the cortico-striato-thalamic loop; "
                            f"check DA\u2192D1/D2 receptor connectivity "
                            f"(Seamans & Yang 2004; Murthy & Fetz 1996)"
                        ),
                    ))


def check_d1_d2_da_balance(
    rec: "DiagnosticsRecorder",
    pop_stats: Dict[Tuple[str, str], PopulationStats],
    issues: List[HealthIssue],
) -> None:
    """Check that striatal D1/D2 MSN activity balance reflects DA concentration.

    Dopamine modulates the two striatal output pathways in opposing directions:

    * **D1-MSN (direct, "go") pathway**: excited by DA via D1 receptors.
    * **D2-MSN (indirect, "no-go") pathway**: inhibited by DA via D2 receptors.

    When DA is chronically depleted (hypo-DA) the expected result is D2-MSN
    over-activity relative to D1-MSNs — the core circuit pathology of
    Parkinson's disease (Frank 2005; Gerfen & Surmeier 2011).  When DA is
    in the high-burst range (hyper-DA), D1-MSN over-dominance predicts
    excessively biased reward-driven action selection.

    Two complementary checks are performed per striatal region that contains
    both D1 and D2 MSN populations **and** at least one DA receptor instance:

    * **Hypo-DA state**: mean DA < lower tonic threshold AND D2_FR / D1_FR > 2.0
      → warning: D2-MSN dominance in hypodopaminergic state (Parkinson-like).
    * **Hyper-DA state**: mean DA > 0.60 AND D1_FR / D2_FR > 3.0
      → warning: D1-MSN dominance in hyperdopaminergic state.

    Requires ≥ 10 NM samples for a reliable DA mean estimate.  Skips any
    region where either D1 or D2 mean FR is below 0.5 Hz (insufficient data).

    References: Frank 2005 *Psychol Rev*; Gerfen & Surmeier 2011 *Annu Rev
    Neurosci*.
    """
    n_nm_steps = rec._gain_sample_step
    if n_nm_steps < 10 or rec._n_nm_receptors == 0:
        return

    da_lo, _da_hi = nm_tonic_range("dopamine")

    for rn in rec._region_keys:
        if not matches_any(rn, STRIATAL_TAGS):
            continue

        region = rec.brain.regions[rn] if rn in rec.brain.regions else None
        if region is None:
            continue

        # Collect D1 and D2 mean FRs for this region.
        d1_fr_vals = [
            pop_stats[(rn, pn)].mean_fr_hz
            for pn in region.neuron_populations.keys()
            if "d1" in pn.lower()
            and (rn, pn) in pop_stats
            and not np.isnan(pop_stats[(rn, pn)].mean_fr_hz)
        ]
        d2_fr_vals = [
            pop_stats[(rn, pn)].mean_fr_hz
            for pn in region.neuron_populations.keys()
            if "d2" in pn.lower()
            and (rn, pn) in pop_stats
            and not np.isnan(pop_stats[(rn, pn)].mean_fr_hz)
        ]
        if not d1_fr_vals or not d2_fr_vals:
            continue

        d1_fr = float(np.mean(d1_fr_vals))
        d2_fr = float(np.mean(d2_fr_vals))
        if d1_fr < 0.5 and d2_fr < 0.5:
            continue  # Both essentially silent — not enough data.

        # Find mean DA concentration for any DA receptor in this region.
        da_means: List[float] = []
        for nm_idx, (r, mod_name) in enumerate(rec._nm_receptor_keys):
            if r != rn:
                continue
            if "dopamine" not in mod_name.lower():
                continue
            vals = rec._nm_concentration_history[:n_nm_steps, nm_idx]
            valid = vals[~np.isnan(vals)]
            if len(valid) >= 10:
                da_means.append(float(np.mean(valid)))
        if not da_means:
            continue

        mean_da = float(np.mean(da_means))

        # Hypo-DA: D2-MSN dominance (Parkinson-like)
        if mean_da < da_lo and d1_fr > 0.5:
            ratio = d2_fr / d1_fr
            if ratio > 2.0:
                issues.append(HealthIssue(
                    severity="warning",
                    category=HealthCategory.NEUROMODULATORS,
                    region=rn,
                    message=(
                        f"Hypodopaminergic D2-MSN dominance: {rn}  "
                        f"mean_DA={mean_da:.4f} (< {da_lo:.3f} tonic threshold)  "
                        f"D2_FR/D1_FR={ratio:.2f} (> 2.0)  "
                        f"\u2014 D2-MSN over-activity in low-DA state; indirect pathway "
                        f"may be overactive (Parkinson-like; Frank 2005)"
                    ),
                ))

        # Hyper-DA: D1-MSN dominance
        if mean_da > 0.60 and d2_fr > 0.5:
            ratio = d1_fr / d2_fr
            if ratio > 3.0:
                issues.append(HealthIssue(
                    severity="warning",
                    category=HealthCategory.NEUROMODULATORS,
                    region=rn,
                    message=(
                        f"Hyperdopaminergic D1-MSN dominance: {rn}  "
                        f"mean_DA={mean_da:.3f} (> 0.60 burst range)  "
                        f"D1_FR/D2_FR={ratio:.2f} (> 3.0)  "
                        f"\u2014 D1 direct pathway dominates excessively under high DA; "
                        f"check reward-learning dynamics (Frank 2005)"
                    ),
                ))
