"""Interneuron coverage health checks — ratio, subtype balance, FR balance."""

from __future__ import annotations

from typing import List

import numpy as np

from thalia.typing import PopulationPolarity

from ._helpers import bin_counts_2d
from .diagnostics_report import HealthCategory, HealthIssue
from .health_context import HealthCheckContext
from .region_tags import CORTICAL_TAGS, PV_TAGS, PYRAMIDAL_TAGS, SST_TAGS, VIP_TAGS, matches_any


def check_interneuron_ratio(ctx: HealthCheckContext) -> None:
    """Check inhibitory interneuron neuron-count ratio in neocortical regions.

    A healthy neocortex contains ~20–30 % GABAergic interneurons (Tremblay et
    al. 2016).  This check uses :attr:`~.NeuralRegion._population_polarities`
    (set at population-registration time per Dale's Law) to count
    ``INHIBITORY`` vs total neurons, then warns when the ratio falls outside
    ``[0.15, 0.40]``.

    Restricted to regions whose names contain ``"cortex"`` or
    ``"prefrontal"`` — the biological reference range does not apply to
    striatum, thalamus, cerebellum, or neuromodulatory nuclei.
    """
    rec, issues = ctx.rec, ctx.issues
    config = ctx.thresholds.connectivity
    for rn in rec._region_keys:
        if not matches_any(rn, CORTICAL_TAGS):
            continue
        n_inhibitory = 0
        n_total = 0
        for pop_idx in rec._region_pop_indices[rn]:
            _, pn = rec._pop_keys[pop_idx]
            n = int(rec._pop_sizes[pop_idx])
            n_total += n
            polarity = rec._pop_polarities.get((rn, pn), PopulationPolarity.ANY)
            if polarity == PopulationPolarity.INHIBITORY:
                n_inhibitory += n
        if n_total == 0:
            continue
        ratio = n_inhibitory / n_total
        if ratio < config.interneuron_ratio_low:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.INTERNEURON_COVERAGE,
                region=rn,
                message=(
                    f"Low interneuron ratio: {rn}  "
                    f"{n_inhibitory}/{n_total} neurons = {ratio:.0%} inhibitory  "
                    f"(expected 15–40 %; Tremblay et al. 2016) "
                    f"— region may lack sufficient GABAergic coverage"
                ),
            ))
        elif ratio > config.interneuron_ratio_high:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.INTERNEURON_COVERAGE,
                region=rn,
                message=(
                    f"High interneuron ratio: {rn}  "
                    f"{n_inhibitory}/{n_total} neurons = {ratio:.0%} inhibitory  "
                    f"(expected 15–40 %; Tremblay et al. 2016) "
                    f"— more inhibitory than excitatory neurons is atypical for neocortex"
                ),
            ))


def check_interneuron_subtype_balance(ctx: HealthCheckContext) -> None:
    """Check that cortical interneuron subtypes (PV, SST, VIP) are proportionally balanced.

    When a cortical region contains populations identified as at least two of
    the three cardinal GABAergic subtypes, verifies that their neuron-count
    fractions fall within the expected reference ranges:

    * PV  (fast-spiking basket / chandelier):   ≥20 % of inhibitory pool
    * SST/SOM (Martinotti / bitufted):           ≥10 % of inhibitory pool
    * VIP (disinhibitory; targets PV/SST):         ≥5 % of inhibitory pool

    Population name matching (case-insensitive):
    * PV  — substring ``"pv"`` or ``"parvalbumin"``
    * SST — substring ``"sst"`` or ``"somatostatin"`` or ``"som"``
    * VIP — substring ``"vip"``

    Only checked for cortical / prefrontal regions when ≥ 2 subtype labels are
    present.  Silently skips regions with a single named subtype or zero
    inhibitory neurons.
    """
    rec, issues = ctx.rec, ctx.issues
    config = ctx.thresholds.connectivity
    for rn in rec._region_keys:
        if not matches_any(rn, CORTICAL_TAGS):
            continue

        pv_n = sst_n = vip_n = other_in_n = 0
        for pop_idx in rec._region_pop_indices[rn]:
            _, pn = rec._pop_keys[pop_idx]
            polarity = rec._pop_polarities.get((rn, pn), PopulationPolarity.ANY)
            if polarity != PopulationPolarity.INHIBITORY:
                continue
            n = int(rec._pop_sizes[pop_idx])
            if matches_any(pn, PV_TAGS):
                pv_n += n
            elif matches_any(pn, SST_TAGS):
                sst_n += n
            elif matches_any(pn, VIP_TAGS):
                vip_n += n
            else:
                other_in_n += n

        n_in_total = pv_n + sst_n + vip_n + other_in_n
        if n_in_total == 0:
            continue

        # Need at least two named subtypes to make a proportionality claim.
        n_labelled_types = sum(1 for x in (pv_n, sst_n, vip_n) if x > 0)
        if n_labelled_types < 2:
            continue

        pv_frac  = pv_n  / n_in_total
        sst_frac = sst_n / n_in_total
        vip_frac = vip_n / n_in_total

        if pv_n > 0 and pv_frac < config.pv_fraction_min:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.INTERNEURON_COVERAGE,
                region=rn,
                message=(
                    f"Low PV interneuron fraction: {rn}  "
                    f"PV={pv_frac:.0%} of inhibitory pool "
                    f"(PV={pv_n}, SST={sst_n}, VIP={vip_n}, other={other_in_n}) "
                    f"— expected ≥20 %; perisomatic inhibition and gamma oscillations "
                    f"may be impaired (Tremblay et al. 2016)"
                ),
            ))
        if sst_n > 0 and sst_frac < config.sst_fraction_min:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.INTERNEURON_COVERAGE,
                region=rn,
                message=(
                    f"Low SST interneuron fraction: {rn}  "
                    f"SST={sst_frac:.0%} of inhibitory pool "
                    f"(PV={pv_n}, SST={sst_n}, VIP={vip_n}, other={other_in_n}) "
                    f"— expected ≥10 %; dendritic inhibition circuit may be "
                    f"under-represented (Tremblay et al. 2016)"
                ),
            ))
        if vip_n > 0 and vip_frac < config.vip_fraction_min:
            issues.append(HealthIssue(
                severity="info",
                category=HealthCategory.INTERNEURON_COVERAGE,
                region=rn,
                message=(
                    f"Low VIP interneuron fraction: {rn}  "
                    f"VIP={vip_frac:.0%} of inhibitory pool "
                    f"(PV={pv_n}, SST={sst_n}, VIP={vip_n}, other={other_in_n}) "
                    f"— expected ≥5 %; disinhibitory gain-control circuit "
                    f"may be under-represented (Jiang et al. 2015)"
                ),
            ))


def check_interneuron_fr_balance(ctx: HealthCheckContext) -> None:
    """Check PV/pyramidal firing-rate ratio and VIP–SST anticorrelation.

    Two firing-rate-based interneuron circuit checks for cortical regions:

    **PV:pyramidal FR ratio**
    PV (fast-spiking basket) cells must fire substantially faster than the
    excitatory pyramidal cells they inhibit.  A neuron-weighted mean PV FR
    below 2× the mean pyramidal FR indicates that the inhibitory brake is
    under-recruited relative to its target population, which can allow
    recurrent excitation to run away.
    Threshold: PV mean FR < 2 × excitatory mean FR → warning.
    Reference: DeWeese & Zador 2006, *J Neurosci*.

    **VIP–SST anticorrelation**
    VIP interneurons disinhibit pyramidal cells by inhibiting SST (and PV)
    interneurons.  A healthy disinhibitory circuit produces a negative
    correlation between the binned VIP population rate and the binned SST
    population rate.  Positive correlation indicates that VIP is either not
    driving SST suppression or that both subtypes are coactivated by shared
    drive, which would short-circuit disinhibition.  Gate: ≥ 50 rate bins
    required; Pearson r > config.vip_sst_corr_max triggers a warning.
    """
    rec, pop_stats, issues = ctx.rec, ctx.pop_stats, ctx.issues
    config = ctx.thresholds.connectivity
    bin_steps = max(1, int(rec.config.rate_bin_ms / rec.dt_ms))
    T = rec._n_recorded or rec.config.n_timesteps
    n_bins = T // bin_steps

    for rn in rec._region_keys:
        if not matches_any(rn, CORTICAL_TAGS):
            continue

        # ------------------------------------------------------------------ #
        # 1. PV FR / pyramidal FR ratio
        # ------------------------------------------------------------------ #
        pv_fr_sum = pv_fr_n = 0.0
        pyr_fr_sum = pyr_fr_n = 0.0
        for pop_idx in rec._region_pop_indices[rn]:
            _, pn = rec._pop_keys[pop_idx]
            ps = pop_stats.get((rn, pn))
            if ps is None:
                continue
            n = float(rec._pop_sizes[pop_idx])
            if matches_any(pn, PV_TAGS):
                pv_fr_sum += ps.mean_fr_hz * n
                pv_fr_n   += n
            elif matches_any(pn, PYRAMIDAL_TAGS):
                pyr_fr_sum += ps.mean_fr_hz * n
                pyr_fr_n   += n

        if pv_fr_n > 0 and pyr_fr_n > 0:
            pv_mean  = pv_fr_sum  / pv_fr_n
            pyr_mean = pyr_fr_sum / pyr_fr_n
            if pyr_mean > 0.1 and pv_mean < config.pv_pyramidal_fr_ratio * pyr_mean:
                issues.append(HealthIssue(
                    severity="warning",
                    category=HealthCategory.INTERNEURON_COVERAGE,
                    region=rn,
                    message=(
                        f"Low PV/pyramidal FR ratio: {rn}  "
                        f"PV={pv_mean:.1f} Hz, pyramidal={pyr_mean:.1f} Hz "
                        f"(ratio={pv_mean/pyr_mean:.2f}, expected ≥2.0) "
                        f"— PV inhibitory drive under-recruited relative to pyramidal "
                        f"target; perisomatic inhibition may be insufficient "
                        f"(DeWeese & Zador 2006)"
                    ),
                ))

        # ------------------------------------------------------------------ #
        # 2. VIP–SST anticorrelation
        # ------------------------------------------------------------------ #
        if n_bins < 50:
            continue

        # Collect population indices for VIP and SST in this region.
        vip_indices: List[int] = []
        sst_indices: List[int] = []
        vip_sizes: List[float] = []
        sst_sizes: List[float] = []
        for pop_idx in rec._region_pop_indices[rn]:
            _, pn = rec._pop_keys[pop_idx]
            idx = pop_idx
            n = float(rec._pop_sizes[pop_idx])
            if matches_any(pn, VIP_TAGS):
                vip_indices.append(idx)
                vip_sizes.append(n)
            elif matches_any(pn, SST_TAGS):
                sst_indices.append(idx)
                sst_sizes.append(n)

        if not vip_indices or not sst_indices:
            continue

        # Neuron-weighted binned rates for each subtype.
        spike_counts = rec._pop_spike_counts[:n_bins * bin_steps]
        binned = bin_counts_2d(spike_counts, n_bins, bin_steps)

        vip_sizes_arr = np.array(vip_sizes)
        sst_sizes_arr = np.array(sst_sizes)
        vip_trace = (binned[:, vip_indices] * vip_sizes_arr).sum(axis=1) / vip_sizes_arr.sum()
        sst_trace = (binned[:, sst_indices] * sst_sizes_arr).sum(axis=1) / sst_sizes_arr.sum()

        # Only test if both populations have meaningful activity.
        if vip_trace.std() < 1e-9 or sst_trace.std() < 1e-9:
            continue

        r = float(np.corrcoef(vip_trace, sst_trace)[0, 1])
        if r > config.vip_sst_corr_max:
            issues.append(HealthIssue(
                severity="warning",
                category=HealthCategory.INTERNEURON_COVERAGE,
                region=rn,
                message=(
                    f"VIP–SST positive correlation: {rn}  "
                    f"r={r:.2f} (expected r < 0) "
                    f"— VIP should suppress SST via disinhibitory circuit; "
                    f"positive correlation indicates VIP is not driving SST "
                    f"suppression or shared excitatory drive dominates "
                    f"(Pi et al. 2013)"
                ),
            ))
