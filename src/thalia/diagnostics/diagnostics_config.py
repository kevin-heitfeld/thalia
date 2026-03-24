"""Diagnostics configuration — thresholds and recording parameters.

Pure data module: no simulation dependencies.

Health thresholds are organised into semantic sub-dataclasses so each domain
(firing, homeostasis, learning, …) can be found, validated, and overridden
independently.

Regional overrides
------------------
``HealthThresholds.for_region(region_name)`` returns a merged copy where
region-specific overrides shadow the defaults::

    th = HealthThresholds(
        regional_overrides={
            "cerebellum": {"firing": {"purkinje_cv_warn": 0.35}},
        },
    )
    cb_th = th.for_region("cerebellum")
    assert cb_th.firing.purkinje_cv_warn == 0.35
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict


# =============================================================================
# THRESHOLD SUB-DATACLASSES
# =============================================================================


@dataclass
class FiringThresholds:
    """Firing-rate, ISI, burst, synchrony, up/down state, and adaptation thresholds."""

    # ── Burst / epileptiform synchrony ────────────────────────────────────
    burst_coactivation_fraction: float = 0.30
    epileptiform_burst_threshold: float = 0.05
    pairwise_rho_critical: float = 0.40
    pairwise_rho_warn: float = 0.15

    # ── ISI irregularity ──────────────────────────────────────────────────
    isi_cv_regular_threshold: float = 0.35
    isi_cv2_low_threshold: float = 0.25
    isi_cv_burst: float = 2.5
    isi_cv2_high: float = 1.8

    # ── Fano factor ───────────────────────────────────────────────────────
    fano_factor_high: float = 3.0
    fano_factor_low: float = 0.3

    # ── DA burst mode detection ───────────────────────────────────────────
    da_burst_isi_cv: float = 1.0
    da_burst_isi_fraction: float = 0.20

    # ── FR range multipliers ──────────────────────────────────────────────
    fr_severely_low_multiplier: float = 0.2
    fr_hyperactive_multiplier: float = 5.0

    # ── Purkinje regularity ───────────────────────────────────────────────
    purkinje_cv_warn: float = 0.25
    purkinje_cv_high: float = 0.5

    # ── Spike-frequency adaptation ────────────────────────────────────────
    sfa_absent: float = 1.30
    sfa_runaway: float = 3.0
    sfa_unexpected: float = 2.0
    sfa_subphysiological: float = 0.8
    sfa_tau_fast_ms: float = 20.0
    sfa_tau_slow_ms: float = 1000.0

    # ── Spatial FR heterogeneity ──────────────────────────────────────────
    fr_heterogeneity_low: float = 0.10
    fr_heterogeneity_high: float = 4.0

    # ── Up/down state detection ───────────────────────────────────────────
    voltage_bimodality_threshold: float = 0.555
    up_state_max_ms: float = 700.0
    up_state_min_ms: float = 200.0
    up_down_ratio_max: float = 4.0

    def __post_init__(self) -> None:
        if not (0 < self.epileptiform_burst_threshold < 1.0):
            raise ValueError(
                f"epileptiform_burst_threshold must be in (0, 1), "
                f"got {self.epileptiform_burst_threshold}"
            )
        _validate_ordered_pairs(self, [
            ("pairwise_rho_warn", "pairwise_rho_critical"),
            ("purkinje_cv_warn", "purkinje_cv_high"),
            ("sfa_absent", "sfa_runaway"),
            ("sfa_tau_fast_ms", "sfa_tau_slow_ms"),
        ])


@dataclass
class HomeostasisThresholds:
    """Homeostatic gain, STP directionality, and firing-rate target thresholds."""

    # ── Homeostatic gain ──────────────────────────────────────────────────
    gain_collapse_threshold: float = 0.30
    gain_drift_pct: float = 15.0
    gain_slope_pct: float = 15.0
    stp_drift_pct: float = 50.0

    # ── FR target mismatch (homeostasis converged) ────────────────────────
    fr_target_ratio_high: float = 2.0
    fr_target_ratio_low: float = 0.5

    # ── STP directionality / final state ──────────────────────────────────
    stp_facilitating_tau_ratio: float = 0.5
    stp_facilitating_u_max: float = 0.25
    stp_depressing_tau_ratio: float = 2.0
    stp_depressing_u_min: float = 0.35
    stp_depletion: float = 0.05
    stp_ceiling: float = 0.95


@dataclass
class LearningThresholds:
    """Weight distribution, BCM, eligibility trace, and plasticity thresholds."""

    weight_sparsity_critical: float = 0.95
    weight_sparsity_warn: float = 0.80
    weight_saturation_cv: float = 0.01
    weight_homogenization_cv: float = 0.01
    weight_drift_warn: float = 0.5
    weight_drift_critical: float = 1.0
    dead_plasticity: float = 1e-8
    unstable_learning_warn: float = 0.1
    unstable_learning_critical: float = 0.5
    ltp_ltd_ratio_high: float = 10.0
    ltp_ltd_ratio_low: float = 0.1
    bcm_theta_collapsed: float = 1e-5
    bcm_theta_saturated: float = 0.45
    da_eligibility_min: float = 0.05
    homeostatic_correction_doubling: float = 2.0
    popvec_stability_min: float = 0.5
    stdp_timing_window_ms: float = 50.0
    stdp_timing_max_pairs: int = 200
    stdp_ltp_fraction_high: float = 0.90
    stdp_ltp_fraction_low: float = 0.10

    def __post_init__(self) -> None:
        _validate_ordered_pairs(self, [
            ("weight_sparsity_warn", "weight_sparsity_critical"),
            ("weight_drift_warn", "weight_drift_critical"),
            ("unstable_learning_warn", "unstable_learning_critical"),
        ])


@dataclass
class NeuromodulatorThresholds:
    """DA, ACh, and 5-HT modulation thresholds."""

    nm_saturation: float = 0.90
    nm_d2_d1_dominance_ratio: float = 2.0
    nm_d1_decline_ratio: float = 0.6
    nm_d2_rise_ratio: float = 1.4
    nm_ach_delta_gate: float = 0.35
    nm_ach_delta_fraction: float = 0.40
    nm_da_beta_gate: float = 0.30
    nm_da_beta_fraction: float = 0.10
    nm_hyper_da: float = 0.60
    nm_hyper_da_d1_d2_ratio: float = 3.0
    nm_serotonin_gate: float = 0.20
    nm_serotonin_da_gate: float = 0.50
    nm_ach_gamma_gate: float = 0.40
    nm_ach_gamma_fraction: float = 0.05


@dataclass
class OscillationThresholds:
    """Criticality, band power, theta/gamma, and hippocampal SWR thresholds."""

    # ── Criticality ───────────────────────────────────────────────────────
    branching_ratio_supercritical: float = 1.05
    branching_ratio_subcritical: float = 0.5
    d1_d2_coactivation: float = 0.3
    beta_burst_max_ms: float = 400.0
    gamma_hypercoherence: float = 0.85

    # ── Aperiodic (1/f) exponent ──────────────────────────────────────────
    # PSD(f) ∝ 1/f^χ  (He 2014; Donoghue et al. 2020).
    # Healthy cortex: χ ≈ 1.0–2.0.  Flattened (χ < 0.5) suggests epileptiform;
    # steep (χ > 3.0) suggests over-inhibition / disconnected activity.
    aperiodic_exponent_low: float = 0.5
    aperiodic_exponent_high: float = 3.0

    # ── Hippocampal theta PLV ─────────────────────────────────────────────
    plv_theta_warn: float = 0.10
    plv_theta_crit: float = 0.05
    plv_theta_high: float = 0.80

    # ── Hippocampal SWR / theta sequence ──────────────────────────────────
    hfo_power_swr: float = 0.02
    hfo_xcorr_min: float = 0.05
    theta_xcorr_min: float = 0.05
    ms_pacemaker_min_fr_hz: float = 2.0

    def __post_init__(self) -> None:
        _validate_ordered_pairs(self, [
            ("plv_theta_crit", "plv_theta_warn"),
        ])


@dataclass
class ConnectivityThresholds:
    """Delay accuracy, E/I balance, NMDA/GABA, and interneuron coverage thresholds."""

    # ── Connectivity / delay accuracy ─────────────────────────────────────
    delay_noise_floor_ms: float = 15.0
    delay_tolerance_fraction: float = 0.50
    jitter_high_ms: float = 10.0
    multihop_tolerance_fraction: float = 0.90
    multihop_min_tolerance_ms: float = 35.0

    # ── E/I balance ───────────────────────────────────────────────────────
    nmda_fraction_low: float = 0.05
    nmda_fraction_high: float = 0.70
    gaba_ba_ratio_critical: float = 3.0
    gaba_ba_ratio_warn: float = 1.0
    gaba_ba_ratio_low: float = 0.02

    # ── Interneuron coverage ──────────────────────────────────────────────
    interneuron_ratio_low: float = 0.15
    interneuron_ratio_high: float = 0.40
    pv_fraction_min: float = 0.20
    sst_fraction_min: float = 0.10
    vip_fraction_min: float = 0.05
    pv_pyramidal_fr_ratio: float = 2.0
    vip_sst_corr_max: float = 0.2

    def __post_init__(self) -> None:
        _validate_ordered_pairs(self, [
            ("gaba_ba_ratio_low", "gaba_ba_ratio_warn"),
            ("gaba_ba_ratio_warn", "gaba_ba_ratio_critical"),
            ("nmda_fraction_low", "nmda_fraction_high"),
        ])


@dataclass
class RegionalThresholds:
    """Region-specific thresholds (basal ganglia, cerebellum, thalamus, laminar, compartment)."""

    # ── Basal ganglia ─────────────────────────────────────────────────────
    gpe_min_fr_hz: float = 20.0
    stn_max_fr_hz: float = 60.0
    bg_output_min_fr_hz: float = 10.0

    # ── Cerebellar ────────────────────────────────────────────────────────
    purkinje_dcn_cofiring: float = 0.3
    purkinje_dcn_anticorr: float = -0.1
    io_synchrony_low: float = 0.1
    purkinje_active_fr_hz: float = 1.0

    # ── Thalamus ──────────────────────────────────────────────────────────
    trn_min_fr_hz: float = 30.0
    trn_relay_corr_max: float = 0.2
    relay_burst_isi_fraction: float = 0.05

    # ── Laminar ───────────────────────────────────────────────────────────
    l4_l23_delay_min_ms: float = 0.3
    l4_l23_delay_max_ms: float = 5.0
    l5_output_rate_ratio: float = 0.25

    # ── Two-compartment ───────────────────────────────────────────────────
    apical_silent: float = 0.02
    basal_min_drive: float = 0.01
    compartment_ratio_tolerance: float = 0.02


# =============================================================================
# VALIDATION HELPER
# =============================================================================


def _validate_ordered_pairs(
    obj: object,
    pairs: list[tuple[str, str]],
) -> None:
    """Raise :class:`ValueError` if any ``(lo, hi)`` pair violates ``lo < hi``."""
    for lo_name, hi_name in pairs:
        lo_val = getattr(obj, lo_name)
        hi_val = getattr(obj, hi_name)
        if lo_val >= hi_val:
            raise ValueError(
                f"{lo_name} ({lo_val}) must be < {hi_name} ({hi_val})"
            )


# =============================================================================
# HEALTH THRESHOLDS — composing all sub-groups
# =============================================================================


# Map each sub-dataclass to the registry of known sub-threshold types.
_THRESHOLD_GROUPS: Dict[str, type] = {
    "firing": FiringThresholds,
    "homeostasis": HomeostasisThresholds,
    "learning": LearningThresholds,
    "neuromodulators": NeuromodulatorThresholds,
    "oscillations": OscillationThresholds,
    "connectivity": ConnectivityThresholds,
    "regional": RegionalThresholds,
}


@dataclass
class HealthThresholds:
    """Biological threshold constants that govern all health checks.

    Thresholds are organised into semantic sub-groups accessible as attributes:

    * ``firing`` — :class:`FiringThresholds`
    * ``homeostasis`` — :class:`HomeostasisThresholds`
    * ``learning`` — :class:`LearningThresholds`
    * ``neuromodulators`` — :class:`NeuromodulatorThresholds`
    * ``oscillations`` — :class:`OscillationThresholds`
    * ``connectivity`` — :class:`ConnectivityThresholds`
    * ``regional`` — :class:`RegionalThresholds`

    Use :meth:`for_region` to obtain a merged copy with region-specific overrides.
    """

    firing: FiringThresholds = field(default_factory=FiringThresholds)
    homeostasis: HomeostasisThresholds = field(default_factory=HomeostasisThresholds)
    learning: LearningThresholds = field(default_factory=LearningThresholds)
    neuromodulators: NeuromodulatorThresholds = field(default_factory=NeuromodulatorThresholds)
    oscillations: OscillationThresholds = field(default_factory=OscillationThresholds)
    connectivity: ConnectivityThresholds = field(default_factory=ConnectivityThresholds)
    regional: RegionalThresholds = field(default_factory=RegionalThresholds)

    # ── Regional overrides ────────────────────────────────────────────────
    # Keys are case-insensitive substring patterns matched against region names.
    # Values are nested dicts: {"group_name": {"field_name": value}}.
    # Example:
    #   {"cerebellum": {"firing": {"purkinje_cv_warn": 0.35}}}
    regional_overrides: Dict[str, Dict[str, Dict[str, Any]]] = field(
        default_factory=dict,
    )

    def for_region(self, region: str) -> HealthThresholds:
        """Return thresholds with region-specific overrides applied.

        Matches *region* against all keys in :attr:`regional_overrides` using
        case-insensitive substring containment (longest match wins for each
        group/field pair).  Returns ``self`` unchanged if no overrides match.
        """
        if not self.regional_overrides:
            return self

        # Collect all matching overrides (longest pattern first for priority)
        merged: Dict[str, Dict[str, Any]] = {}
        region_lower = region.lower()
        for pattern in sorted(self.regional_overrides, key=len):
            if pattern.lower() in region_lower:
                for group_name, field_overrides in self.regional_overrides[pattern].items():
                    merged.setdefault(group_name, {}).update(field_overrides)

        if not merged:
            return self

        # Build a new HealthThresholds with overridden sub-groups
        result = copy.copy(self)
        for group_name, overrides in merged.items():
            if group_name not in _THRESHOLD_GROUPS:
                continue
            original_group = getattr(self, group_name)
            new_group = copy.copy(original_group)
            for field_name, value in overrides.items():
                if hasattr(new_group, field_name):
                    object.__setattr__(new_group, field_name, value)
            object.__setattr__(result, group_name, new_group)
        return result


# =============================================================================
# DIAGNOSTICS CONFIG
# =============================================================================


@dataclass
class DiagnosticsConfig:
    """Diagnostics Configuration.

    Args:
        n_timesteps: Number of timesteps to record.  Used to pre-allocate
            buffers.  Must be set before calling ``reset()`` / ``record()``.
        dt_ms: Simulation timestep in milliseconds.
        voltage_sample_size: Number of neurons to sample per population for
            voltage traces.
        conductance_sample_size: Number of neurons to sample for conductance
            traces.
        conductance_sample_interval_steps: How frequently (in timesteps) to snapshot
            conductances.
        gain_sample_interval_ms: How frequently (in ms) to snapshot homeostatic
            gains.  Applies in both modes.
        rate_bin_ms: Bin width (ms) for population firing-rate estimation and
            FFT analysis.
        coherence_n_pairs: Maximum number of region pairs to compute coherence
            for (avoids O(n²) scaling with many regions).
    """

    n_timesteps: int
    dt_ms: float = 1.0

    # Sampling resolution
    voltage_sample_size: int = 8
    conductance_sample_size: int = 8
    conductance_sample_interval_steps: int = 1
    gain_sample_interval_ms: int = 10

    # Analysis parameters
    rate_bin_ms: float = 10.0
    coherence_n_pairs: int = 30
    coherence_seed: int = 42

    # Streaming / windowed recording (bounded memory for long runs)
    window_size: int | None = None

    # Spike avalanche / criticality analysis
    compute_avalanches: bool = False
    avalanche_bin_ms: float = 4.0

    # SFA health-check suppression
    skip_sfa_health_check: bool = False
    sensory_pattern: str = ""
    thresholds: HealthThresholds = field(default_factory=HealthThresholds)

    # Transient detection thresholds (used by detect_transient_step)
    transient_gain_threshold_pct: float = 0.02
    transient_stp_threshold_pct: float = 0.05
    transient_rate_threshold_pct: float = 0.10
