"""Tests for health-check functions — firing, E/I balance, connectivity."""

from __future__ import annotations

import numpy as np

from thalia.typing import ReceptorType, SynapseId
from thalia.diagnostics.diagnostics_config import (
    DiagnosticsConfig,
    HealthThresholds,
)
from thalia.diagnostics.diagnostics_metrics import (
    ConnectivityStats,
    HomeostaticStats,
    OscillatoryStats,
    PopulationStats,
    RegionStats,
)
from thalia.diagnostics.diagnostics_report import HealthCategory
from thalia.diagnostics.diagnostics_snapshot import RecorderSnapshot
from thalia.diagnostics.health_context import HealthCheckContext
from thalia.diagnostics.health_firing import check_population_firing
from thalia.diagnostics.health_ei_balance import check_ei_balance
from thalia.diagnostics.health_connectivity import check_connectivity


# ---------------------------------------------------------------------------
# Fixture factories
# ---------------------------------------------------------------------------


def _make_pop_stats(**overrides) -> PopulationStats:
    """Build a PopulationStats with sane defaults; override any field."""
    defaults = dict(
        region_name="cortex",
        population_name="pyr",
        n_neurons=100,
        mean_fr_hz=10.0,
        std_fr_hz=3.0,
        fraction_silent=0.0,
        fraction_hyperactive=0.0,
        hyperactive_threshold_hz=100.0,
        total_spikes=5000,
        isi_mean_ms=100.0,
        isi_cv=1.0,
        isi_cv_corrected=1.0,
        fraction_bursting=0.05,
        fraction_refractory_violations=0.0,
        isi_cv2=1.0,
        isi_cv2_population=1.0,
        fraction_isi_lt_80ms=0.1,
        da_burst_events_per_s=0.0,
        sfa_index=1.0,
        per_neuron_ff=1.0,
        pairwise_correlation=0.05,
        fraction_burst_events=0.01,
        fr_histogram=np.zeros(20, dtype=np.float32),
        fr_histogram_edges=np.zeros(21, dtype=np.float32),
        bio_range_hz=(5.0, 25.0),
    )
    defaults.update(overrides)
    return PopulationStats(**defaults)


def _make_region_stats(
    rn: str = "cortex",
    populations: dict | None = None,
    **overrides,
) -> RegionStats:
    """Build a RegionStats with defaults."""
    ps = _make_pop_stats(region_name=rn)
    defaults = dict(
        region_name=rn,
        populations=populations or {"pyr": ps},
        mean_fr_hz=10.0,
        total_spikes=5000,
        mean_g_exc=0.5,
        mean_g_nmda=0.3,
        mean_g_inh=0.4,
        mean_g_gaba_a=0.3,
        mean_g_gaba_b=0.1,
    )
    defaults.update(overrides)
    return RegionStats(**defaults)


def _make_oscillatory_stats() -> OscillatoryStats:
    return OscillatoryStats(
        region_band_power={},
        region_band_power_absolute={},
        region_dominant_freq={},
        region_dominant_band={},
        coherence_theta=np.zeros((0, 0)),
        coherence_beta=np.zeros((0, 0)),
        coherence_gamma=np.zeros((0, 0)),
        region_order=[],
        global_dominant_freq_hz=0.0,
        global_band_power={},
    )


def _make_connectivity_stats(
    tracts: list | None = None,
    n_broken: list | None = None,
) -> ConnectivityStats:
    return ConnectivityStats(
        tracts=tracts or [],
        n_functional=len(tracts or []),
        n_broken=n_broken or [],
    )


def _make_homeostatic_stats() -> HomeostaticStats:
    return HomeostaticStats(
        gain_trajectories={},
        gain_sample_times_ms=np.array([]),
        stp_efficacy_history={},
        stp_final_state={},
    )


def _make_minimal_snapshot(
    n_timesteps: int = 100,
    thresholds: HealthThresholds | None = None,
    skip_sfa: bool = False,
) -> RecorderSnapshot:
    """Minimal RecorderSnapshot just to satisfy HealthCheckContext."""
    config = DiagnosticsConfig(
        n_timesteps=n_timesteps,
        thresholds=thresholds or HealthThresholds(),
        skip_sfa_health_check=skip_sfa,
    )
    return RecorderSnapshot(
        config=config,
        dt_ms=1.0,
        _pop_keys=[],
        _pop_index={},
        _n_pops=0,
        _pop_sizes=np.array([], dtype=np.int32),
        _region_keys=[],
        _region_index={},
        _n_regions=0,
        _region_pop_indices={},
        _tract_keys=[],
        _tract_index={},
        _n_tracts=0,
        _stp_keys=[],
        _nm_receptor_keys=[],
        _n_nm_receptors=0,
        _nm_source_pop_keys=[],
        _v_sample_idx=[],
        _c_sample_idx=[],
        _n_recorded=n_timesteps,
        _gain_sample_step=0,
        _cond_sample_step=0,
        _gain_sample_times=[],
        _pop_spike_counts=np.zeros((n_timesteps, 0), dtype=np.int32),
        _per_neuron_spike_counts=[],
        _region_spike_counts=np.zeros((n_timesteps, 0), dtype=np.int32),
        _tract_sent=np.zeros((n_timesteps, 0), dtype=np.int32),
        _spike_times={},
        _voltages=None,
        _g_exc_samples=None,
        _g_inh_samples=None,
        _g_nmda_samples=None,
        _g_gaba_b_samples=None,
        _g_apical_samples=None,
        _g_L_scale_history=np.zeros((0, 0), dtype=np.float32),
        _stp_efficacy_history=np.zeros((0, 0), dtype=np.float32),
        _nm_concentration_history=np.zeros((0, 0), dtype=np.float32),
        _pop_polarities={},
        _tract_delay_ms=[],
        _homeostasis_target_hz={},
        _stp_configs=[],
        _stp_final_state={},
    )


def _make_ctx(
    pop_stats: dict | None = None,
    region_stats: dict | None = None,
    connectivity: ConnectivityStats | None = None,
    rec: RecorderSnapshot | None = None,
    **kwargs,
) -> HealthCheckContext:
    """Build a HealthCheckContext with defaults for unspecified fields."""
    return HealthCheckContext(
        rec=rec or _make_minimal_snapshot(**{k: v for k, v in kwargs.items() if k in ("thresholds", "skip_sfa")}),
        pop_stats=pop_stats or {},
        region_stats=region_stats or {},
        connectivity=connectivity or _make_connectivity_stats(),
        oscillations=_make_oscillatory_stats(),
        homeostasis=_make_homeostatic_stats(),
        learning=None,
    )


# =====================================================================
# Firing health checks
# =====================================================================


class TestCheckPopulationFiring:

    def test_healthy_population_no_issues(self) -> None:
        """A population within bio range should not trigger critical or warning issues."""
        ps = _make_pop_stats(mean_fr_hz=12.0, bio_range_hz=(5.0, 25.0))
        ctx = _make_ctx(pop_stats={("cortex", "pyr"): ps})
        check_population_firing(ctx)
        firing_issues = [i for i in ctx.issues if i.category == HealthCategory.FIRING
                         and i.severity in ("critical", "warning")]
        assert len(firing_issues) == 0

    def test_silent_population_critical(self) -> None:
        """Zero spikes + expected FR > 0 → critical SILENT issue."""
        ps = _make_pop_stats(
            mean_fr_hz=0.0,
            total_spikes=0,
            fraction_silent=1.0,
            bio_range_hz=(5.0, 25.0),
            isi_cv=float("nan"),
            isi_cv2=float("nan"),
            per_neuron_ff=float("nan"),
        )
        ctx = _make_ctx(pop_stats={("cortex", "pyr"): ps})
        check_population_firing(ctx)
        critical = [i for i in ctx.issues if i.severity == "critical"]
        assert any("SILENT" in i.message for i in critical)

    def test_hyperactive_population_critical(self) -> None:
        """FR far above bio range → critical HYPERACTIVE issue."""
        ps = _make_pop_stats(
            mean_fr_hz=300.0,
            bio_range_hz=(5.0, 25.0),
            fraction_hyperactive=0.5,
        )
        ctx = _make_ctx(pop_stats={("cortex", "pyr"): ps})
        check_population_firing(ctx)
        critical = [i for i in ctx.issues if i.severity == "critical"]
        assert any("HYPERACTIVE" in i.message for i in critical)

    def test_low_fr_warning(self) -> None:
        """FR below bio range but not severely → warning."""
        ps = _make_pop_stats(
            mean_fr_hz=2.0,
            bio_range_hz=(5.0, 25.0),
        )
        ctx = _make_ctx(pop_stats={("cortex", "pyr"): ps})
        check_population_firing(ctx)
        warnings = [i for i in ctx.issues if i.severity == "warning"]
        assert any("Low FR" in i.message for i in warnings)

    def test_refractory_violations_critical(self) -> None:
        """Nonzero fraction_refractory_violations → critical issue."""
        ps = _make_pop_stats(fraction_refractory_violations=0.05)
        ctx = _make_ctx(pop_stats={("cortex", "pyr"): ps})
        check_population_firing(ctx)
        critical = [i for i in ctx.issues if i.severity == "critical"]
        assert any("REFRACTORY" in i.message for i in critical)

    def test_high_fano_factor_warning(self) -> None:
        """FF > threshold → warning about epileptiform variability."""
        ps = _make_pop_stats(per_neuron_ff=6.0)
        ctx = _make_ctx(pop_stats={("cortex", "pyr"): ps})
        check_population_firing(ctx)
        warnings = [i for i in ctx.issues if i.severity == "warning"]
        assert any("Fano Factor" in i.message for i in warnings)

    def test_population_status_populated(self) -> None:
        """check_population_firing should populate population_status dict."""
        ps = _make_pop_stats(mean_fr_hz=10.0, bio_range_hz=(5.0, 25.0))
        ctx = _make_ctx(pop_stats={("cortex", "pyr"): ps})
        check_population_firing(ctx)
        assert "cortex:pyr" in ctx.population_status
        assert ctx.population_status["cortex:pyr"] == "ok"


# =====================================================================
# E/I balance health checks
# =====================================================================


class TestCheckEiBalance:

    def test_balanced_ei_no_issues(self) -> None:
        """Balanced E/I ratio → no critical or warning issues."""
        rs = _make_region_stats(
            mean_g_exc=0.5,
            mean_g_nmda=0.3,
            mean_g_inh=0.6,
            mean_g_gaba_a=0.5,
            mean_g_gaba_b=0.1,
        )
        ctx = _make_ctx(region_stats={"cortex": rs})
        check_ei_balance(ctx)
        ei_issues = [i for i in ctx.issues
                     if i.category == HealthCategory.EI_BALANCE
                     and i.severity in ("critical", "warning")]
        assert len(ei_issues) == 0

    def test_nan_conductances_no_crash(self) -> None:
        """NaN conductances should not crash the checker."""
        rs = _make_region_stats(
            mean_g_exc=float("nan"),
            mean_g_nmda=float("nan"),
            mean_g_inh=float("nan"),
            mean_g_gaba_a=float("nan"),
            mean_g_gaba_b=float("nan"),
        )
        ctx = _make_ctx(region_stats={"cortex": rs})
        check_ei_balance(ctx)
        # No crash is the test

    def test_gaba_b_dominant_critical(self) -> None:
        """GABA-B >> GABA-A → critical issue."""
        rs = _make_region_stats(
            mean_g_gaba_a=0.1,
            mean_g_gaba_b=5.0,
        )
        ctx = _make_ctx(region_stats={"cortex": rs})
        check_ei_balance(ctx)
        critical = [i for i in ctx.issues if i.severity == "critical"]
        assert any("GABA-B" in i.message for i in critical)


# =====================================================================
# Connectivity health checks
# =====================================================================


class TestCheckConnectivity:

    def test_no_tracts_no_issues(self) -> None:
        """Empty tract list → no issues."""
        ctx = _make_ctx(connectivity=_make_connectivity_stats())
        check_connectivity(ctx)
        assert len(ctx.issues) == 0

    def test_broken_tract_critical(self) -> None:
        """A broken tract should produce a critical issue."""
        sid = SynapseId("cortex", "pyr", "striatum", "msn_d1", ReceptorType.AMPA)
        broken = ConnectivityStats.TractStats(
            synapse_id=sid,
            spikes_sent=1000,
            transmission_ratio=0.0,
            is_functional=False,
            measured_delay_ms=float("nan"),
            expected_delay_ms=5.0,
            anticausal_peak_ms=float("nan"),
        )
        conn = _make_connectivity_stats(n_broken=[broken])
        ctx = _make_ctx(connectivity=conn)
        check_connectivity(ctx)
        critical = [i for i in ctx.issues if i.severity == "critical"]
        assert any("BROKEN TRACT" in i.message for i in critical)

    def test_delay_mismatch_warning(self) -> None:
        """Large measured vs expected delay → warning."""
        sid = SynapseId("cortex", "pyr", "thalamus", "relay", ReceptorType.AMPA)
        ts = ConnectivityStats.TractStats(
            synapse_id=sid,
            spikes_sent=1000,
            transmission_ratio=0.95,
            is_functional=True,
            measured_delay_ms=50.0,
            expected_delay_ms=5.0,
            anticausal_peak_ms=float("nan"),
        )
        conn = _make_connectivity_stats(tracts=[ts])
        ctx = _make_ctx(connectivity=conn)
        check_connectivity(ctx)
        warnings = [i for i in ctx.issues if i.severity == "warning"]
        assert any("Delay mismatch" in i.message for i in warnings)

    def test_accurate_delay_no_issue(self) -> None:
        """Measured delay within tolerance → no warning."""
        sid = SynapseId("cortex", "pyr", "thalamus", "relay", ReceptorType.AMPA)
        ts = ConnectivityStats.TractStats(
            synapse_id=sid,
            spikes_sent=1000,
            transmission_ratio=0.95,
            is_functional=True,
            measured_delay_ms=6.0,
            expected_delay_ms=5.0,
            anticausal_peak_ms=float("nan"),
        )
        conn = _make_connectivity_stats(tracts=[ts])
        ctx = _make_ctx(connectivity=conn)
        check_connectivity(ctx)
        warnings = [i for i in ctx.issues if i.severity == "warning"
                    and "Delay mismatch" in i.message]
        assert len(warnings) == 0
