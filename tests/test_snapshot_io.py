"""Roundtrip tests for snapshot save/load (diagnostics_snapshot_io.py)."""

from __future__ import annotations

import os

import numpy as np

from thalia.typing import ReceptorType, SynapseId
from thalia.diagnostics.diagnostics_config import DiagnosticsConfig, HealthThresholds
from thalia.diagnostics.diagnostics_snapshot import RecorderSnapshot
from thalia.diagnostics.diagnostics_snapshot_io import (
    _reconstruct_thresholds,
    load_snapshot,
    save_snapshot,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_synapse_id(src_r: str = "cortex", src_p: str = "pyr",
                     tgt_r: str = "striatum", tgt_p: str = "msn_d1",
                     receptor: ReceptorType = ReceptorType.AMPA) -> SynapseId:
    return SynapseId(src_r, src_p, tgt_r, tgt_p, receptor)


def _make_minimal_snapshot(
    *,
    n_pops: int = 2,
    n_regions: int = 1,
    T: int = 100,
    n_neurons_per_pop: int = 20,
    n_tracts: int = 1,
    n_stp: int = 1,
    n_nm: int = 1,
    V_sample: int = 4,
    C_sample: int = 4,
    gain_steps: int = 5,
    cond_steps: int = 10,
    include_voltages: bool = True,
    include_conductances: bool = True,
    include_learning: bool = False,
) -> RecorderSnapshot:
    """Build a synthetic RecorderSnapshot with controlled data."""
    rng = np.random.default_rng(42)
    config = DiagnosticsConfig(
        n_timesteps=T,
        dt_ms=1.0,
        voltage_sample_size=V_sample,
        conductance_sample_size=C_sample,
        thresholds=HealthThresholds(),
    )

    pop_keys = [("region_A", f"pop_{i}") for i in range(n_pops)]
    pop_index = {k: i for i, k in enumerate(pop_keys)}
    pop_sizes = np.array([n_neurons_per_pop] * n_pops, dtype=np.int32)

    region_keys = [f"region_{chr(65 + i)}" for i in range(n_regions)]
    region_index = {r: i for i, r in enumerate(region_keys)}
    region_pop_indices = {"region_A": list(range(n_pops))}

    tract_sid = _make_synapse_id()
    tract_keys = [tract_sid] * n_tracts
    tract_index = {tract_sid: 0}

    stp_keys = [("region_A", tract_sid)] * n_stp
    nm_receptor_keys = [("region_A", "dopamine")] * n_nm
    nm_source_pop_keys = [("region_A", "pop_0")]

    v_sample_idx = [rng.choice(n_neurons_per_pop, size=V_sample, replace=False).astype(np.int64)
                    for _ in range(n_pops)]
    c_sample_idx = [rng.choice(n_neurons_per_pop, size=C_sample, replace=False).astype(np.int64)
                    for _ in range(n_pops)]

    pop_spike_counts = rng.integers(0, 5, size=(T, n_pops), dtype=np.int32)
    per_neuron_spike_counts = [
        rng.integers(0, 100, size=n_neurons_per_pop, dtype=np.int32) for _ in range(n_pops)
    ]
    region_spike_counts = rng.integers(0, 10, size=(T, n_regions), dtype=np.int32)
    tract_sent = rng.integers(0, 3, size=(T, n_tracts), dtype=np.int32)

    # Spike times
    spike_times = {}
    for key in pop_keys:
        n = n_neurons_per_pop
        nested = [sorted(rng.choice(T, size=rng.integers(1, 10), replace=False).tolist()) for _ in range(n)]
        spike_times[key] = nested

    voltages = rng.standard_normal((T, n_pops, V_sample)).astype(np.float32) if include_voltages else None
    g_exc = rng.random((cond_steps, n_pops, C_sample)).astype(np.float32) if include_conductances else None
    g_inh = rng.random((cond_steps, n_pops, C_sample)).astype(np.float32) if include_conductances else None
    g_nmda = rng.random((cond_steps, n_pops, C_sample)).astype(np.float32) if include_conductances else None
    g_gaba_b = rng.random((cond_steps, n_pops, C_sample)).astype(np.float32) if include_conductances else None

    g_L_scale = rng.random((gain_steps, n_pops)).astype(np.float32)
    stp_efficacy = rng.random((gain_steps, n_stp)).astype(np.float32)
    nm_conc = rng.random((gain_steps, n_nm)).astype(np.float32)

    pop_polarities = {k: "excitatory" for k in pop_keys}
    tract_delay_ms = [5.0] * n_tracts
    homeostasis_target_hz = {k: 10.0 for k in pop_keys}
    stp_configs = [(0.5, 200.0, 50.0)] * n_stp
    stp_final_state = {"tract_0": {"u": 0.3, "x": 0.9}}
    tract_weight_stats = {tract_sid.to_key(): {"mean": 0.1, "std": 0.05}}
    pop_neuron_params = {k: {"tau_mem": 20.0, "v_thresh": -50.0} for k in pop_keys}

    learning_keys: list[tuple[str, str]] = []
    weight_dist = None
    weight_update = None
    elig_mean = None
    elig_ratio = None
    bcm_theta = None
    bcm_keys: list[int] = []
    homeo_corr = None
    popvec_snaps = None
    popvec_times: list[int] = []

    if include_learning:
        learning_keys = [("group_0", "stdp")]
        n_learn = 1
        weight_dist = rng.random((gain_steps, n_learn, 5)).astype(np.float32)
        weight_update = rng.random((gain_steps, n_learn)).astype(np.float32)
        elig_mean = rng.random((gain_steps, n_learn)).astype(np.float32)
        elig_ratio = rng.random((gain_steps, n_learn)).astype(np.float32)
        bcm_theta = rng.random((gain_steps, 1)).astype(np.float32)
        bcm_keys = [0]
        homeo_corr = rng.random((gain_steps, n_pops)).astype(np.float32)

    return RecorderSnapshot(
        config=config,
        dt_ms=1.0,
        _pop_keys=pop_keys,
        _pop_index=pop_index,
        _n_pops=n_pops,
        _pop_sizes=pop_sizes,
        _region_keys=region_keys,
        _region_index=region_index,
        _n_regions=n_regions,
        _region_pop_indices=region_pop_indices,
        _tract_keys=tract_keys,
        _tract_index=tract_index,
        _n_tracts=n_tracts,
        _stp_keys=stp_keys,
        _nm_receptor_keys=nm_receptor_keys,
        _n_nm_receptors=n_nm,
        _nm_source_pop_keys=nm_source_pop_keys,
        _v_sample_idx=v_sample_idx,
        _c_sample_idx=c_sample_idx,
        _n_recorded=T,
        _gain_sample_step=gain_steps,
        _cond_sample_step=cond_steps,
        _gain_sample_times=list(range(0, gain_steps * 10, 10)),
        _pop_spike_counts=pop_spike_counts,
        _per_neuron_spike_counts=per_neuron_spike_counts,
        _region_spike_counts=region_spike_counts,
        _tract_sent=tract_sent,
        _spike_times=spike_times,
        _voltages=voltages,
        _g_exc_samples=g_exc,
        _g_inh_samples=g_inh,
        _g_nmda_samples=g_nmda,
        _g_gaba_b_samples=g_gaba_b,
        _g_apical_samples=None,
        _g_L_scale_history=g_L_scale,
        _stp_efficacy_history=stp_efficacy,
        _nm_concentration_history=nm_conc,
        _pop_polarities=pop_polarities,
        _tract_delay_ms=tract_delay_ms,
        _homeostasis_target_hz=homeostasis_target_hz,
        _stp_configs=stp_configs,
        _stp_final_state=stp_final_state,
        _tract_weight_stats=tract_weight_stats,
        _pop_neuron_params=pop_neuron_params,
        _learning_keys=learning_keys,
        _weight_dist_history=weight_dist,
        _weight_update_magnitude_history=weight_update,
        _eligibility_mean_history=elig_mean,
        _eligibility_ltp_ltd_ratio_history=elig_ratio,
        _bcm_theta_history=bcm_theta,
        _bcm_keys=bcm_keys,
        _homeostatic_correction_rate=homeo_corr,
        _popvec_snapshots=popvec_snaps,
        _popvec_snapshot_times=popvec_times,
    )


# =====================================================================
# Roundtrip tests
# =====================================================================


class TestSnapshotRoundtrip:
    """save_snapshot → load_snapshot preserves all fields."""

    def test_basic_roundtrip(self, tmp_path: str) -> None:
        orig = _make_minimal_snapshot()
        path = os.path.join(str(tmp_path), "test_snap.npz")
        save_snapshot(orig, path)
        loaded = load_snapshot(path)

        # Scalar / metadata
        assert loaded.dt_ms == orig.dt_ms
        assert loaded._n_pops == orig._n_pops
        assert loaded._n_regions == orig._n_regions
        assert loaded._n_tracts == orig._n_tracts
        assert loaded._n_recorded == orig._n_recorded
        assert loaded._gain_sample_step == orig._gain_sample_step
        assert loaded._cond_sample_step == orig._cond_sample_step
        assert loaded._gain_sample_times == orig._gain_sample_times

        # Pop / region keys
        assert loaded._pop_keys == orig._pop_keys
        assert loaded._pop_index == orig._pop_index
        assert loaded._region_keys == orig._region_keys
        assert loaded._region_pop_indices == orig._region_pop_indices

        # Tract keys roundtrip through SynapseId.to_key/from_key
        for lo, oo in zip(loaded._tract_keys, orig._tract_keys):
            assert lo.to_key() == oo.to_key()

        # Pop sizes
        np.testing.assert_array_equal(loaded._pop_sizes, orig._pop_sizes)

        # Spike counts
        np.testing.assert_array_equal(loaded._pop_spike_counts, orig._pop_spike_counts)
        np.testing.assert_array_equal(loaded._region_spike_counts, orig._region_spike_counts)
        np.testing.assert_array_equal(loaded._tract_sent, orig._tract_sent)
        for lo, oo in zip(loaded._per_neuron_spike_counts, orig._per_neuron_spike_counts):
            np.testing.assert_array_equal(lo, oo)

        # Sample indices
        for lo, oo in zip(loaded._v_sample_idx, orig._v_sample_idx):
            np.testing.assert_array_equal(lo, oo)
        for lo, oo in zip(loaded._c_sample_idx, orig._c_sample_idx):
            np.testing.assert_array_equal(lo, oo)

    def test_spike_times_roundtrip(self, tmp_path: str) -> None:
        orig = _make_minimal_snapshot()
        path = os.path.join(str(tmp_path), "spike_snap.npz")
        save_snapshot(orig, path)
        loaded = load_snapshot(path)

        for key in orig._pop_keys:
            assert key in loaded._spike_times
            for ni in range(len(orig._spike_times[key])):
                assert loaded._spike_times[key][ni] == orig._spike_times[key][ni]

    def test_voltage_traces_roundtrip(self, tmp_path: str) -> None:
        orig = _make_minimal_snapshot(include_voltages=True)
        path = os.path.join(str(tmp_path), "v_snap.npz")
        save_snapshot(orig, path)
        loaded = load_snapshot(path)

        assert loaded._voltages is not None
        np.testing.assert_allclose(loaded._voltages, orig._voltages, rtol=1e-6)

    def test_no_voltages(self, tmp_path: str) -> None:
        orig = _make_minimal_snapshot(include_voltages=False)
        path = os.path.join(str(tmp_path), "no_v.npz")
        save_snapshot(orig, path)
        loaded = load_snapshot(path)
        assert loaded._voltages is None

    def test_conductance_traces_roundtrip(self, tmp_path: str) -> None:
        orig = _make_minimal_snapshot(include_conductances=True)
        path = os.path.join(str(tmp_path), "cond_snap.npz")
        save_snapshot(orig, path)
        loaded = load_snapshot(path)

        for attr in ("_g_exc_samples", "_g_inh_samples", "_g_nmda_samples", "_g_gaba_b_samples"):
            lo = getattr(loaded, attr)
            oo = getattr(orig, attr)
            assert lo is not None and oo is not None
            np.testing.assert_allclose(lo, oo, rtol=1e-6)

    def test_no_conductances(self, tmp_path: str) -> None:
        orig = _make_minimal_snapshot(include_conductances=False, cond_steps=0)
        path = os.path.join(str(tmp_path), "no_cond.npz")
        save_snapshot(orig, path)
        loaded = load_snapshot(path)
        assert loaded._g_exc_samples is None

    def test_trajectory_buffers_roundtrip(self, tmp_path: str) -> None:
        orig = _make_minimal_snapshot()
        path = os.path.join(str(tmp_path), "traj_snap.npz")
        save_snapshot(orig, path)
        loaded = load_snapshot(path)

        np.testing.assert_allclose(loaded._g_L_scale_history, orig._g_L_scale_history, rtol=1e-6)
        np.testing.assert_allclose(loaded._stp_efficacy_history, orig._stp_efficacy_history, rtol=1e-6)
        np.testing.assert_allclose(loaded._nm_concentration_history, orig._nm_concentration_history, rtol=1e-6)

    def test_static_metadata_roundtrip(self, tmp_path: str) -> None:
        orig = _make_minimal_snapshot()
        path = os.path.join(str(tmp_path), "meta_snap.npz")
        save_snapshot(orig, path)
        loaded = load_snapshot(path)

        assert loaded._pop_polarities == orig._pop_polarities
        assert loaded._tract_delay_ms == orig._tract_delay_ms
        assert loaded._homeostasis_target_hz == orig._homeostasis_target_hz
        assert loaded._stp_configs == orig._stp_configs
        assert loaded._stp_final_state == orig._stp_final_state
        assert loaded._pop_neuron_params == orig._pop_neuron_params

    def test_learning_buffers_roundtrip(self, tmp_path: str) -> None:
        orig = _make_minimal_snapshot(include_learning=True)
        path = os.path.join(str(tmp_path), "learn_snap.npz")
        save_snapshot(orig, path)
        loaded = load_snapshot(path)

        assert loaded._learning_keys == orig._learning_keys
        # bcm_keys undergo int→str coercion through JSON roundtrip
        assert [str(k) for k in loaded._bcm_keys] == [str(k) for k in orig._bcm_keys]
        np.testing.assert_allclose(loaded._weight_dist_history, orig._weight_dist_history, rtol=1e-6)
        np.testing.assert_allclose(loaded._weight_update_magnitude_history, orig._weight_update_magnitude_history, rtol=1e-6)
        np.testing.assert_allclose(loaded._eligibility_mean_history, orig._eligibility_mean_history, rtol=1e-6)
        np.testing.assert_allclose(loaded._bcm_theta_history, orig._bcm_theta_history, rtol=1e-6)
        np.testing.assert_allclose(loaded._homeostatic_correction_rate, orig._homeostatic_correction_rate, rtol=1e-6)

    def test_config_roundtrip(self, tmp_path: str) -> None:
        orig = _make_minimal_snapshot()
        path = os.path.join(str(tmp_path), "cfg_snap.npz")
        save_snapshot(orig, path)
        loaded = load_snapshot(path)

        assert loaded.config.n_timesteps == orig.config.n_timesteps
        assert loaded.config.dt_ms == orig.config.dt_ms
        assert loaded.config.voltage_sample_size == orig.config.voltage_sample_size
        assert loaded.config.rate_bin_ms == orig.config.rate_bin_ms

    def test_npz_extension_auto_appended(self, tmp_path: str) -> None:
        orig = _make_minimal_snapshot()
        path = os.path.join(str(tmp_path), "no_ext")
        save_snapshot(orig, path)
        assert os.path.exists(path + ".npz")
        loaded = load_snapshot(path)
        assert loaded._n_pops == orig._n_pops


# =====================================================================
# Threshold reconstruction
# =====================================================================


class TestReconstructThresholds:

    def test_nested_format(self) -> None:
        """New nested format with sub-group keys."""
        from dataclasses import asdict
        ht = HealthThresholds()
        d = asdict(ht)
        result = _reconstruct_thresholds(d)
        assert result.firing.fr_severely_low_multiplier == ht.firing.fr_severely_low_multiplier
        assert result.homeostasis.gain_drift_pct == ht.homeostasis.gain_drift_pct

    def test_empty_dict(self) -> None:
        """Empty dict → default thresholds."""
        result = _reconstruct_thresholds({})
        assert result.firing.fr_severely_low_multiplier > 0
