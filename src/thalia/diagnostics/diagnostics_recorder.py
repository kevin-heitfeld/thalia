"""
DiagnosticsRecorder: Training-loop-integrated brain diagnostics.

Architecture
============
The recorder is a thin coordinator that delegates to:

- :class:`PopulationIndex` — immutable brain topology metadata
- :class:`BufferManager` — all mutable recording buffers
- :func:`build_snapshot` — one-shot snapshot assembly

The hot-path ``record()`` and ``_sample_*()`` methods remain here because
they need the live ``Brain`` reference for tensor reads.

The recorder is completely decoupled from the simulation loop.  It never
drives its own loop; instead, call ``record(t, inputs, outputs)`` once per
timestep from whatever loop you own (training, evaluation, pre-training
diagnostic run).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import torch

from thalia.learning.strategies import (
    BCMStrategy,
    D1STDPStrategy,
    STDPStrategy,
    ThreeFactorStrategy,
)
from thalia.typing import BrainOutput, SynapticInput

from .brain_protocol import BrainLike
from .buffer_manager import BufferManager
from .diagnostics_config import DiagnosticsConfig
from .diagnostics_snapshot import RecorderSnapshot
from .population_index import PopulationIndex
from .snapshot_builder import build_snapshot

if TYPE_CHECKING:
    from thalia.brain.neurons import ConductanceLIF, TwoCompartmentLIF


# =============================================================================
# HELPERS
# =============================================================================


def _get_attr_fallback(obj: Any, *attrs: str) -> Optional[torch.Tensor]:
    """Return the first non-None attribute from *obj*, or ``None`` if all are absent.

    Used to transparently handle models that expose conductances under
    different names (e.g. ``g_E`` on :class:`ConductanceLIF` vs
    ``g_E_basal`` on :class:`TwoCompartmentLIF`).
    """
    for a in attrs:
        v = getattr(obj, a, None)
        if v is not None:
            return v
    return None


# =============================================================================
# DIAGNOSTICS RECORDER
# =============================================================================


class DiagnosticsRecorder:
    """Records brain activity for comprehensive diagnostics.

    Thin coordinator: call ``record(t, inputs, outputs)`` once per timestep,
    then ``to_snapshot()`` to get a serialisable :class:`RecorderSnapshot`.

    Delegates to :class:`PopulationIndex` (immutable metadata),
    :class:`BufferManager` (mutable recording state), and
    :func:`build_snapshot` (snapshot assembly).
    """

    def __init__(self, brain: BrainLike, config: DiagnosticsConfig) -> None:
        self.brain = brain
        self.config = config
        self.dt_ms = brain.dt_ms

        self._index = PopulationIndex(brain, config)
        self._buffers = BufferManager(self._index, config, brain.dt_ms)

    # =========================================================================
    # RESET
    # =========================================================================

    def reset(self) -> None:
        """Re-zero all recording buffers in-place (faster than re-allocating).

        Call before starting a new recording window.  Re-uses the same
        pre-allocated arrays so memory stays stable across repeated runs.
        """
        self._buffers.reset(self._index.n_learning)

    # =========================================================================
    # RECORD  — the single integration point
    # =========================================================================

    def record(self, timestep: int, inputs: SynapticInput, outputs: BrainOutput) -> None:
        """Record one timestep of brain output.

        Call this immediately after ``brain.forward()`` inside your loop.

        Args:
            timestep: Zero-based timestep index.  In full-recording mode
                (``window_size is None``) this must be ``< config.n_timesteps``.
                In windowed mode the recorder accepts any non-negative timestep.
            inputs: Dict of synaptic inputs to the brain at this timestep.
            outputs: Dict ``{region_name: {pop_name: spike_tensor}}`` returned
                by ``brain.forward()``.
        """
        idx = self._index
        buf = self._buffers
        buf.recording_started = True
        t = timestep

        # ── Buffer index computation ──────────────────────────────────────────
        buf_idx = buf.compute_write_index(t, self.config.n_timesteps)
        if buf_idx is None:
            warnings.warn(
                f"DiagnosticsRecorder: timestep {t} exceeds n_timesteps "
                f"({self.config.n_timesteps}).  Ignoring.",
                stacklevel=2,
            )
            return

        # --- Spike counts per population & per region ----------------------------------
        for region_name, region_outputs in outputs.items():
            region_idx = idx.region_index.get(region_name)
            region_total = 0

            for pop_name, pop_spikes in region_outputs.items():
                key = (region_name, pop_name)
                pop_idx = idx.pop_index.get(key)
                if pop_idx is None:
                    continue

                n_active = int(pop_spikes.sum().item())
                buf.pop_spike_counts[buf_idx, pop_idx] = n_active
                region_total += n_active

                if n_active > 0:
                    spike_arr = pop_spikes.cpu().numpy()
                    # Flat-buffer accumulation — one numpy slice per pop per step.
                    nidx = np.where(spike_arr)[0].astype(np.int32)
                    n_new = len(nidx)
                    cnt = int(buf.spike_flat_n[pop_idx])
                    cap = len(buf.spike_flat_nidx[pop_idx])
                    if cnt + n_new > cap:
                        new_cap = max(cnt + n_new, cap * 2)
                        buf.spike_flat_nidx[pop_idx] = np.resize(
                            buf.spike_flat_nidx[pop_idx], new_cap
                        )
                        buf.spike_flat_ts[pop_idx] = np.resize(
                            buf.spike_flat_ts[pop_idx], new_cap
                        )
                    buf.spike_flat_nidx[pop_idx][cnt:cnt + n_new] = nidx
                    buf.spike_flat_ts[pop_idx][cnt:cnt + n_new] = t
                    buf.spike_flat_n[pop_idx] = cnt + n_new
                    # Per-neuron cumulative counts — enables correct
                    # fraction_silent / fraction_hyperactive.
                    buf.per_neuron_spike_counts[pop_idx] += spike_arr.astype(np.int32)

            if region_idx is not None:
                buf.region_spike_counts[buf_idx, region_idx] = region_total

        # --- Axonal tract transmissions ------------------------------------------------
        for synapse_id, tract_idx in idx.tract_index.items():
            src_spikes = outputs.get(synapse_id.source_region, {}).get(
                synapse_id.source_population, None
            )
            if src_spikes is not None:
                buf.tract_sent[buf_idx, tract_idx] = int(src_spikes.sum().item())

        # --- Internal state sampling ---------------------------------------------------
        gi_steps = max(1, int(self.config.gain_sample_interval_ms / self.dt_ms))
        if t % gi_steps == 0:
            self._sample_gains(t)

        self._sample_voltages(buf_idx)
        ci = self.config.conductance_sample_interval_steps
        if ci > 0 and t % ci == 0:
            self._sample_conductances(t)

        buf.n_recorded = t + 1

        # --- Windowed bookkeeping ------------------------------------------------------
        buf.advance_cursor()

    # =========================================================================
    # INTERNAL STATE SAMPLING (hot path — needs live Brain reference)
    # =========================================================================

    def _sample_gains(self, timestep: int) -> None:
        """Sample homeostatic g_L_scale and STP efficacy for all populations/synapses."""
        idx = self._index
        buf = self._buffers
        step = buf.gain_sample_step
        if step >= buf.g_L_scale_history.shape[0]:
            return

        for pop_i, (rn, pn) in enumerate(idx.pop_keys):
            pop_obj = self.brain.regions[rn].neuron_populations[pn]
            if hasattr(pop_obj, "g_L_scale") and pop_obj.g_L_scale is not None:
                val = pop_obj.g_L_scale
                buf.g_L_scale_history[step, pop_i] = float(
                    val.mean().item() if isinstance(val, torch.Tensor) else val
                )

        # STP x·u efficacy
        for stp_idx, (rn, syn_id) in enumerate(idx.stp_keys):
            stp_mod = self.brain.regions[rn].get_stp_module(syn_id)
            assert stp_mod is not None
            buf.stp_efficacy_history[step, stp_idx] = float(
                (stp_mod.x * stp_mod.u).mean().item()
            )

        # Neuromodulator receptor concentrations
        if idx.n_nm_receptors > 0:
            for nm_idx, (rn, mod_name) in enumerate(idx.nm_receptor_keys):
                try:
                    obj: Any = self.brain.regions[rn]
                    for part in mod_name.split("."):
                        obj = getattr(obj, part)
                    conc: torch.Tensor = obj.concentration
                    buf.nm_concentration_history[step, nm_idx] = float(
                        conc.mean().item()
                    )
                except (AttributeError, KeyError):
                    pass

        buf.gain_sample_times.append(timestep)
        buf.gain_sample_step += 1

        # Sample learning state on the same schedule as gains
        self._sample_learning_state(step)

    def _sample_learning_state(self, step: int) -> None:
        """Sample weight distributions, eligibility traces, BCM thresholds, and homeostatic correction."""
        idx = self._index
        buf = self._buffers

        # ── Weight distributions and update magnitudes ───────────────────
        if idx.n_learning > 0 and buf.weight_dist_history is not None:
            for learn_idx, (rn, syn_id) in enumerate(idx.learning_keys):
                region = self.brain.regions[rn]
                w = region.get_synaptic_weights(syn_id).data

                w_cpu = w.cpu().numpy().astype(np.float32)
                w_mean = float(np.mean(w_cpu))
                w_std = float(np.std(w_cpu))
                w_min = float(np.min(w_cpu))
                w_max = float(np.max(w_cpu))
                n_total = w_cpu.size
                sparsity = float(np.sum(np.abs(w_cpu) < 1e-6) / max(1, n_total))

                buf.weight_dist_history[step, learn_idx, :] = [
                    w_mean, w_std, w_min, w_max, sparsity
                ]

                # Weight update magnitude |ΔW|/|W|
                assert buf.weight_update_magnitude_history is not None
                assert buf.prev_weight_snapshots is not None
                prev = buf.prev_weight_snapshots[learn_idx]
                if prev is not None:
                    dw = np.abs(w_cpu - prev)
                    w_abs_mean = max(float(np.mean(np.abs(prev))), 1e-10)
                    buf.weight_update_magnitude_history[step, learn_idx] = (
                        float(np.mean(dw)) / w_abs_mean
                    )
                buf.prev_weight_snapshots[learn_idx] = w_cpu.copy()

                # ── Eligibility trace statistics ─────────────────────────
                assert buf.eligibility_mean_history is not None
                assert buf.eligibility_ltp_ltd_ratio_history is not None
                strategy = region.get_learning_strategy(syn_id)
                if strategy is not None:
                    elig: Optional[torch.Tensor] = None
                    if isinstance(strategy, (STDPStrategy, ThreeFactorStrategy)):
                        if hasattr(strategy, "trace_manager"):
                            elig = strategy.trace_manager.eligibility
                    elif isinstance(strategy, D1STDPStrategy):
                        if hasattr(strategy, "fast_trace"):
                            elig = strategy.fast_trace + getattr(strategy, "slow_trace", 0)
                    if elig is not None:
                        elig_cpu = elig.detach().cpu().numpy()
                        buf.eligibility_mean_history[step, learn_idx] = float(
                            np.mean(np.abs(elig_cpu))
                        )
                        pos = float(np.sum(elig_cpu[elig_cpu > 0]))
                        neg = float(np.abs(np.sum(elig_cpu[elig_cpu < 0])))
                        ratio = pos / max(neg, 1e-10) if neg > 1e-10 else (float("inf") if pos > 0 else 1.0)
                        buf.eligibility_ltp_ltd_ratio_history[step, learn_idx] = min(ratio, 100.0)

        # ── BCM threshold ────────────────────────────────────────────────
        if idx.n_bcm > 0 and buf.bcm_theta_history is not None:
            for bcm_idx_in_list, learn_idx in enumerate(idx.bcm_keys):
                rn, syn_id = idx.learning_keys[learn_idx]
                strategy = self.brain.regions[rn].get_learning_strategy(syn_id)
                if strategy is not None and isinstance(strategy, BCMStrategy):
                    if hasattr(strategy, "theta"):
                        buf.bcm_theta_history[step, bcm_idx_in_list] = float(
                            strategy.theta.mean().item()
                        )

        # ── Homeostatic correction rate ──────────────────────────────────
        if step > 0:
            for pop_i in range(idx.n_pops):
                prev_g = buf.g_L_scale_history[step - 1, pop_i]
                curr_g = buf.g_L_scale_history[step, pop_i]
                if not (np.isnan(prev_g) or np.isnan(curr_g)):
                    buf.homeostatic_correction_rate[step, pop_i] = abs(curr_g - prev_g)

        # ── Population vector snapshot for representational stability ────
        if (
            step > 0
            and step % buf.popvec_interval == 0
        ):
            snap_idx = len(buf.popvec_snapshot_times)
            if snap_idx < buf.popvec_snapshots.shape[0]:
                # Use recent spike counts as the population vector
                gi_steps = max(1, int(self.config.gain_sample_interval_ms / self.dt_ms))
                t_now = buf.gain_sample_times[-1] if buf.gain_sample_times else 0
                t_start = max(0, t_now - gi_steps * buf.popvec_interval)
                for pop_i in range(idx.n_pops):
                    total_spikes = float(
                        buf.pop_spike_counts[t_start:t_now + 1, pop_i].sum()
                    )
                    n_neurons = max(1, int(idx.pop_sizes[pop_i]))
                    duration_s = max(1e-6, (t_now - t_start + 1) * self.dt_ms / 1000.0)
                    buf.popvec_snapshots[snap_idx, pop_i] = total_spikes / n_neurons / duration_s
                buf.popvec_snapshot_times.append(t_now)

    def _sample_voltages(self, buf_idx: int) -> None:
        """Sample membrane voltages for fixed neurons in each population."""
        buf = self._buffers
        if buf_idx >= buf.voltages.shape[0]:
            return

        idx = self._index
        for pop_i, (rn, pn) in enumerate(idx.pop_keys):
            if rn not in self.brain.regions:
                raise ValueError(f"Region '{rn}' not found in brain")
            if pn not in self.brain.regions[rn].neuron_populations:
                raise ValueError(f"Population '{pn}' not found in region '{rn}'")
            pop_obj: Union[ConductanceLIF, TwoCompartmentLIF] = self.brain.regions[rn].neuron_populations[pn]
            v_idx = idx.v_sample_idx[pop_i]
            if len(v_idx) == 0:
                continue
            vals = pop_obj.V_soma[torch.from_numpy(v_idx).long().to(pop_obj.V_soma.device)]
            buf.voltages[buf_idx, pop_i, : len(v_idx)] = vals.cpu().numpy()

            # Two-compartment: also sample dendritic voltage.
            v_dend = getattr(pop_obj, "V_dend", None)
            if v_dend is not None:
                dend_vals = v_dend[torch.from_numpy(v_idx).long().to(v_dend.device)]
                buf.v_dend[buf_idx, pop_i, : len(v_idx)] = dend_vals.cpu().numpy()

    def _sample_conductances(self, timestep: int) -> None:
        """Sample g_E, g_I, g_nmda, g_gaba_b for fixed neurons."""
        buf = self._buffers
        idx = self._index
        cap = buf.g_exc_samples.shape[0]
        if buf._windowed:
            step = buf.cond_sample_step % cap
        else:
            step = buf.cond_sample_step
            if step >= cap:
                return

        for pop_i, (rn, pn) in enumerate(idx.pop_keys):
            c_idx = idx.c_sample_idx[pop_i]
            if len(c_idx) == 0:
                continue

            dev_idx = torch.from_numpy(c_idx).long()

            pop_obj = self.brain.regions[rn].neuron_populations[pn]

            # ConductanceLIF: g_E / g_I / g_nmda
            # TwoCompartmentLIF: g_E_basal / g_I_basal / g_nmda_basal
            g_exc  = _get_attr_fallback(pop_obj, "g_E",    "g_E_basal")
            g_inh  = _get_attr_fallback(pop_obj, "g_I",    "g_I_basal")
            g_nmda = _get_attr_fallback(pop_obj, "g_nmda", "g_nmda_basal")
            if g_exc is not None:
                dev_idx_dev = dev_idx.to(g_exc.device)
                buf.g_exc_samples[step, pop_i, : len(c_idx)] = (
                    g_exc[dev_idx_dev].cpu().numpy()
                )
            if g_inh is not None:
                dev_idx_dev = dev_idx.to(g_inh.device)
                buf.g_inh_samples[step, pop_i, : len(c_idx)] = (
                    g_inh[dev_idx_dev].cpu().numpy()
                )
            if g_nmda is not None:
                dev_idx_dev = dev_idx.to(g_nmda.device)
                buf.g_nmda_samples[step, pop_i, : len(c_idx)] = (
                    g_nmda[dev_idx_dev].cpu().numpy()
                )
            # GABA_B (slow inhibition; drives theta, up/down states, burst termination)
            g_gaba_b = _get_attr_fallback(pop_obj, "g_gaba_b", "g_gaba_b_basal")
            if g_gaba_b is not None:
                dev_idx_dev = dev_idx.to(g_gaba_b.device)
                buf.g_gaba_b_samples[step, pop_i, : len(c_idx)] = (
                    g_gaba_b[dev_idx_dev].cpu().numpy()
                )

            # Apical AMPA (TwoCompartmentLIF only; NaN for ConductanceLIF populations)
            g_exc_apical = getattr(pop_obj, "g_E_apical", None)
            if g_exc_apical is not None:
                dev_idx_dev = dev_idx.to(g_exc_apical.device)
                buf.g_apical_samples[step, pop_i, : len(c_idx)] = (
                    g_exc_apical[dev_idx_dev].cpu().numpy()
                )

            # NMDA plateau conductance (TwoCompartmentLIF only)
            g_plateau = getattr(pop_obj, "g_plateau_dend", None)
            if g_plateau is not None:
                dev_idx_dev = dev_idx.to(g_plateau.device)
                buf.g_plateau_samples[step, pop_i, : len(c_idx)] = (
                    g_plateau[dev_idx_dev].cpu().numpy()
                )

        buf.cond_sample_step += 1

    # =========================================================================
    # SNAPSHOT — decouple analysis / plotting from live brain state
    # =========================================================================

    def to_snapshot(self) -> RecorderSnapshot:
        """Extract a :class:`RecorderSnapshot` from the current recorder state.

        The snapshot is a self-contained, serialisable copy of everything the
        analysis and plotting functions need.  It carries no references to the
        live ``Brain`` or any ``torch.Tensor`` objects, so it is safe to save
        to disk, share between processes, or use after the simulation has ended.

        In windowed mode, circular buffers are linearised so the snapshot looks
        like a fresh recording of ``min(window_size, total_steps)`` timesteps
        starting at index 0.  A :class:`RunningStats` attachment provides
        full-run context (total duration, lifetime per-neuron spike counts).

        Call :meth:`RecorderSnapshot.save` on the returned object to persist it.
        Call :meth:`RecorderSnapshot.load` to reload and re-run analysis later.
        """
        return build_snapshot(
            self.brain, self._index, self._buffers, self.config, self.dt_ms
        )
