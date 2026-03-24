"""Cerebellar timing metrics: Purkinje–DCN anti-correlation and IO pairwise synchrony."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from thalia.diagnostics._helpers import bin_counts_1d, safe_pearson_r, bin_spike_times_to_array
from thalia.diagnostics.diagnostics_metrics import CerebellarCouplingStats
from thalia.diagnostics.diagnostics_snapshot import RecorderSnapshot


def compute_cerebellar_metrics(rec: RecorderSnapshot, T: int) -> CerebellarCouplingStats:
    """Compute cerebellar timing metrics: Purkinje–DCN anti-correlation and IO pairwise synchrony.

    Returns:
        :class:`CerebellarCouplingStats` with ``purkinje_dcn_corr`` and ``io_pairwise_corr`` dicts.
    """
    purkinje_dcn_corr: Dict[str, float] = {}
    io_pairwise_corr: Dict[str, float] = {}
    bin_steps_cb = max(1, int(10.0 / rec.dt_ms))  # 10 ms bins for Purkinje-DCN
    for rn_cb in rec._region_keys:
        purk_idx = [
            i for i, (r, p) in enumerate(rec._pop_keys)
            if r == rn_cb and "purkinje" in p.lower()
        ]
        dcn_idx = [
            i for i, (r, p) in enumerate(rec._pop_keys)
            if r == rn_cb and "dcn" in p.lower()
        ]
        if purk_idx and dcn_idx:
            n_bins_cb = T // bin_steps_cb
            if n_bins_cb >= 4:
                purk_counts = rec._pop_spike_counts[:T, purk_idx].sum(axis=1).astype(np.float64)
                dcn_counts  = rec._pop_spike_counts[:T, dcn_idx ].sum(axis=1).astype(np.float64)
                p_b = bin_counts_1d(purk_counts, n_bins_cb, bin_steps_cb)
                d_b = bin_counts_1d(dcn_counts, n_bins_cb, bin_steps_cb)
                r_val = safe_pearson_r(p_b, d_b)
                if not np.isnan(r_val):
                    purkinje_dcn_corr[rn_cb] = r_val

        io_idx = [
            i for i, (r, p) in enumerate(rec._pop_keys)
            if r == rn_cb and "inferior_olive" in p.lower()
        ]
        if io_idx:
            # 500 ms bins: IO neurons fire at 0.3–3 Hz, so 200 ms bins
            # capture at most 1 spike per neuron — too few for meaningful
            # Pearson ρ.  500 ms gives ≥2 spikes even at the low end.
            bin_steps_io = max(1, int(500.0 / rec.dt_ms))
            n_bins_io = T // bin_steps_io
            if n_bins_io >= 4 and len(io_idx) >= 2:
                io_key_list = [
                    (rn_cb, p)
                    for _, (r2, p) in enumerate(rec._pop_keys)
                    if r2 == rn_cb and "inferior_olive" in p.lower()
                ]
                io_pair_rs: List[float] = []
                for ki, io_key_i in enumerate(io_key_list):
                    if io_key_i not in rec._spike_times:
                        continue
                    for _kj, io_key_j in enumerate(io_key_list[ki + 1:], start=ki + 1):
                        if io_key_j not in rec._spike_times:
                            continue
                        vi = bin_spike_times_to_array(rec._spike_times[io_key_i], n_bins_io, bin_steps_io)
                        vj = bin_spike_times_to_array(rec._spike_times[io_key_j], n_bins_io, bin_steps_io)
                        r_val = safe_pearson_r(vi, vj)
                        if not np.isnan(r_val):
                            io_pair_rs.append(r_val)
                if io_pair_rs:
                    io_pairwise_corr[rn_cb] = float(np.mean(io_pair_rs))
    return CerebellarCouplingStats(purkinje_dcn_corr=purkinje_dcn_corr, io_pairwise_corr=io_pairwise_corr)
