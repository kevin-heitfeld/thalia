"""Neural coupling analysis — domain-focused sub-modules.

Organised into six domain modules:

* ``cerebellar``       — Purkinje–DCN anti-correlation, IO synchrony
* ``cross_frequency``  — generic CFC: PAC, AAC, PFC for arbitrary band pairs
* ``hippocampal``      — PLV, HFO, SWR coupling, theta-sequence timing
* ``network_dynamics`` — spike avalanches, beta bursts, synaptic gain
* ``spike_field``      — general spike-field coherence (PLV/PPC) across all regions
* ``thalamocortical``  — relay burst mode, laminar cascade latency

All public functions are re-exported here so that existing
``from .coupling import compute_plv_theta_per_region`` imports continue
to work.
"""

from __future__ import annotations

from .cerebellar import compute_cerebellar_metrics
from .cross_frequency import compute_cfc_per_region
from .hippocampal import (
    compute_ca3_ca1_theta_sequence,
    compute_hfo_per_region,
    compute_plv_theta_per_region,
    compute_swr_ca3_ca1_coupling,
)
from .network_dynamics import (
    compute_beta_burst_stats,
    compute_effective_synaptic_gain,
    compute_spike_avalanches,
)
from .lfp_proxy import build_all_region_lfp_proxies, build_region_lfp_proxy
from .spike_field import compute_spike_field_coherence
from .thalamocortical import (
    compute_laminar_cascade,
    compute_relay_burst_mode,
    detect_thalamic_volleys,
)

__all__ = [
    "build_all_region_lfp_proxies",
    "build_region_lfp_proxy",
    "compute_beta_burst_stats",
    "compute_ca3_ca1_theta_sequence",
    "compute_cerebellar_metrics",
    "compute_cfc_per_region",
    "compute_effective_synaptic_gain",
    "compute_hfo_per_region",
    "compute_laminar_cascade",
    "compute_plv_theta_per_region",
    "compute_relay_burst_mode",
    "compute_spike_avalanches",
    "compute_spike_field_coherence",
    "compute_swr_ca3_ca1_coupling",
    "detect_thalamic_volleys",
]
