"""Diagnostics result metrics — population, region, oscillatory, connectivity,
homeostatic, and learning data structures.

Pure data module: no simulation dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from thalia.typing import PopulationName, RegionName, SynapseId

from ._helpers import compute_ei_current_ratio as _compute_ei_current_ratio
from ._helpers import compute_ei_ratio as _compute_ei_ratio


# =============================================================================
# POPULATION STATISTICS
# =============================================================================


@dataclass
class PopulationStats:
    """Complete statistics for one population over the recorded window."""

    region_name: RegionName
    population_name: PopulationName
    n_neurons: int

    # Firing rate
    mean_fr_hz: float
    std_fr_hz: float
    fraction_silent: float
    fraction_hyperactive: float
    hyperactive_threshold_hz: float
    total_spikes: int

    # ISI statistics
    isi_mean_ms: float
    isi_cv: float
    isi_cv_corrected: float
    fraction_bursting: float
    fraction_refractory_violations: float
    isi_cv2: float
    isi_cv2_population: float
    fraction_isi_lt_80ms: float
    da_burst_events_per_s: float
    sfa_index: float

    # Variability / synchrony metrics
    per_neuron_ff: float
    pairwise_correlation: float
    fraction_burst_events: float

    # Per-neuron FR histogram (for CDF plots)
    fr_histogram: np.ndarray
    fr_histogram_edges: np.ndarray

    # Biological reference
    bio_range_hz: Optional[Tuple[float, float]]

    # Network state classifier
    network_state: str = "unknown"

    # Up/Down state detection
    voltage_bimodality: float = float("nan")
    up_state_duration_ms: float = float("nan")
    down_state_duration_ms: float = float("nan")

    # SFA time constant (single-exponential fit)
    sfa_tau_ms: float = float("nan")

    # SFA two-component exponential fit
    sfa_tau_fast_ms: float = float("nan")
    sfa_tau_slow_ms: float = float("nan")

    # Two-compartment apical conductance
    mean_g_exc_apical: float = float("nan")

    # Two-compartment dendritic validation (E7)
    bap_attenuation_ratio: float = float("nan")    # V_dend deflection / V_soma spike height
    nmda_plateau_fraction: float = float("nan")     # fraction of time g_plateau_dend > 0
    coincidence_gain: float = float("nan")          # FR(both inputs) / FR(basal only)

    # Fano factor scaling across bin widths
    fano_scaling: List[Tuple[float, float]] = field(default_factory=list)

    # Pairwise correlation distribution
    pairwise_correlation_distribution: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def is_silent(self) -> bool:
        return self.fraction_silent > 0.99

    @property
    def is_hyperactive(self) -> bool:
        return self.fraction_hyperactive > 0.10

    @property
    def bio_plausibility(self) -> str:
        """'ok', 'low', 'high', or 'unknown'."""
        if self.bio_range_hz is None:
            return "unknown"
        lo, hi = self.bio_range_hz
        if self.mean_fr_hz < lo * 0.5:
            return "low"
        if self.mean_fr_hz > hi * 2.0:
            return "high"
        return "ok"

    @property
    def fr_cv_across_neurons(self) -> float:
        """Coefficient of variation of per-neuron firing rates (spatial heterogeneity)."""
        if self.mean_fr_hz < 1e-9:
            return float("nan")
        return self.std_fr_hz / self.mean_fr_hz


# =============================================================================
# REGION STATISTICS
# =============================================================================


@dataclass
class RegionStats:
    """Statistics aggregated across all populations in a region."""

    region_name: RegionName
    populations: Dict[PopulationName, PopulationStats]
    mean_fr_hz: float
    total_spikes: int

    # E/I balance (conductance-based where available, else NaN)
    mean_g_exc: float
    mean_g_nmda: float
    mean_g_inh: float
    mean_g_gaba_a: float = float("nan")
    mean_g_gaba_b: float = float("nan")

    # E/I current ratio (driving-force-weighted)
    mean_voltage: float = float("nan")
    mean_E_E: float = float("nan")
    mean_E_nmda: float = float("nan")
    mean_E_I: float = float("nan")
    mean_E_GABA_B: float = float("nan")

    # D1/D2 competition index (striatal regions only)
    d1_d2_competition_index_200ms: float = float("nan")
    d1_d2_competition_index_50ms: float = float("nan")

    # E/I lag cross-correlation
    ei_lag_ms: float = float("nan")
    ei_xcorr_peak: float = float("nan")

    @property
    def ei_ratio(self) -> float:
        """Excitatory (AMPA + NMDA) to inhibitory (GABA-A + GABA-B) conductance ratio."""
        return _compute_ei_ratio(self.mean_g_exc, self.mean_g_nmda, self.mean_g_inh)

    @property
    def ei_current_ratio(self) -> float:
        """Driving-force-weighted E/I current ratio at mean membrane voltage."""
        return _compute_ei_current_ratio(
            self.mean_g_exc, self.mean_g_nmda,
            self.mean_g_gaba_a, self.mean_g_gaba_b,
            self.mean_E_E, self.mean_E_nmda,
            self.mean_E_I, self.mean_E_GABA_B,
            self.mean_voltage,
        )

    @property
    def is_active(self) -> bool:
        return self.mean_fr_hz > 0.1

    @property
    def has_pathological_populations(self) -> bool:
        for p in self.populations.values():
            if p.is_silent or p.is_hyperactive:
                return True
        return False


# =============================================================================
# COUPLING RESULT TYPES
# =============================================================================


@dataclass
class AvalancheStats:
    """Power-law fit and branching ratio from spike avalanche analysis."""

    exponent: float = float("nan")
    r2: float = float("nan")
    branching_ratio: float = float("nan")


@dataclass
class CerebellarCouplingStats:
    """Purkinje–DCN anti-correlation and IO pairwise synchrony."""

    purkinje_dcn_corr: Dict[str, float] = field(default_factory=dict)
    io_pairwise_corr: Dict[str, float] = field(default_factory=dict)


@dataclass
class PlvThetaStats:
    """Spike–theta phase-locking values for hippocampal regions."""

    values: Dict[str, float] = field(default_factory=dict)
    used_fallback: Dict[str, bool] = field(default_factory=dict)


@dataclass
class SpikeFieldResult:
    """Spike-field coherence result for one (population, band) pair.

    ``plv`` is the standard phase-locking value (biased for small *n_spikes*).
    ``ppc`` is the pairwise phase consistency (Vinck et al. 2010), an unbiased
    estimator defined as ``(plv² · N - 1) / (N - 1)`` for *N* ≥ 2.
    """

    plv: float
    ppc: float
    n_spikes: int
    band: str


@dataclass
class BetaBurstRegionStats:
    """Per-region beta burst detection results."""

    n_bursts: float = 0.0
    mean_duration_ms: float = float("nan")
    max_duration_ms: float = float("nan")
    mean_ibi_ms: float = float("nan")


@dataclass
class SwrCouplingRegionStats:
    """Per-region SWR CA3→CA1 cross-correlation result."""

    ca3_ca1_xcorr_peak: float = float("nan")
    ca3_ca1_lag_ms: float = float("nan")


@dataclass
class ThetaSequenceRegionStats:
    """Per-region CA3→CA1 theta-sequence cross-correlation result."""

    xcorr_peak: float = float("nan")
    peak_lag_ms: float = float("nan")


@dataclass
class LaminarCascadeRegionStats:
    """Per-region cortical layer latencies after thalamic volleys."""

    l4_lat_ms: float = float("nan")
    l23_lat_ms: float = float("nan")
    l5_lat_ms: float = float("nan")
    l6_lat_ms: float = float("nan")


# =============================================================================
# CROSS-FREQUENCY COUPLING
# =============================================================================


@dataclass
class CFCResult:
    """Single cross-frequency coupling measurement for a region."""

    region: str          # region name
    coupling_type: str   # "pac", "aac", or "pfc"
    phase_band: str      # EEG band name for phase / low-freq signal
    amp_band: str        # EEG band name for amplitude / high-freq signal
    value: float         # coupling metric (NaN when not computable)


# =============================================================================
# OSCILLATORY STATISTICS
# =============================================================================


@dataclass
class OscillatoryStats:
    """Spectral and coherence analysis."""

    region_band_power: Dict[RegionName, Dict[str, float]]
    region_band_power_absolute: Dict[RegionName, Dict[str, float]]

    region_dominant_freq: Dict[RegionName, float]
    region_dominant_band: Dict[RegionName, str]

    coherence_theta: np.ndarray
    coherence_beta:  np.ndarray
    coherence_gamma: np.ndarray
    region_order: List[RegionName]

    global_dominant_freq_hz: float
    global_band_power: Dict[str, float]

    freq_resolution_hz: float = float("nan")

    cfc_results: List["CFCResult"] = field(default_factory=list)
    hfo_band_power: Dict[str, float] = field(default_factory=dict)

    swr_ca3_ca1_coupling: Dict[str, SwrCouplingRegionStats] = field(default_factory=dict)
    ca3_ca1_theta_sequence: Dict[str, ThetaSequenceRegionStats] = field(default_factory=dict)

    plv_theta: PlvThetaStats = field(default_factory=PlvThetaStats)

    # General spike-field coherence: (region, population) → SpikeFieldResult.
    spike_field_coherence: Dict[Tuple[str, str], SpikeFieldResult] = field(default_factory=dict)

    avalanche: AvalancheStats = field(default_factory=AvalancheStats)

    cerebellar: CerebellarCouplingStats = field(default_factory=CerebellarCouplingStats)

    beta_burst_stats: Dict[str, BetaBurstRegionStats] = field(default_factory=dict)

    region_integration_tau_ms: Dict[str, float] = field(default_factory=dict)

    # Per-region aperiodic (1/f) exponent from PSD fit: PSD(f) ∝ 1/f^χ.
    # Healthy cortex χ ≈ 1.0–2.0 (He 2014; Donoghue et al. 2020).
    region_aperiodic_exponent: Dict[str, float] = field(default_factory=dict)

    relay_burst_mode: Dict[str, float] = field(default_factory=dict)

    laminar_cascade_latencies: Dict[str, LaminarCascadeRegionStats] = field(default_factory=dict)

    # LFP proxy method used per region: "current" (conductance-based) or "spike_rate".
    lfp_proxy_methods: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# CONNECTIVITY STATISTICS
# =============================================================================


@dataclass
class ConnectivityStats:
    """Axonal tract transmission statistics."""

    @dataclass
    class TractStats:
        synapse_id: SynapseId
        spikes_sent: int
        transmission_ratio: float
        is_functional: bool
        measured_delay_ms: float
        expected_delay_ms: float
        anticausal_peak_ms: float
        transmission_jitter_ms: float = float("nan")

    tracts: List[TractStats]
    n_functional: int
    n_broken: List["ConnectivityStats.TractStats"]


# =============================================================================
# HOMEOSTATIC STATISTICS
# =============================================================================


@dataclass
class HomeostaticStats:
    """Homeostatic gain and STP state over time."""

    gain_trajectories: Dict[str, np.ndarray]
    gain_sample_times_ms: np.ndarray

    stp_efficacy_history: Dict[str, np.ndarray]
    stp_final_state: Dict[str, Dict[str, float]]


# =============================================================================
# LEARNING / TRAINING STATISTICS
# =============================================================================


@dataclass
class WeightDistStats:
    """Weight distribution summary for a single synapse group at a point in time."""

    mean: float
    std: float
    min_val: float
    max_val: float
    sparsity: float


@dataclass
class STDPTimingStats:
    """Pre-post spike timing distribution for one learning pathway.

    Δt = t_post − t_pre (using emission times, not arrival times).
    Positive Δt → pre fires before post → canonical LTP window.
    Negative Δt → post fires before pre → canonical LTD window.
    """

    mean_delta_ms: float
    std_delta_ms: float
    ltp_fraction: float  # Fraction of pairs with Δt > 0
    n_pairs: int
    histogram_edges_ms: np.ndarray
    histogram_counts: np.ndarray


@dataclass
class SynapseLearningSummary:
    """End-of-run learning summary for one synapse group."""

    synapse_id: str
    strategy_type: str

    weight_start: WeightDistStats
    weight_end: WeightDistStats

    mean_update_magnitude: float
    weight_drift: float

    mean_eligibility: float
    ltp_ltd_ratio: float

    bcm_theta_start: float
    bcm_theta_end: float


@dataclass
class LearningStats:
    """Aggregate learning/training diagnostics."""

    synapse_summaries: Dict[str, SynapseLearningSummary]
    weight_trajectories: Dict[str, List[Tuple[float, WeightDistStats]]]
    update_magnitude_trajectories: Dict[str, np.ndarray]
    eligibility_trajectories: Dict[str, np.ndarray]
    bcm_theta_trajectories: Dict[str, np.ndarray]
    homeostatic_correction_rate: Dict[str, np.ndarray]
    popvec_stability: float
    sample_times_ms: np.ndarray
    da_eligibility_alignment: Dict[str, float] = field(default_factory=dict)
    stdp_timing: Dict[str, STDPTimingStats] = field(default_factory=dict)
