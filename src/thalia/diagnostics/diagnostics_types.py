"""
Diagnostics data structures — configuration and result dataclasses.

Pure data module: no simulation dependencies, no torch, no matplotlib.
Can be imported freely by notebooks, scripts, and tests without pulling
in the full recorder stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from thalia.typing import PopulationName, RegionName, SynapseId


# =============================================================================
# HEALTH CATEGORY
# =============================================================================


class HealthCategory(StrEnum):
    """Enumerated category tags for :class:`HealthIssue`.

    Using ``StrEnum`` means members compare equal to their string values
    (``HealthCategory.FIRING == "firing"``), so existing JSON serialisation
    and string-based filtering continue to work without changes.
    """

    FIRING = "firing"
    EI_BALANCE = "ei_balance"
    OSCILLATIONS = "oscillations"
    NEUROMODULATORS = "neuromodulators"
    HOMEOSTASIS = "homeostasis"
    CONNECTIVITY = "connectivity"
    CEREBELLAR = "cerebellar"
    THALAMUS = "thalamus"
    INTERNEURON_COVERAGE = "interneuron_coverage"
    LAMINAR = "laminar"


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class HealthThresholds:
    """Biological threshold constants that govern all health checks.

    All values are literature-sourced defaults.  Construct a custom instance
    and assign it to :attr:`DiagnosticsConfig.thresholds` to swap the entire
    threshold profile — e.g. a tighter epileptiform criterion for sparse
    populations, or more permissive gain-drift limits during early training —
    without rebuilding the recorder or touching recording configuration.
    """

    # ── Stability-score weights ───────────────────────────────────────────
    # stability = max(0, 1 - n_crit * critical_weight - n_warn * warning_weight)
    # Calibration: 3–4 CRITICALs drives score to 0; ~14 warnings tolerated.
    critical_weight: float = 0.3
    warning_weight: float = 0.05

    # ── Burst / epileptiform synchrony ────────────────────────────────────
    # Burst co-activation fraction: Binomial(N, p) mean parameter p.
    # Burst threshold per 20 ms window = p·N + 2·√(p·(1−p)·N).
    # 0.30 is a conservative default; use 0.15 for sparser populations.
    burst_coactivation_fraction: float = 0.30
    # Fraction of windows exceeding threshold before CRITICAL epileptiform.
    epileptiform_burst_threshold: float = 0.05
    # Pairwise Pearson ρ (spike-count, 100 ms bins) thresholds.
    pairwise_rho_critical: float = 0.40
    pairwise_rho_warn: float = 0.15

    # ── ISI irregularity ──────────────────────────────────────────────────
    # Global CV below this → regular / pacemaker-like firing (pyramidal).
    # Healthy asynchronous-irregular CV ≈ 0.7–1.3 (Softky & Koch 1993).
    # Lowered 0.5→0.35: many populations (interneurons, subcortical, deep
    # cortical layers) fire more regularly than L2/3 pyramids; 0.5 over-flagged
    # physiologically regular interneurons and L5/L6 pyramids (CV 0.35–0.50).
    isi_cv_regular_threshold: float = 0.35
    # Local CV₂ below this → local pacemaker-like regularity warning.
    # Lowered 0.6→0.35→0.25: many subcortical/pacemaker populations legitimately
    # have CV₂ ≈ 0.25–0.35; threshold 0.35 still flagged regular-but-normal
    # populations (NGCs, GPe, BLA) that fire with consistent pacemaker rhythm.
    isi_cv2_low_threshold: float = 0.25

    # ── Homeostatic gain ──────────────────────────────────────────────────
    # g_L_scale < gain_collapse_threshold → CRITICAL gain collapse.
    gain_collapse_threshold: float = 0.30
    # > gain_drift_pct change between first/last 10 % of trajectory → warning.
    gain_drift_pct: float = 15.0
    # Linear slope over last 50 % implies > gain_slope_pct change → still drifting.
    gain_slope_pct: float = 15.0
    # > stp_drift_pct change in x·u between first/last 10 % → STP not converged.
    # Separate from gain_drift_pct because STP equilibrium is slower to reach
    # than homeostatic gain (tau_d up to 600 ms; full convergence needs 3–5 s
    # even when firing rates are stable).  Raised 35→50% because even after gain
    # converges, STP state drifts slowly at low firing rates (< 3 Hz) where the
    # effective equilibrium takes > 10 s to reach.
    stp_drift_pct: float = 50.0

    # ── Spatial FR heterogeneity ──────────────────────────────────────────
    # FR-CV = std(FR) / mean(FR) across neurons.  Cortical AI state: ≈ 1.4–2.0
    # (stimulus driven).  At rest / spontaneous activity, FR-CV = 0.1–0.5 is
    # biologically plausible — neurons share similar tonic inputs without the
    # stimulus-driven differentiation that produces high FR-CV.  Only FR-CV <
    # 0.10 indicates truly extreme artificial synchrony where all neurons fire
    # at essentially identical rates (< 10% coefficient of variation).
    fr_heterogeneity_low: float = 0.10
    fr_heterogeneity_high: float = 4.0

    # ── Up/down state detection ───────────────────────────────────────────
    # Sarle's BC > threshold → bimodal voltage (Steriade et al. 1993).
    voltage_bimodality_threshold: float = 0.555

    # ── Hippocampal theta PLV ─────────────────────────────────────────────
    # CA1 pyramidal healthy range: PLV 0.15–0.40 (O'Keefe & Recce 1993).
    plv_theta_warn: float = 0.10   # sub-clinical warning
    plv_theta_crit: float = 0.05   # critical — coupling effectively absent
    plv_theta_high: float = 0.80   # excessive lock-in → seizure-like synchrony

    def __post_init__(self) -> None:
        if not (0 < self.critical_weight <= 1.0):
            raise ValueError(
                f"critical_weight must be in (0, 1.0], got {self.critical_weight}"
            )
        if not (0 < self.warning_weight <= 0.5):
            raise ValueError(
                f"warning_weight must be in (0, 0.5], got {self.warning_weight}"
            )
        if self.critical_weight < self.warning_weight:
            raise ValueError(
                f"critical_weight ({self.critical_weight}) must be >= warning_weight ({self.warning_weight})"
            )
        if not (0 < self.epileptiform_burst_threshold < 1.0):
            raise ValueError(
                f"epileptiform_burst_threshold must be in (0, 1), got {self.epileptiform_burst_threshold}"
            )
        if self.pairwise_rho_warn >= self.pairwise_rho_critical:
            raise ValueError(
                f"pairwise_rho_warn ({self.pairwise_rho_warn}) must be < "
                f"pairwise_rho_critical ({self.pairwise_rho_critical})"
            )


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
    coherence_seed: int = 42  # RNG seed for random sampling of coherence pairs

    # Spike avalanche / criticality analysis
    # When True, fits a power-law exponent to the avalanche size distribution
    # (Beggs & Plenz 2003).  Auto-enabled when n_timesteps ≥ 2000 (enough
    # events for a reliable fit).  The O(T) cost is negligible vs simulation.
    compute_avalanches: bool = False
    # Bin width (ms) for avalanche detection.  Beggs & Plenz (2003) recommend
    # ≈ mean inter-electrode delay (~3–4 ms).  Using dt (1 ms) fragments
    # avalanches into singletons and steepens the power-law exponent.
    avalanche_bin_ms: float = 4.0

    # SFA health-check suppression.
    # The SFA index compares firing rate in the first 25 % vs last 25 % of the
    # *analysis* window (post-transient).  One condition makes the index unreliable
    # even after the transient is excluded:
    #
    # Ramping input pattern (``_sensory_ramp``): a monotonically increasing
    # stimulus raises FR throughout the window so every population appears
    # adapted regardless of cellular SFA properties.  The index reflects
    # stimulus dynamics rather than intrinsic adaptation.
    #
    # Set to True when using a ramp pattern.  The driver sets it automatically.
    skip_sfa_health_check: bool = False
    # Sensory input pattern: ``"random"``, ``"rhythmic"``, ``"none"`` …
    # Set by the driver before each ``recorder.analyze()`` call.  Empty = unknown.
    # Used by thalamic and hippocampal health checks to distinguish waking-state
    # stimulation from background / slow-wave contexts.
    sensory_pattern: str = ""
    # Health-check thresholds (biological constants).  Swap for strict/permissive
    # profiles without rebuilding the recorder or changing recording buffers.
    thresholds: HealthThresholds = field(default_factory=HealthThresholds)


# =============================================================================
# RECORDER SNAPSHOT
# =============================================================================


@dataclass
class RecorderSnapshot:
    """Complete, serialisable snapshot of a :class:`DiagnosticsRecorder`'s recorded state."""

    # ── Config ─────────────────────────────────────────────────────────────
    config: DiagnosticsConfig
    dt_ms: float

    # ── Index metadata ──────────────────────────────────────────────────────
    _pop_keys: List[Tuple[str, str]]
    _pop_index: Dict[Tuple[str, str], int]
    _n_pops: int
    _pop_sizes: np.ndarray  # int32 [n_pops]

    _region_keys: List[str]
    _region_index: Dict[str, int]
    _n_regions: int
    _region_pop_indices: Dict[str, List[int]]

    _tract_keys: List[SynapseId]
    _tract_index: Dict[SynapseId, int]
    _n_tracts: int

    _stp_keys: List[Tuple[str, SynapseId]]
    _nm_receptor_keys: List[Tuple[str, str]]
    _n_nm_receptors: int
    _nm_source_pop_keys: List[Tuple[str, str]]

    _v_sample_idx: List[np.ndarray]
    _c_sample_idx: List[np.ndarray]

    # ── Recording counters ──────────────────────────────────────────────────
    _n_recorded: int
    _gain_sample_step: int
    _cond_sample_step: int
    _gain_sample_times: List[int]

    # ── Spike buffers ───────────────────────────────────────────────────────
    _pop_spike_counts: np.ndarray            # int32 [T, n_pops]
    _per_neuron_spike_counts: List[np.ndarray]  # int32 [n_neurons] per pop
    _region_spike_counts: np.ndarray         # int32 [T, n_regions]
    _tract_sent: np.ndarray                  # int32 [T, n_tracts]
    _spike_times: Dict[Tuple[str, str], List[List[int]]]

    # ── State sample buffers ───────────────
    _voltages: Optional[np.ndarray]          # float32 [T, n_pops, V]
    _g_exc_samples: Optional[np.ndarray]     # float32 [n_cond, n_pops, C]
    _g_inh_samples: Optional[np.ndarray]
    _g_nmda_samples: Optional[np.ndarray]
    _g_gaba_b_samples: Optional[np.ndarray]
    _g_apical_samples: Optional[np.ndarray]

    # ── Trajectory buffers ──────────────────────────────────────────────────
    _g_L_scale_history: np.ndarray           # float32 [n_gain, n_pops]
    _stp_efficacy_history: np.ndarray        # float32 [n_gain, n_stp]
    _nm_concentration_history: np.ndarray    # float32 [n_gain, n_nm_receptors]

    # ── Static brain metadata (snapshotted at recording time) ────────────────
    # Population polarity: "excitatory" | "inhibitory" | "any" per (region, pop)
    _pop_polarities: Dict[Tuple[str, str], str]
    # Expected axonal delay (ms) per tract, indexed by tract_idx (same order as _tract_keys)
    _tract_delay_ms: List[float]
    # Homeostatic target firing rate (Hz) per population; absent = no homeostasis configured
    _homeostasis_target_hz: Dict[Tuple[str, str], float]
    # STP config (U, tau_d, tau_f) per stp_idx (same order as _stp_keys)
    _stp_configs: List[Tuple[float, float, float]]
    # STP state at end of recording: synapse_id_str → {"mean_x", "mean_u", "efficacy"}
    _stp_final_state: Dict[str, Dict[str, float]]
    # Per-tract weight stats: synapse_id_str → {"mean", "n_nonzero", "n_total"}
    _tract_weight_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Per-population neuron parameters for analytical rate prediction
    # (region, pop) → {"g_L", "v_threshold", "v_reset", "E_L", "E_E", "E_I",
    #                   "tau_E", "tau_I", "tau_ref", "adapt_increment", "tau_adapt",
    #                   plus specialized: "i_h_conductance", "sk_conductance", etc.}
    _pop_neuron_params: Dict[Tuple[str, str], Dict[str, float]] = field(default_factory=dict)
    # Per-population config class name for type identification by calibration advisor
    # (region, pop) → class name string (e.g. "NorepinephrineNeuronConfig")
    _pop_config_types: Dict[Tuple[str, str], str] = field(default_factory=dict)


# =============================================================================
# RESULT DATA STRUCTURES
# =============================================================================


@dataclass
class PopulationStats:
    """Complete statistics for one population over the recorded window."""

    region_name: RegionName
    population_name: PopulationName
    n_neurons: int

    # Firing rate
    mean_fr_hz: float         # Mean firing rate across all neurons (Hz)
    std_fr_hz: float          # Std across neurons (Hz)
    fraction_silent: float    # Fraction of neurons with zero spikes
    fraction_hyperactive: float  # Fraction firing above population-specific hyperactive threshold
    hyperactive_threshold_hz: float  # Threshold used (1.5 × bio_range upper bound, or 50 Hz if unknown)
    total_spikes: int

    # ISI statistics
    isi_mean_ms: float        # Mean ISI
    isi_cv: float             # Coefficient of variation (irregularity; ~1 = Poisson)
    fraction_bursting: float  # Fraction of ISIs < 10 ms (burst criterion)

    # Refractory period violations
    # Any ISI < 2 ms (absolute refractory period) indicates a model implementation
    # error: the spike-reset mechanism is not enforcing refractoriness correctly.
    # Any value > 0 should be treated as a CRITICAL health issue.
    fraction_refractory_violations: float  # Fraction of ISIs < 2 ms

    # CV₂ — local ISI irregularity
    # Computed as mean(2|ISI_{n+1}−ISI_n| / (ISI_{n+1}+ISI_n)) over consecutive ISI pairs.
    # Unlike global CV, CV₂ is insensitive to rate non-stationarity caused by adaptation.
    # Healthy cortical AI state: CV₂ ≈ 0.8–1.2; regular bursting drives it higher.
    # NaN when fewer than 3 ISI pairs are available.
    isi_cv2: float

    # Fraction of ISIs below 80 ms
    # Used for DA burst-mode detection: when isi_cv > 1 and fraction_isi_lt_80ms > 0.2
    # the population is in reward-context burst firing rather than tonic pacemaker mode.
    fraction_isi_lt_80ms: float

    # DA burst-event rate
    # Events are defined as ≥3 consecutive ISIs < 80 ms followed by a pause ISI > 200 ms.
    # Rate is expressed per second of simulation.
    da_burst_events_per_s: float

    # Spike-frequency adaptation index
    # Ratio of mean FR in the first 25 % of the recording window to the mean FR
    # in the last 25 %.  sfa_index > 1 → adapting (expected for pyramidal cells,
    # relay neurons, MSNs); sfa_index ≈ 1 → non-adapting (PV, FSI, TAN).
    # NaN when fewer than 4 total spikes are recorded.
    sfa_index: float

    # Variability / synchrony metrics
    # Two distinct Fano-factor fields reflecting their different semantics:
    #
    # per_neuron_ff — mean of per-neuron var/mean spike count
    #     in 50 ms bins, averaged over up to 50 sampled neurons.  Unaffected by
    #     between-neuron correlations (Poisson ≈ 1).
    per_neuron_ff: float
    pairwise_correlation: float  # Mean Pearson r across neuron pairs in 100 ms bins
    fraction_burst_events: float # Fraction of 20 ms windows where >30% of pop spikes (epilepsy marker)

    # Per-neuron FR histogram (for CDF plots)
    fr_histogram: np.ndarray       # [20] bins from 0 to max_fr_hz
    fr_histogram_edges: np.ndarray # [21] bin edges in Hz

    # Biological reference
    bio_range_hz: Optional[Tuple[float, float]]  # (min, max) or None

    # ── Network State Classifier ──────────────────────────────────────
    # Joint diagnosis from CV / FF / pairwise ρ:
    #   "AI"    Asynchronous-Irregular  CV 0.7–1.3, FF ~1, ρ < 0.15
    #   "SR"    Synchronous-Regular     CV < 0.5,   FF < 0.5, ρ > 0.3
    #   "SI"    Synchronous-Irregular   CV > 1.3,   FF > 2, ρ > 0.2
    #   "burst" Epileptiform            fraction_burst_events > 0.05 or CV > 2
    #   "unknown" insufficient data
    network_state: str = "unknown"

    # ── Up/Down State Detection ────────────────────────────
    # Sarle's bimodality coefficient BC = (skewness² + 1) / kurtosis applied to
    # the sampled voltage distribution.  BC > 0.555 → bimodal (likely up/down).
    # NaN when fewer than 20 voltage samples are available.
    voltage_bimodality: float = float("nan")

    # Mean duration of up-state and down-state epochs (ms).  Derived from
    # run-length analysis on the mean population voltage binarised at its grand
    # mean.  Healthy cortical slow oscillations: up ≈ 300–500 ms,
    # down ≈ 200–400 ms.  NaN when voltage_bimodality ≤ 0.555,
    # or when insufficient voltage samples exist.
    up_state_duration_ms: float = float("nan")
    down_state_duration_ms: float = float("nan")

    # ── Spatial FR heterogeneity ──────────────────────────────────────
    # Computed as a property; see fr_cv_across_neurons below.

    # ── SFA time constant (τ_sfa) ─────────────────────────────────────
    # Time constant (ms) of an exponential decay fit to the binned firing-rate
    # trajectory: FR(t) = FR_ss + (FR_0 − FR_ss)·exp(−t/τ_sfa).
    # NaN when fit fails, fewer than 10 rate bins, or total_spikes < 20.
    sfa_tau_ms: float = float("nan")

    # ── Two-compartment apical conductance (TwoCompartmentLIF only) ──
    # Mean sampled g_E_apical conductance (AMPA, apical/distal dendrite).
    # NaN for ConductanceLIF populations.
    # Complementary to the basal AMPA captured in RegionStats.mean_g_exc.
    # A near-zero apical mean during a waking-state pattern indicates the
    # apical compartment is not being driven by top-down inputs.
    mean_g_exc_apical: float = float("nan")

    # ── Fano factor scaling across bin widths ────────
    # List of (bin_ms, fano_factor) tuples at multiple time scales [10, 20, 50,
    # 100, 200, 500] ms.  A Poisson process yields FF≈1 at all scales; bursty
    # or correlated activity shows FF increasing with bin width.
    # Empty list when < 50 total spikes.
    fano_scaling: List[Tuple[float, float]] = field(default_factory=list)

    # ── Pairwise correlation distribution ────────────
    # Full array of all sampled pairwise Pearson r values (up to C(30,2) pairs).
    # Used for histogram plotting to assess the shape of the correlation
    # distribution (should be peaked near zero for healthy AI state).
    # None when < 2 neurons, or insufficient data.
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
        """Coefficient of variation of per-neuron firing rates (spatial heterogeneity).

        Healthy AI state for cortical pyramidal cells: FR-CV ≈ 1.4–2.0 (log-normal
        distribution, small fraction of neurons carry most spikes).
        FR-CV < 0.5 → artificially homogeneous (uniform parameter set).
        FR-CV > 4.0 → winner-take-all pathology.
        NaN when mean_fr_hz ≈ 0.
        """
        if self.mean_fr_hz < 1e-9:
            return float("nan")
        return self.std_fr_hz / self.mean_fr_hz


@dataclass
class RegionStats:
    """Statistics aggregated across all populations in a region."""

    region_name: RegionName
    populations: Dict[PopulationName, PopulationStats]
    mean_fr_hz: float
    total_spikes: int

    # E/I balance (conductance-based where available, else NaN)
    mean_g_exc: float   # AMPA conductance only
    mean_g_nmda: float  # NMDA conductance (NaN when not sampled)
    mean_g_inh: float   # GABA-A + GABA-B conductance (combined; used for ei_ratio)
    # Separate GABA subtypes so the GABA-B/GABA-A ratio can be checked
    # independently — GABA-B (K⁺, τ≈100–200 ms) ≠ GABA-A (Cl⁻, τ≈10–20 ms).
    mean_g_gaba_a: float = float("nan")  # GABA-A conductance (NaN when not sampled)
    mean_g_gaba_b: float = float("nan")  # GABA-B conductance (NaN when not sampled)

    # ── D1/D2 competition index (striatal regions only) ───────────────
    # Pearson correlation between D1 and D2 MSN population rates.
    # Healthy mutual inhibition → negative/near-zero.  Positive (> 0.3) means
    # both pathways are co-activated → no action-selection competition.
    # NaN for non-striatal regions or when D1/D2 populations are not present.
    #
    # Two bin widths are reported:
    #   d1_d2_competition_index_200ms — 200 ms bins (Mink 1996 action-selection epoch)
    #   d1_d2_competition_index_50ms — 50 ms bins (striatal mutual-inhibition timescale)
    # The 200 ms measure can mask within-window alternation (D1 fires first half,
    # D2 fires second half → correlation near zero despite correct competition).
    # The 50 ms measure catches this sub-second structure.
    d1_d2_competition_index_200ms: float = float("nan")
    d1_d2_competition_index_50ms: float = float("nan")

    # ── E/I lag cross-correlation ────────────────────
    # Peak cross-correlation between excitatory (AMPA) and inhibitory (GABA-A)
    # conductance time series, and the lag in ms at which it occurs.
    # Healthy cortex: inhibition tracks excitation with 1–5 ms lag (feedforward
    # inhibition) or 5–15 ms lag (feedback inhibition).  Near-zero lag suggests
    # common drive; negative lag or very long lag (>20 ms) indicates pathology.
    # NaN when conductance samples are unavailable.
    ei_lag_ms: float = float("nan")
    ei_xcorr_peak: float = float("nan")

    @property
    def ei_ratio(self) -> float:
        """Excitatory (AMPA + NMDA) to inhibitory (GABA-A + GABA-B) conductance ratio."""
        exc_total = self.mean_g_exc
        if not np.isnan(self.mean_g_nmda):
            exc_total = exc_total + self.mean_g_nmda
        if self.mean_g_inh > 0:
            return exc_total / self.mean_g_inh
        return float("nan")

    @property
    def is_active(self) -> bool:
        return self.mean_fr_hz > 0.1  # > 0.1 Hz average

    @property
    def has_pathological_populations(self) -> bool:
        for p in self.populations.values():
            if p.is_silent or p.is_hyperactive:
                return True
        return False


@dataclass
class OscillatoryStats:
    """Spectral and coherence analysis."""

    # Per-region power spectra (region → band → normalised power)
    region_band_power: Dict[RegionName, Dict[str, float]]

    # Per-region dominant frequency (Hz) and band name
    region_dominant_freq: Dict[RegionName, float]
    region_dominant_band: Dict[RegionName, str]

    # Cross-regional coherence matrices — [n_regions × n_regions]
    # theta (4–8 Hz): hippocampal–septal–cortical timing
    # beta  (13–30 Hz): prefrontal–striatum motor loops
    # gamma (30–100 Hz): thalamocortical sensory binding
    coherence_theta: np.ndarray
    coherence_beta:  np.ndarray
    coherence_gamma: np.ndarray
    region_order: List[RegionName]  # Row/column labels for coherence matrices

    # Global oscillation (from weighted sum of all regions)
    global_dominant_freq_hz: float
    global_band_power: Dict[str, float]

    # Frequency resolution of the population-rate PSD (E9)
    # = sampling_rate / nperseg.  NaN when fewer than 8 rate bins are available.
    # A value > 1 Hz indicates the simulation was too short for reliable spectral analysis.
    freq_resolution_hz: float = float("nan")

    # Theta–gamma phase–amplitude coupling per hippocampal region (E7)
    # Key = region_name; value = Mean Vector Length (MVL) modulation index ∈ [0, 1].
    # NaN when the simulation is < 500 ms or the sampling rate is too low for gamma.
    pac_modulation_index: Dict[str, float] = field(default_factory=dict)

    # Normalised HFO (100–250 Hz) band power for hippocampal CA1 populations (E12).
    # Key = region_name; value = HFO / broadband power ratio.
    # Only populated when dt_ms ≤ 1.0 and simulation ≥ 200 ms.
    hfo_band_power: Dict[str, float] = field(default_factory=dict)

    # ── SWR CA3→CA1 temporal coupling ────────────────────────────────────────
    # Cross-correlation of Gaussian-smoothed (σ=5 ms) CA3 and CA1 spike-count
    # signals at causal lags 10–30 ms (physiological CA3 sharp-wave→CA1 ripple
    # delay; Buzsáki 2015).  Only populated for hippocampal regions containing
    # both CA3 and CA1 populations when dt_ms ≤ 1.0 and simulation ≥ 200 ms.
    # Inner dict keys:
    #   ca3_ca1_xcorr_peak — peak Pearson cross-correlation in the 10–30 ms
    #                         causal window (CA3 leading CA1)
    #   ca3_ca1_lag_ms     — lag in ms at which the peak occurs
    # Healthy: xcorr_peak ≥ 0.05 when hfo_band_power > 0.02 → genuine SWR cascade.
    # Uncoupled HFO (xcorr_peak < 0.05) → noise or gamma artefact in CA1.
    # Reference: Buzsáki 2015 "Hippocampal sharp wave-ripple" Prog Neurobiol.
    swr_ca3_ca1_coupling: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # ── CA3→CA1 theta-sequence coupling ────────────────────────────────────
    # Cross-correlogram peak lag between Gaussian-smoothed (sigma=25 ms)
    # CA3 and CA1 pyramidal spike-count signals at causal lags 5–30 ms.
    # Within each theta cycle, CA3 place cells should activate before CA1
    # (Foster & Wilson 2007; Dragoi & Buzsáki 2006).  A peak lag in 5–30 ms
    # indicates healthy Schaffer-collateral feedforward timing; near-zero or
    # negative peak lag indicates disrupted sequence compression.
    # Populated for hippocampal regions with both CA3 and CA1 populations
    # when simulation ≥ 500 ms.  Empty when populations are absent.
    # Inner dict keys:
    #   xcorr_peak  — peak Pearson cross-correlation in the 5–30 ms causal window
    #   peak_lag_ms — lag in ms at which that peak occurs
    # Healthy: xcorr_peak ≥ 0.05 at lag 5–30 ms.
    # Disrupted: xcorr_peak < 0.05 or peak_lag_ms ≤ 0 ms.
    # References: Foster & Wilson 2007 *Nature*;
    #             Dragoi & Buzsáki 2006 *Cell*.
    ca3_ca1_theta_sequence: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # ── Phase-Locking Value (PLV) — hippocampal spike–theta coupling ──
    # |mean(exp(j·θ_spike))| where θ_spike is the instantaneous theta phase at
    # each CA1 pyramidal spike.  NaN when < 10 spikes or fs low.
    # Healthy: CA1 pyramidal PLV ≈ 0.15–0.40; medial septum GABA PLV ≈ 0.50–0.80.
    plv_theta: Dict[str, float] = field(default_factory=dict)
    # True when the MS GABA reference was absent and the region's own spike train
    # was used as the phase reference (circular → inflated PLV; interpret with caution).
    plv_theta_used_fallback: Dict[str, bool] = field(default_factory=dict)

    # ── Spike avalanche / criticality ─────────────────────────────────
    # Power-law exponent of the avalanche-size distribution (log-log slope).
    # Critical cortex: exponent ≈ −1.5.  Super-critical (epileptic) > −1;
    # sub-critical (down-regulated) < −2.  NaN when the run is too short
    # (< 2000 timesteps) or fewer than 20 avalanche events were detected.
    avalanche_exponent: float = float("nan")
    avalanche_r2: float = float("nan")  # R² of the log-log linear fit
    # Branching ratio σ (Beggs & Plenz 2003): pooled ratio of total descendant
    # spikes to total ancestor spikes across all consecutive active bins.
    # σ ≈ 1.0 → critical (maximal information transfer)
    # σ < 1.0 → subcritical (exponential decay; dominant in healthy rest)
    # σ > 1.0 → supercritical (runaway, seizure-like)
    # NaN when the run is too short (< 2000 timesteps) or fewer than 4 active-bin pairs.
    avalanche_branching_ratio: float = float("nan")

    # ── Cerebellar timing metrics ─────────────────────────────────────
    # Purkinje–DCN population-rate Pearson correlation per cerebellar region.
    # Expected: negative (DCN rebounds during Purkinje silence, ~anti-phase).
    # Positive correlation → Purkinje inhibition is not suppressing DCN.
    purkinje_dcn_corr: Dict[str, float] = field(default_factory=dict)
    # IO pairwise spike-count correlation (gap junction-coupled synchrony).
    # Expected ρ > 0.3 even at 0.3–3 Hz.  Low ρ → gap junctions absent/broken.
    io_pairwise_corr: Dict[str, float] = field(default_factory=dict)

    # ── Beta burst analysis (BG and motor cortex regions) ───────────────────
    # Per-region beta burst statistics detected via Hilbert envelope threshold
    # at 75th percentile.  Only populated for BG/motor regions (stn, gpe, gpi,
    # striatum, cortex_motor, putamen, caudate) when rate_bin_ms ≤ 16 ms.
    # Inner dict keys:
    #   n_bursts         — number of bursts detected (≥ 100 ms minimum duration)
    #   mean_duration_ms — mean burst duration in ms
    #   max_duration_ms  — maximum burst duration in ms
    #   mean_ibi_ms      — mean inter-burst interval in ms (NaN when ≤ 1 burst)
    # Healthy: mean < 200 ms, max < 400 ms (Tinkhauser et al. 2017).
    # Pathological: max > 500 ms → STN–GPe loop-driven Parkinsonian synchrony.
    beta_burst_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # ── Firing-rate autocorrelation time constant (τ_int) ─────────────────
    # Exponential decay time constant (ms) of the population-rate
    # autocorrelation for each region, fit over lags 1–500 ms.
    # PFC:            τ ≈ 100–400 ms (Murray et al. 2014)
    # Motor cortex:   τ ≈ 40–150 ms
    # Sensory cortex: τ ≈ 20–50 ms
    # NaN when the simulation is < 1 s, the fit fails, or R² < 0.5.
    region_integration_tau_ms: Dict[str, float] = field(default_factory=dict)

    # ── Thalamic relay burst mode (LTS / T-channel) ───────────────────────
    # Per-region short-ISI fraction for relay populations.
    # Short ISI defined as < 15 ms (characteristic LTS burst doublets/triplets).
    # Key = region name; value = fraction of all relay-cell ISIs < 15 ms.
    #
    # Burst mode: fraction ≥ 0.05 (≥ 5 % short ISIs → LTS active)
    # Tonic mode: fraction < 0.02 (< 2 % short ISIs → Poisson-like)
    #
    # Reference: McCormick & Huguenard 1992 "Properties of the T-type Ca2+
    # current in relay neurons of the thalamus" J Neurophysiol.
    relay_burst_mode: Dict[str, float] = field(default_factory=dict)

    # ── Cortical laminar cascade latencies ────────────────────────────────
    # Mean first-spike latency (ms) per cortical layer after thalamic relay
    # volleys.  Key = cortical region name; inner dict keys:
    #   l4_lat_ms  — mean latency to first L4 pyramidal spike post-volley
    #   l23_lat_ms — mean latency to first L2/3 pyramidal spike post-volley
    #   l5_lat_ms  — mean latency to first L5 pyramidal spike post-volley
    #   l6_lat_ms  — mean latency to first L6 pyramidal spike post-volley
    # Expected feedforward order: l4_lat_ms < l23_lat_ms < l5_lat_ms.
    # Empty when no cortical or thalamic relay populations are present.
    laminar_cascade_latencies: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class ConnectivityStats:
    """Axonal tract transmission statistics."""

    @dataclass
    class TractStats:
        synapse_id: SynapseId
        spikes_sent: int
        transmission_ratio: float  # Fraction of timesteps source was active
        is_functional: bool
        # Cross-correlation delay verification
        measured_delay_ms: float   # Peak lag of source→target cross-corr (causal window)
        expected_delay_ms: float   # From axonal tract spec
        # Anti-causal peak: xcorr peak in the negative-lag window.
        # If this exceeds the causal peak, the connection is likely reversed.
        anticausal_peak_ms: float  # NaN when not computed

    tracts: List[TractStats]
    n_functional: int
    n_broken: List["ConnectivityStats.TractStats"]


@dataclass
class HomeostaticStats:
    """Homeostatic gain and STP state over time."""

    # g_L_scale trajectories per population: key = "region:pop" → 1D array of gains
    gain_trajectories: Dict[str, np.ndarray]
    gain_sample_times_ms: np.ndarray  # Timestep-to-ms for x-axis

    # STP efficacy (x·u) trajectories per synapse: key = str(synapse_id) → 1D array
    # Sampled at the same interval as homeostatic gains (gain_sample_interval_ms).
    # Allows verifying that STP converges to a stable efficacy rather than drifting.
    stp_efficacy_history: Dict[str, np.ndarray]

    # STP state at end of recording: synapse_id_str → {"mean_x": ..., "mean_u": ...}
    stp_final_state: Dict[str, Dict[str, float]]


@dataclass
class HealthIssue:
    """A single health issue with severity, source category, and message.

    Collected by ``assess_health`` during diagnostic evaluation.  The
    ``category`` field identifies which checker produced the issue, enabling
    deduplication and priority-based filtering without string parsing.
    """

    severity: Literal["critical", "warning", "info"]
    category: HealthCategory
    message: str
    population: Optional[str] = None   # "region:pop" key, when applicable
    region: Optional[str] = None       # region name, when applicable


@dataclass
class HealthReport:
    """Per-population and global health assessment."""

    # Global
    is_healthy: bool
    stability_score: float          # 0–1
    critical_issues: List[str]
    warnings: List[str]

    # Per-population biological plausibility
    population_status: Dict[str, str]  # "region:pop" → "ok" | "low" | "high" | "unknown"

    # Summary counts
    n_populations_ok: int
    n_populations_low: int
    n_populations_high: int
    n_populations_unknown: int

    # Structured issue list — allows filtering by category/severity without
    # parsing the flat critical_issues / warnings string lists.
    all_issues: List[HealthIssue] = field(default_factory=list)

    # Global brain-state summary — a slash-separated three-part label:
    #   "<dominant_state>/<dominant_oscillation>/<criticality>"
    # e.g. "AI/theta/sub-crit"  or  "SI/beta/supercrit"
    #
    # dominant_state:    majority-vote of adaptation_expected=True populations'
    #                    network_state ("AI", "SR", "SI", "burst", "unknown")
    # dominant_oscillation: global dominant EEG band from OscillatoryStats,
    #                    e.g. "theta", "beta", "gamma", "none"
    # criticality:       derived from avalanche branching ratio σ:
    #                    "supercrit" (σ>1.05), "critical" (0.9<σ≤1.05),
    #                    "sub-crit"  (σ≤0.9),  "unknown" (no avalanche data)
    global_brain_state: str = "unknown/none/unknown"


@dataclass
class DiagnosticsReport:
    """Complete diagnostics report."""

    # Meta
    timestamp: float
    simulation_time_ms: float
    n_timesteps: int

    # Core data
    regions: Dict[RegionName, RegionStats]
    oscillations: OscillatoryStats
    connectivity: ConnectivityStats
    homeostasis: HomeostaticStats
    health: HealthReport

    # Number of steps detected as onset transient and excluded from steady-state
    # analyses (population stats, binned rates, oscillations).  Set automatically
    # by detect_transient_step() in analyze().  0 when T < 200 or detection fails.
    transient_steps: int = 0

    # Neuromodulator receptor concentration trajectories (both modes):
    # key = "{region}/{receptor_attr}" → 1D float32 array, length = n_gain_steps
    neuromodulator_levels: Optional[Dict[str, np.ndarray]] = None

    # Raw traces kept for plotting
    # shape: [T, n_pops] – spike counts per ms per population
    raw_spike_counts: Optional[np.ndarray] = None
    # shape: [T_cond, n_pops, sample_size] – sampled voltages
    raw_voltages: Optional[np.ndarray] = None
    # Timestamps for voltage/conductance samples
    voltage_sample_times_ms: Optional[np.ndarray] = None
    conductance_sample_times_ms: Optional[np.ndarray] = None
    # g_exc / g_inh traces
    raw_g_exc: Optional[np.ndarray] = None
    raw_g_inh: Optional[np.ndarray] = None
    # Population-level firing rates (binned): [n_bins, n_pops]
    # READ-ONLY — shared with PAC, coherence, and plotting code.  Do not modify in-place.
    pop_rate_binned: Optional[np.ndarray] = field(default=None, repr=False)
    # Region-level firing rates (binned, sum across populations): [n_bins, n_regions]
    region_rate_binned: Optional[np.ndarray] = None
    # Population and region index
    pop_keys: Optional[List[Tuple[str, str]]] = None
    region_keys: Optional[List[str]] = None

    # ── Effective synaptic gain per tract ─────────────────────────────
    # Estimated as Pearson correlation between pre-region and post-region
    # binned firing rate time series at the causal lag matching the tract's
    # axonal delay.  Keys are synapse_id strings (e.g. "PFC→striatum_dorsal").
    # Healthy range: 0.05–0.60; values near 0 indicate ineffective connections;
    # values > 0.8 suggest pathological synchrony.
    effective_synaptic_gain: Optional[Dict[str, float]] = None
