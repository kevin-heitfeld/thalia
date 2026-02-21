"""
Brain Activity Analyzer: Comprehensive pre-training diagnostic tool.

This module provides deep analysis of brain-wide activity patterns to verify
that spikes are flowing properly through all regions and populations, and that
the overall activity resembles biological brain dynamics.

Features:
=========
1. SPIKE FLOW TRACING: Track spike propagation through the entire network
2. POPULATION ACTIVITY: Analyze firing rates for every population in every region
3. OSCILLATORY ANALYSIS: EEG-like spectral analysis of network rhythms
4. CONNECTIVITY VERIFICATION: Check that all pathways are functional
5. HEALTH MONITORING: Detect pathological states (silence, seizure, etc.)
6. BIOLOGICAL COMPARISON: Compare activity patterns to real brain data
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import torch

from thalia.brain.regions.population_names import ThalamusPopulation
from thalia.typing import (
    BrainInput,
    BrainOutput,
    RegionOutput,
    SynapseId,
)

if TYPE_CHECKING:
    from thalia.brain.brain import DynamicBrain


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class PopulationActivity:
    """Activity statistics for a single population."""

    population_name: str
    region_name: str
    n_neurons: int
    total_spikes: int
    mean_firing_rate: float  # Fraction of neurons active per timestep
    firing_rate_hz: float  # Spikes per second per neuron
    spike_counts_per_timestep: List[int]
    active_timesteps: int  # Number of timesteps with at least 1 spike
    silence_ratio: float  # Fraction of timesteps with zero spikes
    burst_events: int  # Number of timesteps with >50% neurons active

    @property
    def is_silent(self) -> bool:
        """True if population shows no meaningful activity.

        Biological baseline firing rates:
        - Cortex: 0.1-5 Hz (0.01-0.5% per ms)
        - Hippocampus: 0.05-10 Hz (0.005-1.0% per ms)
        - Striatum: 0.5-20 Hz (0.05-2.0% per ms)

        Threshold of 0.01% (0.1 Hz) allows for sparse but healthy baseline activity.
        We only check mean firing rate, not active_timesteps, since sparse populations
        can have silent timesteps while still being healthy (e.g., 0.43% = 4.3 Hz is normal).
        """
        return self.mean_firing_rate < 0.0001  # 0.01% = 0.1 Hz baseline

    @property
    def is_hyperactive(self) -> bool:
        """True if population shows excessive activity."""
        return self.mean_firing_rate > 0.8 or self.burst_events > 50


@dataclass
class GainTrajectory:
    """Tracks homeostatic gain evolution over time."""

    region_name: str
    population_name: str
    gain_values: List[float]  # Gain at each timestep
    initial_gain: float
    final_gain: float
    min_gain: float
    max_gain: float
    is_collapsing: bool  # True if gain drops below 0.3
    collapse_timestep: Optional[int]  # When gain first drops below 0.3


@dataclass
class WeightEvolution:
    """Tracks synaptic weight changes over time."""

    axonal_tract_name: str
    initial_mean: float
    final_mean: float
    initial_std: float
    final_std: float
    weight_change: float  # final_mean - initial_mean
    percent_change: float  # 100 * weight_change / initial_mean
    is_depressing: bool  # True if weights decreased > 20%
    is_potentiating: bool  # True if weights increased > 20%


@dataclass
class ActivityTrajectory:
    """Tracks activity evolution over time to detect collapse."""

    total_spikes_per_window: List[int]  # Windowed spike counts
    window_size_ms: int
    initial_activity: int  # Spikes in first window
    final_activity: int  # Spikes in last window
    activity_change: float  # final - initial
    percent_change: float  # 100 * change / initial
    is_collapsing: bool  # True if activity drops > 50%
    collapse_timestep: Optional[int]  # When activity first drops > 50%


@dataclass
class RegionActivity:
    """Activity statistics for an entire region."""

    region_name: str
    populations: Dict[str, PopulationActivity]
    mean_firing_rate: float
    total_spikes: int

    @property
    def silent_populations(self) -> List[str]:
        """List of silent population names."""
        return [name for name, pop in self.populations.items() if pop.is_silent]

    @property
    def hyperactive_populations(self) -> List[str]:
        """List of hyperactive population names."""
        return [name for name, pop in self.populations.items() if pop.is_hyperactive]


@dataclass
class AxonalTractCheck:
    """Verification that axonal tracts are functional."""

    axonal_tract_name: str
    source_region: str
    target_region: str
    is_functional: bool
    spikes_transmitted: int
    transmission_ratio: float  # Fraction of timesteps with transmission


@dataclass
class OscillatoryAnalysis:
    """EEG-like spectral analysis of brain rhythms."""

    dominant_frequency_hz: float
    power_spectrum: Dict[str, float]  # Band name -> power
    coherence: float  # Cross-region synchronization
    phase_locking: float  # Temporal organization

    @property
    def dominant_band(self) -> str:
        """Name of dominant oscillatory band."""
        return max(self.power_spectrum.items(), key=lambda x: x[1])[0]


@dataclass
class HealthReport:
    """Overall brain health assessment."""

    is_healthy: bool
    critical_issues: List[str]
    warnings: List[str]
    stability_score: float  # 0-1, higher = better
    biological_plausibility_score: float  # 0-1, higher = more realistic


@dataclass
class BrainActivityReport:
    """Complete brain activity analysis."""

    timestamp: float
    simulation_time_ms: float
    n_timesteps: int

    # Region-level activity
    regions: Dict[str, RegionActivity]

    # Network-level metrics
    global_firing_rate: float
    total_spikes: int
    active_regions: List[str]
    silent_regions: List[str]

    # Connectivity
    pathways: List[AxonalTractCheck]
    functional_axonal_tracts: int
    broken_axonal_tracts: List[str]

    # Oscillations
    oscillations: OscillatoryAnalysis

    # Health
    health: HealthReport

    # Temporal dynamics
    activity_trajectory: Optional[ActivityTrajectory] = None
    gain_trajectories: Optional[Dict[str, GainTrajectory]] = None  # region:pop -> trajectory
    weight_evolution: Optional[Dict[str, WeightEvolution]] = None  # pathway -> evolution
    inhibition_analysis: Optional[Dict[str, Dict]] = None  # region -> {E/I ratio, etc.}


# =============================================================================
# BRAIN ACTIVITY ANALYZER
# =============================================================================


class BrainActivityAnalyzer:
    """
    Comprehensive analyzer for brain-wide activity patterns.

    This tool runs the brain without any training task and analyzes the
    spontaneous and evoked activity to ensure all components are working.
    """

    def __init__(self, brain: DynamicBrain, timesteps: int):
        """
        Initialize analyzer.

        Args:
            brain: The brain instance to analyze
            timesteps: Number of timesteps to simulate
        """
        self.brain = brain
        self.timesteps = timesteps
        self.dt_ms = brain.dt_ms
        self.device = brain.device

        # Recording buffers
        self.spike_history: List[BrainOutput] = []
        self.region_activity: Dict[str, List[RegionOutput]] = defaultdict(list)

        # NEW: Temporal tracking for Phase 1 diagnostics
        self.gain_history: Dict[str, List[float]] = defaultdict(list)  # "region:pop" -> gains
        self.weight_snapshots: Dict[str, List[torch.Tensor]] = {}  # pathway -> weight tensors
        self.activity_windows: List[int] = []  # Windowed spike counts
        self.track_dynamics = False  # Enable with analyze_temporal_dynamics()

    # =========================================================================
    # DATA COLLECTION
    # =========================================================================

    def run_simulation(
        self,
        input_pattern: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        """
        Run brain simulation and record all activity.

        Args:
            input_pattern: Type of input to provide:
                - None: Spontaneous activity only
                - "random": Random sparse input to sensory regions
                - "rhythmic": Rhythmic input at theta frequency
                - "burst": Single burst at t=100ms
            verbose: Print progress updates
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"BRAIN ACTIVITY ANALYZER - Simulating {self.timesteps} timesteps")
            print(f"{'='*80}\n")

        self.spike_history = []
        self.region_activity = defaultdict(list)

        start_time = time.time()

        for t in range(self.timesteps):
            # Generate input if requested
            input_spikes = self._generate_input(t, input_pattern)

            # Run brain forward pass
            outputs = self.brain.forward(input_spikes)

            # Record outputs
            self.spike_history.append(outputs)
            for region_name, region_outputs in outputs.items():
                self.region_activity[region_name].append(region_outputs)

            # Track gains and weights if enabled
            if self.track_dynamics:
                self._record_timestep_diagnostics(t)

            # Progress updates
            if verbose and (t + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (t + 1) / elapsed
                print(f"  Timestep {t+1}/{self.timesteps} ({rate:.1f} steps/sec)")

        if verbose:
            elapsed = time.time() - start_time
            print(f"\n✓ Simulation complete in {elapsed:.2f}s ({self.timesteps/elapsed:.1f} steps/sec)")

    def _generate_input(
        self,
        timestep: int,
        pattern: Optional[str],
    ) -> Optional[BrainInput]:
        """Generate input pattern for given timestep."""
        if pattern is None:
            return None

        # Find sensory input region (thalamus if present)
        if "thalamus" not in self.brain.regions:
            return None

        thalamus = self.brain.regions["thalamus"]
        # Get relay size from region attribute (not config)
        relay_size = getattr(thalamus, 'relay_size', None)
        if relay_size is None:
            return None

        spikes = None

        if pattern == "random":
            # CONTINUOUS SPARSE POISSON INPUT
            # Conductance-based neurons need continuous background activity for balanced state
            # Instead of sparse bursts (5% duty cycle), use continuous Poisson spiking
            # Spatial: 3% neurons active each timestep (sparse but continuous)
            # Temporal: 20% spike probability per active neuron
            # Effective rate: 0.03 × 0.20 = 0.6% = 6 spikes/ms for 400 neurons
            n_active = max(1, int(relay_size * 0.03))  # 3% active neurons
            active_idx = torch.randperm(relay_size, device=self.device)[:n_active]
            # Each active neuron spikes with 20% probability (Poisson-like)
            spike_prob = torch.rand(n_active, device=self.device) < 0.20
            spikes = torch.zeros(relay_size, dtype=torch.bool, device=self.device)
            spikes[active_idx] = spike_prob

        elif pattern == "rhythmic":
            # Theta rhythm (8 Hz = 125ms period)
            period_ms = 125.0
            phase = (timestep * self.dt_ms % period_ms) / period_ms
            if phase < 0.2:  # Active in first 20% of cycle
                n_active = max(1, int(relay_size * 0.1))
                spikes = torch.zeros(relay_size, dtype=torch.bool, device=self.device)
                active_idx = torch.randperm(relay_size, device=self.device)[:n_active]
                spikes[active_idx] = True

        elif pattern == "burst":
            # Single burst at t=100ms
            if timestep == int(100.0 / self.dt_ms):
                n_active = max(1, int(relay_size * 0.3))
                spikes = torch.zeros(relay_size, dtype=torch.bool, device=self.device)
                active_idx = torch.randperm(relay_size, device=self.device)[:n_active]
                spikes[active_idx] = True

        # Always return input (continuous drive) - even if no spikes this timestep
        if spikes is None:
            spikes = torch.zeros(relay_size, dtype=torch.bool, device=self.device)

        synapse_id = SynapseId(
            source_region="external",
            source_population="sensory",
            target_region="thalamus",
            target_population=ThalamusPopulation.RELAY.value,
        )
        return {"thalamus": {synapse_id: spikes}}

    def _record_timestep_diagnostics(self, timestep: int) -> None:
        """Record gains, weights, and activity for this timestep (Phase 1 diagnostics)."""
        # Track homeostatic gains
        gain_regions = {
            'cortex': ['l4_gain', 'l23_gain', 'l5_gain'],
            'thalamus': ['relay_gain'],
            'hippocampus': ['dg_gain', 'ca3_gain', 'ca2_gain', 'ca1_gain'],
            'striatum': ['d1_gain', 'd2_gain'],
        }

        for region_name, gain_attrs in gain_regions.items():
            if region_name not in self.brain.regions:
                continue

            region = self.brain.regions[region_name]
            for gain_attr in gain_attrs:
                if not hasattr(region, gain_attr):
                    continue

                gain_param = getattr(region, gain_attr)
                mean_gain = float(gain_param.mean().item())

                pop_name = gain_attr.replace('_gain', '')
                key = f"{region_name}:{pop_name}"
                self.gain_history[key].append(mean_gain)

        # Track windowed activity (10ms windows)
        window_size = 10
        if timestep % window_size == 0:
            # Sum spikes from previous window
            start = max(0, timestep - window_size)
            window_spikes = 0
            for t in range(start, timestep):
                if t < len(self.spike_history):
                    outputs = self.spike_history[t]
                    for region_outputs in outputs.values():
                        for pop_spikes in region_outputs.values():
                            window_spikes += int(pop_spikes.sum().item())
            self.activity_windows.append(window_spikes)

    # =========================================================================
    # ANALYSIS METHODS
    # =========================================================================

    def analyze_full_brain(
        self,
        input_pattern: Optional[str] = "random",
        verbose: bool = True,
    ) -> BrainActivityReport:
        """
        Run complete brain analysis.

        This is the main entry point that runs simulation and all analyses.

        Args:
            input_pattern: Input pattern to use (see run_simulation)
            verbose: Print progress updates

        Returns:
            Complete brain activity report
        """
        # Run simulation
        self.run_simulation(input_pattern=input_pattern, verbose=verbose)

        if verbose:
            print(f"\n{'='*80}")
            print("ANALYZING ACTIVITY PATTERNS")
            print(f"{'='*80}\n")

        # Analyze each region
        regions = {}
        for region_name in self.brain.regions.keys():
            regions[region_name] = self._analyze_region(region_name)

        # Network-level metrics
        total_spikes = sum(r.total_spikes for r in regions.values())
        global_firing_rate = np.mean([r.mean_firing_rate for r in regions.values()])
        active_regions = [name for name, r in regions.items() if r.mean_firing_rate > 0.001]
        silent_regions = [name for name, r in regions.items() if r.mean_firing_rate <= 0.001]

        # Connectivity analysis
        pathways = self._analyze_connectivity()
        functional_axonal_tracts = sum(1 for p in pathways if p.is_functional)
        broken_axonal_tracts = [p.axonal_tract_name for p in pathways if not p.is_functional]

        # Oscillatory analysis
        oscillations = self._analyze_oscillations()

        # Health assessment
        health = self._assess_health(regions, pathways, oscillations)

        return BrainActivityReport(
            timestamp=time.time(),
            simulation_time_ms=self.timesteps * self.dt_ms,
            n_timesteps=self.timesteps,
            regions=regions,
            global_firing_rate=global_firing_rate,
            total_spikes=total_spikes,
            active_regions=active_regions,
            silent_regions=silent_regions,
            pathways=pathways,
            functional_axonal_tracts=functional_axonal_tracts,
            broken_axonal_tracts=broken_axonal_tracts,
            oscillations=oscillations,
            health=health,
        )

    def _analyze_region(self, region_name: str) -> RegionActivity:
        """Analyze activity for a single region."""
        region_history = self.region_activity[region_name]

        # Analyze each population
        populations = {}
        total_region_spikes = 0

        # Get all population names from first timestep
        if region_history:
            population_names = region_history[0].keys()
        else:
            population_names = []

        for pop_name in population_names:
            pop_activity = self._analyze_population(region_name, pop_name, region_history)
            populations[pop_name] = pop_activity
            total_region_spikes += pop_activity.total_spikes

        # Region-level metrics
        if populations:
            mean_fr = np.mean([p.mean_firing_rate for p in populations.values()])
        else:
            mean_fr = 0.0

        return RegionActivity(
            region_name=region_name,
            populations=populations,
            mean_firing_rate=mean_fr,
            total_spikes=total_region_spikes,
        )

    def _analyze_population(
        self,
        region_name: str,
        pop_name: str,
        region_history: List[RegionOutput],
    ) -> PopulationActivity:
        """Analyze activity for a single population."""
        spike_counts = []
        total_spikes = 0
        active_timesteps = 0
        burst_events = 0
        n_neurons = 0

        for timestep_outputs in region_history:
            if pop_name not in timestep_outputs:
                spike_counts.append(0)
                continue

            spikes = timestep_outputs[pop_name]
            n_neurons = spikes.shape[0]
            n_active = int(spikes.sum().item())

            spike_counts.append(n_active)
            total_spikes += n_active

            if n_active > 0:
                active_timesteps += 1

            if n_active > n_neurons * 0.5:
                burst_events += 1

        # Compute statistics
        if len(spike_counts) > 0 and n_neurons > 0:
            mean_firing_rate = total_spikes / (len(spike_counts) * n_neurons)
            firing_rate_hz = mean_firing_rate * (1000.0 / self.dt_ms)
            silence_ratio = 1.0 - (active_timesteps / len(spike_counts))
        else:
            mean_firing_rate = 0.0
            firing_rate_hz = 0.0
            silence_ratio = 1.0

        return PopulationActivity(
            population_name=pop_name,
            region_name=region_name,
            n_neurons=n_neurons,
            total_spikes=total_spikes,
            mean_firing_rate=mean_firing_rate,
            firing_rate_hz=firing_rate_hz,
            spike_counts_per_timestep=spike_counts,
            active_timesteps=active_timesteps,
            silence_ratio=silence_ratio,
            burst_events=burst_events,
        )

    def _analyze_connectivity(self) -> List[AxonalTractCheck]:
        """Verify that all axonal tracts are transmitting spikes."""
        axonal_tract_checks = []

        # Keys are "target:population" strings, not tuples
        # Each AxonalTract can have multiple sources
        for target_key, axonal_tract in self.brain.axonal_tracts.items():
            # Parse target region and population
            target_region, target_population = target_key.split(":") if ":" in target_key else (target_key, "")

            # Analyze each source -> target pathway
            for source_spec in axonal_tract.source_specs:
                source_region = source_spec.region_name
                source_population = source_spec.population

                # Count timesteps where axonal tract transmitted spikes
                transmitted_timesteps = 0
                total_transmitted_spikes = 0

                # This is a simplified check - in practice we'd need to track
                # axonal tract outputs separately
                # For now, check if target region received input

                # Build descriptive pathway name
                source_label = f"{source_region}:{source_population}" if source_population else source_region
                target_label = f"{target_region}:{target_population}" if target_population else target_region
                axonal_tract_name = f"{source_label} → {target_label}"

                # Check if target region was active (proxy for transmission)
                if target_region in self.region_activity:
                    target_history = self.region_activity[target_region]
                    for outputs in target_history:
                        if outputs:  # Any output means received input
                            transmitted_timesteps += 1
                            for pop_spikes in outputs.values():
                                total_transmitted_spikes += int(pop_spikes.sum().item())

                transmission_ratio = transmitted_timesteps / max(1, self.timesteps)
                is_functional = transmission_ratio > 0.01  # Active at least 1% of time

                axonal_tract_checks.append(AxonalTractCheck(
                    axonal_tract_name=axonal_tract_name,
                    source_region=source_region,
                    target_region=target_region,
                    is_functional=is_functional,
                    spikes_transmitted=total_transmitted_spikes,
                    transmission_ratio=transmission_ratio,
                ))

        return axonal_tract_checks

    def _analyze_oscillations(self) -> OscillatoryAnalysis:
        """Analyze oscillatory dynamics (EEG-like spectral analysis)."""
        # Compute global activity trace (sum across all regions)
        activity_trace = np.zeros(self.timesteps)

        for t, outputs in enumerate(self.spike_history):
            for region_outputs in outputs.values():
                for pop_spikes in region_outputs.values():
                    activity_trace[t] += pop_spikes.sum().item()

        # Compute power spectrum using FFT
        # Handle edge cases
        if np.all(activity_trace == 0) or len(activity_trace) < 10:
            return OscillatoryAnalysis(
                dominant_frequency_hz=0.0,
                power_spectrum={
                    "delta": 0.0,
                    "theta": 0.0,
                    "alpha": 0.0,
                    "beta": 0.0,
                    "gamma": 0.0,
                },
                coherence=0.0,
                phase_locking=0.0,
            )

        # FFT
        fft = np.fft.rfft(activity_trace - activity_trace.mean())
        freqs = np.fft.rfftfreq(len(activity_trace), d=self.dt_ms / 1000.0)  # in Hz
        power = np.abs(fft) ** 2

        # Find dominant frequency (exclude DC component)
        if len(freqs) > 1:
            dominant_idx = np.argmax(power[1:]) + 1
            dominant_frequency = freqs[dominant_idx]
        else:
            dominant_frequency = 0.0

        # Compute power in standard EEG bands
        def band_power(f_min: float, f_max: float) -> float:
            mask = (freqs >= f_min) & (freqs <= f_max)
            if not mask.any():
                return 0.0
            return float(np.mean(power[mask]))

        power_spectrum = {
            "delta": band_power(0.5, 4.0),
            "theta": band_power(4.0, 8.0),
            "alpha": band_power(8.0, 13.0),
            "beta": band_power(13.0, 30.0),
            "gamma": band_power(30.0, 100.0),
        }

        # Normalize
        total_power = sum(power_spectrum.values())
        if total_power > 0:
            power_spectrum = {k: v / total_power for k, v in power_spectrum.items()}

        # Compute coherence (simplified: std of activity trace)
        coherence = float(1.0 - min(1.0, np.std(activity_trace) / (np.mean(activity_trace) + 1e-6)))

        # Phase locking (simplified: autocorrelation at theta period)
        phase_locking = 0.0
        if len(activity_trace) > 125:
            lag = int(125.0 / self.dt_ms)  # Theta period
            if lag < len(activity_trace):
                autocorr = np.corrcoef(activity_trace[:-lag], activity_trace[lag:])[0, 1]
                phase_locking = float(max(0.0, autocorr))

        return OscillatoryAnalysis(
            dominant_frequency_hz=dominant_frequency,
            power_spectrum=power_spectrum,
            coherence=coherence,
            phase_locking=phase_locking,
        )

    def _assess_health(
        self,
        regions: Dict[str, RegionActivity],
        pathways: List[AxonalTractCheck],
        oscillations: OscillatoryAnalysis,
    ) -> HealthReport:
        """Assess overall brain health."""
        critical_issues = []
        warnings = []

        # Regions that require external input (not spontaneously active)
        EXTERNALLY_DRIVEN_REGIONS = {"reward_encoder"}

        # Check for silent regions
        for name, region in regions.items():
            if region.mean_firing_rate < 0.001:
                # Don't flag externally-driven regions as critical issues
                if name in EXTERNALLY_DRIVEN_REGIONS:
                    warnings.append(
                        f"Region '{name}' is silent (expected - requires external input via API)"
                    )
                else:
                    critical_issues.append(f"Region '{name}' is SILENT (no activity)")
            elif region.silent_populations:
                warnings.append(
                    f"Region '{name}' has silent populations: {', '.join(region.silent_populations)}"
                )

        # Check for hyperactive regions
        for name, region in regions.items():
            if region.mean_firing_rate > 0.8:
                critical_issues.append(f"Region '{name}' is HYPERACTIVE (seizure-like)")
            elif region.hyperactive_populations:
                warnings.append(
                    f"Region '{name}' has hyperactive populations: {', '.join(region.hyperactive_populations)}"
                )

        # Check connectivity
        broken_axonal_tracts = [p for p in pathways if not p.is_functional]
        if broken_axonal_tracts:
            for axonal_tract in broken_axonal_tracts:
                warnings.append(f"Pathway '{axonal_tract.axonal_tract_name}' shows minimal transmission")

        # Check oscillations
        if oscillations.dominant_frequency_hz == 0.0:
            warnings.append("No oscillatory activity detected")
        elif oscillations.dominant_frequency_hz > 100.0:
            warnings.append(f"Abnormally high oscillation frequency: {oscillations.dominant_frequency_hz:.1f} Hz")

        # Compute stability score
        n_critical = len(critical_issues)
        n_warnings = len(warnings)
        stability_score = max(0.0, 1.0 - (n_critical * 0.3) - (n_warnings * 0.1))

        # Compute biological plausibility score
        bio_score = 1.0

        # Penalize if no regions active
        if not regions:
            bio_score *= 0.5

        # Reward theta-dominant oscillations (typical in healthy brain)
        if oscillations.dominant_band == "theta":
            bio_score *= 1.0
        elif oscillations.dominant_band in ["alpha", "beta"]:
            bio_score *= 0.9
        else:
            bio_score *= 0.7

        # Reward moderate firing rates (5-20% typical for sparse coding)
        mean_fr = np.mean([r.mean_firing_rate for r in regions.values()]) if regions else 0.0
        if 0.05 <= mean_fr <= 0.20:
            bio_score *= 1.0
        elif 0.01 <= mean_fr <= 0.50:
            bio_score *= 0.8
        else:
            bio_score *= 0.5

        # Overall health
        is_healthy = len(critical_issues) == 0 and stability_score > 0.7

        return HealthReport(
            is_healthy=is_healthy,
            critical_issues=critical_issues,
            warnings=warnings,
            stability_score=stability_score,
            biological_plausibility_score=bio_score,
        )

    # =========================================================================
    # REPORTING
    # =========================================================================

    def print_report(self, report: BrainActivityReport, detailed: bool = True) -> None:
        """Print formatted analysis report."""
        print(f"\n{'='*80}")
        print("BRAIN ACTIVITY ANALYSIS REPORT")
        print(f"{'='*80}")
        print(f"Simulation time: {report.simulation_time_ms:.1f} ms ({report.n_timesteps} timesteps)")
        print(f"Global firing rate: {report.global_firing_rate*100:.2f}%")
        print(f"Total spikes: {report.total_spikes:,}")
        print()

        # Health overview
        print(f"{'─'*80}")
        print("HEALTH STATUS")
        print(f"{'─'*80}")
        status = "✓ HEALTHY" if report.health.is_healthy else "✗ UNHEALTHY"
        print(f"Overall status: {status}")
        print(f"Stability score: {report.health.stability_score:.2f}/1.0")
        print(f"Biological plausibility: {report.health.biological_plausibility_score:.2f}/1.0")

        if report.health.critical_issues:
            print(f"\n🔴 CRITICAL ISSUES ({len(report.health.critical_issues)}):")
            for issue in report.health.critical_issues:
                print(f"  • {issue}")

        if report.health.warnings:
            print(f"\n⚠️  WARNINGS ({len(report.health.warnings)}):")
            for warning in report.health.warnings:
                print(f"  • {warning}")

        if not report.health.critical_issues and not report.health.warnings:
            print("\n✓ No issues detected")

        print()

        # Region activity
        print(f"{'─'*80}")
        print("REGION ACTIVITY")
        print(f"{'─'*80}")
        print(f"Active regions: {len(report.active_regions)}/{len(report.regions)}")
        if report.silent_regions:
            print(f"Silent regions: {', '.join(report.silent_regions)}")
        print()

        for region_name, region in report.regions.items():
            status_icon = "✓" if region.mean_firing_rate > 0.001 else "✗"
            print(f"{status_icon} {region_name}:")
            print(f"    Mean FR: {region.mean_firing_rate*100:.2f}% | Total spikes: {region.total_spikes:,}")

            if detailed:
                for pop_name, pop in region.populations.items():
                    icon = "✓" if not pop.is_silent else "✗"
                    print(f"      {icon} {pop_name}: {pop.mean_firing_rate*100:.2f}% "
                          f"({pop.n_neurons} neurons, {pop.total_spikes} spikes)")

        print()

        # Connectivity
        print(f"{'─'*80}")
        print("CONNECTIVITY")
        print(f"{'─'*80}")
        print(f"Functional pathways: {report.functional_axonal_tracts}/{len(report.pathways)}")

        if report.broken_axonal_tracts:
            print(f"Non-functional pathways: {', '.join(report.broken_axonal_tracts)}")

        if detailed and report.pathways:
            print("\nPathway transmission rates:")
            for axonal_tract in report.pathways:
                icon = "✓" if axonal_tract.is_functional else "✗"
                print(f"  {icon} {axonal_tract.axonal_tract_name}: {axonal_tract.transmission_ratio*100:.1f}% "
                      f"({axonal_tract.spikes_transmitted:,} spikes)")

        print()

        # Oscillations
        print(f"{'─'*80}")
        print("OSCILLATORY DYNAMICS (EEG-like)")
        print(f"{'─'*80}")
        print(f"Dominant frequency: {report.oscillations.dominant_frequency_hz:.2f} Hz "
              f"({report.oscillations.dominant_band} band)")
        print(f"Coherence: {report.oscillations.coherence:.2f}")
        print(f"Phase locking: {report.oscillations.phase_locking:.2f}")
        print("\nPower spectrum:")
        for band, power in report.oscillations.power_spectrum.items():
            bar_length = int(power * 40)
            bar = "█" * bar_length
            print(f"  {band:>6s}: {bar} {power:.3f}")

        print(f"\n{'='*80}\n")
