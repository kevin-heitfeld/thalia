"""
Region Test Runner — isolated single-region simulation for rapid tuning.

Runs one brain region in isolation with user-specified Poisson spike inputs,
without needing to build and simulate the full brain.  Useful for:

- Diagnosing silent or overactive populations before a full diagnostic run
- Tuning individual region parameters (baseline_drive, weight_scale, etc.)
- Verifying that a region fires in its target rate band given realistic inputs
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch

from thalia.brain import (
    BrainBuilder,
    BrainConfig,
    NeuralRegionConfig,
    NeuralRegionRegistry,
    ConductanceScaledSpec,
    STPConfig,
    apply_stp_correction,
)
from thalia.typing import (
    NeuromodulatorInput,
    PopulationName,
    PopulationSizes,
    ReceptorType,
    RegionName,
    SynapseId,
    SynapticInput,
)

from .analysis import analyze
from .diagnostics_recorder import DiagnosticsRecorder

if TYPE_CHECKING:
    from .diagnostics_types import DiagnosticsConfig, DiagnosticsReport


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PoissonInputSpec:
    """Specification for one Poisson spike input source injected into the test region."""

    synapse_id: SynapseId
    """Full routing key (source_region / source_population / target_region / target_population / receptor_type)."""

    n_input: int
    """Number of Poisson neurons in the input population."""

    rate_hz: float
    """Mean firing rate of each input neuron (Hz)."""

    connectivity: float
    """Sparse connection probability (0–1)."""

    weight_scale: Union[float, ConductanceScaledSpec]
    """Weight scale for the synaptic connection."""

    stp_config: Optional[STPConfig] = None
    """Optional short-term plasticity configuration."""


@dataclass
class RegionTestResult:
    """Firing-rate statistics from an isolated region test run."""

    rates_hz: Dict[str, float]
    """Mean firing rate per population (Hz).  Measured over ``duration_ms`` after warmup."""

    spike_counts: Dict[str, int]
    """Total spike count per population during the measurement window."""

    duration_ms: float
    """Measurement window duration in milliseconds (warmup excluded)."""

    n_neurons: Dict[str, int]
    """Number of neurons in each population."""

    region_name: str
    """Name of the tested region."""

    diagnostics: Optional[DiagnosticsReport] = field(default=None)
    """Full diagnostics report, or ``None`` when *diagnostics_config* was not passed to :meth:`RegionTestRunner.run`."""

    def print(self) -> None:
        """Print a human-readable firing rate summary to stdout."""
        print(f"\n{'─' * 65}")
        print(f"  RegionTestResult: {self.region_name}  ({self.duration_ms:.0f} ms measurement)")
        print(f"{'─' * 65}")
        print(f"  {'Population':<38} {'Rate':>8}  {'Spikes':>8}  {'N':>6}")
        print(f"  {'-' * 38} {'-' * 8}  {'-' * 8}  {'-' * 6}")
        for pop_name in sorted(self.rates_hz):
            rate = self.rates_hz[pop_name]
            spikes = self.spike_counts.get(pop_name, 0)
            n = self.n_neurons.get(pop_name, 0)
            print(f"  {pop_name:<38} {rate:>7.1f}Hz  {spikes:>8}  {n:>6}")
        print(f"{'─' * 65}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

class RegionTestRunner:
    """Run a single brain region in isolation with Poisson spike inputs.

    Supports two construction paths:

    1. **From preset** (recommended for debugging existing regions)::

        runner = RegionTestRunner.from_preset("default", "cortex_sensory")

    2. **Manual construction** (for testing custom configs or sizes)::

        runner = RegionTestRunner(
            region_name="cortex_sensory",
            population_sizes={"l4_pyr": 400, ...},
            config=CorticalColumnConfig(baseline_drive=0.01),
            registry_name="cortical_column",
        )

    After construction, add Poisson inputs for each population you want to
    drive, then call :meth:`run`::

        runner.add_poisson_input(target_population="l4_pyr", rate_hz=15.0, ...)
        result = runner.run(duration_ms=1000)
        result.print()
    """

    def __init__(
        self,
        region_name: RegionName,
        population_sizes: PopulationSizes,
        *,
        config: NeuralRegionConfig,
        registry_name: Optional[RegionName] = None,
        dt_ms: float = 1.0,
    ) -> None:
        """Initialise the runner.

        Args:
            region_name: Instance name for the region (also used by the registry
                lookup when ``registry_name`` is not provided).
            population_sizes: Population size dict (matches the region's expected keys).
            config: Region configuration.
            registry_name: Registry key for the region class.  Defaults to
                ``region_name`` when not provided.
            dt_ms: Simulation timestep in milliseconds.
        """
        self.region_name: RegionName = region_name
        self.registry_name: RegionName = registry_name or region_name
        self.population_sizes: PopulationSizes = population_sizes
        self.config: NeuralRegionConfig = config
        self.dt_ms: float = dt_ms
        self._poisson_inputs: List[PoissonInputSpec] = []

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_preset(
        cls,
        preset_name: str,
        region_name: RegionName,
        *,
        dt_ms: float = 1.0,
        brain_config: Optional[BrainConfig] = None,
    ) -> "RegionTestRunner":
        """Create a runner whose region spec is extracted from a preset.

        This is the most convenient way to test an existing region in isolation
        using exactly the same population sizes and configuration that the full
        brain would use.

        Example::

            runner = RegionTestRunner.from_preset("default", "cortex_sensory")

        Args:
            preset_name: Name of the registered preset (e.g. ``"default"``).
            region_name: Name of the specific region to isolate.
            dt_ms: Simulation timestep in milliseconds.
            brain_config: Optional brain configuration overrides.

        Returns:
            :class:`RegionTestRunner` ready for :meth:`add_poisson_input` calls.

        Raises:
            KeyError: If ``preset_name`` or ``region_name`` not found.
        """
        builder = BrainBuilder.preset_builder(preset_name, brain_config)

        if region_name not in builder._region_specs:
            available = sorted(builder._region_specs.keys())
            raise KeyError(
                f"Region '{region_name}' not found in preset '{preset_name}'. "
                f"Available regions: {available}"
            )

        spec = builder._region_specs[region_name]
        return cls(
            region_name=region_name,
            population_sizes=spec.population_sizes,
            config=spec.config,
            registry_name=spec.registry_name,
            dt_ms=dt_ms,
        )

    # ------------------------------------------------------------------ #
    # Input registration                                                   #
    # ------------------------------------------------------------------ #

    def add_poisson_input(
        self,
        target_population: PopulationName,
        rate_hz: float,
        n_input: int,
        connectivity: float = 0.2,
        weight_scale: Union[float, ConductanceScaledSpec] = 0.0001,
        *,
        receptor_type: ReceptorType = ReceptorType.AMPA,
        source_label: str = "poisson",
        stp_config: Optional[STPConfig] = None,
    ) -> "RegionTestRunner":
        """Register a Poisson spike input targeting a specific population.

        Multiple inputs can be chained::

            runner.add_poisson_input("l4_pyr",   rate_hz=15.0, n_input=100, ...)
                  .add_poisson_input("l4_pv_bsk", rate_hz=10.0, n_input=50,  ...)

        Args:
            target_population: Name of the target population within the region.
            rate_hz: Mean Poisson firing rate (Hz).
            n_input: Number of Poisson neurons (pre-synaptic population size).
            connectivity: Sparse connection probability (0–1).
            weight_scale: Float or :class:`~thalia.brain.synapses.ConductanceScaledSpec`.
            receptor_type: Receptor type (default AMPA).
            source_label: Human-readable label for the Poisson source.
            stp_config: Optional short-term plasticity config.

        Returns:
            Self, for method chaining.
        """
        synapse_id = SynapseId(
            source_region="__poisson__",
            source_population=source_label,
            target_region=self.region_name,
            target_population=target_population,
            receptor_type=receptor_type,
        )
        self._poisson_inputs.append(PoissonInputSpec(
            synapse_id=synapse_id,
            n_input=n_input,
            rate_hz=rate_hz,
            connectivity=connectivity,
            weight_scale=weight_scale,
            stp_config=stp_config,
        ))
        return self

    # ------------------------------------------------------------------ #
    # Run                                                                  #
    # ------------------------------------------------------------------ #

    def run(
        self,
        duration_ms: float = 1000.0,
        warmup_ms: float = 500.0,
        device: str = "cpu",
        *,
        diagnostics_config: Optional[DiagnosticsConfig] = None,
    ) -> RegionTestResult:
        """Run the region in isolation and return per-population firing rates.

        Args:
            duration_ms: Measurement window in milliseconds (after warmup).
            warmup_ms: Warmup period whose spikes are discarded.
            device: Torch device string (e.g. ``"cpu"`` or ``"cuda"``).
            diagnostics_config: When provided, a :class:`DiagnosticsRecorder` is
                attached for the measurement window and a full
                :class:`DiagnosticsReport` is attached to the returned result
                (``result.diagnostics``).  The recorder uses this region as a
                single-region "brain".

        Returns:
            :class:`RegionTestResult` with per-population firing rates, and
            optionally a :attr:`~RegionTestResult.diagnostics` report.
        """
        # ── 1. Instantiate the region ────────────────────────────────────────
        self.config.dt_ms = self.dt_ms

        region = NeuralRegionRegistry.create(
            self.registry_name,
            config=self.config,
            population_sizes=self.population_sizes,
            region_name=self.region_name,
            device=device,
        )

        # ── 2. Register Poisson inputs ───────────────────────────────────────
        for inp in self._poisson_inputs:
            corrected_ws = apply_stp_correction(inp.weight_scale, inp.stp_config)
            region.add_input_source(
                synapse_id=inp.synapse_id,
                n_input=inp.n_input,
                connectivity=inp.connectivity,
                weight_scale=corrected_ws,
                stp_config=inp.stp_config,
                learning_strategy=None,
                device=device,
            )
            # Pre-load STP state to steady state to avoid onset transient.
            if inp.stp_config is not None and isinstance(corrected_ws, ConductanceScaledSpec):
                stp = region.get_stp_module(inp.synapse_id)
                if stp is not None:
                    stp.initialize_to_steady_state(corrected_ws.source_rate_hz)

        # Some regions need post-connection finalisation (e.g. gap junctions).
        if hasattr(region, "finalize_initialization"):
            region.finalize_initialization()

        # ── 3. Optionally attach a DiagnosticsRecorder ───────────────────────
        recorder = None
        if diagnostics_config is not None:
            n_meas_steps = int(duration_ms / self.dt_ms)
            diag_cfg = diagnostics_config
            # Override n_timesteps to match our measurement window.
            if diag_cfg.n_timesteps != n_meas_steps:
                diag_cfg = copy.replace(diag_cfg, n_timesteps=n_meas_steps)

            # Build a minimal Brain-compatible wrapper so DiagnosticsRecorder
            # can inspect region/population structure without a full Brain.
            class _FakeBrain:
                def __init__(self, region_name: str, region_obj: Any) -> None:
                    self.regions: Dict[str, Any] = {region_name: region_obj}
                    self.axonal_tracts: Dict[Any, Any] = {}

            fake_brain = _FakeBrain(self.region_name, region)
            recorder = DiagnosticsRecorder(fake_brain, diag_cfg)

        # ── 4. Simulate ──────────────────────────────────────────────────────
        total_steps   = int((warmup_ms + duration_ms) / self.dt_ms)
        warmup_steps  = int(warmup_ms / self.dt_ms)

        # Pre-compute per-input Poisson probabilities (proportional to dt_ms)
        poisson_specs: List[tuple[SynapseId, float, int]] = [
            (inp.synapse_id, inp.rate_hz / 1000.0 * self.dt_ms, inp.n_input)
            for inp in self._poisson_inputs
        ]

        spike_accum: Dict[str, int] = {}
        empty_neuromod: NeuromodulatorInput = {}

        for step in range(total_steps):
            # Generate Poisson spikes: each neuron fires with prob = rate * dt
            synaptic_inputs: SynapticInput = {
                sid: torch.rand(n_in, device=device) < prob
                for sid, prob, n_in in poisson_specs
            }

            region_output = region.forward(synaptic_inputs, empty_neuromod)

            # Accumulate only after warmup
            if step >= warmup_steps:
                meas_step = step - warmup_steps
                for pop_name, spikes_t in region_output.items():
                    key = str(pop_name)
                    if key not in spike_accum:
                        spike_accum[key] = 0
                    spike_accum[key] += int(spikes_t.sum().item())
                # Record into diagnostics if configured
                if recorder is not None:
                    recorder.record(meas_step, synaptic_inputs, {self.region_name: region_output})

        # ── 5. Compute rates ─────────────────────────────────────────────────
        measurement_ms = total_steps * self.dt_ms - warmup_ms
        rates:    Dict[str, float] = {}
        n_neurons: Dict[str, int]  = {}

        for pop_key, spikes_t in spike_accum.items():
            try:
                n = region.get_population_size(pop_key)
            except (KeyError, ValueError):
                n = 1  # Fallback — prevents division errors for unknown keys
            n_neurons[pop_key] = n
            rates[pop_key] = (
                spikes_t / n / (measurement_ms / 1000.0)
                if n > 0 and measurement_ms > 0 else 0.0
            )

        if recorder is not None:
            if recorder.config.mode == "full":
                recorder._build_spike_times()
            diag_report = analyze(recorder)
        else:
            diag_report = None

        return RegionTestResult(
            rates_hz=rates,
            spike_counts=spike_accum,
            duration_ms=measurement_ms,
            n_neurons=n_neurons,
            region_name=self.region_name,
            diagnostics=diag_report,
        )
