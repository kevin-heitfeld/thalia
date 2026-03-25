"""Two-tier health monitoring for the training loop.

Tier 1 — Per-trial lightweight spike-count checks (called every trial).
Tier 2 — Periodic full diagnostics via a probe trial (called every N trials).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from thalia.brain.brain import Brain
from thalia.diagnostics import (
    DiagnosticsConfig,
    DiagnosticsRecorder,
    DiagnosticsReport,
)
from thalia.diagnostics.analysis import analyze
from thalia.diagnostics.simulation_loop import simulate
from thalia.typing import BrainOutput, PopulationKey


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROBE_N_STEPS = 1000
PROBE_PATTERN = "random"
PROBE_REPORT_INTERVAL = 500  # print progress every 500 steps during probe


# ---------------------------------------------------------------------------
# Tier 1 — lightweight per-trial summary
# ---------------------------------------------------------------------------


@dataclass
class TrialHealthSummary:
    """Lightweight health metrics computed from spike counts accumulated
    during a single trial.

    Attributes:
        population_spike_counts: ``(region, population)`` → total spike count.
        trial_steps: Number of timesteps in the trial.
        silent_populations: Populations with zero spikes across the entire trial.
        hyperactive_populations: Populations exceeding the hyperactivity threshold.
    """

    population_spike_counts: dict[PopulationKey, int]
    trial_steps: int
    silent_populations: list[PopulationKey] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]
    hyperactive_populations: list[PopulationKey] = field(default_factory=list)  # pyright: ignore[reportUnknownVariableType]


# ---------------------------------------------------------------------------
# Health Monitor
# ---------------------------------------------------------------------------


class HealthMonitor:
    """Two-tier health monitor integrated into the training loop.

    **Usage**::

        monitor = HealthMonitor(brain)

        # During each trial:
        for t in range(trial.duration_steps):
            outputs = brain.forward(inputs)
            monitor.record_step(outputs)

        summary = monitor.end_trial()

        # Periodically:
        if trial_idx % 100 == 0:
            report = monitor.run_full_diagnostics()
    """

    def __init__(
        self,
        brain: Brain,
        *,
        hyperactive_hz: float = 100.0,
    ) -> None:
        self.brain = brain
        self.hyperactive_hz = hyperactive_hz

        # Tier 1 accumulators (reset each trial)
        self._spike_counts: dict[PopulationKey, int] = {}
        self._trial_steps: int = 0

        # Population sizes — cached for rate calculation
        self._pop_sizes: dict[PopulationKey, int] = {}
        for region in brain.regions.values():
            for pop_name, pop in region.neuron_populations.items():
                key: PopulationKey = (region.region_name, pop_name)
                self._pop_sizes[key] = pop.n_neurons

    # -- Tier 1: per-timestep accumulation -----------------------------------

    def record_step(self, outputs: BrainOutput) -> None:
        """Accumulate spike counts from one ``brain.forward()`` call."""
        self._trial_steps += 1
        for region_name, pops in outputs.items():
            for pop_name, spikes in pops.items():
                key: PopulationKey = (region_name, pop_name)
                self._spike_counts[key] = (
                    self._spike_counts.get(key, 0) + int(spikes.sum().item())
                )

    def end_trial(self) -> TrialHealthSummary:
        """Finalise Tier 1 metrics for the current trial and reset accumulators.

        Returns:
            TrialHealthSummary with spike counts, silent/hyperactive detection.
        """
        dt_ms = 1.0  # simulation timestep
        trial_duration_s = self._trial_steps * dt_ms / 1000.0

        silent: list[PopulationKey] = []
        hyperactive: list[PopulationKey] = []

        for key, count in self._spike_counts.items():
            if count == 0:
                silent.append(key)
            elif trial_duration_s > 0:
                n_neurons = self._pop_sizes.get(key, 1)
                rate_hz = count / (n_neurons * trial_duration_s)
                if rate_hz > self.hyperactive_hz:
                    hyperactive.append(key)

        summary = TrialHealthSummary(
            population_spike_counts=dict(self._spike_counts),
            trial_steps=self._trial_steps,
            silent_populations=silent,
            hyperactive_populations=hyperactive,
        )

        # Reset for next trial
        self._spike_counts.clear()
        self._trial_steps = 0

        return summary

    # -- Tier 2: full diagnostic probe trial ---------------------------------

    def run_full_diagnostics(self) -> DiagnosticsReport:
        """Run a probe trial with full recording and return a diagnostics report.

        Learning is temporarily disabled during the probe so that diagnostic
        input patterns don't corrupt ongoing training.
        """
        self.brain.set_learning_disabled(True)
        try:
            return self._run_probe()
        finally:
            self.brain.set_learning_disabled(False)

    def _run_probe(self) -> DiagnosticsReport:
        """Execute the probe simulation, analyse, and return the report."""
        config = DiagnosticsConfig(n_timesteps=PROBE_N_STEPS)
        recorder = DiagnosticsRecorder(self.brain, config)

        simulate(
            self.brain,
            recorder,
            n_steps=PROBE_N_STEPS,
            pattern=PROBE_PATTERN,
            report_interval=PROBE_REPORT_INTERVAL,
            label="probe",
        )

        snapshot = recorder.to_snapshot()
        return analyze(snapshot)
