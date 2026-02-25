"""
Comprehensive Brain Diagnostics — thin driver script.

This script owns the simulation loop.  All recording and analysis is
delegated to ``DiagnosticsRecorder``, which can equally be plugged into a
training loop with a single ``recorder.record(t, outputs)`` call.

Usage::

    python scripts/comprehensive_diagnostics.py
    python scripts/comprehensive_diagnostics.py --timesteps 5000 --input-pattern rhythmic
    python scripts/comprehensive_diagnostics.py --output-dir data/diagnostics --no-plots
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

import torch

from thalia import GlobalConfig
from thalia.brain import BrainBuilder, DynamicBrain
from thalia.brain.regions.population_names import ThalamusPopulation
from thalia.diagnostics import DiagnosticsConfig, DiagnosticsRecorder
from thalia.typing import SynapseId, SynapticInput

# ── Configure line buffering so progress prints appear immediately ─────────────
sys.stdout.reconfigure(line_buffering=True)


# =============================================================================
# BRAIN INFO HELPERS
# =============================================================================


def print_brain_config(brain: DynamicBrain) -> None:
    """Print a compact overview of the brain's region list."""
    print(f"\n{'═'*80}")
    print("BRAIN CONFIGURATION")
    print(f"{'═'*80}")
    print(f"  Device        : {brain.device}")
    print(f"  dt_ms         : {brain.dt_ms} ms")
    print(f"  Axonal tracts : {len(brain.axonal_tracts)}")
    print(f"  Regions       : {len(brain.regions)}")
    for region_name in brain.regions:
        print(f"    - {region_name}")


def print_neuron_populations(brain: DynamicBrain) -> None:
    """Print per-population neuron counts for every region."""
    print(f"\n{'═'*80}")
    print("NEURON POPULATION SIZES")
    print(f"{'═'*80}")
    total: int = 0
    for region_name, region in brain.regions.items():
        region_total: int = sum(int(p.n_neurons) for p in region.neuron_populations.values())
        total += region_total
        print(f"  {region_name}  [{region_total} neurons]")
        for pop_name, pop in region.neuron_populations.items():
            print(f"    {pop_name:<42s} {int(pop.n_neurons):>6d}  ({pop.__class__.__name__})")
    print(f"\n  TOTAL : {total:,} neurons")


def print_synaptic_weights(brain: DynamicBrain, heading: str = "SYNAPTIC WEIGHTS") -> None:
    """Print weight statistics and STP parameters for every synapse."""
    print(f"\n{'═'*80}")
    print(heading)
    print(f"{'═'*80}")
    for region in brain.regions.values():
        for synapse_id, weights in region.synaptic_weights.items():
            stp = region.stp_modules.get(synapse_id, None)
            if stp is not None:
                stp_str = (
                    f"STP U={stp.config.U:.2f}  "
                    f"τd={stp.config.tau_d:.0f}ms  "
                    f"τf={stp.config.tau_f:.0f}ms"
                )
            else:
                stp_str = "no STP"
            shape_str = "×".join(str(d) for d in weights.shape)
            print(
                f"  {str(synapse_id):<80s}"
                f"  {shape_str:>11s}  "
                f"μ={weights.mean():.5f}  "
                f"σ={weights.std():.5f}  "
                f"min={weights.min():.5f}  "
                f"max={weights.max():.5f}  "
                f"[{stp_str}]"
            )


# =============================================================================
# SENSORY INPUT GENERATOR
# =============================================================================


def make_sensory_input(
    brain: DynamicBrain,
    timestep: int,
    pattern: str,
) -> Optional[SynapticInput]:
    """
    Generate a ``SynapticInput`` dict targeting the thalamus relay population.

    Parameters
    ----------
    brain:
        The live ``DynamicBrain`` instance.
    timestep:
        Current integer timestep index.
    pattern:
        One of ``"random"``, ``"rhythmic"``, ``"burst"``, ``"none"``.
    """
    if pattern == "none":
        return None

    thalamus = brain.get_region_by_name("thalamus")
    if thalamus is None:
        return None

    relay_size: int = thalamus.get_population_size(ThalamusPopulation.RELAY)
    device = brain.device
    dt_ms = brain.dt_ms
    spikes: torch.Tensor

    if pattern == "random":
        # Sparse Poisson: ~3 % of relay neurons, each with 20 % spike probability
        n_active = max(1, int(relay_size * 0.03))
        active_idx = torch.randperm(relay_size, device=device)[:n_active]
        mask = torch.rand(n_active, device=device) < 0.20
        spikes = torch.zeros(relay_size, dtype=torch.bool, device=device)
        spikes[active_idx] = mask

    elif pattern == "rhythmic":
        # Theta rhythm (8 Hz → 125 ms period): active for first 20 % of each cycle
        period_ms = 125.0
        phase = (timestep * dt_ms % period_ms) / period_ms
        if phase < 0.20:
            n_active = max(1, int(relay_size * 0.10))
            spikes = torch.zeros(relay_size, dtype=torch.bool, device=device)
            spikes[torch.randperm(relay_size, device=device)[:n_active]] = True
        else:
            spikes = torch.zeros(relay_size, dtype=torch.bool, device=device)

    elif pattern == "burst":
        # Single burst at t = 100 ms
        if timestep == int(100.0 / dt_ms):
            n_active = max(1, int(relay_size * 0.30))
            spikes = torch.zeros(relay_size, dtype=torch.bool, device=device)
            spikes[torch.randperm(relay_size, device=device)[:n_active]] = True
        else:
            spikes = torch.zeros(relay_size, dtype=torch.bool, device=device)

    else:
        raise ValueError(f"Unknown input pattern: {pattern!r}")

    synapse_id = SynapseId.external_sensory_to_thalamus_relay(thalamus.region_name)
    return {synapse_id: spikes}


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive Thalia brain diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000,
        help="Number of simulation timesteps (1 step = dt_ms).",
    )
    parser.add_argument(
        "--warmup", type=int, default=0,
        help=(
            "Number of warmup timesteps run before recording begins. "
            "Allows STP and adaptation to reach near-equilibrium so the "
            "measurement window reflects steady-state dynamics, not startup "
            "transients. At dt=1ms and 30 Hz thalamic input the thalamocortical "
            "STP effective time constant is ~67 ms; 300 ms ≈ 4.5τ. "
            "WARNING: circuits without STP-compensated weights (BLA, LHb, "
            "hippocampus) will be starved and appear silent with warmup>0."
        ),
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/diagnostics",
        help="Directory for diagnostic outputs (JSON, NPZ, PNG).",
    )
    parser.add_argument(
        "--input-pattern", type=str, default="random",
        choices=["random", "rhythmic", "burst", "none"],
        help="Sensory input pattern fed to the thalamus relay.",
    )
    parser.add_argument(
        "--mode", type=str, default="full",
        choices=["full", "stats"],
        help=(
            "'full' records spike times, voltages, and conductances. "
            "'stats' records only spike counts and gains (lighter, suitable "
            "for long training-loop checks)."
        ),
    )
    parser.add_argument(
        "--no-plots", action="store_true",
        help="Skip matplotlib plot generation.",
    )
    args = parser.parse_args()

    # ── Header ────────────────────────────────────────────────────────────────
    print("\n" + "═" * 80)
    print("THALIA  ·  COMPREHENSIVE BRAIN DIAGNOSTICS")
    print("═" * 80)

    print("\n  GlobalConfig:")
    print(f"    DEFAULT_DT_MS            : {GlobalConfig.DEFAULT_DT_MS}")
    print(f"    HOMEOSTASIS_DISABLED     : {GlobalConfig.HOMEOSTASIS_DISABLED}")
    print(f"    LEARNING_DISABLED        : {GlobalConfig.LEARNING_DISABLED}")
    print(f"    NEUROMODULATION_DISABLED : {GlobalConfig.NEUROMODULATION_DISABLED}")
    print(f"    SYNAPTIC_WEIGHT_SCALE    : {GlobalConfig.SYNAPTIC_WEIGHT_SCALE}")

    print(f"\n  Run parameters:")
    print(f"    timesteps    : {args.timesteps}")
    print(f"    warmup       : {args.warmup}")
    print(f"    input-pattern: {args.input_pattern}")
    print(f"    mode         : {args.mode}")
    print(f"    output-dir   : {args.output_dir}")
    print(f"    plots        : {'no' if args.no_plots else 'yes'}")

    # ── Build brain ───────────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print("BUILDING BRAIN")
    print(f"{'═'*80}\n")
    t_build = time.perf_counter()
    brain = BrainBuilder.preset("default")
    print(f"  ✓ Brain built in {time.perf_counter() - t_build:.3f} s")

    print_brain_config(brain)
    print_neuron_populations(brain)
    print_synaptic_weights(brain, heading="INITIAL SYNAPTIC WEIGHTS")

    # ── Create recorder ───────────────────────────────────────────────────────
    config = DiagnosticsConfig(
        n_timesteps=args.timesteps,
        dt_ms=brain.dt_ms,
        mode=args.mode,
        voltage_sample_size=8,
        conductance_sample_size=8,
        conductance_sample_interval_ms=1,
        gain_sample_interval_ms=10,
        rate_bin_ms=10.0,
    )
    recorder = DiagnosticsRecorder(brain, config)

    # ── Warmup ────────────────────────────────────────────────────────────────
    if args.warmup > 0:
        print(f"\n{'═'*80}")
        print(
            f"WARMUP {args.warmup} TIMESTEPS"
            f"  input={args.input_pattern!r}  (not recorded)"
        )
        print(f"{'═'*80}\n")
        t_warmup = time.perf_counter()
        for t in range(args.warmup):
            synaptic_inputs = make_sensory_input(brain, t, args.input_pattern)
            brain.forward(synaptic_inputs)  # no recorder.record — warmup only
            if (t + 1) % 100 == 0:
                elapsed = time.perf_counter() - t_warmup
                rate = (t + 1) / elapsed
                eta = (args.warmup - t - 1) / rate
                print(
                    f"  warmup step {t+1:>6d}/{args.warmup}"
                    f"  {rate:6.2f} steps/s"
                    f"  ETA {eta:6.2f} s"
                )
        print(f"  ✓ Warmup complete: {time.perf_counter() - t_warmup:.2f} s")

    # ── Simulation loop ───────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print(
        f"RUNNING {args.timesteps} TIMESTEPS"
        f"  input={args.input_pattern!r}  mode={args.mode!r}"
    )
    print(f"{'═'*80}\n")

    t_sim = time.perf_counter()
    for t in range(args.timesteps):
        synaptic_inputs = make_sensory_input(brain, t, args.input_pattern)
        outputs = brain.forward(synaptic_inputs)
        recorder.record(t, outputs)

        if (t + 1) % 100 == 0:
            elapsed = time.perf_counter() - t_sim
            rate = (t + 1) / elapsed
            eta = (args.timesteps - t - 1) / rate
            print(
                f"  step {t+1:>6d}/{args.timesteps}"
                f"  {rate:6.2f} steps/s"
                f"  ETA {eta:6.2f} s"
            )

    elapsed_sim = time.perf_counter() - t_sim
    print(
        f"\n  ✓ Simulation complete: {elapsed_sim:.2f} s"
        f"  ({args.timesteps / elapsed_sim:.1f} steps/s)"
    )

    print_synaptic_weights(brain, heading="FINAL SYNAPTIC WEIGHTS")

    # ── Analyse ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print("ANALYSING")
    print(f"{'═'*80}\n")

    t_analyse = time.perf_counter()
    report = recorder.analyze()
    print(f"  ✓ Analysis complete in {time.perf_counter() - t_analyse:.3f} s")

    # ── Print report ──────────────────────────────────────────────────────────
    recorder.print_report(report, detailed=True)

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print(f"SAVING  →  {args.output_dir}")
    print(f"{'═'*80}\n")
    recorder.save(report, args.output_dir)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if not args.no_plots:
        print(f"\n{'═'*80}")
        print("GENERATING PLOTS")
        print(f"{'═'*80}\n")
        recorder.plot(report, args.output_dir)

    print(f"\n{'═'*80}")
    print("DONE")
    print(f"{'═'*80}\n")


if __name__ == "__main__":
    main()
