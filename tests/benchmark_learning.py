#!/usr/bin/env python3
"""Benchmark performance and memory usage of learning mechanisms.

This script measures:
1. Execution time per call for each learning function
2. Memory usage scaling with neuron count
3. Throughput (operations per second)

Run with: python tests/benchmark_learning.py
"""

import gc
from dataclasses import dataclass
from typing import Callable, List, Tuple
import time
import torch
import numpy as np

# Import our learning mechanisms
from thalia.learning import (
    hebbian_update,
    synaptic_scaling,
    update_bcm_threshold,
    update_homeostatic_excitability,
    PredictiveCoding,
    PhaseHomeostasis,
)

# Import simulation components
from thalia.core import LIFNeuron, LIFConfig
from thalia.core import DendriticNeuron, DendriticNeuronConfig, DendriticBranchConfig
from thalia.core import ConductanceLIF, ConductanceLIFConfig
from thalia.dynamics import NetworkState, NetworkConfig, forward_timestep


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""
    name: str
    n_neurons: int
    time_per_call_us: float  # microseconds
    memory_mb: float  # megabytes
    calls_per_second: float


def get_tensor_memory_mb(tensor: torch.Tensor) -> float:
    """Get memory usage of a tensor in MB."""
    return tensor.element_size() * tensor.nelement() / (1024 * 1024)


def benchmark_function(
    name: str,
    setup_fn: Callable[[], Tuple],
    run_fn: Callable[..., None],
    n_neurons: int,
    n_iterations: int = 1000,
    warmup: int = 100,
) -> BenchmarkResult:
    """Benchmark a function with given setup and run functions."""
    # Setup
    args = setup_fn()

    # Calculate memory from tensors in args
    memory_mb = sum(
        get_tensor_memory_mb(arg) for arg in args
        if isinstance(arg, torch.Tensor)
    )

    # Warmup
    for _ in range(warmup):
        run_fn(*args)

    # Force sync if CUDA
    if args[0].is_cuda if isinstance(args[0], torch.Tensor) else False:
        torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iterations):
        run_fn(*args)

    if args[0].is_cuda if isinstance(args[0], torch.Tensor) else False:
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start
    time_per_call_us = (elapsed / n_iterations) * 1_000_000
    calls_per_second = n_iterations / elapsed

    return BenchmarkResult(
        name=name,
        n_neurons=n_neurons,
        time_per_call_us=time_per_call_us,
        memory_mb=memory_mb,
        calls_per_second=calls_per_second,
    )


def benchmark_hebbian_update(n_input: int, n_output: int, device: torch.device) -> BenchmarkResult:
    """Benchmark hebbian_update function."""
    def setup():
        weights = torch.rand(n_output, n_input, device=device)
        input_spikes = torch.zeros(1, n_input, device=device)
        output_spikes = torch.zeros(1, n_output, device=device)
        # Simulate ~10% sparsity
        input_spikes[0, ::10] = 1.0
        output_spikes[0, ::10] = 1.0
        return (weights, input_spikes, output_spikes)

    def run(weights, input_spikes, output_spikes):
        hebbian_update(weights, input_spikes, output_spikes, learning_rate=0.01, w_max=1.0, heterosynaptic_ratio=0.5)

    return benchmark_function(
        name="hebbian_update",
        setup_fn=setup,
        run_fn=run,
        n_neurons=n_output,
    )


def benchmark_synaptic_scaling(n_input: int, n_output: int, device: torch.device) -> BenchmarkResult:
    """Benchmark synaptic_scaling function."""
    def setup():
        weights = torch.rand(n_output, n_input, device=device)
        return (weights,)

    def run(weights):
        synaptic_scaling(weights, target_norm_fraction=0.3, tau=10.0, w_max=1.0)

    return benchmark_function(
        name="synaptic_scaling",
        setup_fn=setup,
        run_fn=run,
        n_neurons=n_output,
    )


def benchmark_bcm_threshold(n_output: int, device: torch.device) -> BenchmarkResult:
    """Benchmark update_bcm_threshold function."""
    def setup():
        threshold = torch.ones(1, n_output, device=device) * 0.5
        return (threshold, 20.0, 10.0, 100.0, 0.01, 1.0)

    def run(threshold, avg_activity, target_rate, tau, min_thresh, max_thresh):
        update_bcm_threshold(threshold, avg_activity, target_rate, tau, min_thresh, max_thresh)

    return benchmark_function(
        name="update_bcm_threshold",
        setup_fn=setup,
        run_fn=run,
        n_neurons=n_output,
    )


def benchmark_homeostatic_excitability(n_output: int, device: torch.device) -> BenchmarkResult:
    """Benchmark update_homeostatic_excitability function."""
    def setup():
        current_rate = torch.rand(1, n_output, device=device) * 0.01
        avg_rate = torch.rand(1, n_output, device=device) * 0.01
        excitability = torch.zeros(1, n_output, device=device)
        return (current_rate, avg_rate, excitability)

    def run(current_rate, avg_rate, excitability):
        update_homeostatic_excitability(
            current_rate, avg_rate, excitability,
            target_rate=0.002, tau=500.0, strength=0.01, v_threshold=1.0,
            bounds=(-0.5, 0.5)
        )

    return benchmark_function(
        name="homeostatic_excitability",
        setup_fn=setup,
        run_fn=run,
        n_neurons=n_output,
    )


def benchmark_predictive_coding(n_output: int, device: torch.device) -> BenchmarkResult:
    """Benchmark PredictiveCoding class operations."""
    gamma_period = 100
    learning_phase = 90

    def setup():
        pc = PredictiveCoding(
            n_output=n_output,
            gamma_period=gamma_period,
            learning_phase=learning_phase,
            start_cycle=0,
            device=device,
        )
        output_spikes = torch.zeros(1, n_output, device=device)
        output_spikes[0, 0] = 1.0
        recurrent_weights = torch.rand(n_output, n_output, device=device)
        return (pc, output_spikes, recurrent_weights)

    def run(pc, output_spikes, recurrent_weights):
        pc.accumulate_spikes(output_spikes)
        # Simulate learning phase
        pc.update_recurrent(learning_phase, recurrent_weights, current_cycle=10)

    return benchmark_function(
        name="PredictiveCoding",
        setup_fn=setup,
        run_fn=run,
        n_neurons=n_output,
    )


def benchmark_phase_homeostasis(n_output: int, device: torch.device) -> BenchmarkResult:
    """Benchmark PhaseHomeostasis class operations."""
    def setup():
        ph = PhaseHomeostasis(n_output=n_output, device=device)
        return (ph,)

    def run(ph):
        ph.record_win(0)
        ph.record_win(1)
        ph.update_cycle()
        ph.reset_cycle()

    return benchmark_function(
        name="PhaseHomeostasis",
        setup_fn=setup,
        run_fn=run,
        n_neurons=n_output,
    )


def benchmark_lif_neuron(n_output: int, device: torch.device) -> BenchmarkResult:
    """Benchmark LIF neuron forward pass."""
    def setup():
        config = LIFConfig(tau_mem=20.0, v_threshold=1.0, noise_std=0.1, dt=0.1)
        neurons = LIFNeuron(n_neurons=n_output, config=config).to(device)
        neurons.reset_state(batch_size=1)
        input_current = torch.rand(1, n_output, device=device) * 0.5
        return (neurons, input_current)

    def run(neurons, input_current):
        neurons(input_current)

    return benchmark_function(
        name="LIFNeuron.forward",
        setup_fn=setup,
        run_fn=run,
        n_neurons=n_output,
    )


def benchmark_conductance_lif(n_output: int, device: torch.device) -> BenchmarkResult:
    """Benchmark ConductanceLIF neuron forward pass."""
    def setup():
        config = ConductanceLIFConfig(v_threshold=1.0, dt=0.1)
        neurons = ConductanceLIF(n_neurons=n_output, config=config).to(device)
        neurons.reset_state(batch_size=1)
        g_exc = torch.rand(1, n_output, device=device) * 0.5
        g_inh = torch.rand(1, n_output, device=device) * 0.2
        return (neurons, g_exc, g_inh)

    def run(neurons, g_exc, g_inh):
        neurons(g_exc, g_inh)

    return benchmark_function(
        name="ConductanceLIF.forward",
        setup_fn=setup,
        run_fn=run,
        n_neurons=n_output,
    )


def benchmark_dendritic_neuron(n_input: int, n_output: int, device: torch.device) -> BenchmarkResult:
    """Benchmark DendriticNeuron forward pass (with NMDA branches + synaptic dynamics)."""
    n_branches = 4
    inputs_per_branch = n_input // n_branches

    def setup():
        branch_config = DendriticBranchConfig(
            nmda_threshold=0.3,
            nmda_gain=2.5,
            tau_syn_ms=15.0,  # Synaptic temporal integration
            plateau_tau_ms=50.0,
            dt=0.1,
        )
        soma_config = ConductanceLIFConfig(v_threshold=1.0, dt=0.1)
        config = DendriticNeuronConfig(
            n_branches=n_branches,
            inputs_per_branch=inputs_per_branch,
            branch_config=branch_config,
            soma_config=soma_config,
        )
        neurons = DendriticNeuron(n_neurons=n_output, config=config).to(device)
        neurons.reset_state(batch_size=1)
        inputs = torch.zeros(1, n_input, device=device)
        inputs[0, ::5] = 0.1  # 20% sparsity
        g_inh = torch.rand(1, n_output, device=device) * 0.2
        return (neurons, inputs, g_inh)

    def run(neurons, inputs, g_inh):
        neurons(inputs, g_inh)

    return benchmark_function(
        name="DendriticNeuron.forward",
        setup_fn=setup,
        run_fn=run,
        n_neurons=n_output,
    )


def benchmark_forward_timestep(n_input: int, n_output: int, device: torch.device) -> BenchmarkResult:
    """Benchmark the full forward_timestep function."""
    def setup():
        # Create all the components needed for forward_timestep
        dt = 0.1
        recurrent_delay = 100
        interneuron_delay = 20

        config = LIFConfig(tau_mem=20.0, v_threshold=1.0, noise_std=0.1, dt=dt)
        neurons = LIFNeuron(n_neurons=n_output, config=config).to(device)
        neurons.reset_state(batch_size=1)

        # Create network state
        state = NetworkState.create(
            n_output, recurrent_delay, interneuron_delay,
            config.v_threshold, 0.002, device
        )
        # Skip spike tracking for benchmark (simulates training mode)
        state.skip_spike_tracking = True

        # Create network config
        theta_period = 1600
        gamma_period = 100
        cycle_duration = 1600
        effective_cycle = 2100

        # Inhibition kernel
        output_positions = torch.arange(n_output, device=device, dtype=torch.float32)
        distance_matrix = torch.abs(output_positions.unsqueeze(0) - output_positions.unsqueeze(1))
        inhibition_kernel = torch.exp(-distance_matrix**2 / 8.0)
        inhibition_kernel = inhibition_kernel * (1 - torch.eye(n_output, device=device))

        theta_phase_preference = torch.zeros(n_output, device=device)
        for i in range(n_output):
            theta_phase_preference[i] = (i / n_output) * 2 * np.pi

        net_config = NetworkConfig(
            n_input=n_input,
            n_output=n_output,
            device=device,
            dt=dt,
            recurrent_delay=recurrent_delay,
            interneuron_delay=interneuron_delay,
            theta_period=theta_period,
            gamma_period=gamma_period,
            cycle_duration=cycle_duration,
            effective_cycle=effective_cycle,
            shunting_strength=3.0,
            shunting_decay=0.5,
            blanket_inhibition_strength=0.5,
            gamma_reset_factor=0.3,
            sfa_strength=1.5,
            sfa_increment=0.15,
            sfa_decay=0.995,
            absolute_refractory=20,
            relative_refractory=30,
            relative_refractory_factor=0.3,
            theta_phase_preference=theta_phase_preference,
            theta_modulation_strength=2.5,
            v_threshold=config.v_threshold,
            target_rate=0.002,
            homeostatic_tau=500.0,
            homeostatic_strength_fraction=0.01,
            intrinsic_strength_fraction=0.002,
            inhibition_kernel=inhibition_kernel,
        )

        # Weights
        weights = torch.rand(n_output, n_input, device=device) * 0.3 + 0.2
        recurrent_weights = torch.rand(n_output, n_output, device=device) * 0.15
        recurrent_weights = recurrent_weights * (1 - torch.eye(n_output, device=device))

        # Input spikes
        input_spikes = torch.zeros(1, n_input, device=device)
        input_spikes[0, 0] = 1.0  # One input spike

        return (neurons, state, net_config, weights, recurrent_weights, input_spikes)

    # Need a mutable counter for timestep
    timestep = [0]

    def run(neurons, state, net_config, weights, recurrent_weights, input_spikes):
        forward_timestep(
            timestep[0], input_spikes, state, net_config,
            weights, recurrent_weights, neurons
        )
        timestep[0] += 1

    return benchmark_function(
        name="forward_timestep",
        setup_fn=setup,
        run_fn=run,
        n_neurons=n_output,
    )


def benchmark_forward_timestep_no_skip(n_input: int, n_output: int, device: torch.device) -> BenchmarkResult:
    """Benchmark forward_timestep WITH CPU spike tracking (original slow path)."""
    def setup():
        dt = 0.1
        recurrent_delay = 100
        interneuron_delay = 20

        config = LIFConfig(tau_mem=20.0, v_threshold=1.0, noise_std=0.1, dt=dt)
        neurons = LIFNeuron(n_neurons=n_output, config=config).to(device)
        neurons.reset_state(batch_size=1)

        state = NetworkState.create(
            n_output, recurrent_delay, interneuron_delay,
            config.v_threshold, 0.002, device
        )
        # DO NOT skip spike tracking - test the slow path
        state.skip_spike_tracking = False

        theta_period = 1600
        gamma_period = 100
        cycle_duration = 1600
        effective_cycle = 2100

        output_positions = torch.arange(n_output, device=device, dtype=torch.float32)
        distance_matrix = torch.abs(output_positions.unsqueeze(0) - output_positions.unsqueeze(1))
        inhibition_kernel = torch.exp(-distance_matrix**2 / 8.0)
        inhibition_kernel = inhibition_kernel * (1 - torch.eye(n_output, device=device))

        theta_phase_preference = torch.zeros(n_output, device=device)
        for i in range(n_output):
            theta_phase_preference[i] = (i / n_output) * 2 * np.pi

        net_config = NetworkConfig(
            n_input=n_input,
            n_output=n_output,
            device=device,
            dt=dt,
            recurrent_delay=recurrent_delay,
            interneuron_delay=interneuron_delay,
            theta_period=theta_period,
            gamma_period=gamma_period,
            cycle_duration=cycle_duration,
            effective_cycle=effective_cycle,
            shunting_strength=3.0,
            shunting_decay=0.5,
            blanket_inhibition_strength=0.5,
            gamma_reset_factor=0.3,
            sfa_strength=1.5,
            sfa_increment=0.15,
            sfa_decay=0.995,
            absolute_refractory=20,
            relative_refractory=30,
            relative_refractory_factor=0.3,
            theta_phase_preference=theta_phase_preference,
            theta_modulation_strength=2.5,
            v_threshold=config.v_threshold,
            target_rate=0.002,
            intrinsic_strength_fraction=0.002,
            inhibition_kernel=inhibition_kernel,
        )

        weights = torch.rand(n_output, n_input, device=device) * 0.3 + 0.2
        recurrent_weights = torch.rand(n_output, n_output, device=device) * 0.15
        recurrent_weights = recurrent_weights * (1 - torch.eye(n_output, device=device))

        input_spikes = torch.zeros(1, n_input, device=device)
        input_spikes[0, 0] = 1.0

        return (neurons, state, net_config, weights, recurrent_weights, input_spikes)

    timestep = [0]

    def run(neurons, state, net_config, weights, recurrent_weights, input_spikes):
        forward_timestep(
            timestep[0], input_spikes, state, net_config,
            weights, recurrent_weights, neurons
        )
        timestep[0] += 1

    return benchmark_function(
        name="forward_timestep (CPU track)",
        setup_fn=setup,
        run_fn=run,
        n_neurons=n_output,
        n_iterations=500,  # Fewer iterations since it's slower
    )


def run_scaling_benchmark(device: torch.device) -> List[BenchmarkResult]:
    """Run benchmarks at different scales."""
    results = []

    # Different neuron counts to test scaling
    scales = [
        (20, 10),      # Tiny (like exp2)
        (100, 50),     # Small
        (200, 100),    # Medium
        (1000, 500),   # Large
        (2000, 1000),  # Very large
    ]

    print(f"\nBenchmarking on device: {device}")
    print("=" * 80)

    for n_input, n_output in scales:
        print(f"\nScale: {n_input} inputs → {n_output} outputs")
        print("-" * 60)

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # Run each benchmark
        benchmarks = [
            benchmark_lif_neuron(n_output, device),
            benchmark_conductance_lif(n_output, device),
            benchmark_dendritic_neuron(n_input, n_output, device),
            # Skip forward_timestep benchmarks - NetworkConfig has changed
            # benchmark_forward_timestep(n_input, n_output, device),
            # benchmark_forward_timestep_no_skip(n_input, n_output, device),
            benchmark_hebbian_update(n_input, n_output, device),
            benchmark_synaptic_scaling(n_input, n_output, device),
            benchmark_bcm_threshold(n_output, device),
            benchmark_homeostatic_excitability(n_output, device),
            benchmark_predictive_coding(n_output, device),
            benchmark_phase_homeostasis(n_output, device),
        ]

        for result in benchmarks:
            print(f"  {result.name:30s}: {result.time_per_call_us:8.2f} μs/call, "
                  f"{result.memory_mb:6.3f} MB, {result.calls_per_second:,.0f} calls/s")
            results.append(result)

    return results


def analyze_scaling(results: List[BenchmarkResult]):
    """Analyze how performance scales with neuron count."""
    print("\n" + "=" * 80)
    print("SCALING ANALYSIS")
    print("=" * 80)

    # Group by function name
    by_name = {}
    for r in results:
        if r.name not in by_name:
            by_name[r.name] = []
        by_name[r.name].append(r)

    print("\nTime scaling (ratio of time at max scale vs min scale):")
    print("-" * 60)

    for name, func_results in by_name.items():
        func_results.sort(key=lambda x: x.n_neurons)
        min_result = func_results[0]
        max_result = func_results[-1]

        time_ratio = max_result.time_per_call_us / min_result.time_per_call_us
        neuron_ratio = max_result.n_neurons / min_result.n_neurons

        # Estimate scaling exponent: time ∝ n^k → k = log(time_ratio) / log(n_ratio)
        import math
        if neuron_ratio > 1:
            scaling_exponent = math.log(time_ratio) / math.log(neuron_ratio)
        else:
            scaling_exponent = 0

        scaling_type = "O(1)" if scaling_exponent < 0.5 else \
                       "O(n)" if scaling_exponent < 1.5 else \
                       "O(n²)" if scaling_exponent < 2.5 else "O(n³+)"

        print(f"  {name:30s}: {time_ratio:6.1f}× slower at {neuron_ratio:.0f}× neurons "
              f"→ ~{scaling_type} (exponent: {scaling_exponent:.2f})")

    print("\nMemory scaling:")
    print("-" * 60)

    for name, func_results in by_name.items():
        func_results.sort(key=lambda x: x.n_neurons)
        min_result = func_results[0]
        max_result = func_results[-1]

        memory_ratio = max_result.memory_mb / min_result.memory_mb if min_result.memory_mb > 0 else 0
        neuron_ratio = max_result.n_neurons / min_result.n_neurons

        print(f"  {name:30s}: {min_result.memory_mb:.4f} MB → {max_result.memory_mb:.3f} MB "
              f"({memory_ratio:.1f}× at {neuron_ratio:.0f}× neurons)")


def main():
    print("=" * 80)
    print("THALIA LEARNING MECHANISMS - PERFORMANCE BENCHMARK")
    print("=" * 80)

    # Test on CPU first
    device = torch.device("cpu")
    results = run_scaling_benchmark(device)

    # Analyze scaling
    analyze_scaling(results)

    # Test on GPU if available
    if torch.cuda.is_available():
        print("\n" + "=" * 80)
        print("GPU BENCHMARK")
        print("=" * 80)
        device = torch.device("cuda")
        gpu_results = run_scaling_benchmark(device)
        analyze_scaling(gpu_results)

        # Compare CPU vs GPU
        print("\n" + "=" * 80)
        print("CPU vs GPU COMPARISON (at largest scale)")
        print("=" * 80)

        cpu_by_name = {r.name: r for r in results if r.n_neurons == 1000}
        gpu_by_name = {r.name: r for r in gpu_results if r.n_neurons == 1000}

        for name in cpu_by_name:
            cpu_time = cpu_by_name[name].time_per_call_us
            gpu_time = gpu_by_name[name].time_per_call_us
            speedup = cpu_time / gpu_time
            print(f"  {name:30s}: CPU {cpu_time:8.2f} μs, GPU {gpu_time:8.2f} μs "
                  f"→ {speedup:.1f}× {'faster' if speedup > 1 else 'slower'} on GPU")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Find bottlenecks at large scale
    large_scale_results = [r for r in results if r.n_neurons == 1000]
    large_scale_results.sort(key=lambda x: x.time_per_call_us, reverse=True)

    print("\nSlowest operations at 1000 neurons (potential bottlenecks):")
    for r in large_scale_results[:3]:
        print(f"  {r.name}: {r.time_per_call_us:.2f} μs/call")

    print("\nFastest operations at 1000 neurons:")
    for r in large_scale_results[-3:]:
        print(f"  {r.name}: {r.time_per_call_us:.2f} μs/call")

    # Estimate total time for one training step
    total_time = sum(r.time_per_call_us for r in large_scale_results)
    print(f"\nTotal time for all learning operations: {total_time:.2f} μs/step")
    print(f"Maximum training steps per second: {1_000_000 / total_time:,.0f}")


if __name__ == "__main__":
    main()
