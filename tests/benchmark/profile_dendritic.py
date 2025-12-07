"""Profile DendriticNeuron to identify optimization opportunities."""

import torch
import time
from contextlib import contextmanager
from typing import Dict, List
from dataclasses import dataclass, field

from thalia.core.dendritic import (
    DendriticNeuron, DendriticNeuronConfig, DendriticBranchConfig
)
from thalia.core.neuron import ConductanceLIFConfig


@dataclass
class TimingStats:
    """Accumulated timing statistics."""
    times: List[float] = field(default_factory=list)
    
    def add(self, t: float):
        self.times.append(t)
    
    @property
    def total(self) -> float:
        return sum(self.times)
    
    @property
    def mean(self) -> float:
        return self.total / len(self.times) if self.times else 0
    
    @property
    def count(self) -> int:
        return len(self.times)


class Profiler:
    """Simple profiler for timing code sections."""
    
    def __init__(self):
        self.stats: Dict[str, TimingStats] = {}
    
    @contextmanager
    def section(self, name: str):
        if name not in self.stats:
            self.stats[name] = TimingStats()
        start = time.perf_counter()
        yield
        self.stats[name].add(time.perf_counter() - start)
    
    def report(self, total_time: float):
        print("\n" + "=" * 60)
        print("PROFILING RESULTS")
        print("=" * 60)
        print(f"{'Section':<35} {'Total (ms)':<12} {'Mean (μs)':<12} {'Count':<8} {'%':<6}")
        print("-" * 60)
        
        sorted_stats = sorted(
            self.stats.items(),
            key=lambda x: x[1].total,
            reverse=True
        )
        
        for name, stats in sorted_stats:
            total_ms = stats.total * 1000
            mean_us = stats.mean * 1e6
            pct = (stats.total / total_time) * 100 if total_time > 0 else 0
            print(f"{name:<35} {total_ms:>10.2f}  {mean_us:>10.2f}  {stats.count:>6}  {pct:>5.1f}%")


def profile_dendritic_forward(
    n_neurons: int = 100,
    n_branches: int = 4,
    inputs_per_branch: int = 50,
    batch_size: int = 1,
    n_iterations: int = 1000,
    device: str = "cpu"
):
    """Profile individual operations in DendriticNeuron.forward()."""
    
    device = torch.device(device)
    n_input = n_branches * inputs_per_branch
    
    # Setup
    branch_config = DendriticBranchConfig(
        nmda_threshold=0.3,
        nmda_gain=2.5,
        tau_syn_ms=15.0,
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
    
    neuron = DendriticNeuron(n_neurons=n_neurons, config=config).to(device)
    neuron.reset_state()
    
    inputs = torch.zeros(batch_size, n_input, device=device)
    inputs[:, ::5] = 0.1  # 20% sparsity
    g_inh = torch.rand(batch_size, n_neurons, device=device) * 0.2
    
    profiler = Profiler()
    
    print(f"\nProfiling DendriticNeuron on {device}")
    print(f"Config: {n_neurons} neurons, {n_branches} branches, {inputs_per_branch} inputs/branch")
    print(f"Running {n_iterations} iterations...")
    
    # Pre-allocate for analysis
    weights_clamped = neuron.branch_weights.clamp(min=0)
    
    # Warm up
    for _ in range(10):
        neuron(inputs, g_inh)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Profile
    total_start = time.perf_counter()
    
    for _ in range(n_iterations):
        # Manually profile each step of forward()
        
        # 1. Get membrane potential
        with profiler.section("get_membrane"):
            membrane_potential = neuron.soma.membrane
        
        # 2. Route inputs to branches
        with profiler.section("route_inputs"):
            branch_inputs = neuron._route_inputs_to_branches(inputs)
        
        # 3. Clamp weights
        with profiler.section("clamp_weights"):
            weights_clamped = neuron.branch_weights.clamp(min=0)
        
        # 4. Compute weighted sum
        with profiler.section("weighted_sum"):
            instantaneous_input = (branch_inputs * weights_clamped).sum(dim=-1)
        
        # 5. Synaptic dynamics
        with profiler.section("synaptic_dynamics"):
            neuron.branch_g_syn = neuron.branch_g_syn * neuron.syn_decay + instantaneous_input
            linear_sum = neuron.branch_g_syn
        
        # 6. Voltage-dependent NMDA
        with profiler.section("voltage_gate"):
            if membrane_potential is not None:
                voltage_gate = torch.sigmoid(membrane_potential.unsqueeze(-1) * 2)
                effective_gain = 1 + (neuron.nmda_gain - 1) * voltage_gate
            else:
                effective_gain = neuron.nmda_gain
        
        # 7. Above threshold sigmoid
        with profiler.section("threshold_sigmoid"):
            above_threshold = torch.sigmoid((linear_sum - neuron.nmda_threshold) * 10)
        
        # 8. Subthreshold/suprathreshold blend
        with profiler.section("nmda_blend"):
            subthreshold_output = linear_sum * neuron.subthreshold_attenuation
            suprathreshold_output = linear_sum * effective_gain
            instantaneous_output = (
                subthreshold_output * (1 - above_threshold) +
                suprathreshold_output * above_threshold
            )
        
        # 9. Saturation
        with profiler.section("saturation"):
            instantaneous_output = neuron.saturation_level * torch.tanh(
                instantaneous_output / neuron.saturation_level
            )
        
        # 10. Plateau dynamics
        with profiler.section("plateau_dynamics"):
            neuron.branch_plateaus = neuron.branch_plateaus * neuron.plateau_decay
            plateau_boost = above_threshold * instantaneous_output * 0.5
            neuron.branch_plateaus = torch.maximum(neuron.branch_plateaus, plateau_boost)
        
        # 11. Max of instantaneous and plateau
        with profiler.section("max_output"):
            branch_output = torch.maximum(instantaneous_output, neuron.branch_plateaus)
        
        # 12. Apply coupling
        with profiler.section("coupling"):
            branch_output = branch_output * neuron.branch_coupling
        
        # 13. Sum branches
        with profiler.section("sum_branches"):
            g_exc_soma = branch_output.sum(dim=-1)
        
        # 14. Soma forward
        with profiler.section("soma_forward"):
            spikes, membrane = neuron.soma(g_exc_soma, g_inh)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    total_time = time.perf_counter() - total_start
    
    print(f"\nTotal time: {total_time*1000:.2f} ms ({total_time*1e6/n_iterations:.1f} μs/iteration)")
    
    profiler.report(total_time)
    
    # Also time the full forward() for comparison
    print("\n" + "=" * 60)
    print("FULL FORWARD() COMPARISON")
    print("=" * 60)
    
    neuron.reset_state()
    
    start = time.perf_counter()
    for _ in range(n_iterations):
        neuron(inputs, g_inh)
    if device.type == "cuda":
        torch.cuda.synchronize()
    full_time = time.perf_counter() - start
    
    print(f"Full forward(): {full_time*1000:.2f} ms ({full_time*1e6/n_iterations:.1f} μs/iteration)")
    print(f"Profiled sections: {total_time*1000:.2f} ms")
    print(f"Overhead from profiling: {(total_time - full_time)*1000:.2f} ms ({((total_time/full_time)-1)*100:.1f}%)")


def profile_with_torch_profiler(
    n_neurons: int = 100,
    n_branches: int = 4,
    inputs_per_branch: int = 50,
    device: str = "cpu"
):
    """Use torch.profiler for detailed breakdown."""
    
    device = torch.device(device)
    n_input = n_branches * inputs_per_branch
    
    branch_config = DendriticBranchConfig()
    config = DendriticNeuronConfig(
        n_branches=n_branches,
        inputs_per_branch=inputs_per_branch,
        branch_config=branch_config,
    )
    
    neuron = DendriticNeuron(n_neurons=n_neurons, config=config).to(device)
    neuron.reset_state(1)
    
    inputs = torch.zeros(1, n_input, device=device)
    inputs[:, ::5] = 0.1
    g_inh = torch.rand(1, n_neurons, device=device) * 0.2
    
    # Warm up
    for _ in range(10):
        neuron(inputs, g_inh)
    
    print(f"\nTorch Profiler on {device}:")
    print("-" * 60)
    
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU] + 
                   ([torch.profiler.ProfilerActivity.CUDA] if device.type == "cuda" else []),
        record_shapes=True,
    ) as prof:
        for _ in range(100):
            neuron(inputs, g_inh)
    
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--neurons", type=int, default=100)
    parser.add_argument("--branches", type=int, default=4)
    parser.add_argument("--inputs-per-branch", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--torch-profiler", action="store_true", help="Use torch.profiler")
    
    args = parser.parse_args()
    
    if args.torch_profiler:
        profile_with_torch_profiler(
            n_neurons=args.neurons,
            n_branches=args.branches,
            inputs_per_branch=args.inputs_per_branch,
            device=args.device,
        )
    else:
        profile_dendritic_forward(
            n_neurons=args.neurons,
            n_branches=args.branches,
            inputs_per_branch=args.inputs_per_branch,
            n_iterations=args.iterations,
            device=args.device,
        )
        
        if torch.cuda.is_available():
            print("\n" + "=" * 60)
            print("COMPARING CPU vs CUDA")
            print("=" * 60)
            
            profile_dendritic_forward(
                n_neurons=args.neurons,
                n_branches=args.branches,
                inputs_per_branch=args.inputs_per_branch,
                n_iterations=args.iterations,
                device="cuda",
            )
