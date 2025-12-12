"""
Performance Profiler: Track timing and memory metrics during training.

This module provides lightweight performance monitoring for neural network
training, tracking:
- Forward pass timing
- Steps per second throughput
- CPU/GPU memory usage
- Tensor allocation statistics

Designed for minimal overhead (<2% slowdown) while providing comprehensive
performance insights.

Usage:
======
    from thalia.diagnostics import PerformanceProfiler

    profiler = PerformanceProfiler()

    # In training loop:
    profiler.start_forward()
    output = brain.forward(input_data)
    profiler.end_forward()
    profiler.record_step()

    # Periodic memory sampling:
    if step % 100 == 0:
        profiler.record_memory(brain)

    # Get statistics:
    stats = profiler.get_stats()
    print(f"Steps/sec: {stats['steps_per_sec']:.2f}")
    print(f"Forward time: {stats['avg_forward_ms']:.2f} ms")
    print(f"GPU memory: {stats['gpu_memory_mb']:.1f} MB")

Author: Thalia Project
Date: December 12, 2025
"""

from __future__ import annotations

import time
from collections import deque
from typing import Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
import torch


# Optional psutil for system memory (graceful degradation if not available)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore
    PSUTIL_AVAILABLE = False


@dataclass
class PerformanceStats:
    """Performance statistics snapshot.

    Attributes:
        steps_per_sec: Training throughput (steps/second)
        avg_forward_ms: Average forward pass time (milliseconds)
        std_forward_ms: Std dev of forward pass time
        cpu_memory_mb: CPU memory usage (MB)
        gpu_memory_mb: GPU memory usage (MB)
        gpu_memory_allocated_mb: GPU memory allocated by PyTorch (MB)
        gpu_memory_reserved_mb: GPU memory reserved by PyTorch (MB)
        tensor_count: Approximate number of tensors in brain
        total_parameters: Total trainable parameters
        region_times_ms: Per-region average forward time (milliseconds)
        spike_stats: Spike propagation statistics
    """
    steps_per_sec: float = 0.0
    avg_forward_ms: float = 0.0
    std_forward_ms: float = 0.0
    cpu_memory_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    gpu_memory_allocated_mb: float = 0.0
    gpu_memory_reserved_mb: float = 0.0
    tensor_count: int = 0
    total_parameters: int = 0
    region_times_ms: Dict[str, float] = None
    spike_stats: Dict[str, Any] = None

    def __post_init__(self):
        """Initialize default mutable fields."""
        if self.region_times_ms is None:
            self.region_times_ms = {}
        if self.spike_stats is None:
            self.spike_stats = {}


class PerformanceProfiler:
    """Lightweight performance profiler for training.

    Tracks timing and memory metrics with minimal overhead.
    """

    def __init__(self, window_size: int = 100):
        """Initialize profiler.

        Args:
            window_size: Number of samples to keep for moving averages
        """
        self.window_size = window_size

        # Timing buffers (circular buffers for efficiency)
        self.forward_times: deque = deque(maxlen=window_size)
        self.step_times: deque = deque(maxlen=window_size)

        # Per-region timing (region_name -> deque of times)
        self.region_times: Dict[str, deque] = {}

        # Spike propagation tracking
        self.spike_counts: deque = deque(maxlen=window_size)
        self.spike_rates: Dict[str, deque] = {}  # region_name -> firing rates

        # Memory buffers (sample less frequently)
        self.memory_samples: deque = deque(maxlen=window_size)

        # State
        self._forward_start: Optional[float] = None
        self._region_start: Optional[float] = None
        self._current_region: Optional[str] = None
        self._last_step_time: float = time.time()
        self._total_steps: int = 0

        # Cached values
        self._last_stats: Optional[PerformanceStats] = None

    def reset(self) -> None:
        """Reset all counters and buffers."""
        self.forward_times.clear()
        self.step_times.clear()
        self.region_times.clear()
        self.spike_counts.clear()
        self.spike_rates.clear()
        self.memory_samples.clear()
        self._forward_start = None
        self._region_start = None
        self._last_step_time = time.time()
        self._total_steps = 0
        self._last_stats = None

    def start_forward(self) -> None:
        """Start timing a forward pass.

        Call this immediately before brain.forward().
        """
        self._forward_start = time.time()

    def end_forward(self) -> None:
        """End timing a forward pass.

        Call this immediately after brain.forward().
        """
        if self._forward_start is None:
            return  # start_forward() was not called

        duration = time.time() - self._forward_start
        self.forward_times.append(duration)
        self._forward_start = None

    def start_region(self, region_name: str) -> None:
        """Start timing a region's forward pass.

        Args:
            region_name: Name of the region being profiled
        """
        self._region_start = time.time()
        self._current_region = region_name

    def end_region(self, region_name: str) -> None:
        """End timing a region's forward pass.

        Args:
            region_name: Name of the region (should match start_region())
        """
        if self._region_start is None:
            return

        duration = time.time() - self._region_start

        # Initialize deque for this region if needed
        if region_name not in self.region_times:
            self.region_times[region_name] = deque(maxlen=self.window_size)

        self.region_times[region_name].append(duration)
        self._region_start = None

    def record_spikes(self, region_name: str, spike_count: int, n_neurons: int) -> None:
        """Record spike statistics for a region.

        Args:
            region_name: Name of the region
            spike_count: Number of neurons that spiked
            n_neurons: Total number of neurons in region
        """
        firing_rate = spike_count / n_neurons if n_neurons > 0 else 0.0

        # Initialize deque for this region if needed
        if region_name not in self.spike_rates:
            self.spike_rates[region_name] = deque(maxlen=self.window_size)

        self.spike_rates[region_name].append(firing_rate)

        # Also track total spikes
        self.spike_counts.append(spike_count)

    def record_step(self) -> None:
        """Record completion of a training step.

        Call this once per training iteration.
        """
        now = time.time()
        step_duration = now - self._last_step_time
        self.step_times.append(step_duration)
        self._last_step_time = now
        self._total_steps += 1

    def record_memory(self, brain: Any, device: Optional[str] = None) -> None:
        """Record current memory usage.

        Call this periodically (e.g., every 100 steps) to avoid overhead.

        Args:
            brain: Brain object to analyze
            device: Device to check ('cpu', 'cuda', or auto-detect)
        """
        if device is None:
            device = str(brain.device) if hasattr(brain, 'device') else 'cpu'

        memory_info = {
            'cpu_mb': self._get_cpu_memory(),
            'gpu_mb': 0.0,
            'gpu_allocated_mb': 0.0,
            'gpu_reserved_mb': 0.0,
            'tensor_count': self._count_brain_tensors(brain),
            'total_parameters': self._count_parameters(brain),
        }

        # GPU memory (if available)
        if 'cuda' in device and torch.cuda.is_available():
            gpu_stats = self._get_gpu_memory(device)
            memory_info.update(gpu_stats)

        self.memory_samples.append(memory_info)

    def get_stats(self) -> PerformanceStats:
        """Get current performance statistics.

        Returns:
            PerformanceStats object with current metrics
        """
        stats = PerformanceStats()

        # Throughput (steps per second)
        if self.step_times:
            avg_step_time = np.mean(self.step_times)
            stats.steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0.0

        # Forward pass timing
        if self.forward_times:
            stats.avg_forward_ms = np.mean(self.forward_times) * 1000.0
            stats.std_forward_ms = np.std(self.forward_times) * 1000.0

        # Per-region timing
        stats.region_times_ms = {}
        for region_name, times in self.region_times.items():
            if times:
                stats.region_times_ms[region_name] = float(np.mean(times) * 1000.0)

        # Spike statistics
        stats.spike_stats = {}
        for region_name, rates in self.spike_rates.items():
            if rates:
                stats.spike_stats[region_name] = {
                    'avg_firing_rate': float(np.mean(rates)),
                    'std_firing_rate': float(np.std(rates)),
                }

        if self.spike_counts:
            stats.spike_stats['total_avg_spikes'] = float(np.mean(self.spike_counts))

        # Memory (latest sample)
        if self.memory_samples:
            latest = self.memory_samples[-1]
            stats.cpu_memory_mb = latest['cpu_mb']
            stats.gpu_memory_mb = latest['gpu_mb']
            stats.gpu_memory_allocated_mb = latest['gpu_allocated_mb']
            stats.gpu_memory_reserved_mb = latest['gpu_reserved_mb']
            stats.tensor_count = latest['tensor_count']
            stats.total_parameters = latest['total_parameters']

        self._last_stats = stats
        return stats

    def get_metrics_dict(self) -> Dict[str, float]:
        """Get metrics as flat dictionary for logging.

        Returns:
            Dictionary of metric_name -> value
        """
        stats = self.get_stats()

        return {
            'performance/steps_per_sec': stats.steps_per_sec,
            'performance/avg_forward_ms': stats.avg_forward_ms,
            'performance/std_forward_ms': stats.std_forward_ms,
            'memory/cpu_mb': stats.cpu_memory_mb,
            'memory/gpu_mb': stats.gpu_memory_mb,
            'memory/gpu_allocated_mb': stats.gpu_memory_allocated_mb,
            'memory/gpu_reserved_mb': stats.gpu_memory_reserved_mb,
            'memory/tensor_count': float(stats.tensor_count),
            'memory/total_parameters': float(stats.total_parameters),
        }

    def print_summary(self) -> None:
        """Print a formatted summary of current performance."""
        stats = self.get_stats()

        print("\n" + "="*60)
        print("⚡ PERFORMANCE SUMMARY")
        print("="*60)
        print(f"  Throughput:      {stats.steps_per_sec:>8.2f} steps/sec")
        print(f"  Forward pass:    {stats.avg_forward_ms:>8.2f} ± {stats.std_forward_ms:.2f} ms")
        print(f"  Total steps:     {self._total_steps:>8,d}")
        print()

        # Per-region timing
        if stats.region_times_ms:
            print("  Region timing:")
            for region_name, avg_time_ms in sorted(stats.region_times_ms.items()):
                print(f"    {region_name:<20} {avg_time_ms:>8.2f} ms")
            print()

        # Spike statistics
        if stats.spike_stats:
            print("  Spike activity:")
            for region_name, spike_info in sorted(stats.spike_stats.items()):
                if region_name != 'total_avg_spikes' and isinstance(spike_info, dict):
                    avg_rate = spike_info['avg_firing_rate']
                    std_rate = spike_info['std_firing_rate']
                    print(f"    {region_name:<20} {avg_rate*100:>6.2f}% ± {std_rate*100:.2f}%")
            if 'total_avg_spikes' in stats.spike_stats:
                print(f"    Total avg spikes:    {stats.spike_stats['total_avg_spikes']:>8.1f}")
            print()

        print(f"  CPU memory:      {stats.cpu_memory_mb:>8.1f} MB")
        print(f"  GPU memory:      {stats.gpu_memory_mb:>8.1f} MB")
        print(f"  GPU allocated:   {stats.gpu_memory_allocated_mb:>8.1f} MB")
        print(f"  GPU reserved:    {stats.gpu_memory_reserved_mb:>8.1f} MB")
        print(f"  Tensor count:    {stats.tensor_count:>8,d}")
        print(f"  Parameters:      {stats.total_parameters:>8,d}")
        print("="*60 + "\n")

    # =========================================================================
    # Internal helper methods
    # =========================================================================

    def _get_cpu_memory(self) -> float:
        """Get current process CPU memory usage in MB.

        Returns:
            Memory usage in megabytes (0.0 if psutil unavailable)
        """
        if not PSUTIL_AVAILABLE:
            return 0.0

        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 ** 2)  # Convert to MB
        except Exception:
            return 0.0

    def _get_gpu_memory(self, device: str) -> Dict[str, float]:
        """Get GPU memory usage in MB.

        Args:
            device: Device string ('cuda', 'cuda:0', etc.)

        Returns:
            Dictionary with gpu_mb, gpu_allocated_mb, gpu_reserved_mb
        """
        if not torch.cuda.is_available():
            return {
                'gpu_mb': 0.0,
                'gpu_allocated_mb': 0.0,
                'gpu_reserved_mb': 0.0,
            }

        try:
            # Extract device index
            if ':' in device:
                device_idx = int(device.split(':')[1])
            else:
                device_idx = 0

            # PyTorch memory stats
            allocated = torch.cuda.memory_allocated(device_idx) / (1024 ** 2)
            reserved = torch.cuda.memory_reserved(device_idx) / (1024 ** 2)

            # Total GPU memory (if available via nvidia-smi)
            try:
                _total_memory = torch.cuda.get_device_properties(device_idx).total_memory
                used_memory = allocated  # Approximation
                gpu_mb = used_memory
            except Exception:
                gpu_mb = allocated

            return {
                'gpu_mb': gpu_mb,
                'gpu_allocated_mb': allocated,
                'gpu_reserved_mb': reserved,
            }

        except Exception:
            return {
                'gpu_mb': 0.0,
                'gpu_allocated_mb': 0.0,
                'gpu_reserved_mb': 0.0,
            }

    def _count_brain_tensors(self, brain: Any) -> int:
        """Count tensors in brain (approximate).

        Args:
            brain: Brain object

        Returns:
            Approximate tensor count
        """
        try:
            count = 0

            # Count parameters (weights)
            if hasattr(brain, 'parameters'):
                count += sum(1 for _ in brain.parameters())

            # Count buffers (states)
            if hasattr(brain, 'buffers'):
                count += sum(1 for _ in brain.buffers())

            return count

        except Exception:
            return 0

    def _count_parameters(self, brain: Any) -> int:
        """Count total trainable parameters.

        Args:
            brain: Brain object

        Returns:
            Total parameter count
        """
        try:
            if hasattr(brain, 'parameters'):
                return sum(p.numel() for p in brain.parameters() if p.requires_grad)
            return 0
        except Exception:
            return 0


def quick_profile(brain: Any, n_steps: int = 100, verbose: bool = True) -> PerformanceStats:
    """Quick performance profiling.

    Args:
        brain: Brain object to profile
        n_steps: Number of forward passes to profile
        verbose: Whether to print summary

    Returns:
        PerformanceStats with results
    """
    profiler = PerformanceProfiler()

    # Warm-up (exclude from timing)
    dummy_input = torch.randn(100, device=brain.device if hasattr(brain, 'device') else 'cpu')
    _ = brain.forward(dummy_input, n_timesteps=10)

    # Profile
    for i in range(n_steps):
        profiler.start_forward()
        _ = brain.forward(dummy_input, n_timesteps=10)
        profiler.end_forward()
        profiler.record_step()

        if i % 10 == 0:
            profiler.record_memory(brain)

    stats = profiler.get_stats()

    if verbose:
        profiler.print_summary()

    return stats
