"""Test utilities and common assertion helpers.

This module provides reusable utilities for testing THALIA components,
including validation helpers, assertion functions, and test data generators.
"""

from typing import Tuple
import torch
import numpy as np


# =============================================================================
# ASSERTION HELPERS
# =============================================================================

def assert_spike_train_valid(
    spikes: torch.Tensor,
    allow_float: bool = True,
    name: str = "spikes"
) -> None:
    """Assert that a spike train is valid.

    Args:
        spikes: Tensor containing spike data
        allow_float: If True, allow float values (0.0/1.0), otherwise require int
        name: Name of the tensor for error messages

    Raises:
        AssertionError: If spike train is invalid
    """
    # Check for NaN/Inf
    assert not torch.isnan(spikes).any(), f"{name} contains NaN values"
    assert not torch.isinf(spikes).any(), f"{name} contains Inf values"

    # Check that values are binary
    unique_values = torch.unique(spikes)
    assert len(unique_values) <= 2, \
        f"{name} must be binary, found values: {unique_values.tolist()}"
    assert all(v in [0, 1, 0.0, 1.0] for v in unique_values.tolist()), \
        f"{name} must contain only 0/1, found: {unique_values.tolist()}"

    # Check dtype
    if not allow_float:
        assert spikes.dtype in [torch.int32, torch.int64], \
            f"{name} must be integer type, got {spikes.dtype}"


def assert_weights_healthy(
    weights: torch.Tensor,
    min_val: float = 1e-6,
    max_val: float = 100.0,
    name: str = "weights"
) -> None:
    """Assert that weight matrix is in a healthy range.

    Args:
        weights: Weight tensor to validate
        min_val: Minimum acceptable weight value
        max_val: Maximum acceptable weight value
        name: Name of the weight matrix for error messages

    Raises:
        AssertionError: If weights are unhealthy
    """
    assert not torch.isnan(weights).any(), f"{name} contains NaN values"
    assert not torch.isinf(weights).any(), f"{name} contains Inf values"

    w_min = weights.min().item()
    w_max = weights.max().item()

    assert w_min >= min_val, \
        f"{name} collapsed: min={w_min:.2e} < threshold={min_val:.2e}"
    assert w_max <= max_val, \
        f"{name} exploded: max={w_max:.2e} > threshold={max_val:.2e}"

    # Check for dead weights (all zeros)
    if weights.numel() > 0:
        assert weights.abs().sum() > 0, f"{name} are all zero (dead weights)"


def assert_membrane_potential_valid(
    membrane: torch.Tensor,
    v_rest: float = 0.0,
    v_threshold: float = 1.0,
    tolerance: float = 2.0,
    name: str = "membrane"
) -> None:
    """Assert that membrane potential is in valid range.

    Args:
        membrane: Membrane potential tensor
        v_rest: Resting potential
        v_threshold: Spike threshold
        tolerance: How many standard deviations above threshold to allow
        name: Name for error messages

    Raises:
        AssertionError: If membrane potential is invalid
    """
    assert not torch.isnan(membrane).any(), f"{name} contains NaN"
    assert not torch.isinf(membrane).any(), f"{name} contains Inf"

    v_min = membrane.min().item()
    v_max = membrane.max().item()

    # Should be roughly between rest and a bit above threshold
    # Allow membrane to go ~2x below rest (for strong inhibition/adaptation)
    reasonable_min = v_rest - abs(v_rest) - 2.0
    reasonable_max = v_threshold * tolerance

    assert v_min > reasonable_min, \
        f"{name} too low: {v_min:.3f} (expected > {reasonable_min:.3f})"
    assert v_max < reasonable_max, \
        f"{name} too high: {v_max:.3f} (expected < {reasonable_max:.3f})"


def assert_activity_in_range(
    spikes: torch.Tensor,
    min_rate: float = 0.0,
    max_rate: float = 1.0,
    name: str = "activity"
) -> None:
    """Assert that spike rate is within acceptable range.

    Args:
        spikes: Spike tensor (any shape)
        min_rate: Minimum acceptable spike rate (fraction of 1)
        max_rate: Maximum acceptable spike rate (fraction of 1)
        name: Name for error messages

    Raises:
        AssertionError: If activity is out of range
    """
    spike_rate = spikes.float().mean().item()

    assert spike_rate >= min_rate, \
        f"{name} too low: {spike_rate:.3f} < {min_rate:.3f} (activity collapse?)"
    assert spike_rate <= max_rate, \
        f"{name} too high: {spike_rate:.3f} > {max_rate:.3f} (runaway activity?)"


def assert_tensor_shape(
    tensor: torch.Tensor,
    expected_shape: Tuple[int, ...],
    name: str = "tensor"
) -> None:
    """Assert tensor has expected shape.

    Args:
        tensor: Tensor to check
        expected_shape: Expected shape tuple
        name: Name for error messages

    Raises:
        AssertionError: If shape doesn't match
    """
    assert tensor.shape == expected_shape, \
        f"{name} shape mismatch: got {tensor.shape}, expected {expected_shape}"


def assert_convergence(
    values: list,
    window: int = 10,
    threshold: float = 0.01,
    name: str = "metric"
) -> None:
    """Assert that a metric has converged (stabilized).

    Args:
        values: List of metric values over time
        window: Window size to check for stability
        threshold: Maximum allowed standard deviation in window
        name: Name for error messages

    Raises:
        AssertionError: If metric hasn't converged
    """
    assert len(values) >= window, \
        f"Need at least {window} values to check convergence, got {len(values)}"

    recent_values = values[-window:]
    std = np.std(recent_values)

    assert std < threshold, \
        f"{name} hasn't converged: std={std:.4f} > {threshold:.4f} " \
        f"(recent values: {recent_values})"


def assert_monotonic_decrease(
    values: list,
    tolerance: float = 0.1,
    name: str = "metric"
) -> None:
    """Assert that values generally decrease (e.g., loss, error).

    Args:
        values: List of values over time
        tolerance: Fraction of non-decreasing steps allowed
        name: Name for error messages

    Raises:
        AssertionError: If not generally decreasing
    """
    assert len(values) >= 2, f"Need at least 2 values, got {len(values)}"

    first_half_mean = np.mean(values[:len(values)//2])
    second_half_mean = np.mean(values[len(values)//2:])

    assert second_half_mean < first_half_mean, \
        f"{name} not decreasing: first_half={first_half_mean:.4f}, " \
        f"second_half={second_half_mean:.4f}"


# =============================================================================
# TEST DATA GENERATORS
# =============================================================================

def generate_poisson_spikes(
    rate: float,
    n_neurons: int,
    n_timesteps: int,
    batch_size: int = 1,
    device: str = "cpu"
) -> torch.Tensor:
    """Generate Poisson spike train.

    Args:
        rate: Firing rate (probability per timestep, 0-1)
        n_neurons: Number of neurons
        n_timesteps: Number of timesteps
        batch_size: Batch size
        device: Device to create tensor on

    Returns:
        Spike tensor of shape (n_timesteps, batch_size, n_neurons)
    """
    assert 0 <= rate <= 1, f"Rate must be in [0,1], got {rate}"

    spikes = (torch.rand(n_timesteps, batch_size, n_neurons, device=device) < rate).float()
    return spikes


def generate_clustered_spikes(
    cluster_size: int,
    n_clusters: int,
    n_timesteps: int,
    batch_size: int = 1,
    device: str = "cpu"
) -> torch.Tensor:
    """Generate clustered spike patterns (for testing dendritic clustering).

    Args:
        cluster_size: Neurons per cluster
        n_clusters: Number of clusters
        n_timesteps: Number of timesteps
        batch_size: Batch size
        device: Device to create tensor on

    Returns:
        Spike tensor with clustered activity
    """
    n_neurons = cluster_size * n_clusters
    spikes = torch.zeros(n_timesteps, batch_size, n_neurons, device=device)

    # Activate random clusters at random times
    for t in range(n_timesteps):
        if torch.rand(1).item() > 0.7:  # 30% chance of cluster activation
            active_cluster = torch.randint(0, n_clusters, (1,)).item()
            start_idx = active_cluster * cluster_size
            end_idx = start_idx + cluster_size
            spikes[t, :, start_idx:end_idx] = 1.0

    return spikes


def generate_pattern_sequence(
    n_patterns: int,
    pattern_size: int,
    n_repeats: int = 5,
    device: str = "cpu"
) -> Tuple[torch.Tensor, list]:
    """Generate a sequence of repeating patterns.

    Useful for testing learning and memory.

    Args:
        n_patterns: Number of distinct patterns
        pattern_size: Size of each pattern
        n_repeats: Times to repeat each pattern
        device: Device to create tensor on

    Returns:
        Tuple of (pattern_sequence, pattern_labels)
    """
    # Create random patterns
    patterns = torch.randint(0, 2, (n_patterns, pattern_size), device=device).float()

    # Create sequence
    sequence = []
    labels = []
    for _ in range(n_repeats):
        for i in range(n_patterns):
            sequence.append(patterns[i])
            labels.append(i)

    sequence = torch.stack(sequence)
    return sequence, labels


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def count_parameters(module: torch.nn.Module) -> int:
    """Count trainable parameters in a module.

    Args:
        module: PyTorch module

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def get_memory_usage(device: str = "cuda") -> float:
    """Get current GPU memory usage in MB.

    Args:
        device: Device to check ("cuda" or specific device like "cuda:0")

    Returns:
        Memory usage in MB, or 0.0 if CUDA not available
    """
    if not torch.cuda.is_available():
        return 0.0

    return torch.cuda.memory_allocated(device) / 1024**2


def reset_cuda_memory():
    """Reset CUDA memory cache (useful between tests)."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


# =============================================================================
# TEST MARKERS & DECORATORS
# =============================================================================

def requires_cuda(func):
    """Decorator to skip test if CUDA is not available."""
    import pytest
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )(func)


def slow_test(func):
    """Decorator to mark a test as slow."""
    import pytest
    return pytest.mark.slow(func)


# =============================================================================
# ADVANCED TEST UTILITIES
# =============================================================================

def assert_no_memory_leak(
    func,
    n_iterations: int = 100,
    max_growth_mb: float = 10.0,
    device: str = "cuda"
) -> None:
    """Assert that function doesn't leak memory.
    
    Args:
        func: Function to test (no arguments)
        n_iterations: Number of iterations to run
        max_growth_mb: Maximum allowed memory growth in MB
        device: Device to monitor ("cuda" or "cpu")
        
    Raises:
        AssertionError: If memory leak detected
    """
    if device == "cuda" and not torch.cuda.is_available():
        return  # Skip if CUDA not available
    
    if device == "cuda":
        import tracemalloc
        
        # Reset CUDA memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Warmup
        func()
        torch.cuda.synchronize()
        
        # Measure initial memory
        initial_mb = torch.cuda.memory_allocated(device) / 1024**2
        
        # Run iterations
        for _ in range(n_iterations):
            func()
        
        torch.cuda.synchronize()
        
        # Measure final memory
        final_mb = torch.cuda.memory_allocated(device) / 1024**2
        growth_mb = final_mb - initial_mb
        
        assert growth_mb < max_growth_mb, \
            f"Memory leak detected: {growth_mb:.1f}MB growth over {n_iterations} iterations"
    else:
        # CPU memory monitoring with tracemalloc
        import tracemalloc
        tracemalloc.start()
        
        # Warmup
        func()
        
        snapshot1 = tracemalloc.take_snapshot()
        
        # Run iterations
        for _ in range(n_iterations):
            func()
        
        snapshot2 = tracemalloc.take_snapshot()
        
        # Calculate growth
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        if top_stats:
            growth_mb = sum(stat.size_diff for stat in top_stats) / 1024**2
            
            assert growth_mb < max_growth_mb, \
                f"Memory leak detected: {growth_mb:.1f}MB growth over {n_iterations} iterations"
        
        tracemalloc.stop()


def create_test_spike_pattern(
    pattern_type: str = 'poisson',
    **kwargs
) -> torch.Tensor:
    """Factory for creating common test spike patterns.
    
    Args:
        pattern_type: Type of pattern ('poisson', 'clustered', 'burst', 'regular')
        **kwargs: Pattern-specific parameters
        
    Returns:
        Spike tensor
        
    Examples:
        >>> spikes = create_test_spike_pattern('poisson', rate=0.1, n_neurons=100, n_timesteps=1000)
        >>> spikes = create_test_spike_pattern('burst', n_neurons=50, burst_length=10)
    """
    if pattern_type == 'poisson':
        return generate_poisson_spikes(**kwargs)
    
    elif pattern_type == 'clustered':
        return generate_clustered_spikes(**kwargs)
    
    elif pattern_type == 'burst':
        # Generate burst pattern
        n_neurons = kwargs.get('n_neurons', 100)
        n_timesteps = kwargs.get('n_timesteps', 100)
        batch_size = kwargs.get('batch_size', 1)
        burst_length = kwargs.get('burst_length', 5)
        burst_rate = kwargs.get('burst_rate', 0.1)
        device = kwargs.get('device', 'cpu')
        
        spikes = torch.zeros(n_timesteps, batch_size, n_neurons, device=device)
        
        # Add bursts at random times
        t = 0
        while t < n_timesteps:
            if torch.rand(1).item() < burst_rate:
                # Create burst
                burst_end = min(t + burst_length, n_timesteps)
                spikes[t:burst_end, :, :] = (torch.rand(burst_end - t, batch_size, n_neurons) > 0.3).float()
                t = burst_end
            else:
                t += 1
        
        return spikes
    
    elif pattern_type == 'regular':
        # Generate regular periodic spikes
        n_neurons = kwargs.get('n_neurons', 100)
        n_timesteps = kwargs.get('n_timesteps', 100)
        batch_size = kwargs.get('batch_size', 1)
        period = kwargs.get('period', 10)
        device = kwargs.get('device', 'cpu')
        
        spikes = torch.zeros(n_timesteps, batch_size, n_neurons, device=device)
        
        # Spike every 'period' timesteps
        for t in range(0, n_timesteps, period):
            spikes[t, :, :] = 1.0
        
        return spikes
    
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")


def assert_distribution_match(
    tensor: torch.Tensor,
    expected_mean: float,
    expected_std: float,
    tolerance: float = 0.1,
    name: str = "tensor"
) -> None:
    """Assert that tensor distribution matches expected statistics.
    
    Args:
        tensor: Tensor to check
        expected_mean: Expected mean value
        expected_std: Expected standard deviation
        tolerance: Tolerance as fraction of expected value
        name: Name for error messages
        
    Raises:
        AssertionError: If distribution doesn't match
    """
    actual_mean = tensor.mean().item()
    actual_std = tensor.std().item()
    
    mean_error = abs(actual_mean - expected_mean) / (abs(expected_mean) + 1e-6)
    std_error = abs(actual_std - expected_std) / (abs(expected_std) + 1e-6)
    
    assert mean_error < tolerance, \
        f"{name} mean mismatch: expected {expected_mean:.3f}, got {actual_mean:.3f} " \
        f"(error: {mean_error*100:.1f}%)"
    
    assert std_error < tolerance, \
        f"{name} std mismatch: expected {expected_std:.3f}, got {actual_std:.3f} " \
        f"(error: {std_error*100:.1f}%)"


def benchmark_function(
    func,
    n_warmup: int = 5,
    n_runs: int = 50
) -> dict:
    """Benchmark a function with warmup.
    
    Args:
        func: Function to benchmark (no arguments)
        n_warmup: Number of warmup runs
        n_runs: Number of timed runs
        
    Returns:
        Dict with timing statistics (mean, std, min, max, median in seconds)
    """
    import time
    
    # Warmup
    for _ in range(n_warmup):
        func()
    
    # Timed runs
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append(end - start)
    
    times_tensor = torch.tensor(times)
    
    return {
        'mean': times_tensor.mean().item(),
        'std': times_tensor.std().item(),
        'min': times_tensor.min().item(),
        'max': times_tensor.max().item(),
        'median': times_tensor.median().item(),
    }


def assert_performance_regression(
    func,
    baseline_time: float,
    max_regression: float = 0.2,
    n_runs: int = 50
) -> None:
    """Assert that function performance hasn't regressed.
    
    Args:
        func: Function to benchmark
        baseline_time: Baseline time in seconds
        max_regression: Maximum allowed regression (fraction, e.g., 0.2 = 20%)
        n_runs: Number of runs for benchmark
        
    Raises:
        AssertionError: If performance regressed beyond threshold
    """
    stats = benchmark_function(func, n_runs=n_runs)
    current_time = stats['mean']
    
    regression = (current_time - baseline_time) / baseline_time
    
    assert regression < max_regression, \
        f"Performance regression: {regression*100:.1f}% slower " \
        f"(baseline: {baseline_time*1000:.2f}ms, current: {current_time*1000:.2f}ms)"


def save_test_artifact(
    artifact,
    name: str,
    test_name: str,
    format: str = 'pt'
) -> str:
    """Save test artifact for debugging.
    
    Args:
        artifact: Object to save (tensor, dict, etc.)
        name: Artifact name
        test_name: Name of the test
        format: Format ('pt' for torch, 'json', 'pkl')
        
    Returns:
        Path to saved artifact
    """
    import os
    import json
    import pickle
    
    # Create artifacts directory
    artifacts_dir = os.path.join('tests', 'artifacts', test_name)
    os.makedirs(artifacts_dir, exist_ok=True)
    
    if format == 'pt':
        path = os.path.join(artifacts_dir, f'{name}.pt')
        torch.save(artifact, path)
    elif format == 'json':
        path = os.path.join(artifacts_dir, f'{name}.json')
        with open(path, 'w') as f:
            json.dump(artifact, f, indent=2)
    elif format == 'pkl':
        path = os.path.join(artifacts_dir, f'{name}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(artifact, f)
    else:
        raise ValueError(f"Unknown format: {format}")
    
    return path
