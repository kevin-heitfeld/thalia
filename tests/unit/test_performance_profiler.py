"""
Unit tests for enhanced performance profiler.

Tests per-region timing and spike statistics tracking.
"""

import time

from thalia.diagnostics.performance_profiler import PerformanceProfiler, PerformanceStats


def test_basic_profiling():
    """Test basic timing functionality."""
    profiler = PerformanceProfiler(window_size=10)

    # Simulate forward passes
    for _ in range(5):
        profiler.start_forward()
        time.sleep(0.001)  # 1ms
        profiler.end_forward()
        profiler.record_step()

    stats = profiler.get_stats()

    # Should have recorded timing
    assert stats.avg_forward_ms > 0
    assert stats.steps_per_sec > 0


def test_per_region_timing():
    """Test per-region profiling."""
    profiler = PerformanceProfiler(window_size=10)

    regions = ["cortex", "hippocampus", "striatum"]

    for _ in range(3):
        for region in regions:
            profiler.start_region(region)
            time.sleep(0.001)  # 1ms per region
            profiler.end_region(region)

    stats = profiler.get_stats()

    # Test contract: should track all expected regions
    expected_regions = ["cortex", "hippocampus", "striatum"]
    assert len(stats.region_times_ms) == len(
        expected_regions
    ), f"Should track {len(expected_regions)} regions"
    for region in expected_regions:
        assert region in stats.region_times_ms, f"Should track region '{region}'"
        assert stats.region_times_ms[region] > 0, f"Region '{region}' should have positive timing"


def test_spike_statistics():
    """Test spike counting and firing rate tracking."""
    profiler = PerformanceProfiler(window_size=10)

    # Record spikes for different regions
    profiler.record_spikes("cortex", 50, 1000)  # 5% firing rate
    profiler.record_spikes("cortex", 60, 1000)  # 6%
    profiler.record_spikes("cortex", 40, 1000)  # 4%

    profiler.record_spikes("hippocampus", 10, 100)  # 10%
    profiler.record_spikes("hippocampus", 15, 100)  # 15%

    stats = profiler.get_stats()

    # Should have spike stats for both regions
    assert "cortex" in stats.spike_stats
    assert "hippocampus" in stats.spike_stats

    # Cortex average should be ~5%
    cortex_stats = stats.spike_stats["cortex"]
    assert 0.04 < cortex_stats["avg_firing_rate"] < 0.06

    # Hippocampus average should be ~12.5%
    hipp_stats = stats.spike_stats["hippocampus"]
    assert 0.10 < hipp_stats["avg_firing_rate"] < 0.15

    # Total spikes should be tracked
    assert "total_avg_spikes" in stats.spike_stats


def test_reset():
    """Test that reset clears all data."""
    profiler = PerformanceProfiler(window_size=10)

    # Add some data
    profiler.start_forward()
    time.sleep(0.001)
    profiler.end_forward()
    profiler.record_step()

    profiler.start_region("test_region")
    time.sleep(0.001)
    profiler.end_region("test_region")

    profiler.record_spikes("test_region", 10, 100)

    # Reset
    profiler.reset()

    # Contract: after reset, profiler should work normally
    profiler.start_forward()
    time.sleep(0.001)
    profiler.end_forward()
    profiler.record_step()

    stats = profiler.get_stats()

    # Contract: should produce valid stats after reset
    assert stats.avg_forward_ms > 0.0, "Should track timing after reset"
    assert stats.steps_per_sec > 0.0, "Should track steps after reset"


def test_window_size_limiting():
    """Test that circular buffers respect window size."""
    window_size = 5
    profiler = PerformanceProfiler(window_size=window_size)

    # Add more samples than window size
    for _ in range(10):
        profiler.start_forward()
        time.sleep(0.001)
        profiler.end_forward()

    # Should only keep last window_size samples
    assert len(profiler.forward_times) == window_size


def test_performance_stats_dataclass():
    """Test PerformanceStats dataclass has required structure."""
    stats = PerformanceStats()

    # Test contract: dataclass should have expected fields for profiling
    assert hasattr(stats, "steps_per_sec"), "Should have steps_per_sec field"
    assert hasattr(stats, "avg_forward_ms"), "Should have avg_forward_ms field"
    assert hasattr(stats, "region_times_ms"), "Should have region_times_ms dict"
    assert hasattr(stats, "spike_stats"), "Should have spike_stats dict"

    # Test contract: dict fields should be mutable
    stats.region_times_ms["test_region"] = 2.5
    stats.spike_stats["test_region"] = {"active": 10}
    assert "test_region" in stats.region_times_ms, "Should allow adding region times"
    assert "test_region" in stats.spike_stats, "Should allow adding spike stats"


def test_print_summary_doesnt_crash():
    """Test that print_summary runs without errors."""
    profiler = PerformanceProfiler(window_size=10)

    # Add some data
    profiler.start_forward()
    time.sleep(0.001)
    profiler.end_forward()
    profiler.record_step()

    profiler.start_region("cortex")
    time.sleep(0.001)
    profiler.end_region("cortex")

    profiler.record_spikes("cortex", 50, 1000)

    # Should not raise
    profiler.print_summary()


def test_concurrent_region_profiling():
    """Test profiling multiple regions in quick succession."""
    profiler = PerformanceProfiler(window_size=10)

    # Simulate brain forward pass with multiple regions
    for _ in range(3):
        profiler.start_forward()

        # Process each region
        for region in ["input", "cortex", "output"]:
            profiler.start_region(region)
            time.sleep(0.001)
            profiler.end_region(region)
            profiler.record_spikes(region, 10, 100)

        profiler.end_forward()
        profiler.record_step()

    stats = profiler.get_stats()

    # Test contract: should track all expected regions
    expected_region_count = 3
    expected_regions = ["input", "cortex", "output"]
    assert (
        len(stats.region_times_ms) == expected_region_count
    ), f"Should track {expected_region_count} regions"
    assert (
        len(stats.spike_stats) >= expected_region_count
    ), f"Should track spike stats for at least {expected_region_count} regions (+ total_avg_spikes)"

    # Forward time should be sum of region times (approximately)
    total_region_time = sum(stats.region_times_ms.values())
    # Allow some tolerance for overhead
    assert (
        0.5 * total_region_time < stats.avg_forward_ms < 2.0 * total_region_time
    ), "Forward time should be approximately sum of region times (with overhead tolerance)"
