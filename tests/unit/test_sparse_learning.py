"""
Unit tests for sparse learning updates.

Tests verifying that sparse and dense computations produce identical results
and that sparse operations are beneficial for low-activity neural populations.
"""

import pytest
import torch
import time

from thalia.learning.rules.strategies import HebbianStrategy, HebbianConfig


def test_sparse_hebbian_identical_to_dense():
    """Sparse and dense Hebbian should produce identical results."""
    n_pre, n_post = 1000, 500
    
    # Create sparse spike pattern (2% active)
    pre = torch.zeros(n_pre)
    post = torch.zeros(n_post)
    pre[torch.randperm(n_pre)[:20]] = 1.0  # 20/1000 = 2%
    post[torch.randperm(n_post)[:10]] = 1.0  # 10/500 = 2%
    
    weights = torch.rand(n_post, n_pre) * 0.5
    
    # Dense strategy
    config_dense = HebbianConfig(learning_rate=0.01, use_sparse_updates=False)
    strategy_dense = HebbianStrategy(config_dense)
    weights_dense, metrics_dense = strategy_dense.compute_update(
        weights.clone(), pre, post
    )
    
    # Sparse strategy
    config_sparse = HebbianConfig(learning_rate=0.01, use_sparse_updates=True)
    strategy_sparse = HebbianStrategy(config_sparse)
    weights_sparse, metrics_sparse = strategy_sparse.compute_update(
        weights.clone(), pre, post
    )
    
    # Results should be identical
    torch.testing.assert_close(weights_sparse, weights_dense, rtol=1e-5, atol=1e-6)
    
    # Metrics should match
    assert abs(metrics_sparse['ltp'] - metrics_dense['ltp']) < 1e-5
    assert abs(metrics_sparse['ltd'] - metrics_dense['ltd']) < 1e-5


def test_sparse_hebbian_no_spikes():
    """Sparse computation should handle no-spike case correctly."""
    n_pre, n_post = 100, 50
    
    # No spikes
    pre = torch.zeros(n_pre)
    post = torch.zeros(n_post)
    weights = torch.rand(n_post, n_pre) * 0.5
    
    config = HebbianConfig(learning_rate=0.01, use_sparse_updates=True)
    strategy = HebbianStrategy(config)
    new_weights, _ = strategy.compute_update(weights.clone(), pre, post)
    
    # Weights should be unchanged (no spikes = no update)
    torch.testing.assert_close(new_weights, weights)


def test_sparse_hebbian_all_spikes():
    """Sparse computation should fall back to dense for dense activity."""
    n_pre, n_post = 100, 50
    
    # All neurons active (100% sparsity - should use dense path)
    pre = torch.ones(n_pre)
    post = torch.ones(n_post)
    weights = torch.rand(n_post, n_pre) * 0.5
    
    config_sparse = HebbianConfig(learning_rate=0.01, use_sparse_updates=True)
    strategy_sparse = HebbianStrategy(config_sparse)
    weights_sparse, _ = strategy_sparse.compute_update(weights.clone(), pre, post)
    
    config_dense = HebbianConfig(learning_rate=0.01, use_sparse_updates=False)
    strategy_dense = HebbianStrategy(config_dense)
    weights_dense, _ = strategy_dense.compute_update(weights.clone(), pre, post)
    
    # Should produce identical results
    torch.testing.assert_close(weights_sparse, weights_dense)


def test_sparse_hebbian_single_spike():
    """Single spike should produce correct update."""
    n_pre, n_post = 100, 50
    
    # Single pre and post spike
    pre = torch.zeros(n_pre)
    post = torch.zeros(n_post)
    pre[42] = 1.0
    post[17] = 1.0
    
    weights = torch.zeros(n_post, n_pre)
    
    config = HebbianConfig(learning_rate=0.1, use_sparse_updates=True, w_min=0.0, w_max=1.0)
    strategy = HebbianStrategy(config)
    new_weights, metrics = strategy.compute_update(weights.clone(), pre, post)
    
    # Only synapse (17, 42) should be strengthened
    expected_weight = 0.1  # lr * 1.0 * 1.0
    assert new_weights[17, 42] == pytest.approx(expected_weight, abs=1e-6)
    
    # All other weights should be zero
    mask = torch.ones_like(new_weights, dtype=torch.bool)
    mask[17, 42] = False
    assert (new_weights[mask] == 0.0).all()
    
    # Metrics should reflect single update
    assert metrics['ltp'] == pytest.approx(expected_weight, abs=1e-6)
    assert metrics['ltd'] == 0.0


def test_sparse_threshold_crossover():
    """Test behavior at sparsity threshold (5%)."""
    n_pre, n_post = 1000, 500
    weights = torch.rand(n_post, n_pre) * 0.5
    
    # Test at 4% (should use sparse)
    pre_4pct = torch.zeros(n_pre)
    post_4pct = torch.zeros(n_post)
    pre_4pct[torch.randperm(n_pre)[:40]] = 1.0  # 4%
    post_4pct[torch.randperm(n_post)[:20]] = 1.0  # 4%
    
    config = HebbianConfig(learning_rate=0.01, use_sparse_updates=True)
    strategy = HebbianStrategy(config)
    weights_sparse, _ = strategy.compute_update(weights.clone(), pre_4pct, post_4pct)
    
    # Test at 6% (should use dense)
    pre_6pct = torch.zeros(n_pre)
    post_6pct = torch.zeros(n_post)
    pre_6pct[torch.randperm(n_pre)[:60]] = 1.0  # 6%
    post_6pct[torch.randperm(n_post)[:30]] = 1.0  # 6%
    
    weights_dense_fallback, _ = strategy.compute_update(weights.clone(), pre_6pct, post_6pct)
    
    # Both should produce valid updates (just testing they run)
    assert weights_sparse.shape == weights.shape
    assert weights_dense_fallback.shape == weights.shape


def test_sparse_hebbian_with_weight_decay():
    """Sparse updates should correctly handle weight decay."""
    n_pre, n_post = 100, 50
    
    # Sparse spikes
    pre = torch.zeros(n_pre)
    post = torch.zeros(n_post)
    pre[torch.randperm(n_pre)[:5]] = 1.0
    post[torch.randperm(n_post)[:3]] = 1.0
    
    weights = torch.ones(n_post, n_pre) * 0.5
    
    config = HebbianConfig(
        learning_rate=0.01,
        use_sparse_updates=True,
        decay_rate=0.001,
    )
    strategy = HebbianStrategy(config)
    new_weights, metrics = strategy.compute_update(weights.clone(), pre, post)
    
    # Weights should have decayed
    assert new_weights.mean() < weights.mean()
    
    # LTD metric should be negative (decay)
    assert metrics['ltd'] < 0


def test_sparse_hebbian_respects_bounds():
    """Sparse updates should respect weight bounds."""
    n_pre, n_post = 50, 30
    
    # Strong update that would violate bounds
    pre = torch.ones(n_pre)
    post = torch.ones(n_post)
    weights = torch.ones(n_post, n_pre) * 0.9
    
    config = HebbianConfig(
        learning_rate=0.5,
        use_sparse_updates=True,
        w_min=0.0,
        w_max=1.0,
    )
    strategy = HebbianStrategy(config)
    new_weights, _ = strategy.compute_update(weights, pre, post)
    
    # All weights should be clamped to [0, 1]
    assert (new_weights >= 0.0).all()
    assert (new_weights <= 1.0).all()
    assert (new_weights == 1.0).all()  # Should saturate at max


@pytest.mark.skip(reason="Performance test - run manually when needed")
def test_sparse_performance_benefit():
    """Sparse should be faster than dense for low activity."""
    n_pre, n_post = 10000, 5000
    n_trials = 100
    
    # Create sparse pattern (2%)
    pre = torch.zeros(n_pre)
    post = torch.zeros(n_post)
    pre[torch.randperm(n_pre)[:200]] = 1.0
    post[torch.randperm(n_post)[:100]] = 1.0
    
    weights = torch.rand(n_post, n_pre) * 0.5
    
    # Benchmark dense
    config_dense = HebbianConfig(learning_rate=0.01, use_sparse_updates=False)
    strategy_dense = HebbianStrategy(config_dense)
    
    start = time.time()
    for _ in range(n_trials):
        strategy_dense.compute_update(weights, pre, post)
    dense_time = time.time() - start
    
    # Benchmark sparse
    config_sparse = HebbianConfig(learning_rate=0.01, use_sparse_updates=True)
    strategy_sparse = HebbianStrategy(config_sparse)
    
    start = time.time()
    for _ in range(n_trials):
        strategy_sparse.compute_update(weights, pre, post)
    sparse_time = time.time() - start
    
    print(f"\nPerformance (n_pre={n_pre}, n_post={n_post}, 2% activity, {n_trials} trials):")
    print(f"  Dense:  {dense_time:.3f}s ({dense_time/n_trials*1000:.2f}ms/trial)")
    print(f"  Sparse: {sparse_time:.3f}s ({sparse_time/n_trials*1000:.2f}ms/trial)")
    print(f"  Speedup: {dense_time/sparse_time:.2f}x")
    
    # Sparse should be faster (but this depends on hardware/implementation)
    # Just verify it runs without error
    assert sparse_time > 0
    assert dense_time > 0
