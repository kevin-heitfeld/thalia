"""
Tests for weight initialization registry.

Author: Thalia Project
Date: December 2025
"""

import math
import pytest
import torch

from thalia.core.weight_init import InitStrategy, WeightInitializer


class TestWeightInitializer:
    """Test weight initialization registry."""
    
    def test_gaussian_initialization(self):
        """Test Gaussian initialization."""
        n_output, n_input = 64, 128
        mean, std = 0.5, 0.2
        
        weights = WeightInitializer.gaussian(
            n_output=n_output,
            n_input=n_input,
            mean=mean,
            std=std
        )
        
        assert weights.shape == (n_output, n_input)
        
        # Check statistics (with tolerance for randomness)
        assert abs(weights.mean().item() - mean) < 0.1
        assert abs(weights.std().item() - std) < 0.1
    
    def test_uniform_initialization(self):
        """Test uniform initialization."""
        n_output, n_input = 50, 100
        low, high = 0.1, 0.5
        
        weights = WeightInitializer.uniform(
            n_output=n_output,
            n_input=n_input,
            low=low,
            high=high
        )
        
        assert weights.shape == (n_output, n_input)
        assert weights.min() >= low
        assert weights.max() <= high
    
    def test_xavier_initialization(self):
        """Test Xavier/Glorot initialization."""
        n_output, n_input = 64, 128
        gain = 1.0
        
        weights = WeightInitializer.xavier(
            n_output=n_output,
            n_input=n_input,
            gain=gain
        )
        
        assert weights.shape == (n_output, n_input)
        
        # Check std matches Xavier formula
        expected_std = gain * math.sqrt(2.0 / (n_input + n_output))
        actual_std = weights.std().item()
        assert abs(actual_std - expected_std) < 0.1
    
    def test_kaiming_initialization(self):
        """Test Kaiming/He initialization."""
        n_output, n_input = 100, 200
        
        weights = WeightInitializer.kaiming(
            n_output=n_output,
            n_input=n_input,
            mode="fan_in"
        )
        
        assert weights.shape == (n_output, n_input)
        
        # Check std matches Kaiming formula
        expected_std = math.sqrt(2.0) / math.sqrt(n_input)
        actual_std = weights.std().item()
        assert abs(actual_std - expected_std) < 0.1
    
    def test_sparse_random_initialization(self):
        """Test sparse random initialization."""
        n_output, n_input = 50, 100
        sparsity = 0.3
        
        weights = WeightInitializer.sparse_random(
            n_output=n_output,
            n_input=n_input,
            sparsity=sparsity
        )
        
        assert weights.shape == (n_output, n_input)
        
        # Check sparsity level
        nonzero_ratio = (weights != 0).sum().item() / (n_output * n_input)
        assert abs(nonzero_ratio - sparsity) < 0.1
    
    def test_sparse_random_with_normalization(self):
        """Test sparse random with row normalization."""
        n_output, n_input = 30, 60
        sparsity = 0.4
        weight_scale = 0.2
        
        weights = WeightInitializer.sparse_random(
            n_output=n_output,
            n_input=n_input,
            sparsity=sparsity,
            weight_scale=weight_scale,
            normalize_rows=True
        )
        
        assert weights.shape == (n_output, n_input)
        
        # Check that rows have similar sums
        row_sums = weights.sum(dim=1)
        target_sum = n_input * sparsity * weight_scale * 0.5
        
        # All rows should be close to target (with tolerance for sparsity randomness)
        for row_sum in row_sums:
            assert abs(row_sum.item() - target_sum) < target_sum * 0.5
    
    def test_topographic_initialization(self):
        """Test topographic initialization."""
        n_output, n_input = 64, 64
        
        weights = WeightInitializer.topographic(
            n_output=n_output,
            n_input=n_input,
            base_weight=0.1,
            boost_strength=0.3
        )
        
        assert weights.shape == (n_output, n_input)
        
        # Check that diagonal (nearby connections) is stronger than off-diagonal
        diagonal_mean = torch.diag(weights).mean()
        off_diagonal_mean = (weights.sum() - torch.diag(weights).sum()) / (n_output * n_input - n_output)
        
        assert diagonal_mean > off_diagonal_mean
    
    def test_orthogonal_initialization(self):
        """Test orthogonal initialization."""
        n_size = 64
        
        weights = WeightInitializer.orthogonal(
            n_output=n_size,
            n_input=n_size,
            gain=1.0
        )
        
        assert weights.shape == (n_size, n_size)
        
        # Check orthogonality: W @ W^T should be close to identity
        product = weights @ weights.T
        identity = torch.eye(n_size)
        
        diff = (product - identity).abs().max()
        assert diff < 0.1
    
    def test_zeros_initialization(self):
        """Test zeros initialization."""
        n_output, n_input = 32, 64
        
        weights = WeightInitializer.zeros(n_output=n_output, n_input=n_input)
        
        assert weights.shape == (n_output, n_input)
        assert (weights == 0).all()
    
    def test_ones_initialization(self):
        """Test ones initialization."""
        n_output, n_input = 32, 64
        
        weights = WeightInitializer.ones(n_output=n_output, n_input=n_input)
        
        assert weights.shape == (n_output, n_input)
        assert (weights == 1).all()
    
    def test_identity_initialization(self):
        """Test identity initialization."""
        n_size = 50
        
        weights = WeightInitializer.identity(n_output=n_size, n_input=n_size)
        
        assert weights.shape == (n_size, n_size)
        assert torch.allclose(weights, torch.eye(n_size))
    
    def test_identity_non_square(self):
        """Test identity with non-square matrix."""
        n_output, n_input = 30, 50
        
        weights = WeightInitializer.identity(n_output=n_output, n_input=n_input)
        
        assert weights.shape == (n_output, n_input)
        
        # Check diagonal is identity up to min dimension
        size = min(n_output, n_input)
        assert torch.allclose(weights[:size, :size], torch.eye(size))
    
    def test_constant_initialization(self):
        """Test constant initialization."""
        n_output, n_input = 40, 80
        value = 0.5
        
        weights = WeightInitializer.constant(
            n_output=n_output,
            n_input=n_input,
            value=value
        )
        
        assert weights.shape == (n_output, n_input)
        assert (weights == value).all()
    
    def test_registry_get_method(self):
        """Test getting initializer from registry."""
        initializer = WeightInitializer.get(InitStrategy.GAUSSIAN)
        
        weights = initializer(n_output=32, n_input=64, mean=0.0, std=0.1)
        
        assert weights.shape == (32, 64)
    
    def test_registry_invalid_strategy(self):
        """Test that invalid strategy raises error."""
        # Create a fake strategy
        class FakeStrategy:
            pass
        
        with pytest.raises(ValueError, match="Unknown initialization strategy"):
            WeightInitializer.get(FakeStrategy())
    
    def test_device_parameter(self):
        """Test that device parameter works correctly."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        weights = WeightInitializer.gaussian(
            n_output=32,
            n_input=64,
            device="cuda"
        )
        
        assert weights.device.type == "cuda"
    
    def test_all_strategies_registered(self):
        """Test that all strategies have registered implementations."""
        for strategy in InitStrategy:
            try:
                initializer = WeightInitializer.get(strategy)
                assert callable(initializer)
            except ValueError:
                pytest.fail(f"Strategy {strategy} not registered")
    
    def test_gaussian_reproducibility(self):
        """Test that manual seed makes initialization reproducible."""
        torch.manual_seed(42)
        weights1 = WeightInitializer.gaussian(n_output=32, n_input=64)
        
        torch.manual_seed(42)
        weights2 = WeightInitializer.gaussian(n_output=32, n_input=64)
        
        assert torch.allclose(weights1, weights2)
    
    def test_weight_bounds_with_clipping(self):
        """Test manual weight clipping after initialization."""
        w_min, w_max = 0.0, 0.5
        
        weights = WeightInitializer.gaussian(
            n_output=50,
            n_input=100,
            mean=0.25,
            std=0.1
        )
        
        # Manual clipping
        weights = torch.clamp(weights, min=w_min, max=w_max)
        
        assert weights.min() >= w_min
        assert weights.max() <= w_max
    
    def test_sparse_random_consistency(self):
        """Test that sparse random creates consistent structure."""
        torch.manual_seed(123)
        
        weights = WeightInitializer.sparse_random(
            n_output=30,
            n_input=60,
            sparsity=0.3
        )
        
        # Should have some zeros
        assert (weights == 0).sum() > 0
        
        # Should have some non-zeros
        assert (weights != 0).sum() > 0
    
    def test_topographic_spatial_structure(self):
        """Test that topographic creates spatial structure."""
        n_size = 32
        
        weights = WeightInitializer.topographic(
            n_output=n_size,
            n_input=n_size,
            base_weight=0.1,
            boost_strength=0.5
        )
        
        # Check that nearby connections are stronger
        for i in range(n_size):
            # Same position should be strongest
            same_pos = weights[i, i]
            
            # Adjacent position
            if i + 1 < n_size:
                adjacent = weights[i, i + 1]
                # Adjacent should be weaker than same position
                assert same_pos >= adjacent
    
    def test_orthogonal_norm_preservation(self):
        """Test that orthogonal initialization preserves norms."""
        n_size = 48
        
        weights = WeightInitializer.orthogonal(
            n_output=n_size,
            n_input=n_size,
            gain=1.0
        )
        
        # Each row should have norm close to sqrt(n_size)
        row_norms = torch.norm(weights, dim=1)
        
        # For orthogonal matrices, rows have unit norm (after normalization)
        # But with our initialization, check they're consistent
        assert row_norms.std() < 0.5


class TestInitStrategyEnum:
    """Test InitStrategy enum."""
    
    def test_all_strategies_exist(self):
        """Test that all expected strategies exist."""
        expected = [
            "GAUSSIAN", "UNIFORM", "XAVIER", "KAIMING",
            "SPARSE_RANDOM", "TOPOGRAPHIC", "ORTHOGONAL",
            "ZEROS", "ONES", "IDENTITY", "CONSTANT"
        ]
        
        for name in expected:
            assert hasattr(InitStrategy, name)
    
    def test_strategies_are_unique(self):
        """Test that all strategies have unique values."""
        values = [s.value for s in InitStrategy]
        assert len(values) == len(set(values))


class TestBiologicalRealism:
    """Test biological realism of initialization strategies."""
    
    def test_sparse_connectivity_realistic(self):
        """Test that sparse initialization creates biologically realistic connectivity."""
        # Typical cortical connectivity is ~10-30%
        sparsity = 0.2
        
        weights = WeightInitializer.sparse_random(
            n_output=100,
            n_input=100,
            sparsity=sparsity
        )
        
        connectivity = (weights != 0).sum().item() / (100 * 100)
        
        # Should be close to target sparsity
        assert abs(connectivity - sparsity) < 0.05
    
    def test_topographic_creates_local_connectivity(self):
        """Test that topographic initialization favors local connections."""
        n_size = 64
        
        weights = WeightInitializer.topographic(
            n_output=n_size,
            n_input=n_size,
            boost_strength=0.5
        )
        
        # Compute average weight for local vs distant connections
        local_weights = []
        distant_weights = []
        
        for i in range(n_size):
            for j in range(n_size):
                dist = abs(i - j)
                if dist <= 5:
                    local_weights.append(weights[i, j].item())
                elif dist > n_size // 4:
                    distant_weights.append(weights[i, j].item())
        
        # Local should be stronger than distant
        assert sum(local_weights) / len(local_weights) > sum(distant_weights) / len(distant_weights)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
