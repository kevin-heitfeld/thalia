"""
Tests for GrowthMixin._create_new_weights() method.

This validates the centralized weight creation method that replaced
duplicate new_weights_for() helpers across 7+ regions.

Architecture Review 2025-12-21, Tier 1.1 Implementation
"""

import pytest
import torch

from thalia.components.synapses import WeightInitializer
from thalia.config.size_calculator import LayerSizeCalculator
from thalia.regions.thalamus import ThalamicRelay, ThalamicRelayConfig


def create_test_thalamus(input_size: int, relay_size: int, device: str = "cpu", **kwargs) -> ThalamicRelay:
    """Create a ThalamicRelay for testing with new (config, sizes, device) pattern."""
    config = ThalamicRelayConfig(device=device, **kwargs)
    calc = LayerSizeCalculator()
    sizes = calc.thalamus_from_relay(relay_size, trn_ratio=0.0)  # trn_size=0 for tests
    sizes["input_size"] = input_size
    return ThalamicRelay(config, sizes, device)


@pytest.fixture
def device():
    """Device for testing."""
    return torch.device("cpu")


@pytest.fixture
def thalamus_config(device):
    """Minimal thalamus configuration for testing GrowthMixin."""
    # Return the region directly (tests use region fixture anyway)
    return None


@pytest.fixture
def region(device):
    """Create a test region with GrowthMixin (using Thalamus as representative)."""
    return create_test_thalamus(input_size=50, relay_size=100, device=str(device), w_min=0.0, w_max=1.0)


class TestGrowthMixinCreateNewWeights:
    """Test suite for _create_new_weights() method."""

    def test_xavier_initialization(self, region):
        """Test xavier initialization strategy."""
        n_output, n_input = 20, 30
        weights = region._create_new_weights(
            n_output=n_output,
            n_input=n_input,
            initialization='xavier',
        )

        # Check shape
        assert weights.shape == (n_output, n_input)

        # Check device
        assert weights.device.type == torch.device(region.device).type

        # Xavier should be roughly centered around 0 with bounded variance
        assert weights.mean().abs() < 0.1  # Roughly centered
        assert 0.0 < weights.std() < 0.5  # Has variance but not too large

        # Check bounds (should be within reasonable range)
        assert weights.min() > -2.0
        assert weights.max() < 2.0

    def test_sparse_random_initialization(self, region):
        """Test sparse_random initialization with specified sparsity."""
        n_output, n_input = 100, 80
        sparsity = 0.2  # Keep 20% of connections (80% will be zero)

        weights = region._create_new_weights(
            n_output=n_output,
            n_input=n_input,
            initialization='sparse_random',
            sparsity=sparsity,
        )

        # Check shape
        assert weights.shape == (n_output, n_input)

        # Check device
        assert weights.device.type == torch.device(region.device).type

        # Check sparsity: sparsity parameter means fraction TO KEEP
        # So sparsity=0.2 → ~20% non-zero, ~80% zero
        non_zero_count = (weights != 0).sum().item()
        total_count = weights.numel()
        actual_connectivity = non_zero_count / total_count

        # Allow 5% tolerance on connectivity fraction
        assert abs(actual_connectivity - sparsity) < 0.05

        # Non-zero weights should be within valid range
        non_zero_weights = weights[weights != 0]
        if len(non_zero_weights) > 0:
            assert non_zero_weights.min() >= 0.0
            assert non_zero_weights.max() <= 1.0

    def test_uniform_initialization(self, region):
        """Test uniform initialization (default fallback)."""
        n_output, n_input = 50, 40
        weights = region._create_new_weights(
            n_output=n_output,
            n_input=n_input,
            initialization='uniform',
        )

        # Check shape
        assert weights.shape == (n_output, n_input)

        # Check device
        assert weights.device.type == torch.device(region.device).type

        # Uniform should be between 0 and 1 (or w_min and w_max)
        assert weights.min() >= 0.0
        assert weights.max() <= 1.0

        # Should have some variation (not all zeros or ones)
        assert weights.std() > 0.0

    def test_unknown_initialization_defaults_to_uniform(self, region):
        """Test that unknown initialization falls back to uniform."""
        n_output, n_input = 30, 25
        weights = region._create_new_weights(
            n_output=n_output,
            n_input=n_input,
            initialization='unknown_strategy',  # Invalid strategy
        )

        # Should still work and return uniform initialization
        assert weights.shape == (n_output, n_input)
        assert weights.min() >= 0.0
        assert weights.max() <= 1.0

    def test_single_neuron_weights(self, region):
        """Test edge case with single neuron."""
        weights = region._create_new_weights(
            n_output=1,
            n_input=50,
            initialization='xavier',
        )

        assert weights.shape == (1, 50)
        assert not torch.isnan(weights).any()
        assert not torch.isinf(weights).any()

    def test_single_input_weights(self, region):
        """Test edge case with single input."""
        weights = region._create_new_weights(
            n_output=50,
            n_input=1,
            initialization='xavier',
        )

        assert weights.shape == (50, 1)
        assert not torch.isnan(weights).any()
        assert not torch.isinf(weights).any()

    def test_large_dimensions(self, region):
        """Test with large weight matrices."""
        n_output, n_input = 1000, 800
        weights = region._create_new_weights(
            n_output=n_output,
            n_input=n_input,
            initialization='sparse_random',
            sparsity=0.3,
        )

        assert weights.shape == (n_output, n_input)
        assert weights.device.type == torch.device(region.device).type

        # Should not cause memory issues or NaN/Inf
        assert not torch.isnan(weights).any()
        assert not torch.isinf(weights).any()

    def test_consistency_across_calls(self, region):
        """Test that multiple calls produce different random weights."""
        weights1 = region._create_new_weights(
            n_output=50,
            n_input=40,
            initialization='xavier',
        )

        weights2 = region._create_new_weights(
            n_output=50,
            n_input=40,
            initialization='xavier',
        )

        # Should be different (random initialization)
        assert not torch.allclose(weights1, weights2)

    def test_different_devices(self):
        """Test weight creation on different devices."""
        # CPU region
        cpu_region = create_test_thalamus(input_size=50, relay_size=100, device="cpu")

        cpu_weights = cpu_region._create_new_weights(
            n_output=30,
            n_input=20,
            initialization='xavier',
        )
        assert cpu_weights.device.type == "cpu"

        # CUDA region (if available)
        if torch.cuda.is_available():
            cuda_region = create_test_thalamus(input_size=50, relay_size=100, device="cuda")

            cuda_weights = cuda_region._create_new_weights(
                n_output=30,
                n_input=20,
                initialization='xavier',
            )
            assert cuda_weights.device.type == "cuda"

    def test_integration_with_grow_input(self, region):
        """Test that _create_new_weights produces correct dimensions for grow_input."""
        n_new = 20

        # Create new weight columns (typical grow_input pattern)
        new_cols = region._create_new_weights(
            n_output=region.n_output,
            n_input=n_new,
            initialization='xavier',
        )

        # Verify dimensions are correct for concatenation
        assert new_cols.shape == (region.n_output, n_new)
        # Verify it would concatenate correctly with existing weights
        # (testing pattern, not actual region state)

    def test_integration_with_grow_output(self, region):
        """Test that _create_new_weights produces correct dimensions for grow_output."""
        n_new = 30

        # Create new weight rows (typical grow_output pattern)
        new_rows = region._create_new_weights(
            n_output=n_new,
            n_input=region.input_size,
            initialization='sparse_random',
            sparsity=0.15,
        )

        # Verify dimensions are correct for concatenation
        assert new_rows.shape == (n_new, region.input_size)
        # Verify it would concatenate correctly with existing weights
        # (testing pattern, not actual region state)

    def test_sparsity_parameter_validation(self, region):
        """Test that sparsity parameter is handled correctly."""
        # Valid sparsity values
        for sparsity in [0.0, 0.1, 0.5, 0.9, 1.0]:
            weights = region._create_new_weights(
                n_output=50,
                n_input=40,
                initialization='sparse_random',
                sparsity=sparsity,
            )
            assert weights.shape == (50, 40)

        # Note: WeightInitializer should handle invalid sparsity values
        # (negative or > 1.0), but we test graceful handling
        try:
            weights = region._create_new_weights(
                n_output=50,
                n_input=40,
                initialization='sparse_random',
                sparsity=-0.1,  # Invalid
            )
            # If it doesn't raise, just ensure it returns valid tensor
            assert weights.shape == (50, 40)
        except (ValueError, AssertionError):
            # Expected behavior - invalid sparsity rejected
            pass


class TestRegressionAgainstOldPattern:
    """Regression tests comparing new centralized method against old per-region pattern."""

    def old_new_weights_for_xavier(
        self, n_out: int, n_in: int, device: torch.device
    ) -> torch.Tensor:
        """Old pattern: inline xavier initialization."""
        return WeightInitializer.xavier(n_out, n_in, device=device)

    def old_new_weights_for_sparse(
        self, n_out: int, n_in: int, sparsity: float, device: torch.device
    ) -> torch.Tensor:
        """Old pattern: inline sparse_random initialization."""
        return WeightInitializer.sparse_random(n_out, n_in, sparsity, device=device)

    def test_xavier_produces_equivalent_distribution(self, region):
        """Test that new method produces statistically similar results to old pattern."""
        n_samples = 100
        n_out, n_in = 50, 40

        # Generate samples using both methods
        new_method_samples = [
            region._create_new_weights(n_out, n_in, 'xavier')
            for _ in range(n_samples)
        ]
        old_method_samples = [
            self.old_new_weights_for_xavier(n_out, n_in, region.device)
            for _ in range(n_samples)
        ]

        # Compare statistical properties
        new_means = torch.stack([w.mean() for w in new_method_samples])
        old_means = torch.stack([w.mean() for w in old_method_samples])

        new_stds = torch.stack([w.std() for w in new_method_samples])
        old_stds = torch.stack([w.std() for w in old_method_samples])

        # Means should be similar (both close to 0)
        assert abs(new_means.mean() - old_means.mean()) < 0.05

        # Standard deviations should be similar
        assert abs(new_stds.mean() - old_stds.mean()) < 0.05

    def test_sparse_random_produces_equivalent_sparsity(self, region):
        """Test that sparsity is preserved between old and new patterns."""
        n_out, n_in = 100, 80
        sparsity = 0.25  # Keep 25% of connections

        new_weights = region._create_new_weights(
            n_out, n_in, 'sparse_random', sparsity
        )
        old_weights = self.old_new_weights_for_sparse(
            n_out, n_in, sparsity, torch.device(region.device)
        )

        # Sparsity parameter means fraction TO KEEP, not fraction to zero
        new_connectivity = (new_weights != 0).float().mean()
        old_connectivity = (old_weights != 0).float().mean()

        # Both should be close to target connectivity (within 5%)
        assert abs(new_connectivity - sparsity) < 0.05
        assert abs(old_connectivity - sparsity) < 0.05
        assert abs(new_connectivity - old_connectivity) < 0.05
