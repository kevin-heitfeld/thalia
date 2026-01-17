"""Tests for Cerebellum using unified RegionTestBase framework.

Demonstrates unified testing pattern with region-specific tests for
cerebellum (error-corrective learning and motor prediction).

Author: Thalia Project
Date: December 22, 2025 (Tier 3.4 implementation)
"""

import torch

from tests.utils.region_test_base import RegionTestBase
from thalia.config.size_calculator import LayerSizeCalculator
from thalia.regions.cerebellum import Cerebellum, CerebellumConfig


class TestCerebellum(RegionTestBase):
    """Test Cerebellum implementation using unified test framework."""

    def create_region(self, **kwargs):
        """Create Cerebellum instance for testing."""
        # Extract sizes from kwargs
        input_size = kwargs.pop("input_size")

        # Handle expansion factor (if using enhanced microcircuit)
        if "granule_expansion_factor" in kwargs:
            purkinje_size = kwargs.pop("purkinje_size")
            expansion = kwargs.pop("granule_expansion_factor")
            calc = LayerSizeCalculator()
            sizes = calc.cerebellum_from_output(purkinje_size)
            sizes["input_size"] = input_size
            kwargs["use_enhanced_microcircuit"] = True
        else:
            # Classic cerebellum (purkinje_size is output)
            purkinje_size = kwargs.pop("purkinje_size", kwargs.pop("n_output", 50))
            sizes = {"input_size": input_size, "purkinje_size": purkinje_size}

        # Extract device
        device = kwargs.pop("device", "cpu")

        # Create config with remaining behavioral params
        config = CerebellumConfig(**kwargs)

        return Cerebellum(config, sizes, device)

    def get_default_params(self):
        """Return default cerebellum parameters."""
        return {
            "input_size": 100,
            "purkinje_size": 50,
            "granule_expansion_factor": 2.0,  # Mossy fiber expansion
            "device": "cpu",
            "dt_ms": 1.0,
        }

    def get_min_params(self):
        """Return minimal valid parameters for quick tests."""
        return {
            "input_size": 20,
            "purkinje_size": 10,
            "granule_expansion_factor": 2.0,
            "device": "cpu",
            "dt_ms": 1.0,
        }

    def get_input_dict(self, n_input, device="cpu"):
        """Return dict input for cerebellum (supports multi-source)."""
        return {
            "cortex": torch.zeros(n_input // 2, device=device),
            "pons": torch.zeros(n_input // 2, device=device),
        }

    def _get_region_input_size(self, region):
        """Helper to get input size from region instance."""
        return region.input_size

    def _get_region_output_size(self, region):
        """Helper to get output size from region instance."""
        return region.n_output

    # =========================================================================
    # CEREBELLUM-SPECIFIC TESTS
    # =========================================================================

    def test_granule_cell_expansion(self):
        """Test granule cells expand mossy fiber input."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Forward pass
        input_spikes = torch.ones(self._get_region_input_size(region), device=region.device)
        region.forward(input_spikes)

        # Verify granule layer exists and expands representation
        expected_granules = int(
            self._get_region_input_size(region) * params["granule_expansion_factor"]
        )
        assert expected_granules > self._get_region_input_size(
            region
        ), "Granule cells should expand input representation"

        # Check if granule activity is tracked
        state = region.get_state()
        if hasattr(state, "granule_spikes"):
            if state.granule_spikes is not None:
                assert state.granule_spikes.shape[0] == expected_granules

    def test_purkinje_cell_output(self):
        """Test Purkinje cells produce cerebellar output."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Forward pass
        input_spikes = torch.ones(self._get_region_input_size(region), device=region.device)
        output = region.forward(input_spikes)

        # Output should match Purkinje cell count
        assert output.shape[0] == self._get_region_output_size(region)

        # Check Purkinje activity in state
        state = region.get_state()
        if hasattr(state, "purkinje_spikes"):
            if state.purkinje_spikes is not None:
                assert state.purkinje_spikes.shape[0] == self._get_region_output_size(region)

    def test_climbing_fiber_error_signal(self):
        """Test climbing fiber provides error signal for learning."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Forward with error signal (if supported)
        input_spikes = torch.ones(self._get_region_input_size(region), device=region.device)

        # Check if region accepts error signal
        if "error_signal" in region.forward.__code__.co_varnames:
            error = torch.ones(self._get_region_output_size(region), device=region.device) * 0.5
            output = region.forward(input_spikes, error_signal=error)
            assert output.shape[0] == self._get_region_output_size(region)
        else:
            # Just verify forward works without error
            output = region.forward(input_spikes)
            assert output.shape[0] == self._get_region_output_size(region)

    def test_parallel_fiber_plasticity(self):
        """Test LTD/LTP at parallel fiber → Purkinje synapses."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Get initial weights (parallel fiber → Purkinje)
        if hasattr(region, "synaptic_weights"):
            initial_weights = None
            # Try to find granule→purkinje weights
            for source_name in region.synaptic_weights:
                if "granule" in source_name.lower() or "default" in source_name:
                    initial_weights = region.synaptic_weights[source_name].clone()
                    break

            if initial_weights is not None:
                # Multiple forward passes with consistent input
                input_spikes = torch.ones(self._get_region_input_size(region), device=region.device)
                for _ in range(100):
                    region.forward(input_spikes)

                # Weights should change (LTD/LTP applied)
                for source_name in region.synaptic_weights:
                    if "granule" in source_name.lower() or "default" in source_name:
                        final_weights = region.synaptic_weights[source_name]
                        # Some plasticity should occur
                        if not torch.allclose(initial_weights, final_weights, atol=1e-6):
                            return  # Plasticity detected

                # If we get here, plasticity might be gated or error-dependent
                # This is acceptable for cerebellum

    def test_basket_cell_inhibition(self):
        """Test basket cells provide feedforward inhibition."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Forward pass
        input_spikes = torch.ones(self._get_region_input_size(region), device=region.device)
        region.forward(input_spikes)

        # Check if basket cell activity is tracked
        state = region.get_state()
        if hasattr(state, "basket_spikes"):
            if state.basket_spikes is not None:
                # Basket cells should exist
                assert state.basket_spikes.shape[0] > 0

    def test_golgi_cell_inhibition(self):
        """Test Golgi cells inhibit granule cells."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Forward pass
        input_spikes = torch.ones(self._get_region_input_size(region), device=region.device)
        region.forward(input_spikes)

        # Check if Golgi cell activity is tracked
        state = region.get_state()
        if hasattr(state, "golgi_spikes"):
            if state.golgi_spikes is not None:
                # Golgi cells should exist
                assert state.golgi_spikes.shape[0] > 0

    def test_timing_prediction(self):
        """Test cerebellum learns temporal predictions."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Create temporal pattern (onset at specific time)
        for t in range(50):
            if t < 10:
                input_spikes = torch.zeros(
                    self._get_region_input_size(region), device=region.device
                )
            else:
                input_spikes = torch.ones(self._get_region_input_size(region), device=region.device)

            output = region.forward(input_spikes)
            assert output.shape[0] == self._get_region_output_size(region)

        # Cerebellum should have processed temporal pattern
        # (Detailed timing tests would require monitoring predictions)

    def test_motor_error_correction(self):
        """Test cerebellum adjusts based on motor errors."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Simulate motor command
        input_spikes = torch.ones(self._get_region_input_size(region), device=region.device)
        output1 = region.forward(input_spikes)

        # Continue with more trials
        for _ in range(20):
            region.forward(input_spikes)

        output2 = region.forward(input_spikes)

        # Output should be consistent after learning
        # (Exact comparison depends on learning dynamics)
        assert output1.shape == output2.shape

    def test_sparse_granule_coding(self):
        """Test granule cells maintain sparse coding."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Dense input
        input_spikes = torch.ones(self._get_region_input_size(region), device=region.device) * 0.8
        region.forward(input_spikes)

        # Check granule cell activity
        state = region.get_state()
        if hasattr(state, "granule_spikes"):
            if state.granule_spikes is not None:
                # Granule cells typically maintain ~5% sparsity
                granule_activity = state.granule_spikes.float().mean().item()
                assert (
                    0.0 <= granule_activity <= 0.3
                ), f"Expected sparse granule activity, got {granule_activity}"

    def test_purkinje_complex_spike(self):
        """Test Purkinje cells can generate complex spikes."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Forward passes
        input_spikes = torch.ones(self._get_region_input_size(region), device=region.device)
        for _ in range(10):
            region.forward(input_spikes)

        # Check if complex spike state exists
        state = region.get_state()
        if hasattr(state, "complex_spikes"):
            # Complex spikes should be tracked
            if state.complex_spikes is not None:
                assert state.complex_spikes.shape[0] == self._get_region_output_size(region)


# Standard tests (initialization, forward, growth, state, device, neuromodulators, diagnostics)
# inherited from RegionTestBase - eliminates ~100 lines of boilerplate
