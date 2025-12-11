"""
Tests for region migrations to learning strategy pattern.

Tests that Cerebellum and LayeredCortex correctly use learning strategies
after migration from custom implementations.

Author: Thalia Project
Date: December 11, 2025
"""

import pytest
import torch

from thalia.regions.cortex import LayeredCortex, LayeredCortexConfig
from thalia.learning import BCMStrategy


class TestLayeredCortexStrategyMigration:
    """Test LayeredCortex migration to BCMStrategy."""

    def test_layered_cortex_has_bcm_strategies(self):
        """Test LayeredCortex instantiates BCMStrategy for each layer."""
        config = LayeredCortexConfig(
            n_input=32,
            n_output=16,
            bcm_enabled=True,
            bcm_tau_theta=5000.0,
            bcm_theta_init=0.01,
        )
        cortex = LayeredCortex(config)

        # Verify strategies exist
        assert hasattr(cortex, 'bcm_l4')
        assert hasattr(cortex, 'bcm_l23')
        assert hasattr(cortex, 'bcm_l5')

        # Verify all are BCMStrategy instances
        assert isinstance(cortex.bcm_l4, BCMStrategy)
        assert isinstance(cortex.bcm_l23, BCMStrategy)
        assert isinstance(cortex.bcm_l5, BCMStrategy)

        # Verify config matches
        assert cortex.bcm_l4.bcm_config.tau_theta == 5000.0
        assert cortex.bcm_l4.bcm_config.theta_init == 0.01

    def test_layered_cortex_bcm_disabled_works(self):
        """Test LayeredCortex works when BCM is disabled."""
        config = LayeredCortexConfig(
            n_input=32,
            n_output=16,
            bcm_enabled=False,
        )
        cortex = LayeredCortex(config)

        # Verify strategies are None
        assert cortex.bcm_l4 is None
        assert cortex.bcm_l23 is None
        assert cortex.bcm_l5 is None

        # Forward pass should still work
        input_spikes = (torch.rand(32) < 0.3).bool()
        output = cortex.forward(input_spikes)
        # Output is L2/3 + L5 concatenated
        expected_size = cortex.l23_size + cortex.l5_size
        assert output.shape[0] == expected_size

    def test_layered_cortex_bcm_compute_phi_works(self):
        """Test BCMStrategy compute_phi() method works (backward compatibility)."""
        config = LayeredCortexConfig(
            n_input=32,
            n_output=16,
            bcm_enabled=True,
        )
        cortex = LayeredCortex(config)

        # Test compute_phi on L4
        l4_activity = torch.rand(cortex.l4_size)
        phi = cortex.bcm_l4.compute_phi(l4_activity)

        # Verify phi shape
        assert phi.shape == l4_activity.shape

    def test_layered_cortex_bcm_update_threshold_works(self):
        """Test BCMStrategy update_threshold() method works (backward compatibility)."""
        config = LayeredCortexConfig(
            n_input=32,
            n_output=16,
            bcm_enabled=True,
        )
        cortex = LayeredCortex(config)

        # Test update_threshold on L4
        l4_activity = torch.rand(cortex.l4_size) * 0.5

        # Update threshold multiple times
        for _ in range(10):
            cortex.bcm_l4.update_threshold(l4_activity)

        # Verify theta exists and is reasonable
        assert cortex.bcm_l4.theta is not None
        assert cortex.bcm_l4.theta.shape == (cortex.l4_size,)
        assert (cortex.bcm_l4.theta > 0).all()

    def test_layered_cortex_forward_backward_compatible(self):
        """Test LayeredCortex forward pass works with BCMStrategy."""
        config = LayeredCortexConfig(
            n_input=32,
            n_output=16,
            bcm_enabled=True,
        )
        cortex = LayeredCortex(config)

        # Multiple forward passes
        for _ in range(5):
            input_spikes = (torch.rand(32) < 0.3).bool()
            output = cortex.forward(input_spikes)

            # Verify output shape (L2/3 + L5)
            expected_size = cortex.l23_size + cortex.l5_size
            assert output.shape[0] == expected_size
            assert output.dtype == torch.bool

    def test_layered_cortex_theta_adapts_with_activity(self):
        """Test BCM thresholds adapt to layer activity."""
        config = LayeredCortexConfig(
            n_input=32,
            n_output=16,
            bcm_enabled=True,
        )
        cortex = LayeredCortex(config)

        # Run forward passes to build activity
        for _ in range(20):
            input_spikes = (torch.rand(32) < 0.5).bool()  # High activity
            cortex.forward(input_spikes)

        # Thresholds should be initialized now
        assert cortex.bcm_l4.theta is not None
        assert cortex.bcm_l23.theta is not None
        assert cortex.bcm_l5.theta is not None

        # Store initial thresholds
        initial_l4_theta = cortex.bcm_l4.theta.clone()

        # Continue with high activity
        for _ in range(20):
            input_spikes = (torch.rand(32) < 0.7).bool()  # Even higher
            cortex.forward(input_spikes)

        # Thresholds should increase with activity
        assert cortex.bcm_l4.theta.mean() > initial_l4_theta.mean()


class TestStrategyStateManagement:
    """Test strategy state management in regions."""

    def test_layered_cortex_reset_state(self):
        """Test LayeredCortex reset_state() resets BCM thresholds."""
        config = LayeredCortexConfig(
            n_input=32,
            n_output=16,
            bcm_enabled=True,
        )
        cortex = LayeredCortex(config)

        # Build up state
        for _ in range(10):
            input_spikes = (torch.rand(32) < 0.3).bool()
            cortex.forward(input_spikes)

        # Store thetas
        assert cortex.bcm_l4.theta is not None

        # Reset BCM strategies explicitly
        cortex.bcm_l4.reset_state()
        cortex.bcm_l23.reset_state()
        cortex.bcm_l5.reset_state()

        # Thetas should be cleared
        assert cortex.bcm_l4.theta is None
        assert cortex.bcm_l23.theta is None
        assert cortex.bcm_l5.theta is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
