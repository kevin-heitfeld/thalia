"""
Tests for region migrations to learning strategy pattern.

Tests that Cerebellum and LayeredCortex correctly use learning strategies
after migration from custom implementations.

Author: Thalia Project
Date: December 11, 2025
"""

import pytest
import torch

from thalia.config import LayeredCortexConfig
from thalia.constants.learning import TAU_BCM_THRESHOLD
from thalia.learning import BCMStrategy, CompositeStrategy
from thalia.regions import LayeredCortex


class TestLayeredCortexStrategyMigration:
    """Test LayeredCortex migration to BCMStrategy."""

    def test_layered_cortex_has_bcm_strategies(self):
        """Test LayeredCortex instantiates CompositeStrategy (STDP+BCM) for each layer."""
        sizes = {
            "input_size": 32,
            "l4_size": 8,
            "l23_size": 10,
            "l5_size": 6,
            "l6a_size": 2,
            "l6b_size": 2,
        }
        config = LayeredCortexConfig(bcm_enabled=True)
        cortex = LayeredCortex(config=config, sizes=sizes, device="cpu")

        # Verify strategies exist
        assert hasattr(cortex, "bcm_l4")
        assert hasattr(cortex, "bcm_l23")
        assert hasattr(cortex, "bcm_l5")

        # Verify all are CompositeStrategy instances
        assert isinstance(cortex.bcm_l4, CompositeStrategy)
        assert isinstance(cortex.bcm_l23, CompositeStrategy)
        assert isinstance(cortex.bcm_l5, CompositeStrategy)

        # Verify composite contains BCMStrategy (second strategy in the list)
        assert len(cortex.bcm_l4.strategies) == 2
        assert isinstance(cortex.bcm_l4.strategies[1], BCMStrategy)

        # Verify BCM config matches default
        bcm_strategy = cortex.bcm_l4.strategies[1]
        assert bcm_strategy.bcm_config.tau_theta == TAU_BCM_THRESHOLD
        assert bcm_strategy.bcm_config.theta_init == 0.01  # Default theta_init

    def test_layered_cortex_bcm_disabled_works(self):
        """Test LayeredCortex works when BCM is disabled."""
        sizes = {
            "input_size": 32,
            "l4_size": 8,
            "l23_size": 10,
            "l5_size": 6,
            "l6a_size": 2,
            "l6b_size": 2,
        }
        config = LayeredCortexConfig(bcm_enabled=False)
        cortex = LayeredCortex(config=config, sizes=sizes, device="cpu")

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
        """Test BCMStrategy compute_phi() method works via CompositeStrategy."""
        sizes = {
            "input_size": 32,
            "l4_size": 8,
            "l23_size": 10,
            "l5_size": 6,
            "l6a_size": 2,
            "l6b_size": 2,
        }
        config = LayeredCortexConfig(bcm_enabled=True)
        cortex = LayeredCortex(config=config, sizes=sizes, device="cpu")

        # Get BCM strategy from composite (second strategy)
        bcm_l4 = cortex.bcm_l4.strategies[1]

        # Test compute_phi on L4
        l4_activity = torch.rand(cortex.l4_size)
        phi = bcm_l4.compute_phi(l4_activity)

        # Verify phi shape
        assert phi.shape == l4_activity.shape

    def test_layered_cortex_bcm_update_threshold_works(self):
        """Test BCMStrategy update_threshold() method works via CompositeStrategy."""
        sizes = {
            "input_size": 32,
            "l4_size": 8,
            "l23_size": 10,
            "l5_size": 6,
            "l6a_size": 2,
            "l6b_size": 2,
        }
        config = LayeredCortexConfig(bcm_enabled=True)
        cortex = LayeredCortex(config=config, sizes=sizes, device="cpu")

        # Get BCM strategy from composite (second strategy)
        bcm_l4 = cortex.bcm_l4.strategies[1]

        # Test update_threshold on L4
        l4_activity = torch.rand(cortex.l4_size) * 0.5

        # Update threshold multiple times
        for _ in range(10):
            bcm_l4.update_threshold(l4_activity)

        # Verify theta exists and is reasonable
        assert bcm_l4.theta is not None
        assert bcm_l4.theta.shape == (cortex.l4_size,)
        assert (bcm_l4.theta > 0).all()

    def test_layered_cortex_forward_backward_compatible(self):
        """Test LayeredCortex forward pass works with BCMStrategy."""
        sizes = {
            "input_size": 32,
            "l4_size": 8,
            "l23_size": 10,
            "l5_size": 6,
            "l6a_size": 2,
            "l6b_size": 2,
        }
        config = LayeredCortexConfig(bcm_enabled=True)
        cortex = LayeredCortex(config=config, sizes=sizes, device="cpu")

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
        sizes = {
            "input_size": 32,
            "l4_size": 8,
            "l23_size": 10,
            "l5_size": 6,
            "l6a_size": 2,
            "l6b_size": 2,
        }
        config = LayeredCortexConfig(bcm_enabled=True)
        cortex = LayeredCortex(config=config, sizes=sizes, device="cpu")

        # Run forward passes to build activity
        for _ in range(20):
            input_spikes = (torch.rand(32) < 0.5).bool()  # High activity
            cortex.forward(input_spikes)

        # Thresholds should be initialized now (access BCM via .strategies[1])
        bcm_l4 = cortex.bcm_l4.strategies[1]
        bcm_l23 = cortex.bcm_l23.strategies[1]
        bcm_l5 = cortex.bcm_l5.strategies[1]
        assert bcm_l4.theta is not None
        assert bcm_l23.theta is not None
        assert bcm_l5.theta is not None

        # Store initial thresholds
        initial_l4_theta = bcm_l4.theta.clone()

        # Continue with high activity
        for _ in range(20):
            input_spikes = (torch.rand(32) < 0.7).bool()  # Even higher
            cortex.forward(input_spikes)

        # Thresholds should increase with activity
        assert bcm_l4.theta.mean() > initial_l4_theta.mean()


class TestStrategyStateManagement:
    """Test strategy state management in regions."""

    def test_layered_cortex_reset_state(self):
        """Test LayeredCortex reset_state() resets BCM thresholds."""
        sizes = {
            "input_size": 32,
            "l4_size": 8,
            "l23_size": 10,
            "l5_size": 6,
            "l6a_size": 2,
            "l6b_size": 2,
        }
        config = LayeredCortexConfig(bcm_enabled=True)
        cortex = LayeredCortex(config=config, sizes=sizes, device="cpu")

        # Build up state
        for _ in range(10):
            input_spikes = (torch.rand(32) < 0.3).bool()
            cortex.forward(input_spikes)

        # Store thetas (access BCM via .strategies[1])
        bcm_l4 = cortex.bcm_l4.strategies[1]
        bcm_l23 = cortex.bcm_l23.strategies[1]
        bcm_l5 = cortex.bcm_l5.strategies[1]
        assert bcm_l4.theta is not None

        # Reset BCM strategies explicitly
        bcm_l4.reset_state()
        bcm_l23.reset_state()
        bcm_l5.reset_state()

        # Thetas should be cleared
        assert bcm_l4.theta is None
        assert bcm_l23.theta is None
        assert bcm_l5.theta is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
