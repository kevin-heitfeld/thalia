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
from thalia.regions import LayeredCortex


class TestLayeredCortexStrategyMigration:
    """Test LayeredCortex migration to BCMStrategy."""

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

        # BCM strategies are integrated into composite strategies, not separate attributes
        # The config disables BCM so strategies will use STDP only
        # Check that forward pass still works
        input_spikes = (torch.rand(32) < 0.3).bool()
        output = cortex.forward({"input": input_spikes})
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
