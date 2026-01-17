"""
Unit tests for gap junction electrical coupling.

Tests functional connectivity-based coupling, voltage synchronization dynamics,
and integration with TRN neurons in thalamus.

Author: Thalia Project
Date: December 23, 2025
"""

import pytest
import torch

from thalia.components.gap_junctions import (
    GapJunctionConfig,
    GapJunctionCoupling,
    create_gap_junction_coupling,
)


class TestGapJunctionConfig:
    """Test gap junction configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GapJunctionConfig()
        assert config.enabled is True
        assert 0.05 <= config.coupling_strength <= 0.3  # Biological range
        assert 0 < config.connectivity_threshold < 1
        assert config.max_neighbors > 0
        assert config.interneuron_only is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = GapJunctionConfig(
            enabled=False,
            coupling_strength=0.2,
            connectivity_threshold=0.5,
            max_neighbors=5,
            interneuron_only=False,
        )
        assert config.enabled is False
        assert config.coupling_strength == 0.2
        assert config.connectivity_threshold == 0.5
        assert config.max_neighbors == 5
        assert config.interneuron_only is False


class TestGapJunctionCoupling:
    """Test gap junction coupling module."""

    @pytest.fixture
    def simple_weights(self):
        """Create simple afferent weights for testing."""
        # 5 neurons, 10 inputs
        # First 3 neurons share inputs [0,1,2]
        # Last 2 neurons share inputs [7,8,9]
        weights = torch.zeros(5, 10)
        weights[0, [0, 1, 2]] = 1.0
        weights[1, [0, 1, 2]] = 1.0
        weights[2, [0, 1, 2]] = 1.0
        weights[3, [7, 8, 9]] = 1.0
        weights[4, [7, 8, 9]] = 1.0
        return weights

    def test_disabled_coupling(self, simple_weights):
        """Test that disabled coupling returns zero current."""
        config = GapJunctionConfig(enabled=False)
        gap_junctions = GapJunctionCoupling(
            n_neurons=5,
            afferent_weights=simple_weights,
            config=config,
            device="cpu",
        )

        voltages = torch.randn(5)
        coupling_current = gap_junctions(voltages)

        assert torch.allclose(coupling_current, torch.zeros(5))
        assert gap_junctions.coupling_matrix.sum() == 0

    def test_functional_connectivity_neighborhoods(self, simple_weights):
        """Test that neurons with shared inputs get coupled."""
        config = GapJunctionConfig(
            enabled=True,
            coupling_strength=0.1,
            connectivity_threshold=0.5,  # Share ≥50% of inputs
            max_neighbors=None,  # No limit
        )
        gap_junctions = GapJunctionCoupling(
            n_neurons=5,
            afferent_weights=simple_weights,
            config=config,
            device="cpu",
        )

        # Neurons 0,1,2 should be coupled (share all 3 inputs)
        # Neurons 3,4 should be coupled (share all 3 inputs)
        # No coupling between these two groups

        coupling = gap_junctions.coupling_matrix

        # Check intra-group coupling
        assert coupling[0, 1] > 0  # 0↔1 coupled
        assert coupling[1, 0] > 0
        assert coupling[3, 4] > 0  # 3↔4 coupled
        assert coupling[4, 3] > 0

        # Check inter-group isolation
        assert coupling[0, 3] == 0  # No coupling between groups
        assert coupling[0, 4] == 0
        assert coupling[1, 3] == 0

    def test_voltage_coupling_dynamics(self):
        """Test that gap junctions create attractive voltage dynamics."""
        # Create coupled pair
        weights = torch.ones(2, 10) * 0.5  # Both neurons have same inputs
        config = GapJunctionConfig(
            enabled=True,
            coupling_strength=0.2,
            connectivity_threshold=0.0,  # Couple all
            max_neighbors=10,
        )
        gap_junctions = GapJunctionCoupling(
            n_neurons=2,
            afferent_weights=weights,
            config=config,
            device="cpu",
        )

        # Start with voltage difference
        voltages = torch.tensor([-60.0, -50.0])  # 10mV difference
        coupling_current = gap_junctions(voltages)

        # Neuron 0 (lower voltage) should get positive current
        # Neuron 1 (higher voltage) should get negative current
        # This reduces voltage difference
        assert coupling_current[0] > 0  # Pull neuron 0 up
        assert coupling_current[1] < 0  # Pull neuron 1 down
        assert abs(coupling_current[0] + coupling_current[1]) < 1e-5  # Current conserved

    def test_max_neighbors_limit(self, simple_weights):
        """Test that max_neighbors limits coupling density."""
        config_unlimited = GapJunctionConfig(
            enabled=True,
            coupling_strength=0.1,
            connectivity_threshold=0.0,  # Low threshold (couple all)
            max_neighbors=None,  # No limit
        )
        gap_unlimited = GapJunctionCoupling(
            n_neurons=5,
            afferent_weights=simple_weights,
            config=config_unlimited,
            device="cpu",
        )

        config_limited = GapJunctionConfig(
            enabled=True,
            coupling_strength=0.1,
            connectivity_threshold=0.0,
            max_neighbors=2,  # Limit to 2 neighbors
        )
        gap_limited = GapJunctionCoupling(
            n_neurons=5,
            afferent_weights=simple_weights,
            config=config_limited,
            device="cpu",
        )

        # Limited should have fewer connections
        n_unlimited = (gap_unlimited.coupling_matrix > 0).sum().item()
        n_limited = (gap_limited.coupling_matrix > 0).sum().item()

        assert n_limited <= n_unlimited
        assert n_limited <= 5 * 2  # At most max_neighbors per neuron

    def test_interneuron_mask(self):
        """Test that interneuron_only restricts coupling."""
        weights = torch.ones(5, 10) * 0.5  # All neurons have same inputs
        interneuron_mask = torch.tensor([True, True, False, False, False])

        config = GapJunctionConfig(
            enabled=True,
            coupling_strength=0.1,
            connectivity_threshold=0.0,
            interneuron_only=True,
        )
        gap_junctions = GapJunctionCoupling(
            n_neurons=5,
            afferent_weights=weights,
            config=config,
            interneuron_mask=interneuron_mask,
            device="cpu",
        )

        coupling = gap_junctions.coupling_matrix

        # Only interneurons (0,1) should be coupled
        assert coupling[0, 1] > 0
        assert coupling[1, 0] > 0

        # No coupling to pyramidal neurons (2,3,4)
        assert coupling[0, 2] == 0
        assert coupling[2, 0] == 0
        assert coupling[2, 3] == 0

    def test_coupling_stats(self, simple_weights):
        """Test network statistics computation."""
        config = GapJunctionConfig(
            enabled=True,
            coupling_strength=0.1,
            connectivity_threshold=0.3,
            max_neighbors=None,
        )
        gap_junctions = GapJunctionCoupling(
            n_neurons=5,
            afferent_weights=simple_weights,
            config=config,
            device="cpu",
        )

        stats = gap_junctions.get_coupling_stats()

        assert "n_coupled_neurons" in stats
        assert "n_connections" in stats
        assert "avg_neighbors" in stats
        assert "coupling_density" in stats

        assert stats["n_coupled_neurons"] > 0
        assert stats["n_connections"] > 0
        assert stats["avg_neighbors"] >= 0
        assert 0 <= stats["coupling_density"] <= 1

    def test_disabled_coupling_stats(self, simple_weights):
        """Test that disabled coupling returns zero stats."""
        config = GapJunctionConfig(enabled=False)
        gap_junctions = GapJunctionCoupling(
            n_neurons=5,
            afferent_weights=simple_weights,
            config=config,
            device="cpu",
        )

        stats = gap_junctions.get_coupling_stats()

        assert stats["n_coupled_neurons"] == 0
        assert stats["n_connections"] == 0
        assert stats["avg_neighbors"] == 0.0
        assert stats["coupling_density"] == 0.0

    def test_factory_function(self, simple_weights):
        """Test convenience factory function."""
        gap_junctions = create_gap_junction_coupling(
            n_neurons=5,
            afferent_weights=simple_weights,
            coupling_strength=0.15,
            connectivity_threshold=0.4,
            max_neighbors=8,
            interneuron_only=True,
            device="cpu",
        )

        assert isinstance(gap_junctions, GapJunctionCoupling)
        assert gap_junctions.config.enabled is True
        assert gap_junctions.config.coupling_strength == 0.15
        assert gap_junctions.config.connectivity_threshold == 0.4
        assert gap_junctions.config.max_neighbors == 8

    def test_repr(self, simple_weights):
        """Test string representation."""
        config = GapJunctionConfig(enabled=True, coupling_strength=0.1)
        gap_junctions = GapJunctionCoupling(
            n_neurons=5,
            afferent_weights=simple_weights,
            config=config,
            device="cpu",
        )

        repr_str = repr(gap_junctions)
        assert "GapJunctionCoupling" in repr_str
        assert "coupled" in repr_str
        assert "connections" in repr_str

    def test_disabled_repr(self, simple_weights):
        """Test repr for disabled coupling."""
        config = GapJunctionConfig(enabled=False)
        gap_junctions = GapJunctionCoupling(
            n_neurons=5,
            afferent_weights=simple_weights,
            config=config,
            device="cpu",
        )

        repr_str = repr(gap_junctions)
        assert "disabled" in repr_str

    def test_bidirectional_coupling(self):
        """Test that gap junctions are bidirectional."""
        # Create strongly coupled pair
        weights = torch.ones(2, 10)
        config = GapJunctionConfig(
            enabled=True,
            coupling_strength=0.3,
            connectivity_threshold=0.0,
        )
        gap_junctions = GapJunctionCoupling(
            n_neurons=2,
            afferent_weights=weights,
            config=config,
            device="cpu",
        )

        # Both directions should have coupling
        coupling = gap_junctions.coupling_matrix
        assert coupling[0, 1] > 0
        assert coupling[1, 0] > 0

        # Coupling strength should be similar (symmetric)
        assert torch.abs(coupling[0, 1] - coupling[1, 0]) < 0.01


class TestGapJunctionIntegration:
    """Test integration with neuron dynamics."""

    def test_synchronization_effect(self):
        """Test that gap junctions promote synchrony over time."""
        # Create 3 coupled neurons
        weights = torch.ones(3, 10)
        gap_junctions = create_gap_junction_coupling(
            n_neurons=3,
            afferent_weights=weights,
            coupling_strength=0.2,
            connectivity_threshold=0.0,
            device="cpu",
        )

        # Start with different voltages
        voltages = torch.tensor([-60.0, -55.0, -50.0])

        # Simulate coupling dynamics
        voltage_history = [voltages.clone()]
        for _ in range(10):
            coupling_current = gap_junctions(voltages)
            # Simple Euler integration (dt=1ms, tau=10ms)
            voltages = voltages + 0.1 * coupling_current
            voltage_history.append(voltages.clone())

        # Voltage variance should decrease (more synchronized)
        initial_variance = voltage_history[0].var().item()
        final_variance = voltage_history[-1].var().item()

        assert final_variance < initial_variance  # Converging
        assert final_variance < initial_variance * 0.5  # At least 50% reduction


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
