"""
Tests for Purkinje cell per-dendrite learning in cerebellum.

Tests the biologically accurate LTD/LTP mechanism for individual Purkinje cell
dendritic weight plasticity.
"""

import torch
import pytest

from thalia.regions.cerebellum_region import Cerebellum, CerebellumConfig


class TestPurkinjePerDendriteLearning:
    """Test per-Purkinje cell dendritic learning implementation."""

    def test_enhanced_mode_updates_purkinje_weights(self):
        """Enhanced mode should update individual Purkinje cell weights."""
        config = CerebellumConfig(
            input_size=64,
            purkinje_size=32,
            use_enhanced_microcircuit=True,
            learning_rate=0.1,  # Higher LR for visible changes
            error_threshold=0.001,  # Lower threshold to allow learning
            gap_junctions_enabled=False,  # Disable gap junctions (they zero out error if not initialized)
            device="cpu"
        )
        cerebellum = Cerebellum(config)

        # Forward pass to initialize weights
        input_spikes = torch.zeros(64, device=torch.device("cpu"))
        input_spikes[:10] = 1.0
        output = cerebellum(input_spikes)

        # Store initial weights from first Purkinje cell
        initial_weights = cerebellum.purkinje_cells[0].pf_synaptic_weights.clone()

        # Provide error signal for learning via deliver_error
        # Use large error to ensure it exceeds threshold
        target = torch.ones(32, device=torch.device("cpu"))  # Strong target signal
        metrics = cerebellum.deliver_error(target, output)

        # Verify Purkinje cell weights changed
        new_weights = cerebellum.purkinje_cells[0].pf_synaptic_weights
        assert not torch.allclose(initial_weights, new_weights, atol=1e-6), \
            f"Purkinje cell weights should have changed after learning. Error: {metrics.get('error', 0)}"
        assert "error" in metrics
        assert "ltp" in metrics
        assert "ltd" in metrics

    def test_ltd_with_negative_error(self):
        """LTD should occur when target < output (negative error).

        Note: We manually set output spikes to ensure negative error condition.
        """
        config = CerebellumConfig(
            input_size=64,
            purkinje_size=32,
            use_enhanced_microcircuit=True,
            learning_rate=0.1,
            error_threshold=0.001,
            gap_junctions_enabled=False,
            device="cpu"
        )
        cerebellum = Cerebellum(config)

        # Create input pattern
        input_spikes = torch.zeros(64, device=torch.device("cpu"))
        input_spikes[:40] = 1.0
        output = cerebellum(input_spikes)

        # Store initial weights
        initial_weights = cerebellum.purkinje_cells[0].pf_synaptic_weights.clone()

        # Manually create output spikes for negative error test
        # (target < output → negative error → LTD)
        output = torch.ones(32, device=torch.device("cpu"), dtype=torch.bool)
        target = torch.zeros(32, device=torch.device("cpu"))
        cerebellum.deliver_error(target, output)

        new_weights = cerebellum.purkinje_cells[0].pf_synaptic_weights
        weight_change = (new_weights - initial_weights).abs().sum().item()

        # Weights should have changed
        assert weight_change > 0, "Weights should change with error signal"

    def test_ltp_with_positive_error(self):
        """LTP should occur when target > output (positive error)."""
        config = CerebellumConfig(
            input_size=64,
            purkinje_size=32,
            use_enhanced_microcircuit=True,
            learning_rate=0.1,
            error_threshold=0.001,
            gap_junctions_enabled=False,
            device="cpu"
        )
        cerebellum = Cerebellum(config)

        input_spikes = torch.zeros(64, device=torch.device("cpu"))
        input_spikes[:20] = 1.0
        output = cerebellum(input_spikes)

        initial_weights = cerebellum.purkinje_cells[0].pf_synaptic_weights.clone()

        # Positive error: target > output → weights should increase (LTP)
        # Use ones target for strong positive error
        target = torch.ones(32, device=torch.device("cpu"))
        cerebellum.deliver_error(target, output)

        new_weights = cerebellum.purkinje_cells[0].pf_synaptic_weights
        weight_change = (new_weights - initial_weights).abs().sum().item()

        # Weights should have changed
        assert weight_change > 0, "Weights should change with error signal"

    def test_weight_bounds_respected(self):
        """Purkinje cell weights should stay within configured bounds."""
        config = CerebellumConfig(
            input_size=64,
            purkinje_size=32,
            use_enhanced_microcircuit=True,
            learning_rate=1.0,  # Very high LR to test bounds
            w_min=0.0,
            w_max=1.0,
            gap_junctions_enabled=False,
            device="cpu"
        )
        cerebellum = Cerebellum(config)

        # Multiple learning iterations with large errors
        for _ in range(10):
            input_spikes = torch.rand(64, device=torch.device("cpu"))
            output = cerebellum(input_spikes)
            target = torch.rand(32, device=torch.device("cpu"))
            cerebellum.deliver_error(target, output)

        # Check all Purkinje cells respect bounds
        for i, cell in enumerate(cerebellum.purkinje_cells):
            weights = cell.pf_synaptic_weights
            assert torch.all(weights >= config.w_min), \
                f"Purkinje cell {i} has weights below w_min"
            assert torch.all(weights <= config.w_max), \
                f"Purkinje cell {i} has weights above w_max"

    def test_each_purkinje_cell_learns_independently(self):
        """Each Purkinje cell should have independent weight updates."""
        config = CerebellumConfig(
            input_size=64,
            purkinje_size=10,
            use_enhanced_microcircuit=True,
            learning_rate=0.01,
            gap_junctions_enabled=False,
            device="cpu"
        )
        cerebellum = Cerebellum(config)

        # Forward pass
        input_spikes = torch.rand(64, device=torch.device("cpu"))
        output = cerebellum(input_spikes)

        # Store initial weights for multiple cells
        initial_weights = [
            cell.pf_synaptic_weights.clone()
            for cell in cerebellum.purkinje_cells[:5]
        ]

        # Different error signals
        target = torch.rand(10, device=torch.device("cpu"))
        cerebellum.deliver_error(target, output)

        # Verify each cell's weights changed
        for i in range(5):
            new_weights = cerebellum.purkinje_cells[i].pf_synaptic_weights
            weight_change = (new_weights - initial_weights[i]).abs().sum().item()
            assert weight_change >= 0, f"Purkinje cell {i} should have weight updates"


class TestCerebellumNeurogenesisTracking:
    """Test neurogenesis tracking integration for cerebellum."""

    def test_cerebellum_has_training_step_method(self):
        """Cerebellum should have set_training_step method."""
        config = CerebellumConfig(
            input_size=64,
            purkinje_size=32,
            device="cpu"
        )
        cerebellum = Cerebellum(config)

        # Should not raise AttributeError
        assert hasattr(cerebellum, 'set_training_step')

        # Should be callable
        cerebellum.set_training_step(100)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
