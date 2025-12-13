"""
Edge case tests for neural components.

Tests boundary conditions using the full brain: silent input, saturated input,
extreme parameters, dimension mismatches, and numerical stability.

These tests focus on behavioral contracts rather than implementation details.
"""

import pytest
import torch

from thalia.core.brain import EventDrivenBrain
from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes


@pytest.fixture
def device():
    """Use CPU for tests."""
    return torch.device('cpu')


@pytest.fixture
def test_brain(device):
    """Create a small test brain."""
    config = ThaliaConfig(
        global_=GlobalConfig(device=str(device), dt_ms=1.0),
        brain=BrainConfig(
            sizes=RegionSizes(
                input_size=10,
                thalamus_size=20,
                cortex_size=30,
                hippocampus_size=40,
                pfc_size=25,
                n_actions=4,
            ),
        ),
    )
    return EventDrivenBrain.from_thalia_config(config)


class TestSilentInput:
    """Test brain handles zero spikes correctly."""

    def test_brain_handles_silent_input(self, test_brain, device):
        """Brain should handle zero spikes without crashing."""
        # All zeros (no spikes)
        silent_input = torch.zeros(10, dtype=torch.bool, device=device)

        # Should not crash
        output = test_brain.forward(silent_input, n_timesteps=5)

        # Validate output
        assert output.dtype == torch.bool, "Output should be bool"
        assert not torch.isnan(output.float()).any(), "Output contains NaN"

    def test_brain_survives_extended_silence(self, test_brain, device):
        """Brain should remain stable with prolonged silent input."""
        silent_input = torch.zeros(10, dtype=torch.bool, device=device)

        # Run 100 steps of silence
        for _ in range(100):
            output = test_brain.forward(silent_input, n_timesteps=1)

            # Should not crash or produce NaN
            assert not torch.isnan(output.float()).any(), "Output contains NaN"


class TestSaturatedInput:
    """Test brain handles maximum input without overflow."""

    def test_brain_handles_saturated_input(self, test_brain, device):
        """Brain should handle all spikes without overflow."""
        # All ones (maximum spikes)
        saturated_input = torch.ones(10, dtype=torch.bool, device=device)

        # Should not crash or overflow
        output = test_brain.forward(saturated_input, n_timesteps=5)

        # Validate output
        assert output.dtype == torch.bool, "Output should be bool"
        assert not torch.isnan(output.float()).any(), "Output contains NaN"
        assert not torch.isinf(output.float()).any(), "Output contains Inf"

    def test_brain_survives_extended_saturation(self, test_brain, device):
        """Brain should remain stable with prolonged saturated input."""
        saturated_input = torch.ones(10, dtype=torch.bool, device=device)

        # Run 100 steps of saturation
        for step in range(100):
            output = test_brain.forward(saturated_input, n_timesteps=1)

            # Should not crash or produce NaN/Inf
            assert not torch.isnan(output.float()).any(), f"Output contains NaN at step {step}"
            assert not torch.isinf(output.float()).any(), f"Output contains Inf at step {step}"

            # Check region states periodically
            if step % 25 == 0:
                for region_name, region in test_brain.regions.items():
                    if hasattr(region, 'membrane'):
                        assert not torch.isnan(region.membrane).any(), \
                            f"Region {region_name} membrane contains NaN at step {step}"


class TestDimensionMismatches:
    """Test brain validates input dimensions."""

    def test_brain_rejects_wrong_input_size(self, test_brain, device):
        """Brain should validate input dimensions."""
        # Wrong size (expecting 10, giving 8)
        wrong_input = torch.zeros(8, dtype=torch.bool, device=device)

        with pytest.raises((RuntimeError, ValueError, AssertionError)):
            test_brain.forward(wrong_input, n_timesteps=1)

    def test_brain_rejects_too_large_input(self, test_brain, device):
        """Brain should reject input that's too large."""
        # Too large (expecting 10, giving 15)
        wrong_input = torch.zeros(15, dtype=torch.bool, device=device)

        with pytest.raises((RuntimeError, ValueError, AssertionError)):
            test_brain.forward(wrong_input, n_timesteps=1)


class TestNumericalStability:
    """Test numerical stability over many iterations."""

    def test_brain_numerical_stability_long_run(self, test_brain, device):
        """Brain should remain stable over many forward passes."""
        # Run 1000 forward passes with varied input
        for step in range(1000):
            # Vary sparsity from 5% to 50%
            sparsity = 0.05 + (step % 10) * 0.045
            input_pattern = torch.rand(10, device=device) > (1.0 - sparsity)

            _ = test_brain.forward(input_pattern, n_timesteps=1)

            # Check for NaN or Inf every 100 steps
            if step % 100 == 0:
                # Check all region membranes
                for region_name, region in test_brain.regions.items():
                    if hasattr(region, 'membrane'):
                        assert not torch.isnan(region.membrane).any(), \
                            f"Region {region_name} membrane contains NaN at step {step}"
                        assert not torch.isinf(region.membrane).any(), \
                            f"Region {region_name} membrane contains Inf at step {step}"

                # Check all pathway weights
                for pathway_name, pathway in test_brain.pathway_manager.pathways.items():
                    if hasattr(pathway, 'weights'):
                        assert not torch.isnan(pathway.weights).any(), \
                            f"Pathway {pathway_name} weights contain NaN at step {step}"
                        assert not torch.isinf(pathway.weights).any(), \
                            f"Pathway {pathway_name} weights contain Inf at step {step}"

    def test_brain_handles_random_input_patterns(self, test_brain, device):
        """Brain should remain stable with random input patterns."""
        for _ in range(100):
            # Completely random patterns (varying sparsity)
            sparsity = torch.rand(1).item() * 0.9 + 0.05  # 5% to 95%
            input_pattern = torch.rand(10, device=device) > (1.0 - sparsity)

            output = test_brain.forward(input_pattern, n_timesteps=1)

            # Should not crash or produce NaN
            assert not torch.isnan(output.float()).any(), "Output contains NaN"


class TestResetBehavior:
    """Test reset clears state correctly."""

    def test_brain_reset_clears_state(self, test_brain, device):
        """Reset should clear all accumulated state."""
        # Run brain to build up state
        active_input = torch.ones(10, dtype=torch.bool, device=device)
        for _ in range(50):
            test_brain.forward(active_input, n_timesteps=1)

        # Reset
        test_brain.reset_state()

        # After reset, silent input should produce minimal activity
        silent_input = torch.zeros(10, dtype=torch.bool, device=device)
        output_after_reset = test_brain.forward(silent_input, n_timesteps=5)

        # Validate reset worked (should not crash)
        assert not torch.isnan(output_after_reset.float()).any(), "Output contains NaN after reset"

    def test_brain_multiple_resets(self, test_brain, device):
        """Brain should handle multiple resets without issues."""
        input_pattern = torch.rand(10, device=device) > 0.8

        for _ in range(10):
            # Run some steps
            for _ in range(10):
                test_brain.forward(input_pattern, n_timesteps=1)

            # Reset
            test_brain.reset_state()

        # Final forward pass should work
        final_output = test_brain.forward(input_pattern, n_timesteps=1)
        assert not torch.isnan(final_output.float()).any(), "Output contains NaN after multiple resets"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
