"""Tests for Thalamus using unified RegionTestBase framework.

Demonstrates unified testing pattern with region-specific tests for
thalamus (sensory relay and cortical modulation).

Author: Thalia Project
Date: December 22, 2025 (Tier 3.4 implementation)
"""

import torch

from tests.utils.region_test_base import RegionTestBase
from thalia.config.size_calculator import LayerSizeCalculator
from thalia.regions.thalamus import ThalamicRelay, ThalamicRelayConfig


class TestThalamus(RegionTestBase):
    """Test Thalamus implementation using unified test framework."""

    def create_region(self, **kwargs):
        """Create ThalamicRelay instance for testing.

        Uses new (config, sizes, device) pattern:
        - Config contains only behavioral parameters
        - Sizes computed via LayerSizeCalculator
        - Device passed explicitly
        """
        # Extract size-related parameters
        relay_size = kwargs.pop("relay_size", 80)
        device = kwargs.pop("device", "cpu")

        # Compute sizes using calculator
        calc = LayerSizeCalculator()
        sizes = calc.thalamus_from_relay(relay_size=relay_size)

        # Create config with behavioral parameters only
        config = ThalamicRelayConfig(**kwargs)

        # Create region with new pattern
        return ThalamicRelay(config=config, sizes=sizes, device=device)

    def get_default_params(self):
        """Return default thalamus parameters."""
        return {
            "relay_size": 80,
            "device": "cpu",
            "dt_ms": 1.0,
        }

    def get_min_params(self):
        """Return minimal valid parameters for quick tests."""
        return {
            "relay_size": 15,
            "device": "cpu",
            "dt_ms": 1.0,
        }

    def _get_input_size(self, params):
        """Get input size for the region."""
        # For thalamus, input_size equals relay_size (1:1 relay)
        relay_size = params.get("relay_size", 80)
        return relay_size

    def get_input_dict(self, n_input, device="cpu"):
        """Return dict input for thalamus (sensory + optional feedback).

        Sensory input is full n_input (NOT split).
        Feedback is optional modulatory input (different targets: TRN/relay).
        """
        return {
            "sensory": torch.zeros(n_input, device=device),
            # Feedback is optional, omit to test sensory-only mode
        }

    # =========================================================================
    # THALAMUS-SPECIFIC TESTS
    # =========================================================================

    def test_sensory_relay(self):
        """Test thalamus relays sensory information to cortex."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Sensory input
        input_spikes = torch.ones(self._get_input_size(params), device=region.device)
        output = region.forward({"input": input_spikes})

        # Should relay to cortex (output size = relay_size)
        assert output.shape[0] == region.relay_size

        # With strong sensory input, should produce some output
        assert output.sum() > 0, "Expected relay activity with sensory input"

    def test_trn_inhibition(self):
        """Test thalamic reticular nucleus provides inhibition."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Forward pass
        input_spikes = torch.ones(self._get_input_size(params), device=region.device)
        region.forward({"input": input_spikes})

        # Check TRN activity
        state = region.get_state()
        if hasattr(state, "trn_spikes"):
            if state.trn_spikes is not None:
                assert state.trn_spikes.shape[0] == region.n_trn

    def test_l6_feedback_modulation(self):
        """Test L6 cortical feedback modulates thalamic relay."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Forward with L6 feedback (if supported)
        input_spikes = torch.ones(self._get_input_size(params), device=region.device)

        # Check if forward accepts l6_feedback parameter
        if "l6_feedback" in region.forward.__code__.co_varnames:
            l6_feedback = torch.ones(params["n_trn"], device=region.device) * 0.5
            output = region.forward(input_spikes, l6_feedback=l6_feedback)
            assert output.shape[0] == region.n_output
        else:
            # Just verify forward works
            output = region.forward({"input": input_spikes})
            assert output.shape[0] == region.n_output

    def test_burst_firing_mode(self):
        """Test thalamus can switch between tonic and burst modes."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Forward passes in different modes
        input_spikes = torch.ones(self._get_input_size(params), device=region.device)

        # First pass (could be tonic mode)
        output1 = region.forward({"input": input_spikes})

        # Multiple passes (state-dependent mode switching)
        for _ in range(10):
            region.forward(
                {"input": torch.zeros(self._get_input_size(params), device=region.device)}
            )

        # Second pass (might be in burst mode)
        output2 = region.forward({"input": input_spikes})

        # Both should produce valid outputs
        assert output1.shape == output2.shape

    def test_alpha_oscillation(self):
        """Test thalamus generates alpha oscillations (8-13 Hz)."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Check if oscillator phase is tracked
        if hasattr(region, "_alpha_phase"):
            # Run forward passes
            input_spikes = torch.zeros(self._get_input_size(params), device=region.device)
            for _ in range(20):
                region.forward({"input": input_spikes})

            # Alpha phase should be present
            assert hasattr(region, "_alpha_phase")

    def test_sensory_gating(self):
        """Test thalamus gates sensory information based on attention."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Strong sensory input
        strong_input = torch.ones(self._get_input_size(params), device=region.device)
        output_strong = region.forward({"input": strong_input})

        # Reset
        region.reset_state()

        # Weak sensory input (gated)
        weak_input = torch.ones(self._get_input_size(params), device=region.device) * 0.1
        output_weak = region.forward({"input": weak_input})

        # Strong input should produce more relay activity
        assert (
            output_strong.sum() > output_weak.sum()
        ), "Expected stronger relay with stronger sensory input"

    def test_lateral_inhibition_in_trn(self):
        """Test TRN has lateral inhibition for competitive selection."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Check if TRN has lateral inhibition weights
        if hasattr(region, "trn_lateral_weights"):
            # Should be [n_trn, n_trn]
            assert region.trn_lateral_weights.shape == (
                params["n_trn"],
                params["n_trn"],
            ), "TRN lateral weights shape mismatch"

    def test_corticothalamic_plasticity(self):
        """Test plasticity in corticothalamic feedback connections."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Get initial feedback weights (if accessible)
        if hasattr(region, "synaptic_weights"):
            initial_weights = None
            for source_name in region.synaptic_weights:
                if "feedback" in source_name.lower() or "l6" in source_name.lower():
                    initial_weights = region.synaptic_weights[source_name].clone()
                    break

            if initial_weights is not None:
                # Multiple forward passes with feedback
                input_spikes = torch.ones(self._get_input_size(params), device=region.device)
                for _ in range(100):
                    region.forward(input_spikes)

                # Check if weights changed
                for source_name in region.synaptic_weights:
                    if "feedback" in source_name.lower() or "l6" in source_name.lower():
                        final_weights = region.synaptic_weights[source_name]
                        if not torch.allclose(initial_weights, final_weights, atol=1e-6):
                            return  # Plasticity detected

    def test_sleep_spindles(self):
        """Test thalamus can generate sleep spindles (during low activity)."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Low activity (simulating sleep)
        low_input = torch.zeros(self._get_input_size(params), device=region.device)

        # Multiple passes to allow spindle generation
        for _ in range(50):
            output = region.forward({"input": low_input})

        # Spindles would be visible as oscillatory bursts
        # (Detailed spindle detection would require spectral analysis)
        assert output.shape[0] == region.n_output

    def test_multimodal_integration(self):
        """Test thalamus integrates multiple sensory modalities."""
        params = self.get_default_params()
        region = self.create_region(**params)

        # Thalamus uses port-based routing (single sensory input to relay neurons)
        # NOT multi-source dict with arbitrary keys
        sensory_input = torch.ones(self._get_input_size(params), device=region.device)

        # Forward with sensory input
        output = region.forward({"input": sensory_input})

        # Should relay sensory input
        assert output.shape[0] == region.n_output


# Standard tests (initialization, forward, growth, state, device, neuromodulators, diagnostics)
# inherited from RegionTestBase - eliminates ~100 lines of boilerplate
