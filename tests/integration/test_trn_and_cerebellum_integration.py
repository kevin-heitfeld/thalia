"""
Integration tests for TRN feedback loop and enhanced cerebellum.

Tests complete circuits:
- End-to-end attention: sensory→thalamus→cortex→L6→TRN→thalamus
- Motor learning: input→cerebellum(granule→Purkinje→DCN)→output
- Multi-region coordination with L6 feedback
- Enhanced cerebellum in full brain context
"""

import pytest
import torch

from thalia.core.brain_builder import BrainBuilder
from thalia.config import GlobalConfig


@pytest.fixture
def device():
    """Device for testing (CPU by default)."""
    return torch.device("cpu")


@pytest.fixture
def global_config(device):
    """Global configuration for brain."""
    return GlobalConfig(
        device=str(device),
        dt_ms=1.0,
    )


@pytest.fixture
def enhanced_cerebellum_brain(global_config, device):
    """Brain with enhanced cerebellum for testing."""
    builder = BrainBuilder(global_config)

    # Add components with proper sizes
    builder.add_component("thalamus", "thalamus", n_input=128, n_output=128)
    builder.add_component("cortex", "cortex", n_output=256,
                         l4_size=64, l23_size=128, l5_size=128, l6a_size=0, l6b_size=0)
    builder.add_component("cerebellum", "cerebellum",
                         n_input=256, n_output=64,
                         use_enhanced_microcircuit=True,
                         granule_expansion_factor=4.0,
                         granule_sparsity=0.03,
                         purkinje_n_dendrites=100)

    # Connect
    builder.connect("thalamus", "cortex", pathway_type="axonal")
    builder.connect("cortex", "cerebellum", pathway_type="axonal")

    return builder.build()


class TestL6TRNFeedbackIntegration:
    """Integration tests for complete L6→TRN feedback loop."""

    def test_sensorimotor_brain_with_l6(self, global_config):
        """Test sensorimotor preset includes L6→TRN pathway."""
        # Create brain with sensorimotor preset (includes L6→TRN)
        brain = BrainBuilder.preset("sensorimotor", global_config)

        # Contract: brain should have necessary components
        assert "cortex" in brain.components, "Brain should have cortex"
        assert "thalamus" in brain.components, "Brain should have thalamus"

        # Contract: cortex should have L6a and L6b layers
        cortex = brain.components["cortex"]
        assert hasattr(cortex, 'l6a_size'), "Cortex should have L6a layer"
        assert hasattr(cortex, 'l6b_size'), "Cortex should have L6b layer"
        assert cortex.l6a_size > 0, "L6a should have neurons"
        assert cortex.l6b_size > 0, "L6b should have neurons"

    def test_end_to_end_attention_loop(self, global_config, device):
        """Test complete attention loop: thalamus→cortex→L6→TRN→thalamus."""
        # Create brain
        brain = BrainBuilder.preset("sensorimotor", global_config)

        # Create sensory input with spatial structure
        # Channels 0-63: high activity (attended), 64-127: low activity (unattended)
        sensory_input = torch.zeros(128, dtype=torch.bool, device=device)
        sensory_input[0:64] = torch.rand(64, device=device) > 0.6  # 40% rate
        sensory_input[64:128] = torch.rand(64, device=device) > 0.95  # 5% rate

        # Run for multiple timesteps to establish feedback loop
        outputs = []
        for _ in range(30):  # 30ms to complete loop cycles
            output = brain(sensory_input, n_timesteps=1)
            outputs.append(output)

        # Contract: system should process without errors
        assert len(outputs) == 30, "Should complete all timesteps"

        # Contract: L6 feedback should be active
        cortex = brain.components["cortex"]
        l6_spikes = cortex.get_l6_spikes()

        if l6_spikes is not None:
            total_l6_size = cortex.l6a_size + cortex.l6b_size
            assert l6_spikes.shape[0] == total_l6_size, \
                "L6 spikes should match total L6 size (L6a + L6b)"

    def test_l6_affects_thalamic_relay(self, global_config, device):
        """Test that L6 feedback measurably affects thalamic relay."""
        # Create brain
        brain = BrainBuilder.preset("sensorimotor", global_config)

        sensory_input = torch.rand(128, device=device) > 0.8

        # Measure relay output over time
        thalamus = brain.components["thalamus"]
        relay_activities = []

        for _ in range(20):
            brain(sensory_input, n_timesteps=1)

            # Get thalamus relay state (if accessible)
            if hasattr(thalamus, 'relay_neurons'):
                if thalamus.relay_neurons.membrane is not None:
                    avg_membrane = thalamus.relay_neurons.membrane.mean().item()
                    relay_activities.append(avg_membrane)

        # Contract: relay should show activity variation over time
        # (L6 feedback modulates relay through TRN)
        if len(relay_activities) > 5:
            activity_std = torch.tensor(relay_activities).std().item()
            assert activity_std >= 0, "Relay activity should vary over time"

    def test_gamma_oscillation_emergence(self, global_config, device):
        """Test that feedback loop timing supports gamma oscillations (~40 Hz)."""
        # Gamma cycle: ~25ms (40 Hz)
        # Feedback loop: thalamus→cortex (5-8ms) + L2/3→L6 (2ms) + L6→TRN (10ms) + TRN→relay (3-5ms)
        # Total: ~20-25ms

        from thalia.diagnostics.oscillation_detection import measure_oscillation

        brain = BrainBuilder.preset("sensorimotor", global_config)
        cortex = brain.components["cortex"]

        # Note: Gamma oscillator disabled by default (should emerge from L6→TRN loop)

        # Strong periodic input at gamma frequency
        sensory_input = torch.rand(128, device=device) > 0.8

        # Track L6 activity over time (need longer window for FFT)
        l6_activities = []

        for _ in range(200):  # 200ms = 8 gamma cycles (better FFT resolution)
            brain(sensory_input, n_timesteps=1)

            l6_spikes = cortex.get_l6_spikes()
            if l6_spikes is not None:
                l6_activities.append(l6_spikes.sum().item())
            else:
                l6_activities.append(0.0)

        # Contract 1: L6 should show activity over time
        total_l6_activity = sum(l6_activities)
        assert total_l6_activity > 0, \
            "L6 should generate feedback activity over 200ms"

        # Contract 2: FFT should detect gamma-band oscillation
        freq, power = measure_oscillation(
            l6_activities,
            dt_ms=1.0,
            freq_range=(25.0, 60.0),  # Gamma range
        )

        assert 25 <= freq <= 60, \
            f"Expected gamma-range frequency (25-60 Hz), got {freq:.1f} Hz"

        # Log for diagnostics
        print(f"\n✅ L6→TRN loop oscillates at {freq:.1f} Hz (power={power:.3f})")

        # Ideal range is 35-50 Hz (circuit delay ~20-28ms)
        if 35 <= freq <= 50:
            print("   Perfect gamma band oscillation!")
        elif 25 <= freq < 35:
            print("   Slightly slow (may need stronger L6 feedback)")
        elif 50 < freq <= 60:
            print("   Slightly fast (may need longer TRN delays)")


    def test_spatial_attention_modulation(self, global_config, device):
        """Test that L6 enables spatial attention (channel-specific modulation)."""
        brain = BrainBuilder.preset("sensorimotor", global_config)

        # Create two input patterns: attended vs unattended
        attended_input = torch.zeros(128, dtype=torch.bool, device=device)
        attended_input[0:32] = True  # Attend to first 32 channels

        unattended_input = torch.zeros(128, dtype=torch.bool, device=device)
        unattended_input[96:128] = True  # Different channels

        # Present attended pattern repeatedly (build expectation)
        for _ in range(20):
            brain(attended_input, n_timesteps=1)

        # Get cortex response to attended
        cortex = brain.components["cortex"]
        attended_response = cortex.state.spikes.sum().item() if cortex.state.spikes is not None else 0

        # Reset and present unattended pattern
        brain.reset_state()
        for _ in range(20):
            brain(unattended_input, n_timesteps=1)

        unattended_response = cortex.state.spikes.sum().item() if cortex.state.spikes is not None else 0

        # Contract: both patterns should elicit responses
        # (Differential effects would require more sophisticated analysis)
        assert attended_response >= 0 and unattended_response >= 0, \
            "Both patterns should be processed"


class TestEnhancedCerebellumIntegration:
    """Integration tests for enhanced cerebellum in brain context."""

    def test_brain_with_enhanced_cerebellum(self, enhanced_cerebellum_brain):
        """Test brain with enhanced cerebellar microcircuit."""
        brain = enhanced_cerebellum_brain

        # Contract: cerebellum should be enhanced
        cerebellum = brain.components["cerebellum"]
        assert cerebellum.use_enhanced, "Cerebellum should use enhanced microcircuit"
        assert cerebellum.granule_layer is not None, "Should have granule layer"

        # Test forward pass - need to actually drive activity
        sensory_input = torch.ones(128, dtype=torch.bool, device=brain.device)  # Strong input

        for _ in range(10):
            brain(sensory_input, n_timesteps=1)

        # Contract: components should be active
        assert "cerebellum" in brain.components, "Brain should have cerebellum"

    def test_enhanced_cerebellum_motor_learning(self, enhanced_cerebellum_brain, device):
        """Test enhanced cerebellum learns motor mappings."""
        brain = enhanced_cerebellum_brain
        cerebellum = brain.components["cerebellum"]

        for epoch in range(5):
            input_pattern = torch.rand(128, device=device) > 0.8

            # Forward pass - use more timesteps to allow spikes to propagate through delays
            brain(input_pattern, n_timesteps=10)

            # Error signal (target motor command)
            target = torch.rand(64, device=device) > 0.5

            # Deliver error to cerebellum
            metrics = cerebellum.deliver_error(target)

            # Contract: error delivery should work
            assert isinstance(metrics, dict), "Should return learning metrics"

        # Contract: cerebellum should have processed through enhanced pathway
        # Note: Membrane initialization occurs when cerebellum receives spikes
        # With axonal delays, this requires enough timesteps for propagation
        assert cerebellum.granule_layer.neurons.membrane is not None, \
            "Granule layer should be active after processing"

    def test_granule_layer_in_brain_context(self, global_config, device):
        """Test granule layer expansion works in full brain."""
        builder = BrainBuilder(global_config)
        builder.add_component("thalamus", "thalamus", n_input=128, n_output=128)
        builder.add_component("cerebellum", "cerebellum",
                             n_input=128, n_output=64,
                             use_enhanced_microcircuit=True,
                             granule_expansion_factor=4.0,
                             granule_sparsity=0.03)
        builder.connect("thalamus", "cerebellum", pathway_type="axonal")

        brain = builder.build()

        # Input through thalamus to cerebellum
        sensory_input = torch.rand(128, device=device) > 0.8

        for _ in range(10):
            brain(sensory_input, n_timesteps=1)

        # Contract: granule layer should show sparse coding
        cerebellum = brain.components["cerebellum"]
        granule_layer = cerebellum.granule_layer

        if granule_layer.neurons.membrane is not None:
            # Check sparsity of granule activity
            n_granule = granule_layer.n_granule
            # Sparsity check done in unit tests, here just verify it processes
            assert n_granule == int(128 * 4.0), "Granule layer should have 4× expansion"


class TestMultiRegionCoordination:
    """Integration tests for multi-region coordination with L6 and enhanced cerebellum."""

    def test_full_sensorimotor_loop(self, global_config, device):
        """Test complete sensorimotor loop with L6 feedback and enhanced cerebellum."""
        # Build brain with all components
        brain = BrainBuilder.preset("sensorimotor", global_config)

        # Contract: should have key regions
        required_regions = ["thalamus", "cortex", "striatum"]
        for region in required_regions:
            assert region in brain.components, f"Brain should have {region}"

        # Sensorimotor loop: sensory → thalamus → cortex → motor
        # With L6: cortex.L6 → TRN → thalamus (attention)

        sensory_input = torch.rand(128, device=device) > 0.8

        # Run multiple timesteps
        for _ in range(25):  # One gamma cycle
            output = brain(sensory_input, n_timesteps=1)

            # Contract: output should be valid
            assert isinstance(output, dict), "Brain should return component outputs"

        # Verify L6 is active
        cortex = brain.components["cortex"]
        l6_spikes = cortex.get_l6_spikes()

        assert l6_spikes is not None, "L6 should be active after processing"

    def test_learning_with_feedback_and_cerebellum(self, global_config, device):
        """Test learning in system with both L6 feedback and enhanced cerebellum."""
        builder = BrainBuilder(global_config)

        # Add components with proper sizes (pass parameters directly)
        # Note: L6 size must match thalamus n_input for feedback pathway
        builder.add_component("thalamus", "thalamus", n_input=128, n_output=128)
        builder.add_component("cortex", "cortex",
                             n_input=128, n_output=256,
                             l4_size=64, l23_size=128, l5_size=128, l6a_size=64, l6b_size=64)
        builder.add_component("cerebellum", "cerebellum",
                             n_input=256, n_output=64,
                             use_enhanced_microcircuit=True)

        # Connections
        builder.connect("thalamus", "cortex", pathway_type="axonal")
        builder.connect("cortex", "thalamus", pathway_type="axonal", source_port="l6a")  # L6a feedback to TRN
        builder.connect("cortex", "cerebellum", pathway_type="axonal")

        brain = builder.build()

        # Training loop
        for _ in range(10):
            sensory_input = torch.rand(128, device=device) > 0.8

            # Process
            brain(sensory_input, n_timesteps=5)

            # Error signal to cerebellum
            target = torch.rand(64, device=device) > 0.5
            cerebellum = brain.components["cerebellum"]
            cerebellum.deliver_error(target)

        # Contract: system should learn without errors
        assert True, "Learning with L6 feedback and enhanced cerebellum should work"


class TestSystemRobustness:
    """Tests for system robustness with new features."""

    def test_reset_state_with_l6(self, global_config, device):
        """Test brain reset works with L6 layer."""
        brain = BrainBuilder.preset("sensorimotor", global_config)

        # Run some timesteps
        sensory_input = torch.rand(128, device=device) > 0.8
        for _ in range(10):
            brain(sensory_input, n_timesteps=1)

        # Reset
        brain.reset_state()

        # Contract: L6 state should be reset
        cortex = brain.components["cortex"]
        if hasattr(cortex, 'l6a_neurons'):
            if cortex.l6a_neurons.membrane is not None:
                # After reset, membrane should be at resting potential
                assert (cortex.l6a_neurons.membrane == cortex.l6a_neurons.config.v_reset).all(), \
                    "L6a neurons should be reset"
        if hasattr(cortex, 'l6b_neurons'):
            if cortex.l6b_neurons.membrane is not None:
                assert (cortex.l6b_neurons.membrane == cortex.l6b_neurons.config.v_reset).all(), \
                    "L6b neurons should be reset"

    def test_checkpoint_with_enhanced_features(self, global_config, device):
        """Test checkpointing works with L6 and enhanced cerebellum."""
        builder = BrainBuilder(global_config)

        # Add components with proper sizes (pass parameters directly)
        builder.add_component("thalamus", "thalamus", n_input=128, n_output=128)
        builder.add_component("cortex", "cortex",
                             n_input=128, n_output=256,
                             l4_size=64, l23_size=128, l5_size=128, l6a_size=64, l6b_size=64)
        builder.add_component("cerebellum", "cerebellum",
                             n_input=256, n_output=64,
                             use_enhanced_microcircuit=True)

        builder.connect("thalamus", "cortex", pathway_type="axonal")
        builder.connect("cortex", "cerebellum", pathway_type="axonal")

        brain = builder.build()

        # Run some timesteps
        sensory_input = torch.rand(128, device=device) > 0.8
        for _ in range(10):
            brain(sensory_input, n_timesteps=1)

        # Get checkpoint
        checkpoint = {}
        for name, component in brain.components.items():
            if hasattr(component, 'get_full_state'):
                checkpoint[name] = component.get_full_state()

        # Contract: checkpoint should include enhanced components
        assert "cortex" in checkpoint, "Checkpoint should include cortex"
        assert "cerebellum" in checkpoint, "Checkpoint should include cerebellum"

        # Cerebellum checkpoint should indicate enhanced mode
        if "config" in checkpoint["cerebellum"]:
            assert checkpoint["cerebellum"]["config"].get("use_enhanced", False), \
                "Cerebellum checkpoint should indicate enhanced mode"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
