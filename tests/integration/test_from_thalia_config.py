"""
Integration tests for DynamicBrain.from_thalia_config() backward compatibility.

These tests validate that DynamicBrain can serve as a drop-in replacement
for EventDrivenBrain when constructed from ThaliaConfig.

Tests:
    - Component sizes match config
    - All expected components exist
    - Connections are correct
    - Forward pass executes successfully
    - RL interface works (select_action, deliver_reward)
    - Output dimensions match EventDrivenBrain
"""

import pytest
import torch

from thalia.core.dynamic_brain import DynamicBrain
from thalia.core.brain import EventDrivenBrain
from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes


@pytest.fixture
def device():
    """Get device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def minimal_config(device):
    """Create minimal ThaliaConfig for testing."""
    return ThaliaConfig(
        global_=GlobalConfig(
            device=device,
            dt_ms=1.0,
            theta_frequency_hz=8.0,
        ),
        brain=BrainConfig(
            device=device,  # Must match global_.device
            sizes=RegionSizes(
                input_size=64,
                thalamus_size=64,
                cortex_size=128,
                hippocampus_size=64,
                pfc_size=32,
                n_actions=4,
            ),
        ),
    )


@pytest.fixture
def standard_config(device):
    """Create standard ThaliaConfig with default sizes."""
    return ThaliaConfig(
        global_=GlobalConfig(device=device, dt_ms=1.0),
        brain=BrainConfig(
            device=device,  # Must match global_.device
            sizes=RegionSizes(
                input_size=784,     # MNIST size
                thalamus_size=784,
                cortex_size=256,
                hippocampus_size=128,
                pfc_size=64,
                n_actions=10,
            ),
        ),
    )


class TestFromThaliaConfigBasic:
    """Basic validation of DynamicBrain.from_thalia_config()."""

    def test_from_thalia_config_creates_brain(self, minimal_config):
        """Test that from_thalia_config() creates a DynamicBrain instance."""
        brain = DynamicBrain.from_thalia_config(minimal_config)

        assert isinstance(brain, DynamicBrain)
        assert brain.device == torch.device(minimal_config.global_.device)

    def test_has_expected_components(self, minimal_config):
        """Test that all expected components are created."""
        brain = DynamicBrain.from_thalia_config(minimal_config)

        # Expected components from sensorimotor architecture
        expected_components = ["thalamus", "cortex", "hippocampus", "pfc", "striatum", "cerebellum"]

        for component in expected_components:
            assert component in brain.components, f"Missing component: {component}"

    def test_component_sizes_match_config(self, minimal_config):
        """Test that component sizes match ThaliaConfig."""
        brain = DynamicBrain.from_thalia_config(minimal_config)
        sizes = minimal_config.brain.sizes

        # Check thalamus size
        thalamus = brain.components["thalamus"]
        assert hasattr(thalamus.config, "n_output")
        assert thalamus.config.n_output == sizes.thalamus_size

        # Check cortex size (LayeredCortex outputs L2/3 + L5 = 2.5× n_output)
        cortex = brain.components["cortex"]
        assert hasattr(cortex.config, "n_output")
        # LayeredCortex: l23_ratio=1.5 + l5_ratio=1.0 = 2.5× the base size
        expected_cortex_output = int(sizes.cortex_size * 2.5)
        assert cortex.config.n_output == expected_cortex_output

        # Check hippocampus size
        hippo = brain.components["hippocampus"]
        assert hasattr(hippo.config, "n_output")
        assert hippo.config.n_output == sizes.hippocampus_size

        # Check PFC size
        pfc = brain.components["pfc"]
        assert hasattr(pfc.config, "n_output")
        assert pfc.config.n_output == sizes.pfc_size

        # Check striatum actions (stored as instance attribute, not in config)
        striatum = brain.components["striatum"]
        assert hasattr(striatum, "n_actions")
        assert striatum.n_actions == sizes.n_actions

    def test_has_expected_connections(self, minimal_config):
        """Test that expected connections are created."""
        brain = DynamicBrain.from_thalia_config(minimal_config)

        # Expected connections from sensorimotor architecture
        expected_connections = [
            ("thalamus", "cortex"),
            ("cortex", "hippocampus"),
            ("hippocampus", "cortex"),  # Bidirectional
            ("cortex", "pfc"),
            ("pfc", "striatum"),
            ("striatum", "pfc"),  # Bidirectional
            ("pfc", "cerebellum"),
        ]

        for (source, target) in expected_connections:
            assert (source, target) in brain.connections, \
                f"Missing connection: {source} -> {target}"

    def test_forward_pass_executes(self, minimal_config):
        """Test that forward pass executes without errors."""
        brain = DynamicBrain.from_thalia_config(minimal_config)

        # Create input matching thalamus size (which takes input_size)
        input_data = {
            "thalamus": torch.randn(minimal_config.brain.sizes.input_size, device=brain.device)
        }

        # Execute forward pass
        result = brain.forward(input_data, n_timesteps=10)

        assert "outputs" in result
        assert "thalamus" in result["outputs"]


class TestFromThaliaConfigRLInterface:
    """Test that RL interface works with from_thalia_config() brains."""

    def test_select_action_works(self, minimal_config):
        """Test that select_action() works after from_thalia_config()."""
        brain = DynamicBrain.from_thalia_config(minimal_config)

        # Run some forward passes to activate striatum
        input_data = {
            "thalamus": torch.randn(minimal_config.brain.sizes.input_size, device=brain.device)
        }
        for _ in range(3):
            brain.forward(input_data, n_timesteps=10)

        # Select action
        action, confidence = brain.select_action(explore=True, use_planning=False)

        assert isinstance(action, int)
        assert 0 <= action < minimal_config.brain.sizes.n_actions
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_deliver_reward_works(self, minimal_config):
        """Test that deliver_reward() works after from_thalia_config()."""
        brain = DynamicBrain.from_thalia_config(minimal_config)

        # Run forward pass and select action
        input_data = {
            "thalamus": torch.randn(minimal_config.brain.sizes.input_size, device=brain.device)
        }
        brain.forward(input_data, n_timesteps=10)
        action, _ = brain.select_action(explore=True, use_planning=False)

        # Deliver reward (should not raise)
        brain.deliver_reward(external_reward=1.0)

    def test_rl_episode_loop(self, minimal_config):
        """Test full RL episode loop with from_thalia_config() brain."""
        brain = DynamicBrain.from_thalia_config(minimal_config)

        # Run 5-step episode
        for step in range(5):
            # Forward pass
            input_data = {
                "thalamus": torch.randn(minimal_config.brain.sizes.input_size, device=brain.device)
            }
            brain.forward(input_data, n_timesteps=10)

            # Select action
            action, confidence = brain.select_action(explore=True, use_planning=False)

            # Deliver reward
            reward = 1.0 if step % 2 == 0 else 0.0
            brain.deliver_reward(external_reward=reward)

        # Should complete without errors
        assert True


class TestFromThaliaConfigOutputDimensions:
    """Test that output dimensions match expectations."""

    def test_thalamus_output_dimension(self, minimal_config):
        """Test thalamus output has correct dimension."""
        brain = DynamicBrain.from_thalia_config(minimal_config)

        input_data = {
            "thalamus": torch.randn(minimal_config.brain.sizes.input_size, device=brain.device)
        }
        result = brain.forward(input_data, n_timesteps=5)

        # Thalamus output should match thalamus_size
        thalamus_out = result["outputs"]["thalamus"]
        if thalamus_out is not None:
            assert thalamus_out.shape[0] == minimal_config.brain.sizes.thalamus_size

    def test_cortex_output_dimension(self, minimal_config):
        """Test cortex output has correct dimension.

        Note: LayeredCortex concatenates L2/3 and L5 outputs, so total size
        is larger than config.n_output.
        """
        brain = DynamicBrain.from_thalia_config(minimal_config)

        input_data = {
            "thalamus": torch.randn(minimal_config.brain.sizes.input_size, device=brain.device)
        }
        result = brain.forward(input_data, n_timesteps=10)

        # Cortex output exists (may be None if no spikes, but should be tensor)
        cortex_out = result["outputs"]["cortex"]
        if cortex_out is not None:
            # LayeredCortex: L2/3 (1.5x) + L5 (1.0x) = 2.5x config size
            # For config.cortex_size=128: L2/3=192 + L5=128 = 320
            expected_total = int(minimal_config.brain.sizes.cortex_size * 2.5)
            assert cortex_out.shape[0] == expected_total, \
                f"Expected cortex output {expected_total}, got {cortex_out.shape[0]}"


class TestFromThaliaConfigEquivalence:
    """Test that DynamicBrain behavior is equivalent to EventDrivenBrain."""

    def test_same_config_produces_similar_outputs(self, minimal_config):
        """Test that same config produces architecturally similar brains.

        Note: Full equivalence is NOT expected due to fundamental differences:
        1. **Layer Routing**: EventDrivenBrain routes cortex L2/3 vs L5 separately;
           DynamicBrain routes whole cortex output as one signal
        2. **Direct Sensory Paths**: EventDrivenBrain has direct sensory→hippocampus
           (ec_l3); DynamicBrain only has sensory→thalamus→cortex→hippocampus
        3. **Top-down Modulation**: EventDrivenBrain handles PFC→cortex via separate
           parameter; DynamicBrain would sum it into n_input
        4. **Random Initialization**: Different random seeds produce different weights
        5. **Execution Models**: Event-driven scheduler vs component graph

        This test validates what CAN be equivalent:
        - Same region types present (6 regions)
        - Same region names
        - Same n_actions value
        - Same output sizes (where comparable)
        - Same device
        """
        dynamic_brain = DynamicBrain.from_thalia_config(minimal_config)
        event_brain = EventDrivenBrain.from_thalia_config(minimal_config)

        # Both should have 6 regions in sensorimotor architecture
        assert len(dynamic_brain.components) == 6
        assert len(event_brain.adapters) == 6

        # Check region names match
        expected_regions = {"thalamus", "cortex", "hippocampus", "pfc", "striatum", "cerebellum"}
        assert set(dynamic_brain.components.keys()) == expected_regions
        assert set(event_brain.adapters.keys()) == expected_regions

        # Compare thalamus (simplest region - should match exactly)
        dynamic_thalamus = dynamic_brain.components["thalamus"]
        event_thalamus = event_brain.adapters["thalamus"].impl
        assert dynamic_thalamus.config.n_input == event_thalamus.config.n_input
        assert dynamic_thalamus.config.n_output == event_thalamus.config.n_output

        # Compare cortex output size (both should have 2.5× multiplier)
        dynamic_cortex = dynamic_brain.components["cortex"]
        event_cortex = event_brain.adapters["cortex"].impl
        assert dynamic_cortex.config.n_output == event_cortex.config.n_output  # Both 320
        assert dynamic_cortex.l23_size == event_cortex.l23_size  # Both 192
        assert dynamic_cortex.l5_size == event_cortex.l5_size   # Both 128
        # NOTE: n_input differs due to topology (DynamicBrain: 64, EventDrivenBrain: 64+32 top-down)

        # Compare hippocampus output (both should match hippocampus_size)
        dynamic_hippo = dynamic_brain.components["hippocampus"]
        event_hippo = event_brain.adapters["hippocampus"].impl
        assert dynamic_hippo.config.n_output == event_hippo.config.n_output  # Both 64
        # NOTE: n_input differs due to layer routing and ec_l3 path

        # Compare PFC output
        dynamic_pfc = dynamic_brain.components["pfc"]
        event_pfc = event_brain.adapters["pfc"].impl
        assert dynamic_pfc.config.n_output == event_pfc.config.n_output  # Both 32
        # NOTE: n_input differs due to topology

        # Compare striatum (n_actions is key compatibility metric)
        dynamic_striatum = dynamic_brain.components["striatum"]
        event_striatum = event_brain.adapters["striatum"].impl
        assert dynamic_striatum.n_actions == event_striatum.n_actions
        assert dynamic_striatum.n_actions == minimal_config.brain.sizes.n_actions
        # NOTE: n_input differs due to layer routing

        # Compare cerebellum output
        dynamic_cerebellum = dynamic_brain.components["cerebellum"]
        event_cerebellum = event_brain.adapters["cerebellum"].impl
        assert dynamic_cerebellum.config.n_output == event_cerebellum.config.n_output
        # NOTE: n_input differs due to topology

        # Both should use same device
        assert dynamic_brain.device == event_brain.device

        # Document key differences for future reference
        print("\n=== Known Architectural Differences ===")
        print(f"Cortex n_input: Dynamic={dynamic_cortex.config.n_input}, Event={event_cortex.config.n_input}")
        print(f"Hippo n_input: Dynamic={dynamic_hippo.config.n_input}, Event={event_hippo.config.n_input}")
        print(f"PFC n_input: Dynamic={dynamic_pfc.config.n_input}, Event={event_pfc.config.n_input}")
        print("These differences are expected due to layer-specific routing in EventDrivenBrain.")


class TestFromThaliaConfigEdgeCases:
    """Test edge cases and error handling."""

    def test_with_standard_sizes(self, standard_config):
        """Test with standard MNIST-sized configuration."""
        brain = DynamicBrain.from_thalia_config(standard_config)

        assert brain is not None
        assert len(brain.components) == 6
        # Striatum stores n_actions as instance attribute, not in config
        assert brain.components["striatum"].n_actions == 10

    def test_multiple_forward_passes(self, minimal_config):
        """Test that multiple forward passes work correctly."""
        brain = DynamicBrain.from_thalia_config(minimal_config)

        input_data = {
            "thalamus": torch.randn(minimal_config.brain.sizes.input_size, device=brain.device)
        }

        # Execute 10 forward passes
        for _ in range(10):
            result = brain.forward(input_data, n_timesteps=5)
            assert "outputs" in result

    def test_with_different_input_each_time(self, minimal_config):
        """Test with varying input patterns."""
        brain = DynamicBrain.from_thalia_config(minimal_config)

        # Different input patterns
        patterns = [
            torch.zeros(minimal_config.brain.sizes.input_size, device=brain.device),
            torch.ones(minimal_config.brain.sizes.input_size, device=brain.device),
            torch.randn(minimal_config.brain.sizes.input_size, device=brain.device),
        ]

        for pattern in patterns:
            result = brain.forward({"thalamus": pattern}, n_timesteps=5)
            assert "outputs" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
