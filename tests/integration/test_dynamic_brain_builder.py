"""
Integration tests for DynamicBrain and BrainBuilder.

Tests end-to-end functionality:
    - Brain creation from builder API
    - Preset architecture execution
    - Custom brain construction
    - Memory usage validation
"""

import pytest
import torch

from thalia.config import (
    BrainConfig,
    LayeredCortexConfig,
    LayerSizeCalculator,
)
from thalia.core.brain_builder import BrainBuilder
from thalia.core.dynamic_brain import DynamicBrain
from thalia.pathways.axonal_projection import AxonalProjection
from thalia.regions import LayeredCortex


@pytest.fixture
def device():
    """Get device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def brain_config(device):
    """Create BrainConfig for testing."""
    return BrainConfig(device=device, dt_ms=1.0, theta_frequency_hz=8.0)


@pytest.fixture
def input_data(device):
    """Create sample input data."""
    return {
        "thalamus": torch.randn(128, device=device),
    }


class TestDynamicBrainIntegration:
    """Integration tests for DynamicBrain."""

    def test_simple_brain_creation_and_execution(self, device, brain_config):
        """Test creating and executing a simple custom brain."""
        brain = (
            BrainBuilder(brain_config)
            .add_component("input", "thalamic_relay", input_size=64, relay_size=64, trn_size=0)
            .add_component(
                "cortex", "layered_cortex", **LayerSizeCalculator().cortex_from_output(32)
            )  # n_input inferred!
            .connect("input", "cortex", source_port="relay", target_port="feedforward", pathway_type="axonal_projection", axonal_delay_ms=5.0)
            .build()
        )

        assert isinstance(brain, DynamicBrain)
        assert "input" in brain.components
        assert "cortex" in brain.components

        # Execute forward pass
        input_data = {"input": torch.randn(64, device=device)}
        result = brain.forward(input_data, n_timesteps=10)

        assert "cortex" in result["outputs"]
        # LayeredCortex outputs both L2/3 and L5 layers concatenated
        # With L2/3:L5 = 2:1 ratio (biological default):
        # n_output=32 → L2/3=21 (32*2/3) + L5=10 (32*1/3) = 31 total
        assert result["outputs"]["cortex"].shape == (31,)

    def test_three_region_chain(self, device, brain_config):
        """Test creating a 3-region chain: A -> B -> C."""
        brain = (
            BrainBuilder(brain_config)
            .add_component("region_a", "thalamic_relay", input_size=32, relay_size=32, trn_size=0)
            .add_component(
                "region_b", "layered_cortex", **LayerSizeCalculator().cortex_from_output(64)
            )  # n_input inferred from region_a
            .add_component(
                "region_c", "layered_cortex", **LayerSizeCalculator().cortex_from_output(16)
            )  # n_input inferred from region_b
            .connect("region_a", "region_b", source_port="relay", target_port="feedforward", pathway_type="axonal_projection", axonal_delay_ms=3.0)
            .connect("region_b", "region_c", source_port="l23", target_port="feedforward", pathway_type="axonal_projection", axonal_delay_ms=3.0)
            .build()
        )

        input_data = {"region_a": torch.randn(32, device=device)}
        result = brain.forward(input_data, n_timesteps=50)  # Longer to allow propagation

        # All regions should have output (region_a always, b/c may be None if cortex doesn't spike)
        assert "region_a" in result["outputs"]
        assert "region_b" in result["outputs"]
        # Note: region_c may be None if region_b (cortex) doesn't spike
        if result["outputs"]["region_c"] is not None:
            # LayeredCortex: n_output=16 → L2/3=10 (16*2/3) + L5=5 (16*1/3) = 15 total
            assert result["outputs"]["region_c"].shape == (15,)

    def test_diamond_topology(self, device, brain_config):
        """Test diamond topology: A -> B, C -> D."""
        brain = (
            BrainBuilder(brain_config)
            .add_component("source", "thalamic_relay", input_size=32, relay_size=32, trn_size=0)
            .add_component(
                "branch1", "layered_cortex", **LayerSizeCalculator().cortex_from_output(16)
            )  # n_input=32 inferred
            .add_component(
                "branch2", "layered_cortex", **LayerSizeCalculator().cortex_from_output(16)
            )  # n_input=32 inferred
            .add_component(
                "sink", "layered_cortex", **LayerSizeCalculator().cortex_from_output(8)
            )  # n_input=32 inferred (16+16)
            .connect("source", "branch1", source_port="relay", target_port="feedforward", pathway_type="axonal_projection", axonal_delay_ms=3.0)
            .connect("source", "branch2", source_port="relay", target_port="feedforward", pathway_type="axonal_projection", axonal_delay_ms=3.0)
            .connect("branch1", "sink", source_port="l23", target_port="feedforward", pathway_type="axonal_projection", axonal_delay_ms=3.0)
            .connect("branch2", "sink", source_port="l5", target_port="feedforward", pathway_type="axonal_projection", axonal_delay_ms=3.0)
            .build()
        )

        input_data = {"source": torch.randn(32, device=device)}
        result = brain.forward(input_data, n_timesteps=50)  # Longer to allow propagation

        # All regions should be present (but cortex may return None)
        assert len(result["outputs"]) == 4
        # Note: sink may be None if branch1/branch2 (cortex) don't spike
        if result["outputs"]["sink"] is not None:
            # LayeredCortex: n_output=8 → L2/3=5 (8*2/3) + L5=2 (8*1/3) = 7 total
            assert result["outputs"]["sink"].shape == (7,)

    def test_dynamic_component_addition(self, device, brain_config):
        """Test adding components dynamically after brain creation."""
        brain = (
            BrainBuilder(brain_config)
            .add_component("input", "thalamic_relay", input_size=32, relay_size=32, trn_size=0)
            .build()
        )

        # Add component dynamically
        cortex_sizes = {
            "input_size": 32,
            "l4_size": 4,
            "l23_size": 6,
            "l5_size": 4,
            "l6a_size": 1,
            "l6b_size": 1,
        }
        cortex_config = LayeredCortexConfig(dt_ms=1.0)
        cortex = LayeredCortex(config=cortex_config, sizes=cortex_sizes, device=device).to(device)

        brain.add_component("cortex", cortex)

        # Create AxonalProjection from input to cortex
        pathway = AxonalProjection(
            sources=[("input", None, 32, 2.0)],  # (region_name, port, size, delay_ms)
            device=device,
        )
        brain.add_connection(
            source="input",
            target="cortex",
            pathway=pathway,
        )

        # Test execution
        input_data = {"input": torch.randn(32, device=device)}
        result = brain.forward(input_data, n_timesteps=10)

        assert "cortex" in result["outputs"]
        # LayeredCortex: n_output=10 → L2/3=6 + L5=4 = 10 total
        assert result["outputs"]["cortex"].shape == (10,)


class TestPresetArchitectures:
    """Integration tests for preset brain architectures."""

    def test_default_preset_execution(self, device, brain_config):
        """Test default preset builds and executes."""
        brain = BrainBuilder.preset("default", brain_config)

        assert isinstance(brain, DynamicBrain)

        # Check all 6 regions exist
        expected_regions = [
            "thalamus",
            "cortex",
            "hippocampus",
            "pfc",
            "striatum",
            "cerebellum",
        ]
        for region_name in expected_regions:
            assert region_name in brain.components, f"Missing region: {region_name}"

        # Execute forward pass
        input_data = {"thalamus": torch.randn(128, device=device)}
        result = brain.forward(input_data, n_timesteps=10)

        # All regions should produce output
        for region_name in expected_regions:
            assert region_name in result["outputs"], f"Missing output from: {region_name}"

    def test_preset_with_modifications(self, device, brain_config):
        """Test building from preset with custom modifications."""
        builder = BrainBuilder.preset_builder("default", brain_config)
        builder.add_component(
            "custom_region", "prefrontal", input_size=256, n_neurons=64
        )  # Custom PFC region
        builder.connect(
            "cortex", "custom_region", source_port="l23", target_port="feedforward", pathway_type="axonal_projection", axonal_delay_ms=5.0
        )
        brain = builder.build()

        # Should have original + new component
        assert "thalamus" in brain.components
        assert "cortex" in brain.components
        assert "striatum" in brain.components
        assert "custom_region" in brain.components

        input_data = {"thalamus": torch.randn(128, device=device)}
        result = brain.forward(input_data, n_timesteps=10)

        assert "custom_region" in result["outputs"]


class TestBuilderValidation:
    """Test error handling and validation in BrainBuilder."""

    def test_invalid_component_type(self, device, brain_config):
        """Test error when using invalid component type."""
        with pytest.raises(KeyError):
            BrainBuilder(brain_config).add_component(
                "test", "invalid_region_type", n_input=32, n_output=16
            ).build()

    def test_connection_to_nonexistent_component(self, device, brain_config):
        """Test error when connecting to non-existent component."""
        with pytest.raises((ValueError, KeyError)):
            BrainBuilder(brain_config).add_component(
                "input", "thalamic_relay", n_input=32, n_output=32
            ).connect("input", "nonexistent", source_port=None, target_port=None).build()

    def test_invalid_preset(self, device, brain_config):
        """Test error with invalid preset name."""
        with pytest.raises(KeyError):
            BrainBuilder.preset("invalid_preset_name", brain_config)


class TestSaveAndLoad:
    """Test saving and loading brain specifications."""

    def test_save_and_load_brain_spec(self, device, brain_config, tmp_path):
        """Test saving and loading brain specification."""
        original_builder = (
            BrainBuilder(brain_config)
            .add_component("input", "thalamic_relay", input_size=32, relay_size=32, trn_size=0)
            .add_component(
                "cortex", "layered_cortex", **LayerSizeCalculator().cortex_from_output(16)
            )  # n_input inferred
            .connect("input", "cortex", source_port="relay", target_port="feedforward", pathway_type="axonal_projection", axonal_delay_ms=5.0)
        )

        # Save spec
        spec_path = tmp_path / "brain_spec.json"
        original_builder.save_spec(str(spec_path))

        # Load spec and build
        loaded_builder = BrainBuilder.load_spec(str(spec_path), brain_config)
        loaded_brain = loaded_builder.build()

        # Verify structure matches
        assert set(loaded_brain.components.keys()) == {"input", "cortex"}

        # Verify execution works
        input_data = {"input": torch.randn(32, device=device)}
        result = loaded_brain.forward(input_data, n_timesteps=10)
        assert "cortex" in result["outputs"]


class TestRLInterface:
    """Test DynamicBrain RL interface."""

    def test_select_action_basic(self, device, brain_config):
        """Test basic action selection after forward pass."""
        brain = BrainBuilder.preset("default", brain_config)

        # Forward pass to generate striatum activity
        input_data = {"thalamus": torch.randn(128, device=device)}
        brain.forward(input_data, n_timesteps=20)

        # Select action
        action, confidence = brain.select_action(explore=True)

        # Verify action is valid
        assert isinstance(action, int)
        assert 0 <= action < brain.components["striatum"].n_actions
        assert 0.0 <= confidence <= 1.0

    def test_select_action_exploration(self, device, brain_config):
        """Test exploration vs exploitation in action selection."""
        brain = BrainBuilder.preset("default", brain_config)

        input_data = {"thalamus": torch.randn(128, device=device)}

        # Run multiple trials
        exploratory_actions = []
        exploit_actions = []

        for _ in range(5):
            brain.forward(input_data, n_timesteps=10)
            action_explore, _ = brain.select_action(explore=True)
            exploratory_actions.append(action_explore)

            brain.forward(input_data, n_timesteps=10)
            action_exploit, _ = brain.select_action(explore=False)
            exploit_actions.append(action_exploit)

        # Both should produce valid actions
        assert all(0 <= a < brain.components["striatum"].n_actions for a in exploratory_actions)
        assert all(0 <= a < brain.components["striatum"].n_actions for a in exploit_actions)

    def test_deliver_reward_basic(self, device, brain_config):
        """Test reward delivery after action selection."""
        brain = BrainBuilder.preset("default", brain_config)

        # Complete RL cycle
        input_data = {"thalamus": torch.randn(128, device=device)}
        brain.forward(input_data, n_timesteps=20)
        _action, _ = brain.select_action(explore=True)

        # Deliver reward - should not raise
        brain.deliver_reward(external_reward=1.0)
        brain.deliver_reward(external_reward=-1.0)
        brain.deliver_reward(external_reward=0.0)

    def test_rl_episode_loop(self, device, brain_config):
        """Test complete multi-step RL episode."""
        brain = BrainBuilder.preset("default", brain_config)

        n_steps = 5
        actions = []
        rewards = []

        for _ in range(n_steps):
            # Forward pass
            input_data = {"thalamus": torch.randn(128, device=device)}
            result = brain.forward(input_data, n_timesteps=10)
            assert "outputs" in result

            # Action selection
            action, confidence = brain.select_action(explore=True)
            actions.append(action)
            assert 0.0 <= confidence <= 1.0

            # Simulated environment reward
            reward = 1.0 if action % 2 == 0 else -0.5
            rewards.append(reward)

            # Reward delivery
            brain.deliver_reward(external_reward=reward)

        # Verify complete episode
        assert len(actions) == n_steps
        assert len(rewards) == n_steps
        assert all(isinstance(a, int) for a in actions)

    def test_deliver_reward_without_action_allowed(self, device, brain_config):
        """Test that deliver_reward works without prior select_action().

        DynamicBrain allows deliver_reward() without prior
        select_action() for streaming/continuous learning scenarios.
        The brain simply uses None/_last_action which may be None initially.
        """
        brain = BrainBuilder.preset("default", brain_config)

        # Forward pass but no action selection
        input_data = {"thalamus": torch.randn(128, device=device)}
        brain.forward(input_data, n_timesteps=20)

        # Both implementations allow this - they check _last_action internally
        # and only update if an action exists
        brain.deliver_reward(external_reward=1.0)  # Works fine


class TestNeuromodulationAndConsolidation:
    """Test neuromodulation and consolidation features (Phase 1.6.3)."""

    def test_neuromodulator_systems_initialized(self, device, brain_config):
        """Test that neuromodulator systems are properly initialized."""
        brain = BrainBuilder.preset("default", brain_config)

        # Check neuromodulator manager exists
        assert hasattr(brain, "neuromodulator_manager")
        assert brain.neuromodulator_manager is not None

        # Check neuromodulator systems exist in manager
        assert hasattr(brain.neuromodulator_manager, "vta")
        assert hasattr(brain.neuromodulator_manager, "locus_coeruleus")
        assert hasattr(brain.neuromodulator_manager, "nucleus_basalis")

        # Verify they are properly initialized
        assert brain.neuromodulator_manager.vta is not None
        assert brain.neuromodulator_manager.locus_coeruleus is not None
        assert brain.neuromodulator_manager.nucleus_basalis is not None

    def test_update_neuromodulators(self, device, brain_config):
        """Test neuromodulator update and broadcasting."""
        brain = BrainBuilder.preset("default", brain_config)

        # Run a forward pass first to initialize state
        input_data = {"thalamus": torch.randn(128, device=device)}
        brain.forward(input_data, n_timesteps=1)

        # Update neuromodulators
        brain._update_neuromodulators()

        # Dopamine should be accessible
        current_da = brain.neuromodulator_manager.vta.get_global_dopamine()
        assert isinstance(current_da, float)
        assert -10.0 <= current_da <= 10.0  # Reasonable range

    def test_neuromodulator_broadcast_to_components(self, device, brain_config):
        """Test that neuromodulators are broadcast to components."""
        brain = BrainBuilder.preset("default", brain_config)

        # Run a forward pass first to initialize state
        input_data = {"thalamus": torch.randn(128, device=device)}
        brain.forward(input_data, n_timesteps=1)

        # Deliver reward to trigger dopamine release
        brain.neuromodulator_manager.vta.deliver_reward(external_reward=1.0, expected_value=0.0)

        # Update and broadcast
        brain._update_neuromodulators()

        # Check that striatum received dopamine
        if "striatum" in brain.components:
            striatum = brain.components["striatum"]
            if hasattr(striatum, "state") and hasattr(striatum.state, "dopamine"):
                # Dopamine should have been broadcast
                assert striatum.state.dopamine >= 0.0

    def test_consolidate_basic(self, device, brain_config):
        """Test basic consolidation functionality."""
        brain = BrainBuilder.preset("default", brain_config)

        # First, add some experiences by running forward passes
        for _ in range(3):
            input_data = {"thalamus": torch.randn(128, device=device)}
            brain.forward(input_data, n_timesteps=10)
            if "striatum" in brain.components:
                _action, _ = brain.select_action(explore=True)
                brain.deliver_reward(external_reward=1.0)

        # Consolidation should work with hippocampus
        consolidation_duration_ms = 100.0
        stats = brain.consolidate(duration_ms=consolidation_duration_ms, verbose=False)

        # Check stats structure (current implementation returns ripple stats)
        assert isinstance(stats, dict)
        assert "ripples" in stats
        assert "duration_ms" in stats
        assert "ripple_rate_hz" in stats

        # Basic sanity checks
        assert stats["duration_ms"] == consolidation_duration_ms
        assert stats["ripples"] >= 0  # May or may not detect ripples
        assert stats["ripple_rate_hz"] >= 0.0  # Hz can be zero if no ripples


class TestDiagnosticsAndGrowth:
    """Test diagnostics and growth features (Phase 1.6.4)."""

    def test_get_diagnostics_basic(self, device, brain_config):
        """Test basic diagnostics collection."""
        brain = BrainBuilder.preset("default", brain_config)

        # Run forward pass to generate activity
        input_data = {"thalamus": torch.randn(128, device=device)}
        brain.forward(input_data, n_timesteps=10)

        # Get diagnostics
        diagnostics = brain.get_diagnostics()

        # Should have diagnostics for all components
        assert isinstance(diagnostics, dict)
        assert len(diagnostics) > 0
        assert "components" in diagnostics

        # Check that we got diagnostics for all expected components
        expected_components = ["thalamus", "cortex", "hippocampus", "pfc", "striatum", "cerebellum"]
        for component_name in expected_components:
            assert component_name in diagnostics["components"]
            assert isinstance(diagnostics["components"][component_name], dict)

    def test_check_growth_needs(self, device, brain_config):
        """Test growth needs detection."""
        brain = BrainBuilder.preset("default", brain_config)

        # Run some activity to generate metrics
        for _ in range(5):
            input_data = {"thalamus": torch.randn(128, device=device)}
            brain.forward(input_data, n_timesteps=10)

        # Check growth needs
        report = brain.check_growth_needs()

        # Should have report for all components
        assert isinstance(report, dict)
        assert len(report) > 0

        # Each component should have growth metrics
        for _component_name, metrics in report.items():
            assert isinstance(metrics, dict)
            assert "growth_recommended" in metrics
            assert "growth_reason" in metrics
            assert isinstance(metrics["growth_recommended"], bool)

    def test_check_growth_needs_structure(self, device, brain_config):
        """Test structure of growth needs report."""
        brain = BrainBuilder.preset("default", brain_config)

        report = brain.check_growth_needs()

        # Verify report structure for components that support metrics
        for _component_name, metrics in report.items():
            # All components should have these fields
            assert "growth_recommended" in metrics
            assert "growth_reason" in metrics

            # Components with full metrics should have additional fields
            if "firing_rate" in metrics:
                assert "weight_saturation" in metrics
                assert "synapse_usage" in metrics
                assert "neuron_count" in metrics


class TestStateManagement:
    """Test state save/load functionality (Phase 1.6.5)."""

    @pytest.fixture
    def simple_brain(self, device, brain_config):
        """Create a simple brain for state testing."""
        return (
            BrainBuilder(brain_config)
            .add_component("thalamus", "thalamic_relay", input_size=10, relay_size=10, trn_size=0)
            .add_component(
                "cortex", "layered_cortex", **LayerSizeCalculator().cortex_from_output(20)
            )
            .add_component("hippocampus", "hippocampus", n_output=15)
            .add_component("pfc", "prefrontal", input_size=20, n_neurons=12)
            .add_component("striatum", "striatum", n_actions=3, neurons_per_action=2)
            .add_component("cerebellum", "cerebellum", purkinje_size=10)
            .connect("thalamus", "cortex", source_port="relay", target_port="feedforward", pathway_type="axonal_projection", axonal_delay_ms=3.0)
            .connect("cortex", "hippocampus", source_port="l5", target_port="feedforward", pathway_type="axonal_projection", axonal_delay_ms=5.0)
            .connect("cortex", "pfc", source_port="l23", target_port="feedforward", pathway_type="axonal_projection", axonal_delay_ms=4.0)
            .connect("pfc", "striatum", source_port="executive", target_port="feedforward", pathway_type="axonal_projection", axonal_delay_ms=3.0)
            .connect("cortex", "cerebellum", source_port="l5", target_port="feedforward", pathway_type="axonal_projection", axonal_delay_ms=4.0)
            .build()
        )

    def test_get_full_state_basic(self, simple_brain):
        """Test that get_full_state() returns complete serializable state."""
        brain = simple_brain

        state = brain.get_full_state()

        # Check top-level keys
        assert "brain_config" in state
        assert "current_time" in state
        assert "topology" in state
        # DynamicBrain uses "regions" key for CheckpointManager compatibility
        assert "regions" in state
        assert "pathways" in state
        assert "neuromodulators" in state

        # Check components are all present (stored as "regions")
        assert len(state["regions"]) == 6
        assert "thalamus" in state["regions"]
        assert "cortex" in state["regions"]
        assert "hippocampus" in state["regions"]
        assert "pfc" in state["regions"]
        assert "striatum" in state["regions"]
        assert "cerebellum" in state["regions"]

        # Check pathways (DynamicBrain uses "pathways" not "connections")
        assert len(state["pathways"]) >= 2  # At least thal→cortex and cortex→hippo

        # Check neuromodulators
        assert "vta" in state["neuromodulators"]
        assert "locus_coeruleus" in state["neuromodulators"]
        assert "nucleus_basalis" in state["neuromodulators"]

    def test_load_full_state_basic(self, simple_brain):
        """Test that load_full_state() restores brain state correctly."""
        brain = simple_brain

        # Run a few steps to build up state
        for _ in range(5):
            spikes = torch.rand(10, device=brain.device) > 0.8
            input_data = {"thalamus": spikes}
            brain.forward(input_data, n_timesteps=1)

        # Save state
        state1 = brain.get_full_state()

        # Run more steps
        for _ in range(5):
            spikes = torch.rand(10, device=brain.device) > 0.8
            input_data = {"thalamus": spikes}
            brain.forward(input_data, n_timesteps=1)

        # Get different state
        state2 = brain.get_full_state()
        assert state1["current_time"] != state2["current_time"]

        # Load old state
        brain.load_full_state(state1)

        # Verify restoration
        state3 = brain.get_full_state()
        assert state3["current_time"] == state1["current_time"]

    def test_state_fidelity_components(self, simple_brain):
        """Test that component states are faithfully preserved through save/load."""
        brain = simple_brain

        # Run to build up interesting state
        for _ in range(10):
            spikes = torch.rand(10, device=brain.device) > 0.7
            input_data = {"thalamus": spikes}
            brain.forward(input_data, n_timesteps=1)

        # Save state and component weights
        state = brain.get_full_state()
        original_weights = {}
        for name, component in brain.components.items():
            # Save all nn.Parameters, not just 'weights' attribute
            # (hippocampus has multiple weight matrices, and .weights can point to different ones)
            original_weights[name] = {
                param_name: param.data.clone() for param_name, param in component.named_parameters()
            }

        # Modify brain
        for _ in range(10):
            spikes = torch.rand(10, device=brain.device) > 0.7
            input_data = {"thalamus": spikes}
            brain.forward(input_data, n_timesteps=1)

        # Load state
        brain.load_full_state(state)

        # Verify weights restored
        for name, saved_params in original_weights.items():
            component = brain.components[name]
            for param_name, original_param in saved_params.items():
                # Get current parameter value
                current_param = dict(component.named_parameters())[param_name]
                try:
                    assert torch.allclose(
                        current_param, original_param, atol=1e-6
                    ), f"Parameter {name}.{param_name} not restored correctly"
                except RuntimeError as e:
                    raise RuntimeError(
                        f"Parameter {name}.{param_name} has incompatible shapes: "
                        f"current={current_param.shape}, original={original_param.shape}"
                    ) from e

    def test_state_fidelity_neuromodulators(self, simple_brain):
        """Test that neuromodulator states are preserved through save/load."""
        brain = simple_brain

        # Run a few steps to let dopamine evolve
        for _ in range(5):
            spikes = torch.rand(10, device=brain.device) > 0.7
            input_data = {"thalamus": spikes}
            brain.forward(input_data, n_timesteps=1)

        # Save state
        state = brain.get_full_state()
        original_da = state["neuromodulators"]["vta"]["global_dopamine"]

        # Run more steps to change dopamine
        for _ in range(10):
            spikes = torch.rand(10, device=brain.device) > 0.7
            input_data = {"thalamus": spikes}
            brain.forward(input_data, n_timesteps=1)

        state2 = brain.get_full_state()
        # Dopamine might have changed (not guaranteed, but check state is different)
        assert state2["current_time"] != state["current_time"]

        # Load state
        brain.load_full_state(state)

        # Verify dopamine restored
        state3 = brain.get_full_state()
        restored_da = state3["neuromodulators"]["vta"]["global_dopamine"]
        assert (
            abs(restored_da - original_da) < 1e-6
        ), f"Dopamine level not restored: {restored_da} != {original_da}"

    def test_state_topology_preserved(self, device, brain_config):
        """Test that topology information is preserved through save/load."""
        brain1 = (
            BrainBuilder(brain_config)
            .add_component("thalamus", "thalamic_relay", input_size=10, relay_size=10, trn_size=0)
            .add_component(
                "cortex", "layered_cortex", **LayerSizeCalculator().cortex_from_output(20)
            )
            .connect("thalamus", "cortex", source_port="relay", target_port="feedforward", pathway_type="axonal_projection", axonal_delay_ms=3.0)
            .build()
        )

        # Save state
        state = brain1.get_full_state()
        original_topology = state["topology"]

        # Verify topology is meaningful
        assert isinstance(original_topology, dict)
        assert len(original_topology) > 0

        # Load state into new brain
        brain2 = (
            BrainBuilder(brain_config)
            .add_component("thalamus", "thalamic_relay", input_size=10, relay_size=10, trn_size=0)
            .add_component(
                "cortex", "layered_cortex", **LayerSizeCalculator().cortex_from_output(20)
            )
            .connect("thalamus", "cortex", source_port="relay", target_port="feedforward", pathway_type="axonal_projection", axonal_delay_ms=3.0)
            .build()
        )
        brain2.load_full_state(state)

        # Verify topology matches
        state2 = brain2.get_full_state()
        assert state2["topology"] == original_topology

    def test_state_time_preserved(self, simple_brain):
        """Test that simulation time is preserved through save/load."""
        brain = simple_brain

        # Run to specific time
        for _ in range(15):
            input_data = {"thalamus": torch.zeros(10, device=brain.device)}
            brain.forward(input_data, n_timesteps=1)

        # Save state
        state = brain.get_full_state()
        original_time = state["current_time"]
        assert original_time == 15.0

        # Run more
        for _ in range(10):
            input_data = {"thalamus": torch.zeros(10, device=brain.device)}
            brain.forward(input_data, n_timesteps=1)
        assert brain.current_time == 25.0

        # Load state
        brain.load_full_state(state)

        # Verify time restored
        assert brain.current_time == original_time


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
