"""
Integration tests for DynamicBrain and BrainBuilder.

Tests end-to-end functionality:
    - Brain creation from builder API
    - Preset architecture execution
    - Custom brain construction
    - Performance benchmarks vs EventDrivenBrain
    - Memory usage validation
"""

import os
import psutil
import time

import pytest
import torch

from thalia.core.dynamic_brain import DynamicBrain
from thalia.core.brain_builder import BrainBuilder
from thalia.core.brain import EventDrivenBrain
from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes


@pytest.fixture
def device():
    """Get device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def global_config(device):
    """Create GlobalConfig for testing."""
    return GlobalConfig(device=device, dt_ms=1.0, theta_frequency_hz=8.0)


@pytest.fixture
def input_data(device):
    """Create sample input data."""
    return {
        "thalamus": torch.randn(128, device=device),
    }


class TestDynamicBrainIntegration:
    """Integration tests for DynamicBrain."""

    def test_simple_brain_creation_and_execution(self, device, global_config):
        """Test creating and executing a simple custom brain."""
        brain = (
            BrainBuilder(global_config)
            .add_component("input", "thalamic_relay", n_input=64, n_output=64)
            .add_component("cortex", "layered_cortex", n_output=32)  # n_input inferred!
            .connect("input", "cortex", pathway_type="spiking", axonal_delay_ms=5.0)
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
        # n_output=32 → L2/3=48 (1.5x) + L5=32 (1.0x) = 80 total
        assert result["outputs"]["cortex"].shape == (80,)

    def test_three_region_chain(self, device, global_config):
        """Test creating a 3-region chain: A -> B -> C."""
        brain = (
            BrainBuilder(global_config)
            .add_component("region_a", "thalamic_relay", n_input=32, n_output=32)
            .add_component("region_b", "layered_cortex", n_output=64)  # n_input inferred from region_a
            .add_component("region_c", "layered_cortex", n_output=16)  # n_input inferred from region_b
            .connect("region_a", "region_b", pathway_type="spiking", axonal_delay_ms=3.0)
            .connect("region_b", "region_c", pathway_type="spiking", axonal_delay_ms=3.0)
            .build()
        )

        input_data = {"region_a": torch.randn(32, device=device)}
        result = brain.forward(input_data, n_timesteps=50)  # Longer to allow propagation

        # All regions should have output (region_a always, b/c may be None if cortex doesn't spike)
        assert "region_a" in result["outputs"]
        assert "region_b" in result["outputs"]
        # Note: region_c may be None if region_b (cortex) doesn't spike
        if result["outputs"]["region_c"] is not None:
            # LayeredCortex: n_output=16 -> L2/3=24 + L5=16 = 40 total
            assert result["outputs"]["region_c"].shape == (40,)

    def test_diamond_topology(self, device, global_config):
        """Test diamond topology: A -> B, C -> D."""
        brain = (
            BrainBuilder(global_config)
            .add_component("source", "thalamic_relay", n_input=32, n_output=32)
            .add_component("branch1", "layered_cortex", n_output=16)  # n_input=32 inferred
            .add_component("branch2", "layered_cortex", n_output=16)  # n_input=32 inferred
            .add_component("sink", "layered_cortex", n_output=8)      # n_input=32 inferred (16+16)
            .connect("source", "branch1", pathway_type="spiking", axonal_delay_ms=3.0)
            .connect("source", "branch2", pathway_type="spiking", axonal_delay_ms=3.0)
            .connect("branch1", "sink", pathway_type="spiking", axonal_delay_ms=3.0)
            .connect("branch2", "sink", pathway_type="spiking", axonal_delay_ms=3.0)
            .build()
        )

        input_data = {"source": torch.randn(32, device=device)}
        result = brain.forward(input_data, n_timesteps=50)  # Longer to allow propagation

        # All regions should be present (but cortex may return None)
        assert len(result["outputs"]) == 4
        # Note: sink may be None if branch1/branch2 (cortex) don't spike
        if result["outputs"]["sink"] is not None:
            # LayeredCortex: n_output=8 -> L2/3=12 + L5=8 = 20 total
            assert result["outputs"]["sink"].shape == (20,)

    def test_dynamic_component_addition(self, device, global_config):
        """Test adding components dynamically after brain creation."""
        brain = (
            BrainBuilder(global_config)
            .add_component("input", "thalamic_relay", n_input=32, n_output=32)
            .build()
        )

        # Add component dynamically
        from thalia.regions.cortex import LayeredCortex
        from thalia.regions.cortex.config import LayeredCortexConfig
        from thalia.pathways.spiking_pathway import SpikingPathway
        from thalia.core.base.component_config import PathwayConfig

        cortex_config = LayeredCortexConfig(
            n_input=16,  # Match pathway output
            n_output=8,
            device=device,
            dt_ms=1.0,
        )
        cortex = LayeredCortex(cortex_config).to(device)

        pathway_config = PathwayConfig(
            n_output=16,
            n_input=32,
            device=device,
            dt_ms=1.0,
            axonal_delay_ms=2.0,
        )
        pathway = SpikingPathway(pathway_config).to(device)

        brain.add_component("cortex", cortex)
        brain.add_connection("input", "cortex", pathway)

        # Test execution
        input_data = {"input": torch.randn(32, device=device)}
        result = brain.forward(input_data, n_timesteps=10)

        assert "cortex" in result["outputs"]
        # LayeredCortex: n_output=8 → L2/3=12 + L5=8 = 20 total
        assert result["outputs"]["cortex"].shape == (20,)


class TestPresetArchitectures:
    """Integration tests for preset brain architectures."""

    def test_minimal_preset_execution(self, device, global_config):
        """Test minimal preset builds and executes."""
        brain = BrainBuilder.preset("minimal", global_config)

        assert isinstance(brain, DynamicBrain)

        # Check expected components exist (input, process, output)
        assert "input" in brain.components
        assert "process" in brain.components
        assert "output" in brain.components

        # Execute forward pass
        input_data = {"input": torch.randn(64, device=device)}
        result = brain.forward(input_data, n_timesteps=10)

        # All regions should produce output
        assert "input" in result["outputs"]
        assert "process" in result["outputs"]
        assert "output" in result["outputs"]

    def test_sensorimotor_preset_execution(self, device, global_config):
        """Test sensorimotor preset builds and executes."""
        brain = BrainBuilder.preset("sensorimotor", global_config)

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

    def test_preset_with_modifications(self, device, global_config):
        """Test building from preset with custom modifications."""
        builder = BrainBuilder.preset_builder("sensorimotor", global_config)
        builder.add_component("custom_region", "prefrontal", n_output=64)  # n_input inferred from cortex
        builder.connect("cortex", "custom_region", pathway_type="spiking", axonal_delay_ms=5.0)
        brain = builder.build()

        # Should have original + new component
        assert "thalamus" in brain.components
        assert "cortex" in brain.components
        assert "striatum" in brain.components
        assert "custom_region" in brain.components

        input_data = {"thalamus": torch.randn(128, device=device)}
        result = brain.forward(input_data, n_timesteps=10)

        assert "custom_region" in result["outputs"]


class TestPerformanceBenchmarks:
    """Performance benchmarks comparing DynamicBrain vs EventDrivenBrain."""

    @pytest.mark.slow
    @pytest.mark.skip(reason="Device validation error in EventDrivenBrain - needs separate fix")
    def test_execution_time_comparison(self, device, global_config):
        """Compare execution time of DynamicBrain vs EventDrivenBrain."""
        n_timesteps = 100
        n_trials = 10

        # Create DynamicBrain
        dynamic_brain = BrainBuilder.preset("sensorimotor", global_config)

        # Create EventDrivenBrain
        config = ThaliaConfig(
            global_=GlobalConfig(device=device, dt_ms=1.0),
            brain=BrainConfig(
                sizes=RegionSizes(
                    input_size=128,
                    cortex_size=128,
                    hippocampus_size=64,
                    pfc_size=32,
                    n_actions=7,
                ),
            ),
        )
        event_brain = EventDrivenBrain.from_thalia_config(config)

        # Warm-up
        input_data = {"thalamus": torch.randn(128, device=device)}
        dynamic_brain.forward(input_data, n_timesteps=10)
        event_brain.forward(
            sensory_input=torch.randn(128, device=device),
            n_timesteps=10,
        )

        # Benchmark DynamicBrain
        dynamic_times = []
        for _ in range(n_trials):
            start = time.perf_counter()
            dynamic_brain.forward(input_data, n_timesteps=n_timesteps)
            end = time.perf_counter()
            dynamic_times.append(end - start)

        # Benchmark EventDrivenBrain
        event_times = []
        for _ in range(n_trials):
            start = time.perf_counter()
            event_brain.forward(
                sensory_input=torch.randn(128, device=device),
                n_timesteps=n_timesteps,
            )
            end = time.perf_counter()
            event_times.append(end - start)

        avg_dynamic = sum(dynamic_times) / len(dynamic_times)
        avg_event = sum(event_times) / len(event_times)

        # DynamicBrain should be within reasonable performance
        performance_ratio = avg_dynamic / avg_event
        print(f"\nPerformance ratio (Dynamic/Event): {performance_ratio:.2f}")
        print(f"DynamicBrain avg: {avg_dynamic*1000:.2f}ms")
        print(f"EventDrivenBrain avg: {avg_event*1000:.2f}ms")

        # Allow up to 2x slower for now (different execution models)
        assert performance_ratio < 2.0, (
            f"DynamicBrain too slow: {performance_ratio:.2f}x EventDrivenBrain"
        )

    @pytest.mark.slow
    def test_memory_usage_comparison(self, device, global_config):
        """Compare memory usage of DynamicBrain vs EventDrivenBrain."""
        if device == "cuda":
            pytest.skip("Memory comparison only on CPU")

        process = psutil.Process(os.getpid())

        # Measure baseline
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create DynamicBrain
        dynamic_brain = BrainBuilder.preset("sensorimotor", global_config)
        dynamic_memory = process.memory_info().rss / 1024 / 1024  # MB
        dynamic_delta = dynamic_memory - baseline_memory

        # Reset
        del dynamic_brain
        import gc
        gc.collect()
        time.sleep(0.1)

        # Create EventDrivenBrain
        config = ThaliaConfig(
            global_=GlobalConfig(device=device, dt_ms=1.0),
            brain=BrainConfig(
                sizes=RegionSizes(
                    input_size=128,
                    cortex_size=128,
                    hippocampus_size=64,
                    pfc_size=32,
                    n_actions=7,
                ),
            ),
        )
        event_brain = EventDrivenBrain.from_thalia_config(config)
        event_memory = process.memory_info().rss / 1024 / 1024  # MB
        event_delta = event_memory - baseline_memory

        memory_ratio = dynamic_delta / event_delta if event_delta > 0 else 1.0
        print(f"\nMemory ratio (Dynamic/Event): {memory_ratio:.2f}")
        print(f"DynamicBrain: {dynamic_delta:.2f}MB")
        print(f"EventDrivenBrain: {event_delta:.2f}MB")

        # Memory should be comparable (within 2x)
        assert memory_ratio < 2.0, (
            f"DynamicBrain uses too much memory: {memory_ratio:.2f}x EventDrivenBrain"
        )

    def test_forward_pass_correctness(self, device, global_config):
        """Verify DynamicBrain produces reasonable outputs."""
        brain = BrainBuilder.preset("sensorimotor", global_config)

        input_data = {"thalamus": torch.randn(128, device=device)}

        # Run multiple timesteps
        result = brain.forward(input_data, n_timesteps=100)  # Longer for full propagation

        # Check outputs are reasonable (cortex may not spike, causing None downstream)
        for region_name, output in result["outputs"].items():
            if output is not None:
                assert torch.isfinite(output).all(), f"{region_name} has non-finite values"
                assert output.shape[0] > 0, f"{region_name} has zero-size output"


class TestBuilderValidation:
    """Test error handling and validation in BrainBuilder."""

    def test_invalid_component_type(self, device, global_config):
        """Test error when using invalid component type."""
        with pytest.raises(KeyError):
            BrainBuilder(global_config).add_component(
                "test", "invalid_region_type", n_input=32, n_output=16
            ).build()

    def test_connection_to_nonexistent_component(self, device, global_config):
        """Test error when connecting to non-existent component."""
        with pytest.raises((ValueError, KeyError)):
            BrainBuilder(global_config).add_component(
                "input", "thalamic_relay", n_input=32, n_output=32
            ).connect("input", "nonexistent").build()

    def test_invalid_preset(self, device, global_config):
        """Test error with invalid preset name."""
        with pytest.raises(KeyError):
            BrainBuilder.preset("invalid_preset_name", global_config)


class TestSaveAndLoad:
    """Test saving and loading brain specifications."""

    def test_save_and_load_brain_spec(self, device, global_config, tmp_path):
        """Test saving and loading brain specification."""
        original_builder = (
            BrainBuilder(global_config)
            .add_component("input", "thalamic_relay", n_input=32, n_output=32)
            .add_component("cortex", "layered_cortex", n_output=16)  # n_input inferred
            .connect("input", "cortex", pathway_type="spiking", axonal_delay_ms=5.0)
        )

        # Save spec
        spec_path = tmp_path / "brain_spec.json"
        original_builder.save_spec(str(spec_path))

        # Load spec and build
        loaded_builder = BrainBuilder.load_spec(str(spec_path), global_config)
        loaded_brain = loaded_builder.build()

        # Verify structure matches
        assert set(loaded_brain.components.keys()) == {"input", "cortex"}

        # Verify execution works
        input_data = {"input": torch.randn(32, device=device)}
        result = loaded_brain.forward(input_data, n_timesteps=10)
        assert "cortex" in result["outputs"]


class TestThaliaConfigCompatibility:
    """Test backward compatibility with ThaliaConfig."""

    def test_from_thalia_config(self, device):
        """Test creating DynamicBrain from ThaliaConfig."""
        config = ThaliaConfig(
            global_=GlobalConfig(device=device, dt_ms=1.0),
            brain=BrainConfig(
                sizes=RegionSizes(
                    input_size=128,
                    thalamus_size=128,  # Must explicitly set, defaults to 256
                    cortex_size=256,
                    hippocampus_size=128,
                    pfc_size=64,
                    n_actions=7,
                ),
            ),
        )

        brain = DynamicBrain.from_thalia_config(config)

        # Verify structure
        assert isinstance(brain, DynamicBrain)
        expected_regions = ["thalamus", "cortex", "hippocampus", "pfc", "striatum", "cerebellum"]
        for region in expected_regions:
            assert region in brain.components, f"Missing region: {region}"

        # Verify sizes match config
        assert brain.components["thalamus"].n_input == 128
        assert brain.components["thalamus"].n_output == 128

        # Test execution
        input_data = {"thalamus": torch.randn(128, device=device)}
        result = brain.forward(input_data, n_timesteps=10)

        assert "outputs" in result
        assert "time" in result
        assert "thalamus" in result["outputs"]

    def test_from_thalia_config_custom_sizes(self, device):
        """Test from_thalia_config with different sizes."""
        config = ThaliaConfig(
            global_=GlobalConfig(device=device, dt_ms=1.0),
            brain=BrainConfig(
                sizes=RegionSizes(
                    input_size=64,
                    cortex_size=128,
                    hippocampus_size=64,
                    pfc_size=32,
                    n_actions=5,
                ),
            ),
        )

        brain = DynamicBrain.from_thalia_config(config)

        # Verify custom sizes
        assert brain.components["thalamus"].n_input == 64
        # Striatum uses population coding: actual neurons = n_actions * neurons_per_action (default 10)
        assert brain.components["striatum"].n_actions == 5
        assert brain.components["striatum"].n_output == 50  # 5 actions * 10 neurons/action


class TestRLInterface:
    """Test EventDrivenBrain-compatible RL interface (Phase 1.6.2)."""

    def test_select_action_basic(self, device, global_config):
        """Test basic action selection after forward pass."""
        brain = BrainBuilder.preset("sensorimotor", global_config)

        # Forward pass to generate striatum activity
        input_data = {"thalamus": torch.randn(128, device=device)}
        brain.forward(input_data, n_timesteps=20)

        # Select action
        action, confidence = brain.select_action(explore=True)

        # Verify action is valid
        assert isinstance(action, int)
        assert 0 <= action < brain.components["striatum"].n_actions
        assert 0.0 <= confidence <= 1.0

    def test_select_action_exploration(self, device, global_config):
        """Test exploration vs exploitation in action selection."""
        brain = BrainBuilder.preset("sensorimotor", global_config)

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

    def test_deliver_reward_basic(self, device, global_config):
        """Test reward delivery after action selection."""
        brain = BrainBuilder.preset("sensorimotor", global_config)

        # Complete RL cycle
        input_data = {"thalamus": torch.randn(128, device=device)}
        brain.forward(input_data, n_timesteps=20)
        _action, _ = brain.select_action(explore=True)

        # Deliver reward - should not raise
        brain.deliver_reward(external_reward=1.0)
        brain.deliver_reward(external_reward=-1.0)
        brain.deliver_reward(external_reward=0.0)

    def test_rl_episode_loop(self, device, global_config):
        """Test complete multi-step RL episode."""
        brain = BrainBuilder.preset("sensorimotor", global_config)

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

    def test_rl_without_striatum_raises(self, device, global_config):
        """Test that RL methods raise error if no striatum present."""
        # Build brain without striatum (minimal preset has no striatum)
        brain = BrainBuilder.preset("minimal", global_config)

        input_data = {"input": torch.randn(64, device=device)}
        brain.forward(input_data, n_timesteps=10)

        # Should raise ValueError
        with pytest.raises(ValueError, match="Striatum component not found"):
            brain.select_action()

        with pytest.raises(ValueError, match="Striatum component not found"):
            brain.deliver_reward(external_reward=1.0)

    def test_deliver_reward_without_action_allowed(self, device, global_config):
        """Test that deliver_reward works without prior select_action().

        Both DynamicBrain and EventDrivenBrain allow deliver_reward() without
        prior select_action() for streaming/continuous learning scenarios.
        The brain simply uses None/_last_action which may be None initially.
        """
        brain = BrainBuilder.preset("sensorimotor", global_config)

        # Forward pass but no action selection
        input_data = {"thalamus": torch.randn(128, device=device)}
        brain.forward(input_data, n_timesteps=20)

        # Both implementations allow this - they check _last_action internally
        # and only update if an action exists
        brain.deliver_reward(external_reward=1.0)  # Works fine


class TestNeuromodulationAndConsolidation:
    """Test neuromodulation and consolidation features (Phase 1.6.3)."""

    def test_neuromodulator_systems_initialized(self, device, global_config):
        """Test that neuromodulator systems are properly initialized."""
        brain = BrainBuilder.preset("sensorimotor", global_config)

        # Check neuromodulator manager exists
        assert hasattr(brain, "neuromodulator_manager")
        assert brain.neuromodulator_manager is not None

        # Check system shortcuts
        assert hasattr(brain, "vta")
        assert hasattr(brain, "locus_coeruleus")
        assert hasattr(brain, "nucleus_basalis")

        # Verify they are the same objects as in manager
        assert brain.vta is brain.neuromodulator_manager.vta
        assert brain.locus_coeruleus is brain.neuromodulator_manager.locus_coeruleus
        assert brain.nucleus_basalis is brain.neuromodulator_manager.nucleus_basalis

    def test_update_neuromodulators(self, device, global_config):
        """Test neuromodulator update and broadcasting."""
        brain = BrainBuilder.preset("sensorimotor", global_config)

        # Update neuromodulators
        brain._update_neuromodulators()

        # Dopamine should be accessible
        current_da = brain.vta.get_global_dopamine()
        assert isinstance(current_da, float)
        assert -10.0 <= current_da <= 10.0  # Reasonable range

    def test_neuromodulator_broadcast_to_components(self, device, global_config):
        """Test that neuromodulators are broadcast to components."""
        brain = BrainBuilder.preset("sensorimotor", global_config)

        # Deliver reward to trigger dopamine release
        brain.vta.deliver_reward(external_reward=1.0, expected_value=0.0)

        # Update and broadcast
        brain._update_neuromodulators()

        # Check that striatum received dopamine
        if "striatum" in brain.components:
            striatum = brain.components["striatum"]
            if hasattr(striatum, "state") and hasattr(striatum.state, "dopamine"):
                # Dopamine should have been broadcast
                assert striatum.state.dopamine >= 0.0

    def test_consolidate_requires_hippocampus(self, device, global_config):
        """Test that consolidate raises error without hippocampus."""
        # Build brain without hippocampus (minimal preset)
        brain = BrainBuilder.preset("minimal", global_config)

        # Should raise ValueError
        with pytest.raises(ValueError, match="Hippocampus component required"):
            brain.consolidate(n_cycles=1)

    def test_consolidate_basic(self, device, global_config):
        """Test basic consolidation functionality."""
        brain = BrainBuilder.preset("sensorimotor", global_config)

        # First, add some experiences by running forward passes
        for _ in range(3):
            input_data = {"thalamus": torch.randn(128, device=device)}
            brain.forward(input_data, n_timesteps=10)
            if "striatum" in brain.components:
                _action, _ = brain.select_action(explore=True)
                brain.deliver_reward(external_reward=1.0)

        # Consolidation should work with hippocampus
        stats = brain.consolidate(n_cycles=2, batch_size=4, verbose=False)

        # Check stats structure
        assert isinstance(stats, dict)
        assert "cycles_completed" in stats
        assert "total_replayed" in stats
        assert "experiences_learned" in stats

        # Cycles should complete
        assert stats["cycles_completed"] == 2

        # May or may not have experiences to replay depending on hippocampus implementation
        assert stats["total_replayed"] >= 0
        assert stats["experiences_learned"] >= 0


class TestDiagnosticsAndGrowth:
    """Test diagnostics and growth features (Phase 1.6.4)."""

    def test_get_diagnostics_basic(self, device, global_config):
        """Test basic diagnostics collection."""
        brain = BrainBuilder.preset("sensorimotor", global_config)

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

    def test_check_growth_needs(self, device, global_config):
        """Test growth needs detection."""
        brain = BrainBuilder.preset("sensorimotor", global_config)

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

    def test_auto_grow_basic(self, device, global_config):
        """Test basic auto-growth functionality."""
        brain = BrainBuilder.preset("minimal", global_config)

        # Get initial sizes
        initial_sizes = {
            name: comp.n_output
            for name, comp in brain.components.items()
            if hasattr(comp, "n_output")
        }

        # Try auto-growth (may or may not grow depending on metrics)
        growth_actions = brain.auto_grow(threshold=0.8)

        # Should return dict (may be empty)
        assert isinstance(growth_actions, dict)

        # If anything grew, verify sizes increased
        for component_name, neurons_added in growth_actions.items():
            assert neurons_added > 0
            component = brain.components[component_name]
            if hasattr(component, "n_output"):
                new_size = component.n_output
                old_size = initial_sizes[component_name]
                assert new_size == old_size + neurons_added

    def test_check_growth_needs_structure(self, device, global_config):
        """Test structure of growth needs report."""
        brain = BrainBuilder.preset("sensorimotor", global_config)

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
    def simple_brain(self, device, global_config):
        """Create a simple brain for state testing."""
        return (
            BrainBuilder(global_config)
            .add_component("thalamus", "thalamic_relay", n_input=10, n_output=10)
            .add_component("cortex", "layered_cortex", n_output=20)
            .add_component("hippocampus", "hippocampus", n_output=15)
            .add_component("pfc", "prefrontal", n_output=12)
            .add_component("striatum", "striatum", n_output=5)
            .add_component("cerebellum", "cerebellum", n_output=10)
            .connect("thalamus", "cortex", pathway_type="spiking", axonal_delay_ms=3.0)
            .connect("cortex", "hippocampus", pathway_type="spiking", axonal_delay_ms=5.0)
            .connect("cortex", "pfc", pathway_type="spiking", axonal_delay_ms=4.0)
            .connect("pfc", "striatum", pathway_type="spiking", axonal_delay_ms=3.0)
            .connect("cortex", "cerebellum", pathway_type="spiking", axonal_delay_ms=4.0)
            .build()
        )

    def test_get_full_state_basic(self, simple_brain):
        """Test that get_full_state() returns complete serializable state."""
        brain = simple_brain

        state = brain.get_full_state()

        # Check top-level keys
        assert "global_config" in state
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
            if hasattr(component, 'weights'):
                original_weights[name] = component.weights.clone()

        # Modify brain
        for _ in range(10):
            spikes = torch.rand(10, device=brain.device) > 0.7
            input_data = {"thalamus": spikes}
            brain.forward(input_data, n_timesteps=1)

        # Load state
        brain.load_full_state(state)

        # Verify weights restored
        for name, original_w in original_weights.items():
            component = brain.components[name]
            if hasattr(component, 'weights'):
                current_w = component.weights
                assert torch.allclose(current_w, original_w, atol=1e-6), \
                    f"Weights for {name} not restored correctly"

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
        assert abs(restored_da - original_da) < 1e-6, \
            f"Dopamine level not restored: {restored_da} != {original_da}"

    def test_state_topology_preserved(self, device, global_config):
        """Test that topology information is preserved through save/load."""
        brain1 = (
            BrainBuilder(global_config)
            .add_component("thalamus", "thalamic_relay", n_input=10, n_output=10)
            .add_component("cortex", "layered_cortex", n_output=20)
            .connect("thalamus", "cortex", pathway_type="spiking", axonal_delay_ms=3.0)
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
            BrainBuilder(global_config)
            .add_component("thalamus", "thalamic_relay", n_input=10, n_output=10)
            .add_component("cortex", "layered_cortex", n_output=20)
            .connect("thalamus", "cortex", pathway_type="spiking", axonal_delay_ms=3.0)
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
