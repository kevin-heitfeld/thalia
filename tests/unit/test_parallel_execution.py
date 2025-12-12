"""
Tests for parallel execution mode.

Verifies that parallel mode produces equivalent results to sequential mode
and properly handles multiprocessing, tensor serialization, and event ordering.

Author: Thalia Project
Date: December 2025
"""

import pytest
import torch
from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes
from thalia.core.brain import EventDrivenBrain
from thalia.events import Event, EventType, SpikePayload
from thalia.events.parallel import (
    serialize_event,
    deserialize_event,
    create_region_creator,
)


class TestTensorSerialization:
    """Test tensor serialization for multiprocessing."""

    def test_serialize_cpu_tensor(self):
        """CPU tensors should pass through unchanged."""
        spikes = torch.zeros(10)
        event = Event(
            time=1.0,
            event_type=EventType.SPIKE,
            source="test",
            target="test",
            payload=SpikePayload(spikes=spikes),
        )

        serialized = serialize_event(event)
        assert torch.equal(serialized.payload.spikes, spikes)
        assert not serialized.payload.spikes.is_cuda

    def test_serialize_gpu_tensor(self):
        """GPU tensors should be moved to CPU for pickling."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        spikes = torch.zeros(10, device="cuda")
        event = Event(
            time=1.0,
            event_type=EventType.SPIKE,
            source="test",
            target="test",
            payload=SpikePayload(spikes=spikes),
        )

        serialized = serialize_event(event)
        assert not serialized.payload.spikes.is_cuda
        assert serialized.payload.spikes.device.type == "cpu"

    def test_deserialize_to_cpu(self):
        """Deserialize should respect target device."""
        spikes = torch.zeros(10)
        event = Event(
            time=1.0,
            event_type=EventType.SPIKE,
            source="test",
            target="test",
            payload=SpikePayload(spikes=spikes),
        )

        deserialized = deserialize_event(event, "cpu")
        assert deserialized.payload.spikes.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_deserialize_to_gpu(self):
        """Deserialize should move tensors to GPU if requested."""
        spikes = torch.zeros(10)
        event = Event(
            time=1.0,
            event_type=EventType.SPIKE,
            source="test",
            target="test",
            payload=SpikePayload(spikes=spikes),
        )

        deserialized = deserialize_event(event, "cuda")
        assert deserialized.payload.spikes.is_cuda

    def test_serialize_preserves_metadata(self):
        """Serialization should preserve event metadata."""
        spikes = torch.ones(10)
        event = Event(
            time=42.0,
            event_type=EventType.SENSORY,
            source="sensory",
            target="cortex",
            payload=SpikePayload(spikes=spikes, source_layer="L4"),
        )

        serialized = serialize_event(event)
        assert serialized.time == 42.0
        assert serialized.event_type == EventType.SENSORY
        assert serialized.source == "sensory"
        assert serialized.target == "cortex"
        assert serialized.payload.source_layer == "L4"


class TestRegionCreators:
    """Test module-level region creator functions."""

    def test_create_cortex_creator(self):
        """Should create cortex region creator."""
        config = {
            "name": "cortex",
            "n_input": 784,
            "n_layers": 3,
            "layer_sizes": [256, 256, 256],
            "output_targets": ["hippocampus"],
        }

        creator = create_region_creator("cortex", config, "cpu")
        region = creator()

        assert region.name == "cortex"
        assert hasattr(region, "process_event")
        assert hasattr(region, "impl")

    def test_create_hippocampus_creator(self):
        """Should create hippocampus region creator."""
        config = {
            "name": "hippocampus",
            "n_input": 256,
            "dg_size": 400,
            "ca3_size": 300,
            "ca1_size": 200,
            "output_targets": ["pfc"],
        }

        creator = create_region_creator("hippocampus", config, "cpu")
        region = creator()

        assert region.name == "hippocampus"
        assert hasattr(region, "process_event")

    def test_create_pfc_creator(self):
        """Should create PFC region creator."""
        config = {
            "name": "pfc",
            "n_neurons": 128,
            "n_input": 256,
            "output_targets": ["striatum"],
        }

        creator = create_region_creator("pfc", config, "cpu")
        region = creator()

        assert region.name == "pfc"
        assert hasattr(region, "process_event")

    def test_create_striatum_creator(self):
        """Should create striatum region creator."""
        config = {
            "name": "striatum",
            "n_actions": 10,
            "n_input": 512,
            "n_context": 200,
            "output_targets": [],
        }

        creator = create_region_creator("striatum", config, "cpu")
        region = creator()

        assert region.name == "striatum"
        assert hasattr(region, "process_event")

    def test_invalid_region_type(self):
        """Should raise error for invalid region type."""
        with pytest.raises(ValueError, match="Unknown region type"):
            create_region_creator("invalid_region", {}, "cpu")


class TestParallelExecution:
    """Test parallel execution produces equivalent results to sequential."""

    @pytest.fixture
    def sequential_config(self):
        """Config for sequential mode."""
        return ThaliaConfig(
            global_=GlobalConfig(
                device="cpu",  # CPU required for fair comparison
                dt_ms=1.0,
            ),
            brain=BrainConfig(
                sizes=RegionSizes(
                    input_size=784,
                    thalamus_size=784,  # Must match input_size for proper relay
                    cortex_size=128,
                    hippocampus_size=100,
                    pfc_size=64,
                    n_actions=4,
                ),
                parallel=False,  # Sequential mode
                encoding_timesteps=10,
            ),
        )

    @pytest.fixture
    def parallel_config(self):
        """Config for parallel mode."""
        return ThaliaConfig(
            global_=GlobalConfig(
                device="cpu",  # CPU required for multiprocessing
                dt_ms=1.0,
            ),
            brain=BrainConfig(
                sizes=RegionSizes(
                    input_size=784,
                    thalamus_size=784,  # Must match input_size for proper relay
                    cortex_size=128,
                    hippocampus_size=100,
                    pfc_size=64,
                    n_actions=4,
                ),
                parallel=True,  # Parallel mode
                encoding_timesteps=10,
            ),
        )

    def test_parallel_initialization(self, parallel_config):
        """Parallel mode should initialize without errors."""
        brain = EventDrivenBrain.from_thalia_config(parallel_config)
        assert brain._parallel_executor is not None
        assert brain._parallel_executor.device == "cpu"

    def test_sequential_initialization(self, sequential_config):
        """Sequential mode should not create parallel executor."""
        brain = EventDrivenBrain.from_thalia_config(sequential_config)
        assert brain._parallel_executor is None

    def test_parallel_forward_pass(self, parallel_config):
        """Parallel mode should execute forward pass without errors."""
        brain = EventDrivenBrain.from_thalia_config(parallel_config)

        # Create input pattern
        input_pattern = torch.rand(784) > 0.5
        input_pattern = input_pattern.float()

        # Execute forward pass
        result = brain.forward(input_pattern, n_timesteps=5)

        assert "spike_counts" in result
        assert "events_processed" in result
        assert "final_time" in result

    def test_parallel_vs_sequential_spike_counts(self, sequential_config, parallel_config):
        """Parallel and sequential modes should produce similar spike patterns.

        Note: Exact equivalence not expected due to timing variations in
        multiprocessing, but spike counts should be in similar range.
        """
        # Create same input for both
        torch.manual_seed(42)
        input_pattern = torch.rand(784) > 0.5
        input_pattern = input_pattern.float()

        # Sequential execution
        brain_seq = EventDrivenBrain.from_thalia_config(sequential_config)
        result_seq = brain_seq.forward(input_pattern.clone(), n_timesteps=5)

        # Parallel execution
        brain_par = EventDrivenBrain.from_thalia_config(parallel_config)
        result_par = brain_par.forward(input_pattern.clone(), n_timesteps=5)

        # Check spike counts are in similar range (within 50% for stochastic systems)
        for region in ["cortex", "hippocampus", "pfc", "striatum"]:
            if region in result_seq["spike_counts"] and region in result_par["spike_counts"]:
                seq_count = result_seq["spike_counts"][region]
                par_count = result_par["spike_counts"][region]

                # Allow significant variation due to async processing
                if seq_count > 0:
                    ratio = par_count / seq_count
                    assert 0.3 < ratio < 3.0, (
                        f"Spike counts too different for {region}: "
                        f"seq={seq_count}, par={par_count}"
                    )

    def test_parallel_action_selection(self, parallel_config):
        """Parallel mode should support action selection."""
        brain = EventDrivenBrain.from_thalia_config(parallel_config)

        # Process some input first
        input_pattern = torch.rand(784) > 0.5
        input_pattern = input_pattern.float()
        brain.forward(input_pattern, n_timesteps=5)

        # Select action
        action, confidence = brain.select_action(explore=True)

        assert isinstance(action, int)
        assert 0 <= action < 4
        assert 0.0 <= confidence <= 1.0

    def test_parallel_cleanup(self, parallel_config):
        """Parallel executor should cleanup worker processes."""
        brain = EventDrivenBrain.from_thalia_config(parallel_config)
        executor = brain._parallel_executor

        # Verify processes started
        assert len(executor.processes) > 0
        for process in executor.processes.values():
            assert process.is_alive()

        # Cleanup
        brain.__del__()

        # Give processes time to terminate
        import time
        time.sleep(0.5)

        # Verify processes terminated
        for process in executor.processes.values():
            assert not process.is_alive()


class TestParallelEdgeCases:
    """Test edge cases and error handling in parallel mode."""

    def test_parallel_with_no_input(self):
        """Parallel mode should handle None input (maintenance)."""
        config = ThaliaConfig(
            global_=GlobalConfig(device="cpu", dt_ms=1.0),
            brain=BrainConfig(
                sizes=RegionSizes(
                    input_size=784,
                    cortex_size=128,
                    hippocampus_size=100,
                    pfc_size=64,
                    n_actions=4,
                ),
                parallel=True,
            ),
        )

        brain = EventDrivenBrain.from_thalia_config(config)

        # Forward with no input (maintenance mode)
        result = brain.forward(None, n_timesteps=5)

        assert "spike_counts" in result

    def test_parallel_multiple_forward_passes(self):
        """Parallel mode should handle multiple sequential forward passes."""
        config = ThaliaConfig(
            global_=GlobalConfig(device="cpu", dt_ms=1.0),
            brain=BrainConfig(
                sizes=RegionSizes(
                    input_size=784,
                    cortex_size=128,
                    hippocampus_size=100,
                    pfc_size=64,
                    n_actions=4,
                ),
                parallel=True,
            ),
        )

        brain = EventDrivenBrain.from_thalia_config(config)
        input_pattern = torch.rand(784) > 0.5
        input_pattern = input_pattern.float()

        # Multiple forward passes
        for _ in range(3):
            result = brain.forward(input_pattern, n_timesteps=3)
            assert "spike_counts" in result

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_parallel_gpu_warning(self):
        """Parallel mode with GPU config should force CPU device."""
        config = ThaliaConfig(
            global_=GlobalConfig(device="cuda"),  # GPU requested
            brain=BrainConfig(
                sizes=RegionSizes(
                    input_size=784,
                    cortex_size=128,
                    hippocampus_size=100,
                    pfc_size=64,
                    n_actions=4,
                ),
                parallel=True,
            ),
        )

        brain = EventDrivenBrain.from_thalia_config(config)

        # Parallel executor should use CPU despite GPU config
        assert brain._parallel_executor.device == "cpu"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
