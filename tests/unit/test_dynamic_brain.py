"""
Unit tests for DynamicBrain and BrainBuilder.

Tests:
    - DynamicBrain component graph execution
    - Topological sorting
    - Input gathering and routing
    - BrainBuilder fluent API
    - Preset architectures
    - Component validation
"""

from pathlib import Path
import tempfile
import json
import pytest
import torch

from thalia.core.dynamic_brain import DynamicBrain, ComponentSpec, ConnectionSpec
from thalia.core.brain_builder import BrainBuilder
from thalia.config import GlobalConfig
from thalia.regions.thalamus import ThalamicRelay, ThalamicRelayConfig
from thalia.pathways.axonal_projection import AxonalProjection


# ============================================================================
# DynamicBrain Tests
# ============================================================================

def test_dynamic_brain_creation():
    """Test DynamicBrain instantiation."""
    global_config = GlobalConfig(device="cpu", dt_ms=1.0)

    components = {
        "region1": ThalamicRelay(ThalamicRelayConfig(n_input=32, n_output=64)),
        "region2": ThalamicRelay(ThalamicRelayConfig(n_input=64, n_output=128)),
    }

    connections = {
        ("region1", "region2"): AxonalProjection(
            sources=[("region1", None, 64, 1.0)],
            device="cpu"
        ),
    }

    brain = DynamicBrain(components, connections, global_config)

    assert len(brain.components) == 2
    assert len(brain.connections) == 1
    assert "region1" in brain.components
    assert "region2" in brain.components


def test_dynamic_brain_topology_graph():
    """Test topology graph construction."""
    global_config = GlobalConfig(device="cpu", dt_ms=1.0)

    components = {
        "a": ThalamicRelay(ThalamicRelayConfig(n_input=32, n_output=64)),
        "b": ThalamicRelay(ThalamicRelayConfig(n_input=64, n_output=64)),
        "c": ThalamicRelay(ThalamicRelayConfig(n_input=64, n_output=64)),
    }

    connections = {
        ("a", "b"): AxonalProjection(
            sources=[("a", None, 64, 1.0)],
            device="cpu"
        ),
        ("b", "c"): AxonalProjection(
            sources=[("b", None, 64, 1.0)],
            device="cpu"
        ),
    }

    brain = DynamicBrain(components, connections, global_config)

    # Check topology
    assert "b" in brain._topology["a"]
    assert "c" in brain._topology["b"]
    assert len(brain._topology["c"]) == 0


def test_dynamic_brain_forward():
    """Test forward pass execution."""
    global_config = GlobalConfig(device="cpu", dt_ms=1.0)

    components = {
        "input": ThalamicRelay(ThalamicRelayConfig(n_input=32, n_output=64)),
        "output": ThalamicRelay(ThalamicRelayConfig(n_input=64, n_output=64)),
    }

    connections = {
        ("input", "output"): AxonalProjection(
            sources=[("input", None, 64, 1.0)],
            device="cpu"
        ),
    }

    brain = DynamicBrain(components, connections, global_config)

    # Set mock registry to avoid adapter requirement
    from unittest.mock import Mock
    brain._registry = Mock()

    # Execute forward - just verify it runs without errors
    input_data = {"input": torch.ones(32)}  # Match n_input=32
    result = brain.forward(input_data, n_timesteps=1)

    # Verify basic structure
    assert "outputs" in result
    assert "input" in result["outputs"]
    assert "output" in result["outputs"]


def test_dynamic_brain_get_component():
    """Test get_component method."""
    global_config = GlobalConfig(device="cpu", dt_ms=1.0)

    components = {
        "region1": ThalamicRelay(ThalamicRelayConfig(n_input=32, n_output=64)),
    }

    brain = DynamicBrain(components, {}, global_config)

    comp = brain.get_component("region1")
    assert comp is components["region1"]

    with pytest.raises(KeyError):
        brain.get_component("nonexistent")


def test_dynamic_brain_add_component():
    """Test dynamic component addition."""
    global_config = GlobalConfig(device="cpu", dt_ms=1.0)

    components = {
        "region1": ThalamicRelay(ThalamicRelayConfig(n_input=32, n_output=64)),
    }

    brain = DynamicBrain(components, {}, global_config)

    # Add new component
    new_region = ThalamicRelay(ThalamicRelayConfig(n_input=64, n_output=128))
    brain.add_component("region2", new_region)

    assert "region2" in brain.components
    assert brain.get_component("region2") is new_region


def test_dynamic_brain_add_connection():
    """Test dynamic connection addition."""
    global_config = GlobalConfig(device="cpu", dt_ms=1.0)

    components = {
        "a": ThalamicRelay(ThalamicRelayConfig(n_input=32, n_output=64)),
        "b": ThalamicRelay(ThalamicRelayConfig(n_input=64, n_output=64)),
    }

    brain = DynamicBrain(components, {}, global_config)

    # Add connection
    pathway = AxonalProjection(
        sources=[("a", None, 64, 1.0)],
        device="cpu"
    )
    brain.add_connection("a", "b", pathway)

    # connections dict uses tuple keys, not string keys
    assert ("a", "b") in brain.connections
    assert brain.connections[("a", "b")] is pathway


def test_dynamic_brain_reset_state():
    """Test state reset."""
    global_config = GlobalConfig(device="cpu", dt_ms=1.0)

    components = {
        "region": ThalamicRelay(ThalamicRelayConfig(n_input=32, n_output=64)),
    }

    brain = DynamicBrain(components, {}, global_config)

    # Set mock registry to avoid adapter requirement
    from unittest.mock import Mock
    brain._registry = Mock()

    # Execute to set state
    brain.forward({"region": torch.ones(32)}, n_timesteps=1)  # Match n_input=32

    # Reset - just verify it completes without error
    brain.reset_state()

    # Verify basic functionality after reset
    result = brain.forward({"region": torch.ones(32)}, n_timesteps=1)
    assert "outputs" in result


# ============================================================================
# BrainBuilder Tests
# ============================================================================

def test_brain_builder_creation():
    """Test BrainBuilder instantiation."""
    global_config = GlobalConfig(device="cpu", dt_ms=1.0)
    builder = BrainBuilder(global_config)

    assert builder.global_config is global_config
    assert len(builder._components) == 0
    assert len(builder._connections) == 0


def test_brain_builder_add_component():
    """Test add_component method."""
    global_config = GlobalConfig(device="cpu", dt_ms=1.0)
    builder = BrainBuilder(global_config)

    # Note: This will fail without registered components
    # This test demonstrates the API but won't pass without registry setup
    # TODO: Add mock registry components for testing


def test_brain_builder_connect():
    """Test connect method."""
    global_config = GlobalConfig(device="cpu", dt_ms=1.0)
    builder = BrainBuilder(global_config)

    # Add mock components (would need registry)
    # builder.add_component("a", "mock_region", n_neurons=64)
    # builder.add_component("b", "mock_region", n_neurons=64)
    # builder.connect("a", "b", "mock_pathway")

    # TODO: Complete test with mock registry


def test_brain_builder_validation():
    """Test component graph validation."""
    global_config = GlobalConfig(device="cpu", dt_ms=1.0)
    builder = BrainBuilder(global_config)

    issues = builder.validate()
    assert len(issues) == 0  # Empty graph is valid


def test_brain_builder_save_load_spec():
    """Test save/load component specifications."""
    global_config = GlobalConfig(device="cpu", dt_ms=1.0)
    builder = BrainBuilder(global_config)

    # Mock some specs (without registry)
    builder._components["region1"] = ComponentSpec(
        name="region1",
        component_type="region",
        registry_name="mock_region",
        config_params={"n_neurons": 64},
    )

    builder._connections.append(ConnectionSpec(
        source="region1",
        target="region2",
        pathway_type="mock_pathway",
        config_params={},
    ))

    # Save to temp file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        temp_path = Path(f.name)

    try:
        builder.save_spec(temp_path)

        # Load and verify
        with open(temp_path, "r") as f:
            spec = json.load(f)

        assert len(spec["components"]) == 1
        assert spec["components"][0]["name"] == "region1"
        assert len(spec["connections"]) == 1

    finally:
        temp_path.unlink()


def test_brain_builder_list_presets():
    """Test listing preset architectures."""
    presets = BrainBuilder.list_presets()

    # Should have at least "minimal" and "sensorimotor"
    preset_names = [name for name, _ in presets]
    assert "minimal" in preset_names
    assert "sensorimotor" in preset_names


def test_brain_builder_preset_registration():
    """Test preset architecture registration."""
    def custom_preset(builder: BrainBuilder, **overrides):
        # Mock preset builder
        pass

    BrainBuilder.register_preset(
        name="test_preset",
        description="Test preset for unit tests",
        builder_fn=custom_preset,
    )

    presets = BrainBuilder.list_presets()
    preset_names = [name for name, _ in presets]
    assert "test_preset" in preset_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
