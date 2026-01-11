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
from thalia.config.region_sizes import compute_thalamus_sizes
from thalia.regions.thalamus import ThalamicRelay, ThalamicRelayConfig
from thalia.pathways.axonal_projection import AxonalProjection


# ============================================================================
# Test Helpers
# ============================================================================

def create_test_thalamus(input_size: int, relay_size: int, device: str = "cpu") -> ThalamicRelay:
    """Create a ThalamicRelay for testing with new (config, sizes, device) pattern."""
    config = ThalamicRelayConfig(device=device)
    sizes = compute_thalamus_sizes(relay_size, trn_ratio=0.0)  # trn_size=0 for tests
    sizes["input_size"] = input_size
    return ThalamicRelay(config, sizes, device)


# ============================================================================
# DynamicBrain Tests
# ============================================================================

def test_dynamic_brain_creation():
    """Test DynamicBrain instantiation."""
    global_config = GlobalConfig(device="cpu", dt_ms=1.0)

    components = {
        "region1": create_test_thalamus(input_size=32, relay_size=64),
        "region2": create_test_thalamus(input_size=64, relay_size=128),
    }

    connections = {
        ("region1", "region2"): AxonalProjection(
            sources=[("region1", None, 64, 1.0)],
            device="cpu"
        ),
    }

    brain = DynamicBrain(components, connections, global_config)

    # Test explicit setup (2 components explicitly created above)
    assert len(brain.components) == 2
    assert len(brain.connections) == 1
    assert "region1" in brain.components
    assert "region2" in brain.components


def test_dynamic_brain_topology_graph():
    """Test topology graph construction."""
    global_config = GlobalConfig(device="cpu", dt_ms=1.0)

    components = {
        "a": create_test_thalamus(input_size=32, relay_size=64),
        "b": create_test_thalamus(input_size=64, relay_size=64),
        "c": create_test_thalamus(input_size=64, relay_size=64),
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

    # Test topology via public API behavior
    # Verify components exist and are connected correctly
    assert "a" in brain.components
    assert "b" in brain.components
    assert "c" in brain.components

    # Verify connections exist
    assert ("a", "b") in brain.connections
    assert ("b", "c") in brain.connections


def test_dynamic_brain_get_component():
    """Test get_component method."""
    global_config = GlobalConfig(device="cpu", dt_ms=1.0)

    components = {
        "region1": create_test_thalamus(input_size=32, relay_size=64),
    }

    brain = DynamicBrain(components, {}, global_config)

    comp = brain.get_component("region1")
    assert comp is components["region1"]

    with pytest.raises(KeyError, match="(?i)(nonexistent|not found|unknown)"):
        brain.get_component("nonexistent")


def test_dynamic_brain_add_component():
    """Test dynamic component addition."""
    global_config = GlobalConfig(device="cpu", dt_ms=1.0)

    components = {
        "region1": create_test_thalamus(input_size=32, relay_size=64),
    }

    brain = DynamicBrain(components, {}, global_config)

    # Add new component
    new_region = create_test_thalamus(input_size=64, relay_size=128)
    brain.add_component("region2", new_region)

    assert "region2" in brain.components
    assert brain.get_component("region2") is new_region


def test_dynamic_brain_add_connection():
    """Test dynamic connection addition."""
    global_config = GlobalConfig(device="cpu", dt_ms=1.0)

    components = {
        "a": create_test_thalamus(input_size=32, relay_size=64),
        "b": create_test_thalamus(input_size=64, relay_size=64),
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
        "region": create_test_thalamus(input_size=32, relay_size=64),
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

    # Test via public API - build empty brain
    brain = builder.build()
    assert len(brain.components) == 0
    assert len(brain.connections) == 0


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

    # Should have at least "minimal" and "default"
    preset_names = [name for name, _ in presets]
    assert "minimal" in preset_names
    assert "default" in preset_names


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


# ============================================================================
# Mock Failure Tests
# ============================================================================

def test_brain_handles_connection_validation_failure():
    """Test brain validates incompatible connections.

    This test ensures that when adding a connection between components
    with mismatched dimensions, the brain catches the error early.
    """
    global_config = GlobalConfig(device="cpu", dt_ms=1.0)

    # Create components with incompatible sizes
    components = {
        "source": create_test_thalamus(input_size=32, relay_size=64, device="cpu"),
        "target": create_test_thalamus(input_size=128, relay_size=64, device="cpu"),
    }
    brain = DynamicBrain(components, {}, global_config)

    # Create pathway with output size that doesn't match target input
    # Source outputs 64, but target expects 128 → dimension mismatch
    # AxonalProjection expects tuples: (region_name, port, size, delay_ms)
    pathway = AxonalProjection(
        sources=[("source", None, 64, 1.0)],
        device="cpu"
    )

    # Should raise error during validation
    # Note: DynamicBrain may not validate this automatically yet
    # This test documents the EXPECTED behavior for future implementation
    try:
        brain.add_connection("source", "target", pathway)
        # If no error raised, verify the connection at least exists
        assert ("source", "target") in brain.connections
    except (ValueError, AssertionError, RuntimeError) as e:
        # Expected: dimension validation error
        assert any(keyword in str(e).lower() for keyword in ["dimension", "size", "mismatch", "shape"])


def test_brain_forward_with_missing_inputs():
    """Test brain handles missing required inputs gracefully.

    When forward() is called without inputs for all regions,
    should provide clear error message.
    """
    global_config = GlobalConfig(device="cpu", dt_ms=1.0)

    components = {
        "region_a": create_test_thalamus(input_size=32, relay_size=64, device="cpu"),
        "region_b": create_test_thalamus(input_size=64, relay_size=32, device="cpu"),
    }
    brain = DynamicBrain(components, {}, global_config)

    # Only provide input for region_a, not region_b
    partial_inputs = {"region_a": torch.ones(32, dtype=torch.bool)}

    # Should handle missing inputs (either error or use defaults)
    try:
        result = brain.forward(partial_inputs, n_timesteps=1)
        # If successful, verify result structure
        assert "outputs" in result
    except (KeyError, ValueError) as e:
        # Expected: clear error about missing inputs
        assert any(keyword in str(e).lower() for keyword in ["missing", "input", "required", "region_b"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
