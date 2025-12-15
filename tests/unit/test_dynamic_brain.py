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
from thalia.regions.base import NeuralComponent
from thalia.core.base.component_config import NeuralComponentConfig


# Mock components for testing
class MockComponentConfig(NeuralComponentConfig):
    """Mock configuration for test components."""
    n_neurons: int = 100


class MockRegion(NeuralComponent):
    """Mock region for testing."""

    def __init__(self, config: MockComponentConfig, device: str = "cpu"):
        super().__init__()
        self.n_neurons = config.n_neurons
        self.device = torch.device(device)
        self._last_input = None
        self._last_output = None

    def forward(self, input_data: torch.Tensor | None) -> torch.Tensor:
        """Forward pass."""
        if input_data is None:
            # Source node - generate output
            output = torch.ones(self.n_neurons, device=self.device)
        else:
            # Process input
            output = input_data + 1.0

        self._last_input = input_data
        self._last_output = output
        return output

    def reset_state(self) -> None:
        """Reset internal state."""
        self._last_input = None
        self._last_output = None

    def get_diagnostics(self) -> dict:
        """Get diagnostics."""
        return {
            "n_neurons": self.n_neurons,
            "last_input_shape": self._last_input.shape if self._last_input is not None else None,
            "last_output_shape": self._last_output.shape if self._last_output is not None else None,
        }

    def get_full_state(self) -> dict:
        """Get full state."""
        return {
            "n_neurons": self.n_neurons,
            "last_input": self._last_input,
            "last_output": self._last_output,
        }

    def load_full_state(self, state: dict) -> None:
        """Load full state."""
        self.n_neurons = state["n_neurons"]
        self._last_input = state["last_input"]
        self._last_output = state["last_output"]


class MockPathway(NeuralComponent):
    """Mock pathway for testing."""

    def __init__(self, config: MockComponentConfig, device: str = "cpu"):
        super().__init__()
        self.n_neurons = config.n_neurons  # Not really used, just for interface
        self.device = torch.device(device)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Forward pass - simple identity with small transform."""
        return input_data * 2.0

    def reset_state(self) -> None:
        """Reset state."""
        pass

    def get_diagnostics(self) -> dict:
        """Get diagnostics."""
        return {}

    def get_full_state(self) -> dict:
        """Get full state."""
        return {}

    def load_full_state(self, state: dict) -> None:
        """Load full state."""
        pass


# ============================================================================
# DynamicBrain Tests
# ============================================================================

def test_dynamic_brain_creation():
    """Test DynamicBrain instantiation."""
    global_config = GlobalConfig(device="cpu", dt_ms=1.0)

    components = {
        "region1": MockRegion(MockComponentConfig(n_neurons=64)),
        "region2": MockRegion(MockComponentConfig(n_neurons=128)),
    }

    connections = {
        ("region1", "region2"): MockPathway(MockComponentConfig(n_neurons=64)),
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
        "a": MockRegion(MockComponentConfig(n_neurons=64)),
        "b": MockRegion(MockComponentConfig(n_neurons=64)),
        "c": MockRegion(MockComponentConfig(n_neurons=64)),
    }

    connections = {
        ("a", "b"): MockPathway(MockComponentConfig(n_neurons=64)),
        ("b", "c"): MockPathway(MockComponentConfig(n_neurons=64)),
    }

    brain = DynamicBrain(components, connections, global_config)

    # Check topology
    assert "b" in brain._topology["a"]
    assert "c" in brain._topology["b"]
    assert len(brain._topology["c"]) == 0


def test_dynamic_brain_topological_sort():
    """Test topological sorting of components."""
    global_config = GlobalConfig(device="cpu", dt_ms=1.0)

    components = {
        "a": MockRegion(MockComponentConfig(n_neurons=64)),
        "b": MockRegion(MockComponentConfig(n_neurons=64)),
        "c": MockRegion(MockComponentConfig(n_neurons=64)),
    }

    connections = {
        ("a", "b"): MockPathway(MockComponentConfig(n_neurons=64)),
        ("b", "c"): MockPathway(MockComponentConfig(n_neurons=64)),
    }

    brain = DynamicBrain(components, connections, global_config)
    order = brain._get_execution_order()

    # Order should be: a, b, c
    assert order.index("a") < order.index("b")
    assert order.index("b") < order.index("c")


def test_dynamic_brain_cycle_detection():
    """Test cycle detection in component graph."""
    global_config = GlobalConfig(device="cpu", dt_ms=1.0)

    components = {
        "a": MockRegion(MockComponentConfig(n_neurons=64)),
        "b": MockRegion(MockComponentConfig(n_neurons=64)),
    }

    connections = {
        ("a", "b"): MockPathway(MockComponentConfig(n_neurons=64)),
        ("b", "a"): MockPathway(MockComponentConfig(n_neurons=64)),  # Cycle!
    }

    brain = DynamicBrain(components, connections, global_config)

    with pytest.raises(ValueError, match="contains cycles"):
        brain._get_execution_order()


def test_dynamic_brain_forward():
    """Test forward pass execution."""
    global_config = GlobalConfig(device="cpu", dt_ms=1.0)

    components = {
        "input": MockRegion(MockComponentConfig(n_neurons=64)),
        "output": MockRegion(MockComponentConfig(n_neurons=64)),
    }

    connections = {
        ("input", "output"): MockPathway(MockComponentConfig(n_neurons=64)),
    }

    brain = DynamicBrain(components, connections, global_config)

    # Execute forward
    input_data = {"input": torch.ones(64)}
    result = brain.forward(input_data, n_timesteps=1)

    assert "outputs" in result
    assert "input" in result["outputs"]
    assert "output" in result["outputs"]
    assert result["outputs"]["input"] is not None
    assert result["outputs"]["output"] is not None


def test_dynamic_brain_get_component():
    """Test get_component method."""
    global_config = GlobalConfig(device="cpu", dt_ms=1.0)

    components = {
        "region1": MockRegion(MockComponentConfig(n_neurons=64)),
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
        "region1": MockRegion(MockComponentConfig(n_neurons=64)),
    }

    brain = DynamicBrain(components, {}, global_config)

    # Add new component
    new_region = MockRegion(MockComponentConfig(n_neurons=128))
    brain.add_component("region2", new_region)

    assert "region2" in brain.components
    assert brain.get_component("region2") is new_region


def test_dynamic_brain_add_connection():
    """Test dynamic connection addition."""
    global_config = GlobalConfig(device="cpu", dt_ms=1.0)

    components = {
        "a": MockRegion(MockComponentConfig(n_neurons=64)),
        "b": MockRegion(MockComponentConfig(n_neurons=64)),
    }

    brain = DynamicBrain(components, {}, global_config)

    # Add connection
    pathway = MockPathway(MockComponentConfig(n_neurons=64))
    brain.add_connection("a", "b", pathway)

    assert "a_to_b" in brain.connections


def test_dynamic_brain_reset_state():
    """Test state reset."""
    global_config = GlobalConfig(device="cpu", dt_ms=1.0)

    components = {
        "region": MockRegion(MockComponentConfig(n_neurons=64)),
    }

    brain = DynamicBrain(components, {}, global_config)

    # Execute to set state
    brain.forward({"region": torch.ones(64)}, n_timesteps=1)

    # Reset
    brain.reset_state()

    # State should be cleared
    assert components["region"]._last_input is None


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
