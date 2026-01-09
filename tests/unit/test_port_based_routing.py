"""
Unit tests for port-based routing in DynamicBrain.

This test suite defines the expected behavior for layer-specific routing,
multiple input types, and biologically accurate connectivity patterns.

Author: Thalia Project
Date: December 15, 2025
"""

import pytest
import torch

from thalia.core.brain_builder import BrainBuilder, ConnectionSpec
from thalia.config import GlobalConfig


@pytest.fixture
def device():
    """Get device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def global_config(device):
    """Create minimal GlobalConfig."""
    return GlobalConfig(device=device, dt_ms=1.0)


class TestConnectionSpecPorts:
    """Test ConnectionSpec with source and target ports."""

    def test_connection_spec_with_ports(self):
        """Test creating ConnectionSpec with source and target ports."""
        spec = ConnectionSpec(
            source="cortex",
            target="hippocampus",
            source_port="l23",
            target_port="feedforward",
            pathway_type="axonal_projection",
        )

        assert spec.source == "cortex"
        assert spec.target == "hippocampus"
        assert spec.source_port == "l23"
        assert spec.target_port == "feedforward"
        assert spec.pathway_type == "axonal_projection"

    def test_connection_spec_without_ports_defaults_to_none(self):
        """Test that ports default to None for backward compatibility."""
        spec = ConnectionSpec(
            source="thalamus",
            target="cortex",
            pathway_type="axonal_projection",
        )

        assert spec.source_port is None
        assert spec.target_port is None

    def test_connection_spec_with_only_source_port(self):
        """Test specifying only source port."""
        spec = ConnectionSpec(
            source="cortex",
            target="striatum",
            source_port="l5",
            pathway_type="axonal_projection",
        )

        assert spec.source_port == "l5"
        assert spec.target_port is None

    def test_connection_spec_with_only_target_port(self):
        """Test specifying only target port."""
        spec = ConnectionSpec(
            source="pfc",
            target="cortex",
            target_port="top_down",
            pathway_type="axonal_projection",
        )

        assert spec.source_port is None
        assert spec.target_port == "top_down"


class TestBrainBuilderPortBasedConnections:
    """Test BrainBuilder with port-based connections."""

    def test_connect_with_source_port(self, global_config):
        """Test connecting with source port specification."""
        builder = BrainBuilder(global_config)

        # Add components (cortex needs all layer sizes and input_size specified)
        builder.add_component("cortex", "cortex", input_size=64, l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0)
        builder.add_component("hippocampus", "hippocampus", input_size=64, dg_size=128, ca3_size=96, ca2_size=32, ca1_size=64)

        # Connect using L2/3 output specifically
        builder.connect(
            "cortex", "hippocampus",
            source_port="l23",
            pathway_type="axonal_projection"
        )

        # Build and verify connection exists with correct routing
        brain = builder.build()
        assert len(brain.connections) == 1
        assert ("cortex", "hippocampus") in brain.connections

        # Verify hippocampus receives input from cortex
        pathway = brain.connections[("cortex", "hippocampus")]
        assert hasattr(pathway, 'forward')  # Valid pathway exists

    def test_connect_with_target_port(self, global_config):
        """Test connecting with target port specification."""
        builder = BrainBuilder(global_config)

        builder.add_component("pfc", "prefrontal", input_size=64, wm_size=32)
        builder.add_component("cortex", "cortex", input_size=64, l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0)

        # Connect to top_down input specifically
        builder.connect(
            "pfc", "cortex",
            target_port="top_down",
            pathway_type="axonal_projection"
        )

        conn = builder._connections[0]
        assert conn.target_port == "top_down"

    def test_connect_with_both_ports(self, global_config):
        """Test connecting with both source and target ports."""
        builder = BrainBuilder(global_config)

        builder.add_component("cortex", "cortex", input_size=64, l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0)
        builder.add_component("striatum", "striatum", n_actions=4, neurons_per_action=1, input_sources={"default": 32})

        builder.connect(
            "cortex", "striatum",
            source_port="l5",
            target_port="cortical_input",
            pathway_type="axonal_projection"
        )

        conn = builder._connections[0]
        assert conn.source_port == "l5"
        assert conn.target_port == "cortical_input"

    def test_multiple_connections_to_same_target_different_ports(self, global_config):
        """Test multiple connections to same target using different ports."""
        builder = BrainBuilder(global_config)

        builder.add_component("thalamus", "thalamus", n_input=64, n_output=64)
        builder.add_component("pfc", "prefrontal", n_input=64, n_output=32)
        builder.add_component("cortex", "cortex", n_output=128, l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0)

        # Feedforward path
        builder.connect(
            "thalamus", "cortex",
            target_port="feedforward",
            pathway_type="axonal_projection"
        )

        # Top-down path
        builder.connect(
            "pfc", "cortex",
            target_port="top_down",
            pathway_type="axonal_projection"
        )

        # Build and verify connections exist
        brain = builder.build()
        assert len(brain.connections) == 2

        # When ports are specified, connection keys include port suffix
        # Check that both pathways exist (may be keyed with port suffix)
        connection_keys = list(brain.connections.keys())
        assert len(connection_keys) == 2

        # Verify connections exist (with or without port suffix)
        thalamus_cortex_exists = any(
            'thalamus' in str(k[0]) and 'cortex' in str(k[1])
            for k in connection_keys
        )
        pfc_cortex_exists = any(
            'pfc' in str(k[0]) and 'cortex' in str(k[1])
            for k in connection_keys
        )
        assert thalamus_cortex_exists, f"Thalamus->Cortex connection not found in {connection_keys}"
        assert pfc_cortex_exists, f"PFC->Cortex connection not found in {connection_keys}"


class TestLayerSpecificCorticalRouting:
    """Test layer-specific cortical output routing (L2/3 vs L5)."""

    def test_cortex_l23_to_hippocampus(self, global_config):
        """Test that cortex L2/3 output routes to hippocampus."""
        builder = BrainBuilder(global_config)

        builder.add_component("thalamus", "thalamus", n_input=64, n_output=64)
        builder.add_component("cortex", "cortex", n_output=128, l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0)
        builder.add_component("hippocampus", "hippocampus", n_output=64)

        builder.connect("thalamus", "cortex", pathway_type="axonal_projection")
        builder.connect("cortex", "hippocampus", source_port="l23", pathway_type="axonal_projection")

        brain = builder.build()

        # Verify cortex component exists and has layer structure
        cortex = brain.components["cortex"]
        assert hasattr(cortex, 'l23_size')
        assert hasattr(cortex, 'l5_size')

        # Verify hippocampus receives only L2/3 size
        hippo = brain.components["hippocampus"]
        # Should infer input_size from cortex L2/3 output, not full output
        assert hippo.config.input_size == cortex.l23_size

    def test_cortex_l5_to_striatum(self, global_config):
        """Test that cortex L5 output routes to striatum."""
        builder = BrainBuilder(global_config)

        builder.add_component("thalamus", "thalamus", input_size=64, relay_size=64, trn_size=0)
        builder.add_component("cortex", "cortex", input_size=64, l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0)
        builder.add_component("striatum", "striatum", n_actions=4, neurons_per_action=1, input_sources={"default": 32})

        builder.connect("thalamus", "cortex", pathway_type="axonal_projection")
        builder.connect("cortex", "striatum", source_port="l5", pathway_type="axonal_projection")

        brain = builder.build()

        cortex = brain.components["cortex"]
        striatum = brain.components["striatum"]

        # Striatum should receive only L5 size
        assert striatum.config.total_input == cortex.l5_size

    def test_cortex_outputs_to_multiple_targets_with_different_layers(self, global_config):
        """Test cortex routing L2/3 to one target and L5 to another."""
        builder = BrainBuilder(global_config)

        builder.add_component("thalamus", "thalamus", n_input=64, n_output=64)
        builder.add_component("cortex", "cortex", n_output=128, l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0)
        builder.add_component("hippocampus", "hippocampus", n_output=64)
        builder.add_component("striatum", "striatum", n_output=4)

        builder.connect("thalamus", "cortex")
        builder.connect("cortex", "hippocampus", source_port="l23")  # Cortico-cortical
        builder.connect("cortex", "striatum", source_port="l5")      # Cortico-subcortical

        brain = builder.build()

        cortex = brain.components["cortex"]
        hippo = brain.components["hippocampus"]
        striatum = brain.components["striatum"]

        # Each target receives appropriate layer size
        assert hippo.config.input_size == cortex.l23_size
        assert striatum.config.total_input == cortex.l5_size


class TestMultipleInputPorts:
    """Test components with multiple named input ports."""

    def test_cortex_feedforward_and_topdown_inputs(self, global_config):
        """Test cortex receiving both feedforward and top-down inputs."""
        builder = BrainBuilder(global_config)

        builder.add_component("thalamus", "thalamus", n_input=64, n_output=64)
        builder.add_component("pfc", "prefrontal", n_input=64, n_output=32)
        builder.add_component("cortex", "cortex", n_output=128, l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0)

        # Feedforward from thalamus
        builder.connect("thalamus", "cortex", target_port="feedforward")

        # Top-down from PFC
        builder.connect("pfc", "cortex", target_port="top_down")

        brain = builder.build()

        cortex = brain.components["cortex"]
        thalamus = brain.components["thalamus"]

        # Cortex input_size should only count feedforward, not top_down
        # (top_down is separate parameter in forward())
        assert cortex.config.input_size == thalamus.relay_size  # Dimension compatibility

    def test_hippocampus_cortical_and_entorhinal_inputs(self, global_config):
        """Test hippocampus receiving both cortical and direct entorhinal inputs."""
        builder = BrainBuilder(global_config)

        builder.add_component("thalamus", "thalamus", n_input=64, n_output=64)
        builder.add_component("cortex", "cortex", n_output=128, l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0)
        builder.add_component("hippocampus", "hippocampus", n_output=64)

        # Main cortical input (L2/3)
        builder.connect("thalamus", "cortex")
        builder.connect("cortex", "hippocampus", source_port="l23", target_port="cortical")

        # Direct entorhinal input
        builder.connect("thalamus", "hippocampus", target_port="ec_l3")

        brain = builder.build()

        cortex = brain.components["cortex"]
        hippo = brain.components["hippocampus"]

        # Hippocampus n_input is cortical input size
        assert hippo.config.n_input == cortex.l23_size

        # ec_l3_input_size should be set separately
        assert hippo.config.ec_l3_input_size == 64

    def test_striatum_multiple_input_sources(self, global_config):
        """Test striatum receiving inputs from cortex, hippocampus, and PFC."""
        builder = BrainBuilder(global_config)

        builder.add_component("thalamus", "thalamus", n_input=64, n_output=64)
        builder.add_component("cortex", "cortex", n_output=128, l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0)
        builder.add_component("hippocampus", "hippocampus", n_output=64)
        builder.add_component("pfc", "prefrontal", n_output=32)
        builder.add_component("striatum", "striatum", n_output=4)

        # Standard connections
        builder.connect("thalamus", "cortex")
        builder.connect("cortex", "hippocampus", source_port="l23")
        builder.connect("cortex", "pfc", source_port="l23")  # PFC needs input

        # Striatum inputs from multiple sources
        builder.connect("cortex", "striatum", source_port="l5", target_port="cortical")
        builder.connect("hippocampus", "striatum", target_port="hippocampal")
        builder.connect("pfc", "striatum", target_port="pfc_modulation")

        brain = builder.build()

        cortex = brain.components["cortex"]
        hippo = brain.components["hippocampus"]
        pfc = brain.components["pfc"]
        striatum = brain.components["striatum"]

        # Striatum n_input should sum cortical and hippocampal
        # (pfc_modulation is separate for goal conditioning)
        expected_input = cortex.l5_size + hippo.config.n_output
        assert striatum.config.n_input == expected_input


class TestPortBasedForwardPass:
    """Test forward pass with port-based routing."""

    def test_forward_routes_correct_layer_outputs(self, global_config):
        """Test that forward pass routes layer-specific outputs correctly."""
        builder = BrainBuilder(global_config)

        builder.add_component("thalamus", "thalamus", n_input=64, n_output=64)
        builder.add_component("cortex", "cortex", n_output=128, l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0)
        builder.add_component("hippocampus", "hippocampus", n_output=64)
        builder.add_component("striatum", "striatum", n_output=4)

        builder.connect("thalamus", "cortex")
        builder.connect("cortex", "hippocampus", source_port="l23")
        builder.connect("cortex", "striatum", source_port="l5")

        brain = builder.build()

        # Create input
        input_data = {
            "thalamus": torch.randn(64, device=brain.device)
        }

        # Run forward pass
        result = brain.forward(input_data, n_timesteps=5)

        # Verify outputs exist for all components
        assert "thalamus" in result["outputs"]
        assert "cortex" in result["outputs"]
        assert "hippocampus" in result["outputs"]
        assert "striatum" in result["outputs"]

        # Verify cortex produced layered output
        cortex_output = result["outputs"]["cortex"]
        if cortex_output is not None:
            cortex = brain.components["cortex"]
            # Cortex should output concatenated L2/3 + L5
            assert cortex_output.shape[0] == cortex.l23_size + cortex.l5_size


class TestBackwardCompatibility:
    """Test that port-based routing maintains backward compatibility."""

    def test_connections_without_ports_still_work(self, global_config):
        """Test that existing code without ports continues to work."""
        builder = BrainBuilder(global_config)

        builder.add_component("thalamus", "thalamus", n_input=64, n_output=64)
        builder.add_component("cortex", "cortex", n_output=128, l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0)

        # Old-style connection without ports
        builder.connect("thalamus", "cortex", pathway_type="axonal_projection")

        brain = builder.build()

        # Should build successfully
        assert "thalamus" in brain.components
        assert "cortex" in brain.components

        # Should infer sizes correctly (dimension compatibility)
        cortex = brain.components["cortex"]
        thalamus = brain.components["thalamus"]
        assert cortex.config.n_input == thalamus.n_output

    def test_mixed_ports_and_no_ports_connections(self, global_config):
        """Test mixing port-based and traditional connections."""
        builder = BrainBuilder(global_config)

        builder.add_component("thalamus", "thalamus", n_input=64, n_output=64)
        builder.add_component("cortex", "cortex", n_output=128, l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0)
        builder.add_component("hippocampus", "hippocampus", n_output=64)
        builder.add_component("striatum", "striatum", n_output=4)

        # Mix of old and new style
        builder.connect("thalamus", "cortex")  # Old style
        builder.connect("cortex", "hippocampus", source_port="l23")  # New style
        builder.connect("cortex", "striatum", source_port="l5")  # New style

        brain = builder.build()

        # Should build successfully
        assert len(brain.components) == 4
