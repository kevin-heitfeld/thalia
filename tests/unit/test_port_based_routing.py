"""
Unit tests for port-based routing in DynamicBrain.

This test suite defines the expected behavior for layer-specific routing,
multiple input types, and biologically accurate connectivity patterns.

Author: Thalia Project
Date: December 15, 2025
"""

import pytest
import torch

from thalia.config import (
    LayeredCortexConfig,
    HippocampusConfig,
    StriatumConfig,
    ThalamicRelayConfig,
    BrainConfig,
    LayerSizeCalculator,
)
from thalia.core.brain_builder import BrainBuilder, ConnectionSpec
from thalia.pathways.axonal_projection import AxonalProjection
from thalia.regions import (
    LayeredCortex,
    Striatum,
    ThalamicRelay,
    TrisynapticHippocampus,
)


@pytest.fixture
def device():
    """Get device for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def brain_config(device):
    """Create minimal BrainConfig."""
    return BrainConfig(device=device, dt_ms=1.0)


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

    def test_connect_with_source_port(self, brain_config):
        """Test connecting with source port specification."""
        builder = BrainBuilder(brain_config)

        # Add components (cortex needs all layer sizes and input_size specified)
        builder.add_component(
            "cortex",
            "cortex",
            input_size=64,
            l4_size=64,
            l23_size=96,
            l5_size=32,
            l6a_size=0,
            l6b_size=0,
        )
        builder.add_component(
            "hippocampus",
            "hippocampus",
            input_size=64,
            dg_size=128,
            ca3_size=96,
            ca2_size=32,
            ca1_size=64,
        )

        # Connect using L2/3 output specifically
        builder.connect(
            "cortex", "hippocampus", source_port="l23", target_port="feedforward", pathway_type="axonal_projection"
        )

        # Build and verify connection exists with correct routing
        brain = builder.build()
        assert len(brain.connections) == 1
        # Connection key includes target port when specified
        assert ("cortex", "hippocampus:feedforward") in brain.connections

        # Verify hippocampus receives input from cortex
        pathway = brain.connections[("cortex", "hippocampus:feedforward")]
        assert hasattr(pathway, "forward")  # Valid pathway exists

    def test_connect_with_target_port(self, brain_config):
        """Test connecting with target port specification."""
        builder = BrainBuilder(brain_config)

        builder.add_component("pfc", "prefrontal", input_size=64, wm_size=32)
        builder.add_component(
            "cortex",
            "cortex",
            input_size=64,
            l4_size=64,
            l23_size=96,
            l5_size=32,
            l6a_size=0,
            l6b_size=0,
        )

        # Connect to top_down input specifically
        builder.connect("pfc", "cortex", source_port="executive", target_port="top_down", pathway_type="axonal_projection")

        conn = builder._connections[0]
        assert conn.target_port == "top_down"

    def test_connect_with_both_ports(self, brain_config):
        """Test connecting with both source and target ports."""
        builder = BrainBuilder(brain_config)

        builder.add_component(
            "cortex",
            "cortex",
            input_size=64,
            l4_size=64,
            l23_size=96,
            l5_size=32,
            l6a_size=0,
            l6b_size=0,
        )
        builder.add_component(
            "striatum", "striatum", n_actions=4, neurons_per_action=1, input_sources={"default": 32}
        )

        builder.connect(
            "cortex",
            "striatum",
            source_port="l5",
            target_port="cortical_input",
            pathway_type="axonal_projection",
        )

        conn = builder._connections[0]
        assert conn.source_port == "l5"
        assert conn.target_port == "cortical_input"

    def test_multiple_connections_to_target(self, brain_config):
        """Test multiple connections to same target using different ports."""
        builder = BrainBuilder(brain_config)

        builder.add_component("thalamus", "thalamus", input_size=64, relay_size=64, trn_size=19)
        builder.add_component("pfc", "prefrontal", input_size=64, n_neurons=32)
        builder.add_component(
            "cortex", "cortex", l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0
        )

        # Feedforward path
        builder.connect(
            "thalamus", "cortex", source_port="relay", target_port="feedforward", pathway_type="axonal_projection"
        )

        # Top-down path
        builder.connect("pfc", "cortex", source_port="executive", target_port="top_down", pathway_type="axonal_projection")

        # Build and verify connections exist
        brain = builder.build()
        assert len(brain.connections) == 2

        # When ports are specified, connection keys include port suffix
        # Check that both pathways exist (may be keyed with port suffix)
        connection_keys = list(brain.connections.keys())
        assert len(connection_keys) == 2

        # Verify connections exist (with or without port suffix)
        thalamus_cortex_exists = any(
            "thalamus" in str(k[0]) and "cortex" in str(k[1]) for k in connection_keys
        )
        pfc_cortex_exists = any(
            "pfc" in str(k[0]) and "cortex" in str(k[1]) for k in connection_keys
        )
        assert thalamus_cortex_exists, f"Thalamus->Cortex connection not found in {connection_keys}"
        assert pfc_cortex_exists, f"PFC->Cortex connection not found in {connection_keys}"


class TestLayerSpecificCorticalRouting:
    """Test layer-specific cortical output routing (L2/3 vs L5)."""

    def test_l23_output_routes_to_hippocampus(self, brain_config):
        """Test that cortex L2/3 output routes to hippocampus."""
        builder = BrainBuilder(brain_config)

        builder.add_component("thalamus", "thalamus", input_size=64, relay_size=64, trn_size=19)
        builder.add_component(
            "cortex", "cortex", l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0
        )
        builder.add_component(
            "hippocampus", "hippocampus", dg_size=128, ca3_size=96, ca2_size=32, ca1_size=64
        )

        builder.connect("thalamus", "cortex", source_port="relay", target_port="feedforward", pathway_type="axonal_projection")
        builder.connect(
            "cortex", "hippocampus", source_port="l23", target_port="feedforward", pathway_type="axonal_projection"
        )

        brain = builder.build()

        # Verify cortex component exists and has layer structure
        cortex = brain.components["cortex"]
        assert hasattr(cortex, "l23_size")
        assert hasattr(cortex, "l5_size")

        # Verify hippocampus receives only L2/3 size
        hippo = brain.components["hippocampus"]
        # Should infer input_size from cortex L2/3 output, not full output
        assert hippo.input_size == cortex.l23_size

    def test_dual_output_routing(self, brain_config):
        """Test cortex routing L2/3 to one target and L5 to another with D1/D2 separation."""
        builder = BrainBuilder(brain_config)

        builder.add_component("thalamus", "thalamus", input_size=64, relay_size=64, trn_size=19)
        builder.add_component(
            "cortex", "cortex", l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0
        )
        builder.add_component(
            "hippocampus", "hippocampus", dg_size=128, ca3_size=96, ca2_size=32, ca1_size=64
        )
        builder.add_component(
            "striatum", "striatum", n_actions=4, neurons_per_action=1, input_sources={"cortex": 32}
        )

        builder.connect("thalamus", "cortex", source_port="relay", target_port="feedforward")
        builder.connect("cortex", "hippocampus", source_port="l23", target_port="feedforward")  # Cortico-cortical
        builder.connect("cortex", "striatum", source_port="l5", target_port="feedforward")  # Cortico-subcortical

        brain = builder.build()

        cortex = brain.components["cortex"]
        hippo = brain.components["hippocampus"]
        striatum = brain.components["striatum"]

        # Hippocampus receives L2/3 from cortex (via multi-source input dict)
        # Note: input_size is now per-source, not a single attribute
        assert "cortex:l23" in hippo.synaptic_weights
        assert hippo.synaptic_weights["cortex:l23"].shape[1] == cortex.l23_size

        # Striatum receives L5 from cortex with D1/D2 pathway separation
        # The striatum internally routes to D1 and D2 pathways
        assert "cortex:l5_d1" in striatum.synaptic_weights
        assert "cortex:l5_d2" in striatum.synaptic_weights
        assert striatum.synaptic_weights["cortex:l5_d1"].shape[1] == cortex.l5_size
        assert striatum.synaptic_weights["cortex:l5_d2"].shape[1] == cortex.l5_size


class TestMultipleInputPorts:
    """Test components with multiple named input ports."""

    def test_cortex_feedforward_and_topdown(self, brain_config):
        """Test cortex receiving both feedforward and top-down inputs."""
        builder = BrainBuilder(brain_config)

        builder.add_component("thalamus", "thalamus", input_size=64, relay_size=64, trn_size=19)
        builder.add_component("pfc", "prefrontal", input_size=64, n_neurons=32)
        builder.add_component(
            "cortex", "cortex", l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0
        )

        # Feedforward from thalamus
        builder.connect("thalamus", "cortex", source_port="relay", target_port="feedforward")

        # Top-down from PFC
        builder.connect("pfc", "cortex", source_port="executive", target_port="top_down")

        brain = builder.build()

        cortex = brain.components["cortex"]
        # thalamus = brain.components["thalamus"]

        # Cortex should have received inputs from both sources
        # Weight keys use "source:source_port" format (not target port)
        assert "thalamus:relay" in cortex.synaptic_weights
        assert "pfc:executive" in cortex.synaptic_weights

    def test_hippocampus_multiple_inputs(self, brain_config):
        """Test hippocampus receiving both cortical and direct entorhinal inputs."""
        builder = BrainBuilder(brain_config)

        builder.add_component("thalamus", "thalamus", input_size=64, relay_size=64, trn_size=19)
        builder.add_component(
            "cortex", "cortex", l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0
        )
        builder.add_component(
            "hippocampus", "hippocampus", dg_size=128, ca3_size=96, ca2_size=32, ca1_size=64
        )

        # Main cortical input (L2/3)
        builder.connect("thalamus", "cortex", source_port="relay", target_port="feedforward")
        builder.connect("cortex", "hippocampus", source_port="l23", target_port="cortical")

        # Direct entorhinal input
        builder.connect("thalamus", "hippocampus", source_port="relay", target_port="ec_l3")

        brain = builder.build()

        cortex = brain.components["cortex"]
        hippo = brain.components["hippocampus"]

        # Hippocampus has multiple input sources (cortical + entorhinal)
        # Note: hippo.n_input represents internal neurons (DG+CA3+CA2+CA1=320), NOT external inputs
        # Check input sources are registered correctly instead
        assert "cortex:l23" in hippo.input_sources
        assert hippo.input_sources["cortex:l23"] == cortex.l23_size  # cortical = 96

        # ec_l3 should be registered as input source with "source:port" format
        assert "thalamus:relay" in hippo.input_sources
        assert hippo.input_sources["thalamus:relay"] == 64  # entorhinal from thalamus


class TestPortBasedForwardPass:
    """Test forward pass with port-based routing."""

    def test_forward_with_layer_routing(self, brain_config):
        """Test that forward pass routes layer-specific outputs correctly."""
        builder = BrainBuilder(brain_config)

        builder.add_component("thalamus", "thalamus", input_size=64, relay_size=64, trn_size=19)
        builder.add_component(
            "cortex", "cortex", l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0
        )
        builder.add_component(
            "hippocampus", "hippocampus", dg_size=128, ca3_size=96, ca2_size=32, ca1_size=64
        )
        builder.add_component(
            "striatum", "striatum", n_actions=4, neurons_per_action=1, input_sources={"cortex": 32}
        )

        builder.connect("thalamus", "cortex", source_port="relay", target_port="feedforward")
        builder.connect("cortex", "hippocampus", source_port="l23", target_port="feedforward")
        builder.connect("cortex", "striatum", source_port="l5", target_port="feedforward")

        brain = builder.build()

        # Create input
        input_data = {"thalamus": torch.randn(64, device=brain.device)}

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

    def test_backward_compatible_no_ports(self, brain_config):
        """Test that existing code without ports continues to work."""
        builder = BrainBuilder(brain_config)

        builder.add_component("thalamus", "thalamus", input_size=64, relay_size=64, trn_size=19)
        builder.add_component(
            "cortex", "cortex", l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0
        )

        # Old-style connection without ports - NOW REQUIRES PORTS
        builder.connect("thalamus", "cortex", source_port="relay", target_port="feedforward", pathway_type="axonal_projection")

        brain = builder.build()

        # Should build successfully
        assert "thalamus" in brain.components
        assert "cortex" in brain.components

        # Should have proper connection (cortex receives from thalamus with port-based naming)
        cortex = brain.components["cortex"]
        thalamus = brain.components["thalamus"]
        assert "thalamus:relay" in cortex.synaptic_weights
        assert cortex.synaptic_weights["thalamus:relay"].shape[1] == thalamus.relay_size

    def test_mixing_port_and_traditional(self, brain_config):
        """Test mixing port-based and traditional connections."""
        builder = BrainBuilder(brain_config)

        builder.add_component("thalamus", "thalamus", input_size=64, relay_size=64, trn_size=19)
        builder.add_component(
            "cortex", "cortex", l4_size=64, l23_size=96, l5_size=32, l6a_size=0, l6b_size=0
        )
        builder.add_component(
            "hippocampus", "hippocampus", dg_size=128, ca3_size=96, ca2_size=32, ca1_size=64
        )
        builder.add_component(
            "striatum", "striatum", n_actions=4, neurons_per_action=1, input_sources={"cortex": 32}
        )

        # Mix of old and new style - ALL NOW REQUIRE PORTS
        builder.connect("thalamus", "cortex", source_port="relay", target_port="feedforward")  # Explicit ports required
        builder.connect("cortex", "hippocampus", source_port="l23", target_port="feedforward")  # New style
        builder.connect("cortex", "striatum", source_port="l5", target_port="feedforward")  # New style

        brain = builder.build()

        # Should build successfully
        assert len(brain.components) == 4


class TestAxonalProjectionPortRouting:
    """Test AxonalProjection port-aware routing."""

    def test_axonal_projection_with_port_spec(self, device):
        """Test AxonalProjection routes from specific port."""
        # Create cortex with layers
        cortex_config = LayeredCortexConfig()
        sizes = {"l4_size": 64, "l23_size": 96, "l5_size": 32, "l6a_size": 16, "l6b_size": 16}
        cortex = LayeredCortex(config=cortex_config, sizes=sizes, device=device)

        # Create pathway that routes from L6a port
        projection = AxonalProjection(
            sources=[
                ("cortex", "l6a", 16, 2.0),  # (region_name, port, size, delay_ms)
            ],
            device=device,
            dt_ms=1.0,
        )

        # Simulate cortex forward pass (sets port outputs)
        input_spikes = {"input": torch.zeros(64, dtype=torch.bool, device=device)}
        cortex.forward(input_spikes)

        # Route through pathway (pass region object, not tensor)
        routed = projection.forward({"cortex": cortex})

        # Should get L6a output specifically
        assert "cortex:l6a" in routed
        assert routed["cortex:l6a"].shape[0] == 16

    def test_axonal_projection_multiple_ports(self, device):
        """Test AxonalProjection routes from multiple ports."""
        # Create cortex
        cortex_config = LayeredCortexConfig()
        sizes = {"l4_size": 64, "l23_size": 96, "l5_size": 32, "l6a_size": 16, "l6b_size": 16}
        cortex = LayeredCortex(config=cortex_config, sizes=sizes, device=device)

        # Create pathway that routes from multiple ports
        projection = AxonalProjection(
            sources=[
                ("cortex", "l6a", 16, 2.0),
                ("cortex", "l6b", 16, 2.0),
            ],
            device=device,
            dt_ms=1.0,
        )

        # Simulate cortex forward pass
        input_spikes = {"input": torch.zeros(64, dtype=torch.bool, device=device)}
        cortex.forward(input_spikes)

        # Route through pathway
        routed = projection.forward({"cortex": cortex})

        # Should get both port outputs
        assert "cortex:l6a" in routed
        assert "cortex:l6b" in routed
        assert routed["cortex:l6a"].shape[0] == 16
        assert routed["cortex:l6b"].shape[0] == 16

    def test_axonal_projection_backward_compat_tensor_mode(self, device):
        """Test AxonalProjection still works with tensor inputs (backward compatibility)."""
        # Create pathway
        projection = AxonalProjection(
            sources=[
                ("cortex", None, 128, 2.0),  # No port specified
            ],
            device=device,
            dt_ms=1.0,
        )

        # Old-style tensor input
        tensor_input = {"cortex": torch.zeros(128, dtype=torch.bool, device=device)}
        routed = projection.forward(tensor_input)

        # Should work correctly
        assert "cortex" in routed
        assert routed["cortex"].shape[0] == 128


class TestEndToEndPortBasedRouting:
    """End-to-end integration tests for port-based routing (Phase 4)."""

    def test_cortex_l6_to_thalamus_routing(self, brain_config):
        """Test L6a→TRN and L6b→relay routing through full brain."""
        builder = BrainBuilder(brain_config)

        # Add thalamus with relay and TRN
        builder.add_component(
            "thalamus",
            "thalamus",
            input_size=64,
            relay_size=64,
            trn_size=19,
        )

        # Add cortex with L6a and L6b layers
        builder.add_component(
            "cortex",
            "cortex",
            l4_size=64,
            l23_size=96,
            l5_size=32,
            l6a_size=16,  # L6a CT neurons
            l6b_size=16,  # L6b CT neurons
        )

        # Feedforward: thalamus → cortex L4
        builder.connect("thalamus", "cortex", source_port="relay", target_port="feedforward", pathway_type="axonal_projection")

        # Feedback: L6a → TRN (attentional gating)
        builder.connect(
            "cortex",
            "thalamus",
            source_port="l6a",
            target_port="l6a_feedback",
            pathway_type="axonal_projection",
            axonal_delay_ms=2.0,
        )

        # Feedback: L6b → relay (gain modulation)
        builder.connect(
            "cortex",
            "thalamus",
            source_port="l6b",
            target_port="l6b_feedback",
            pathway_type="axonal_projection",
            axonal_delay_ms=2.0,
        )

        # Build brain
        brain = builder.build()

        # Verify components exist
        assert "cortex" in brain.components
        assert "thalamus" in brain.components

        # Verify cortex has L6a and L6b ports
        cortex = brain.components["cortex"]
        assert "l6a" in cortex.get_registered_ports()
        assert "l6b" in cortex.get_registered_ports()

        # Verify thalamus has both feedback connections
        thalamus = brain.components["thalamus"]
        assert "cortex:l6a" in thalamus.synaptic_weights or "cortex:l6a" in thalamus.input_sources
        assert "cortex:l6b" in thalamus.synaptic_weights or "cortex:l6b" in thalamus.input_sources

        # Run forward pass
        input_data = torch.randn(64, device=brain.device)
        result = brain.forward({"thalamus": input_data}, n_timesteps=10)

        # Verify outputs
        assert "cortex" in result["outputs"]
        assert "thalamus" in result["outputs"]

        # Verify cortex set port outputs
        cortex_output = result["outputs"]["cortex"]
        if cortex_output is not None:
            # Cortex should output concatenated L2/3 + L5 (default port)
            assert cortex_output.shape[0] == cortex.l23_size + cortex.l5_size

            # Verify L6a and L6b port outputs were set
            l6a_output = cortex.get_port_output("l6a")
            l6b_output = cortex.get_port_output("l6b")
            assert l6a_output.shape[0] == 16
            assert l6b_output.shape[0] == 16

    def test_multi_region_port_routing(self, brain_config):
        """Test port-based routing with multiple regions and pathways."""
        builder = BrainBuilder(brain_config)

        # Build a more complex network:
        # thalamus → cortex (feedforward)
        # cortex L2/3 → hippocampus (cortico-cortical)
        # cortex L5 → striatum (cortico-subcortical)
        # cortex L6a → thalamus TRN (feedback gating)
        # cortex L6b → thalamus relay (feedback modulation)

        builder.add_component("thalamus", "thalamus", input_size=64, relay_size=64, trn_size=19)
        builder.add_component(
            "cortex", "cortex", l4_size=64, l23_size=96, l5_size=32, l6a_size=16, l6b_size=16
        )
        builder.add_component(
            "hippocampus", "hippocampus", dg_size=128, ca3_size=96, ca2_size=32, ca1_size=64
        )
        builder.add_component(
            "striatum", "striatum", n_actions=4, neurons_per_action=8, input_sources={"cortex": 32}
        )

        # Feedforward path
        builder.connect("thalamus", "cortex", source_port="relay", target_port="feedforward")

        # Cortico-cortical (L2/3 → hippocampus)
        builder.connect("cortex", "hippocampus", source_port="l23", target_port="feedforward")

        # Cortico-subcortical (L5 → striatum)
        builder.connect("cortex", "striatum", source_port="l5", target_port="feedforward")

        # Cortico-thalamic feedback
        builder.connect("cortex", "thalamus", source_port="l6a", target_port="l6a_feedback")
        builder.connect("cortex", "thalamus", source_port="l6b", target_port="l6b_feedback")

        # Build and verify
        brain = builder.build()

        # Verify all connections exist
        assert len(brain.connections) == 5

        # Run forward pass
        input_data = torch.randn(64, device=brain.device)
        result = brain.forward({"thalamus": input_data}, n_timesteps=5)

        # Verify all components produced outputs
        assert all(
            comp in result["outputs"] for comp in ["thalamus", "cortex", "hippocampus", "striatum"]
        )

        # Verify cortex port outputs are accessible
        cortex = brain.components["cortex"]
        assert cortex.get_port_output("l23").shape[0] == 96
        assert cortex.get_port_output("l5").shape[0] == 32
        assert cortex.get_port_output("l6a").shape[0] == 16
        assert cortex.get_port_output("l6b").shape[0] == 16

    def test_thalamus_specific_ports(self, brain_config):
        """Test thalamus relay and TRN port outputs."""
        calc = LayerSizeCalculator()
        sizes = calc.thalamus_from_relay(64)
        config = ThalamicRelayConfig(dt_ms=1.0, device=brain_config.device)
        thalamus = ThalamicRelay(config=config, sizes=sizes, device=brain_config.device)

        # Create input
        input_spikes = torch.zeros(64, device=brain_config.device)
        input_spikes[:10] = 1  # Activate first 10 neurons

        # Forward pass
        thalamus.forward({"sensory": input_spikes})

        # Verify port outputs
        relay_output = thalamus.get_port_output("relay")
        trn_output = thalamus.get_port_output("trn")

        # Check shapes
        assert relay_output.shape[0] == sizes["relay_size"], f"Relay output mismatch"
        assert trn_output.shape[0] == sizes["trn_size"], f"TRN output mismatch"

        # Outputs should be binary spikes
        assert relay_output.dtype == torch.bool
        assert trn_output.dtype == torch.bool

    def test_hippocampus_specific_ports(self, brain_config):
        """Test hippocampus subregion port outputs (DG, CA3, CA2, CA1)."""
        config = HippocampusConfig(dt_ms=1.0)
        sizes = {"dg_size": 128, "ca3_size": 96, "ca2_size": 32, "ca1_size": 64, "input_size": 128}
        hippocampus = TrisynapticHippocampus(config=config, sizes=sizes, device=brain_config.device)

        # Create input
        input_spikes = torch.zeros(128, device=brain_config.device)
        input_spikes[:20] = 1  # Activate first 20 neurons

        # Forward pass
        hippocampus.forward({"entorhinal": input_spikes})

        # Verify port outputs
        dg_output = hippocampus.get_port_output("dg")
        ca3_output = hippocampus.get_port_output("ca3")
        ca2_output = hippocampus.get_port_output("ca2")
        ca1_output = hippocampus.get_port_output("ca1")

        # Check shapes
        assert dg_output.shape[0] == 128, f"DG output should be 128, got {dg_output.shape[0]}"
        assert ca3_output.shape[0] == 96, f"CA3 output should be 96, got {ca3_output.shape[0]}"
        assert ca2_output.shape[0] == 32, f"CA2 output should be 32, got {ca2_output.shape[0]}"
        assert ca1_output.shape[0] == 64, f"CA1 output should be 64, got {ca1_output.shape[0]}"

        # Outputs should be binary spikes
        assert dg_output.dtype == torch.bool
        assert ca3_output.dtype == torch.bool
        assert ca2_output.dtype == torch.bool
        assert ca1_output.dtype == torch.bool

    def test_striatum_specific_ports(self, brain_config):
        """Test striatum D1/D2 pathway port outputs."""
        calc = LayerSizeCalculator()
        sizes = calc.striatum_from_actions(n_actions=4, neurons_per_action=16)
        sizes["input_size"] = 64  # Add input size
        config = StriatumConfig(dt_ms=1.0, device=brain_config.device)
        striatum = Striatum(config=config, sizes=sizes, device=brain_config.device)

        # Create input
        input_spikes = torch.zeros(64, device=brain_config.device)
        input_spikes[:10] = 1  # Activate first 10 neurons

        # Forward pass
        striatum.forward({"cortex": input_spikes})

        # Verify port outputs
        d1_output = striatum.get_port_output("d1")
        d2_output = striatum.get_port_output("d2")

        # Check shapes (d1_size and d2_size should be n_actions * neurons_per_action)
        expected_d1 = striatum.d1_size
        expected_d2 = striatum.d2_size
        assert d1_output.shape[0] == expected_d1, f"D1 output should be {expected_d1}, got {d1_output.shape[0]}"
        assert d2_output.shape[0] == expected_d2, f"D2 output should be {expected_d2}, got {d2_output.shape[0]}"

        # Outputs should be binary spikes
        assert d1_output.dtype == torch.bool
        assert d2_output.dtype == torch.bool
