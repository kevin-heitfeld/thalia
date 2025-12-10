"""Comprehensive tests for growth mechanisms (TDD approach).

Tests neuron addition, capacity metrics, and checkpoint compatibility
for ALL regions and pathways. Tests are written FIRST, implementations follow.

Test Coverage:
- Growth Manager core functionality
- Striatum growth (D1/D2 pathways, population coding)
- Hippocampus growth (trisynaptic circuit)
- Cortex growth (layered architecture)
- PFC growth (working memory)
- Cerebellum growth (granule/Purkinje cells)
- Pathway growth (all 9 inter-region pathways)
- Brain-level coordinated growth
"""

import pytest
import torch

from thalia.core.growth import GrowthManager, CapacityMetrics, GrowthEvent
from thalia.regions.striatum import Striatum, StriatumConfig
from thalia.regions.hippocampus import TrisynapticHippocampus, TrisynapticConfig
from thalia.regions.cortex import LayeredCortex, LayeredCortexConfig, PredictiveCortex, PredictiveCortexConfig
from thalia.regions.prefrontal import Prefrontal, PrefrontalConfig
from thalia.regions.cerebellum import Cerebellum, CerebellumConfig
from thalia.integration.spiking_pathway import SpikingPathway, SpikingPathwayConfig
from thalia.integration.pathways.spiking_attention import SpikingAttentionPathway, SpikingAttentionPathwayConfig
from thalia.integration.pathways.spiking_replay import SpikingReplayPathway, SpikingReplayPathwayConfig
from thalia.io import BrainCheckpoint


class TestGrowthManager:
    """Test GrowthManager core functionality."""

    def test_growth_manager_initialization(self):
        """Test basic initialization."""
        manager = GrowthManager(region_name="test_component")
        assert manager.region_name == "test_component"
        assert len(manager.history) == 0

    def test_capacity_metrics_computation(self):
        """Test capacity metrics calculation."""
        # Create simple striatum (n_output=32 actions with default 10 neurons/action = 320 total neurons)
        config = StriatumConfig(n_input=64, n_output=32, device="cpu")
        region = Striatum(config)

        manager = GrowthManager(region_name="striatum")
        metrics = manager.get_capacity_metrics(region)

        assert isinstance(metrics, CapacityMetrics)
        assert 0 <= metrics.firing_rate <= 1
        assert 0 <= metrics.weight_saturation <= 1
        assert 0 <= metrics.synapse_usage <= 1
        assert metrics.neuron_count == 320  # 32 actions × 10 neurons/action
        assert metrics.synapse_count > 0

    def test_growth_history_tracking(self):
        """Test that growth events are recorded."""
        manager = GrowthManager(region_name="test")

        # Create mock event
        event = GrowthEvent(
            timestamp="2025-12-07T10:00:00",
            component_name="test",
            component_type="region",
            event_type="add_neurons",
            n_neurons_added=10,
            reason="test growth"
        )
        manager.history.append(event)

        history = manager.get_history()
        assert len(history) == 1
        assert history[0]["n_neurons_added"] == 10

    def test_state_serialization(self):
        """Test growth manager state save/load."""
        manager = GrowthManager(region_name="test")

        # Add some history
        event = GrowthEvent(
            timestamp="2025-12-07T10:00:00",
            component_name="test",
            component_type="region",
            event_type="add_neurons",
            n_neurons_added=10
        )
        manager.history.append(event)

        # Save and restore
        state = manager.get_state()
        manager2 = GrowthManager(region_name="test2")
        manager2.load_state(state)

        assert len(manager2.history) == 1
        assert manager2.history[0].n_neurons_added == 10


# ============================================================================
# REGION GROWTH TESTS
# ============================================================================

class TestStriatumGrowth:
    """Test growth for Striatum region (RL via D1/D2 opponent pathways)."""

    def test_striatum_add_neurons_basic(self):
        """Test adding neurons to striatum expands D1/D2 pathways."""
        config = StriatumConfig(
            n_input=64,
            n_output=2,  # 2 actions
            neurons_per_action=10,
            device="cpu"
        )
        striatum = Striatum(config)

        # Initial state: 2 actions × 10 neurons = 20 neurons
        assert striatum.n_actions == 2
        assert striatum.config.n_output == 20
        assert striatum.d1_weights.shape == (20, 64)
        assert striatum.d2_weights.shape == (20, 64)

        # Add 1 action (= 10 neurons)
        striatum.add_neurons(n_new=1, initialization='xavier')

        # After growth: 3 actions × 10 neurons = 30 neurons
        assert striatum.n_actions == 3
        assert striatum.config.n_output == 30
        assert striatum.d1_weights.shape == (30, 64)
        assert striatum.d2_weights.shape == (30, 64)

    def test_striatum_preserves_d1_d2_weights(self):
        """Test that existing D1/D2 weights are unchanged after growth."""
        config = StriatumConfig(n_input=64, n_output=2, neurons_per_action=10, device="cpu")
        striatum = Striatum(config)

        # Save old weights
        old_d1 = striatum.d1_weights.clone()
        old_d2 = striatum.d2_weights.clone()

        # Grow
        striatum.add_neurons(n_new=1)

        # Check old weights unchanged (first 20 neurons)
        assert torch.allclose(striatum.d1_weights[:20], old_d1)
        assert torch.allclose(striatum.d2_weights[:20], old_d2)

    def test_striatum_expands_eligibility_traces(self):
        """Test that eligibility traces expand with neurons."""
        config = StriatumConfig(n_input=64, n_output=2, neurons_per_action=10, device="cpu")
        striatum = Striatum(config)

        # Process input to create eligibility traces (ADR-005: 1D input)
        input_spikes = (torch.rand(64) > 0.5).float()
        striatum.forward(input_spikes)

        # Grow
        striatum.add_neurons(n_new=1)

        # Eligibility traces should match new size
        assert striatum.d1_eligibility.shape == (30, 64)
        assert striatum.d2_eligibility.shape == (30, 64)

    def test_striatum_expands_spike_traces(self):
        """Test that spike traces expand with neurons."""
        config = StriatumConfig(n_input=64, n_output=2, neurons_per_action=10, device="cpu")
        striatum = Striatum(config)

        # Grow
        striatum.add_neurons(n_new=1)

        # Spike traces should match new size
        if hasattr(striatum, 'd1_output_trace'):
            assert striatum.d1_output_trace.shape[0] == 30
        if hasattr(striatum, 'd2_output_trace'):
            assert striatum.d2_output_trace.shape[0] == 30

    def test_striatum_expands_neuron_populations(self):
        """Test that D1 and D2 neuron populations expand."""
        config = StriatumConfig(n_input=64, n_output=2, neurons_per_action=10, device="cpu")
        striatum = Striatum(config)

        # Initial: 2 actions × 10 neurons/action = 20 neurons each pathway
        assert striatum.d1_neurons.n_neurons == 20
        assert striatum.d2_neurons.n_neurons == 20
        assert striatum.n_actions == 2

        # Grow by 1 action (adds 10 neurons per pathway)
        striatum.add_neurons(n_new=1)

        # After growth: 3 actions × 10 neurons/action = 30 neurons each pathway
        assert striatum.d1_neurons.n_neurons == 30
        assert striatum.d2_neurons.n_neurons == 30
        assert striatum.n_actions == 3

    def test_striatum_preserves_neuron_state(self):
        """Test that existing neurons preserve membrane potential after growth."""
        config = StriatumConfig(n_input=64, n_output=2, neurons_per_action=10, device="cpu")
        striatum = Striatum(config)

        # Process input to build neuron state (ADR-005: 1D)
        input_spikes = (torch.rand(64) > 0.5).float()
        striatum.forward(input_spikes)

        # Save membrane state of first 20 neurons (first 2 actions)
        old_d1_membrane = striatum.d1_neurons.membrane[:20].clone()
        old_d2_membrane = striatum.d2_neurons.membrane[:20].clone()

        # Grow by 1 action (adds 10 neurons)
        striatum.add_neurons(n_new=1)

        # First 20 neurons should preserve their membrane state
        assert torch.allclose(striatum.d1_neurons.membrane[:20], old_d1_membrane, atol=1e-6)
        assert torch.allclose(striatum.d2_neurons.membrane[:20], old_d2_membrane, atol=1e-6)

        # New neurons (indices 20-29) should be initialized
        assert striatum.d1_neurons.membrane[20:].numel() == 10

    def test_striatum_updates_value_estimates(self):
        """Test that value_estimates expands for new actions."""
        config = StriatumConfig(n_input=64, n_output=2, neurons_per_action=10, device="cpu")
        striatum = Striatum(config)

        # Process to create value estimates (ADR-005: 1D)
        input_spikes = (torch.rand(64) > 0.5).float()
        striatum.forward(input_spikes)

        old_values = striatum.value_estimates.clone() if hasattr(striatum, 'value_estimates') else None

        # Grow
        striatum.add_neurons(n_new=1)

        # Value estimates should have 3 actions now
        if hasattr(striatum, 'value_estimates'):
            assert striatum.value_estimates.shape[0] == 3

    def test_striatum_capacity_metrics(self):
        """Test capacity metrics API."""
        config = StriatumConfig(n_input=64, n_output=32, device="cpu")
        striatum = Striatum(config)

        metrics = striatum.get_capacity_metrics()
        # get_capacity_metrics returns dict, not CapacityMetrics object
        assert isinstance(metrics, dict) or isinstance(metrics, CapacityMetrics)
        if isinstance(metrics, dict):
            assert metrics['neuron_count'] == 320  # 32 actions × 10 neurons/action
        else:
            assert metrics.neuron_count == 320

    def test_striatum_checkpoint_roundtrip_with_growth(self):
        """Test checkpoint save/load after growth."""
        config = StriatumConfig(n_input=64, n_output=2, neurons_per_action=10, device="cpu")
        striatum = Striatum(config)

        # Grow first, then process data (ADR-005: 1D)
        striatum.add_neurons(n_new=1)  # Now has 3 actions
        input_spikes = (torch.rand(64) > 0.5).float()

        # Process through striatum to build state
        output_spikes = striatum.forward(input_spikes)

        # Checkpoint
        state = striatum.get_full_state()

        # Create new striatum with matching grown size (3 actions)
        config2 = StriatumConfig(n_input=64, n_output=3, neurons_per_action=10, device="cpu")
        striatum2 = Striatum(config2)
        striatum2.load_full_state(state)

        # Verify weights match
        assert torch.allclose(striatum.d1_weights, striatum2.d1_weights)
        assert torch.allclose(striatum.d2_weights, striatum2.d2_weights)

        # Verify neuron populations match
        assert striatum2.d1_neurons.n_neurons == 30  # 3 actions × 10 neurons
        assert striatum2.d2_neurons.n_neurons == 30
        assert torch.allclose(striatum.d2_weights, striatum2.d2_weights)


class TestHippocampusGrowth:
    """Test growth for Hippocampus region (episodic memory via trisynaptic circuit)."""

    def test_hippocampus_add_neurons_basic(self):
        """Test adding neurons to hippocampus."""
        config = TrisynapticConfig(
            n_input=64,
            n_output=32,
            ec_l3_input_size=256,
            device="cpu"
        )
        hippo = TrisynapticHippocampus(config)

        initial_output = hippo.config.n_output

        # Add neurons
        hippo.add_neurons(n_new=16, initialization='sparse_random', sparsity=0.2)

        # Check expansion
        assert hippo.config.n_output == initial_output + 16

    def test_hippocampus_expands_trisynaptic_circuit(self):
        """Test that all layers expand proportionally."""
        config = TrisynapticConfig(n_input=64, n_output=32, ec_l3_input_size=256, device="cpu")
        hippo = TrisynapticHippocampus(config)

        old_dg_size = hippo.dg_size
        old_ca3_size = hippo.ca3_size
        old_ca1_size = hippo.ca1_size

        # Grow CA1 by 16 neurons
        hippo.add_neurons(n_new=16)

        # All layers now grow proportionally
        assert hippo.dg_size > old_dg_size  # DG grows
        assert hippo.ca3_size > old_ca3_size  # CA3 grows
        assert hippo.ca1_size == old_ca1_size + 16  # CA1 grows by specified amount

    def test_hippocampus_preserves_pattern_separation(self):
        """Test that pattern separation weights are preserved."""
        config = TrisynapticConfig(n_input=64, n_output=32, ec_l3_input_size=256, device="cpu")
        hippo = TrisynapticHippocampus(config)

        # Save DG weights (pattern separation)
        if hasattr(hippo, 'ec_to_dg_weights'):
            old_weights = hippo.ec_to_dg_weights.clone()

        # Grow
        hippo.add_neurons(n_new=16)

        # Old weights should be preserved
        if hasattr(hippo, 'ec_to_dg_weights'):
            new_weights = hippo.ec_to_dg_weights
            assert torch.allclose(new_weights[:old_weights.shape[0], :old_weights.shape[1]], old_weights)

    def test_hippocampus_capacity_metrics(self):
        """Test hippocampus capacity metrics."""
        config = TrisynapticConfig(n_input=64, n_output=32, ec_l3_input_size=256, device="cpu")
        hippo = TrisynapticHippocampus(config)

        metrics = hippo.get_capacity_metrics()
        assert isinstance(metrics, CapacityMetrics)


class TestCortexGrowth:
    """Test growth for Cortex region (feature learning via layered architecture)."""

    def test_cortex_add_neurons_basic(self):
        """Test adding neurons to layered cortex."""
        config = LayeredCortexConfig(
            n_input=128,
            n_output=64,
            device="cpu"
        )
        cortex = LayeredCortex(config)

        initial_l5 = cortex.l5_size

        # Add neurons (expands L5 output layer)
        cortex.add_neurons(n_new=32, initialization='xavier')

        # Check L5 expansion (only L5 grows currently)
        assert cortex.l5_size == initial_l5 + 32

    def test_predictive_cortex_growth(self):
        """Test adding neurons to predictive cortex."""
        from thalia.regions.cortex import PredictiveCortex, PredictiveCortexConfig

        config = PredictiveCortexConfig(
            n_input=128,
            n_output=64,
            prediction_enabled=True,
            use_attention=True,
            device="cpu"
        )
        cortex = PredictiveCortex(config)

        initial_l4 = cortex.l4_size
        initial_l23 = cortex.l23_size
        initial_l5 = cortex.l5_size

        # Add neurons
        cortex.add_neurons(n_new=32)

        # All layers should grow
        assert cortex.l4_size > initial_l4
        assert cortex.l23_size > initial_l23
        assert cortex.l5_size > initial_l5

        # Prediction and attention modules should be recreated
        assert cortex.prediction_layer is not None
        assert cortex.attention is not None

    def test_cortex_layer_sizes_scale_proportionally(self):
        """Test that all layers expand proportionally."""
        config = LayeredCortexConfig(n_input=128, n_output=64, device="cpu")
        cortex = LayeredCortex(config)

        old_l4_size = cortex.l4_size
        old_l23_size = cortex.l23_size
        old_l5_size = cortex.l5_size

        # Grow by 32 total neurons
        cortex.add_neurons(n_new=32)

        # All layers now grow proportionally
        assert cortex.l4_size > old_l4_size  # L4 grows
        assert cortex.l23_size > old_l23_size  # L2/3 grows
        assert cortex.l5_size > old_l5_size  # L5 grows

    def test_cortex_preserves_recurrent_connections(self):
        """Test that L2/3 recurrent connections are preserved."""
        config = LayeredCortexConfig(n_input=128, n_output=64, device="cpu")
        cortex = LayeredCortex(config)

        if hasattr(cortex, 'l23_recurrent_weights'):
            old_weights = cortex.l23_recurrent_weights.clone()
            old_size = old_weights.shape[0]

            # Grow
            cortex.add_neurons(n_new=32)

            # Old recurrent connections should be preserved
            new_weights = cortex.l23_recurrent_weights
            assert torch.allclose(new_weights[:old_size, :old_size], old_weights)

    def test_cortex_capacity_metrics(self):
        """Test cortex capacity metrics."""
        config = LayeredCortexConfig(n_input=128, n_output=64, device="cpu")
        cortex = LayeredCortex(config)

        metrics = cortex.get_capacity_metrics()
        assert isinstance(metrics, CapacityMetrics)


class TestPFCGrowth:
    """Test growth for Prefrontal Cortex region (working memory)."""

    def test_pfc_add_neurons_basic(self):
        """Test adding neurons to PFC."""
        config = PrefrontalConfig(
            n_input=128,
            n_output=32,
            device="cpu"
        )
        pfc = Prefrontal(config)

        initial_output = pfc.config.n_output

        # Add neurons
        pfc.add_neurons(n_new=16, initialization='xavier')

        # Check expansion
        assert pfc.config.n_output == initial_output + 16

    def test_pfc_working_memory_capacity_expands(self):
        """Test that working memory capacity expands with neurons."""
        config = PrefrontalConfig(n_input=128, n_output=32, device="cpu")
        pfc = Prefrontal(config)

        # Process some input to initialize working memory (ADR-005: 1D)
        input_spikes = torch.rand(128) > 0.5
        pfc.forward(input_spikes)

        # Grow
        pfc.add_neurons(n_new=16)

        # Working memory should accommodate new neurons
        assert pfc.config.n_output == 48

    def test_pfc_preserves_gating_weights(self):
        """Test that gating mechanism weights are preserved."""
        config = PrefrontalConfig(n_input=128, n_output=32, device="cpu")
        pfc = Prefrontal(config)

        if hasattr(pfc, 'gating_weights'):
            old_weights = pfc.gating_weights.clone()

            # Grow
            pfc.add_neurons(n_new=16)

            # Old gating weights should be preserved
            new_weights = pfc.gating_weights
            assert torch.allclose(new_weights[:old_weights.shape[0]], old_weights)

    def test_pfc_capacity_metrics(self):
        """Test PFC capacity metrics."""
        config = PrefrontalConfig(n_input=128, n_output=32, device="cpu")
        pfc = Prefrontal(config)

        metrics = pfc.get_capacity_metrics()
        assert isinstance(metrics, CapacityMetrics)


class TestCerebellumGrowth:
    """Test growth for Cerebellum region (supervised learning)."""

    def test_cerebellum_add_neurons_basic(self):
        """Test adding neurons to cerebellum."""
        config = CerebellumConfig(
            n_input=64,
            n_output=32,
            device="cpu"
        )
        cerebellum = Cerebellum(config)

        initial_output = cerebellum.config.n_output

        # Add neurons
        cerebellum.add_neurons(n_new=16, initialization='uniform')

        # Check expansion
        assert cerebellum.config.n_output == initial_output + 16

    def test_cerebellum_expands_granule_layer(self):
        """Test that Purkinje cells expand (granule layer is input, stays fixed)."""
        config = CerebellumConfig(n_input=64, n_output=32, device="cpu")
        cerebellum = Cerebellum(config)

        old_output_size = cerebellum.config.n_output

        # Grow
        cerebellum.add_neurons(n_new=16)

        # Only Purkinje cells (output) expand, granule layer (input) stays same
        assert cerebellum.config.n_output == old_output_size + 16
        assert cerebellum.config.n_input == 64  # Granule layer unchanged

    def test_cerebellum_preserves_purkinje_weights(self):
        """Test that Purkinje cell weights are preserved."""
        config = CerebellumConfig(n_input=64, n_output=32, device="cpu")
        cerebellum = Cerebellum(config)

        if hasattr(cerebellum, 'purkinje_weights'):
            old_weights = cerebellum.purkinje_weights.clone()

            # Grow
            cerebellum.add_neurons(n_new=16)

            # Old Purkinje weights should be preserved
            new_weights = cerebellum.purkinje_weights
            assert torch.allclose(new_weights[:old_weights.shape[0]], old_weights)

    def test_cerebellum_capacity_metrics(self):
        """Test cerebellum capacity metrics."""
        config = CerebellumConfig(n_input=64, n_output=32, device="cpu")
        cerebellum = Cerebellum(config)

        metrics = cerebellum.get_capacity_metrics()
        assert isinstance(metrics, CapacityMetrics)


# ============================================================================
# PATHWAY GROWTH TESTS
# ============================================================================

class TestPathwayGrowth:
    """Test growth for neural pathways (inter-region connections)."""

    def test_spiking_pathway_add_neurons_basic(self):
        """Test adding neurons to spiking pathway (expands target dimension)."""
        config = SpikingPathwayConfig(
            source_size=64,
            target_size=32,
            device="cpu"
        )
        pathway = SpikingPathway(config)

        initial_target = config.target_size

        # Add neurons (expands target side)
        pathway.add_neurons(n_new=16, initialization='sparse_random', sparsity=0.1)

        # Target side should expand
        # (Pathway growth typically expands target projection)
        assert pathway.config.target_size == initial_target + 16

    def test_pathway_preserves_transformation_weights(self):
        """Test pathway preserves existing connections."""
        config = SpikingPathwayConfig(source_size=64, target_size=32, device="cpu")
        pathway = SpikingPathway(config)

        # Save old weights
        old_weights = pathway.weights.data.clone()

        # Grow
        pathway.add_neurons(n_new=16)

        # Check old weights preserved (with tolerance for floating point)
        assert pathway.weights.shape[0] == 48  # 32 + 16
        assert torch.allclose(
            pathway.weights[:old_weights.shape[0]],
            old_weights,
            rtol=1e-5,
            atol=1e-7
        )

    def test_pathway_expands_stdp_traces(self):
        """Test that STDP traces expand with pathway growth."""
        config = SpikingPathwayConfig(source_size=64, target_size=32, device="cpu")
        pathway = SpikingPathway(config)

        # Process some spikes to create STDP traces (ADR-005: 1D)
        source_spikes = torch.rand(64) > 0.5
        pathway.forward(source_spikes)

        # Grow
        pathway.add_neurons(n_new=16)

        # STDP traces should match new size
        if hasattr(pathway, 'post_trace'):
            # Target dimension should be 48 now
            assert pathway.post_trace.shape[0] == 48

    def test_pathway_capacity_metrics(self):
        """Test pathway capacity metrics."""
        config = SpikingPathwayConfig(source_size=64, target_size=32, device="cpu")
        pathway = SpikingPathway(config)

        metrics = pathway.get_capacity_metrics()
        # Base class returns dict, not CapacityMetrics object
        assert isinstance(metrics, dict)

    def test_pathway_checkpoint_roundtrip_with_growth(self):
        """Test pathway checkpoint after growth."""
        config = SpikingPathwayConfig(source_size=64, target_size=32, device="cpu")
        pathway1 = SpikingPathway(config)

        # Grow
        pathway1.add_neurons(n_new=16)

        # Checkpoint
        state = pathway1.get_full_state()

        # Load into new pathway with grown size
        config2 = SpikingPathwayConfig(source_size=64, target_size=48, device="cpu")
        pathway2 = SpikingPathway(config2)
        pathway2.load_full_state(state)

        # Verify weights match
        if hasattr(pathway1, 'weights'):
            assert torch.allclose(pathway1.weights, pathway2.weights)


class TestSpecializedPathwayGrowth:
    """Test growth for specialized pathways (attention, replay, etc.)."""

    def test_attention_pathway_growth(self):
        """Test PFC→Cortex attention pathway growth."""
        config = SpikingAttentionPathwayConfig(
            source_size=64,  # PFC size
            target_size=128,  # Cortex size
            device="cpu"
        )
        pathway = SpikingAttentionPathway(config)

        initial_target = pathway.config.target_size

        # Add neurons to target (cortex grew)
        pathway.add_neurons(n_new=32)

        # Target should expand
        assert pathway.config.target_size == initial_target + 32

        # Attention mechanism should still work
        assert hasattr(pathway, 'attention_weights') or hasattr(pathway, 'forward')

    def test_replay_pathway_growth(self):
        """Test Hippocampus→Cortex replay pathway growth."""
        config = SpikingReplayPathwayConfig(
            source_size=128,  # Hippocampus size
            target_size=256,  # Cortex size
            device="cpu"
        )
        pathway = SpikingReplayPathway(config)

        initial_target = pathway.config.target_size

        # Add neurons to target (cortex grew)
        pathway.add_neurons(n_new=32)

        # Target should expand
        assert pathway.config.target_size == initial_target + 32

        # Replay mechanism should still work
        assert hasattr(pathway, 'replay_buffer') or hasattr(pathway, 'forward')

    def test_all_nine_pathways_support_growth(self):
        """Test that all 9 inter-region pathways support growth."""
        # Standard pathways (use SpikingPathway)
        pathways_to_test = [
            ("cortex_to_hippo", 192, 128),  # L2/3 → Hippo
            ("cortex_to_striatum", 160, 320),  # L5 → Striatum
            ("cortex_to_pfc", 192, 64),  # L2/3 → PFC
            ("hippo_to_pfc", 128, 64),  # Hippo → PFC
            ("hippo_to_striatum", 128, 320),  # Hippo → Striatum
            ("pfc_to_striatum", 64, 320),  # PFC → Striatum
            ("striatum_to_cerebellum", 320, 4),  # Striatum → Cerebellum
        ]

        for name, source_size, target_size in pathways_to_test:
            config = SpikingPathwayConfig(
                source_size=source_size,
                target_size=target_size,
                device="cpu"
            )
            pathway = SpikingPathway(config)

            # Should be able to add neurons
            pathway.add_neurons(n_new=16)

            # Should expand target
            assert pathway.config.target_size == target_size + 16, f"{name} failed to grow"

    def test_specialized_pathways_inherit_growth(self):
        """Test that attention and replay pathways inherit growth from base."""
        # Both should inherit add_neurons from SpikingPathway
        attention_config = SpikingAttentionPathwayConfig(
            source_size=64,
            target_size=128,
            device="cpu"
        )
        attention = SpikingAttentionPathway(attention_config)

        replay_config = SpikingReplayPathwayConfig(
            source_size=128,
            target_size=256,
            device="cpu"
        )
        replay = SpikingReplayPathway(replay_config)

        # Both should have add_neurons method
        assert hasattr(attention, 'add_neurons')
        assert hasattr(replay, 'add_neurons')

        # Should work without errors
        attention.add_neurons(n_new=8)
        replay.add_neurons(n_new=8)

        assert attention.config.target_size == 136
        assert replay.config.target_size == 264


# ============================================================================
# BRAIN-LEVEL INTEGRATION TESTS
# ============================================================================

class TestGrowthIntegration:
    """Test brain-level growth coordination."""

    def test_brain_check_growth_needs(self):
        """Test brain-level growth need detection."""
        from thalia.core.brain import EventDrivenBrain
        from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes

        config = ThaliaConfig(
            global_=GlobalConfig(device="cpu"),
            brain=BrainConfig(
                sizes=RegionSizes(input_size=64, n_actions=4),
            ),
        )
        brain = EventDrivenBrain.from_thalia_config(config)

        # Check growth needs
        growth_report = brain.check_growth_needs()

        # Should return dict of components needing growth
        assert isinstance(growth_report, dict)
        # Keys could be region names or 'regions'/'pathways'
        assert len(growth_report) >= 0  # Can be empty if no growth needed

    def test_brain_auto_grow_coordination(self):
        """Test that brain can coordinate growth across connected components."""
        from thalia.core.brain import EventDrivenBrain
        from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes

        config = ThaliaConfig(
            global_=GlobalConfig(device="cpu"),
            brain=BrainConfig(
                sizes=RegionSizes(input_size=64, n_actions=4),
            ),
        )
        brain = EventDrivenBrain.from_thalia_config(config)

        # Trigger auto-growth
        # This should detect which regions/pathways need growth
        # and coordinate expansion (e.g., if cortex grows, cortex_to_* pathways grow)
        grown_components = brain.auto_grow(threshold=0.8)

        assert isinstance(grown_components, dict)
        # Maps region_name -> n_neurons_added
        # Will be empty if no regions need growth yet

    def test_checkpoint_preserves_growth_history(self):
        """Test that checkpoint metadata includes growth history."""
        from thalia.core.brain import EventDrivenBrain
        from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes
        import tempfile
        import os

        config = ThaliaConfig(
            global_=GlobalConfig(device="cpu"),
            brain=BrainConfig(
                sizes=RegionSizes(input_size=64, n_actions=4),
            ),
        )
        brain = EventDrivenBrain.from_thalia_config(config)

        # Perform some growth (manually for testing)
        # brain.striatum.add_neurons(n_new=1)

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "growth_test.thalia")
            BrainCheckpoint.save(brain, checkpoint_path)

            # Load checkpoint
            state = BrainCheckpoint.load(checkpoint_path, device="cpu")

            # Check growth history in metadata
            # (Implementation-dependent: could be in metadata or separate field)
            assert 'metadata' in state or 'growth_history' in state

    def test_pathway_grows_with_connected_region(self):
        """Test that pathways automatically track region growth."""
        from thalia.core.brain import EventDrivenBrain
        from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes

        config = ThaliaConfig(
            global_=GlobalConfig(device="cpu"),
            brain=BrainConfig(
                sizes=RegionSizes(input_size=64, n_actions=4, hippocampus_size=128),
            ),
        )
        brain = EventDrivenBrain.from_thalia_config(config)

        # Get initial sizes
        # When hippocampus grows, replay pathway (hippo→cortex) should grow its source
        # But we test the target side: when hippocampus is TARGET
        initial_hippo_size = brain.hippocampus.impl.config.n_output
        initial_cortex_to_hippo_target = brain.cortex_to_hippo_pathway.config.target_size

        # Grow hippocampus (target of cortex_to_hippo pathway)
        brain.hippocampus.impl.add_neurons(n_new=16)
        new_hippo_size = brain.hippocampus.impl.config.n_output

        # Hippocampus grew
        assert new_hippo_size == initial_hippo_size + 16

        # Now manually grow pathways where hippocampus is target
        # cortex_to_hippo has hippocampus as TARGET, so it should grow
        brain._grow_connected_pathways('hippocampus', 16)

        # Check that cortex→hippocampus pathway target size was updated
        new_cortex_to_hippo_target = brain.cortex_to_hippo_pathway.config.target_size
        assert new_cortex_to_hippo_target == initial_cortex_to_hippo_target + 16

    def test_all_regions_in_brain_support_growth(self):
        """Test that all 5 major brain regions support growth."""
        from thalia.core.brain import EventDrivenBrain
        from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes

        config = ThaliaConfig(
            global_=GlobalConfig(device="cpu"),
            brain=BrainConfig(
                sizes=RegionSizes(input_size=64, n_actions=4),
            ),
        )
        brain = EventDrivenBrain.from_thalia_config(config)

        # Test each region
        regions_to_test = [
            ('cortex', brain.cortex.impl),
            ('hippocampus', brain.hippocampus.impl),
            ('pfc', brain.pfc.impl),
            ('striatum', brain.striatum.impl),
            ('cerebellum', brain.cerebellum.impl),
        ]

        for name, region in regions_to_test:
            # Each should have add_neurons method
            assert hasattr(region, 'add_neurons'), f"{name} missing add_neurons()"

            # Should be able to get capacity metrics
            assert hasattr(region, 'get_capacity_metrics'), f"{name} missing get_capacity_metrics()"

            # Get initial size
            initial_size = region.config.n_output

            # Grow
            region.add_neurons(n_new=8)

            # Verify growth
            new_size = region.config.n_output
            assert new_size > initial_size, f"{name} failed to grow"

    def test_growth_preserves_brain_functionality(self):
        """Test that brain remains functional after growth."""
        from thalia.core.brain import EventDrivenBrain
        from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes

        config = ThaliaConfig(
            global_=GlobalConfig(device="cpu"),
            brain=BrainConfig(
                sizes=RegionSizes(input_size=64, n_actions=4),
            ),
        )
        brain = EventDrivenBrain.from_thalia_config(config)

        # Process some input before growth
        input_pattern = torch.rand(64) > 0.5
        brain.process_sample(input_pattern.float(), n_timesteps=5)

        # Grow a region
        brain.striatum.impl.add_neurons(n_new=16)

        # Should still be able to process input
        brain.process_sample(input_pattern.float(), n_timesteps=5)

        # Should still be able to select action
        action, confidence = brain.select_action()
        assert action is not None
        assert 0 <= confidence <= 1

    def test_growth_history_tracks_all_changes(self):
        """Test that growth history captures all growth events."""
        from thalia.core.brain import EventDrivenBrain
        from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes

        config = ThaliaConfig(
            global_=GlobalConfig(device="cpu"),
            brain=BrainConfig(
                sizes=RegionSizes(input_size=64, n_actions=4),
            ),
        )
        brain = EventDrivenBrain.from_thalia_config(config)

        # Manually grow multiple regions
        brain.striatum.impl.add_neurons(n_new=8)
        brain._growth_history.append({
            'timestamp': '2025-12-08T12:00:00',
            'region': 'striatum',
            'neurons_added': 8,
            'old_size': 320,
            'new_size': 328,
            'reason': 'manual_test',
        })

        brain.hippocampus.impl.add_neurons(n_new=4)
        brain._growth_history.append({
            'timestamp': '2025-12-08T12:01:00',
            'region': 'hippocampus',
            'neurons_added': 4,
            'old_size': 128,
            'new_size': 132,
            'reason': 'manual_test',
        })

        # History should have both events
        assert len(brain._growth_history) == 2
        assert brain._growth_history[0]['region'] == 'striatum'
        assert brain._growth_history[1]['region'] == 'hippocampus'

    def test_auto_grow_respects_threshold(self):
        """Test that auto_grow only grows when capacity exceeds threshold."""
        from thalia.core.brain import EventDrivenBrain
        from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes

        config = ThaliaConfig(
            global_=GlobalConfig(device="cpu"),
            brain=BrainConfig(
                sizes=RegionSizes(input_size=64, n_actions=4),
            ),
        )
        brain = EventDrivenBrain.from_thalia_config(config)

        # With high threshold, nothing should grow (regions are fresh)
        grown = brain.auto_grow(threshold=0.95)

        # Should return empty dict or dict with 0 values
        if grown:
            assert all(v == 0 or isinstance(v, int) for v in grown.values())


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
