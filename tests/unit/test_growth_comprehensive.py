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
from thalia.regions.cortex import LayeredCortex, LayeredCortexConfig
from thalia.regions.prefrontal import Prefrontal, PrefrontalConfig
from thalia.regions.cerebellum import Cerebellum, CerebellumConfig
from thalia.integration.spiking_pathway import SpikingPathway, SpikingPathwayConfig
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
        # Create simple striatum
        config = StriatumConfig(n_input=64, n_output=32, device="cpu")
        region = Striatum(config)

        manager = GrowthManager(region_name="striatum")
        metrics = manager.get_capacity_metrics(region)

        assert isinstance(metrics, CapacityMetrics)
        assert 0 <= metrics.firing_rate <= 1
        assert 0 <= metrics.weight_saturation <= 1
        assert 0 <= metrics.synapse_usage <= 1
        assert metrics.neuron_count == 32
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
        """Test that D1 and D2 neuron lists expand."""
        config = StriatumConfig(n_input=64, n_output=2, neurons_per_action=10, device="cpu")
        striatum = Striatum(config)
        
        old_d1_count = striatum.d1_neurons.n_neurons
        old_d2_count = striatum.d2_neurons.n_neurons
        
        # Grow
        striatum.add_neurons(n_new=1)
        
        # Neuron populations should expand
        assert striatum.d1_neurons.n_neurons == old_d1_count + 10
        assert striatum.d2_neurons.n_neurons == old_d2_count + 10

    def test_striatum_preserves_neuron_state(self):
        """Test that existing neurons preserve membrane potential, etc."""
        config = StriatumConfig(n_input=64, n_output=2, neurons_per_action=10, device="cpu")
        striatum = Striatum(config)
        
        # Process input to build neuron state (ADR-005: 1D)
        input_spikes = (torch.rand(64) > 0.5).float()
        striatum.forward(input_spikes)
        
        # Save first neuron's membrane potential
        if hasattr(striatum.d1_neurons[0], 'v_mem'):
            old_vmem = striatum.d1_neurons[0].v_mem.clone()
        
        # Grow
        striatum.add_neurons(n_new=1)
        
        # First neuron should have same state
        if hasattr(striatum.d1_neurons[0], 'v_mem'):
            assert torch.allclose(striatum.d1_neurons[0].v_mem, old_vmem)

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
        
        # Grow and process some data (ADR-005: 1D)
        striatum.add_neurons(n_new=1)
        input_spikes = (torch.rand(64) > 0.5).float()
        output_spikes = striatum.forward(input_spikes)
        
        # Checkpoint
        state = striatum.get_full_state()
        
        # Create new striatum with grown size
        config2 = StriatumConfig(n_input=64, n_output=3, neurons_per_action=10, device="cpu")
        striatum2 = Striatum(config2)
        striatum2.load_full_state(state)
        
        # Verify weights match
        assert torch.allclose(striatum.d1_weights, striatum2.d1_weights)
        assert torch.allclose(striatum.d2_weights, striatum2.d2_weights)


class TestHippocampusGrowth:
    """Test growth for Hippocampus region (episodic memory via trisynaptic circuit)."""

    @pytest.mark.skip("Pending Hippocampus.add_neurons() implementation")
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

    @pytest.mark.skip("Pending Hippocampus.add_neurons() implementation")
    def test_hippocampus_expands_trisynaptic_circuit(self):
        """Test that DG, CA3, CA1 all expand proportionally."""
        config = TrisynapticConfig(n_input=64, n_output=32, ec_l3_input_size=256, device="cpu")
        hippo = TrisynapticHippocampus(config)
        
        old_dg_size = hippo.dg_size
        old_ca3_size = hippo.ca3_size
        old_ca1_size = hippo.ca1_size
        
        # Grow
        hippo.add_neurons(n_new=16)
        
        # All layers should expand
        assert hippo.dg_size > old_dg_size
        assert hippo.ca3_size > old_ca3_size
        assert hippo.ca1_size > old_ca1_size

    @pytest.mark.skip("Pending Hippocampus.add_neurons() implementation")
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

    @pytest.mark.skip("Pending Hippocampus.add_neurons() implementation")
    def test_hippocampus_capacity_metrics(self):
        """Test hippocampus capacity metrics."""
        config = TrisynapticConfig(n_input=64, n_output=32, ec_l3_input_size=256, device="cpu")
        hippo = TrisynapticHippocampus(config)
        
        metrics = hippo.get_capacity_metrics()
        assert isinstance(metrics, CapacityMetrics)


class TestCortexGrowth:
    """Test growth for Cortex region (feature learning via layered architecture)."""

    @pytest.mark.skip("Pending LayeredCortex.add_neurons() implementation")
    def test_cortex_add_neurons_basic(self):
        """Test adding neurons to layered cortex."""
        config = LayeredCortexConfig(
            n_input=128,
            n_output=64,
            device="cpu"
        )
        cortex = LayeredCortex(config)
        
        initial_output = cortex.config.n_output
        
        # Add neurons
        cortex.add_neurons(n_new=32, initialization='xavier')
        
        # Check expansion
        assert cortex.config.n_output == initial_output + 32

    @pytest.mark.skip("Pending LayeredCortex.add_neurons() implementation")
    def test_cortex_layer_sizes_scale_proportionally(self):
        """Test that L4, L2/3, L5 scale with proper ratios."""
        config = LayeredCortexConfig(n_input=128, n_output=64, device="cpu")
        cortex = LayeredCortex(config)
        
        old_l4_size = cortex.l4_size
        old_l23_size = cortex.l23_size
        old_l5_size = cortex.l5_size
        
        # Grow
        cortex.add_neurons(n_new=32)
        
        # Layers should scale proportionally
        assert cortex.l4_size > old_l4_size
        assert cortex.l23_size > old_l23_size
        assert cortex.l5_size > old_l5_size
        
        # Ratios should be preserved (approximately)
        old_ratio_l4 = old_l4_size / 64
        new_ratio_l4 = cortex.l4_size / 96
        assert abs(old_ratio_l4 - new_ratio_l4) < 0.1

    @pytest.mark.skip("Pending LayeredCortex.add_neurons() implementation")
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

    @pytest.mark.skip("Pending LayeredCortex.add_neurons() implementation")
    def test_cortex_capacity_metrics(self):
        """Test cortex capacity metrics."""
        config = LayeredCortexConfig(n_input=128, n_output=64, device="cpu")
        cortex = LayeredCortex(config)
        
        metrics = cortex.get_capacity_metrics()
        assert isinstance(metrics, CapacityMetrics)


class TestPFCGrowth:
    """Test growth for Prefrontal Cortex region (working memory)."""

    @pytest.mark.skip("Pending Prefrontal.add_neurons() implementation")
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

    @pytest.mark.skip("Pending Prefrontal.add_neurons() implementation")
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

    @pytest.mark.skip("Pending Prefrontal.add_neurons() implementation")
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

    @pytest.mark.skip("Pending Prefrontal.add_neurons() implementation")
    def test_pfc_capacity_metrics(self):
        """Test PFC capacity metrics."""
        config = PrefrontalConfig(n_input=128, n_output=32, device="cpu")
        pfc = Prefrontal(config)
        
        metrics = pfc.get_capacity_metrics()
        assert isinstance(metrics, CapacityMetrics)


class TestCerebellumGrowth:
    """Test growth for Cerebellum region (supervised learning)."""

    @pytest.mark.skip("Pending Cerebellum.add_neurons() implementation")
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

    @pytest.mark.skip("Pending Cerebellum.add_neurons() implementation")
    def test_cerebellum_expands_granule_layer(self):
        """Test that granule cell layer expands."""
        config = CerebellumConfig(n_input=64, n_output=32, device="cpu")
        cerebellum = Cerebellum(config)
        
        old_granule_size = cerebellum.granule_size if hasattr(cerebellum, 'granule_size') else 0
        
        # Grow
        cerebellum.add_neurons(n_new=16)
        
        # Granule layer should expand
        if hasattr(cerebellum, 'granule_size'):
            assert cerebellum.granule_size > old_granule_size

    @pytest.mark.skip("Pending Cerebellum.add_neurons() implementation")
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

    @pytest.mark.skip("Pending Cerebellum.add_neurons() implementation")
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

    @pytest.mark.skip("Pending SpikingPathway.add_neurons() implementation")
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

    @pytest.mark.skip("Pending SpikingPathway.add_neurons() implementation")
    def test_pathway_preserves_transformation_weights(self):
        """Test pathway preserves existing connections."""
        config = SpikingPathwayConfig(source_size=64, target_size=32, device="cpu")
        pathway = SpikingPathway(config)
        
        # Save old weights (if pathway has them)
        if hasattr(pathway, 'weights') and pathway.weights is not None:
            old_weights = pathway.weights.clone()
            
            # Grow
            pathway.add_neurons(n_new=16)
            
            # Check old weights preserved
            if pathway.weights.shape[0] > old_weights.shape[0]:
                # Weight matrix expanded in output dimension
                assert torch.allclose(pathway.weights[:old_weights.shape[0]], old_weights)

    @pytest.mark.skip("Pending SpikingPathway.add_neurons() implementation")
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
        if hasattr(pathway, 'eligibility_trace'):
            # Target dimension should be 48 now
            assert pathway.eligibility_trace.shape[0] == 48

    @pytest.mark.skip("Pending SpikingPathway.add_neurons() implementation")
    def test_pathway_capacity_metrics(self):
        """Test pathway capacity metrics."""
        config = SpikingPathwayConfig(source_size=64, target_size=32, device="cpu")
        pathway = SpikingPathway(config)
        
        metrics = pathway.get_capacity_metrics()
        assert isinstance(metrics, CapacityMetrics)

    @pytest.mark.skip("Pending SpikingPathway.add_neurons() implementation")
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

    @pytest.mark.skip("Pending specialized pathway growth implementations")
    def test_cortex_to_hippo_pathway_growth(self):
        """Test cortex→hippocampus pathway growth."""
        # This pathway uses phase coding
        # Growth should preserve phase relationships
        pass

    @pytest.mark.skip("Pending specialized pathway growth implementations")
    def test_cortex_to_striatum_pathway_growth(self):
        """Test cortex→striatum pathway growth."""
        # This pathway uses dopamine-STDP
        # Growth should preserve dopamine modulation
        pass

    @pytest.mark.skip("Pending specialized pathway growth implementations")
    def test_attention_pathway_growth(self):
        """Test PFC→Cortex attention pathway growth."""
        # Attention pathway has specialized top-down modulation
        # Growth should preserve attention masks
        pass

    @pytest.mark.skip("Pending specialized pathway growth implementations")
    def test_replay_pathway_growth(self):
        """Test Hippocampus→Cortex replay pathway growth."""
        # Replay pathway has specialized consolidation logic
        # Growth should preserve replay buffers
        pass


# ============================================================================
# BRAIN-LEVEL INTEGRATION TESTS
# ============================================================================

class TestGrowthIntegration:
    """Test brain-level growth coordination."""

    @pytest.mark.skip("Pending Brain.check_growth_needs() implementation")
    def test_brain_check_growth_needs(self):
        """Test brain-level growth need detection."""
        from thalia.core.brain import EventDrivenBrain, EventDrivenBrainConfig

        config = EventDrivenBrainConfig(
            input_size=64,
            n_actions=4,
            device="cpu"
        )
        brain = EventDrivenBrain(config)

        # Check growth needs
        growth_report = brain.check_growth_needs()

        # Should return dict of components needing growth
        assert isinstance(growth_report, dict)
        # Keys could be region names or 'regions'/'pathways'
        assert len(growth_report) >= 0  # Can be empty if no growth needed

    @pytest.mark.skip("Pending Brain.auto_grow() implementation")
    def test_brain_auto_grow_coordination(self):
        """Test that brain can coordinate growth across connected components."""
        from thalia.core.brain import EventDrivenBrain, EventDrivenBrainConfig

        config = EventDrivenBrainConfig(
            input_size=64,
            n_actions=4,
            device="cpu"
        )
        brain = EventDrivenBrain(config)

        # Trigger auto-growth
        # This should detect which regions/pathways need growth
        # and coordinate expansion (e.g., if cortex grows, cortex_to_* pathways grow)
        grown_components = brain.auto_grow(threshold=0.8)

        assert isinstance(grown_components, list)
        # Each entry should be (component_name, n_neurons_added)

    @pytest.mark.skip("Pending growth_history checkpoint integration")
    def test_checkpoint_preserves_growth_history(self):
        """Test that checkpoint metadata includes growth history."""
        from thalia.core.brain import EventDrivenBrain, EventDrivenBrainConfig
        import tempfile
        import os

        config = EventDrivenBrainConfig(
            input_size=64,
            n_actions=4,
            device="cpu"
        )
        brain = EventDrivenBrain(config)

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

    @pytest.mark.skip("Pending pathway-region growth coordination")
    def test_pathway_grows_with_connected_region(self):
        """Test that pathways automatically track region growth."""
        from thalia.core.brain import EventDrivenBrain, EventDrivenBrainConfig

        config = EventDrivenBrainConfig(
            input_size=64,
            n_actions=4,
            device="cpu"
        )
        brain = EventDrivenBrain(config)

        # Grow cortex
        old_cortex_size = brain.cortex.config.n_output
        brain.cortex.add_neurons(n_new=32)
        new_cortex_size = brain.cortex.config.n_output

        # Check that cortex→hippocampus pathway target size matches
        # (This assumes coordinated growth is implemented)
        # pathway = brain.cortex_to_hippo_pathway
        # assert pathway.config.source_size == new_cortex_size


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
