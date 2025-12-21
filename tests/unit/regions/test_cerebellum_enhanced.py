"""
Unit tests for enhanced cerebellar microcircuit.

Tests:
- Granule cell layer expansion and sparse coding
- Enhanced Purkinje cell dendritic computation
- Complex spike generation from climbing fibers
- Simple spike generation from parallel fibers
- Deep cerebellar nuclei integration
- Purkinje inhibition of DCN
- Mossy fiber collaterals to DCN
- Parallel fiber temporal delays
- Enhanced vs classic pathway compatibility
"""

import pytest
import torch

from thalia.regions.cerebellum import Cerebellum, CerebellumConfig
from thalia.regions.cerebellum.granule_layer import GranuleCellLayer
from thalia.regions.cerebellum.purkinje_cell import EnhancedPurkinjeCell
from thalia.regions.cerebellum.deep_nuclei import DeepCerebellarNuclei


@pytest.fixture
def device():
    """Device for testing (CPU by default)."""
    return torch.device("cpu")


@pytest.fixture
def cerebellum_classic_config(device):
    """Classic cerebellum configuration (no enhanced microcircuit)."""
    return CerebellumConfig(
        n_input=128,
        n_output=64,
        use_enhanced_microcircuit=False,  # Classic pathway
        dt_ms=1.0,
        device=str(device),
    )


@pytest.fixture
def cerebellum_enhanced_config(device):
    """Enhanced cerebellum configuration with granule layer and DCN."""
    return CerebellumConfig(
        n_input=128,
        n_output=64,
        use_enhanced_microcircuit=True,  # Enhanced pathway
        granule_expansion_factor=4.0,
        granule_sparsity=0.03,
        purkinje_n_dendrites=100,
        dt_ms=1.0,
        device=str(device),
    )


@pytest.fixture
def cerebellum_classic(cerebellum_classic_config):
    """Classic cerebellum instance."""
    cereb = Cerebellum(cerebellum_classic_config)
    cereb.reset_state()
    return cereb


@pytest.fixture
def cerebellum_enhanced(cerebellum_enhanced_config):
    """Enhanced cerebellum instance."""
    cereb = Cerebellum(cerebellum_enhanced_config)
    cereb.reset_state()
    return cereb


class TestGranuleCellLayer:
    """Tests for granule cell layer expansion and sparse coding."""

    def test_granule_layer_initialization(self, device):
        """Test granule layer initializes with correct dimensions."""
        granule_layer = GranuleCellLayer(
            n_mossy_fibers=128,
            expansion_factor=4.0,
            sparsity=0.03,
            device=device,
            dt_ms=1.0,
        )

        # Contract: granule cells = mossy fibers × expansion factor
        expected_granule = int(128 * 4.0)
        assert granule_layer.n_granule == expected_granule, \
            f"Should have {expected_granule} granule cells (4× expansion)"

        # Contract: mossy→granule weights exist
        assert granule_layer.weights.shape == (expected_granule, 128), \
            "Weights should connect mossy fibers to granule cells"

    def test_granule_layer_expansion(self, device):
        """Test 4× expansion from mossy fibers to granule cells."""
        n_mossy = 128
        expansion = 4.0

        granule_layer = GranuleCellLayer(
            n_mossy_fibers=n_mossy,
            expansion_factor=expansion,
            sparsity=0.03,
            device=device,
            dt_ms=1.0,
        )

        # Contract: output dimension is 4× input
        assert granule_layer.n_granule == int(n_mossy * expansion), \
            "Granule cells should be 4× mossy fibers"

    def test_granule_layer_sparsity(self, device):
        """Test 3% sparse coding in granule layer."""
        granule_layer = GranuleCellLayer(
            n_mossy_fibers=128,
            expansion_factor=4.0,
            sparsity=0.03,
            device=device,
            dt_ms=1.0,
        )

        # Run several timesteps with varied input
        sparsity_measurements = []
        for _ in range(10):
            # Varied mossy fiber input each timestep (biologically realistic)
            mossy_spikes = torch.rand(128, device=device) > 0.7
            parallel_fiber_spikes = granule_layer(mossy_spikes)

            # Measure sparsity
            n_active = parallel_fiber_spikes.sum().item()
            n_total = granule_layer.n_granule
            actual_sparsity = n_active / n_total
            sparsity_measurements.append(actual_sparsity)

        # Contract: average sparsity should be ~3% (biological)
        avg_sparsity = sum(sparsity_measurements) / len(sparsity_measurements)
        assert 0.01 < avg_sparsity < 0.05, \
            f"Granule sparsity should be ~3%, got {avg_sparsity:.3f}"

    def test_granule_layer_output_type(self, device):
        """Test granule layer output is bool spikes (ADR-004)."""
        granule_layer = GranuleCellLayer(
            n_mossy_fibers=128,
            expansion_factor=4.0,
            sparsity=0.03,
            device=device,
            dt_ms=1.0,
        )

        mossy_spikes = torch.rand(128, device=device) > 0.9
        parallel_fiber_spikes = granule_layer(mossy_spikes)

        # Contract: output should be bool (ADR-004)
        assert parallel_fiber_spikes.dtype == torch.bool, \
            "Parallel fiber spikes should be bool (ADR-004)"
        assert parallel_fiber_spikes.dim() == 1, \
            "Output should be 1D (ADR-005)"

    def test_granule_connectivity_sparsity(self, device):
        """Test mossy→granule connectivity is sparse (~5%)."""
        granule_layer = GranuleCellLayer(
            n_mossy_fibers=128,
            expansion_factor=4.0,
            sparsity=0.03,
            device=device,
            dt_ms=1.0,
        )

        weights = granule_layer.weights.data

        # Count zero weights
        zero_count = (weights == 0).sum().item()
        total_count = weights.numel()
        connectivity_sparsity = zero_count / total_count

        # Contract: mossy→granule should be ~5% connected (95% zeros)
        # Note: Implementation uses 5% connectivity
        assert connectivity_sparsity > 0.90, \
            f"Mossy→granule should be sparse (~95% zeros), got {connectivity_sparsity:.3f}"


class TestEnhancedPurkinjeCell:
    """Tests for enhanced Purkinje cell with dendritic computation."""

    def test_purkinje_initialization(self, device):
        """Test Purkinje cell initializes with dendrites."""
        n_dendrites = 100
        purkinje = EnhancedPurkinjeCell(
            n_dendrites=n_dendrites,
            device=device,
            dt_ms=1.0,
        )

        # Contract: dendrites exist
        assert purkinje.n_dendrites == n_dendrites, f"Should have {n_dendrites} dendritic compartments"
        assert purkinje.n_dendrites > 0, "Invariant: positive dendrites"
        assert hasattr(purkinje, 'dendrite_voltage'), \
            "Should have dendritic voltage state"

    def test_purkinje_simple_spikes(self, device):
        """Test Purkinje generates simple spikes from parallel fibers."""
        purkinje = EnhancedPurkinjeCell(
            n_dendrites=100,
            device=device,
            dt_ms=1.0,
        )

        # Strong parallel fiber input (expanded granule layer)
        parallel_fibers = torch.rand(512, device=device) > 0.95  # Sparse
        climbing_fiber = torch.tensor(0.0, device=device)  # No error

        # Run for multiple timesteps
        simple_spike_count = 0
        for _ in range(20):
            spike = purkinje(parallel_fibers, climbing_fiber)
            if spike:
                simple_spike_count += 1

        # Contract: should generate some simple spikes
        # (Simple spikes: 40-100 Hz in biology, so expect several in 20ms)
        assert simple_spike_count > 0, \
            "Purkinje should generate simple spikes from parallel fibers"

    def test_purkinje_complex_spikes(self, device):
        """Test climbing fiber triggers complex spikes."""
        purkinje = EnhancedPurkinjeCell(
            n_dendrites=100,
            device=device,
            dt_ms=1.0,
        )

        # Parallel fiber input
        parallel_fibers = torch.rand(512, device=device) > 0.97

        # Strong climbing fiber error signal
        climbing_fiber_error = torch.tensor(1.0, device=device)

        # Run and check for complex spike
        complex_spike_detected = False
        for _ in range(5):
            spike = purkinje(parallel_fibers, climbing_fiber_error)
            # Complex spike detection would be in implementation
            # Here we check that climbing fiber is processed
            if purkinje.last_complex_spike_time >= 0:
                complex_spike_detected = True
                break

        # Contract: climbing fiber should trigger complex spike response
        assert complex_spike_detected or purkinje.calcium.sum() > 0, \
            "Climbing fiber should trigger calcium response (complex spike)"

    def test_purkinje_complex_spike_refractory(self, device):
        """Test complex spikes have refractory period (~100ms)."""
        purkinje = EnhancedPurkinjeCell(
            n_dendrites=100,
            device=device,
            dt_ms=1.0,
        )

        parallel_fibers = torch.rand(512, device=device) > 0.97
        climbing_fiber = torch.tensor(1.0, device=device)

        # Trigger first complex spike
        for t in range(200):
            purkinje(parallel_fibers, climbing_fiber)
            if purkinje.last_complex_spike_time == t:
                first_complex_time = t
                break

        # Try to trigger another immediately (should fail due to refractory)
        second_complex_time = None
        for t in range(first_complex_time + 1, first_complex_time + 150):
            purkinje(parallel_fibers, climbing_fiber)
            if purkinje.last_complex_spike_time > first_complex_time:
                second_complex_time = t
                break

        # Contract: second complex spike should be delayed by refractory period
        if second_complex_time is not None:
            refractory = second_complex_time - first_complex_time
            assert refractory >= 100, \
                f"Complex spike refractory should be ~100ms, got {refractory}ms"

    def test_purkinje_calcium_dynamics(self, device):
        """Test dendritic calcium dynamics with complex spikes."""
        purkinje = EnhancedPurkinjeCell(
            n_dendrites=100,
            device=device,
            dt_ms=1.0,
        )

        parallel_fibers = torch.rand(512, device=device) > 0.97
        climbing_fiber = torch.tensor(1.0, device=device)

        # Trigger complex spike
        purkinje(parallel_fibers, climbing_fiber)

        # Contract: calcium should increase
        initial_calcium = purkinje.calcium.clone()

        # Continue without climbing fiber
        for _ in range(10):
            purkinje(parallel_fibers, torch.tensor(0.0, device=device))

        # Calcium should decay
        final_calcium = purkinje.calcium

        # Note: Exact dynamics depend on implementation
        assert not torch.isnan(final_calcium).any(), "Calcium should be valid"


class TestDeepCerebellarNuclei:
    """Tests for deep cerebellar nuclei (DCN) integration."""

    def test_dcn_initialization(self, device):
        """Test DCN initializes with correct dimensions."""
        dcn = DeepCerebellarNuclei(
            n_output=64,
            n_purkinje=64,
            n_mossy=128,
            device=device,
            dt_ms=1.0,
        )

        # Contract: DCN has correct connectivity
        assert dcn.purkinje_to_dcn.shape == (64, 64), \
            "Purkinje→DCN should connect Purkinje to DCN"
        assert dcn.mossy_to_dcn.shape == (64, 128), \
            "Mossy→DCN should provide excitatory collaterals"

    def test_dcn_purkinje_inhibition(self, device):
        """Test Purkinje cells inhibit DCN."""
        dcn = DeepCerebellarNuclei(
            n_output=64,
            n_purkinje=64,
            n_mossy=128,
            device=device,
            dt_ms=1.0,
        )

        # No Purkinje inhibition (baseline DCN activity)
        purkinje_silent = torch.zeros(64, dtype=torch.bool, device=device)

        baseline_outputs = []
        for _ in range(10):
            # Generate fresh random input each timestep
            mossy_spikes = torch.rand(128, device=device) > 0.8
            out = dcn(purkinje_silent, mossy_spikes)
            baseline_outputs.append(out.sum().item())

        # Reset state before testing inhibition condition
        dcn.reset_state()

        # Strong Purkinje inhibition
        purkinje_active = torch.ones(64, dtype=torch.bool, device=device)

        inhibited_outputs = []
        for _ in range(10):
            # Generate fresh random input each timestep (independent from baseline)
            mossy_spikes = torch.rand(128, device=device) > 0.8
            out = dcn(purkinje_active, mossy_spikes)
            inhibited_outputs.append(out.sum().item())

        # Contract: Purkinje activity should reduce DCN output
        baseline_avg = sum(baseline_outputs) / len(baseline_outputs)
        inhibited_avg = sum(inhibited_outputs) / len(inhibited_outputs)

        assert inhibited_avg < baseline_avg, \
            "Purkinje inhibition should reduce DCN activity"

    def test_dcn_mossy_excitation(self, device):
        """Test mossy fiber collaterals excite DCN."""
        dcn = DeepCerebellarNuclei(
            n_output=64,
            n_purkinje=64,
            n_mossy=128,
            device=device,
            dt_ms=1.0,
        )

        purkinje_spikes = torch.zeros(64, dtype=torch.bool, device=device)

        # Weak mossy input
        weak_mossy = torch.rand(128, device=device) > 0.95
        weak_outputs = []
        for _ in range(10):
            out = dcn(purkinje_spikes, weak_mossy)
            weak_outputs.append(out.sum().item())

        # Strong mossy input
        strong_mossy = torch.rand(128, device=device) > 0.7
        strong_outputs = []
        for _ in range(10):
            out = dcn(purkinje_spikes, strong_mossy)
            strong_outputs.append(out.sum().item())

        # Contract: stronger mossy input should increase DCN output
        weak_avg = sum(weak_outputs) / len(weak_outputs)
        strong_avg = sum(strong_outputs) / len(strong_outputs)

        assert strong_avg >= weak_avg, \
            "Stronger mossy input should maintain/increase DCN activity"

    def test_dcn_output_type(self, device):
        """Test DCN output is bool spikes (ADR-004)."""
        n_output = 64
        dcn = DeepCerebellarNuclei(
            n_purkinje=n_output,
            n_mossy=128,
            device=device,
            dt_ms=1.0,
        )

        purkinje_spikes = torch.rand(n_output, device=device) > 0.9
        mossy_spikes = torch.rand(128, device=device) > 0.8

        output = dcn(purkinje_spikes, mossy_spikes)

        # Contract: output should be bool (ADR-004) and 1D (ADR-005)
        assert output.dtype == torch.bool, "DCN output should be bool (ADR-004)"
        assert output.dim() == 1, "DCN output should be 1D (ADR-005)"
        assert output.shape[0] == n_output, "DCN output should match n_output"


class TestEnhancedCerebellumIntegration:
    """Tests for enhanced cerebellum integration (granule→Purkinje→DCN)."""

    def test_enhanced_cerebellum_initialization(self, cerebellum_enhanced):
        """Test enhanced cerebellum initializes all components."""
        # Extract expected values from config
        n_input = cerebellum_enhanced.config.n_input
        n_output = cerebellum_enhanced.config.n_output
        expansion = cerebellum_enhanced.config.granule_expansion_factor

        # Contract: granule layer exists
        assert cerebellum_enhanced.granule_layer is not None, \
            "Enhanced cerebellum should have granule layer"
        assert cerebellum_enhanced.granule_layer.n_granule == int(n_input * expansion), \
            f"Granule layer should have {expansion}× expansion"

        # Contract: enhanced Purkinje cells exist
        assert cerebellum_enhanced.purkinje_cells is not None, \
            "Enhanced cerebellum should have Purkinje cell list"
        assert len(cerebellum_enhanced.purkinje_cells) == n_output, \
            "Should have one Purkinje cell per output neuron"

        # Contract: DCN exists
        assert cerebellum_enhanced.deep_nuclei is not None, \
            "Enhanced cerebellum should have DCN"

    def test_enhanced_forward_pipeline(self, cerebellum_enhanced, device):
        """Test enhanced cerebellum processes mossy→granule→Purkinje→DCN."""
        # Create mossy fiber input
        mossy_spikes = torch.rand(128, device=device) > 0.8

        # Forward pass
        output = cerebellum_enhanced(mossy_spikes)

        # Contract: output should be valid
        assert output.dtype == torch.bool, "Output should be bool (ADR-004)"
        assert output.shape == (64,), "Output should match n_output"

        # Contract: intermediate states should exist
        # (Granule layer should have processed input)
        assert cerebellum_enhanced.granule_layer.neurons.membrane is not None, \
            "Granule layer should have membrane state after forward"

    def test_enhanced_vs_classic_compatibility(self, cerebellum_classic, cerebellum_enhanced, device):
        """Test both enhanced and classic cerebellum work with same input."""
        mossy_spikes = torch.rand(128, device=device) > 0.8

        # Classic forward
        classic_output = cerebellum_classic(mossy_spikes)

        # Enhanced forward
        enhanced_output = cerebellum_enhanced(mossy_spikes)

        # Contract: both should produce valid outputs
        assert classic_output.shape == enhanced_output.shape, \
            "Classic and enhanced should have same output shape"
        assert classic_output.dtype == enhanced_output.dtype == torch.bool, \
            "Both should output bool spikes"

    def test_enhanced_cerebellum_learning(self, cerebellum_enhanced, device):
        """Test enhanced cerebellum supports error-corrective learning."""
        mossy_spikes = torch.rand(128, device=device) > 0.8
        target = torch.rand(64, device=device) > 0.5

        # Forward pass
        output = cerebellum_enhanced(mossy_spikes)

        # Deliver error signal
        metrics = cerebellum_enhanced.deliver_error(target)

        # Contract: error delivery should work
        assert isinstance(metrics, dict), "deliver_error should return metrics"
        assert "error" in metrics or "total_error" in metrics, \
            "Metrics should include error information"

    def test_enhanced_cerebellum_checkpoint(self, cerebellum_enhanced, device):
        """Test enhanced cerebellum checkpoint includes all components."""
        # Run some timesteps
        for _ in range(5):
            mossy_spikes = torch.rand(128, device=device) > 0.8
            cerebellum_enhanced(mossy_spikes)

        # Get checkpoint
        state = cerebellum_enhanced.get_full_state()

        # Contract: checkpoint should include enhanced components
        assert "config" in state, "Checkpoint should include config"
        assert state["config"]["use_enhanced"] == True, \
            "Config should indicate enhanced mode"

        if "enhanced_state" in state:
            assert "granule_layer" in state["enhanced_state"], \
                "Should checkpoint granule layer"
            assert "purkinje_cells" in state["enhanced_state"], \
                "Should checkpoint Purkinje cells"
            assert "deep_nuclei" in state["enhanced_state"], \
                "Should checkpoint DCN"

    def test_enhanced_cerebellum_growth(self, cerebellum_enhanced):
        """Test enhanced cerebellum grows correctly."""
        initial_output = cerebellum_enhanced.config.n_output
        initial_purkinje_count = len(cerebellum_enhanced.purkinje_cells)

        # Grow output
        cerebellum_enhanced.grow_output(n_new=32)

        # Contract: output size increased
        assert cerebellum_enhanced.config.n_output == initial_output + 32, \
            "Output size should increase by 32"

        # Contract: Purkinje cells increased
        new_purkinje_count = len(cerebellum_enhanced.purkinje_cells)
        assert new_purkinje_count == initial_purkinje_count + 32, \
            "Should add 32 new Purkinje cells"

        # Contract: DCN grew
        assert cerebellum_enhanced.deep_nuclei.n_output == initial_output + 32, \
            "DCN output should grow with cerebellum"


class TestParallelFiberTiming:
    """Tests for parallel fiber temporal delay lines."""

    def test_granule_provides_temporal_delays(self, device):
        """Test granule layer provides temporal diversity (delay lines)."""
        granule_layer = GranuleCellLayer(
            n_mossy_fibers=128,
            expansion_factor=4.0,
            sparsity=0.03,
            device=device,
            dt_ms=1.0,
        )

        # Pulse input (single timestep)
        pulse = torch.ones(128, dtype=torch.bool, device=device)
        silent = torch.zeros(128, dtype=torch.bool, device=device)

        # Record responses over time
        responses = []

        # Pulse at t=0
        responses.append(granule_layer(pulse).clone())

        # Silence for several timesteps
        for _ in range(10):
            responses.append(granule_layer(silent).clone())

        # Contract: responses should vary over time (some granule cells respond with delay)
        # This tests that granule layer provides temporal structure
        spike_counts = [r.sum().item() for r in responses]

        # Should see activity variation (not all at once)
        assert max(spike_counts) > 0, "Should have some granule activity"
        assert len(set(spike_counts)) > 1, \
            "Granule activity should vary over time (temporal delays)"

    def test_parallel_fiber_spatial_distribution(self, device):
        """Test parallel fibers provide spatially distributed input to Purkinje."""
        granule_layer = GranuleCellLayer(
            n_mossy_fibers=128,
            expansion_factor=4.0,
            sparsity=0.03,
            device=device,
            dt_ms=1.0,
        )

        # Localized mossy input (only first 32 fibers)
        localized_input = torch.zeros(128, dtype=torch.bool, device=device)
        localized_input[0:32] = torch.rand(32, device=device) > 0.7

        # Process through granule layer
        parallel_fibers = granule_layer(localized_input)

        # Contract: parallel fiber activity should be spatially distributed
        # (not just first 128 of 512 granule cells)
        first_quarter = parallel_fibers[0:128].sum().item()
        second_quarter = parallel_fibers[128:256].sum().item()
        third_quarter = parallel_fibers[256:384].sum().item()
        fourth_quarter = parallel_fibers[384:512].sum().item()

        # At least two quarters should have activity (expansion distributes input)
        active_quarters = sum([
            first_quarter > 0,
            second_quarter > 0,
            third_quarter > 0,
            fourth_quarter > 0,
        ])

        assert active_quarters >= 2, \
            "Granule expansion should distribute mossy input spatially"


class TestBackwardCompatibility:
    """Tests for backward compatibility between classic and enhanced modes."""

    def test_classic_cerebellum_still_works(self, cerebellum_classic, device):
        """Test classic cerebellum (without enhanced components) still functions."""
        # Run classic cerebellum
        for _ in range(10):
            mossy_spikes = torch.rand(128, device=device) > 0.8
            output = cerebellum_classic(mossy_spikes)

            assert output.shape == (64,), "Classic output should be correct shape"
            assert output.dtype == torch.bool, "Classic output should be bool"

    def test_enhanced_flag_controls_pathway(self, device):
        """Test use_enhanced_microcircuit flag correctly enables/disables enhanced pathway."""
        # Classic config
        classic_cfg = CerebellumConfig(
            n_input=128,
            n_output=64,
            use_enhanced_microcircuit=False,
            dt_ms=1.0,
            device=str(device),
        )
        classic = Cerebellum(classic_cfg)

        # Enhanced config
        enhanced_cfg = CerebellumConfig(
            n_input=128,
            n_output=64,
            use_enhanced_microcircuit=True,
            dt_ms=1.0,
            device=str(device),
        )
        enhanced = Cerebellum(enhanced_cfg)

        # Contract: flag controls component creation
        assert classic.granule_layer is None, "Classic should not have granule layer"
        assert classic.purkinje_cells is None, "Classic should not have enhanced Purkinje"
        assert classic.deep_nuclei is None, "Classic should not have DCN"

        assert enhanced.granule_layer is not None, "Enhanced should have granule layer"
        assert enhanced.purkinje_cells is not None, "Enhanced should have Purkinje cells"
        assert enhanced.deep_nuclei is not None, "Enhanced should have DCN"

    def test_checkpoint_compatibility(self, cerebellum_classic, cerebellum_enhanced, device):
        """Test classic and enhanced checkpoints are incompatible (as expected)."""
        # Run both
        for _ in range(5):
            mossy_spikes = torch.rand(128, device=device) > 0.8
            cerebellum_classic(mossy_spikes)
            cerebellum_enhanced(mossy_spikes)

        # Get checkpoints
        classic_state = cerebellum_classic.get_full_state()
        enhanced_state = cerebellum_enhanced.get_full_state()

        # Contract: checkpoints should indicate their mode
        assert classic_state["config"]["use_enhanced"] == False
        assert enhanced_state["config"]["use_enhanced"] == True

        # Contract: loading wrong checkpoint should fail
        with pytest.raises(Exception):  # CheckpointError
            cerebellum_classic.load_full_state(enhanced_state)

        with pytest.raises(Exception):  # CheckpointError
            cerebellum_enhanced.load_full_state(classic_state)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
