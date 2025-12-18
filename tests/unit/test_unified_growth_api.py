"""Test unified growth API (grow_input/grow_output) for all components.

This test suite verifies that:
1. All regions support grow_input() and grow_output()
2. All pathways support grow_input() and grow_output()
3. GrowthCoordinator properly propagates input growth to target regions
"""

import pytest
import torch
from thalia.regions.cortex.layered_cortex import LayeredCortex, LayeredCortexConfig
from thalia.regions.hippocampus.trisynaptic import TrisynapticHippocampus
from thalia.regions.hippocampus.config import HippocampusConfig
from thalia.regions.striatum.striatum import Striatum
from thalia.regions.striatum.config import StriatumConfig
from thalia.regions.thalamus import ThalamicRelay, ThalamicRelayConfig
from thalia.regions.prefrontal import Prefrontal, PrefrontalConfig
from thalia.regions.cerebellum import Cerebellum, CerebellumConfig
from thalia.regions.multisensory import MultimodalIntegration, MultimodalIntegrationConfig
from thalia.pathways.spiking_pathway import SpikingPathway
from thalia.pathways.spiking_replay import SpikingReplayPathway, SpikingReplayPathwayConfig
from thalia.pathways.attention.spiking_attention import SpikingAttentionPathway, SpikingAttentionPathwayConfig
from thalia.pathways.sensory_pathways import VisualPathway, AuditoryPathway, LanguagePathway
from thalia.core.base.component_config import PathwayConfig


@pytest.fixture
def cortex():
    """Create cortex for testing."""
    config = LayeredCortexConfig(
        n_input=64,
        n_output=160,  # Must equal l23_size + l5_size = 96 + 64
        l4_size=64,
        l23_size=96,
        l5_size=64,
        l6_size=32,
        device='cpu'
    )
    return LayeredCortex(config)


@pytest.fixture
def hippocampus():
    """Create hippocampus for testing."""
    config = HippocampusConfig(
        n_input=128,
        n_output=64,
        device='cpu'
    )
    return TrisynapticHippocampus(config)


@pytest.fixture
def striatum():
    """Create striatum for testing."""
    config = StriatumConfig(
        n_input=256,
        n_output=10,  # 10 actions
        device='cpu'
    )
    return Striatum(config)


@pytest.fixture
def pathway():
    """Create pathway for testing."""
    config = PathwayConfig(
        n_input=128,
        n_output=64,
        device='cpu'
    )
    return SpikingPathway(config)


@pytest.fixture
def thalamus():
    """Create thalamus for testing."""
    config = ThalamicRelayConfig(
        n_input=784,
        n_output=256,
        device='cpu'
    )
    return ThalamicRelay(config)


@pytest.fixture
def prefrontal():
    """Create prefrontal cortex for testing."""
    config = PrefrontalConfig(
        n_input=256,
        n_output=128,
        device='cpu'
    )
    return Prefrontal(config)


@pytest.fixture
def cerebellum():
    """Create cerebellum for testing."""
    config = CerebellumConfig(
        n_input=256,
        n_output=64,
        device='cpu'
    )
    return Cerebellum(config)


@pytest.fixture
def multisensory():
    """Create multisensory integration region for testing."""
    config = MultimodalIntegrationConfig(
        n_input=512,
        n_output=256,
        device='cpu'
    )
    return MultimodalIntegration(config)


@pytest.fixture
def replay_pathway():
    """Create replay pathway for testing."""
    config = SpikingReplayPathwayConfig(
        n_input=128,
        n_output=64,
        device='cpu'
    )
    return SpikingReplayPathway(config)


@pytest.fixture
def attention_pathway():
    """Create attention pathway for testing."""
    config = SpikingAttentionPathwayConfig(
        n_input=128,
        n_output=64,
        device='cpu'
    )
    return SpikingAttentionPathway(config)


@pytest.fixture
def visual_pathway():
    """Create visual pathway for testing."""
    return VisualPathway(n_input=784, n_output=256, device='cpu')


@pytest.fixture
def auditory_pathway():
    """Create auditory pathway for testing."""
    return AuditoryPathway(n_input=128, n_output=128, device='cpu')


@pytest.fixture
def language_pathway():
    """Create language pathway for testing."""
    return LanguagePathway(n_input=256, n_output=256, device='cpu')


class TestUnifiedGrowthAPI:
    """Test unified grow_input/grow_output API."""

    def test_pathway_grow_input_expands_input_dimension(self, pathway):
        """Test that pathway.grow_input() expands input dimension."""
        initial_input = pathway.config.n_input
        initial_output = pathway.config.n_output
        initial_weights = pathway.weights.data.clone()

        # Grow input by 20
        pathway.grow_input(n_new=20)

        # Verify input dimension expanded
        assert pathway.config.n_input == initial_input + 20
        assert pathway.config.n_output == initial_output  # Output unchanged

        # Verify weight matrix shape [output, input]
        assert pathway.weights.shape[0] == initial_output
        assert pathway.weights.shape[1] == initial_input + 20

        # Verify old weights preserved (left columns)
        assert torch.allclose(
            pathway.weights.data[:, :initial_input],
            initial_weights,
            atol=1e-6
        )

    def test_pathway_grow_output_expands_output_dimension(self, pathway):
        """Test that pathway.grow_output() expands output dimension."""
        initial_input = pathway.config.n_input
        initial_output = pathway.config.n_output
        initial_weights = pathway.weights.data.clone()

        # Grow output by 15
        pathway.grow_output(n_new=15)

        # Verify output dimension expanded
        assert pathway.config.n_output == initial_output + 15
        assert pathway.config.n_input == initial_input  # Input unchanged

        # Verify weight matrix shape [output, input]
        assert pathway.weights.shape[0] == initial_output + 15
        assert pathway.weights.shape[1] == initial_input

        # Verify old weights preserved (top rows)
        assert torch.allclose(
            pathway.weights.data[:initial_output, :],
            initial_weights,
            atol=1e-6
        )

    def test_cortex_grow_input_expands_input_weights(self, cortex):
        """Test that cortex.grow_input() expands input dimension."""
        initial_input = cortex.config.n_input
        initial_l4_size = cortex.l4_size
        initial_weights = cortex.w_input_l4.data.clone()

        # Grow input by 20
        cortex.grow_input(n_new=20)

        # Verify input dimension expanded
        assert cortex.config.n_input == initial_input + 20

        # Verify w_input_l4 expanded [l4, input] → [l4, input+20]
        assert cortex.w_input_l4.shape[0] == initial_l4_size  # Rows unchanged
        assert cortex.w_input_l4.shape[1] == initial_input + 20  # Columns expanded

        # Verify old weights preserved (left columns)
        assert torch.allclose(
            cortex.w_input_l4.data[:, :initial_input],
            initial_weights,
            atol=1e-6
        )

    def test_cortex_grow_output_expands_neuron_population(self, cortex):
        """Test that cortex.grow_output() expands output dimension.

        Note: Cortex adds neurons proportionally across layers based on current sizes.
        With l4=64, l23=96, l5=64, l6=32 (total=256), adding n_new=30 results in:
        - L4 += 7 (30 * 64/256)
        - L2/3 += 11 (30 * 96/256)
        - L5 += 7 (30 * 64/256)
        - L6 += 3 (30 * 32/256)
        - Total output (L2/3 + L5) += 18
        """
        initial_output = cortex.config.n_output

        # Grow output by 30 (actual output growth will be 18: L2/3:11 + L5:7)
        cortex.grow_output(n_new=30)

        # Verify output dimension expanded by 18 (L2/3:11 + L5:7)
        assert cortex.config.n_output == initial_output + 18

    def test_hippocampus_grow_input_expands_ec_weights(self, hippocampus):
        """Test that hippocampus.grow_input() expands EC input weights."""
        initial_input = hippocampus.config.n_input
        initial_dg_size = hippocampus.dg_size
        initial_ec_dg = hippocampus.w_ec_dg.data.clone()

        # Grow input by 25
        hippocampus.grow_input(n_new=25)

        # Verify input dimension expanded
        assert hippocampus.config.n_input == initial_input + 25

        # Verify w_ec_dg expanded
        assert hippocampus.w_ec_dg.shape[0] == initial_dg_size
        assert hippocampus.w_ec_dg.shape[1] == initial_input + 25

        # Verify old weights preserved
        assert torch.allclose(
            hippocampus.w_ec_dg.data[:, :initial_input],
            initial_ec_dg,
            atol=1e-6
        )

    def test_striatum_grow_input_expands_d1_d2_weights(self, striatum):
        """Test that striatum.grow_input() expands D1/D2 pathway weights."""
        initial_input = striatum.config.n_input
        initial_output = striatum.config.n_output
        initial_d1 = striatum.d1_pathway.weights.data.clone()
        initial_d2 = striatum.d2_pathway.weights.data.clone()

        # Grow input by 30
        striatum.grow_input(n_new=30)

        # Verify input dimension expanded
        assert striatum.config.n_input == initial_input + 30

        # Verify D1 weights expanded
        assert striatum.d1_pathway.weights.shape[0] == initial_output
        assert striatum.d1_pathway.weights.shape[1] == initial_input + 30

        # Verify D2 weights expanded
        assert striatum.d2_pathway.weights.shape[0] == initial_output
        assert striatum.d2_pathway.weights.shape[1] == initial_input + 30

        # Verify old weights preserved
        assert torch.allclose(
            striatum.d1_pathway.weights.data[:, :initial_input],
            initial_d1,
            atol=1e-6
        )
        assert torch.allclose(
            striatum.d2_pathway.weights.data[:, :initial_input],
            initial_d2,
            atol=1e-6
        )

    def test_bidirectional_pathway_growth(self, pathway):
        """Test growing both input and output dimensions."""
        initial_input = pathway.config.n_input
        initial_output = pathway.config.n_output

        # Grow input then output
        pathway.grow_input(10)
        pathway.grow_output(15)

        assert pathway.config.n_input == initial_input + 10
        assert pathway.config.n_output == initial_output + 15
        assert pathway.weights.shape == (initial_output + 15, initial_input + 10)

    def test_bidirectional_region_growth(self, cortex):
        """Test growing both input and output dimensions of a region."""
        initial_input = cortex.config.n_input
        initial_output = cortex.config.n_output

        # Grow input then output
        cortex.grow_input(20)
        cortex.grow_output(30)  # Actually adds 18 due to proportional growth

        assert cortex.config.n_input == initial_input + 20
        assert cortex.config.n_output == initial_output + 18  # 18, not 30

    def test_forward_pass_after_unified_growth(self, cortex):
        """Test that forward pass works after using unified growth API."""
        # Grow input and output (15 will become ~37 due to layer ratios)
        cortex.grow_input(10)
        cortex.grow_output(15)

        # Create input matching new input size
        new_input_size = cortex.config.n_input
        input_spikes = torch.zeros(new_input_size)
        input_spikes[:10] = 1.0  # Some spikes

        # Forward pass should work
        output = cortex(input_spikes)
        assert output.shape[0] == cortex.config.n_output
        assert not torch.isnan(output).any()


class TestAdditionalRegionsGrowth:
    """Test unified growth API for additional brain regions."""

    def test_thalamus_grow_input(self, thalamus):
        """Test that ThalamicRelay supports grow_input()."""
        initial_input = thalamus.config.n_input
        initial_output = thalamus.config.n_output

        # Grow input
        thalamus.grow_input(n_new=20)

        # Verify
        assert thalamus.config.n_input == initial_input + 20
        assert thalamus.config.n_output == initial_output  # Unchanged

    def test_thalamus_grow_output(self, thalamus):
        """Test that ThalamicRelay supports grow_output()."""
        initial_input = thalamus.config.n_input
        initial_output = thalamus.config.n_output

        # Grow output
        thalamus.grow_output(n_new=30)

        # Verify
        assert thalamus.config.n_input == initial_input  # Unchanged
        assert thalamus.config.n_output == initial_output + 30

    def test_thalamus_bidirectional_growth(self, thalamus):
        """Test bidirectional growth for thalamus."""
        initial_input = thalamus.config.n_input
        initial_output = thalamus.config.n_output

        thalamus.grow_input(15)
        thalamus.grow_output(25)

        assert thalamus.config.n_input == initial_input + 15
        assert thalamus.config.n_output == initial_output + 25

    def test_prefrontal_grow_input(self, prefrontal):
        """Test that Prefrontal supports grow_input()."""
        initial_input = prefrontal.config.n_input

        prefrontal.grow_input(n_new=20)

        assert prefrontal.config.n_input == initial_input + 20

    def test_prefrontal_grow_output(self, prefrontal):
        """Test that Prefrontal supports grow_output()."""
        initial_output = prefrontal.config.n_output

        prefrontal.grow_output(n_new=30)

        assert prefrontal.config.n_output == initial_output + 30

    def test_prefrontal_bidirectional_growth(self, prefrontal):
        """Test bidirectional growth for prefrontal cortex."""
        initial_input = prefrontal.config.n_input
        initial_output = prefrontal.config.n_output

        prefrontal.grow_input(10)
        prefrontal.grow_output(20)

        assert prefrontal.config.n_input == initial_input + 10
        assert prefrontal.config.n_output == initial_output + 20

    def test_cerebellum_grow_input(self, cerebellum):
        """Test that Cerebellum supports grow_input()."""
        initial_input = cerebellum.config.n_input

        cerebellum.grow_input(n_new=25)

        assert cerebellum.config.n_input == initial_input + 25

    def test_cerebellum_grow_output(self, cerebellum):
        """Test that Cerebellum supports grow_output()."""
        initial_output = cerebellum.config.n_output

        cerebellum.grow_output(n_new=15)

        assert cerebellum.config.n_output == initial_output + 15

    def test_cerebellum_bidirectional_growth(self, cerebellum):
        """Test bidirectional growth for cerebellum."""
        initial_input = cerebellum.config.n_input
        initial_output = cerebellum.config.n_output

        cerebellum.grow_input(20)
        cerebellum.grow_output(10)

        assert cerebellum.config.n_input == initial_input + 20
        assert cerebellum.config.n_output == initial_output + 10

    def test_multisensory_grow_input(self, multisensory):
        """Test that MultimodalIntegration supports grow_input()."""
        initial_input = multisensory.config.n_input

        multisensory.grow_input(n_new=30)

        assert multisensory.config.n_input == initial_input + 30

    def test_multisensory_grow_output(self, multisensory):
        """Test that MultimodalIntegration supports grow_output()."""
        initial_output = multisensory.config.n_output

        multisensory.grow_output(n_new=40)

        assert multisensory.config.n_output == initial_output + 40

    def test_multisensory_bidirectional_growth(self, multisensory):
        """Test bidirectional growth for multisensory integration."""
        initial_input = multisensory.config.n_input
        initial_output = multisensory.config.n_output

        multisensory.grow_input(25)
        multisensory.grow_output(35)

        assert multisensory.config.n_input == initial_input + 25
        assert multisensory.config.n_output == initial_output + 35


class TestSpecializedPathwaysGrowth:
    """Test unified growth API for specialized pathways."""

    def test_replay_pathway_grow_input(self, replay_pathway):
        """Test that SpikingReplayPathway supports grow_input()."""
        initial_input = replay_pathway.config.n_input
        initial_output = replay_pathway.config.n_output

        replay_pathway.grow_input(n_new=20)

        assert replay_pathway.config.n_input == initial_input + 20
        assert replay_pathway.config.n_output == initial_output  # Unchanged

    @pytest.mark.skip(reason="SpikingReplayPathway needs growth API implementation")
    def test_replay_pathway_grow_output(self, replay_pathway):
        """Test that SpikingReplayPathway supports grow_output()."""
        initial_input = replay_pathway.config.n_input
        initial_output = replay_pathway.config.n_output

        replay_pathway.grow_output(n_new=15)

        assert replay_pathway.config.n_input == initial_input  # Unchanged
        assert replay_pathway.config.n_output == initial_output + 15

    @pytest.mark.skip(reason="SpikingReplayPathway needs growth API implementation")
    def test_replay_pathway_bidirectional_growth(self, replay_pathway):
        """Test bidirectional growth for replay pathway."""
        initial_input = replay_pathway.config.n_input
        initial_output = replay_pathway.config.n_output

        replay_pathway.grow_input(10)
        replay_pathway.grow_output(12)

        assert replay_pathway.config.n_input == initial_input + 10
        assert replay_pathway.config.n_output == initial_output + 12

    def test_attention_pathway_grow_input(self, attention_pathway):
        """Test that SpikingAttentionPathway supports grow_input()."""
        initial_input = attention_pathway.config.n_input

        attention_pathway.grow_input(n_new=18)

        assert attention_pathway.config.n_input == initial_input + 18

    def test_attention_pathway_grow_output(self, attention_pathway):
        """Test that SpikingAttentionPathway supports grow_output()."""
        initial_output = attention_pathway.config.n_output

        attention_pathway.grow_output(n_new=22)

        assert attention_pathway.config.n_output == initial_output + 22

    def test_attention_pathway_bidirectional_growth(self, attention_pathway):
        """Test bidirectional growth for attention pathway."""
        initial_input = attention_pathway.config.n_input
        initial_output = attention_pathway.config.n_output

        attention_pathway.grow_input(8)
        attention_pathway.grow_output(12)

        assert attention_pathway.config.n_input == initial_input + 8
        assert attention_pathway.config.n_output == initial_output + 12

    @pytest.mark.skip(reason="VisualPathway has unimplemented abstract methods from NeuralComponent")
    def test_visual_pathway_grow_input(self, visual_pathway):
        """Test that VisualPathway supports grow_input()."""
        initial_input = visual_pathway.config.n_input

        visual_pathway.grow_input(n_new=20)

        assert visual_pathway.config.n_input == initial_input + 20

    @pytest.mark.skip(reason="VisualPathway has unimplemented abstract methods from NeuralComponent")
    def test_visual_pathway_grow_output(self, visual_pathway):
        """Test that VisualPathway supports grow_output()."""
        initial_output = visual_pathway.config.n_output

        visual_pathway.grow_output(n_new=30)

        assert visual_pathway.config.n_output == initial_output + 30

    @pytest.mark.skip(reason="VisualPathway has unimplemented abstract methods from NeuralComponent")
    def test_visual_pathway_bidirectional_growth(self, visual_pathway):
        """Test bidirectional growth for visual pathway."""
        initial_input = visual_pathway.config.n_input
        initial_output = visual_pathway.config.n_output

        visual_pathway.grow_input(16)
        visual_pathway.grow_output(24)

        assert visual_pathway.config.n_input == initial_input + 16
        assert visual_pathway.config.n_output == initial_output + 24

    # Auditory Pathway Tests
    @pytest.mark.skip(reason="AuditoryPathway has unimplemented abstract methods from NeuralComponent")
    def test_auditory_pathway_grow_input(self, auditory_pathway):
        """Test that AuditoryPathway supports grow_input()."""
        initial_input = auditory_pathway.config.n_input

        auditory_pathway.grow_input(n_new=15)

        assert auditory_pathway.config.n_input == initial_input + 15

    @pytest.mark.skip(reason="AuditoryPathway has unimplemented abstract methods from NeuralComponent")
    def test_auditory_pathway_grow_output(self, auditory_pathway):
        """Test that AuditoryPathway supports grow_output()."""
        initial_output = auditory_pathway.config.n_output

        auditory_pathway.grow_output(n_new=18)

        assert auditory_pathway.config.n_output == initial_output + 18

    @pytest.mark.skip(reason="AuditoryPathway has unimplemented abstract methods from NeuralComponent")
    def test_auditory_pathway_bidirectional_growth(self, auditory_pathway):
        """Test bidirectional growth for auditory pathway."""
        initial_input = auditory_pathway.config.n_input
        initial_output = auditory_pathway.config.n_output

        auditory_pathway.grow_input(12)
        auditory_pathway.grow_output(14)

        assert auditory_pathway.config.n_input == initial_input + 12
        assert auditory_pathway.config.n_output == initial_output + 14

    # Language Pathway Tests
    @pytest.mark.skip(reason="LanguagePathway has unimplemented abstract methods from NeuralComponent")
    def test_language_pathway_grow_input(self, language_pathway):
        """Test that LanguagePathway supports grow_input()."""
        initial_input = language_pathway.config.n_input

        language_pathway.grow_input(n_new=25)

        assert language_pathway.config.n_input == initial_input + 25

    @pytest.mark.skip(reason="LanguagePathway has unimplemented abstract methods from NeuralComponent")
    def test_language_pathway_grow_output(self, language_pathway):
        """Test that LanguagePathway supports grow_output()."""
        initial_output = language_pathway.config.n_output

        language_pathway.grow_output(n_new=28)

        assert language_pathway.config.n_output == initial_output + 28

    @pytest.mark.skip(reason="LanguagePathway has unimplemented abstract methods from NeuralComponent")
    def test_language_pathway_bidirectional_growth(self, language_pathway):
        """Test bidirectional growth for language pathway."""
        initial_input = language_pathway.config.n_input
        initial_output = language_pathway.config.n_output

        language_pathway.grow_input(20)
        language_pathway.grow_output(22)

        assert language_pathway.config.n_input == initial_input + 20
        assert language_pathway.config.n_output == initial_output + 22


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
