"""
Tests for Neural Pathway Protocol compliance.

Validates that all pathway types (Sensory, Spiking, Specialized)
correctly implement the NeuralPathway protocol interface.
"""

import pytest
import torch

from thalia.core.pathway_protocol import (
    NeuralPathway,
)
from thalia.sensory import (
    VisualPathway,
    VisualConfig,
    AuditoryPathway,
    AuditoryConfig,
    Modality,
)
from thalia.integration import (
    SpikingPathway,
    SpikingPathwayConfig,
)
from thalia.integration.pathways import (
    SpikingAttentionPathway,
    SpikingAttentionPathwayConfig,
    SpikingReplayPathway,
    SpikingReplayPathwayConfig,
)


@pytest.fixture
def device():
    return "cpu"


# =============================================================================
# Protocol Compliance Tests
# =============================================================================

class TestProtocolCompliance:
    """Test that all pathways implement NeuralPathway protocol."""

    def test_visual_pathway_implements_protocol(self, device):
        """Visual pathway should implement NeuralPathway protocol (ADR-007)."""
        config = VisualConfig(device=device)
        pathway = VisualPathway(config)

        # Check isinstance with protocol
        assert isinstance(pathway, NeuralPathway)

        # Verify required methods exist (ADR-007: all use forward())
        assert hasattr(pathway, 'forward')
        assert hasattr(pathway, 'get_modality')
        assert hasattr(pathway, 'reset_state')
        assert hasattr(pathway, 'get_diagnostics')

    def test_auditory_pathway_implements_protocol(self, device):
        """Auditory pathway should implement NeuralPathway protocol (ADR-007)."""
        config = AuditoryConfig(device=device)
        pathway = AuditoryPathway(config)

        assert isinstance(pathway, NeuralPathway)
        # ADR-007: All pathways use forward(), not encode()
        assert hasattr(pathway, 'forward')
        assert hasattr(pathway, 'get_modality')
        assert hasattr(pathway, 'reset_state')
        assert hasattr(pathway, 'get_diagnostics')

    def test_language_pathway_implements_protocol(self, device):
        """Language pathway should implement NeuralPathway protocol (ADR-007)."""
        from thalia.sensory import LanguagePathway, LanguageConfig
        
        config = LanguageConfig(device=device)
        pathway = LanguagePathway(config)

        assert isinstance(pathway, NeuralPathway)
        # ADR-007: All pathways use forward(), not encode()
        assert hasattr(pathway, 'forward')
        assert hasattr(pathway, 'get_modality')
        assert hasattr(pathway, 'reset_state')
        assert hasattr(pathway, 'get_diagnostics')

    def test_spiking_pathway_implements_protocol(self, device):
        """Spiking pathway should implement NeuralPathway (always learns via STDP)."""
        config = SpikingPathwayConfig(
            source_size=32,
            target_size=64,
            device=device,
        )
        pathway = SpikingPathway(config)

        assert isinstance(pathway, NeuralPathway)
        assert hasattr(pathway, 'forward')
        # No separate learn() method - learning happens automatically in forward()
        assert hasattr(pathway, 'reset_state')
        assert hasattr(pathway, 'get_diagnostics')

    def test_attention_pathway_implements_protocol(self, device):
        """Attention pathway should implement NeuralPathway (always learns)."""
        config = SpikingAttentionPathwayConfig(
            source_size=32,
            target_size=64,
            device=device,
        )
        pathway = SpikingAttentionPathway(config)

        assert isinstance(pathway, NeuralPathway)
        assert hasattr(pathway, 'forward')
        # Learning happens automatically during forward passes
        assert hasattr(pathway, 'reset_state')
        assert hasattr(pathway, 'get_diagnostics')

    def test_replay_pathway_implements_protocol(self, device):
        """Replay pathway should implement NeuralPathway (always learns)."""
        config = SpikingReplayPathwayConfig(
            source_size=32,
            target_size=64,
            device=device,
        )
        pathway = SpikingReplayPathway(config)

        assert isinstance(pathway, NeuralPathway)
        assert hasattr(pathway, 'forward')
        # Learning happens automatically during forward passes
        assert hasattr(pathway, 'reset_state')
        assert hasattr(pathway, 'get_diagnostics')


# =============================================================================
# Sensory Pathway Interface Tests
# =============================================================================

class TestSensoryPathwayInterface:
    """Test sensory pathway encode() interface."""

    def test_visual_encode(self, device):
        """Visual pathway forward() should return spikes and metadata (ADR-007)."""
        config = VisualConfig(output_size=128, device=device)
        pathway = VisualPathway(config)

        # ADR-005: Single brain = single sample input [C, H, W]
        # Use default image size (28x28 in config)
        image = torch.randn(1, 28, 28)  # [channels, H, W]
        spikes, metadata = pathway(image)  # Callable syntax (ADR-007)

        # Check output format
        assert isinstance(spikes, torch.Tensor)
        assert isinstance(metadata, dict)
        assert spikes.dim() >= 2  # At least [batch, neurons]

        # Check modality - VISION not VISUAL
        assert pathway.get_modality() == Modality.VISION

    def test_auditory_encode(self, device):
        """Auditory pathway forward() should return spikes and metadata (ADR-007)."""
        config = AuditoryConfig(output_size=128, device=device)
        pathway = AuditoryPathway(config)

        # ADR-005: Single brain = single audio sample [samples]
        audio = torch.randn(16000)  # [samples]
        spikes, metadata = pathway(audio)  # Callable syntax (ADR-007)

        assert isinstance(spikes, torch.Tensor)
        assert isinstance(metadata, dict)
        assert spikes.dim() >= 2
        # Check modality - AUDITION not AUDITORY
        assert pathway.get_modality() == Modality.AUDITION

    def test_language_encode(self, device):
        """Language pathway forward() should return spikes and metadata (ADR-007)."""
        from thalia.sensory import LanguagePathway, LanguageConfig, Modality
        
        config = LanguageConfig(output_size=128, device=device)
        pathway = LanguagePathway(config)

        # ADR-005: Single brain = single token (scalar or [1])
        token_id = torch.tensor(42)  # Single token
        spikes, metadata = pathway(token_id)  # Callable syntax (ADR-007)

        # Check output format
        assert isinstance(spikes, torch.Tensor)
        assert isinstance(metadata, dict)
        assert spikes.dim() == 2  # [n_timesteps, output_size]
        assert spikes.shape[1] == 128  # output_size
        
        # Check modality
        assert pathway.get_modality() == Modality.LANGUAGE

    def test_sensory_reset_state(self, device):
        """Sensory pathways should have reset_state() method."""
        from thalia.sensory import LanguagePathway, LanguageConfig
        
        pathways = [
            VisualPathway(VisualConfig(device=device)),
            AuditoryPathway(AuditoryConfig(device=device)),
            LanguagePathway(LanguageConfig(device=device)),
        ]

        for pathway in pathways:
            # Should not raise
            pathway.reset_state()

    def test_sensory_diagnostics(self, device):
        """Sensory pathways should return diagnostics."""
        config = VisualConfig(device=device)
        pathway = VisualPathway(config)

        diagnostics = pathway.get_diagnostics()

        assert isinstance(diagnostics, dict)
        assert 'modality' in diagnostics
        assert diagnostics['modality'] == 'vision'  # Modality.VISUAL.value


# =============================================================================
# Spiking Pathway Interface Tests
# =============================================================================

class TestSpikingPathwayInterface:
    """Test spiking pathway forward() and learn() interface."""

    def test_spiking_forward(self, device):
        """Spiking pathway forward should process spikes."""
        config = SpikingPathwayConfig(
            source_size=32,
            target_size=64,
            device=device,
        )
        pathway = SpikingPathway(config)

        source_spikes = torch.rand(32) > 0.9  # Sparse spikes
        target_spikes = pathway(source_spikes.float(), dt=1.0)

        assert isinstance(target_spikes, torch.Tensor)
        assert target_spikes.shape == (64,)
        assert target_spikes.dtype == torch.float32

    def test_spiking_learn(self, device):
        """Spiking pathway learn should return metrics."""
        config = SpikingPathwayConfig(
            source_size=32,
            target_size=64,
            device=device,
        )
        pathway = SpikingPathway(config)

        # Trigger learning
        source = torch.randn(32)
        target = torch.randn(64)
        metrics = pathway.learn(source, target, dopamine=0.5)

        assert isinstance(metrics, dict)
        # Should contain some learning metrics
        assert len(metrics) > 0

    def test_spiking_reset_state(self, device):
        """Spiking pathway should reset membrane, traces, etc."""
        config = SpikingPathwayConfig(
            source_size=32,
            target_size=64,
            device=device,
        )
        pathway = SpikingPathway(config)

        # Run some spikes through
        pathway(torch.randn(32), dt=1.0)
        pathway(torch.randn(32), dt=1.0)

        # Reset
        pathway.reset_state()

        # Check state is cleared
        assert pathway.membrane.abs().max() < 1.0  # Near resting
        assert pathway.pre_trace.abs().max() < 0.01  # Nearly zero
        assert pathway.post_trace.abs().max() < 0.01

    def test_spiking_diagnostics(self, device):
        """Spiking pathway should return comprehensive diagnostics."""
        config = SpikingPathwayConfig(
            source_size=32,
            target_size=64,
            device=device,
        )
        pathway = SpikingPathway(config)

        # Run some activity
        for _ in range(10):
            pathway(torch.rand(32) > 0.9, dt=1.0)

        diagnostics = pathway.get_diagnostics()

        assert isinstance(diagnostics, dict)
        # Should have various metrics (not necessarily 'config' key)
        assert len(diagnostics) > 0
        assert 'weight_mean' in diagnostics  # At least some weight metrics


# =============================================================================
# Specialized Pathway Tests
# =============================================================================

class TestSpecializedPathways:
    """Test specialized pathway variants."""

    def test_attention_pathway_forward(self, device):
        """Attention pathway should modulate input."""
        config = SpikingAttentionPathwayConfig(
            source_size=16,    # PFC/source size
            target_size=32,    # Cortex/target size
            input_size=64,     # Input dimension to modulate
            device=device,
        )
        pathway = SpikingAttentionPathway(config)

        # Test 1: Standard forward() inherited from SpikingPathway
        pfc_spikes = torch.rand(16) > 0.9  # Sparse PFC spikes [source_size]
        output = pathway(pfc_spikes.float(), dt=1.0)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape == (32,)  # target_size

        # Test 2: Specialized modulate() method for attention
        input_signal = torch.randn(64)  # [input_size]
        pfc_activity = torch.randn(16)  # [source_size] - PFC activity
        modulated = pathway.modulate(input_signal, pfc_activity, dt=1.0)
        
        assert isinstance(modulated, torch.Tensor)
        assert modulated.shape == (64,)  # Same as input_signal
        
        # Modulation should have changed the signal (usually, unless gain is exactly 1)
        # Just check it runs without error - exact modulation depends on learned weights

    def test_replay_pathway_forward(self, device):
        """Replay pathway should support replay mode."""
        config = SpikingReplayPathwayConfig(
            source_size=32,
            target_size=64,
            device=device,
        )
        pathway = SpikingReplayPathway(config)

        hippocampal_spikes = torch.rand(32) > 0.9
        output = pathway(hippocampal_spikes.float(), dt=1.0)

        assert isinstance(output, torch.Tensor)
        assert output.shape == (64,)

    def test_specialized_pathways_inherit_interface(self, device):
        """Specialized pathways should inherit full interface."""
        attention = SpikingAttentionPathway(SpikingAttentionPathwayConfig(
            source_size=16, target_size=32, input_size=64, device=device
        ))
        replay = SpikingReplayPathway(SpikingReplayPathwayConfig(
            source_size=16, target_size=32, device=device
        ))

        for pathway in [attention, replay]:
            # All should have full interface
            assert hasattr(pathway, 'forward')
            assert hasattr(pathway, 'learn')
            assert hasattr(pathway, 'reset_state')
            assert hasattr(pathway, 'get_diagnostics')

            # Test reset doesn't crash
            pathway.reset_state()

            # Test diagnostics
            diag = pathway.get_diagnostics()
            assert isinstance(diag, dict)


# =============================================================================
# Polymorphic Usage Tests
# =============================================================================

class TestPolymorphicUsage:
    """Test that pathways can be used polymorphically."""

    def test_uniform_reset(self, device):
        """All pathways should reset uniformly."""
        pathways = [
            VisualPathway(VisualConfig(device=device)),
            SpikingPathway(SpikingPathwayConfig(source_size=32, target_size=64, device=device)),
            SpikingAttentionPathway(SpikingAttentionPathwayConfig(
                source_size=16, target_size=32, input_size=64, device=device
            )),
        ]

        # Uniform reset interface
        for pathway in pathways:
            pathway.reset_state()  # Should work for all

    def test_uniform_diagnostics(self, device):
        """All pathways should provide diagnostics."""
        pathways = [
            VisualPathway(VisualConfig(device=device)),
            SpikingPathway(SpikingPathwayConfig(source_size=32, target_size=64, device=device)),
        ]

        for pathway in pathways:
            diag = pathway.get_diagnostics()
            assert isinstance(diag, dict)
            assert len(diag) > 0

    # Note: Removed test_learnable_pathway_filtering - ALL pathways learn automatically.
    # No need to filter for learnable pathways since learning happens during forward/encode.


# =============================================================================
# Integration Tests
# =============================================================================

class TestPathwayIntegration:
    """Test pathway integration scenarios."""

    def test_sensory_to_spiking_chain(self, device):
        """Test chaining sensory → spiking pathway."""
        # Use Visual pathway (Language has import issues)
        visual_config = VisualConfig(output_size=128, device=device)
        visual_pathway = VisualPathway(visual_config)

        # Spiking pathway transforms to target region
        spiking_config = SpikingPathwayConfig(
            source_size=128,
            target_size=64,
            device=device,
        )
        spiking_pathway = SpikingPathway(spiking_config)

        # Process image → spikes → transformed spikes
        # ADR-005: Single brain = single sample [C, H, W]
        image = torch.randn(1, 28, 28)  # [channels, H, W]
        spikes, _ = visual_pathway(image)  # Callable syntax (ADR-007) [n_timesteps, 128]

        # Process first timestep
        # ADR-006: Temporal coding produces [n_timesteps, n_neurons]
        first_spike = spikes[0, :]  # [128]
        output = spiking_pathway(first_spike, dt=1.0)

        assert output.shape == (64,)

    def test_pathway_state_lifecycle(self, device):
        """Test full pathway lifecycle with state management."""
        config = SpikingPathwayConfig(source_size=32, target_size=64, device=device)
        pathway = SpikingPathway(config)

        # 1. Initial state
        diag_initial = pathway.get_diagnostics()
        assert isinstance(diag_initial, dict)

        # 2. Process some spikes
        for _ in range(10):
            pathway(torch.rand(32) > 0.9, dt=1.0)

        # 3. Check state has changed
        diag_active = pathway.get_diagnostics()
        assert diag_active != diag_initial  # Some metrics should differ

        # 4. Reset
        pathway.reset_state()

        # 5. State should be reset
        assert pathway.membrane.abs().max() < 1.0
