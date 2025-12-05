"""
Tests for the Language Interface module.

Tests the spike encoding/decoding pipeline for language processing.
"""

import pytest
import torch

from thalia.language.encoder import (
    SpikeEncoder,
    SpikeEncoderConfig,
    SparseDistributedRepresentation,
    EncodingType,
    HierarchicalSpikeEncoder,
)
from thalia.language.decoder import (
    SpikeDecoder,
    SpikeDecoderConfig,
    DecodingType,
    ConfidenceEstimator,
    StreamingDecoder,
)
from thalia.language.position import (
    OscillatoryPositionEncoder,
    PositionEncoderConfig,
    PositionEncodingType,
    SequenceTimer,
)
from thalia.language.model import (
    LanguageBrainInterface,
    LanguageInterfaceConfig,
    MinimalSpikingLM,
)
from thalia.core.brain import EventDrivenBrain, EventDrivenBrainConfig


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def device():
    """Get test device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def small_vocab_size():
    """Small vocabulary for fast tests."""
    return 1000


@pytest.fixture
def small_n_neurons():
    """Small number of neurons for fast tests."""
    return 128


@pytest.fixture
def small_n_timesteps():
    """Small number of timesteps for fast tests."""
    return 10


# ============================================================================
# SparseDistributedRepresentation Tests
# ============================================================================

class TestSparseDistributedRepresentation:
    """Tests for SDR encoding."""

    def test_creation(self, device, small_vocab_size, small_n_neurons):
        """Test SDR module creation."""
        config = SpikeEncoderConfig(
            vocab_size=small_vocab_size,
            n_neurons=small_n_neurons,
            device=device,
        )
        sdr = SparseDistributedRepresentation(config)
        assert sdr is not None

    def test_fixed_patterns_shape(self, device, small_vocab_size, small_n_neurons):
        """Test that fixed patterns have correct shape."""
        config = SpikeEncoderConfig(
            vocab_size=small_vocab_size,
            n_neurons=small_n_neurons,
            device=device,
        )
        sdr = SparseDistributedRepresentation(config)

        assert hasattr(sdr, "fixed_patterns")
        assert sdr.fixed_patterns.shape == (small_vocab_size, small_n_neurons)

    def test_sparsity(self, device, small_vocab_size, small_n_neurons):
        """Test that SDR patterns are sparse."""
        sparsity = 0.05
        sdr_on_bits = int(small_n_neurons * sparsity)  # Calculate expected on bits
        config = SpikeEncoderConfig(
            vocab_size=small_vocab_size,
            n_neurons=small_n_neurons,
            sparsity=sparsity,
            sdr_on_bits=sdr_on_bits,  # Explicitly set to match sparsity
            device=device,
        )
        sdr = SparseDistributedRepresentation(config)

        # Check that patterns have approximately correct sparsity
        mean_active = sdr.fixed_patterns.mean().item()
        expected_sparsity = sdr_on_bits / small_n_neurons
        assert abs(mean_active - expected_sparsity) < 0.02  # Within 2%

    def test_forward_fixed(self, device, small_vocab_size, small_n_neurons):
        """Test forward pass with fixed patterns."""
        config = SpikeEncoderConfig(
            vocab_size=small_vocab_size,
            n_neurons=small_n_neurons,
            device=device,
        )
        sdr = SparseDistributedRepresentation(config)

        token_ids = torch.randint(0, small_vocab_size, (2, 5), device=device)
        patterns = sdr(token_ids, use_learned=False)

        assert patterns.shape == (2, 5, small_n_neurons)

    def test_forward_learned(self, device, small_vocab_size, small_n_neurons):
        """Test forward pass with learned patterns."""
        config = SpikeEncoderConfig(
            vocab_size=small_vocab_size,
            n_neurons=small_n_neurons,
            learnable_embedding=True,
            device=device,
        )
        sdr = SparseDistributedRepresentation(config)
        sdr.to(device)

        token_ids = torch.randint(0, small_vocab_size, (2, 5), device=device)
        patterns = sdr(token_ids, use_learned=True)

        assert patterns.shape == (2, 5, small_n_neurons)

    def test_similarity(self, device, small_vocab_size, small_n_neurons):
        """Test SDR similarity computation."""
        config = SpikeEncoderConfig(
            vocab_size=small_vocab_size,
            n_neurons=small_n_neurons,
            device=device,
        )
        sdr = SparseDistributedRepresentation(config)

        # Same token should have high similarity
        token_ids = torch.tensor([[0], [0]], device=device)
        patterns = sdr(token_ids, use_learned=False)

        sim = sdr.get_similarity(patterns[0], patterns[1])
        assert sim.item() > 0.99  # Should be ~1.0


# ============================================================================
# SpikeEncoder Tests
# ============================================================================

class TestSpikeEncoder:
    """Tests for spike encoder."""

    def test_creation(self, device, small_vocab_size, small_n_neurons, small_n_timesteps):
        """Test encoder creation."""
        config = SpikeEncoderConfig(
            vocab_size=small_vocab_size,
            n_neurons=small_n_neurons,
            n_timesteps=small_n_timesteps,
            device=device,
        )
        encoder = SpikeEncoder(config)
        encoder.to(device)
        assert encoder is not None

    def test_forward_shape(self, device, small_vocab_size, small_n_neurons, small_n_timesteps):
        """Test that forward produces correct shapes."""
        config = SpikeEncoderConfig(
            vocab_size=small_vocab_size,
            n_neurons=small_n_neurons,
            n_timesteps=small_n_timesteps,
            device=device,
        )
        encoder = SpikeEncoder(config)
        encoder.to(device)

        token_ids = torch.randint(0, small_vocab_size, (2, 10), device=device)
        spikes, sdr = encoder(token_ids)

        assert spikes.shape == (2, 10, small_n_timesteps, small_n_neurons)
        assert sdr.shape == (2, 10, small_n_neurons)

    def test_spike_sparsity(self, device, small_vocab_size, small_n_neurons, small_n_timesteps):
        """Test that spike patterns are sparse."""
        config = SpikeEncoderConfig(
            vocab_size=small_vocab_size,
            n_neurons=small_n_neurons,
            n_timesteps=small_n_timesteps,
            sparsity=0.05,
            encoding_type=EncodingType.SDR,
            device=device,
        )
        encoder = SpikeEncoder(config)
        encoder.to(device)

        token_ids = torch.randint(0, small_vocab_size, (2, 10), device=device)
        spikes, _ = encoder(token_ids)

        # Mean activity should be low
        mean_activity = spikes.mean().item()
        assert mean_activity < 0.3  # Reasonably sparse

    @pytest.mark.parametrize("encoding_type", [
        EncodingType.SDR,
        EncodingType.RATE,
        EncodingType.TEMPORAL,
        EncodingType.PHASE,
    ])
    def test_encoding_types(self, device, small_vocab_size, small_n_neurons, encoding_type):
        """Test different encoding types."""
        config = SpikeEncoderConfig(
            vocab_size=small_vocab_size,
            n_neurons=small_n_neurons,
            n_timesteps=10,
            encoding_type=encoding_type,
            device=device,
        )
        encoder = SpikeEncoder(config)
        encoder.to(device)

        token_ids = torch.randint(0, small_vocab_size, (1, 5), device=device)
        spikes, _ = encoder(token_ids)

        assert spikes.shape[0] == 1
        assert spikes.shape[1] == 5

    def test_reset_phase(self, device, small_vocab_size, small_n_neurons):
        """Test phase reset."""
        config = SpikeEncoderConfig(
            vocab_size=small_vocab_size,
            n_neurons=small_n_neurons,
            device=device,
        )
        encoder = SpikeEncoder(config)
        encoder.to(device)

        encoder.reset_phase()
        assert encoder.theta_phase.item() == 0.0


# ============================================================================
# SpikeDecoder Tests
# ============================================================================

class TestSpikeDecoder:
    """Tests for spike decoder."""

    def test_creation(self, device, small_vocab_size, small_n_neurons, small_n_timesteps):
        """Test decoder creation."""
        config = SpikeDecoderConfig(
            n_neurons=small_n_neurons,
            vocab_size=small_vocab_size,
            n_timesteps=small_n_timesteps,
            device=device,
        )
        decoder = SpikeDecoder(config)
        decoder.to(device)
        assert decoder is not None

    def test_forward_shape(self, device, small_vocab_size, small_n_neurons, small_n_timesteps):
        """Test that forward produces correct shapes."""
        config = SpikeDecoderConfig(
            n_neurons=small_n_neurons,
            vocab_size=small_vocab_size,
            n_timesteps=small_n_timesteps,
            device=device,
        )
        decoder = SpikeDecoder(config)
        decoder.to(device)

        spikes = (torch.rand(2, 10, small_n_timesteps, small_n_neurons, device=device) > 0.9).float()
        logits = decoder(spikes)

        assert logits.shape == (2, 10, small_vocab_size)

    @pytest.mark.parametrize("decoding_type", [
        DecodingType.RATE,
        DecodingType.TEMPORAL,
        DecodingType.POPULATION,
    ])
    def test_decoding_types(self, device, small_vocab_size, small_n_neurons, decoding_type):
        """Test different decoding types."""
        config = SpikeDecoderConfig(
            n_neurons=small_n_neurons,
            vocab_size=small_vocab_size,
            n_timesteps=10,
            decoding_type=decoding_type,
            device=device,
        )
        decoder = SpikeDecoder(config)
        decoder.to(device)

        spikes = (torch.rand(1, 5, 10, small_n_neurons, device=device) > 0.9).float()
        logits = decoder(spikes)

        assert logits.shape == (1, 5, small_vocab_size)

    def test_sample(self, device, small_vocab_size, small_n_neurons, small_n_timesteps):
        """Test sampling from decoder."""
        config = SpikeDecoderConfig(
            n_neurons=small_n_neurons,
            vocab_size=small_vocab_size,
            n_timesteps=small_n_timesteps,
            device=device,
        )
        decoder = SpikeDecoder(config)
        decoder.to(device)

        spikes = (torch.rand(2, 5, small_n_timesteps, small_n_neurons, device=device) > 0.9).float()
        tokens, log_probs = decoder.sample(spikes, temperature=1.0)

        assert tokens.shape == (2, 5)
        assert log_probs.shape == (2, 5)
        assert (tokens >= 0).all() and (tokens < small_vocab_size).all()

    def test_greedy_decode(self, device, small_vocab_size, small_n_neurons, small_n_timesteps):
        """Test greedy decoding."""
        config = SpikeDecoderConfig(
            n_neurons=small_n_neurons,
            vocab_size=small_vocab_size,
            n_timesteps=small_n_timesteps,
            device=device,
        )
        decoder = SpikeDecoder(config)
        decoder.to(device)

        spikes = (torch.rand(2, 5, small_n_timesteps, small_n_neurons, device=device) > 0.9).float()
        tokens, probs = decoder.greedy_decode(spikes)

        assert tokens.shape == (2, 5)
        assert probs.shape == (2, 5)


# ============================================================================
# PositionEncoder Tests
# ============================================================================

class TestPositionEncoder:
    """Tests for position encoder."""

    def test_creation(self, device, small_n_neurons):
        """Test position encoder creation."""
        config = PositionEncoderConfig(
            n_neurons=small_n_neurons,
            max_positions=512,
            device=device,
        )
        encoder = OscillatoryPositionEncoder(config)
        encoder.to(device)
        assert encoder is not None

    def test_forward_spikes(self, device, small_n_neurons):
        """Test forward with spike output."""
        config = PositionEncoderConfig(
            n_neurons=small_n_neurons,
            max_positions=512,
            n_timesteps=10,
            device=device,
        )
        encoder = OscillatoryPositionEncoder(config)
        encoder.to(device)

        position_ids = torch.arange(20, device=device).unsqueeze(0)
        spikes = encoder(position_ids, as_spikes=True)

        assert spikes.shape == (1, 20, 10, small_n_neurons)

    def test_forward_continuous(self, device, small_n_neurons):
        """Test forward with continuous output."""
        config = PositionEncoderConfig(
            n_neurons=small_n_neurons,
            max_positions=512,
            encoding_type=PositionEncodingType.SINUSOIDAL,
            device=device,
        )
        encoder = OscillatoryPositionEncoder(config)
        encoder.to(device)

        position_ids = torch.arange(20, device=device).unsqueeze(0)
        encoding = encoder(position_ids, as_spikes=False)

        assert encoding.shape == (1, 20, small_n_neurons)

    @pytest.mark.parametrize("encoding_type", [
        PositionEncodingType.SINUSOIDAL,
        PositionEncodingType.OSCILLATORY,
        PositionEncodingType.PHASE_PRECESSION,
        PositionEncodingType.NESTED_GAMMA,
    ])
    def test_encoding_types(self, device, small_n_neurons, encoding_type):
        """Test different position encoding types."""
        config = PositionEncoderConfig(
            n_neurons=small_n_neurons,
            max_positions=512,
            n_timesteps=10,
            encoding_type=encoding_type,
            device=device,
        )
        encoder = OscillatoryPositionEncoder(config)
        encoder.to(device)

        position_ids = torch.arange(10, device=device).unsqueeze(0)
        spikes = encoder(position_ids, as_spikes=True)

        assert spikes.shape[1] == 10

    def test_relative_encoding(self, device, small_n_neurons):
        """Test relative position encoding."""
        config = PositionEncoderConfig(
            n_neurons=small_n_neurons,
            max_positions=512,
            device=device,
        )
        encoder = OscillatoryPositionEncoder(config)
        encoder.to(device)

        keys = torch.arange(10, device=device).unsqueeze(0)
        queries = torch.arange(5, device=device).unsqueeze(0)

        relative = encoder.get_relative_encoding(keys, queries)

        assert relative.shape == (1, 5, 10)


# ============================================================================
# SequenceTimer Tests
# ============================================================================

class TestSequenceTimer:
    """Tests for sequence timer."""

    def test_creation(self, device):
        """Test timer creation."""
        timer = SequenceTimer(n_neurons=64, device=device)
        timer.to(device)
        assert timer is not None

    def test_step(self, device):
        """Test timer step."""
        timer = SequenceTimer(n_neurons=64, device=device)
        timer.to(device)

        spikes = timer.step(n_steps=1)
        assert spikes.shape == (64,)

    def test_reset(self, device):
        """Test timer reset."""
        timer = SequenceTimer(n_neurons=64, device=device)
        timer.to(device)

        timer.step(n_steps=10)
        timer.reset()

        state = timer.get_state()
        assert state["theta_phase"] == 0.0
        assert state["gamma_phase"] == 0.0


# ============================================================================
# LanguageBrainInterface Tests
# ============================================================================

class TestLanguageBrainInterface:
    """Tests for the language-brain interface."""

    def test_creation(self, small_vocab_size, small_n_neurons):
        """Test interface creation with brain."""
        # NOTE: Using CPU explicitly due to device propagation issues in complex systems
        device = "cpu"

        # Create a minimal brain configuration
        brain_config = EventDrivenBrainConfig(
            input_size=small_n_neurons,
            cortex_size=small_n_neurons,
            hippocampus_size=32,
            pfc_size=16,
            n_actions=10,
            device=device,
        )
        brain = EventDrivenBrain(brain_config)

        # Create language interface
        config = LanguageInterfaceConfig(
            vocab_size=small_vocab_size,
            brain_input_size=small_n_neurons,
            n_timesteps=5,
            device=device,
        )
        interface = LanguageBrainInterface(brain, config)
        assert interface is not None

    def test_process_tokens(self, small_vocab_size, small_n_neurons):
        """Test token processing through brain."""
        device = "cpu"

        brain_config = EventDrivenBrainConfig(
            input_size=small_n_neurons,
            cortex_size=small_n_neurons,
            hippocampus_size=32,
            pfc_size=16,
            n_actions=10,
            device=device,
        )
        brain = EventDrivenBrain(brain_config)

        config = LanguageInterfaceConfig(
            vocab_size=small_vocab_size,
            brain_input_size=small_n_neurons,
            n_timesteps=5,
            device=device,
        )
        interface = LanguageBrainInterface(brain, config)

        # Process tokens
        token_ids = torch.randint(0, small_vocab_size, (1, 5), device=device)
        result = interface.process_tokens(token_ids)

        # Result should contain token processing info
        assert result is not None
        assert "n_tokens" in result
        assert "results" in result
        assert result["n_tokens"] == 5

    def test_diagnostics(self, small_vocab_size, small_n_neurons):
        """Test diagnostics output."""
        device = "cpu"

        brain_config = EventDrivenBrainConfig(
            input_size=small_n_neurons,
            cortex_size=small_n_neurons,
            hippocampus_size=32,
            pfc_size=16,
            n_actions=10,
            device=device,
        )
        brain = EventDrivenBrain(brain_config)

        config = LanguageInterfaceConfig(
            vocab_size=small_vocab_size,
            brain_input_size=small_n_neurons,
            n_timesteps=5,
            device=device,
        )
        interface = LanguageBrainInterface(brain, config)

        diagnostics = interface.get_diagnostics()

        assert "encoder" in diagnostics
        assert "decoder" in diagnostics
# ============================================================================
# MinimalSpikingLM Tests
# ============================================================================

class TestMinimalSpikingLM:
    """Tests for minimal spiking language model."""

    def test_creation(self, device, small_vocab_size, small_n_neurons):
        """Test minimal model creation."""
        model = MinimalSpikingLM(
            vocab_size=small_vocab_size,
            n_neurons=small_n_neurons,
            n_timesteps=5,
            device=device,
        )
        assert model is not None

    def test_forward(self, device, small_vocab_size, small_n_neurons):
        """Test minimal model forward pass."""
        model = MinimalSpikingLM(
            vocab_size=small_vocab_size,
            n_neurons=small_n_neurons,
            n_timesteps=5,
            device=device,
        )

        token_ids = torch.randint(0, small_vocab_size, (2, 8), device=device)
        logits = model(token_ids)

        assert logits.shape == (2, 8, small_vocab_size)

    def test_generate(self, device, small_vocab_size, small_n_neurons):
        """Test minimal model generation."""
        model = MinimalSpikingLM(
            vocab_size=small_vocab_size,
            n_neurons=small_n_neurons,
            n_timesteps=5,
            device=device,
        )

        prompt = torch.randint(0, small_vocab_size, (1, 3), device=device)
        generated = model.generate(prompt, max_new_tokens=5)

        assert generated.shape == (1, 8)


# ============================================================================
# Integration Tests
# ============================================================================

class TestEncoderDecoderIntegration:
    """Tests for encoder-decoder integration."""

    def test_encode_decode_pipeline(self, device, small_vocab_size, small_n_neurons, small_n_timesteps):
        """Test full encode-decode pipeline."""
        # Create encoder
        encoder_config = SpikeEncoderConfig(
            vocab_size=small_vocab_size,
            n_neurons=small_n_neurons,
            n_timesteps=small_n_timesteps,
            device=device,
        )
        encoder = SpikeEncoder(encoder_config)
        encoder.to(device)

        # Create decoder
        decoder_config = SpikeDecoderConfig(
            n_neurons=small_n_neurons,
            vocab_size=small_vocab_size,
            n_timesteps=small_n_timesteps,
            device=device,
        )
        decoder = SpikeDecoder(decoder_config)
        decoder.to(device)

        # Forward pipeline
        token_ids = torch.randint(0, small_vocab_size, (2, 10), device=device)
        spikes, _ = encoder(token_ids)
        logits = decoder(spikes)

        assert logits.shape == (2, 10, small_vocab_size)

    def test_encode_decode_with_position(self, device, small_vocab_size, small_n_neurons, small_n_timesteps):
        """Test pipeline with position encoding."""
        # Encoder
        encoder = SpikeEncoder(SpikeEncoderConfig(
            vocab_size=small_vocab_size,
            n_neurons=small_n_neurons,
            n_timesteps=small_n_timesteps,
            device=device,
        ))
        encoder.to(device)

        # Position encoder
        pos_encoder = OscillatoryPositionEncoder(PositionEncoderConfig(
            n_neurons=small_n_neurons // 4,
            n_timesteps=small_n_timesteps,
            device=device,
        ))
        pos_encoder.to(device)

        # Decoder (for combined size)
        combined_neurons = small_n_neurons + small_n_neurons // 4
        decoder = SpikeDecoder(SpikeDecoderConfig(
            n_neurons=combined_neurons,
            vocab_size=small_vocab_size,
            n_timesteps=small_n_timesteps,
            device=device,
        ))
        decoder.to(device)

        # Forward
        token_ids = torch.randint(0, small_vocab_size, (1, 10), device=device)
        position_ids = torch.arange(10, device=device).unsqueeze(0)

        content_spikes, _ = encoder(token_ids)
        position_spikes = pos_encoder(position_ids, as_spikes=True)

        combined = torch.cat([content_spikes, position_spikes], dim=-1)
        logits = decoder(combined)

        assert logits.shape == (1, 10, small_vocab_size)


# ============================================================================
# ConfidenceEstimator Tests
# ============================================================================

class TestConfidenceEstimator:
    """Tests for confidence estimator."""

    def test_creation(self, device, small_n_neurons):
        """Test confidence estimator creation."""
        estimator = ConfidenceEstimator(n_neurons=small_n_neurons, device=device)
        estimator.to(device)
        assert estimator is not None

    def test_forward(self, device, small_n_neurons, small_n_timesteps):
        """Test confidence estimation."""
        estimator = ConfidenceEstimator(n_neurons=small_n_neurons, device=device)
        estimator.to(device)

        spikes = (torch.rand(2, 5, small_n_timesteps, small_n_neurons, device=device) > 0.9).float()
        features = spikes.mean(dim=2)

        confidence = estimator(spikes, features)

        assert confidence.shape == (2, 5)
        assert (confidence >= 0).all() and (confidence <= 1).all()


# ============================================================================
# HierarchicalSpikeEncoder Tests
# ============================================================================

class TestHierarchicalSpikeEncoder:
    """Tests for hierarchical spike encoder."""

    def test_creation(self, device, small_vocab_size, small_n_neurons):
        """Test hierarchical encoder creation."""
        char_config = SpikeEncoderConfig(
            vocab_size=256,  # ASCII
            n_neurons=small_n_neurons // 2,
            device=device,
        )
        subword_config = SpikeEncoderConfig(
            vocab_size=small_vocab_size,
            n_neurons=small_n_neurons,
            device=device,
        )

        encoder = HierarchicalSpikeEncoder(
            char_config=char_config,
            subword_config=subword_config,
        )
        encoder.to(device)
        assert encoder is not None

    def test_forward(self, device, small_vocab_size, small_n_neurons):
        """Test hierarchical encoding."""
        char_config = SpikeEncoderConfig(
            vocab_size=256,
            n_neurons=small_n_neurons // 2,
            device=device,
        )
        subword_config = SpikeEncoderConfig(
            vocab_size=small_vocab_size,
            n_neurons=small_n_neurons,
            device=device,
        )

        encoder = HierarchicalSpikeEncoder(
            char_config=char_config,
            subword_config=subword_config,
        )
        encoder.to(device)

        char_ids = torch.randint(0, 256, (1, 20), device=device)
        subword_ids = torch.randint(0, small_vocab_size, (1, 5), device=device)

        outputs = encoder(char_ids=char_ids, subword_ids=subword_ids)

        assert "char_spikes" in outputs
        assert "subword_spikes" in outputs
