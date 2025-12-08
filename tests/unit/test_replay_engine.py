"""Tests for ReplayEngine.

Validates that the unified replay engine correctly consolidates replay logic
used by both sleep consolidation and hippocampal sequence replay.
"""

import pytest
import torch

from thalia.memory.replay_engine import (
    ReplayEngine,
    ReplayConfig,
    ReplayMode,
    ReplayResult,
)
from thalia.regions.gamma_dynamics import ThetaGammaConfig
from thalia.regions.hippocampus.config import Episode


class TestReplayEngine:
    """Test suite for ReplayEngine."""
    
    @pytest.fixture
    def gamma_config(self):
        """Create gamma oscillator config."""
        return ThetaGammaConfig(
            theta_freq_hz=8.0,
            gamma_freq_hz=40.0,
            n_slots=7,
            coupling_strength=0.8,
        )
    
    @pytest.fixture
    def replay_config(self, gamma_config):
        """Create replay engine config."""
        return ReplayConfig(
            compression_factor=5.0,
            dt_ms=1.0,
            theta_gamma_config=gamma_config,
            mode=ReplayMode.SEQUENCE,
            apply_gating=True,
            pattern_completion=True,
        )
    
    @pytest.fixture
    def engine(self, replay_config):
        """Create replay engine."""
        return ReplayEngine(replay_config)
    
    @pytest.fixture
    def sequence_episode(self):
        """Create episode with sequence."""
        sequence = [torch.randn(100) for _ in range(5)]
        return Episode(
            state=sequence[-1].clone(),
            action=0,
            reward=1.0,
            correct=True,
            sequence=sequence,
        )
    
    @pytest.fixture
    def single_episode(self):
        """Create episode without sequence."""
        return Episode(
            state=torch.randn(100),
            action=0,
            reward=1.0,
            correct=True,
        )
    
    def test_initialization(self, engine):
        """Test replay engine initializes correctly."""
        assert engine.config.compression_factor == 5.0
        assert engine.gamma_oscillator is not None
        assert engine.config.mode == ReplayMode.SEQUENCE
    
    def test_sequence_replay(self, engine, sequence_episode):
        """Test replaying a sequence."""
        result = engine.replay(sequence_episode)
        
        assert isinstance(result, ReplayResult)
        assert result.mode_used == ReplayMode.SEQUENCE
        assert result.slots_replayed == 5
        assert result.sequence_length == 5
        assert result.compression_factor == 5.0
        assert len(result.replayed_patterns) == 5
    
    def test_single_replay_fallback(self, engine, single_episode):
        """Test falling back to single-state replay."""
        result = engine.replay(single_episode)
        
        assert result.mode_used == ReplayMode.SINGLE
        assert result.slots_replayed == 1
        assert result.sequence_length == 1
        assert len(result.replayed_patterns) == 1
    
    def test_pattern_processor(self, engine, sequence_episode):
        """Test pattern processing callback."""
        processed = []
        
        def processor(pattern):
            processed.append(pattern)
            return pattern * 2.0  # Double the pattern
        
        result = engine.replay(
            sequence_episode,
            pattern_processor=processor,
        )
        
        # Processor should have been called for each slot
        assert len(processed) == 5
        
        # Output patterns should be doubled
        for i, pattern in enumerate(result.replayed_patterns):
            expected = sequence_episode.sequence[i] * 2.0
            assert torch.allclose(pattern, expected, atol=0.1)
    
    def test_gating_function(self, engine, sequence_episode):
        """Test gating callback."""
        gating_calls = []
        
        def gating_fn(slot):
            gating_calls.append(slot)
            return 0.5  # Half gating
        
        result = engine.replay(
            sequence_episode,
            gating_fn=gating_fn,
        )
        
        # Gating should have been called for each slot
        assert len(gating_calls) == 5
        assert gating_calls == [0, 1, 2, 3, 4]
    
    def test_time_compression(self, engine, sequence_episode):
        """Test that replay uses compressed timing."""
        # Replay should complete faster than real-time
        result = engine.replay(sequence_episode)
        
        # With 5x compression, sequence should replay quickly
        assert result.compression_factor == 5.0
        
        # Gamma oscillator should advance with compressed dt
        assert engine.gamma_oscillator is not None
    
    def test_without_gating(self, sequence_episode):
        """Test replay without gating."""
        config = ReplayConfig(
            compression_factor=5.0,
            theta_gamma_config=ThetaGammaConfig(),
            apply_gating=False,  # Disable gating
        )
        engine = ReplayEngine(config)
        
        result = engine.replay(sequence_episode)
        
        # Should still work, just without gating
        assert result.slots_replayed == 5
    
    def test_without_pattern_completion(self, sequence_episode):
        """Test replay without pattern processing."""
        config = ReplayConfig(
            compression_factor=5.0,
            theta_gamma_config=ThetaGammaConfig(),
            pattern_completion=False,
        )
        engine = ReplayEngine(config)
        
        result = engine.replay(
            sequence_episode,
            pattern_processor=None,  # No processing
        )
        
        # Patterns should match original sequence
        for i, pattern in enumerate(result.replayed_patterns):
            original = sequence_episode.sequence[i]
            assert torch.allclose(pattern, original, atol=0.1)
    
    def test_variable_compression_factor(self, engine, sequence_episode):
        """Test changing compression factor."""
        # Replay with different compression
        result = engine.replay(
            sequence_episode,
        )
        
        # Change compression
        engine.config.compression_factor = 10.0
        result2 = engine.replay(sequence_episode)
        
        assert result2.compression_factor == 10.0
    
    def test_ripple_modulation(self):
        """Test ripple modulation."""
        config = ReplayConfig(
            ripple_enabled=True,
            ripple_frequency=150.0,
            ripple_duration=80.0,
            ripple_gain=3.0,
        )
        engine = ReplayEngine(config)
        
        # Trigger ripple
        success = engine.trigger_ripple()
        assert success
        
        # Get modulation
        active, modulation = engine.get_ripple_modulation(dt=1.0)
        assert active
        assert modulation > 0.0
        
        # Ripple should decay after duration
        for _ in range(100):
            active, mod = engine.get_ripple_modulation(dt=1.0)
            if not active:
                break
        
        assert not active  # Should have ended
    
    def test_reset_state(self, engine, sequence_episode):
        """Test resetting engine state."""
        # Do some replay
        engine.replay(sequence_episode)
        
        # Oscillator should have advanced
        assert engine.gamma_oscillator.time_ms > 0
        
        # Reset
        engine.reset_state()
        
        # Should be back to initial state
        assert engine.gamma_oscillator.time_ms == 0.0
        assert engine.gamma_oscillator.theta_phase == 0.0
    
    def test_diagnostics(self, engine):
        """Test diagnostics output."""
        diag = engine.get_diagnostics()
        
        assert "compression_factor" in diag
        assert "mode" in diag
        assert "gamma_oscillator" in diag
        assert diag["compression_factor"] == 5.0
    
    def test_empty_sequence_fallback(self, engine):
        """Test handling empty sequence."""
        episode = Episode(
            state=torch.randn(100),
            action=0,
            reward=1.0,
            correct=True,
            sequence=[],  # Empty sequence
        )
        
        result = engine.replay(episode)
        
        # Should fall back to single-state replay
        assert result.mode_used == ReplayMode.SINGLE
        assert result.slots_replayed == 1
    
    def test_without_gamma_oscillator(self, sequence_episode):
        """Test replay without gamma oscillator (should fall back)."""
        config = ReplayConfig(
            compression_factor=5.0,
            theta_gamma_config=None,  # No oscillator
        )
        engine = ReplayEngine(config)
        
        result = engine.replay(sequence_episode)
        
        # Should fall back to single-state
        assert result.mode_used == ReplayMode.SINGLE
    
    def test_activity_tracking(self, engine, sequence_episode):
        """Test total activity tracking."""
        result = engine.replay(sequence_episode)
        
        # Total activity should be sum of all replayed patterns
        expected_activity = sum(
            pattern.sum().item() 
            for pattern in result.replayed_patterns
        )
        
        assert result.total_activity == pytest.approx(expected_activity, abs=0.1)
    
    def test_gamma_cycle_counting(self, engine, sequence_episode):
        """Test gamma cycle counting during replay."""
        result = engine.replay(sequence_episode)
        
        # Should count gamma cycles
        assert result.gamma_cycles >= 0
        
        # With 5 slots and compression, should see at least 1 cycle
        assert result.gamma_cycles >= 1


class TestReplayEngineIntegration:
    """Test integration with real components."""
    
    def test_with_hippocampus(self, striatum):
        """Test integration with real hippocampus (via fixture)."""
        # This is tested via test_neuromodulator_mixin.py's integration tests
        # which use striatum fixture that has hippocampus-like structure
        pass
