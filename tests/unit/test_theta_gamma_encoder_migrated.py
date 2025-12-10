"""
Tests for ThetaGammaEncoder after centralized oscillator migration.

This file contains updated tests that work with the new architecture
where oscillator phases are provided by the brain instead of created locally.
"""

import pytest
import math
import torch

from thalia.tasks.working_memory import (
    ThetaGammaEncoder,
    WorkingMemoryTaskConfig,
)


@pytest.fixture
def wm_config():
    """Create working memory config."""
    return WorkingMemoryTaskConfig(
        theta_freq_hz=8.0,
        gamma_freq_hz=40.0,
        items_per_theta_cycle=8,
        device="cpu"
    )


class TestThetaGammaEncoderMigrated:
    """Test ThetaGammaEncoder with centralized oscillators."""
    
    def test_encoder_initialization(self, wm_config):
        """Test encoder initializes without local oscillators."""
        encoder = ThetaGammaEncoder(wm_config)
        
        # Encoder stores phases, not oscillators
        assert hasattr(encoder, '_theta_phase')
        assert hasattr(encoder, '_gamma_phase')
        assert encoder._theta_phase == 0.0
        assert encoder._gamma_phase == 0.0
        assert encoder.item_count == 0
    
    def test_set_oscillator_phases(self, wm_config):
        """Test receiving oscillator phases from brain."""
        encoder = ThetaGammaEncoder(wm_config)
        
        # Provide phases (as brain would)
        phases = {
            'theta': math.pi / 4,
            'gamma': math.pi / 2,
            'delta': 0.0,
            'alpha': 0.0,
            'beta': 0.0,
        }
        signals = {
            'theta': 0.707,
            'gamma': 1.0,
            'delta': 0.0,
            'alpha': 0.0,
            'beta': 0.0,
        }
        
        encoder.set_oscillator_phases(phases, signals)
        
        assert encoder._theta_phase == math.pi / 4
        assert encoder._gamma_phase == math.pi / 2
        assert encoder._theta_signal == 0.707
        assert encoder._gamma_signal == 1.0
    
    def test_encoding_phase_calculation(self, wm_config):
        """Test theta phase assignment for items."""
        encoder = ThetaGammaEncoder(wm_config)
        
        # First item at phase 0
        theta0, gamma0 = encoder.get_encoding_phase(0)
        assert theta0 == 0.0
        assert abs(gamma0 - math.pi/2) < 0.01  # Peak gamma
        
        # Fourth item at phase π
        theta4, gamma4 = encoder.get_encoding_phase(4)
        assert abs(theta4 - math.pi) < 0.01
        assert abs(gamma4 - math.pi/2) < 0.01
        
        # Eighth item wraps to 0
        theta8, gamma8 = encoder.get_encoding_phase(8)
        assert abs(theta8 - 0.0) < 0.01
    
    def test_retrieval_phase_calculation(self, wm_config):
        """Test phase calculation for N-back retrieval."""
        encoder = ThetaGammaEncoder(wm_config)
        
        # 2-back from position 5
        phase = encoder.get_retrieval_phase(current_index=5, n_back=2)
        expected = (3 / 8) * (2 * math.pi)  # Item 3 (5-2)
        assert abs(phase - expected) < 0.01
        
        # Can't retrieve before start
        phase = encoder.get_retrieval_phase(current_index=1, n_back=2)
        assert phase < 0  # Error indicator
    
    def test_get_current_phases(self, wm_config):
        """Test getter methods for current phases."""
        encoder = ThetaGammaEncoder(wm_config)
        
        # Set phases
        encoder.set_oscillator_phases(
            {'theta': 1.5, 'gamma': 2.0},
            {'theta': 0.5, 'gamma': 0.8}
        )
        
        assert encoder.get_current_theta_phase() == 1.5
        assert encoder.get_current_gamma_phase() == 2.0
    
    def test_excitability_modulation(self, wm_config):
        """Test gamma-based excitability modulation."""
        encoder = ThetaGammaEncoder(wm_config)
        
        # At gamma peak (π/2)
        encoder.set_oscillator_phases(
            {'gamma': math.pi / 2, 'theta': 0.0},
            {'gamma': 1.0, 'theta': 0.0}
        )
        excitability_peak = encoder.get_excitability_modulation()
        assert excitability_peak > 0.9  # Near maximum
        
        # At gamma trough (3π/2)
        encoder.set_oscillator_phases(
            {'gamma': 3 * math.pi / 2, 'theta': 0.0},
            {'gamma': -1.0, 'theta': 0.0}
        )
        excitability_trough = encoder.get_excitability_modulation()
        assert excitability_trough < 0.1  # Near minimum
    
    def test_excitability_varies_continuously(self, wm_config):
        """Test that excitability varies smoothly with gamma phase."""
        encoder = ThetaGammaEncoder(wm_config)
        
        excitabilities = []
        phases = [i * math.pi / 8 for i in range(16)]  # 0 to 2π
        
        for phase in phases:
            encoder.set_oscillator_phases(
                {'gamma': phase, 'theta': 0.0},
                {'gamma': math.sin(phase), 'theta': 0.0}
            )
            excitabilities.append(encoder.get_excitability_modulation())
        
        # Check continuous variation
        assert min(excitabilities) < 0.2
        assert max(excitabilities) > 0.8
        
        # Should be sinusoidal
        assert all(0.0 <= e <= 1.0 for e in excitabilities)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
