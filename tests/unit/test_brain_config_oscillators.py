"""
Unit tests for oscillator configuration in BrainConfig.

Tests that oscillator_couplings can be configured via BrainConfig
and properly passed through to OscillatorManager.

Author: Thalia Project
Date: December 2025
"""

import pytest
from thalia.config import ThaliaConfig, GlobalConfig, BrainConfig, RegionSizes
from thalia.core.oscillator import OscillatorCoupling
from thalia.core.brain import EventDrivenBrain


class TestBrainConfigOscillators:
    """Test oscillator configuration via BrainConfig."""

    def test_default_coupling_when_none(self):
        """Test that default theta-gamma coupling is used when config is None."""
        config = ThaliaConfig(
            global_=GlobalConfig(device="cpu"),
            brain=BrainConfig(
                sizes=RegionSizes(
                    input_size=32,
                    cortex_size=16,
                    hippocampus_size=8,
                    pfc_size=4,
                    n_actions=2,
                ),
                oscillator_couplings=None,  # Use default
            ),
        )
        
        brain = EventDrivenBrain.from_thalia_config(config)
        
        # Should have default theta-gamma coupling
        assert len(brain.oscillators.couplings) == 1
        coupling = brain.oscillators.couplings[0]
        assert coupling.slow_oscillator == 'theta'
        assert coupling.fast_oscillator == 'gamma'
        assert coupling.coupling_strength == 0.8

    def test_custom_single_coupling(self):
        """Test custom coupling configuration with single coupling."""
        config = ThaliaConfig(
            global_=GlobalConfig(device="cpu"),
            brain=BrainConfig(
                sizes=RegionSizes(
                    input_size=32,
                    cortex_size=16,
                    hippocampus_size=8,
                    pfc_size=4,
                    n_actions=2,
                ),
                oscillator_couplings=[
                    OscillatorCoupling(
                        slow_oscillator='delta',
                        fast_oscillator='theta',
                        coupling_strength=0.6,
                        min_amplitude=0.3,
                        modulation_type='cosine',
                    )
                ],
            ),
        )
        
        brain = EventDrivenBrain.from_thalia_config(config)
        
        # Should have custom delta-theta coupling
        assert len(brain.oscillators.couplings) == 1
        coupling = brain.oscillators.couplings[0]
        assert coupling.slow_oscillator == 'delta'
        assert coupling.fast_oscillator == 'theta'
        assert coupling.coupling_strength == 0.6
        assert coupling.min_amplitude == 0.3

    def test_custom_multiple_couplings(self):
        """Test custom configuration with multiple couplings."""
        config = ThaliaConfig(
            global_=GlobalConfig(device="cpu"),
            brain=BrainConfig(
                sizes=RegionSizes(
                    input_size=32,
                    cortex_size=16,
                    hippocampus_size=8,
                    pfc_size=4,
                    n_actions=2,
                ),
                oscillator_couplings=[
                    OscillatorCoupling('theta', 'gamma', coupling_strength=0.8),
                    OscillatorCoupling('delta', 'theta', coupling_strength=0.6),
                    OscillatorCoupling('alpha', 'gamma', coupling_strength=0.7),
                ],
            ),
        )
        
        brain = EventDrivenBrain.from_thalia_config(config)
        
        # Should have all three couplings
        assert len(brain.oscillators.couplings) == 3
        
        # Check each coupling
        assert brain.oscillators.couplings[0].slow_oscillator == 'theta'
        assert brain.oscillators.couplings[0].fast_oscillator == 'gamma'
        
        assert brain.oscillators.couplings[1].slow_oscillator == 'delta'
        assert brain.oscillators.couplings[1].fast_oscillator == 'theta'
        
        assert brain.oscillators.couplings[2].slow_oscillator == 'alpha'
        assert brain.oscillators.couplings[2].fast_oscillator == 'gamma'

    def test_empty_list_disables_coupling(self):
        """Test that empty list disables all coupling."""
        config = ThaliaConfig(
            global_=GlobalConfig(device="cpu"),
            brain=BrainConfig(
                sizes=RegionSizes(
                    input_size=32,
                    cortex_size=16,
                    hippocampus_size=8,
                    pfc_size=4,
                    n_actions=2,
                ),
                oscillator_couplings=[],  # Explicitly disable coupling
            ),
        )
        
        brain = EventDrivenBrain.from_thalia_config(config)
        
        # Should have no couplings
        assert len(brain.oscillators.couplings) == 0

    def test_coupled_amplitude_with_custom_coupling(self):
        """Test that custom couplings affect amplitude modulation."""
        config = ThaliaConfig(
            global_=GlobalConfig(device="cpu"),
            brain=BrainConfig(
                sizes=RegionSizes(
                    input_size=32,
                    cortex_size=16,
                    hippocampus_size=8,
                    pfc_size=4,
                    n_actions=2,
                ),
                oscillator_couplings=[
                    OscillatorCoupling(
                        slow_oscillator='delta',
                        fast_oscillator='theta',
                        coupling_strength=0.8,
                        min_amplitude=0.2,
                        modulation_type='cosine',
                    )
                ],
            ),
        )
        
        brain = EventDrivenBrain.from_thalia_config(config)
        
        # Test amplitude modulation at delta peak (cosine modulation: max at peak/0)
        brain.oscillators.delta._phase = 0.0  # Peak
        amplitude = brain.oscillators.get_coupled_amplitude('theta', 'delta')
        
        # At peak (cosine modulation), should be near maximum
        # With coupling_strength=0.8, min=0.2: max = 0.2 + (1-0.2)*0.8 = 0.84
        assert amplitude > 0.8
        assert amplitude <= 1.0
        
        # Test amplitude modulation at delta trough (should be min theta)
        brain.oscillators.delta._phase = 3.14159  # π (trough)
        amplitude = brain.oscillators.get_coupled_amplitude('theta', 'delta')
        
        # At trough, should be minimum (0.2)
        assert abs(amplitude - 0.2) < 0.1


class TestOscillatorBroadcastingWithConfig:
    """Test that oscillator broadcasting works with configured couplings."""

    def test_broadcast_with_custom_coupling(self):
        """Test that Brain broadcasts coupling info correctly with custom config."""
        config = ThaliaConfig(
            global_=GlobalConfig(device="cpu"),
            brain=BrainConfig(
                sizes=RegionSizes(
                    input_size=32,
                    cortex_size=16,
                    hippocampus_size=8,
                    pfc_size=4,
                    n_actions=2,
                ),
                oscillator_couplings=[
                    OscillatorCoupling('delta', 'theta', coupling_strength=0.6),
                ],
            ),
        )
        
        brain = EventDrivenBrain.from_thalia_config(config)
        
        # Advance oscillators
        brain.oscillators.advance(dt_ms=1.0)
        
        # Get broadcast info
        phases = brain.oscillators.get_phases()
        signals = brain.oscillators.get_signals()
        theta_slot = brain.oscillators.get_theta_slot()
        
        # Compute coupled amplitudes (what brain broadcasts)
        coupled_amplitudes = {}
        for coupling in brain.oscillators.couplings:
            key = f"{coupling.fast_oscillator}_{coupling.slow_oscillator}"
            coupled_amplitudes[key] = brain.oscillators.get_coupled_amplitude(
                coupling.fast_oscillator, coupling.slow_oscillator
            )
        
        # Should have delta-theta coupling
        assert 'theta_delta' in coupled_amplitudes
        amplitude = coupled_amplitudes['theta_delta']
        
        # Should be in valid range
        assert 0.0 <= amplitude <= 1.0

    def test_broadcast_with_no_coupling(self):
        """Test that broadcasting works even with no coupling configured."""
        config = ThaliaConfig(
            global_=GlobalConfig(device="cpu"),
            brain=BrainConfig(
                sizes=RegionSizes(
                    input_size=32,
                    cortex_size=16,
                    hippocampus_size=8,
                    pfc_size=4,
                    n_actions=2,
                ),
                oscillator_couplings=[],  # No coupling
            ),
        )
        
        brain = EventDrivenBrain.from_thalia_config(config)
        
        # Advance oscillators
        brain.oscillators.advance(dt_ms=1.0)
        
        # Get broadcast info
        phases = brain.oscillators.get_phases()
        signals = brain.oscillators.get_signals()
        theta_slot = brain.oscillators.get_theta_slot()
        
        # Compute coupled amplitudes (should be empty)
        coupled_amplitudes = {}
        for coupling in brain.oscillators.couplings:
            key = f"{coupling.fast_oscillator}_{coupling.slow_oscillator}"
            coupled_amplitudes[key] = brain.oscillators.get_coupled_amplitude(
                coupling.fast_oscillator, coupling.slow_oscillator
            )
        
        # Should have no coupled amplitudes
        assert len(coupled_amplitudes) == 0
        
        # But phases and signals should still work
        assert 'theta' in phases
        assert 'gamma' in signals
        assert 0 <= theta_slot < 7
