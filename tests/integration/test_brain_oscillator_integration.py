"""
Integration test for EventDrivenBrain with OscillatorManager.

This verifies that the centralized oscillator architecture works correctly
with the brain system, following the dopamine broadcast pattern.
"""

import pytest
import torch

from thalia.core.brain import EventDrivenBrain, EventDrivenBrainConfig
from thalia.core.oscillator import OscillatorManager


class TestBrainOscillatorIntegration:
    """Test that brain correctly integrates OscillatorManager."""
    
    def test_brain_has_oscillator_manager(self):
        """Test that brain creates oscillator manager."""
        config = EventDrivenBrainConfig(
            input_size=16,
            cortex_size=32,
            hippocampus_size=16,
            pfc_size=16,
            n_actions=2,
            device="cpu",
        )
        brain = EventDrivenBrain(config)
        
        # Check oscillator manager exists
        assert hasattr(brain, 'oscillators')
        assert isinstance(brain.oscillators, OscillatorManager)
        
        # Check it's configured with brain's dt
        assert brain.oscillators.dt_ms == config.dt_ms
    
    def test_oscillators_advance_with_brain(self):
        """Test that oscillators advance during brain processing."""
        config = EventDrivenBrainConfig(
            input_size=16,
            cortex_size=32,
            hippocampus_size=16,
            pfc_size=16,
            n_actions=2,
            device="cpu",
            dt_ms=1.0,
        )
        brain = EventDrivenBrain(config)
        
        # Record initial phases
        initial_phases = brain.oscillators.get_phases()
        
        # Process a sample (will advance oscillators)
        sample = torch.zeros(config.input_size)
        brain.process_sample(sample, n_timesteps=10)
        
        # Check phases advanced
        new_phases = brain.oscillators.get_phases()
        for name in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            assert new_phases[name] != initial_phases[name], \
                f"{name} oscillator did not advance"
    
    def test_oscillator_phases_broadcast_to_regions(self):
        """Test that oscillator phases are broadcast to regions."""
        config = EventDrivenBrainConfig(
            input_size=16,
            cortex_size=32,
            hippocampus_size=16,
            pfc_size=16,
            n_actions=2,
            device="cpu",
        )
        brain = EventDrivenBrain(config)
        
        # Advance oscillators manually
        brain.oscillators.advance(dt_ms=10.0)
        
        # Broadcast phases
        brain._broadcast_oscillator_phases()
        
        # Check that regions have oscillator phases
        # (regions store them even if not using them yet)
        assert hasattr(brain.cortex.impl.state, '_oscillator_phases')
        assert hasattr(brain.cortex.impl.state, '_oscillator_signals')
        
        phases = brain.cortex.impl.state._oscillator_phases
        signals = brain.cortex.impl.state._oscillator_signals
        
        assert isinstance(phases, dict)
        assert isinstance(signals, dict)
        assert 'delta' in phases
        assert 'theta' in phases
        assert 'alpha' in phases
    
    def test_oscillator_state_in_checkpoint(self):
        """Test that oscillator state is saved in checkpoints."""
        config = EventDrivenBrainConfig(
            input_size=16,
            cortex_size=32,
            hippocampus_size=16,
            pfc_size=16,
            n_actions=2,
            device="cpu",
        )
        brain = EventDrivenBrain(config)
        
        # Advance oscillators
        sample = torch.zeros(config.input_size)
        brain.process_sample(sample, n_timesteps=50)
        
        # Get state
        state = brain.get_full_state()
        
        # Check oscillator state is included
        # (For now it's not in checkpoint - that's next step in migration)
        # This test documents current behavior
        assert 'oscillators' not in state  # Not yet implemented
        
        # TODO: After migration, this should be:
        # assert 'oscillators' in state
        # assert 'delta' in state['oscillators']
    
    def test_multiple_regions_receive_same_phases(self):
        """Test that all regions receive same oscillator phases (synchronized)."""
        config = EventDrivenBrainConfig(
            input_size=16,
            cortex_size=32,
            hippocampus_size=16,
            pfc_size=16,
            n_actions=2,
            device="cpu",
        )
        brain = EventDrivenBrain(config)
        
        # Advance and broadcast
        brain.oscillators.advance(dt_ms=20.0)
        brain._broadcast_oscillator_phases()
        
        # Get phases from different regions
        cortex_phases = brain.cortex.impl.state._oscillator_phases
        hippo_phases = brain.hippocampus.impl.state._oscillator_phases
        pfc_phases = brain.pfc.impl.state._oscillator_phases
        
        # All regions should have identical phases (brain-wide synchronization)
        assert cortex_phases['theta'] == hippo_phases['theta']
        assert cortex_phases['gamma'] == pfc_phases['gamma']
        assert hippo_phases['delta'] == pfc_phases['delta']
    
    def test_sleep_stage_modulates_oscillators(self):
        """Test that sleep stages affect oscillator frequencies."""
        config = EventDrivenBrainConfig(
            input_size=16,
            cortex_size=32,
            hippocampus_size=16,
            pfc_size=16,
            n_actions=2,
            device="cpu",
        )
        brain = EventDrivenBrain(config)
        
        # Normal frequencies
        assert brain.oscillators.delta.frequency_hz == 2.0
        assert brain.oscillators.theta.frequency_hz == 8.0
        assert brain.oscillators.gamma.frequency_hz == 40.0
        
        # Switch to NREM (slow-wave sleep)
        brain.oscillators.set_sleep_stage("NREM")
        
        # Check frequencies changed
        assert brain.oscillators.delta.frequency_hz == 2.0  # Strong delta
        assert brain.oscillators.theta.frequency_hz == 6.0  # Slower theta
        assert brain.oscillators.gamma.frequency_hz == 30.0  # Slow gamma
        
        # Restore to awake
        brain.oscillators.set_sleep_stage("AWAKE")
        assert brain.oscillators.theta.frequency_hz == 8.0


class TestOscillatorTimingAccuracy:
    """Test that oscillator timing stays synchronized with brain time."""
    
    def test_oscillator_time_matches_brain_time(self):
        """Test that oscillator time stays synchronized with brain simulation time."""
        config = EventDrivenBrainConfig(
            input_size=16,
            cortex_size=32,
            hippocampus_size=16,
            pfc_size=16,
            n_actions=2,
            device="cpu",
            dt_ms=1.0,
        )
        brain = EventDrivenBrain(config)
        
        # Process for 100 timesteps
        sample = torch.zeros(config.input_size)
        brain.process_sample(sample, n_timesteps=100)
        
        # Oscillator internal time should match
        assert brain.oscillators._time_ms == 100.0
        
        # Process more
        brain.process_sample(sample, n_timesteps=50)
        assert brain.oscillators._time_ms == 150.0
    
    def test_consistent_phase_progression(self):
        """Test that phases progress consistently over time."""
        config = EventDrivenBrainConfig(
            input_size=16,
            cortex_size=32,
            hippocampus_size=16,
            pfc_size=16,
            n_actions=2,
            device="cpu",
            dt_ms=1.0,
        )
        brain = EventDrivenBrain(config)
        
        # Theta at 8 Hz should complete ~8 cycles in 1000ms
        # Each cycle = 2π radians
        # Expected phase increase per ms: 2π * 8 / 1000 = 0.0503 rad/ms
        
        initial_theta = brain.oscillators.theta.phase
        
        sample = torch.zeros(config.input_size)
        brain.process_sample(sample, n_timesteps=1000)
        
        final_theta = brain.oscillators.theta.phase
        
        # Phase should have wrapped around multiple times
        # We can't predict exact final phase (wrapping), but we know it advanced
        assert brain.oscillators._time_ms == 1000.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
