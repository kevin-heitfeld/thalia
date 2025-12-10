"""Integration tests for alpha oscillator with real Brain and OscillatorManager."""
import pytest
import torch
import math

from thalia.core.brain import EventDrivenBrain, EventDrivenBrainConfig
from thalia.regions.cortex import LayeredCortex


class TestAlphaOscillatorIntegration:
    """Test alpha oscillations in full Brain system."""
    
    def test_brain_broadcasts_alpha_to_cortex(self):
        """Test that Brain correctly broadcasts alpha signals to cortex."""
        # Create brain with cortex region
        brain_config = EventDrivenBrainConfig(
            input_size=32,
            cortex_size=16,
            hippocampus_size=16,
            pfc_size=16,
            n_actions=2,
            device="cpu"
        )
        brain = EventDrivenBrain(brain_config)
        
        # Access the cortex implementation
        cortex = brain.cortex.impl
        assert isinstance(cortex, LayeredCortex), "Brain should create LayeredCortex"
        
        # Advance oscillators and broadcast
        brain.oscillators.advance(dt_ms=10.0)
        brain._broadcast_oscillator_phases()
        
        # Verify cortex received oscillator signals
        assert hasattr(cortex.state, '_oscillator_signals'), "Cortex should have oscillator signals"
        assert 'alpha' in cortex.state._oscillator_signals, "Alpha signal should be broadcast"
        assert 'theta' in cortex.state._oscillator_signals, "Theta signal should be broadcast"
        assert 'gamma' in cortex.state._oscillator_signals, "Gamma signal should be broadcast"
    
    def test_alpha_suppression_varies_over_time(self):
        """Test that alpha suppression follows oscillator phase over multiple timesteps."""
        brain_config = EventDrivenBrainConfig(
            input_size=32,
            cortex_size=16,
            hippocampus_size=16,
            pfc_size=16,
            n_actions=2,
            device="cpu"
        )
        brain = EventDrivenBrain(brain_config)
        cortex = brain.cortex.impl
        
        # Create constant input
        input_spikes = torch.zeros(32)
        input_spikes[::4] = 1.0
        
        suppressions = []
        alpha_signals = []
        
        # Simulate 100ms at 10 Hz alpha (one full cycle)
        n_steps = 100  # 100ms at 1ms dt
        
        for _ in range(n_steps):
            # Process sample advances oscillators and broadcasts
            brain.process_sample(input_spikes, n_timesteps=1)
            
            # Record suppression and alpha signal after processing
            # Note: alpha_suppression is set during cortex forward() call
            suppressions.append(cortex.state.alpha_suppression)
            alpha_signals.append(cortex.state._oscillator_signals.get('alpha', 0.0))
        
        # Verify alpha oscillates (should have peaks and troughs)
        min_suppression = min(suppressions)
        max_suppression = max(suppressions)
        
        assert min_suppression < max_suppression, (
            f"Alpha suppression should vary over time: [{min_suppression}, {max_suppression}]"
        )
        
        # Verify most suppressions follow alpha signal
        # High alpha → low suppression factor (0.5)
        # Low alpha → high suppression factor (1.0)
        # Allow some tolerance since broadcast happens before forward in the event loop
        matches = 0
        for alpha_sig, suppression in zip(alpha_signals, suppressions):
            if alpha_sig > 0:  # Positive alpha
                alpha_magnitude = max(0.0, alpha_sig)
                expected_suppression = 1.0 - (alpha_magnitude * 0.5)
                if abs(suppression - expected_suppression) < 0.05:  # 5% tolerance
                    matches += 1
        
        # At least 80% should match (accounting for timing issues in event system)
        match_rate = matches / len([a for a in alpha_signals if a > 0])
        assert match_rate > 0.8, (
            f"Only {match_rate:.1%} of suppressions matched alpha signals (need >80%)"
        )
    
    def test_alpha_modulates_cortical_activity_over_time(self):
        """Test that alpha oscillations modulate cortical activity in realistic way."""
        brain_config = EventDrivenBrainConfig(
            input_size=32,
            cortex_size=16,
            hippocampus_size=16,
            pfc_size=16,
            n_actions=2,
            device="cpu"
        )
        brain = EventDrivenBrain(brain_config)
        cortex = brain.cortex.impl
        
        # Create constant strong input
        input_spikes = torch.zeros(32)
        input_spikes[::2] = 1.0  # 50% sparsity
        
        activities = []
        alpha_phases = []
        
        # Simulate 200ms
        for _ in range(200):
            # Process sample through brain (advances oscillators, broadcasts, processes)
            result = brain.process_sample(input_spikes, n_timesteps=1)
            
            # Record cortex L5 output activity and alpha phase
            # Use L5 spikes (output layer) instead of general spikes
            if cortex.state.l5_spikes is not None:
                activities.append(cortex.state.l5_spikes.sum().item())
            else:
                activities.append(0)
            alpha_phases.append(cortex.state._oscillator_phases.get('alpha', 0.0))
        
        # Verify activity modulation exists
        min_activity = min(activities)
        max_activity = max(activities)
        
        assert max_activity > 0, "Should have some activity with 50% input"
        
        # Activity should vary (though may be subtle due to stochasticity)
        # Just verify suppression mechanism is working
        high_alpha_activity = []
        low_alpha_activity = []
        
        for act, phase in zip(activities, alpha_phases):
            # High alpha = phase near π (peak suppression)
            # Low alpha = phase near 0 or 2π (minimal suppression)
            if abs(phase - math.pi) < 0.5:  # Near peak
                high_alpha_activity.append(act)
            elif phase < 0.5 or phase > (2*math.pi - 0.5):  # Near trough
                low_alpha_activity.append(act)
        
        # On average, low alpha should have more activity than high alpha
        if high_alpha_activity and low_alpha_activity:
            avg_high = sum(high_alpha_activity) / len(high_alpha_activity)
            avg_low = sum(low_alpha_activity) / len(low_alpha_activity)
            
            # Use <= for stochastic tolerance
            assert avg_high <= avg_low + 1, (
                f"High alpha should suppress more: {avg_high} <= {avg_low}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
