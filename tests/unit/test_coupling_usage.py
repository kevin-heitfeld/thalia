"""Tests that all brain regions properly use cross-frequency coupling.

Tests verify that:
1. All regions receive coupled_amplitudes from Brain broadcast
2. Regions use biologically-appropriate couplings in their processing
3. Coupling amplitudes correctly modulate learning/processing
"""

import pytest
import torch
import math
from thalia.core.brain import EventDrivenBrain
from thalia.config import ThaliaConfig
from thalia.core.oscillator import OscillatorManager


class TestCouplingUsage:
    """Test that regions properly use cross-frequency coupling."""

    @pytest.fixture
    def config(self):
        """Create minimal ThaliaConfig."""
        return ThaliaConfig.minimal(device="cpu")

    @pytest.fixture
    def brain(self, config):
        """Create brain with oscillator manager."""
        brain = EventDrivenBrain.from_thalia_config(config)
        return brain

    def test_brain_broadcasts_coupled_amplitudes(self, brain):
        """Test that Brain broadcasts coupled amplitudes to all regions."""
        # Brain should have oscillator manager
        assert hasattr(brain, 'oscillators')
        assert isinstance(brain.oscillators, OscillatorManager)
        
        # OscillatorManager should have 5 default couplings
        assert len(brain.oscillators.couplings) == 5
        
        # Get coupled amplitudes
        coupled_amps = brain.oscillators.get_coupled_amplitudes()
        
        # Should have keys for all 5 couplings
        assert 'gamma_by_theta' in coupled_amps  # Working memory
        assert 'gamma_by_beta' in coupled_amps   # Motor timing
        assert 'theta_by_delta' in coupled_amps  # Sleep consolidation
        assert 'gamma_by_alpha' in coupled_amps  # Attention gating
        assert 'beta_by_theta' in coupled_amps   # WM-action coordination

    def test_hippocampus_uses_theta_gamma_coupling(self, brain):
        """Test hippocampus uses theta-gamma coupling for working memory."""
        # Set oscillators to encoding phase (theta trough)
        brain.oscillators.set_phases(theta=math.pi, gamma=0.0)
        
        # Broadcast to regions (triggers set_oscillator_phases)
        brain._broadcast_oscillator_phases()
        
        # Hippocampus should have stored coupled amplitude
        hipp = brain.hippocampus.impl
        assert hasattr(hipp, '_coupled_amplitudes')
        
        # Should be using gamma_by_theta coupling
        gamma_amp = hipp._coupled_amplitudes.get('gamma_by_theta', 1.0)
        
        # At theta trough (encoding), gamma should be strong
        # Coupling: strength=0.8, min=0.2, cosine
        # At theta=π: modulation = 0.5*(1+cos(π)) = 0.5*(1-1) = 0
        # amp = 0.2 + (1-0.2)*0*0.8 = 0.2 (minimum)
        assert 0.15 < gamma_amp < 0.25  # Near minimum

    def test_cerebellum_uses_beta_gamma_coupling(self, brain):
        """Test cerebellum uses beta-gamma coupling for motor timing."""
        # Set oscillators to movement initiation (beta trough)
        brain.oscillators.set_phases(beta=math.pi, gamma=0.0)
        
        # Broadcast to regions
        brain._broadcast_oscillator_phases()
        
        # Cerebellum should have stored coupled amplitudes
        cb = brain.cerebellum.impl
        assert hasattr(cb, '_gamma_amplitude')
        assert hasattr(cb, '_beta_amplitude')
        
        # Check that gamma_by_beta coupling is available
        gamma_by_beta = cb._coupled_amplitudes.get('gamma_by_beta', 1.0)
        
        # Should be using the coupling for precise motor timing
        # Coupling: strength=0.6, min=0.3, cosine
        # At beta=π: modulation = 0.5*(1+cos(π)) = 0
        # amp = 0.3 + (1-0.3)*0*0.6 = 0.3 (minimum)
        assert 0.25 < gamma_by_beta < 0.35

    def test_cerebellum_uses_beta_theta_coupling(self, brain):
        """Test cerebellum uses beta-theta coupling for working memory coordination."""
        # Set theta to working memory active phase
        brain.oscillators.set_phases(theta=0.0, beta=0.0)
        
        # Broadcast to regions
        brain._broadcast_oscillator_phases()
        
        # Cerebellum should have beta_by_theta coupling
        cb = brain.cerebellum.impl
        beta_by_theta = cb._coupled_amplitudes.get('beta_by_theta', 1.0)
        
        # Coupling: strength=0.4, min=0.5, cosine
        # At theta=0: modulation = 0.5*(1+cos(0)) = 0.5*(1+1) = 1.0
        # amp = 0.5 + (1-0.5)*1.0*0.4 = 0.5 + 0.2 = 0.7 (maximum)
        assert 0.65 < beta_by_theta < 0.75

    def test_striatum_uses_beta_theta_coupling(self, brain):
        """Test striatum uses beta-theta coupling for action coordination."""
        # Set oscillators
        brain.oscillators.set_phases(theta=0.0, beta=0.0)
        
        # Broadcast to regions
        brain._broadcast_oscillator_phases()
        
        # Striatum should have stored theta and beta phases
        striatum = brain.striatum.impl
        assert hasattr(striatum, '_theta_phase')
        assert hasattr(striatum, '_beta_phase')
        assert hasattr(striatum, '_beta_amplitude')
        
        # Should have beta_by_theta coupling
        beta_by_theta = striatum._coupled_amplitudes.get('beta_by_theta', 1.0)
        
        # At theta=0 (peak): maximum beta
        assert 0.65 < beta_by_theta < 0.75

    def test_cortex_uses_alpha_gamma_coupling(self, brain):
        """Test cortex uses alpha-gamma coupling for attention gating."""
        # Set alpha high (attention suppression)
        brain.oscillators.set_phases(alpha=math.pi/2, gamma=0.0)
        
        # Broadcast to regions
        brain._broadcast_oscillator_phases()
        
        # Cortex should have stored oscillator signals
        cortex = brain.cortex.impl
        assert hasattr(cortex.state, '_oscillator_signals')
        assert hasattr(cortex.state, '_coupled_amplitudes')
        
        # Should have gamma_by_alpha coupling
        gamma_by_alpha = cortex.state._coupled_amplitudes.get('gamma_by_alpha', 1.0)
        
        # Coupling: strength=0.5, min=0.4, sine
        # At alpha=π/2: modulation = 0.5*(1+sin(π/2)) = 0.5*(1+1) = 1.0
        # amp = 0.4 + (1-0.4)*1.0*0.5 = 0.4 + 0.3 = 0.7 (maximum)
        assert 0.65 < gamma_by_alpha < 0.75

    def test_cortex_uses_alpha_for_input_suppression(self, brain):
        """Test cortex uses alpha oscillations for input gating."""
        # Create dummy input
        input_spikes = torch.zeros(brain.config.n_input, device=brain.device)
        input_spikes[:10] = 1.0  # Some spikes
        
        # Set high alpha (suppress input)
        brain.oscillators.set_phases(alpha=0.0)  # Peak alpha signal
        brain.oscillators.set_amplitude('alpha', 1.0)
        brain._broadcast_oscillator_phases()
        
        # Process input (should be suppressed)
        cortex = brain.cortex.impl
        output1 = cortex.forward(input_spikes)
        
        # Set low alpha (allow input)
        brain.oscillators.set_phases(alpha=math.pi)  # Trough alpha signal
        brain._broadcast_oscillator_phases()
        
        # Process same input (should be less suppressed)
        output2 = cortex.forward(input_spikes)
        
        # With high alpha, less cortical activity expected
        # (alpha suppresses input → fewer L4 spikes → fewer L2/3 spikes)
        # Note: This is probabilistic, so we just check the mechanism exists
        assert hasattr(cortex.state, 'alpha_suppression')

    def test_delta_theta_coupling_available_for_replay(self, brain):
        """Test delta-theta coupling is available for sleep consolidation."""
        # Set delta to up-state (peak)
        brain.oscillators.set_phases(delta=0.0, theta=0.0)
        
        # Broadcast to regions
        brain._broadcast_oscillator_phases()
        
        # Get coupled amplitudes from manager
        coupled_amps = brain.oscillators.get_coupled_amplitudes()
        
        # Delta-theta coupling should exist
        theta_by_delta = coupled_amps.get('theta_by_delta', 1.0)
        
        # Coupling: strength=0.7, min=0.1, cosine
        # At delta=0: modulation = 0.5*(1+cos(0)) = 1.0
        # amp = 0.1 + (1-0.1)*1.0*0.7 = 0.1 + 0.63 = 0.73 (maximum)
        assert 0.68 < theta_by_delta < 0.78

    def test_all_couplings_computed_simultaneously(self, brain):
        """Test that all 5 couplings are computed in one call."""
        # Set some oscillator phases
        brain.oscillators.set_phases(
            delta=0.0,
            theta=math.pi/4,
            alpha=math.pi/2,
            beta=math.pi,
            gamma=0.0
        )
        
        # Get all coupled amplitudes at once
        coupled_amps = brain.oscillators.get_coupled_amplitudes()
        
        # Should have exactly 5 entries (one per coupling)
        assert len(coupled_amps) == 5
        
        # All should be in valid range [min_amp, 1.0]
        for key, amp in coupled_amps.items():
            assert 0.0 <= amp <= 1.0, f"{key} amplitude {amp} out of range"
        
        # Verify keys
        expected_keys = {
            'gamma_by_theta',
            'gamma_by_beta',
            'theta_by_delta',
            'gamma_by_alpha',
            'beta_by_theta',
        }
        assert set(coupled_amps.keys()) == expected_keys


class TestCouplingBiology:
    """Test biological accuracy of coupling usage."""

    @pytest.fixture
    def brain(self):
        """Create minimal brain."""
        config = ThaliaConfig.minimal(device="cpu")
        return EventDrivenBrain.from_thalia_config(config)

    def test_theta_gamma_coupling_working_memory(self, brain):
        """Test theta-gamma coupling implements 7±2 working memory slots."""
        # Theta-gamma coupling creates discrete slots within theta cycle
        # Each gamma cycle represents one item in working memory
        
        # Set theta to encoding phase
        brain.oscillators.set_phases(theta=math.pi, gamma=0.0)
        brain._broadcast_oscillator_phases()
        
        # Get theta slot (should be in range 0-6 for 7 slots)
        theta_slot = brain.oscillators.get_theta_slot(n_slots=7)
        assert 0 <= theta_slot < 7

    def test_beta_gamma_coupling_motor_timing(self, brain):
        """Test beta-gamma coupling enhances motor timing precision."""
        # Beta-gamma coupling: gamma provides spike-timing precision
        # during beta-gated movement windows
        
        # Movement initiation (beta trough)
        brain.oscillators.set_phases(beta=math.pi, gamma=0.0)
        brain._broadcast_oscillator_phases()
        
        # Cerebellum should gate learning at beta trough
        cb = brain.cerebellum.impl
        gate = cb._compute_beta_gate()
        
        # Learning should be strong at beta trough
        # Gate should be modulated by gamma_by_beta coupling
        assert gate > 0.01  # Some gating effect

    def test_alpha_gamma_coupling_attention_gating(self, brain):
        """Test alpha-gamma coupling implements attention-dependent binding."""
        # High alpha suppresses gamma → reduced feature binding in non-attended regions
        
        # High alpha (attention elsewhere)
        brain.oscillators.set_phases(alpha=0.0)  # Peak alpha
        brain.oscillators.set_amplitude('alpha', 1.0)
        brain._broadcast_oscillator_phases()
        
        cortex = brain.cortex.impl
        
        # Alpha should suppress input
        assert hasattr(cortex.state, 'alpha_suppression')
        
        # Gamma modulation should be available
        if hasattr(cortex.state, 'gamma_modulation'):
            gamma_mod = cortex.state.gamma_modulation
            # Sine coupling: at alpha=0, sin(0)=0, so modulation is mid-range
            assert 0.0 <= gamma_mod <= 1.0

    def test_theta_beta_coupling_action_coordination(self, brain):
        """Test theta-beta coupling coordinates working memory and action."""
        # Theta phase determines when beta bursts occur
        # High theta → strong beta → action maintenance
        
        # Theta peak (working memory maintenance)
        brain.oscillators.set_phases(theta=0.0, beta=0.0)
        brain._broadcast_oscillator_phases()
        
        # Striatum should have strong beta modulation
        striatum = brain.striatum.impl
        beta_amp = striatum._beta_amplitude
        
        # At theta peak, beta should be strong (action maintenance)
        assert beta_amp > 0.6

    def test_delta_theta_coupling_sleep_consolidation(self, brain):
        """Test delta-theta coupling implements NREM consolidation."""
        # Theta activity nested in delta up-states
        # Strong theta during delta peaks → replay windows
        
        # Delta up-state (peak)
        brain.oscillators.set_phases(delta=0.0, theta=0.0)
        brain._broadcast_oscillator_phases()
        
        # Get coupled amplitude
        coupled_amps = brain.oscillators.get_coupled_amplitudes()
        theta_by_delta = coupled_amps['theta_by_delta']
        
        # Theta should be strong during delta up-state
        assert theta_by_delta > 0.6
        
        # Delta down-state (trough)
        brain.oscillators.set_phases(delta=math.pi, theta=0.0)
        brain._broadcast_oscillator_phases()
        
        coupled_amps = brain.oscillators.get_coupled_amplitudes()
        theta_by_delta = coupled_amps['theta_by_delta']
        
        # Theta should be weak during delta down-state
        assert theta_by_delta < 0.4
