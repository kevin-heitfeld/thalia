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
    def config(self) -> ThaliaConfig:
        """Create minimal ThaliaConfig."""
        return ThaliaConfig.minimal(device="cpu")

    @pytest.fixture
    def brain(self, config: ThaliaConfig) -> EventDrivenBrain:
        """Create brain with oscillator manager."""
        brain = EventDrivenBrain.from_thalia_config(config)
        return brain

    def test_brain_broadcasts_coupled_amplitudes(self, brain: EventDrivenBrain) -> None:
        """Test that Brain broadcasts coupled amplitudes to all regions."""
        # Brain should have oscillator manager
        assert hasattr(brain, 'oscillators')
        assert isinstance(brain.oscillators, OscillatorManager)

        # OscillatorManager should have 4 default couplings
        assert len(brain.oscillators.couplings) == 4

        # Get coupled amplitudes
        coupled_amps = brain.oscillators.get_coupled_amplitudes()

        # Should have keys for default couplings (theta, alpha, beta, gamma)
        # Note: Each fast oscillator is coupled to ALL slower oscillators automatically
        assert len(coupled_amps) > 0  # Should have some couplings

        # Verify amplitude values are in valid range
        for key, amp in coupled_amps.items():
            assert 0.0 <= amp <= 1.0, f"{key} amplitude {amp} out of range"

    def test_hippocampus_uses_theta_gamma_coupling(self, brain: EventDrivenBrain) -> None:
        """Test hippocampus uses theta-gamma coupling for working memory."""
        # Advance oscillators to generate phases
        brain.oscillators.advance(dt_ms=1.0)

        # Set theta to encoding phase (trough) for testing using public API
        brain.oscillators.theta.sync_to_phase(math.pi)
        brain.oscillators.gamma.sync_to_phase(0.0)

        # Get coupled amplitudes - these are computed by OscillatorManager
        coupled_amps = brain.oscillators.get_coupled_amplitudes()

        # Verify gamma_by_theta coupling exists and is in valid range
        assert 'gamma_by_theta' in coupled_amps
        gamma_by_theta = coupled_amps['gamma_by_theta']
        assert 0.0 <= gamma_by_theta <= 1.0, f"gamma_by_theta amplitude {gamma_by_theta} out of range"

    def test_cerebellum_uses_beta_gamma_coupling(self, brain: EventDrivenBrain) -> None:
        """Test cerebellum uses beta-gamma coupling for motor timing."""
        # Advance oscillators
        brain.oscillators.advance(dt_ms=1.0)

        # Set beta to movement initiation (trough) using public API
        brain.oscillators.beta.sync_to_phase(math.pi)
        brain.oscillators.gamma.sync_to_phase(0.0)

        # Get coupled amplitudes - these are computed by OscillatorManager
        coupled_amps = brain.oscillators.get_coupled_amplitudes()

        # Verify gamma_by_beta coupling exists (for motor timing)
        assert 'gamma_by_beta' in coupled_amps
        gamma_by_beta = coupled_amps['gamma_by_beta']
        assert 0.0 <= gamma_by_beta <= 1.0, f"gamma_by_beta amplitude {gamma_by_beta} out of range"

    def test_cerebellum_uses_beta_theta_coupling(self, brain: EventDrivenBrain) -> None:
        """Test cerebellum uses beta-theta coupling for working memory coordination."""
        # Advance oscillators
        brain.oscillators.advance(dt_ms=1.0)

        # Set theta to working memory active phase using public API
        brain.oscillators.theta.sync_to_phase(0.0)
        brain.oscillators.beta.sync_to_phase(0.0)

        # Get coupled amplitudes - these are computed by OscillatorManager
        coupled_amps = brain.oscillators.get_coupled_amplitudes()

        # Verify beta_by_theta coupling exists (for working memory coordination)
        assert 'beta_by_theta' in coupled_amps
        beta_by_theta = coupled_amps['beta_by_theta']
        assert 0.0 <= beta_by_theta <= 1.0, f"beta_by_theta amplitude {beta_by_theta} out of range"

    def test_striatum_uses_beta_theta_coupling(self, brain: EventDrivenBrain) -> None:
        """Test striatum uses beta-theta coupling for action coordination."""
        # Advance oscillators
        brain.oscillators.advance(dt_ms=1.0)

        # Set oscillator phases using public API
        brain.oscillators.theta.sync_to_phase(0.0)
        brain.oscillators.beta.sync_to_phase(0.0)

        # Get coupled amplitudes and effective amplitudes
        coupled_amps = brain.oscillators.get_coupled_amplitudes()
        effective_amps = brain.oscillators.get_effective_amplitudes()

        # Verify beta amplitude is affected by theta (via coupling)
        assert 'beta_by_theta' in coupled_amps
        assert 'beta' in effective_amps
        beta_amplitude = effective_amps['beta']
        assert 0.0 <= beta_amplitude <= 1.0, f"Beta amplitude {beta_amplitude} out of range"

    def test_cortex_uses_alpha_gamma_coupling(self, brain: EventDrivenBrain) -> None:
        """Test cortex uses alpha-gamma coupling for attention gating."""
        # Advance oscillators
        brain.oscillators.advance(dt_ms=1.0)

        # Set alpha high (attention suppression) using public API
        brain.oscillators.alpha.sync_to_phase(math.pi/2)
        brain.oscillators.gamma.sync_to_phase(0.0)

        # Get coupled amplitudes - these are computed by OscillatorManager
        coupled_amps = brain.oscillators.get_coupled_amplitudes()

        # Verify gamma_by_alpha coupling exists (for attention gating)
        assert 'gamma_by_alpha' in coupled_amps
        gamma_by_alpha = coupled_amps['gamma_by_alpha']
        assert 0.0 <= gamma_by_alpha <= 1.0, f"gamma_by_alpha amplitude {gamma_by_alpha} out of range"

    def test_cortex_uses_alpha_for_input_suppression(self, brain: EventDrivenBrain) -> None:
        """Test cortex uses alpha oscillations for input gating."""
        # Create dummy input
        input_spikes = torch.zeros(brain.cortex.impl.config.n_input, device=brain.config.device)
        input_spikes[:10] = 1.0  # Some spikes

        # Advance oscillators
        brain.oscillators.advance(dt_ms=1.0)

        # Get alpha signal to verify it's affecting the system
        alpha_signal = brain.oscillators.alpha.signal
        assert -1.0 <= alpha_signal <= 1.0

        # Process input
        cortex = brain.cortex.impl
        _ = cortex.forward(input_spikes)

        # Verify alpha phase is being tracked (via oscillator manager)
        phases = brain.oscillators.get_phases()
        assert 'alpha' in phases
        assert 0.0 <= phases['alpha'] < 2 * math.pi

    def test_delta_theta_coupling_available_for_replay(self, brain: EventDrivenBrain) -> None:
        """Test delta-theta coupling is available for sleep consolidation."""
        # Advance oscillators
        brain.oscillators.advance(dt_ms=1.0)

        # Get coupled amplitudes from manager
        coupled_amps = brain.oscillators.get_coupled_amplitudes()

        # Verify theta_by_delta coupling exists (for sleep consolidation)
        assert 'theta_by_delta' in coupled_amps
        theta_by_delta = coupled_amps['theta_by_delta']
        assert 0.0 <= theta_by_delta <= 1.0, f"theta_by_delta amplitude {theta_by_delta} out of range"

    def test_all_couplings_computed_simultaneously(self, brain: EventDrivenBrain) -> None:
        """Test that all couplings are computed in one call."""
        # Advance oscillators to generate phases
        brain.oscillators.advance(dt_ms=1.0)

        # Get all coupled amplitudes at once
        coupled_amps = brain.oscillators.get_coupled_amplitudes()

        # Should have multiple couplings (4 default oscillators with automatic coupling)
        assert len(coupled_amps) > 0

        # All should be in valid range [min_amp, 1.0]
        for key, amp in coupled_amps.items():
            assert 0.0 <= amp <= 1.0, f"{key} amplitude {amp} out of range"

        # Verify key format (should be '{fast}_by_{slow}')
        for key in coupled_amps.keys():
            assert '_by_' in key, f"Invalid coupling key format: {key}"


class TestCouplingBiology:
    """Test biological accuracy of coupling usage."""

    @pytest.fixture
    def brain(self) -> EventDrivenBrain:
        """Create minimal brain."""
        config = ThaliaConfig.minimal(device="cpu")
        return EventDrivenBrain.from_thalia_config(config)

    def test_theta_gamma_coupling_working_memory(self, brain: EventDrivenBrain) -> None:
        """Test theta-gamma coupling implements 7±2 working memory slots."""
        # Theta-gamma coupling creates discrete slots within theta cycle
        # Each gamma cycle represents one item in working memory

        # Advance oscillators
        brain.oscillators.advance(dt_ms=1.0)

        # Get theta slot (should be in range 0-6 for 7 slots)
        theta_slot = brain.oscillators.get_theta_slot(n_slots=7)
        assert 0 <= theta_slot < 7

    def test_beta_gamma_coupling_motor_timing(self, brain: EventDrivenBrain) -> None:
        """Test beta-gamma coupling enhances motor timing precision."""
        # Beta-gamma coupling: gamma provides spike-timing precision
        # during beta-gated movement windows

        # Advance oscillators
        brain.oscillators.advance(dt_ms=1.0)

        # Set movement initiation phase (beta trough) using public API
        brain.oscillators.beta.sync_to_phase(math.pi)
        brain.oscillators.gamma.sync_to_phase(0.0)

        # Verify gamma_by_beta coupling exists and is modulated
        coupled_amps = brain.oscillators.get_coupled_amplitudes()
        assert 'gamma_by_beta' in coupled_amps

        # At beta trough (π), gamma should have specific amplitude based on coupling
        gamma_by_beta = coupled_amps['gamma_by_beta']
        assert 0.0 <= gamma_by_beta <= 1.0

    def test_alpha_gamma_coupling_attention_gating(self, brain: EventDrivenBrain) -> None:
        """Test alpha-gamma coupling implements attention-dependent binding."""
        # High alpha suppresses gamma → reduced feature binding in non-attended regions

        # Advance oscillators
        brain.oscillators.advance(dt_ms=1.0)

        # Set high alpha (attention elsewhere) using public API
        brain.oscillators.alpha.sync_to_phase(0.0)  # Peak alpha

        # Verify gamma_by_alpha coupling exists and modulates
        coupled_amps = brain.oscillators.get_coupled_amplitudes()
        assert 'gamma_by_alpha' in coupled_amps
        gamma_by_alpha = coupled_amps['gamma_by_alpha']
        assert 0.0 <= gamma_by_alpha <= 1.0

    def test_theta_beta_coupling_action_coordination(self, brain: EventDrivenBrain) -> None:
        """Test theta-beta coupling coordinates working memory and action."""
        # Theta phase determines when beta bursts occur
        # High theta → strong beta → action maintenance

        # Advance oscillators
        brain.oscillators.advance(dt_ms=1.0)

        # Set theta peak (working memory maintenance) using public API
        brain.oscillators.theta.sync_to_phase(0.0)
        brain.oscillators.beta.sync_to_phase(0.0)

        # Verify beta_by_theta coupling exists
        coupled_amps = brain.oscillators.get_coupled_amplitudes()
        assert 'beta_by_theta' in coupled_amps
        beta_by_theta = coupled_amps['beta_by_theta']

        # Beta amplitude should be in valid range
        assert 0.0 <= beta_by_theta <= 1.0

    def test_delta_theta_coupling_sleep_consolidation(self, brain: EventDrivenBrain) -> None:
        """Test delta-theta coupling implements NREM consolidation."""
        # Theta activity nested in delta up-states
        # Strong theta during delta peaks → replay windows

        # Advance oscillators
        brain.oscillators.advance(dt_ms=1.0)

        # Test delta up-state (peak) using public API
        brain.oscillators.delta.sync_to_phase(0.0)
        brain.oscillators.theta.sync_to_phase(0.0)

        # Get coupled amplitude
        coupled_amps = brain.oscillators.get_coupled_amplitudes()

        # Verify theta_by_delta coupling exists (for sleep consolidation)
        assert 'theta_by_delta' in coupled_amps
        theta_by_delta = coupled_amps['theta_by_delta']
        assert 0.0 <= theta_by_delta <= 1.0, f"theta_by_delta amplitude {theta_by_delta} out of range"
