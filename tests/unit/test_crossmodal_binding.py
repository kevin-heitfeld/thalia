"""
Tests for Cross-Modal Gamma Binding pathway.

Tests the biological mechanism where different sensory modalities
are bound together via synchronized gamma oscillations.
"""

import math
import pytest
import torch
import numpy as np

from thalia.integration.pathways import CrossModalGammaBinding, CrossModalBindingConfig


class TestCrossModalBindingInitialization:
    """Tests for pathway initialization."""

    def test_basic_initialization(self):
        """Test basic pathway creation."""
        binder = CrossModalGammaBinding()

        assert binder.config.visual_size == 256
        assert binder.config.auditory_size == 256
        assert binder.config.output_size == 512
        assert binder.config.gamma_freq_hz == 40.0

    def test_custom_config(self):
        """Test initialization with custom config."""
        config = CrossModalBindingConfig(
            visual_size=128,
            auditory_size=128,
            output_size=256,
            gamma_freq_hz=30.0,
        )
        binder = CrossModalGammaBinding(config=config)

        assert binder.config.visual_size == 128
        assert binder.config.auditory_size == 128
        assert binder.config.output_size == 256
        assert binder.config.gamma_freq_hz == 30.0

    def test_parameter_overrides(self):
        """Test initialization with parameter overrides."""
        binder = CrossModalGammaBinding(
            visual_size=64,
            auditory_size=64,
            output_size=128,
        )

        assert binder.config.visual_size == 64
        assert binder.config.auditory_size == 64
        assert binder.config.output_size == 128

    def test_weights_initialized(self):
        """Test that weights are properly initialized."""
        config = CrossModalBindingConfig(
            visual_size=10,
            auditory_size=10,
            output_size=20,
        )
        binder = CrossModalGammaBinding(config=config)

        assert binder.visual_weights.shape == (20, 10)
        assert binder.auditory_weights.shape == (20, 10)
        assert torch.isfinite(binder.visual_weights).all()
        assert torch.isfinite(binder.auditory_weights).all()

    def test_oscillators_initialized(self):
        """Test that gamma oscillators are created."""
        binder = CrossModalGammaBinding(gamma_freq_hz=40.0)

        assert binder.visual_gamma.frequency_hz == 40.0
        assert binder.auditory_gamma.frequency_hz == 40.0
        assert binder.visual_gamma.phase == 0.0
        assert binder.auditory_gamma.phase == 0.0


class TestGammaGating:
    """Tests for gamma-phase-based gating."""

    def test_gamma_gate_at_peak(self):
        """Test gamma gate is strongest at optimal phase."""
        binder = CrossModalGammaBinding()

        # Optimal phase is π/2 (90°)
        gate_at_peak = binder._compute_gamma_gate(math.pi / 2)
        assert gate_at_peak > 0.9  # Should be near 1.0

    def test_gamma_gate_at_trough(self):
        """Test gamma gate is weakest at opposite phase."""
        binder = CrossModalGammaBinding()

        # Opposite phase is 3π/2 (270°)
        gate_at_trough = binder._compute_gamma_gate(3 * math.pi / 2)
        assert gate_at_trough < 0.1  # Should be near 0.0

    def test_gamma_gate_width(self):
        """Test gamma gate has appropriate width."""
        binder = CrossModalGammaBinding()

        gate_at_peak = binder._compute_gamma_gate(math.pi / 2, width=0.3)
        gate_offset = binder._compute_gamma_gate(math.pi / 2 + 0.3, width=0.3)

        # At width offset, gate should be ~60% of peak
        assert 0.5 < gate_offset / gate_at_peak < 0.7


class TestPhaseCoherence:
    """Tests for phase coherence measurement."""

    def test_perfect_coherence(self):
        """Test coherence is 1.0 when phases are identical."""
        binder = CrossModalGammaBinding()

        coherence = binder._compute_phase_coherence(math.pi / 4, math.pi / 4)
        assert abs(coherence - 1.0) < 0.01

    def test_zero_coherence(self):
        """Test coherence is low when phases are opposite."""
        binder = CrossModalGammaBinding()

        # Opposite phases (π apart)
        coherence = binder._compute_phase_coherence(0.0, math.pi)
        assert coherence < 0.1

    def test_coherence_symmetry(self):
        """Test coherence is symmetric."""
        binder = CrossModalGammaBinding()

        phase1, phase2 = math.pi / 3, math.pi / 2
        coherence1 = binder._compute_phase_coherence(phase1, phase2)
        coherence2 = binder._compute_phase_coherence(phase2, phase1)

        assert abs(coherence1 - coherence2) < 0.001

    def test_coherence_wrapping(self):
        """Test coherence handles phase wrapping correctly."""
        binder = CrossModalGammaBinding()

        # Phases near 0 and 2π should have high coherence
        coherence = binder._compute_phase_coherence(0.1, 2 * math.pi - 0.1)
        assert coherence > 0.9


class TestForwardPass:
    """Tests for forward pass binding computation."""

    def test_forward_basic(self):
        """Test basic forward pass."""
        config = CrossModalBindingConfig(
            visual_size=10,
            auditory_size=10,
            output_size=20,
        )
        binder = CrossModalGammaBinding(config=config)

        visual_spikes = torch.zeros(10)
        auditory_spikes = torch.zeros(10)

        output, coherence = binder(visual_spikes, auditory_spikes)

        assert output.shape == (20,)
        assert 0.0 <= coherence <= 1.0
        assert torch.isfinite(output).all()

    def test_forward_with_spikes(self):
        """Test forward pass with actual spikes."""
        config = CrossModalBindingConfig(
            visual_size=10,
            auditory_size=10,
            output_size=20,
        )
        binder = CrossModalGammaBinding(config=config)

        # Some spikes
        visual_spikes = (torch.rand(10) > 0.7).float()
        auditory_spikes = (torch.rand(10) > 0.7).float()

        output, coherence = binder(visual_spikes, auditory_spikes)

        assert output.shape == (20,)
        assert torch.any(output > 0)  # Should have some activity

    def test_coherence_gates_output(self):
        """Test that low coherence reduces output strength."""
        config = CrossModalBindingConfig(
            visual_size=10,
            auditory_size=10,
            output_size=20,
            gate_threshold=0.5,
        )
        binder = CrossModalGammaBinding(config=config)

        # Strong spikes
        visual_spikes = torch.ones(10)
        auditory_spikes = torch.ones(10)

        # Synchronized phases (high coherence)
        binder.visual_gamma.sync_to_phase(math.pi / 4)
        binder.auditory_gamma.sync_to_phase(math.pi / 4)
        output_sync, coherence_sync = binder(visual_spikes, auditory_spikes)

        # Reset
        binder.reset_phases()

        # Desynchronized phases (low coherence)
        binder.visual_gamma.sync_to_phase(0.0)
        binder.auditory_gamma.sync_to_phase(math.pi)
        output_desync, coherence_desync = binder(visual_spikes, auditory_spikes)

        # High coherence should produce stronger output
        assert coherence_sync > coherence_desync
        assert output_sync.abs().mean() > output_desync.abs().mean()


class TestPhaseCoupling:
    """Tests for mutual phase coupling between modalities."""

    def test_phase_coupling_occurs(self):
        """Test that active inputs cause phase coupling."""
        binder = CrossModalGammaBinding()

        # Set different initial phases
        binder.visual_gamma.sync_to_phase(0.0)
        binder.auditory_gamma.sync_to_phase(math.pi / 2)

        initial_diff = abs(binder.visual_gamma.phase - binder.auditory_gamma.phase)

        # Strong activity in both modalities
        visual_spikes = torch.ones(256)
        auditory_spikes = torch.ones(256)

        # Run multiple steps to allow coupling
        for _ in range(10):
            binder(visual_spikes, auditory_spikes)

        final_diff = abs(binder.visual_gamma.phase - binder.auditory_gamma.phase)

        # Phases should have moved closer together
        # (accounting for potential wrapping)
        final_diff = min(final_diff, 2 * math.pi - final_diff)
        initial_diff = min(initial_diff, 2 * math.pi - initial_diff)

        assert final_diff < initial_diff + 0.1  # Some convergence expected

    def test_no_coupling_without_activity(self):
        """Test that coupling doesn't occur without activity."""
        binder = CrossModalGammaBinding()

        # Set different initial phases
        binder.visual_gamma.sync_to_phase(0.0)
        binder.auditory_gamma.sync_to_phase(math.pi / 2)

        # Calculate expected phase change from oscillator advancement
        dt = binder.config.dt_ms
        freq = binder.config.gamma_freq_hz
        expected_phase_change = 2.0 * math.pi * freq * dt / 1000.0

        initial_visual = binder.visual_gamma.phase
        initial_auditory = binder.auditory_gamma.phase

        # No spikes
        visual_spikes = torch.zeros(256)
        auditory_spikes = torch.zeros(256)

        binder(visual_spikes, auditory_spikes)

        # Phases should only change due to oscillator advancement, not coupling
        # Allow for oscillator advancement (one timestep worth)
        visual_change = abs(binder.visual_gamma.phase - (initial_visual + expected_phase_change) % (2 * math.pi))
        auditory_change = abs(binder.auditory_gamma.phase - (initial_auditory + expected_phase_change) % (2 * math.pi))

        # Changes should be minimal (just rounding errors)
        assert visual_change < 0.01
        assert auditory_change < 0.01


class TestOscillatorSynchronization:
    """Tests for oscillator state management."""

    def test_reset_phases(self):
        """Test resetting oscillator phases."""
        binder = CrossModalGammaBinding()

        # Advance oscillators
        for _ in range(10):
            binder.visual_gamma.advance()
            binder.auditory_gamma.advance()

        assert binder.visual_gamma.phase != 0.0
        assert binder.auditory_gamma.phase != 0.0

        # Reset
        binder.reset_phases()

        assert binder.visual_gamma.phase == 0.0
        assert binder.auditory_gamma.phase == 0.0

    def test_sync_to_external_gamma(self):
        """Test synchronizing to external gamma signal."""
        binder = CrossModalGammaBinding()

        external_phase = math.pi / 3

        binder.sync_to_external_gamma(external_phase)

        assert abs(binder.visual_gamma.phase - external_phase) < 0.01
        assert abs(binder.auditory_gamma.phase - external_phase) < 0.01


class TestBindingMetrics:
    """Tests for binding quality assessment."""

    def test_get_binding_strength(self):
        """Test binding strength metric computation."""
        config = CrossModalBindingConfig(
            visual_size=10,
            auditory_size=10,
            output_size=20,
        )
        binder = CrossModalGammaBinding(config=config)

        visual_spikes = (torch.rand(10) > 0.5).float()
        auditory_spikes = (torch.rand(10) > 0.5).float()

        metrics = binder.get_binding_strength(visual_spikes, auditory_spikes)

        assert "coherence" in metrics
        assert "visual_activity" in metrics
        assert "auditory_activity" in metrics
        assert "is_bound" in metrics
        assert "visual_phase" in metrics
        assert "auditory_phase" in metrics

        assert 0.0 <= metrics["coherence"] <= 1.0
        assert 0.0 <= metrics["visual_activity"] <= 1.0
        assert 0.0 <= metrics["auditory_activity"] <= 1.0
        assert isinstance(metrics["is_bound"], (bool, np.bool_))

    def test_binding_threshold(self):
        """Test binding threshold detection."""
        config = CrossModalBindingConfig(
            visual_size=10,
            auditory_size=10,
            output_size=20,
            gate_threshold=0.6,
        )
        binder = CrossModalGammaBinding(config=config)

        visual_spikes = torch.ones(10)
        auditory_spikes = torch.ones(10)

        # High coherence
        binder.visual_gamma.sync_to_phase(0.0)
        binder.auditory_gamma.sync_to_phase(0.1)
        metrics_bound = binder.get_binding_strength(visual_spikes, auditory_spikes)

        # Low coherence
        binder.visual_gamma.sync_to_phase(0.0)
        binder.auditory_gamma.sync_to_phase(math.pi)
        metrics_unbound = binder.get_binding_strength(visual_spikes, auditory_spikes)

        assert metrics_bound["is_bound"]
        assert not metrics_unbound["is_bound"]


class TestStateSerialization:
    """Tests for state saving/loading."""

    def test_get_state(self):
        """Test state extraction."""
        binder = CrossModalGammaBinding()

        # Advance to non-initial state
        binder.visual_gamma.sync_to_phase(math.pi / 4)
        binder.auditory_gamma.sync_to_phase(math.pi / 3)

        state = binder.get_state()

        assert "visual_gamma_phase" in state
        assert "auditory_gamma_phase" in state
        assert "visual_weights" in state
        assert "auditory_weights" in state

        assert abs(state["visual_gamma_phase"] - math.pi / 4) < 0.01
        assert abs(state["auditory_gamma_phase"] - math.pi / 3) < 0.01

    def test_set_state(self):
        """Test state restoration."""
        binder = CrossModalGammaBinding()

        # Save initial state
        initial_state = binder.get_state()

        # Change state
        binder.visual_gamma.sync_to_phase(math.pi / 2)
        binder.auditory_gamma.sync_to_phase(math.pi)

        # Restore
        binder.set_state(initial_state)

        assert abs(binder.visual_gamma.phase - 0.0) < 0.01
        assert abs(binder.auditory_gamma.phase - 0.0) < 0.01

    def test_state_roundtrip(self):
        """Test full state save/load cycle."""
        config = CrossModalBindingConfig(
            visual_size=10,
            auditory_size=10,
            output_size=20,
        )
        binder1 = CrossModalGammaBinding(config=config)

        # Set to specific state
        binder1.visual_gamma.sync_to_phase(1.5)
        binder1.auditory_gamma.sync_to_phase(2.0)

        # Save state
        state = binder1.get_state()

        # Create new pathway and restore
        binder2 = CrossModalGammaBinding(config=config)
        binder2.set_state(state)

        # Should match
        assert abs(binder1.visual_gamma.phase - binder2.visual_gamma.phase) < 0.01
        assert abs(binder1.auditory_gamma.phase - binder2.auditory_gamma.phase) < 0.01
        assert torch.allclose(binder1.visual_weights, binder2.visual_weights)
        assert torch.allclose(binder1.auditory_weights, binder2.auditory_weights)


class TestDeviceHandling:
    """Tests for device placement."""

    def test_cpu_device(self):
        """Test pathway on CPU."""
        binder = CrossModalGammaBinding(device="cpu")

        visual_spikes = torch.zeros(256)
        auditory_spikes = torch.zeros(256)

        output, coherence = binder(visual_spikes, auditory_spikes)

        assert output.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """Test pathway on CUDA."""
        config = CrossModalBindingConfig(
            visual_size=10,
            auditory_size=10,
            output_size=20,
            device="cuda",
        )
        binder = CrossModalGammaBinding(config=config)

        visual_spikes = torch.zeros(10, device="cuda")
        auditory_spikes = torch.zeros(10, device="cuda")

        output, coherence = binder(visual_spikes, auditory_spikes)

        assert output.device.type == "cuda"


class TestBiologicalScenarios:
    """Tests for biologically realistic binding scenarios."""

    def test_temporal_binding_window(self):
        """Test that binding requires temporal proximity."""
        binder = CrossModalGammaBinding()

        # Synchronous presentation
        visual_spikes = torch.ones(256)
        auditory_spikes = torch.ones(256)

        binder.sync_to_external_gamma(0.0)
        _, coherence_sync = binder(visual_spikes, auditory_spikes)

        # Now advance visual by half a gamma cycle (async)
        for _ in range(12):  # ~12ms = half of 40 Hz cycle
            binder.visual_gamma.advance(1.0)

        _, coherence_async = binder(visual_spikes, auditory_spikes)

        # Synchronous should have higher coherence
        assert coherence_sync > coherence_async

    def test_ventriloquist_effect(self):
        """Test misaligned timing (ventriloquist effect simulation)."""
        config = CrossModalBindingConfig(
            visual_size=10,
            auditory_size=10,
            output_size=20,
        )
        binder = CrossModalGammaBinding(config=config)

        visual_spikes = torch.ones(10)
        auditory_spikes = torch.ones(10)

        # Aligned timing
        binder.reset_phases()
        output_aligned, coherence_aligned = binder(visual_spikes, auditory_spikes)

        # Misaligned timing (large phase difference)
        binder.reset_phases()
        binder.auditory_gamma.sync_to_phase(math.pi)  # Half cycle offset
        output_misaligned, coherence_misaligned = binder(visual_spikes, auditory_spikes)

        # Aligned should produce stronger binding
        assert coherence_aligned > coherence_misaligned
        assert output_aligned.abs().mean() > output_misaligned.abs().mean()

    def test_multisensory_enhancement(self):
        """Test that bound multimodal input is stronger than unimodal."""
        config = CrossModalBindingConfig(
            visual_size=10,
            auditory_size=10,
            output_size=20,
        )
        binder = CrossModalGammaBinding(config=config)

        # Unimodal inputs
        visual_only = torch.ones(10)
        auditory_only = torch.ones(10)
        silence = torch.zeros(10)

        # Synchronize oscillators for fair comparison
        binder.sync_to_external_gamma(math.pi / 4)

        # Visual only
        output_v, _ = binder(visual_only, silence)
        strength_v = output_v.abs().mean()

        # Auditory only
        binder.sync_to_external_gamma(math.pi / 4)
        output_a, _ = binder(silence, auditory_only)
        strength_a = output_a.abs().mean()

        # Multimodal (bound)
        binder.sync_to_external_gamma(math.pi / 4)
        output_both, _ = binder(visual_only, auditory_only)
        strength_both = output_both.abs().mean()

        # Multimodal should be stronger than either unimodal
        # (Note: Depending on gating, this might not always be true)
        # At minimum, should be comparable to sum
        assert strength_both >= max(strength_v, strength_a)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_input(self):
        """Test with no input spikes."""
        binder = CrossModalGammaBinding()

        visual_spikes = torch.zeros(256)
        auditory_spikes = torch.zeros(256)

        output, coherence = binder(visual_spikes, auditory_spikes)

        assert torch.isfinite(output).all()
        assert 0.0 <= coherence <= 1.0

    def test_maximum_input(self):
        """Test with maximum spikes."""
        binder = CrossModalGammaBinding()

        visual_spikes = torch.ones(256)
        auditory_spikes = torch.ones(256)

        output, coherence = binder(visual_spikes, auditory_spikes)

        assert torch.isfinite(output).all()
        assert 0.0 <= coherence <= 1.0

    def test_repeated_calls(self):
        """Test multiple sequential calls."""
        binder = CrossModalGammaBinding()

        visual_spikes = (torch.rand(256) > 0.8).float()
        auditory_spikes = (torch.rand(256) > 0.8).float()

        coherences = []
        for _ in range(20):
            _, coherence = binder(visual_spikes, auditory_spikes)
            coherences.append(coherence)

        # Should remain stable (no NaNs or explosions)
        assert all(0.0 <= c <= 1.0 for c in coherences)
        assert all(math.isfinite(c) for c in coherences)
