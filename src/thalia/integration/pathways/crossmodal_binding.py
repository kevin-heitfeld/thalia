"""
Cross-Modal Gamma Binding - Synchronize different sensory modalities via gamma.

This pathway implements the biological mechanism where different sensory
modalities (visual, auditory, etc.) are bound together through gamma-band
oscillatory coherence. This is critical for unified perception.

Biological Background:
=====================

The brain must solve the "binding problem": How do we know that the bark
we hear and the dog we see belong to the same object?

Answer: GAMMA SYNCHRONY
- Visual and auditory cortices fire at same gamma phase (~40 Hz)
- This phase coherence signals "these signals belong together"
- Happens automatically when inputs are temporally aligned
- Breaks down when inputs are asynchronous (ventriloquist effect)

Key Evidence:
- Singer & Gray (1995): Synchrony codes feature binding
- Engel et al. (2001): Dynamic predictions for sensory binding
- Senkowski et al. (2008): Multisensory gamma-band coherence
- Lakatos et al. (2007): Cross-modal phase reset

Mechanisms:
1. PHASE RESET: Strong input resets gamma phase in both modalities
2. MUTUAL COUPLING: Each modality nudges the other's phase
3. TEMPORAL WINDOW: Binding only works if inputs arrive within ~50ms
4. GAMMA GATING: Output is enhanced when both modalities align

Our Implementation:
==================

This pathway:
1. Takes spike trains from two modalities (e.g., visual and auditory)
2. Extracts current gamma phase from each
3. Computes phase coherence (alignment)
4. Gates output by coherence (synchronized = strong, desync = weak)
5. Applies phase nudging to keep modalities synchronized

Usage:
    ```python
    # Create binding pathway
    binder = CrossModalGammaBinding(
        visual_size=256,
        auditory_size=256,
        output_size=512,
        gamma_freq_hz=40.0,
    )

    # Process multimodal input
    visual_spikes = torch.rand(256) > 0.8  # Binary spikes
    auditory_spikes = torch.rand(256) > 0.8

    bound_output, coherence = binder(visual_spikes, auditory_spikes)

    # High coherence (> 0.7) = good binding
    # Low coherence (< 0.3) = weak binding
    ```

Author: Thalia Project
Date: December 8, 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from thalia.core.pathway_protocol import BaseNeuralPathway
from thalia.core.weight_init import WeightInitializer
from thalia.core.oscillator import SinusoidalOscillator


@dataclass
class CrossModalBindingConfig:
    """Configuration for cross-modal gamma binding.

    Attributes:
        visual_size: Number of visual input neurons
        auditory_size: Number of auditory input neurons
        output_size: Number of output neurons (bound representation)
        gamma_freq_hz: Gamma frequency for binding (typically 40 Hz)
        coherence_window: Temporal window for coherence (radians, ~π/4)
        phase_coupling_strength: How strongly modalities influence each other
        gate_threshold: Minimum coherence for output gating
        dt_ms: Simulation timestep
        device: Computation device
    """

    visual_size: int = 256
    auditory_size: int = 256
    output_size: int = 512
    gamma_freq_hz: float = 40.0
    coherence_window: float = math.pi / 4  # ~45° phase tolerance
    phase_coupling_strength: float = 0.1  # Mutual phase nudging
    gate_threshold: float = 0.3  # Minimum coherence for binding
    dt_ms: float = 1.0
    device: str = "cpu"


class CrossModalGammaBinding(BaseNeuralPathway):
    """
    Cross-modal binding through gamma-band synchronization.

    Implements the biological mechanism where different sensory modalities
    are bound together via synchronized gamma oscillations.

    Key Features:
    - Separate gamma oscillators for each modality
    - Phase coherence measurement
    - Mutual phase coupling (bidirectional influence)
    - Coherence-gated output
    - Temporal binding window

    Biological Analog:
        - Visual and auditory cortices synchronizing at 40 Hz
        - Phase-locked responses indicate same object
        - Desynchronized responses indicate separate objects
    """

    def __init__(
        self,
        config: Optional[CrossModalBindingConfig] = None,
        visual_size: Optional[int] = None,
        auditory_size: Optional[int] = None,
        output_size: Optional[int] = None,
        gamma_freq_hz: Optional[float] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize cross-modal gamma binding pathway.

        Args:
            config: Complete configuration object (preferred)
            visual_size: Override visual input size
            auditory_size: Override auditory input size
            output_size: Override output size
            gamma_freq_hz: Override gamma frequency
            device: Override device
        """
        # Build config with overrides
        if config is None:
            config = CrossModalBindingConfig()

        if visual_size is not None:
            config.visual_size = visual_size
        if auditory_size is not None:
            config.auditory_size = auditory_size
        if output_size is not None:
            config.output_size = output_size
        if gamma_freq_hz is not None:
            config.gamma_freq_hz = gamma_freq_hz
        if device is not None:
            config.device = device

        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        # Separate gamma oscillators for each modality
        self.visual_gamma = SinusoidalOscillator(
            frequency_hz=config.gamma_freq_hz,
            dt_ms=config.dt_ms,
        )
        self.auditory_gamma = SinusoidalOscillator(
            frequency_hz=config.gamma_freq_hz,
            dt_ms=config.dt_ms,
        )

        # Initialize weights (modality-specific projections)
        self._initialize_weights()

        # Move to device
        self.to(self.device)

    def _initialize_weights(self) -> None:
        """Initialize projection weights from each modality to output."""
        cfg = self.config

        # Visual → Output projection
        self.visual_weights = WeightInitializer.gaussian(
            cfg.output_size,
            cfg.visual_size,
            mean=0.5,
            std=0.1,
            device=cfg.device,
        )

        # Auditory → Output projection
        self.auditory_weights = WeightInitializer.gaussian(
            cfg.output_size,
            cfg.auditory_size,
            mean=0.5,
            std=0.1,
            device=cfg.device,
        )

        # Register as parameters
        self.visual_weights = nn.Parameter(self.visual_weights)
        self.auditory_weights = nn.Parameter(self.auditory_weights)

    def forward(
        self,
        visual_spikes: torch.Tensor,
        auditory_spikes: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Bind visual and auditory inputs via gamma synchrony.

        Args:
            visual_spikes: Visual spike train [visual_size] (binary)
            auditory_spikes: Auditory spike train [auditory_size] (binary)

        Returns:
            bound_output: Bound multimodal representation [output_size]
            coherence: Phase coherence score [0, 1] (1 = perfect sync)
        """
        # Ensure correct device
        visual_spikes = visual_spikes.to(self.device)
        auditory_spikes = auditory_spikes.to(self.device)

        # Advance oscillators
        self.visual_gamma.advance(self.config.dt_ms)
        self.auditory_gamma.advance(self.config.dt_ms)

        # Phase-based gating for each modality
        visual_gate = self._compute_gamma_gate(self.visual_gamma.phase)
        auditory_gate = self._compute_gamma_gate(self.auditory_gamma.phase)

        # Apply gamma gating to spikes
        visual_gated = visual_spikes * visual_gate
        auditory_gated = auditory_spikes * auditory_gate

        # Project to output space
        visual_output = F.linear(visual_gated.unsqueeze(0), self.visual_weights).squeeze(0)
        auditory_output = F.linear(auditory_gated.unsqueeze(0), self.auditory_weights).squeeze(0)

        # Measure phase coherence
        coherence = self._compute_phase_coherence(
            self.visual_gamma.phase,
            self.auditory_gamma.phase,
        )

        # Apply mutual phase coupling
        self._apply_phase_coupling(visual_spikes, auditory_spikes)

        # Gate combined output by coherence
        coherence_gate = self._coherence_to_gate(coherence)
        bound_output = (visual_output + auditory_output) * coherence_gate

        return bound_output, coherence

    def _compute_gamma_gate(self, gamma_phase: float, width: float = 0.3) -> float:
        """
        Compute gamma-phase-dependent gating.

        Creates a temporal window: inputs are only strongly processed
        during certain phases of gamma (peak excitability).

        Args:
            gamma_phase: Current gamma phase [0, 2π)
            width: Width of the Gaussian gate

        Returns:
            gate: Gating strength [0, 1]
        """
        # Gaussian centered at phase = π/2 (peak of sine wave)
        optimal_phase = math.pi / 2
        phase_diff = abs(gamma_phase - optimal_phase)

        # Wrap around for circular distance
        phase_diff = min(phase_diff, 2 * math.pi - phase_diff)

        # Gaussian gate
        gate = math.exp(-(phase_diff**2) / (2 * width**2))

        return gate

    def _compute_phase_coherence(
        self,
        visual_phase: float,
        auditory_phase: float,
    ) -> float:
        """
        Measure phase coherence between two oscillators.

        High coherence (near 1.0) = phases aligned = bound together
        Low coherence (near 0.0) = phases misaligned = separate objects

        Args:
            visual_phase: Visual gamma phase [0, 2π)
            auditory_phase: Auditory gamma phase [0, 2π)

        Returns:
            coherence: Coherence score [0, 1]
        """
        # Circular distance between phases
        phase_diff = abs(visual_phase - auditory_phase)
        phase_diff = min(phase_diff, 2 * math.pi - phase_diff)

        # Convert to coherence: 0 diff = 1.0 coherence, π diff = 0.0 coherence
        coherence = math.cos(phase_diff / 2.0) ** 2  # Squared cosine for sharper tuning

        return coherence

    def _apply_phase_coupling(
        self,
        visual_spikes: torch.Tensor,
        auditory_spikes: torch.Tensor,
    ) -> None:
        """
        Apply mutual phase coupling between modalities.

        When one modality has strong input, it nudges the other's phase
        toward synchrony. This is how the brain achieves binding.

        Args:
            visual_spikes: Visual activity (used to weight coupling)
            auditory_spikes: Auditory activity (used to weight coupling)
        """
        cfg = self.config

        # Measure activity strength
        visual_activity = float(visual_spikes.mean())
        auditory_activity = float(auditory_spikes.mean())

        # Only couple if both modalities are active
        if visual_activity > 0.01 and auditory_activity > 0.01:
            # Compute phase difference
            phase_diff = self.auditory_gamma.phase - self.visual_gamma.phase

            # Normalize to [-π, π]
            if phase_diff > math.pi:
                phase_diff -= 2 * math.pi
            elif phase_diff < -math.pi:
                phase_diff += 2 * math.pi

            # Nudge each phase toward the other
            coupling_amount = cfg.phase_coupling_strength * phase_diff

            # Visual nudged by auditory
            visual_nudge = coupling_amount * auditory_activity
            new_visual_phase = self.visual_gamma.phase + visual_nudge
            self.visual_gamma.sync_to_phase(new_visual_phase)

            # Auditory nudged by visual (opposite direction)
            auditory_nudge = -coupling_amount * visual_activity
            new_auditory_phase = self.auditory_gamma.phase + auditory_nudge
            self.auditory_gamma.sync_to_phase(new_auditory_phase)

    def _coherence_to_gate(self, coherence: float) -> float:
        """
        Convert phase coherence to output gate.

        Args:
            coherence: Phase coherence [0, 1]

        Returns:
            gate: Output gating strength [0, 1]
        """
        cfg = self.config

        # Soft threshold: gradually increase above threshold
        if coherence < cfg.gate_threshold:
            return 0.0
        else:
            # Sigmoid above threshold for smooth gating
            x = (coherence - cfg.gate_threshold) / (1.0 - cfg.gate_threshold)
            gate = 1.0 / (1.0 + math.exp(-10 * (x - 0.5)))  # Steep sigmoid
            return gate

    def reset_phases(self) -> None:
        """Reset both gamma oscillators to initial phase."""
        self.visual_gamma.reset_state()
        self.auditory_gamma.reset_state()

    def reset_state(self) -> None:
        """
        Reset pathway temporal state (required by BaseNeuralPathway).

        Clears oscillator phases and returns system to initial state.
        """
        self.reset_phases()

    def sync_to_external_gamma(self, external_phase: float) -> None:
        """
        Synchronize both modalities to an external gamma signal.

        Use this to entrain the binding system to a common oscillator.

        Args:
            external_phase: External gamma phase [0, 2π)
        """
        self.visual_gamma.sync_to_phase(external_phase)
        self.auditory_gamma.sync_to_phase(external_phase)

    def get_binding_strength(
        self,
        visual_spikes: torch.Tensor,
        auditory_spikes: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Evaluate binding quality without changing state.

        Args:
            visual_spikes: Visual input
            auditory_spikes: Auditory input

        Returns:
            metrics: Dictionary with coherence and activity measures
        """
        coherence = self._compute_phase_coherence(
            self.visual_gamma.phase,
            self.auditory_gamma.phase,
        )

        visual_activity = float(visual_spikes.mean())
        auditory_activity = float(auditory_spikes.mean())

        return {
            "coherence": coherence,
            "visual_activity": visual_activity,
            "auditory_activity": auditory_activity,
            "is_bound": coherence > self.config.gate_threshold,
            "visual_phase": self.visual_gamma.phase,
            "auditory_phase": self.auditory_gamma.phase,
        }

    def get_diagnostics(self) -> Dict[str, Any]:
        """
        Get pathway diagnostics (required by BaseNeuralPathway).

        Returns:
            Dictionary containing:
            - Phase information for both modalities
            - Weight statistics
            - Oscillator frequencies
        """
        # Weight statistics
        visual_weights_flat = self.visual_weights.detach().flatten()
        auditory_weights_flat = self.auditory_weights.detach().flatten()

        return {
            "visual_gamma_phase": self.visual_gamma.phase,
            "auditory_gamma_phase": self.auditory_gamma.phase,
            "visual_gamma_freq": self.visual_gamma.frequency_hz,
            "auditory_gamma_freq": self.auditory_gamma.frequency_hz,
            "weight_stats": {
                "visual_mean": float(visual_weights_flat.mean()),
                "visual_std": float(visual_weights_flat.std()),
                "visual_min": float(visual_weights_flat.min()),
                "visual_max": float(visual_weights_flat.max()),
                "auditory_mean": float(auditory_weights_flat.mean()),
                "auditory_std": float(auditory_weights_flat.std()),
                "auditory_min": float(auditory_weights_flat.min()),
                "auditory_max": float(auditory_weights_flat.max()),
            },
            "config": {
                "gamma_freq_hz": self.config.gamma_freq_hz,
                "coherence_window": self.config.coherence_window,
                "gate_threshold": self.config.gate_threshold,
                "phase_coupling_strength": self.config.phase_coupling_strength,
            },
        }

    def get_state(self) -> Dict[str, Any]:
        """Get pathway state for checkpointing."""
        return {
            "visual_gamma_phase": self.visual_gamma.phase,
            "auditory_gamma_phase": self.auditory_gamma.phase,
            "visual_weights": self.visual_weights.detach().cpu(),
            "auditory_weights": self.auditory_weights.detach().cpu(),
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore pathway state from checkpoint."""
        self.visual_gamma.sync_to_phase(state["visual_gamma_phase"])
        self.auditory_gamma.sync_to_phase(state["auditory_gamma_phase"])
        self.visual_weights.data = state["visual_weights"].to(self.device)
        self.auditory_weights.data = state["auditory_weights"].to(self.device)
