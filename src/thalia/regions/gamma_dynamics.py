"""
Theta-Gamma Coupling for Hippocampal Sequence Encoding

Theta-gamma coupling is when fast gamma oscillations (30-100 Hz) are nested
within slower theta oscillations (4-10 Hz). This is critical for:

1. WORKING MEMORY CAPACITY:
   - ~7±2 gamma cycles fit in one theta cycle
   - Each gamma cycle = one memory "slot"
   - This explains the 7±2 working memory capacity limit!

2. SEQUENCE ENCODING:
   - Items in a sequence get encoded at successive gamma phases
   - Temporal order is represented by gamma phase within theta

3. PHASE CODING:
   - Same neuron firing at different gamma phases = different items
   - Gamma phase relative to theta encodes recency/order

References:
- Lisman & Jensen (2013): The theta-gamma neural code
- Colgin (2013): Mechanisms and functions of theta rhythms
- Buzsaki (2006): Rhythms of the Brain

Implementation:
- GammaOscillator: Generates gamma rhythm nested in theta
- GammaSlot: Tracks which "slot" in the gamma sequence we're in
- Phase-based gating: Modulates neuron activity by gamma phase

Example:
    gamma = GammaOscillator(gamma_freq_hz=40, theta_freq_hz=8)

    for t in range(1000):
        gamma.advance(dt=1.0)

        # Get current slot (0-6 typically)
        slot = gamma.current_slot

        # Get gating signal for neurons
        gate = gamma.get_phase_gate(preferred_slot=3)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import math

import torch
import torch.nn as nn


@dataclass
class ThetaGammaConfig:
    """Configuration for theta-gamma coupling.

    Biological parameters:
    - Theta: 4-10 Hz (typically ~8 Hz)
    - Gamma: 30-100 Hz (typically ~40 Hz)
    - ~5-8 gamma cycles per theta cycle
    """
    # Theta parameters
    theta_freq_hz: float = 8.0           # Theta frequency

    # Gamma parameters
    gamma_freq_hz: float = 40.0          # Gamma frequency
    gamma_amplitude: float = 1.0         # Base gamma amplitude

    # Coupling parameters
    # Gamma amplitude is modulated by theta phase
    # Strongest gamma at theta trough (encoding), weakest at peak
    coupling_strength: float = 0.8       # How much theta modulates gamma (0-1)
    gamma_at_peak: float = 0.2           # Minimum gamma at theta peak

    # Phase-based gating
    n_slots: int = 7                     # Number of gamma slots per theta cycle
    slot_width: float = 0.6              # Width of gating window (0-1)

    # Timing
    dt_ms: float = 1.0                   # Default timestep


class GammaOscillator(nn.Module):
    """Generates gamma oscillations nested within theta rhythm.

    The gamma amplitude is modulated by the theta phase:
    - At theta trough (encoding phase): Gamma is STRONG
    - At theta peak (retrieval phase): Gamma is WEAK

    This creates "gamma bursts" that are phase-locked to theta.
    Each gamma cycle within a theta cycle represents a memory "slot".
    """

    def __init__(self, config: Optional[ThetaGammaConfig] = None):
        super().__init__()
        self.config = config or ThetaGammaConfig()

        cfg = self.config

        # Compute periods
        self.theta_period_ms = 1000.0 / cfg.theta_freq_hz  # ~125ms for 8Hz
        self.gamma_period_ms = 1000.0 / cfg.gamma_freq_hz  # ~25ms for 40Hz

        # Gamma cycles per theta cycle
        self.gamma_per_theta = self.theta_period_ms / self.gamma_period_ms  # ~5-8

        # Phase state (in radians)
        self._theta_phase = 0.0
        self._gamma_phase = 0.0

        # Time tracking
        self.time_ms = 0.0

        # Precompute phase increments per ms
        self._theta_phase_per_ms = 2.0 * math.pi * cfg.theta_freq_hz / 1000.0
        self._gamma_phase_per_ms = 2.0 * math.pi * cfg.gamma_freq_hz / 1000.0

    def advance(self, dt_ms: Optional[float] = None) -> None:
        """Advance the oscillator by one timestep.

        Args:
            dt_ms: Timestep in ms (uses config default if None)
        """
        dt = dt_ms or self.config.dt_ms

        # Advance phases
        self._theta_phase += self._theta_phase_per_ms * dt
        self._gamma_phase += self._gamma_phase_per_ms * dt

        # Wrap to [0, 2π)
        self._theta_phase = self._theta_phase % (2.0 * math.pi)
        self._gamma_phase = self._gamma_phase % (2.0 * math.pi)

        self.time_ms += dt

    def sync_to_theta_phase(self, theta_phase: float) -> None:
        """Synchronize to an external theta phase.

        Use this to sync with an existing ThetaGenerator.

        Args:
            theta_phase: External theta phase in radians
        """
        self._theta_phase = theta_phase % (2.0 * math.pi)
        # Reset gamma phase to align with theta
        self._gamma_phase = 0.0
    
    def set_to_slot(self, slot: int) -> None:
        """Set theta phase so current_slot equals the given slot.
        
        Useful for encoding items at specific sequence positions.
        
        Args:
            slot: Target slot index [0, n_slots-1]
        """
        cfg = self.config
        target_slot = slot % cfg.n_slots
        # Theta progress for this slot (0 to 1)
        target_progress = (target_slot + 0.5) / cfg.n_slots  # Center of slot
        self._theta_phase = target_progress * 2.0 * math.pi
        self._gamma_phase = 0.0

    @property
    def theta_phase(self) -> float:
        """Current theta phase in radians [0, 2π)."""
        return self._theta_phase

    @property
    def gamma_phase(self) -> float:
        """Current gamma phase in radians [0, 2π)."""
        return self._gamma_phase

    @property
    def gamma_amplitude(self) -> float:
        """Current gamma amplitude, modulated by theta phase.

        Gamma is strongest at theta trough (phase = 0) where encoding happens.
        Gamma is weakest at theta peak (phase = π) where retrieval happens.
        """
        cfg = self.config

        # Theta modulation: 1 at trough, 0 at peak
        theta_mod = 0.5 * (1.0 + math.cos(self._theta_phase))  # [0, 1]

        # Apply coupling strength
        amplitude = (
            cfg.gamma_at_peak +
            (cfg.gamma_amplitude - cfg.gamma_at_peak) * theta_mod * cfg.coupling_strength
        )

        return amplitude

    @property
    def gamma_signal(self) -> float:
        """Current gamma oscillation value [-1, 1] × amplitude.

        This is the raw oscillation signal that can drive neural activity.
        """
        return math.sin(self._gamma_phase) * self.gamma_amplitude

    @property
    def current_slot(self) -> int:
        """Current gamma slot index [0, n_slots-1].

        Each theta cycle is divided into n_slots gamma slots.
        This maps the current gamma phase within theta to a slot number.

        Slot 0 is at theta trough (encoding), slot n-1 is near theta peak.
        """
        cfg = self.config

        # Calculate how far through the theta cycle we are [0, 1)
        theta_progress = self._theta_phase / (2.0 * math.pi)

        # Map to slot number
        slot = int(theta_progress * cfg.n_slots) % cfg.n_slots

        return slot

    @property
    def slot_phase(self) -> float:
        """Phase within current slot [0, 1).

        0 = start of slot, 1 = end of slot.
        """
        cfg = self.config
        theta_progress = self._theta_phase / (2.0 * math.pi)
        slot_progress = (theta_progress * cfg.n_slots) % 1.0
        return slot_progress

    def get_slot_gate(
        self,
        preferred_slot: int,
        width: Optional[float] = None,
    ) -> float:
        """Get gating signal for a specific slot.

        Returns high value when current slot matches preferred slot,
        low value otherwise. Use this to gate which neurons fire
        at which part of the theta-gamma cycle.

        Args:
            preferred_slot: The slot when this gate should be open (0 to n_slots-1)
            width: Gate width (0-1), higher = wider window. Uses config if None.

        Returns:
            Gate value [0, 1]. High when in preferred slot, low otherwise.
        """
        cfg = self.config
        width = width or cfg.slot_width

        # Current slot
        current = self.current_slot
        n = cfg.n_slots

        # Distance to preferred slot (circular)
        dist = min(
            abs(current - preferred_slot),
            n - abs(current - preferred_slot)
        )

        # Gaussian-like gating based on distance
        # width controls how sharp the transition is
        sigma = n * width / 2.0  # Scale width by number of slots
        gate = math.exp(-(dist ** 2) / (2.0 * sigma ** 2 + 1e-6))

        return gate

    def get_phase_gate_tensor(
        self,
        n_neurons: int,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Get gating tensor for neurons with different preferred slots.

        Each neuron is assigned a preferred slot. Returns a gating tensor
        where each neuron's gate value depends on how close the current
        slot is to its preferred slot.

        This enables phase coding: same neuron fires for different items
        depending on when it fires relative to theta/gamma.

        Args:
            n_neurons: Number of neurons
            device: Torch device

        Returns:
            Tensor of shape [n_neurons] with gate values [0, 1]
        """
        cfg = self.config

        # Assign each neuron a preferred slot (evenly distributed)
        preferred_slots = torch.arange(n_neurons, device=device) % cfg.n_slots

        # Compute gate for each
        gates = torch.tensor([
            self.get_slot_gate(int(slot))
            for slot in preferred_slots
        ], device=device)

        return gates

    def encode_sequence_position(
        self,
        position: int,
        max_positions: int,
    ) -> int:
        """Map a sequence position to a preferred gamma slot.

        Items early in sequence get early slots (near theta trough),
        items later in sequence get later slots (toward theta peak).

        Args:
            position: Position in sequence (0, 1, 2, ...)
            max_positions: Maximum sequence length

        Returns:
            Preferred slot for this position
        """
        cfg = self.config

        # Map position to slot (wrapping if sequence is longer than slots)
        slot = position % cfg.n_slots

        return slot

    def get_state(self) -> dict:
        """Get oscillator state for serialization."""
        return {
            "theta_phase": self._theta_phase,
            "gamma_phase": self._gamma_phase,
            "time_ms": self.time_ms,
        }

    def set_state(self, state: dict) -> None:
        """Restore oscillator state from dict."""
        self._theta_phase = state["theta_phase"]
        self._gamma_phase = state["gamma_phase"]
        self.time_ms = state["time_ms"]

    def reset(self) -> None:
        """Reset oscillator to initial state."""
        self._theta_phase = 0.0
        self._gamma_phase = 0.0
        self.time_ms = 0.0


class SequenceEncoder(nn.Module):
    """Encodes sequences using theta-gamma phase coding.

    Items presented in sequence are encoded at different gamma phases.
    This enables:
    1. Temporal order encoding (position → phase)
    2. Chunking (items within a theta cycle are grouped)
    3. Capacity limits (~7 items per theta cycle)

    Example:
        encoder = SequenceEncoder(n_neurons=100, gamma_oscillator=gamma)

        # Encode items in sequence
        for i, item in enumerate(sequence):
            # Item gets encoded at slot i
            encoded = encoder.encode_item(item, position=i)
    """

    def __init__(
        self,
        n_neurons: int,
        gamma_oscillator: GammaOscillator,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.n_neurons = n_neurons
        self.gamma = gamma_oscillator
        self.device = device

        cfg = self.gamma.config

        # Assign neurons to slots
        # Each slot has n_neurons / n_slots neurons
        self.neurons_per_slot = n_neurons // cfg.n_slots
        self.slot_assignment = torch.arange(n_neurons) // self.neurons_per_slot
        self.slot_assignment = self.slot_assignment.clamp(max=cfg.n_slots - 1)

    def encode_item(
        self,
        item_pattern: torch.Tensor,
        position: int,
    ) -> torch.Tensor:
        """Encode an item at a specific sequence position.

        The item is gated by the gamma phase corresponding to its position.

        Args:
            item_pattern: Input pattern [batch, n_features] or [n_features]
            position: Sequence position (0, 1, 2, ...)

        Returns:
            Encoded pattern gated by position's gamma slot
        """
        cfg = self.gamma.config

        if item_pattern.dim() == 1:
            item_pattern = item_pattern.unsqueeze(0)

        # Get the slot for this position
        slot = position % cfg.n_slots

        # Get gating signal for this slot
        gate = self.gamma.get_slot_gate(slot)

        # Gate the pattern
        # Only neurons assigned to this slot get the full pattern
        # Others are suppressed
        slot_mask = (self.slot_assignment == slot).float().to(item_pattern.device)

        # Apply both gamma gating and slot assignment
        encoded = item_pattern * gate * slot_mask.unsqueeze(0)

        return encoded.squeeze(0) if encoded.shape[0] == 1 else encoded

    def decode_position(
        self,
        activity: torch.Tensor,
    ) -> int:
        """Decode sequence position from neural activity.

        Looks at which slot's neurons are most active to infer position.

        Args:
            activity: Neural activity pattern [n_neurons]

        Returns:
            Estimated sequence position (slot number)
        """
        cfg = self.gamma.config

        # Compute activity per slot
        slot_activities = torch.zeros(cfg.n_slots)
        for slot in range(cfg.n_slots):
            mask = (self.slot_assignment == slot)
            slot_activities[slot] = activity[mask].sum()

        # Return slot with highest activity
        return int(slot_activities.argmax().item())


# Convenience function
def create_theta_gamma_system(
    theta_freq_hz: float = 8.0,
    gamma_freq_hz: float = 40.0,
    n_slots: int = 7,
) -> GammaOscillator:
    """Create a configured theta-gamma oscillator system.

    Args:
        theta_freq_hz: Theta frequency (4-10 Hz)
        gamma_freq_hz: Gamma frequency (30-100 Hz)
        n_slots: Number of working memory slots

    Returns:
        Configured GammaOscillator
    """
    config = ThetaGammaConfig(
        theta_freq_hz=theta_freq_hz,
        gamma_freq_hz=gamma_freq_hz,
        n_slots=n_slots,
    )
    return GammaOscillator(config)
