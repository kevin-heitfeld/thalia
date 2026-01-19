"""Unified Replay Engine - Memory Consolidation via Hippocampal Replay.

**DEPRECATED (Phase 4: Emergent RL Migration)**: This module is deprecated and
will be removed in a future version.

As of Phase 4 (January 19, 2026), replay now uses spontaneous CA3 attractor
dynamics instead of explicit episode replay. See SpontaneousReplayGenerator.

**Migration Path**:
- OLD: ReplayEngine.replay(episode=...)
- NEW: SpontaneousReplayGenerator (Phase 2) uses CA3 weights directly

Kept for backward compatibility during migration. May be removed in future versions.

Consolidates replay logic used by both sleep systems and hippocampal sequence
replay into a single, well-tested implementation.

**Common Elements Across Replay Implementations**:
===================================================
1. **TIME COMPRESSION**: Replay 5-20x faster than real-time encoding
2. **GAMMA PHASE MODULATION**: Gamma phase creates temporal windows for replay
3. **SEQUENTIAL PATTERN REPLAY**: Patterns activated based on gamma phase
4. **SHARP-WAVE RIPPLE MODULATION**: High-frequency coordination signal
5. **PATTERN PRIORITIZATION**: Replay important/recent experiences first

**Biological Basis**:
=====================
- **During sleep/rest**: Hippocampus replays experiences at ~5-20x speed
- **Sharp-wave ripples** (150-250 Hz): Coordinate replay timing
- **Gamma oscillations** (40-100 Hz): Create temporal windows for pattern activation
- **Phase Coding**: Sequence position encoded in gamma phase (emergent, not discrete)
- **Function**: Strengthens hippocampus → cortex connections (consolidation)
- **Priority**: Recent, surprising, or high-reward experiences replayed more

Author: Thalia Project
Date: December 23, 2025
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from thalia.constants.time import SECONDS_PER_MS, TAU


# Stub Episode type for backward compatibility during Phase 4 migration
# This allows replay_engine.py to remain functional while Episode is being phased out
@dataclass
class Episode:
    """DEPRECATED: Stub Episode dataclass for backward compatibility.

    This is a minimal stub to prevent breaking code that still uses ReplayEngine.
    See memory_component.py (deleted in Phase 4) for original implementation.

    Phase 4 removes explicit episode buffers - memory is now emergent from
    CA3 synaptic weights via Hebbian learning.
    """

    state: torch.Tensor
    sequence: Optional[List[torch.Tensor]] = None
    action: int = 0
    reward: float = 0.0
    correct: bool = False


class ReplayMode(Enum):
    """Replay execution mode.

    **Biological Context**:
    - AWAKE_FORWARD: Planning at choice points (prospection)
    - AWAKE_REVERSE: Immediate credit assignment after reward
    - SLEEP_REVERSE: Systems consolidation during sleep
    """

    # Awake replay modes (immediate, during theta troughs)
    AWAKE_FORWARD = "awake_forward"  # Planning/prospection
    AWAKE_REVERSE = "awake_reverse"  # Immediate credit assignment

    # Sleep replay modes (consolidation, during sharp-wave ripples)
    SLEEP_REVERSE = "sleep_reverse"  # Systems consolidation


@dataclass
class ReplayConfig:
    """Configuration for replay engine."""

    # Time compression
    compression_factor: float = 5.0  # How much faster than encoding (5-20x typical)

    # Gamma phase-based replay (replaces discrete slots)
    phase_window_width: float = 0.5  # Gaussian width for phase-based pattern selection (radians)
    max_patterns_per_cycle: int = 7  # Approximate capacity (emerges from gamma/theta ratio)

    # Ripple parameters (for sleep replay)
    ripple_enabled: bool = False
    ripple_frequency: float = 150.0  # Hz (150-250 typical)
    ripple_duration: float = 80.0  # ms
    ripple_gain: float = 3.0  # Amplification during ripple

    # Replay control
    max_patterns_per_replay: int = 30  # Safety limit for pattern replay
    mode: ReplayMode = ReplayMode.SLEEP_REVERSE

    # Pattern processing
    apply_phase_modulation: bool = True  # Apply gamma phase modulation to patterns
    pattern_completion: bool = True  # Run through CA3 for completion

    # Mode-specific parameters
    awake_compression: float = 10.0  # Time compression for awake replay (10x typical)
    sleep_compression: float = 5.0  # Time compression for sleep replay (5x typical)
    reverse_replay: bool = False  # Whether to replay in reverse order


@dataclass
class ReplayResult:
    """Results from a replay episode."""

    # Replay metrics
    slots_replayed: int = 0
    total_activity: float = 0.0
    gamma_cycles: int = 0
    compression_factor: float = 1.0

    # Output
    replayed_patterns: List[torch.Tensor] = field(default_factory=list)  # type: ignore[assignment]

    # Diagnostics
    mode_used: ReplayMode = ReplayMode.SLEEP_REVERSE
    sequence_length: int = 0

    def __post_init__(self):
        if self.replayed_patterns is None:
            self.replayed_patterns = []


class ReplayEngine(nn.Module):
    """Unified engine for memory replay and consolidation.

    Provides time-compressed sequence replay with gamma oscillator coordination.
    Supports multiple biologically-motivated replay modes:

    1. AWAKE_FORWARD: Planning and prospection
       - Forward replay at choice points
       - 10x time compression
       - Used for model-based decision making

    2. AWAKE_REVERSE: Immediate credit assignment
       - Reverse replay after reward
       - 10x time compression
       - Links outcomes to earlier actions

    3. SLEEP_REVERSE: Systems consolidation
       - Reverse replay during sleep
       - 5x time compression
       - Strengthens hippocampus → cortex connections

    Example:
        >>> config = ReplayConfig(compression_factor=5.0)
        >>> engine = ReplayEngine(config)
        >>>
        >>> # Replay a sequence episode
        >>> result = engine.replay(
        ...     episode=episode_with_sequence,
        ...     pattern_processor=lambda p: hippocampus.forward(p, phase=DELAY)
        ... )
        >>>
        >>> print(f"Replayed {result.slots_replayed} slots")
        >>> for pattern in result.replayed_patterns:
        >>>     consolidate_to_cortex(pattern)
    """

    def __init__(self, config: ReplayConfig):
        super().__init__()
        self.config = config

        # Ripple generator state
        if config.ripple_enabled:
            self._ripple_phase = 0.0
            self._ripple_active = False
            self._ripple_time = 0.0

    def replay(
        self,
        episode: Episode,
        pattern_processor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        gamma_phase: float = 0.0,
        mode: Optional[ReplayMode] = None,
    ) -> ReplayResult:
        """Replay an episode with time compression and mode-specific behavior.

        Uses gamma_phase for fine-grained timing within the theta cycle,
        allowing precise phase-based replay that respects gamma oscillations.

        **Replay Modes**:
        - AWAKE_FORWARD: Planning/prospection (forward replay, 10x compression)
        - AWAKE_REVERSE: Immediate credit assignment (reverse replay, 10x compression)
        - SLEEP_REVERSE: Systems consolidation (reverse replay, 5x compression)

        Args:
            episode: Episode to replay (contains state or sequence)
            pattern_processor: Function to process each pattern
                               (e.g., lambda p: hippocampus.forward(p, phase=DELAY))
                               If None, patterns are returned as-is
            gamma_phase: Current gamma phase from brain's OscillatorManager (radians [0, 2π])
            mode: Replay mode (overrides config.mode if provided)

        Returns:
            ReplayResult with replayed patterns and metrics
        """
        # Use provided mode or fall back to config mode
        active_mode = mode if mode is not None else self.config.mode

        # Set compression factor based on mode
        if active_mode in (ReplayMode.AWAKE_FORWARD, ReplayMode.AWAKE_REVERSE):
            compression = self.config.awake_compression
        elif active_mode == ReplayMode.SLEEP_REVERSE:
            compression = self.config.sleep_compression
        else:
            compression = self.config.compression_factor

        # Determine if sequence should be reversed
        reverse_sequence = active_mode in (
            ReplayMode.AWAKE_REVERSE,
            ReplayMode.SLEEP_REVERSE,
        )

        # Determine replay mode
        if episode.sequence is not None and len(episode.sequence) > 0:
            # Have sequence → use sequence replay
            return self._replay_sequence(
                episode.sequence,
                pattern_processor,
                gamma_phase,
                compression,
                reverse_sequence,
                active_mode,
            )
        else:
            # No sequence → single-state replay
            return self._replay_single(episode.state, pattern_processor, active_mode)

    def _replay_sequence(
        self,
        sequence: List[torch.Tensor],
        pattern_processor: Optional[Callable],
        gamma_phase: float,
        compression_factor: float,
        reverse: bool,
        mode: ReplayMode,
    ) -> ReplayResult:
        """Replay a sequence using continuous gamma phase modulation.

        Patterns are selected and modulated based on gamma phase using
        Gaussian windows, allowing emergent phase-based activation
        without hardcoded slot assignments.

        Args:
            sequence: List of patterns to replay
            pattern_processor: Optional function to process each pattern
            gamma_phase: Current gamma phase in radians [0, 2π]
            compression_factor: Time compression multiplier
            reverse: Whether to replay in reverse order
            mode: Replay mode for diagnostics
        """
        n_patterns = len(sequence)

        # Reverse sequence if needed (for credit assignment)
        if reverse:
            sequence = list(reversed(sequence))

        # Result tracking
        result = ReplayResult(
            compression_factor=compression_factor,
            mode_used=mode,
            sequence_length=n_patterns,
        )

        # EMERGENT PHASE-BASED PATTERN SELECTION (no hardcoded slots)
        # Each pattern gets a preferred phase distributed across gamma cycle
        # Patterns near current gamma phase are activated with Gaussian weighting

        replayed_patterns = []

        for pattern_idx, pattern in enumerate(sequence):
            # Pattern's preferred phase (evenly distributed across gamma cycle)
            preferred_phase = (2 * math.pi * pattern_idx) / n_patterns

            # Phase distance (circular)
            phase_diff = abs(gamma_phase - preferred_phase)
            phase_diff = min(phase_diff, 2 * math.pi - phase_diff)

            # Gaussian modulation based on phase proximity
            width = self.config.phase_window_width
            phase_modulation = math.exp(-(phase_diff**2) / (2 * width**2))

            # Only activate patterns within phase window (>10% modulation)
            if phase_modulation > 0.1:
                # Apply phase modulation to pattern
                if pattern.dim() == 1:
                    pattern = pattern.unsqueeze(0)

                if self.config.apply_phase_modulation:
                    modulated_pattern = pattern * phase_modulation
                else:
                    modulated_pattern = pattern

                # Process pattern (e.g., through CA3 for completion)
                if pattern_processor is not None:
                    if modulated_pattern.dim() > 1:
                        modulated_pattern = modulated_pattern.squeeze(0)
                    output = pattern_processor(modulated_pattern)
                else:
                    output = modulated_pattern

                replayed_patterns.append(output)
                result.total_activity += output.sum().item()

        result.replayed_patterns = replayed_patterns
        result.slots_replayed = len(replayed_patterns)

        return result

    def _replay_single(
        self,
        state: torch.Tensor,
        pattern_processor: Optional[Callable],
        mode: ReplayMode,
    ) -> ReplayResult:
        """Replay a single state pattern (fallback).

        Args:
            state: State tensor to replay
            pattern_processor: Optional function to process pattern
            mode: Replay mode for diagnostics
        """
        # Ensure 2D
        if state.dim() == 1:
            state = state.unsqueeze(0)

        # Process pattern
        if pattern_processor is not None:
            output = pattern_processor(state.squeeze(0))
        else:
            output = state

        return ReplayResult(
            slots_replayed=1,
            total_activity=output.sum().item(),
            gamma_cycles=0,
            compression_factor=1.0,
            replayed_patterns=[output],
            mode_used=mode,
            sequence_length=1,
        )

    def trigger_ripple(self) -> bool:
        """Trigger a sharp-wave ripple event.

        Returns:
            success: Whether ripple was triggered
        """
        if not self.config.ripple_enabled:
            return False

        self._ripple_active = True
        self._ripple_time = 0.0
        self._ripple_phase = 0.0

        return True

    def get_ripple_modulation(self, dt_ms: float) -> tuple[bool, float]:
        """Get ripple state and modulation value.

        Args:
            dt_ms: Time step in ms

        Returns:
            (ripple_active, modulation_value)
        """
        if not self.config.ripple_enabled or not self._ripple_active:
            return False, 0.0

        # Advance ripple phase
        self._ripple_time += dt_ms

        # Check if ripple duration exceeded
        if self._ripple_time >= self.config.ripple_duration:
            self._ripple_active = False
            return False, 0.0

        # Compute ripple phase (oscillation)
        freq_hz = self.config.ripple_frequency
        self._ripple_phase = TAU * freq_hz * self._ripple_time * SECONDS_PER_MS

        # Ripple value (sinusoid)
        ripple_value = math.sin(self._ripple_phase)

        # Modulation factor
        modulation = 0.5 * (1 + ripple_value) * self.config.ripple_gain

        return True, modulation

    def reset_state(self) -> None:
        """Reset replay engine state."""
        if self.config.ripple_enabled:
            self._ripple_phase = 0.0
            self._ripple_active = False
            self._ripple_time = 0.0

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get replay engine diagnostics."""
        diag = {
            "compression_factor": self.config.compression_factor,
            "mode": self.config.mode.value,
            "ripple_enabled": self.config.ripple_enabled,
            "phase_window_width": self.config.phase_window_width,
            "max_patterns_per_cycle": self.config.max_patterns_per_cycle,
        }

        if self.config.ripple_enabled:
            diag["ripple_state"] = {  # type: ignore[assignment]
                "active": self._ripple_active,
                "phase": self._ripple_phase,
                "time_ms": self._ripple_time,
            }

        return diag

    def get_state(self) -> Dict[str, Any]:
        """Get replay engine state for checkpointing."""
        return {
            "ripple_phase": self._ripple_phase if self.config.ripple_enabled else 0.0,
            "ripple_active": self._ripple_active if self.config.ripple_enabled else False,
            "ripple_time": self._ripple_time if self.config.ripple_enabled else 0.0,
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore replay engine state from checkpoint."""
        if self.config.ripple_enabled:
            self._ripple_phase = state["ripple_phase"]
            self._ripple_active = state["ripple_active"]
            self._ripple_time = state["ripple_time"]
