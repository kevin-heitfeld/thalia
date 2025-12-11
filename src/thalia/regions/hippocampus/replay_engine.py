"""
Unified Replay Engine for Memory Consolidation

Consolidates replay logic used by both SleepSystemMixin and TrisynapticHippocampus.

Common elements across replay implementations:
1. Time compression (replay 5-20x faster than encoding)
2. Gamma oscillator-driven sequence reactivation
3. Slot-by-slot pattern replay
4. Sharp-wave ripple modulation
5. Pattern prioritization and selection

This engine provides a single, configurable implementation that can be used
by both sleep-based consolidation and hippocampal sequence replay.

Biological basis:
- During sleep/rest, hippocampus replays experiences at ~5-20x speed
- Sharp-wave ripples (150-250 Hz) coordinate replay
- Gamma oscillations (40-100 Hz) organize sequence slots
- Replay strengthens hippocampus → cortex connections

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable, TYPE_CHECKING
from enum import Enum

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from thalia.regions.hippocampus.config import Episode


class ReplayMode(Enum):
    """Replay execution mode."""
    SEQUENCE = "sequence"  # Gamma-driven sequence replay
    SINGLE = "single"      # Single-state replay (fallback)
    RIPPLE = "ripple"      # Sharp-wave ripple replay


@dataclass
class ReplayConfig:
    """Configuration for replay engine."""
    
    # Time compression
    compression_factor: float = 5.0  # How much faster than encoding (5-20x typical)
    dt_ms: float = 1.0               # Base time step in ms
    
    # Gamma slot configuration (replaces oscillator config)
    n_slots: int = 7                 # Number of gamma slots for sequence replay
    slot_duration_ms: float = 18.0   # Duration per slot in ms (~18ms for 7 slots in 125ms theta)
    
    # Ripple parameters (for sleep replay)
    ripple_enabled: bool = False
    ripple_frequency: float = 150.0  # Hz (150-250 typical)
    ripple_duration: float = 80.0    # ms
    ripple_gain: float = 3.0         # Amplification during ripple
    
    # Replay control
    max_steps_per_slot: int = 30     # Safety limit for slot replay
    mode: ReplayMode = ReplayMode.SEQUENCE
    
    # Pattern processing
    apply_gating: bool = True        # Apply gamma gating to patterns
    pattern_completion: bool = True  # Run through CA3 for completion


@dataclass
class ReplayResult:
    """Results from a replay episode."""
    
    # Replay metrics
    slots_replayed: int = 0
    total_activity: float = 0.0
    gamma_cycles: int = 0
    compression_factor: float = 1.0
    
    # Output
    replayed_patterns: List[torch.Tensor] = None
    
    # Diagnostics
    mode_used: ReplayMode = ReplayMode.SINGLE
    sequence_length: int = 0
    
    def __post_init__(self):
        if self.replayed_patterns is None:
            self.replayed_patterns = []


class ReplayEngine(nn.Module):
    """Unified engine for memory replay and consolidation.
    
    Provides time-compressed sequence replay with gamma oscillator coordination.
    Can operate in three modes:
    
    1. SEQUENCE: Full gamma-driven sequence replay
       - Uses gamma oscillator to drive slot-by-slot reactivation
       - Time-compressed (5-20x faster than encoding)
       - Best for replaying learned sequences
       
    2. SINGLE: Single-state replay
       - Replays one pattern without sequence structure
       - Fallback when no sequence available
       - Faster, simpler
       
    3. RIPPLE: Sharp-wave ripple modulated replay
       - Modulates replay by ripple phase
       - Used in sleep/offline consolidation
       - Coordinates hippocampus → cortex transfer
    
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
        gating_fn: Optional[Callable[[int], float]] = None,
        gamma_phase: float = 0.0,
    ) -> ReplayResult:
        """Replay an episode with time compression.
        
        Uses gamma_phase for fine-grained timing within the theta cycle,
        allowing precise phase-based replay that respects gamma oscillations.
        
        Args:
            episode: Episode to replay (contains state or sequence)
            pattern_processor: Function to process each pattern
                               (e.g., lambda p: hippocampus.forward(p, phase=DELAY))
                               If None, patterns are returned as-is
            gating_fn: Function to compute gating for each slot
                       (e.g., lambda slot: get_gamma_gating(slot))
                       If None, no gating applied
            gamma_phase: Current gamma phase from brain's OscillatorManager (radians [0, 2π])
        
        Returns:
            ReplayResult with replayed patterns and metrics
        """
        # Determine replay mode
        if episode.sequence is not None and len(episode.sequence) > 0:
            # Have sequence → use sequence replay
            return self._replay_sequence(
                episode.sequence,
                pattern_processor,
                gating_fn,
                gamma_phase,
            )
        else:
            # No sequence → single-state replay
            return self._replay_single(episode.state, pattern_processor)
    
    def _replay_sequence(
        self,
        sequence: List[torch.Tensor],
        pattern_processor: Optional[Callable],
        gating_fn: Optional[Callable],
        gamma_phase: float,
    ) -> ReplayResult:
        """Replay a sequence using gamma phase timing from brain.
        
        The gamma phase (0 to 2π) is converted to a slot index, allowing
        for sub-slot precision and smooth phase-based replay.
        
        Args:
            sequence: List of patterns to replay (one per slot)
            pattern_processor: Optional function to process each pattern
            gating_fn: Optional function to compute slot-specific gating
            gamma_phase: Current gamma phase in radians [0, 2π]
        """
        import math
        
        n_slots = len(sequence)
        
        # Result tracking
        result = ReplayResult(
            compression_factor=self.config.compression_factor,
            mode_used=ReplayMode.SEQUENCE,
            sequence_length=n_slots,
        )
        
        # Convert gamma phase to slot index using finer-grained timing
        # gamma_phase ranges from 0 to 2π over one gamma cycle
        # Map this to slot indices: phase=0 → slot 0, phase=2π → slot n_slots
        normalized_phase = (gamma_phase % (2 * math.pi)) / (2 * math.pi)  # [0, 1]
        current_slot = int(normalized_phase * n_slots) % n_slots
        
        # Replay the pattern for the current slot
        if current_slot < n_slots:
            # Get pattern for this slot
            pattern = sequence[current_slot]
            
            # Apply gating if enabled
            if self.config.apply_gating and gating_fn is not None:
                gating = gating_fn(current_slot)
                if pattern.dim() == 1:
                    pattern = pattern.unsqueeze(0)
                pattern = pattern * gating
            
            # Process pattern (e.g., through CA3 for completion)
            if pattern_processor is not None:
                if pattern.dim() > 1:
                    pattern = pattern.squeeze(0)
                output = pattern_processor(pattern)
            else:
                output = pattern
            
            # Record
            result.replayed_patterns.append(output)
            result.total_activity += output.sum().item()
            result.slots_replayed = 1  # Single slot replayed per call
        
        return result
    
    def _replay_single(
        self,
        state: torch.Tensor,
        pattern_processor: Optional[Callable],
    ) -> ReplayResult:
        """Replay a single state pattern (fallback)."""
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
            mode_used=ReplayMode.SINGLE,
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
    
    def get_ripple_modulation(self, dt: float) -> tuple[bool, float]:
        """Get ripple state and modulation value.
        
        Args:
            dt: Time step in ms
        
        Returns:
            (ripple_active, modulation_value)
        """
        if not self.config.ripple_enabled or not self._ripple_active:
            return False, 0.0
        
        # Advance ripple phase
        self._ripple_time += dt
        
        # Check if ripple duration exceeded
        if self._ripple_time >= self.config.ripple_duration:
            self._ripple_active = False
            return False, 0.0
        
        # Compute ripple phase (oscillation)
        import math
        freq_hz = self.config.ripple_frequency
        self._ripple_phase = 2.0 * math.pi * freq_hz * self._ripple_time / 1000.0
        
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
            "n_slots": self.config.n_slots,
        }
        
        if self.config.ripple_enabled:
            diag["ripple_state"] = {
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
