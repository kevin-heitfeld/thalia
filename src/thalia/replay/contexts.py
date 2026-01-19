"""Replay Contexts for Unified Replay System.

Defines the four biologically-motivated contexts for hippocampal replay,
each corresponding to different behavioral states and neuromodulatory profiles.

Author: Thalia Project
Date: January 2026
"""

from enum import Enum


class ReplayContext(Enum):
    """Biological context for hippocampal replay.

    The same hippocampal CA3→CA1 circuitry produces different replay modes
    depending on behavioral state and neuromodulatory context. These contexts
    map to distinct neuroscience findings:

    **SLEEP_CONSOLIDATION** (Sleep, offline)
    - Timing: During sleep/offline periods
    - Direction: Reverse (credit assignment)
    - Trigger: Spontaneous (CA3 driven)
    - ACh level: Low (0.1) - enables spontaneous CA3 replay
    - NE level: Low (0.1) - reduces noise
    - DA level: Moderate (0.3) - available for learning
    - Compression: 5x (slow, thorough consolidation)
    - Target: Hippocampus → cortex transfer (systems consolidation)
    - Biology: Sharp-wave ripples during slow-wave sleep (Buzsáki 2015)

    **AWAKE_IMMEDIATE** (Post-reward, online)
    - Timing: Immediately after reward/surprise
    - Direction: Reverse (credit assignment)
    - Trigger: Reward/surprise driven
    - ACh level: High (0.8) - encoding mode
    - NE level: Elevated (0.6) - arousal
    - DA level: High (0.9) - reward signal
    - Compression: 10x (fast, focused on recent)
    - Target: Rapid credit assignment to recent actions
    - Biology: Reverse replay during theta troughs (Foster & Wilson 2006)

    **FORWARD_PLANNING** (Choice points, online)
    - Timing: At decision/choice points
    - Direction: Forward (prospection)
    - Trigger: Goal-directed (PFC driven)
    - ACh level: High (0.8) - goal-directed retrieval
    - NE level: Moderate (0.5) - attention
    - DA level: Moderate (0.5) - evaluation
    - Compression: 10-20x (fast simulation)
    - Target: Action selection via outcome prediction
    - Biology: Forward replay in VTE behavior (Pfeiffer & Foster 2013)

    **BACKGROUND_PLANNING** (Idle moments, online)
    - Timing: During awake idle moments
    - Direction: Mixed (forward/reverse)
    - Trigger: Opportunistic (sampling driven)
    - ACh level: Moderate (0.6) - partial CA3 release
    - NE level: Low (0.3) - idle state
    - DA level: Moderate (0.4) - value updates
    - Compression: Variable
    - Target: Value refinement, policy improvement
    - Biology: Default mode network activity (Dyna-like, Sutton 1990)

    References:
    - Buzsáki (2015): Hippocampal sharp wave-ripple
    - Foster & Wilson (2006): Reverse replay in awake state
    - Pfeiffer & Foster (2013): Forward replay to remembered goals
    - Sutton (1990): Dyna architecture for planning
    """

    SLEEP_CONSOLIDATION = "sleep"
    AWAKE_IMMEDIATE = "awake_immediate"
    FORWARD_PLANNING = "forward_plan"
    BACKGROUND_PLANNING = "background"

    def __str__(self) -> str:
        """Return the value as string."""
        return self.value

    def is_awake(self) -> bool:
        """Check if this is an awake replay context."""
        return self in (
            ReplayContext.AWAKE_IMMEDIATE,
            ReplayContext.FORWARD_PLANNING,
            ReplayContext.BACKGROUND_PLANNING,
        )

    def is_sleep(self) -> bool:
        """Check if this is a sleep replay context."""
        return self == ReplayContext.SLEEP_CONSOLIDATION

    def is_forward(self) -> bool:
        """Check if replay is primarily forward direction."""
        return self == ReplayContext.FORWARD_PLANNING

    def is_reverse(self) -> bool:
        """Check if replay is primarily reverse direction."""
        return self in (ReplayContext.SLEEP_CONSOLIDATION, ReplayContext.AWAKE_IMMEDIATE)

    def get_default_compression(self) -> float:
        """Get default time compression factor for this context."""
        if self == ReplayContext.SLEEP_CONSOLIDATION:
            return 5.0  # Slower, thorough consolidation
        elif self == ReplayContext.AWAKE_IMMEDIATE:
            return 10.0  # Fast credit assignment
        elif self == ReplayContext.FORWARD_PLANNING:
            return 15.0  # Fast prospection
        else:  # BACKGROUND_PLANNING
            return 8.0  # Moderate speed
