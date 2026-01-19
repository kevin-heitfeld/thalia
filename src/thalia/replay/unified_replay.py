"""Unified Replay Coordinator - DEPRECATED.

**DEPRECATION NOTICE (January 2026 - Phase 2 Emergent RL)**:
This module is deprecated and will be removed in a future release.

Explicit replay coordination has been replaced by spontaneous replay
via biologically-accurate mechanisms (Phase 2 - Emergent RL):
- Spontaneous replay: CA3 attractor dynamics + synaptic tagging
- Sharp-wave ripples: Acetylcholine-gated (no explicit triggering)
- Priority: Frey-Morris synaptic tags (no explicit Episode.priority)
- Consolidation: Just set ACh low and run forward() (no coordinator)

**Migration Guide**:
Instead of:
```python
coordinator = UnifiedReplayCoordinator(...)
stats = coordinator.sleep_consolidation(n_cycles=5, batch_size=32)
```

Use:
```python
# Spontaneous replay during low acetylcholine
brain.consolidate(duration_ms=5000, verbose=True)
# Hippocampus automatically replays high-priority patterns
```

See: docs/design/emergent_rl_migration.md for full migration details.

This module provides a unified coordinator that replaces separate
ConsolidationManager, MentalSimulationCoordinator, and DynaPlanner
systems with a single biologically-grounded architecture.

**Key Insight**: All replay uses the same hippocampal CA3→CA1 machinery,
differing only in triggering context, neuromodulatory state, replay direction,
and compression speed.

Author: Thalia Project
Date: January 2026
"""

from __future__ import annotations

import warnings
import weakref
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import torch

from thalia.memory.consolidation import SleepStage, SleepStageController
from thalia.regions.hippocampus.replay_engine import ReplayEngine, ReplayMode
from thalia.replay.contexts import ReplayContext

if TYPE_CHECKING:
    from thalia.core.dynamic_brain import DynamicBrain

# Issue deprecation warning when module is imported
warnings.warn(
    "UnifiedReplayCoordinator is deprecated (Phase 2 Emergent RL). "
    "Use brain.consolidate() with spontaneous replay instead. "
    "See docs/design/emergent_rl_migration.md for migration guide.",
    DeprecationWarning,
    stacklevel=2,
)


class UnifiedReplayCoordinator:
    """Unified coordination for all hippocampal replay types.

    **DEPRECATED**: Use brain.consolidate() with spontaneous replay instead.
    This class will be removed in a future release (Phase 2 Emergent RL).

    Replaces separate ConsolidationManager, MentalSimulationCoordinator,
    and DynaPlanner with a single biologically-grounded system.

    **Biological Foundation**:
    All replay modes use the same hippocampal CA3→CA1 circuitry, as documented
    in neuroscience literature (Carr et al. 2011, Foster 2017). The differences
    emerge from:
    - Behavioral context (sleep, reward, choice, idle)
    - Neuromodulatory state (ACh, DA, NE levels)
    - Oscillatory coordination (theta vs ripple)
    - Replay direction (forward, reverse, mixed)

    **Usage** (DEPRECATED):
    ```python
    # OLD (deprecated):
    coordinator = UnifiedReplayCoordinator(...)
    stats = coordinator.sleep_consolidation(n_cycles=50, batch_size=32)

    # NEW (Phase 2):
    stats = brain.consolidate(duration_ms=5000, verbose=True)
    ```

    **Architecture**:
    - Single coordination layer for all replay modes
    - Shared hippocampal sampling with context-specific priorities
    - Unified neuromodulatory control (ACh, DA, NE)
    - ReplayEngine integration for all contexts
    - Full brain architecture support (preserves delays)

    References:
    - Carr et al. (2011): Unified hippocampal replay machinery
    - Foster (2017): Replay comes of age (Annual Review)
    - Buzsáki (2015): Sharp wave-ripples and theta coordination
    - Hasselmo (1999, 2011): Acetylcholine and replay modulation
    """

    def __init__(
        self,
        hippocampus: Any,
        striatum: Any,
        cortex: Any,
        pfc: Any,
        replay_engine: ReplayEngine,
        sleep_controller: SleepStageController,
        config: Any,
        deliver_reward_fn: Callable[[float], None],
    ):
        """Initialize unified replay coordinator.

        **DEPRECATED**: This class is deprecated. Use brain.consolidate() instead.

        Args:
            hippocampus: Hippocampus region (memory storage/retrieval)
            striatum: Striatum region (value evaluation)
            cortex: Cortex region (state representation)
            pfc: Prefrontal cortex (goal context)
            replay_engine: ReplayEngine instance for execution
            sleep_controller: SleepStageController for NREM/REM cycling
            config: Brain configuration (SimpleNamespace)
            deliver_reward_fn: Callback to deliver reward during replay
        """
        # Issue deprecation warning
        warnings.warn(
            "UnifiedReplayCoordinator is deprecated (Phase 2 Emergent RL). "
            "Use brain.consolidate() with spontaneous replay instead. "
            "This class will be removed in a future release.",
            DeprecationWarning,
            stacklevel=2,
        )

        # Core components
        self.hippocampus = hippocampus
        self.striatum = striatum
        self.cortex = cortex
        self.pfc = pfc
        self.replay_engine = replay_engine
        self.sleep_controller = sleep_controller
        self.config = config
        self._deliver_reward = deliver_reward_fn

        # Weak reference to brain for full-architecture replay
        self._brain_ref: Optional[weakref.ref[DynamicBrain]] = None

        # Unified statistics tracking
        self.replay_stats: Dict[str, Any] = {
            "sleep_replays": 0,
            "awake_immediate_replays": 0,
            "forward_planning_replays": 0,
            "background_planning_replays": 0,
            "episode_replay_counts": {},
            "total_replays": 0,
            "nrem_replays": 0,
            "rem_replays": 0,
        }

        # Cache for state reconstruction (matches ConsolidationManager)
        self._cortex_output_size: Optional[int] = None
        self._hippo_size = getattr(config, "hippocampus_size", None) or getattr(
            hippocampus, "n_output", 128
        )
        self._pfc_size = getattr(config, "pfc_size", None) or getattr(pfc, "n_output", 64)

        # Temporary storage for surprise-based DA boost
        self._surprise_da_boost: Optional[float] = None

    def set_brain_reference(self, brain: DynamicBrain) -> None:
        """Set weak reference to brain for full-architecture replay.

        Args:
            brain: DynamicBrain instance
        """
        self._brain_ref = weakref.ref(brain)

    def set_cortex_output_size(self, size: int) -> None:
        """Set cortex output size for state reconstruction.

        Args:
            size: Full cortex output size (L23 + L5)
        """
        self._cortex_output_size = size

    # ========================================================================
    # Public API: Unified replay triggering
    # ========================================================================

    def trigger_replay(
        self,
        context: ReplayContext,
        episode_indices: Optional[List[int]] = None,
        n_episodes: int = 1,
        current_state: Optional[torch.Tensor] = None,
        goal_context: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Unified replay trigger for all contexts.

        This is the core coordination method that handles all replay types.
        It orchestrates:
        1. Neuromodulatory state setting (context-specific ACh/DA/NE)
        2. Replay mode selection (forward/reverse)
        3. Episode sampling (context-specific strategy)
        4. Replay execution (through hippocampus and brain)
        5. Statistics tracking

        Args:
            context: Biological context (sleep, awake, planning)
            episode_indices: Specific episodes to replay (None = sample)
            n_episodes: Number of episodes if sampling
            current_state: Current state (for forward planning)
            goal_context: Goal context (for planning)

        Returns:
            Dictionary with replay results:
                - n_replayed: Number of episodes replayed
                - rollouts: List of replay rollouts (for planning)
                - values: List of state-action values (for planning)
                - quality: Replay quality metrics (for consolidation)
                - mode_used: ReplayMode that was used
        """
        # Set neuromodulatory state for context
        self._set_neuromodulatory_state(context)

        # Select replay mode based on context
        mode = self._get_replay_mode(context)

        # Sample or use provided episodes
        if episode_indices is None:
            episode_indices = self._sample_episodes(context, n_episodes, current_state)

        # Execute replay
        results = self._execute_replay(
            episode_indices=episode_indices,
            mode=mode,
            context=context,
            current_state=current_state,
            goal_context=goal_context,
        )

        # Update statistics
        self._update_statistics(context, episode_indices)

        return results

    # ========================================================================
    # Specialized replay methods (clean API)
    # ========================================================================

    def sleep_consolidation(
        self,
        n_cycles: int,
        batch_size: int,
    ) -> Dict[str, Any]:
        """Execute sleep consolidation with NREM/REM cycling.

        Offline memory consolidation during sleep, with sharp-wave ripple
        coordination and delta oscillator gating. Strengthens hippocampus→cortex
        connections for systems consolidation.

        Args:
            n_cycles: Number of consolidation cycles to run
            batch_size: Number of episodes per cycle

        Returns:
            Statistics dictionary with:
                - cycles_completed: Number of cycles run
                - total_replayed: Total episodes replayed
                - quality_metrics: List of quality scores per cycle
                - nrem_replays: Replays during NREM
                - rem_replays: Replays during REM
        """
        stats: Dict[str, Any] = {
            "cycles_completed": 0,
            "total_replayed": 0,
            "quality_metrics": [],
            "nrem_replays": 0,
            "rem_replays": 0,
        }

        consolidation_step = 0

        for _cycle in range(n_cycles):
            # Get current sleep stage
            stage = self.sleep_controller.get_current_stage(consolidation_step)

            # Track stage-specific replays
            if stage == SleepStage.NREM:
                stats["nrem_replays"] += batch_size
            else:
                stats["rem_replays"] += batch_size

            # Execute replay for this cycle
            result = self.trigger_replay(
                context=ReplayContext.SLEEP_CONSOLIDATION,
                n_episodes=batch_size,
            )

            stats["total_replayed"] += result["n_replayed"]
            stats["cycles_completed"] += 1

            # Track quality metrics if available
            if "quality" in result:
                stats["quality_metrics"].append(result["quality"])

            # Advance consolidation step
            consolidation_step += batch_size

        return stats

    def immediate_replay(
        self,
        episode_index: int,
        surprise_level: float = 1.0,
    ) -> None:
        """Trigger immediate awake replay after surprising event.

        Reverse replay during theta troughs, triggered by reward or surprise.
        Provides rapid credit assignment to recent actions.

        Args:
            episode_index: Episode to replay
            surprise_level: Surprise/saliency (0-1, affects DA modulation)
        """
        # Modulate dopamine based on surprise (higher surprise = more DA)
        # DA level set in _set_neuromodulatory_state, but can boost here
        da_boost = min(0.9, 0.5 + surprise_level * 0.4)

        # Store DA boost for use in neuromodulation
        self._surprise_da_boost = da_boost

        self.trigger_replay(
            context=ReplayContext.AWAKE_IMMEDIATE,
            episode_indices=[episode_index],
        )

        # Clear boost
        self._surprise_da_boost = None

    def plan_action(
        self,
        current_state: torch.Tensor,
        available_actions: List[int],
        goal_context: Optional[torch.Tensor] = None,
        depth: int = 3,
    ) -> int:
        """Plan best action using forward replay.

        Prospective forward replay at choice points. Simulates outcomes
        of available actions and selects best based on predicted value.

        Args:
            current_state: Current brain state
            available_actions: List of action indices to evaluate
            goal_context: Optional PFC goal context
            depth: Planning depth (steps ahead to simulate)

        Returns:
            Index of best action
        """
        # Run forward planning replay
        result = self.trigger_replay(
            context=ReplayContext.FORWARD_PLANNING,
            n_episodes=len(available_actions) * depth,
            current_state=current_state,
            goal_context=goal_context,
        )

        # Extract best action from results
        if "best_action" in result:
            return result["best_action"]
        else:
            # Fallback: return first action as placeholder
            # Full implementation would evaluate all actions
            return available_actions[0] if available_actions else 0

    def background_planning(
        self,
        n_simulations: int,
        goal_context: Optional[torch.Tensor] = None,
        priority_sampling: bool = True,
    ) -> Dict[str, Any]:
        """Execute background planning during idle moments.

        Opportunistic replay during awake idle periods. Refines value estimates
        and improves policy through offline simulation.

        Args:
            n_simulations: Number of simulations to run
            goal_context: Optional goal context for directed planning
            priority_sampling: Use TD-error based prioritized sampling

        Returns:
            Statistics dictionary with replay results
        """
        return self.trigger_replay(
            context=ReplayContext.BACKGROUND_PLANNING,
            n_episodes=n_simulations,
            goal_context=goal_context,
        )

    # ========================================================================
    # Internal: Neuromodulation and mode selection
    # ========================================================================

    def _set_neuromodulatory_state(self, context: ReplayContext) -> None:
        """Set ACh/NE/DA levels appropriate for replay context.

        Each context has biologically-motivated neuromodulator levels:
        - ACh: Controls encoding (high) vs retrieval (low) mode
        - NE: Arousal and attention level
        - DA: Reward signal and learning gate

        Args:
            context: Replay context
        """
        if context == ReplayContext.SLEEP_CONSOLIDATION:
            # Sleep: Low ACh enables spontaneous CA3 replay
            self.hippocampus.set_neuromodulators(
                acetylcholine=0.1,  # LOW: Enable spontaneous replay
                norepinephrine=0.1,  # LOW: Reduce noise
                dopamine=0.3,  # MODERATE: Available for learning
            )

        elif context == ReplayContext.AWAKE_IMMEDIATE:
            # Immediate replay: High ACh, high DA (surprise/reward)
            da_level = getattr(self, "_surprise_da_boost", 0.9)
            self.hippocampus.set_neuromodulators(
                acetylcholine=0.8,  # HIGH: Encoding mode
                norepinephrine=0.6,  # ELEVATED: Arousal
                dopamine=da_level,  # HIGH: Reward signal
            )

        elif context == ReplayContext.FORWARD_PLANNING:
            # Planning: High ACh (goal-directed), moderate DA
            self.hippocampus.set_neuromodulators(
                acetylcholine=0.8,  # HIGH: Goal-directed retrieval
                norepinephrine=0.5,  # MODERATE: Attention
                dopamine=0.5,  # MODERATE: Evaluation
            )

        elif context == ReplayContext.BACKGROUND_PLANNING:
            # Background: Moderate ACh (controlled simulation)
            self.hippocampus.set_neuromodulators(
                acetylcholine=0.6,  # MODERATE: Partial CA3 release
                norepinephrine=0.3,  # LOW: Idle state
                dopamine=0.4,  # MODERATE: Value updates
            )

    def _get_replay_mode(self, context: ReplayContext) -> ReplayMode:
        """Map replay context to replay engine mode.

        Args:
            context: Replay context

        Returns:
            Corresponding ReplayMode for engine
        """
        if context == ReplayContext.SLEEP_CONSOLIDATION:
            return ReplayMode.SLEEP_REVERSE
        elif context == ReplayContext.AWAKE_IMMEDIATE:
            return ReplayMode.AWAKE_REVERSE
        elif context == ReplayContext.FORWARD_PLANNING:
            return ReplayMode.AWAKE_FORWARD
        else:  # BACKGROUND_PLANNING
            # Mixed replay, default to reverse for value propagation
            return ReplayMode.SLEEP_REVERSE

    def _sample_episodes(
        self,
        context: ReplayContext,
        n_episodes: int,
        current_state: Optional[torch.Tensor],
    ) -> List[int]:
        """Sample episodes with context-appropriate strategy.

        Different contexts require different sampling strategies:
        - Sleep: Recent + high-reward experiences
        - Immediate: Just-experienced episode
        - Forward planning: Similar experiences to current state
        - Background: Prioritized by TD error

        Args:
            context: Replay context
            n_episodes: Number of episodes to sample
            current_state: Current state (for similarity-based sampling)

        Returns:
            List of episode indices to replay
        """
        if context == ReplayContext.FORWARD_PLANNING and current_state is not None:
            # For planning: retrieve SIMILAR experiences
            if hasattr(self.hippocampus, "retrieve_similar"):
                similar = self.hippocampus.retrieve_similar(
                    query_state=current_state,
                    k=n_episodes,
                )
                return [exp.get("episode_index", i) for i, exp in enumerate(similar)]

        # Default: prioritized sampling (recent + high-reward)
        if hasattr(self.hippocampus, "sample_episodes_prioritized"):
            episodes = self.hippocampus.sample_episodes_prioritized(n=n_episodes)
            return [i for i, _ in enumerate(episodes) if i < n_episodes]

        # Fallback: recent episodes
        memory_size = len(getattr(self.hippocampus, "episodic_memory", []))
        if memory_size == 0:
            return []
        return list(range(max(0, memory_size - n_episodes), memory_size))

    def _execute_replay(
        self,
        episode_indices: List[int],
        mode: ReplayMode,
        context: ReplayContext,
        current_state: Optional[torch.Tensor],
        goal_context: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        """Execute replay through hippocampus and brain architecture.

        Args:
            episode_indices: Episodes to replay
            mode: ReplayMode to use
            context: Replay context
            current_state: Current state (for planning)
            goal_context: Goal context (for planning)

        Returns:
            Replay results dictionary
        """
        results: Dict[str, Any] = {
            "n_replayed": len(episode_indices),
            "rollouts": [],
            "values": [],
            "mode_used": mode,
        }

        if not episode_indices:
            return results

        # Enter consolidation mode (enables hippocampal replay)
        was_in_consolidation = getattr(self.hippocampus, "_consolidation_mode", False)
        if not was_in_consolidation:
            self.hippocampus.enter_consolidation_mode()

        try:
            for episode_idx in episode_indices:
                # Cue hippocampus for replay
                if hasattr(self.hippocampus, "cue_replay"):
                    self.hippocampus.cue_replay(episode_idx)

                # Execute through full brain architecture (preserves delays)
                if self._brain_ref is not None:
                    brain = self._brain_ref()
                    if brain is not None:
                        # Run through brain architecture
                        brain.forward(sensory_input=None, n_timesteps=10)
                    else:
                        # Fallback if brain reference is dead
                        self._manual_replay()
                else:
                    # No brain reference, manual coordination
                    self._manual_replay()

                # Collect results for planning contexts
                if context in [ReplayContext.FORWARD_PLANNING, ReplayContext.BACKGROUND_PLANNING]:
                    # Evaluate replayed state-action value
                    if hasattr(self.striatum, "evaluate_state"):
                        ca1_spikes = getattr(self.hippocampus.state, "ca1_spikes", None)
                        if ca1_spikes is not None:
                            value = self.striatum.evaluate_state(ca1_spikes)
                            results["values"].append(value)

        finally:
            # Restore original state
            if not was_in_consolidation:
                self.hippocampus.exit_consolidation_mode()

        return results

    def _manual_replay(self) -> None:
        """Fallback: manual region coordination (no brain reference).

        Coordinates hippocampus → cortex → striatum manually when
        full brain architecture is not available.
        """
        # Forward through hippocampus
        ca1_spikes = self.hippocampus.forward({})

        # Forward through cortex
        cortex_spikes = self.cortex.forward({"hippocampus": ca1_spikes})

        # Forward through striatum
        striatum_inputs = {
            "hippocampus": ca1_spikes,
        }

        # Add cortex outputs (handle both dict and tensor)
        if isinstance(cortex_spikes, dict):
            if "l5" in cortex_spikes:
                striatum_inputs["cortex:l5"] = cortex_spikes["l5"]
        else:
            striatum_inputs["cortex"] = cortex_spikes

        self.striatum.forward(striatum_inputs)

    def _update_statistics(
        self,
        context: ReplayContext,
        episode_indices: List[int],
    ) -> None:
        """Update replay statistics.

        Args:
            context: Replay context
            episode_indices: Episodes that were replayed
        """
        n_replayed = len(episode_indices)

        # Update context-specific counters
        if context == ReplayContext.SLEEP_CONSOLIDATION:
            self.replay_stats["sleep_replays"] += n_replayed
        elif context == ReplayContext.AWAKE_IMMEDIATE:
            self.replay_stats["awake_immediate_replays"] += n_replayed
        elif context == ReplayContext.FORWARD_PLANNING:
            self.replay_stats["forward_planning_replays"] += n_replayed
        else:  # BACKGROUND_PLANNING
            self.replay_stats["background_planning_replays"] += n_replayed

        # Update total counter
        self.replay_stats["total_replays"] += n_replayed

        # Update per-episode counters
        for idx in episode_indices:
            if idx not in self.replay_stats["episode_replay_counts"]:
                self.replay_stats["episode_replay_counts"][idx] = 0
            self.replay_stats["episode_replay_counts"][idx] += 1
