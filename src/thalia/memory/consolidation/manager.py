"""
Consolidation coordinator for DynamicBrain.

This module manages memory consolidation and replay logic:
following the existing manager pattern.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any, Dict

import torch

if TYPE_CHECKING:
    from thalia.core.dynamic_brain import DynamicBrain


class ConsolidationManager:
    """Manages memory consolidation and offline replay.

    This coordinator handles hippocampal replay during sleep-like consolidation.
    It samples stored experiences, reactivates patterns, and triggers learning
    to strengthen cortical representations offline.

    Responsibilities:
    1. Experience storage - Store trials in hippocampal memory
    2. Consolidation cycles - Sample and replay experiences
    3. HER integration - Hindsight Experience Replay support
    4. Learning coordination - Trigger striatum learning during replay

    Architecture:
    - Follows existing manager pattern (PathwayManager, NeuromodulatorManager)
    - Delegates to hippocampus for memory operations
    - Coordinates with striatum for replay learning
    - Maintains backward compatibility (same external API)
    """

    def __init__(
        self,
        hippocampus: Any,
        striatum: Any,
        cortex: Any,
        pfc: Any,
        config: Any,
        deliver_reward_fn: Any,
    ):
        """Initialize consolidation manager.

        Args:
            hippocampus: Hippocampus region (for episode storage/sampling)
            striatum: Striatum region (for replay learning)
            cortex: Cortex region (for state construction)
            pfc: PFC region (for goal context)
            config: Brain configuration (SimpleNamespace)
            deliver_reward_fn: Callback to deliver_reward (for replay learning)
        """
        self.hippocampus = hippocampus
        self.striatum = striatum
        self.cortex = cortex
        self.pfc = pfc
        self.config = config
        self._deliver_reward = deliver_reward_fn

        # Store brain reference for full-architecture replay (Phase 1.7.4)
        self._brain_ref: weakref.ref[DynamicBrain] | None = None

        # Cache sizes for state reconstruction - fallback to component sizes if config doesn't have them
        self._cortex_output_size: int | None = (
            None  # Full cortex output (L23+L5), set via set_cortex_output_size()
        )
        self._hippo_size = getattr(config, "hippocampus_size", None) or getattr(
            hippocampus, "n_output", 128
        )
        self._pfc_size = getattr(config, "pfc_size", None) or getattr(pfc, "n_output", 64)

    def set_cortex_output_size(self, size: int) -> None:
        """Set cortex output size (L23+L5 combined, needed for state reconstruction)."""
        self._cortex_output_size = size

    def set_brain_reference(self, brain: DynamicBrain) -> None:
        """Set weak reference to brain for consolidation replay.

        This allows consolidation to use the full brain architecture,
        preserving axonal delays and normal pathway routing.

        Args:
            brain: DynamicBrain instance to store reference to
        """
        self._brain_ref = weakref.ref(brain)

    def _validate_striatum_sources(self) -> None:
        """Ensure striatum uses normal pathway weights (not consolidation weights).

        This validation ensures consolidation refactoring is complete -
        striatum should ONLY have normal source weights (hippocampus_d1/d2,
        cortex:l5_d1/d2, etc.), NOT special consolidation weights.
        """
        if hasattr(self.striatum, "synaptic_weights"):
            # Check for unexpected consolidation weights
            consolidation_keys = [
                key
                for key in self.striatum.synaptic_weights.keys()
                if "consolidation" in key.lower()
            ]

            if consolidation_keys:
                raise ValueError(
                    f"Striatum has 'consolidation' weights: {consolidation_keys}. "
                    "These should be removed. Consolidation should use normal "
                    "hippocampus/cortex pathway weights."
                )

    def store_experience(
        self,
        action: int,
        reward: float,
        last_action_holder: Any,  # Reference to coordinator's _last_action
    ) -> None:
        """Store experience from current brain state.

        Automatically extracts state from region activities and stores
        in hippocampal memory for later consolidation.

        Args:
            action: Action that was taken
            reward: Reward that was received
            last_action_holder: Mutable reference to coordinator's _last_action
        """
        # Infer correctness from reward
        correct = reward > 0.0

        # Construct state from current brain activity
        # Cortex outputs through both L23 and L5 (full output)
        cortex_out = None
        if hasattr(self.cortex, "state") and self.cortex.state:
            l23 = self.cortex.state.l23_spikes
            l5 = self.cortex.state.l5_spikes
            if l23 is not None and l5 is not None:
                cortex_out = torch.cat([l23, l5], dim=-1)  # Concatenate L23+L5
        if cortex_out is None:
            assert (
                self._cortex_output_size is not None
            ), "Cortex output size must be set via set_cortex_output_size()"
            cortex_out = torch.zeros(1, self._cortex_output_size, device=self.config.device)

        hippo_out = (
            self.hippocampus.state.ca1_spikes
            if hasattr(self.hippocampus, "state") and self.hippocampus.state
            else None
        )
        if hippo_out is None:
            assert self._hippo_size is not None, "Hippocampus size not set"
            hippo_out = torch.zeros(1, self._hippo_size, device=self.config.device)

        pfc_out = self.pfc.state.spikes if hasattr(self.pfc, "state") and self.pfc.state else None
        if pfc_out is None:
            assert self._pfc_size is not None, "PFC size not set"
            pfc_out = torch.zeros(1, self._pfc_size, device=self.config.device)

        combined_state = torch.cat(
            [
                cortex_out.view(-1),
                hippo_out.view(-1),
                pfc_out.view(-1),
            ]
        )

        # Priority boost for rare/important experiences
        priority_boost = 0.0
        if correct and action == 1:  # Correct NOMATCH (rare)
            priority_boost += 3.0

        # Goal-conditioned storage for HER
        goal_for_her = None
        if self.hippocampus.her_integration is not None:
            goal_for_her = pfc_out.clone()

        # Get episode index before storing (for replay cuing)
        episode_index = len(self.hippocampus.episodic_memory.episodes) if hasattr(self.hippocampus, 'episodic_memory') else 0

        # Store in hippocampus
        self.hippocampus.store_episode(
            state=combined_state,
            action=action,
            reward=reward,
            correct=correct,
            context=None,
            metadata={"episode_index": episode_index},
            priority_boost=priority_boost,
            goal=goal_for_her,
            achieved_goal=hippo_out.clone(),
            done=correct,
        )

    def consolidate(
        self,
        n_cycles: int,
        batch_size: int,
        verbose: bool,
        last_action_holder: Any,  # Mutable reference to coordinator's _last_action
    ) -> Dict[str, Any]:
        """Perform memory consolidation (replay) cycles.

        Simulates sleep/offline replay where hippocampus replays stored
        episodes to strengthen cortical representations.

        Biologically accurate consolidation:
        1. Sample experiences from hippocampal memory
        2. Replay state through brain (reactivate patterns)
        3. Deliver stored reward → dopamine → striatum learning
        4. HER automatically augments if enabled

        Args:
            n_cycles: Number of replay cycles to run
            batch_size: Number of experiences per cycle
            verbose: Whether to print progress
            last_action_holder: Mutable reference to coordinator's _last_action

        Returns:
            Dict with consolidation statistics
        """
        # Validate striatum doesn't have consolidation weights
        self._validate_striatum_sources()

        stats = {
            "cycles_completed": 0,
            "total_replayed": 0,
            "experiences_learned": 0,
            "her_enabled": (
                self.hippocampus.her_integration is not None
                if hasattr(self.hippocampus, "her_integration")
                else False
            ),
        }

        # Enter consolidation mode if HER enabled
        if (
            hasattr(self.hippocampus, "her_integration")
            and self.hippocampus.her_integration is not None
        ):
            self.hippocampus.enter_consolidation_mode()
            if verbose:
                her_diag = self.hippocampus.get_her_diagnostics()
                print(
                    f"  HER: {her_diag['n_episodes']} episodes, {her_diag['n_transitions']} transitions"
                )

        # Run replay cycles
        for cycle in range(n_cycles):
            if (
                hasattr(self.hippocampus, "her_integration")
                and self.hippocampus.her_integration is not None
            ):
                # Sample mix of real + hindsight experiences
                batch = self.hippocampus.sample_her_replay_batch(batch_size=batch_size)
                if batch:
                    stats["total_replayed"] += len(batch)

                    # Replay each experience and trigger learning
                    for experience in batch:
                        self._replay_experience(experience, last_action_holder, stats)

                    if verbose:
                        print(
                            f"  Cycle {cycle+1}/{n_cycles}: Replayed {len(batch)} experiences, {stats['experiences_learned']} learned"
                        )
            else:
                # Sample normal episodic replay
                episodes = self.hippocampus.sample_episodes_prioritized(n=batch_size)
                if episodes:
                    stats["total_replayed"] += len(episodes)

                    # Replay each episode and trigger learning
                    for episode in episodes:
                        self._replay_experience(episode, last_action_holder, stats)

                    if verbose:
                        print(
                            f"  Cycle {cycle+1}/{n_cycles}: Replayed {len(episodes)} episodes, {stats['experiences_learned']} learned"
                        )

            stats["cycles_completed"] += 1

        # Exit consolidation mode
        if (
            hasattr(self.hippocampus, "her_integration")
            and self.hippocampus.her_integration is not None
        ):
            self.hippocampus.exit_consolidation_mode()

        return stats

    def _replay_experience(
        self,
        experience: Dict[str, Any],
        last_action_holder: Any,
        stats: Dict[str, Any],
    ) -> None:
        """Replay experience using biologically accurate consolidation.

        **Biological mechanism:**
        1. Hippocampus spontaneously reactivates stored CA3→CA1 pattern
        2. CA1 spikes propagate through normal AxonalProjection pathways
        3. Both hippocampus and cortex drive striatum via normal pathways
        4. Learning modifies SAME synapses used during wake
        5. Axonal delays preserved (biologically realistic timing)

        **No special 'consolidation' weights needed!**

        Args:
            experience: Experience dict or Episode dataclass with state, action, reward
            last_action_holder: Mutable reference to coordinator's _last_action
            stats: Statistics dict to update
        """
        # Handle both dict and dataclass (Episode) formats
        if hasattr(experience, "action"):
            # Dataclass format (Episode)
            action = getattr(experience, "action")
            reward = float(getattr(experience, "reward"))
            # Get episode index from metadata
            metadata = getattr(experience, "metadata", {})
            episode_index = metadata.get("episode_index", 0) if metadata else 0
        else:
            # Dict format
            action = experience.get("action", None)  # type: ignore[union-attr]
            reward = experience.get("reward", 0.0)  # type: ignore[union-attr]
            metadata = experience.get("metadata", {})  # type: ignore[union-attr]
            episode_index = metadata.get("episode_index", 0) if metadata else 0

        if action is None:
            return

        # === BIOLOGICALLY ACCURATE REPLAY ===
        # Cue hippocampus to retrieve stored memory (simulates spontaneous reactivation)
        # The hippocampus retrieves from its OWN episodic memory buffer
        # CA3 attractor pattern propagates to CA1 via Schaffer collaterals
        self.hippocampus.cue_replay(episode_index)

        # Execute brain forward pass (consolidation replay)
        # - Hippocampus forward() spontaneously retrieves CA3→CA1 pattern
        # - CA1 output routes through AxonalProjection to cortex and striatum
        # - Striatum receives: {"hippocampus": ca1_spikes, "cortex:l5": l5_spikes}
        # - Learning uses NORMAL synaptic weights (hippocampus_d1/d2, cortex:l5_d1/d2)
        # - Axonal delays preserved (biologically realistic timing)
        self._run_consolidation_replay(n_timesteps=10)  # Brief replay window

        # Set action and deliver reward (triggers dopamine-gated learning!)
        last_action_holder[0] = action
        self._deliver_reward(external_reward=reward)
        stats["experiences_learned"] += 1

    def _run_consolidation_replay(self, n_timesteps: int = 10) -> None:
        """Execute consolidation replay through full brain architecture.

        Uses complete brain forward pass to preserve:
        - Axonal delays (AxonalProjection pathways)
        - Multi-region interactions (hippocampus→cortex→striatum)
        - Normal synaptic routing (no special cases)
        - Biological timing (time-compressed 10x but delays preserved)

        Args:
            n_timesteps: Number of timesteps for replay window (time-compressed)
        """
        # Get brain reference
        brain = self._brain_ref() if self._brain_ref else None

        if brain is not None:
            # PREFERRED: Use full brain architecture
            # sensory_input=None → no external input, hippocampus drives internally
            brain.forward(sensory_input=None, n_timesteps=n_timesteps)
        else:
            # FALLBACK: Manual execution (less ideal - bypasses AxonalProjection delays)
            # Hippocampus spontaneously generates CA1 spikes from stored memory
            ca1_spikes = self.hippocampus.forward({})  # Consolidation mode

            # Cortex receives hippocampal feedback and pattern completes
            # (Systems consolidation mechanism - not implemented in this phase)
            cortex_spikes = self.cortex.forward({})

            # Striatum receives multi-source input via NORMAL source names
            striatum_inputs = {
                "hippocampus": ca1_spikes,
                "cortex:l5": cortex_spikes.get("l5") if isinstance(cortex_spikes, dict) else cortex_spikes,
            }

            # Striatum forward with NORMAL multi-source architecture
            self.striatum.forward(striatum_inputs)
