"""
Consolidation coordinator for EventDrivenBrain.

This module extracts memory consolidation and replay logic from EventDrivenBrain,
following the existing manager pattern.

Author: Thalia Project
Date: December 2025
"""

from typing import Dict, Any
import torch


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

        # Cache sizes for state reconstruction
        self._cortex_l5_size = None
        self._hippo_size = config.hippocampus_size
        self._pfc_size = config.pfc_size

    def set_cortex_l5_size(self, size: int) -> None:
        """Set cortex L5 size (needed for state reconstruction)."""
        self._cortex_l5_size = size

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
        cortex_L5 = self.cortex.impl.state.l5_spikes
        if cortex_L5 is None:
            cortex_L5 = torch.zeros(1, self._cortex_l5_size, device=self.config.device)

        hippo_out = self.hippocampus.impl.state.ca1_spikes
        if hippo_out is None:
            hippo_out = torch.zeros(1, self._hippo_size, device=self.config.device)

        pfc_out = self.pfc.impl.state.spikes
        if pfc_out is None:
            pfc_out = torch.zeros(1, self._pfc_size, device=self.config.device)

        combined_state = torch.cat([
            cortex_L5.view(-1),
            hippo_out.view(-1),
            pfc_out.view(-1),
        ])

        # Priority boost for rare/important experiences
        priority_boost = 0.0
        if correct and action == 1:  # Correct NOMATCH (rare)
            priority_boost += 3.0

        # Goal-conditioned storage for HER
        goal_for_her = None
        if self.hippocampus.impl.her_integration is not None:
            goal_for_her = pfc_out.clone()

        # Store in hippocampus
        self.hippocampus.impl.store_episode(
            state=combined_state,
            action=action,
            reward=reward,
            correct=correct,
            context=None,
            metadata={},
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
        stats = {
            'cycles_completed': 0,
            'total_replayed': 0,
            'experiences_learned': 0,
            'her_enabled': self.hippocampus.impl.her_integration is not None,
        }

        # Enter consolidation mode if HER enabled
        if self.hippocampus.impl.her_integration is not None:
            self.hippocampus.impl.enter_consolidation_mode()
            if verbose:
                her_diag = self.hippocampus.impl.get_her_diagnostics()
                print(f"  HER: {her_diag['n_episodes']} episodes, {her_diag['n_transitions']} transitions")

        # Run replay cycles
        for cycle in range(n_cycles):
            if self.hippocampus.impl.her_integration is not None:
                # Sample mix of real + hindsight experiences
                batch = self.hippocampus.impl.sample_her_replay_batch(batch_size=batch_size)
                if batch:
                    stats['total_replayed'] += len(batch)

                    # Replay each experience and trigger learning
                    for experience in batch:
                        self._replay_experience(experience, last_action_holder, stats)

                    if verbose:
                        print(f"  Cycle {cycle+1}/{n_cycles}: Replayed {len(batch)} experiences, {stats['experiences_learned']} learned")
            else:
                # Sample normal episodic replay
                episodes = self.hippocampus.impl.sample_episodes_prioritized(n=batch_size)
                if episodes:
                    stats['total_replayed'] += len(episodes)

                    # Replay each episode and trigger learning
                    for episode in episodes:
                        self._replay_experience(episode, last_action_holder, stats)

                    if verbose:
                        print(f"  Cycle {cycle+1}/{n_cycles}: Replayed {len(episodes)} episodes, {stats['experiences_learned']} learned")

            stats['cycles_completed'] += 1

        # Exit consolidation mode
        if self.hippocampus.impl.her_integration is not None:
            self.hippocampus.impl.exit_consolidation_mode()

        return stats

    def _replay_experience(
        self,
        experience: Dict[str, Any],
        last_action_holder: Any,
        stats: Dict[str, Any],
    ) -> None:
        """Replay a single experience and trigger learning.

        Args:
            experience: Experience dict with state, action, reward
            last_action_holder: Mutable reference to coordinator's _last_action
            stats: Statistics dict to update
        """
        action = experience.get('action', None)
        reward = experience.get('reward', 0.0)
        state = experience.get('state', None)

        if action is None or state is None:
            return

        # Reconstruct state components
        cortex_size = self._cortex_l5_size
        hippo_size = self._hippo_size
        pfc_size = self._pfc_size

        cortex_state = state[:cortex_size]
        hippo_state = state[cortex_size:cortex_size + hippo_size]
        pfc_state = state[cortex_size + hippo_size:]

        # Reactivate pattern in striatum
        striatum_input = torch.cat([
            cortex_state.unsqueeze(0),
            hippo_state.unsqueeze(0),
            pfc_state.unsqueeze(0),
        ], dim=-1)
        _ = self.striatum.impl.forward(striatum_input)

        # Set action and deliver reward (triggers learning!)
        last_action_holder[0] = action
        self._deliver_reward(external_reward=reward)
        stats['experiences_learned'] += 1
