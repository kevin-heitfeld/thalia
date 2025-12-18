"""
Consolidation coordinator for DynamicBrain.

This module manages memory consolidation and replay logic:
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
        self._cortex_output_size = None  # Full cortex output (L23+L5)
        self._hippo_size = config.hippocampus_size
        self._pfc_size = config.pfc_size

    def set_cortex_output_size(self, size: int) -> None:
        """Set cortex output size (L23+L5 combined, needed for state reconstruction)."""
        self._cortex_output_size = size

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
        if hasattr(self.cortex, 'state') and self.cortex.state:
            l23 = self.cortex.state.l23_spikes
            l5 = self.cortex.state.l5_spikes
            if l23 is not None and l5 is not None:
                cortex_out = torch.cat([l23, l5], dim=-1)  # Concatenate L23+L5
        if cortex_out is None:
            cortex_out = torch.zeros(1, self._cortex_output_size, device=self.config.device)

        hippo_out = self.hippocampus.state.ca1_spikes if hasattr(self.hippocampus, 'state') and self.hippocampus.state else None
        if hippo_out is None:
            hippo_out = torch.zeros(1, self._hippo_size, device=self.config.device)

        pfc_out = self.pfc.state.spikes if hasattr(self.pfc, 'state') and self.pfc.state else None
        if pfc_out is None:
            pfc_out = torch.zeros(1, self._pfc_size, device=self.config.device)

        combined_state = torch.cat([
            cortex_out.view(-1),
            hippo_out.view(-1),
            pfc_out.view(-1),
        ])

        # Priority boost for rare/important experiences
        priority_boost = 0.0
        if correct and action == 1:  # Correct NOMATCH (rare)
            priority_boost += 3.0

        # Goal-conditioned storage for HER
        goal_for_her = None
        if self.hippocampus.her_integration is not None:
            goal_for_her = pfc_out.clone()

        # Store in hippocampus
        self.hippocampus.store_episode(
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
            'her_enabled': (
                self.hippocampus.her_integration is not None if hasattr(self.hippocampus, 'her_integration') else False
            ),
        }

        # Enter consolidation mode if HER enabled
        if hasattr(self.hippocampus, 'her_integration') and self.hippocampus.her_integration is not None:
            self.hippocampus.enter_consolidation_mode()
            if verbose:
                her_diag = self.hippocampus.get_her_diagnostics()
                print(f"  HER: {her_diag['n_episodes']} episodes, {her_diag['n_transitions']} transitions")

        # Run replay cycles
        for cycle in range(n_cycles):
            if hasattr(self.hippocampus, 'her_integration') and self.hippocampus.her_integration is not None:
                # Sample mix of real + hindsight experiences
                batch = self.hippocampus.sample_her_replay_batch(batch_size=batch_size)
                if batch:
                    stats['total_replayed'] += len(batch)

                    # Replay each experience and trigger learning
                    for experience in batch:
                        self._replay_experience(experience, last_action_holder, stats)

                    if verbose:
                        print(f"  Cycle {cycle+1}/{n_cycles}: Replayed {len(batch)} experiences, {stats['experiences_learned']} learned")
            else:
                # Sample normal episodic replay
                episodes = self.hippocampus.sample_episodes_prioritized(n=batch_size)
                if episodes:
                    stats['total_replayed'] += len(episodes)

                    # Replay each episode and trigger learning
                    for episode in episodes:
                        self._replay_experience(episode, last_action_holder, stats)

                    if verbose:
                        print(f"  Cycle {cycle+1}/{n_cycles}: Replayed {len(episodes)} episodes, {stats['experiences_learned']} learned")

            stats['cycles_completed'] += 1

        # Exit consolidation mode
        if hasattr(self.hippocampus, 'her_integration') and self.hippocampus.her_integration is not None:
            self.hippocampus.exit_consolidation_mode()

        return stats

    def _replay_experience(
        self,
        experience: Dict[str, Any],
        last_action_holder: Any,
        stats: Dict[str, Any],
    ) -> None:
        """Replay a single experience and trigger learning.

        Args:
            experience: Experience dict or Episode dataclass with state, action, reward
            last_action_holder: Mutable reference to coordinator's _last_action
            stats: Statistics dict to update
        """
        # Handle both dict and dataclass (Episode) formats
        if hasattr(experience, 'action'):
            # Dataclass format (Episode)
            action = experience.action
            reward = experience.reward
            state = experience.state
        else:
            # Dict format
            action = experience.get('action', None)
            reward = experience.get('reward', 0.0)
            state = experience.get('state', None)

        if action is None or state is None:
            return

        # Reconstruct state components
        cortex_size = self._cortex_output_size  # Full cortex output (L23+L5)
        hippo_size = self._hippo_size
        pfc_size = self._pfc_size

        cortex_state = state[:cortex_size]
        hippo_state = state[cortex_size:cortex_size + hippo_size]
        pfc_state = state[cortex_size + hippo_size:]

        # Reactivate pattern in striatum
        # Note: Striatum expects 1D input (ADR-005), no batch dimension
        striatum_input = torch.cat([
            cortex_state,
            hippo_state,
            pfc_state,
        ], dim=-1)
        _ = self.striatum.forward(striatum_input)

        # Set action and deliver reward (triggers learning!)
        last_action_holder[0] = action
        self._deliver_reward(external_reward=reward)
        stats['experiences_learned'] += 1
