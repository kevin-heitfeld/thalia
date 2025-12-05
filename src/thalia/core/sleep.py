"""
Sleep and Consolidation System for EventDrivenBrain.

This module provides sleep-related functionality as a mixin class,
implementing biologically realistic sleep stages:
- N1 (drowsy): Brief transition, theta waves
- N2 (light): Sleep spindles, moderate consolidation
- N3/SWS (deep): Sharp-wave ripples, MAXIMUM consolidation
- REM: Theta oscillations, stochastic replay, generalization

Sleep Architecture:
- Early cycles: More SWS (memory consolidation priority)
- Late cycles: More REM (generalization, creativity)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Dict, Any, List

import torch

if TYPE_CHECKING:
    from ..regions.hippocampus import Episode


class SleepSystemMixin:
    """Mixin providing sleep/consolidation methods for EventDrivenBrain.
    
    Implements biologically realistic sleep stages for memory consolidation.
    
    Expects the following attributes on the mixed-in class:
    - hippocampus: with episode_buffer and sample_episodes_prioritized()
    - striatum: with d1_weights, d2_weights, forward(), deliver_reward()
    - cortex: with forward()
    - replay_pathway: with set_sleep_stage(), trigger_ripple(), replay_step(), learn()
    - theta: with advance()
    - config: with input_size, hippocampus_size, pfc_size
    - _cortex_l5_size: int
    """
    
    # Type hints for mixin - these are provided by EventDrivenBrain
    hippocampus: Any
    striatum: Any
    cortex: Any
    replay_pathway: Any
    theta: Any
    config: Any
    _cortex_l5_size: int

    def sleep_epoch(
        self,
        n_cycles: int = 3,
        replays_per_cycle: int = 30,
        n_timesteps: int = 10,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run a sleep epoch with biologically realistic sleep stages.

        Models complete sleep architecture with all stages:
        - N1 (drowsy): Brief transition, theta waves
        - N2 (light): Sleep spindles, moderate consolidation
        - N3/SWS (deep): Sharp-wave ripples, MAXIMUM consolidation
        - REM: Theta oscillations, stochastic replay, generalization

        Sleep Architecture Changes Through Night:
        - Early cycles: More SWS (memory consolidation priority)
        - Late cycles: More REM (generalization, creativity)

        Args:
            n_cycles: Number of sleep cycles (real brain: 4-6 per night)
            replays_per_cycle: Total replays per cycle
            n_timesteps: Timesteps per replay
            verbose: Print cycle details

        Returns:
            Dict with comprehensive sleep metrics
        """
        if len(self.hippocampus.episode_buffer) == 0:
            return {"cycles": 0, "total_replays": 0, "message": "No episodes to replay"}

        # Track overall metrics
        total_replays = 0
        total_n2_replays = 0
        total_sws_replays = 0
        total_rem_replays = 0

        d1_start = self.striatum.d1_weights.clone()
        d2_start = self.striatum.d2_weights.clone()

        cycle_metrics: List[Dict[str, Any]] = []

        for cycle_idx in range(n_cycles):
            # Biological sleep architecture:
            # - Early night: ~10% N2, 70% SWS, 20% REM
            # - Late night: ~15% N2, 35% SWS, 50% REM
            progress = cycle_idx / max(1, n_cycles - 1)
            n2_fraction = 0.10 + 0.05 * progress
            sws_fraction = 0.70 - 0.35 * progress
            rem_fraction = 1.0 - n2_fraction - sws_fraction

            n_n2 = max(1, int(replays_per_cycle * n2_fraction))
            n_sws = int(replays_per_cycle * sws_fraction)
            n_rem = replays_per_cycle - n_sws - n_n2

            cycle_d1_before = self.striatum.d1_weights.clone()

            # ================================================================
            # N1 STAGE (DROWSY / FALLING ASLEEP)
            # ================================================================
            self.replay_pathway.set_sleep_stage("n1")
            self._run_drowsy_transition(n_timesteps=10)

            # ================================================================
            # N2 STAGE (LIGHT SLEEP WITH SPINDLES)
            # ================================================================
            self.replay_pathway.set_sleep_stage("n2")
            n2_metrics = self._run_n2_replays(n_n2, n_timesteps)
            total_n2_replays += n2_metrics["replays"]

            # ================================================================
            # SLOW-WAVE SLEEP (N3/SWS) PHASE
            # ================================================================
            self.replay_pathway.set_sleep_stage("sws")
            sws_metrics = self._run_sws_replays(n_sws, n_timesteps)
            total_sws_replays += sws_metrics["replays"]

            # ================================================================
            # REM SLEEP PHASE
            # ================================================================
            self.replay_pathway.set_sleep_stage("rem")
            rem_metrics = self._run_rem_replays(n_rem, n_timesteps)
            total_rem_replays += rem_metrics["replays"]

            total_replays += n_n2 + n_sws + n_rem

            # Compute per-cycle metrics
            cycle_d1_delta = (self.striatum.d1_weights - cycle_d1_before).abs().mean().item()

            cycle_metrics.append({
                "cycle": cycle_idx + 1,
                "n2_replays": n_n2,
                "sws_replays": n_sws,
                "rem_replays": n_rem,
                "n2_fraction": n2_fraction,
                "sws_fraction": sws_fraction,
                "rem_fraction": rem_fraction,
                "d1_delta": cycle_d1_delta,
            })

            if verbose:
                print(f"  Sleep Cycle {cycle_idx + 1}/{n_cycles}: "
                      f"N2={n_n2}, SWS={n_sws}, REM={n_rem}, "
                      f"ΔD1={cycle_d1_delta:.4f}")

        # Return to wake state
        self.replay_pathway.set_sleep_stage("wake")

        # Compute total weight changes
        d1_total_delta = (self.striatum.d1_weights - d1_start).abs().mean().item()
        d2_total_delta = (self.striatum.d2_weights - d2_start).abs().mean().item()

        return {
            "cycles": n_cycles,
            "total_replays": total_replays,
            "n2_replays": total_n2_replays,
            "sws_replays": total_sws_replays,
            "rem_replays": total_rem_replays,
            "d1_delta": d1_total_delta,
            "d2_delta": d2_total_delta,
            "cycle_metrics": cycle_metrics,
        }

    def _run_drowsy_transition(self, n_timesteps: int = 10) -> None:
        """N1 stage - brief transition letting neural activity settle."""
        # Run with no input, letting activity decay naturally
        for _ in range(n_timesteps):
            self.theta.advance()
            # Just let membrane potentials decay

    def _settle_for_replay(self, n_timesteps: int = 5) -> None:
        """Let neural activity settle between replays."""
        null_input = torch.zeros(1, self.config.input_size)
        for _ in range(n_timesteps):
            self.cortex.forward(null_input)

    def _run_n2_replays(
        self,
        n_replays: int,
        n_timesteps: int,
    ) -> Dict[str, Any]:
        """Run N2 (light sleep) replays with moderate learning."""
        n2_match = 0
        n2_nomatch = 0

        # Sample with moderate priority bias for N2 spindles
        episodes = self._sample_episodes_for_n2(n_replays)

        # Striatum expects: cortex_l5 + hippocampus + pfc
        striatum_input_size = (
            self._cortex_l5_size +
            self.config.hippocampus_size +
            self.config.pfc_size
        )

        for episode in episodes:
            self._settle_for_replay(n_timesteps=5)

            # Build striatum input from episode state
            # Episode state is hippocampus activity, pad with zeros for cortex/pfc
            striatum_input = torch.zeros(1, striatum_input_size)
            # Place hippocampus activity in the right position
            hippo_start = self._cortex_l5_size
            state_size = min(episode.state.shape[0], self.config.hippocampus_size)
            striatum_input[0, hippo_start:hippo_start + state_size] = episode.state[:state_size]

            # Replay through striatum with moderate learning
            for _ in range(n_timesteps):
                self.striatum.forward(striatum_input, explore=False)
                self.striatum.last_action = episode.action

            # Moderate reward strength during N2 (60%)
            self.striatum.deliver_reward(reward=episode.reward * 0.6)

            is_match = episode.metadata.get("is_match", True) if episode.metadata else True
            if is_match:
                n2_match += 1
            else:
                n2_nomatch += 1

        return {
            "replays": len(episodes),
            "match": n2_match,
            "nomatch": n2_nomatch,
        }

    def _run_sws_replays(
        self,
        n_replays: int,
        n_timesteps: int,
    ) -> Dict[str, Any]:
        """Run SWS (slow-wave sleep) replays with FULL learning."""
        sws_match = 0
        sws_nomatch = 0

        # Priority-weighted sampling for SWS
        episodes = self.hippocampus.sample_episodes_prioritized(n_replays)

        # Striatum expects: cortex_l5 + hippocampus + pfc
        striatum_input_size = (
            self._cortex_l5_size +
            self.config.hippocampus_size +
            self.config.pfc_size
        )

        for episode in episodes:
            self._settle_for_replay(n_timesteps=5)

            # Build striatum input from episode state
            striatum_input = torch.zeros(1, striatum_input_size)
            hippo_start = self._cortex_l5_size
            state_size = min(episode.state.shape[0], self.config.hippocampus_size)
            striatum_input[0, hippo_start:hippo_start + state_size] = episode.state[:state_size]

            # Replay with FULL learning during SWS
            for _ in range(n_timesteps):
                self.striatum.forward(striatum_input, explore=False)
                self.striatum.last_action = episode.action

            # FULL reward strength during SWS
            self.striatum.deliver_reward(reward=episode.reward)

            # Hippocampus → Cortex consolidation via replay pathway
            if episode.context is not None:
                hippo_activity = episode.state[:self.config.hippocampus_size].unsqueeze(0)
                self.replay_pathway.trigger_ripple()
                replay_signal = self.replay_pathway.replay_step(dt=1.0)

                if replay_signal is not None:
                    cortex_input = torch.zeros(1, self.config.input_size)
                    min_size = min(replay_signal.shape[1], self.config.input_size)
                    cortex_input[:, :min_size] = replay_signal[:, :min_size]
                    cortex_activity = self.cortex.forward(cortex_input)

                    self.replay_pathway.learn(
                        pre_spikes=hippo_activity,
                        post_spikes=cortex_activity,
                        reward=0.5,
                        dt=1.0,
                    )

            is_match = episode.metadata.get("is_match", True) if episode.metadata else True
            if is_match:
                sws_match += 1
            else:
                sws_nomatch += 1

        return {
            "replays": len(episodes),
            "match": sws_match,
            "nomatch": sws_nomatch,
        }

    def _run_rem_replays(
        self,
        n_replays: int,
        n_timesteps: int,
    ) -> Dict[str, Any]:
        """Run REM replays with reduced learning and stochastic noise."""
        rem_match = 0
        rem_nomatch = 0

        # Stochastic sampling for REM (generalization, less priority bias)
        episodes = self._sample_episodes_for_rem(n_replays)

        # Striatum expects: cortex_l5 + hippocampus + pfc
        striatum_input_size = (
            self._cortex_l5_size +
            self.config.hippocampus_size +
            self.config.pfc_size
        )

        for episode in episodes:
            self._settle_for_replay(n_timesteps=5)

            # Build striatum input from episode state
            striatum_input = torch.zeros(1, striatum_input_size)
            hippo_start = self._cortex_l5_size
            state_size = min(episode.state.shape[0], self.config.hippocampus_size)
            striatum_input[0, hippo_start:hippo_start + state_size] = episode.state[:state_size]

            # Replay with REDUCED learning and theta modulation
            for t in range(n_timesteps):
                # Theta modulation (~6 Hz during REM)
                theta_phase = 2 * math.pi * 6.0 * t / 1000.0
                theta_mod = 0.5 * (1 + math.cos(theta_phase))

                # Add stochastic noise for REM-like variability
                noise = torch.randn_like(striatum_input) * 0.1 * theta_mod
                self.striatum.forward(striatum_input + noise, explore=False)
                self.striatum.last_action = episode.action

            # REDUCED reward strength during REM (30%)
            self.striatum.deliver_reward(reward=episode.reward * 0.3)

            is_match = episode.metadata.get("is_match", True) if episode.metadata else True
            if is_match:
                rem_match += 1
            else:
                rem_nomatch += 1

        return {
            "replays": len(episodes),
            "match": rem_match,
            "nomatch": rem_nomatch,
        }

    def _sample_episodes_random(self, n: int) -> List["Episode"]:
        """Sample episodes randomly (for N2 and REM)."""
        buffer = self.hippocampus.episode_buffer
        if len(buffer) == 0:
            return []

        n = min(n, len(buffer))
        indices = torch.randperm(len(buffer))[:n].tolist()
        return [buffer[i] for i in indices]

    def _sample_episodes_for_n2(self, n: int) -> List["Episode"]:
        """
        Sample episodes for N2 (light sleep with spindles).

        N2 spindles have moderate priority bias - between random and
        fully priority-weighted. Spindles help transfer memories from
        hippocampus to cortex.

        Args:
            n: Number of episodes to sample

        Returns:
            List of sampled episodes
        """
        buffer = self.hippocampus.episode_buffer
        if len(buffer) == 0:
            return []

        n = min(n, len(buffer))

        # Moderate priority bias for spindle-related consolidation
        priorities = torch.tensor([
            ep.priority for ep in buffer
        ])

        # Add moderate noise (less than REM, more than SWS)
        noisy_priorities = priorities + torch.randn_like(priorities) * 0.3

        # Softmax with moderate temperature
        probs = torch.softmax(noisy_priorities * 1.0, dim=0)  # temperature = 1.0

        indices = torch.multinomial(probs, n, replacement=False)
        return [buffer[i] for i in indices.tolist()]

    def _sample_episodes_for_rem(self, n: int) -> List["Episode"]:
        """
        Sample episodes for REM sleep with less priority bias.

        REM sleep is characterized by more stochastic/random replay,
        which helps with generalization and creative associations.

        Args:
            n: Number of episodes to sample

        Returns:
            List of sampled episodes
        """
        buffer = self.hippocampus.episode_buffer
        if len(buffer) == 0:
            return []

        n = min(n, len(buffer))

        # Use softmax with lower temperature for more uniform sampling
        # (less priority-biased than SWS)
        priorities = torch.tensor([
            ep.priority for ep in buffer
        ])

        # Add noise to priorities for stochasticity
        noisy_priorities = priorities + torch.randn_like(priorities) * 0.5

        # Softmax with lower temperature (more uniform)
        probs = torch.softmax(noisy_priorities * 0.5, dim=0)  # temperature = 2.0

        indices = torch.multinomial(probs, n, replacement=False)
        return [buffer[i] for i in indices.tolist()]
