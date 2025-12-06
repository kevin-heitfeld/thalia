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
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Dict, Any, List, Optional

import torch

if TYPE_CHECKING:
    from ..regions.hippocampus import Episode


class SleepStage(Enum):
    """Sleep stages with distinct replay characteristics."""
    N2 = auto()   # Light sleep - moderate learning
    SWS = auto()  # Slow-wave sleep - full learning + consolidation
    REM = auto()  # REM - reduced learning + noise


@dataclass
class StageConfig:
    """Configuration for a sleep stage's replay behavior.
    
    Attributes:
        reward_multiplier: Scale factor for reward delivery (0.0-1.0)
        add_noise: Whether to add stochastic noise during replay
        noise_scale: Base amplitude of noise (if add_noise is True)
        do_consolidation: Whether to run hippocampus→cortex consolidation
        temperature: Softmax temperature for episode sampling (higher = more uniform)
        priority_noise: Noise added to priorities before sampling
    """
    reward_multiplier: float = 1.0
    add_noise: bool = False
    noise_scale: float = 0.0
    do_consolidation: bool = False
    temperature: float = 1.0
    priority_noise: float = 0.0


# Default configurations for each sleep stage
STAGE_CONFIGS: Dict[SleepStage, StageConfig] = {
    SleepStage.N2: StageConfig(
        reward_multiplier=0.6,
        add_noise=False,
        noise_scale=0.0,
        do_consolidation=False,
        temperature=1.0,
        priority_noise=0.3,
    ),
    SleepStage.SWS: StageConfig(
        reward_multiplier=1.0,
        add_noise=False,
        noise_scale=0.0,
        do_consolidation=True,
        temperature=float('inf'),  # Use prioritized sampling directly
        priority_noise=0.0,
    ),
    SleepStage.REM: StageConfig(
        reward_multiplier=0.3,
        add_noise=True,
        noise_scale=0.1,
        do_consolidation=False,
        temperature=2.0,  # More uniform sampling
        priority_noise=0.5,
    ),
}


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

    # ========================================================================
    # Issue 8: Striatum Input Builder Helper
    # ========================================================================
    
    def _get_striatum_input_size(self) -> int:
        """Get the expected input size for striatum (cortex_l5 + hippocampus + pfc)."""
        return (
            self._cortex_l5_size +
            self.config.hippocampus_size +
            self.config.pfc_size
        )
    
    def _build_striatum_input(self, episode: "Episode") -> torch.Tensor:
        """Build striatum input tensor from an episode's hippocampal state.
        
        The striatum expects concatenated input from cortex L5, hippocampus, 
        and PFC. This helper constructs that input by placing the episode's
        hippocampal state in the correct position.
        
        Args:
            episode: Episode containing hippocampal state to replay
            
        Returns:
            Tensor of shape (1, striatum_input_size) ready for striatum.forward()
        """
        striatum_input_size = self._get_striatum_input_size()
        striatum_input = torch.zeros(1, striatum_input_size)
        
        # Place hippocampus activity in the correct position (after cortex L5)
        hippo_start = self._cortex_l5_size
        state_size = min(episode.state.shape[0], self.config.hippocampus_size)
        striatum_input[0, hippo_start:hippo_start + state_size] = episode.state[:state_size]
        
        return striatum_input

    # ========================================================================
    # Issue 9: Unified Episode Sampling
    # ========================================================================
    
    def _sample_episodes(
        self,
        n: int,
        temperature: float = 1.0,
        priority_noise: float = 0.0,
    ) -> List["Episode"]:
        """Sample episodes with configurable priority bias and stochasticity.
        
        Uses softmax over (noisy) priorities with temperature control:
        - temperature → ∞: Use hippocampus.sample_episodes_prioritized() directly
        - temperature = 1.0: Standard priority-weighted sampling
        - temperature > 1.0: More uniform sampling (less priority bias)
        - temperature < 1.0: Sharper priority bias
        
        Args:
            n: Number of episodes to sample
            temperature: Softmax temperature (higher = more uniform)
            priority_noise: Gaussian noise scale added to priorities
            
        Returns:
            List of sampled episodes
        """
        buffer = self.hippocampus.episode_buffer
        if len(buffer) == 0:
            return []
        
        n = min(n, len(buffer))
        
        # Special case: infinite temperature means use prioritized sampling directly
        if temperature == float('inf'):
            return self.hippocampus.sample_episodes_prioritized(n)
        
        # Compute priorities with optional noise
        priorities = torch.tensor([ep.priority for ep in buffer])
        
        if priority_noise > 0:
            priorities = priorities + torch.randn_like(priorities) * priority_noise
        
        # Apply temperature: divide logits by temperature before softmax
        # (equivalent to softmax(x * (1/temperature)))
        probs = torch.softmax(priorities / temperature, dim=0)
        
        indices = torch.multinomial(probs, n, replacement=False)
        return [buffer[i] for i in indices.tolist()]

    # ========================================================================
    # Issue 7: Unified Stage Replay Method
    # ========================================================================
    
    def _run_stage_replays(
        self,
        stage: SleepStage,
        n_replays: int,
        n_timesteps: int,
        stage_config: Optional[StageConfig] = None,
    ) -> Dict[str, Any]:
        """Run replays for a sleep stage with stage-specific parameters.
        
        This unified method handles all sleep stages (N2, SWS, REM) with
        configurable behavior for:
        - Episode sampling strategy (temperature, priority noise)
        - Reward scaling
        - Noise injection (REM)
        - Hippocampus→Cortex consolidation (SWS)
        
        Args:
            stage: Sleep stage (N2, SWS, or REM)
            n_replays: Number of episodes to replay
            n_timesteps: Timesteps per replay
            stage_config: Optional custom configuration (defaults to STAGE_CONFIGS[stage])
            
        Returns:
            Dict with replay metrics: replays, match, nomatch counts
        """
        config = stage_config or STAGE_CONFIGS[stage]
        
        # Sample episodes using unified sampler
        episodes = self._sample_episodes(
            n_replays,
            temperature=config.temperature,
            priority_noise=config.priority_noise,
        )
        
        match_count = 0
        nomatch_count = 0
        
        for episode in episodes:
            self._settle_for_replay(n_timesteps=5)
            
            # Build input using helper (Issue 8)
            striatum_input = self._build_striatum_input(episode)
            
            # Run timesteps with optional noise (REM)
            for t in range(n_timesteps):
                current_input = striatum_input
                
                if config.add_noise:
                    # Theta modulation (~6 Hz during REM)
                    theta_phase = 2 * math.pi * 6.0 * t / 1000.0
                    theta_mod = 0.5 * (1 + math.cos(theta_phase))
                    noise = torch.randn_like(striatum_input) * config.noise_scale * theta_mod
                    current_input = striatum_input + noise
                
                self.striatum.forward(current_input, explore=False)
                self.striatum.last_action = episode.action
            
            # Deliver scaled reward
            self.striatum.deliver_reward(reward=episode.reward * config.reward_multiplier)
            
            # SWS-specific: Hippocampus → Cortex consolidation
            if config.do_consolidation and episode.context is not None:
                self._run_consolidation(episode)
            
            # Track match/nomatch
            is_match = episode.metadata.get("is_match", True) if episode.metadata else True
            if is_match:
                match_count += 1
            else:
                nomatch_count += 1
        
        return {
            "replays": len(episodes),
            "match": match_count,
            "nomatch": nomatch_count,
        }
    
    def _run_consolidation(self, episode: "Episode") -> None:
        """Run hippocampus → cortex consolidation for an episode (SWS-specific).
        
        This method uses gamma-driven replay when sequences are available.
        During sleep, sequences that took seconds to encode are replayed
        time-compressed (~5-20x faster), with the gamma oscillator driving
        slot-by-slot reactivation.
        
        For episodes without sequences, falls back to single-state replay.
        """
        # Try gamma-driven sequence replay first
        if hasattr(self.hippocampus, 'replay_sequence') and episode.sequence is not None:
            # Use gamma oscillator for time-compressed sequence replay
            replay_result = self.hippocampus.replay_sequence(
                episode,
                compression_factor=5.0,  # 5x faster than encoding
                dt=1.0,
            )
            
            # Consolidate each replayed pattern to cortex
            for pattern in replay_result.get("replayed_patterns", []):
                if pattern is not None:
                    cortex_input = torch.zeros(1, self.config.input_size)
                    if pattern.dim() == 1:
                        pattern = pattern.unsqueeze(0)
                    min_size = min(pattern.shape[1], self.config.input_size)
                    cortex_input[:, :min_size] = pattern[:, :min_size]
                    cortex_activity = self.cortex.forward(cortex_input)
                    
                    # Learning during replay
                    self.replay_pathway.learn(
                        pre_spikes=pattern,
                        post_spikes=cortex_activity,
                        reward=0.5,
                        dt=1.0,
                    )
        else:
            # Fall back to original single-state replay
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

    # ========================================================================
    # Stage-Specific Methods (now delegate to unified method)
    # ========================================================================

    def _run_n2_replays(
        self,
        n_replays: int,
        n_timesteps: int,
    ) -> Dict[str, Any]:
        """Run N2 (light sleep) replays with moderate learning."""
        return self._run_stage_replays(SleepStage.N2, n_replays, n_timesteps)

    def _run_sws_replays(
        self,
        n_replays: int,
        n_timesteps: int,
    ) -> Dict[str, Any]:
        """Run SWS (slow-wave sleep) replays with FULL learning."""
        return self._run_stage_replays(SleepStage.SWS, n_replays, n_timesteps)

    def _run_rem_replays(
        self,
        n_replays: int,
        n_timesteps: int,
    ) -> Dict[str, Any]:
        """Run REM replays with reduced learning and stochastic noise."""
        return self._run_stage_replays(SleepStage.REM, n_replays, n_timesteps)
