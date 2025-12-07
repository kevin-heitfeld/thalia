"""
Striatum - Reinforcement Learning with Three-Factor Rule

The striatum (part of basal ganglia) learns through dopamine-modulated
plasticity, implementing the classic three-factor learning rule for
reinforcement learning.

Key Features:
=============
1. THREE-FACTOR LEARNING: Δw = eligibility × dopamine
   - Pre-post activity creates eligibility traces
   - Eligibility alone does NOT cause plasticity
   - Dopamine arriving later converts eligibility to weight change
   - DA burst → LTP, DA dip → LTD, No DA → no learning

2. DOPAMINE as REWARD PREDICTION ERROR:
   - Burst: "Better than expected" → reinforce recent actions
   - Dip: "Worse than expected" → weaken recent actions
   - Baseline: "As expected" → maintain current policy

3. LONG ELIGIBILITY TRACES:
   - Biological tau: 500-2000ms (Yagishita et al., 2014)
   - Allows credit assignment for delayed rewards
   - Synaptic tag persists until dopamine arrives

4. ACTION SELECTION:
   - Winner-take-all competition via lateral inhibition
   - Selected action's synapses become eligible
   - Dopamine retroactively credits/blames the action

Biological Basis:
=================
- Medium Spiny Neurons (MSNs) in striatum
- D1-MSNs (direct pathway): DA → LTP → "Go" signal
- D2-MSNs (indirect pathway): DA → LTD → "No-Go" signal
- Schultz et al. (1997): Dopamine as reward prediction error

When to Use:
============
- Reinforcement learning (reward/punishment, not labels)
- Action selection and habit learning
- Delayed reward credit assignment
- When you want to learn from trial and error
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import replace
from typing import Optional, Dict, Any, List, Generator

import torch

from thalia.core.utils import ensure_batch_dim
from thalia.core.diagnostics_mixin import DiagnosticsMixin
from thalia.regions.base import (
    BrainRegion,
    RegionConfig,
    LearningRule,
)
from thalia.core.neuron import ConductanceLIF, ConductanceLIFConfig
from thalia.learning.unified_homeostasis import (
    StriatumHomeostasis,
    UnifiedHomeostasisConfig,
)

from .config import StriatumConfig
from .eligibility import EligibilityTraces
from .action_selection import ActionSelectionMixin


class Striatum(DiagnosticsMixin, ActionSelectionMixin, BrainRegion):
    """Striatal region with three-factor reinforcement learning.

    Implements dopamine-modulated learning:
    - Eligibility traces tag recently active synapses
    - Dopamine signal converts eligibility to plasticity
    - No learning without dopamine (unlike Hebbian)

    Population Coding (optional):
    - Instead of 1 neuron per action, use N neurons per action
    - Decision = which population has highest total spike count
    - Benefits: noise reduction, redundancy, graded confidence
    """
    def __init__(self, config: RegionConfig):
        if not isinstance(config, StriatumConfig):
            config = StriatumConfig(
                n_input=config.n_input,
                n_output=config.n_output,
                neuron_type=config.neuron_type,
                learning_rate=config.learning_rate,
                w_max=config.w_max,
                w_min=config.w_min,
                target_firing_rate_hz=config.target_firing_rate_hz,
                dt_ms=config.dt_ms,
                device=config.device,
            )

        self.striatum_config: StriatumConfig = config  # type: ignore

        # =====================================================================
        # POPULATION CODING SETUP
        # =====================================================================
        # If population_coding is enabled, we create N neurons per action
        # n_output in config = number of ACTIONS
        # actual neurons = n_actions * neurons_per_action
        self.n_actions = config.n_output
        if self.striatum_config.population_coding:
            self.neurons_per_action = self.striatum_config.neurons_per_action
            # Override n_output to be the total number of neurons
            actual_n_output = self.n_actions * self.neurons_per_action
            # Create modified config with expanded output
            config = replace(config, n_output=actual_n_output)
            self.striatum_config = config
        else:
            self.neurons_per_action = 1

        super().__init__(config)

        # Eligibility traces (synapse-specific)
        self.eligibility = EligibilityTraces(
            n_pre=config.n_input,
            n_post=config.n_output,
            tau_ms=self.striatum_config.eligibility_tau_ms,
            device=config.device,
        )

        # NOTE: Dopamine is now managed centrally by Brain (acting as VTA).
        # The Brain computes RPE and broadcasts normalized dopamine via set_dopamine().
        # We no longer have a local DopamineSystem here.

        # Recent spikes for lateral inhibition
        self.recent_spikes = torch.zeros(config.n_output, device=self.device)

        # Track last action for credit assignment
        self.last_action: Optional[int] = None

        # STDP traces for REWARD_MODULATED_STDP (spike-based learning)
        self.input_trace = torch.zeros(config.n_input, device=self.device)
        self.output_trace = torch.zeros(config.n_output, device=self.device)
        self.stdp_eligibility = torch.zeros(
            config.n_output, config.n_input, device=self.device
        )

        # Exploration state (uncertainty-driven, no epsilon-greedy)
        self.exploring = False  # Track if current action was exploratory
        self._last_uncertainty = 0.0
        self._last_exploration_prob = 0.0

        # =====================================================================
        # D1/D2 TRIAL ACCUMULATORS (for robust action selection)
        # =====================================================================
        # Accumulate D1 and D2 votes across all timesteps of a trial.
        # Final action selection uses accumulated NET = D1_total - D2_total.
        # This is more robust than per-timestep decisions because:
        # - Sparse spiking input means many timesteps have no activity
        # - Integrating over time builds reliable evidence
        self._d1_votes_accumulated = torch.zeros(self.n_actions, device=self.device)
        self._d2_votes_accumulated = torch.zeros(self.n_actions, device=self.device)

        # =====================================================================
        # UCB EXPLORATION TRACKING (action counts)
        # =====================================================================
        # Track how many times each action has been chosen for UCB bonus.
        # UCB bonus = c * sqrt(log(total_trials) / N_action)
        # This guarantees exploration of rarely-chosen actions.
        self._action_counts = torch.zeros(self.n_actions, device=self.device)
        self._total_trials = 0

        # =====================================================================
        # ADAPTIVE EXPLORATION TRACKING (performance history)
        # =====================================================================
        # Track recent reward history to adapt exploration based on performance.
        # Poor performance → increase exploration (boost tonic DA)
        # Good performance → decrease exploration (reduce tonic DA)
        self._recent_rewards: List[float] = []
        self._recent_accuracy: float = 0.5  # Running estimate of accuracy

        # =====================================================================
        # D1/D2 OPPONENT PROCESS (Direct/Indirect Pathways) - ALWAYS ENABLED
        # =====================================================================
        # We have TWO sets of weights and eligibility per action:
        # - D1 (direct pathway): Facilitates action selection (GO signal)
        # - D2 (indirect pathway): Suppresses action selection (NOGO signal)
        #
        # Action selection is based on D1 - D2 balance.
        # Learning rules are OPPOSITE for D1 vs D2:
        # - D1: DA+ → LTP, DA- → LTD (standard)
        # - D2: DA+ → LTD, DA- → LTP (inverted!)
        #
        # This naturally solves credit assignment:
        # - When wrong action is punished (DA-), its D2 pathway strengthens
        # - This builds up NOGO signal, making the action less likely
        # - Meanwhile, correct action's D1 weights strengthen when rewarded

        # D1 pathway weights (direct/GO) - same shape as main weights
        self.d1_weights = self._initialize_pathway_weights()
        # D2 pathway weights (indirect/NOGO) - same shape as main weights
        self.d2_weights = self._initialize_pathway_weights()

        # Separate eligibility traces for D1 and D2
        self.d1_eligibility = torch.zeros(
            config.n_output, config.n_input, device=self.device
        )
        self.d2_eligibility = torch.zeros(
            config.n_output, config.n_input, device=self.device
        )

        # D1/D2 spike traces for STDP
        self.d1_input_trace = torch.zeros(config.n_input, device=self.device)
        self.d2_input_trace = torch.zeros(config.n_input, device=self.device)
        self.d1_output_trace = torch.zeros(config.n_output, device=self.device)
        self.d2_output_trace = torch.zeros(config.n_output, device=self.device)

        # =====================================================================
        # HOMEOSTATIC PLASTICITY STATE
        # =====================================================================
        # Track activity for synaptic scaling (maintains stable firing rates)
        # This is critical for preventing D2 runaway inhibition!
        self._activity_ema = 0.5  # Exponential moving average of activity rate
        self._trial_spike_count = 0.0  # Spikes accumulated this trial
        self._trial_timesteps = 0  # Timesteps in current trial
        self._homeostatic_scaling_applied = False  # Track if scaling happened

        # =====================================================================
        # TONIC DOPAMINE STATE
        # =====================================================================
        # Baseline dopamine level that modulates D1 gain and exploration.
        # Updated slowly based on overall reward history (motivational state).
        self.tonic_dopamine = self.striatum_config.tonic_dopamine

        # =====================================================================
        # UNIFIED HOMEOSTASIS (Constraint-Based Stability)
        # =====================================================================
        # Instead of many overlapping correction mechanisms (BCM, synaptic scaling,
        # intrinsic plasticity), we use a single constraint-based approach:
        #
        # Key insight: CONSTRAINTS > CORRECTIONS
        # - Correction: "If D2 gets too small, boost learning" (might not work)
        # - Constraint: "D1 + D2 MUST sum to budget" (mathematically guaranteed)
        #
        # After each reward delivery, we normalize D1/D2 to share a fixed budget.
        # This GUARANTEES:
        # - Neither pathway can completely dominate
        # - If D1 grows, D2 must shrink (and vice versa)
        # - Stable equilibrium is enforced mathematically
        #
        # DYNAMIC BUDGET: Computed from actual initialized weights to adapt to
        # any architecture (population_coding on/off, different n_input, etc.)
        if self.striatum_config.homeostatic_enabled:
            # Compute budget from initialized weights (per-action sum of D1+D2)
            # This ensures the budget matches the actual weight scale
            with torch.no_grad():
                d1_d2_sum = self.d1_weights.sum() + self.d2_weights.sum()
                dynamic_budget = (d1_d2_sum / self.n_actions).item()
            
            unified_config = UnifiedHomeostasisConfig(
                weight_budget=dynamic_budget,
                soft_normalization=self.striatum_config.homeostatic_soft,
                normalization_rate=self.striatum_config.homeostatic_rate,
                w_min=self.config.w_min,
                w_max=self.config.w_max,
            )
            self.unified_homeostasis = StriatumHomeostasis(
                n_actions=self.n_actions,
                neurons_per_action=self.neurons_per_action,
                config=unified_config,
            )
        else:
            self.unified_homeostasis = None

        # =====================================================================
        # D1/D2 SEPARATE NEURON POPULATIONS (Biological Architecture)
        # =====================================================================
        # In biology, D1-MSNs and D2-MSNs are SEPARATE neurons with different
        # dopamine receptor types. This is crucial because:
        # - D1-MSNs express D1 receptors: DA+ → LTP
        # - D2-MSNs express D2 receptors: DA+ → LTD (inverted!)
        #
        # Previous implementation was WRONG: D1/D2 were weight matrices on the
        # SAME neurons. This meant D1/D2 were fighting over the same neural activity.
        #
        # Correct implementation:
        # - D1-MSNs: Separate population, excited by d1_weights
        # - D2-MSNs: Separate population, excited by d2_weights
        # - Action selection: argmax(D1_activity - D2_activity) per action
        self.d1_neurons = self._create_d1_neurons()
        self.d2_neurons = self._create_d2_neurons()

        # =====================================================================
        # REWARD PREDICTION ERROR (RPE) - Value Estimates
        # =====================================================================
        # Track expected value per action for computing prediction error.
        # DA = actual_reward - expected_reward
        # This prevents runaway winners (high expectation → small surprise)
        if self.striatum_config.rpe_enabled:
            self.value_estimates = torch.full(
                (self.n_actions,),
                self.striatum_config.rpe_initial_value,
                device=self.device
            )
        else:
            self.value_estimates = None

        # Plasticity freeze flag (for debugging/evaluation escape hatch)
        self._plasticity_frozen = False

    # =========================================================================
    # PLASTICITY CONTROL (for debugging only)
    # =========================================================================

    @contextmanager
    def freeze_plasticity(self) -> Generator[None, None, None]:
        """Context manager to temporarily freeze learning.

        WARNING: This is an escape hatch for debugging only!
        The brain ALWAYS learns in biology — use sparingly.

        Usage:
            with striatum.freeze_plasticity():
                # No learning during this block
                action = striatum.forward(input_spikes)
                striatum.deliver_reward(reward)  # No weight changes

        Yields:
            None
        """
        old_value = self._plasticity_frozen
        self._plasticity_frozen = True
        try:
            yield
        finally:
            self._plasticity_frozen = old_value

    # =========================================================================
    # VALUE ESTIMATION (for centralized RPE computation in Brain/VTA)
    # =========================================================================

    def get_expected_value(self, action: Optional[int] = None) -> float:
        """Get expected value for an action (used by Brain for RPE computation).

        The striatum maintains value estimates for each action as "Q-values".
        The Brain (acting as VTA) queries these to compute reward prediction error:
            RPE = actual_reward - expected_value

        Args:
            action: Action index to get value for. If None, uses last_action.

        Returns:
            Expected value for the action. Returns 0.0 if:
            - rpe_enabled is False (no value tracking)
            - action is None and no last_action recorded
            - action index is out of range
        """
        if self.value_estimates is None:
            return 0.0

        if action is None:
            action = self.last_action

        if action is None or action < 0 or action >= self.n_actions:
            return 0.0

        return float(self.value_estimates[action].item())

    def update_value_estimate(self, action: int, reward: float) -> None:
        """Update value estimate for an action towards actual reward.

        Called by Brain after computing RPE, to update the Q-value:
            V(a) ← V(a) + α * (reward - V(a))

        Args:
            action: Action index to update
            reward: Actual reward received
        """
        if self.value_estimates is None:
            return

        if action < 0 or action >= self.n_actions:
            return

        cfg = self.striatum_config
        self.value_estimates[action] = (
            self.value_estimates[action]
            + cfg.rpe_learning_rate * (reward - self.value_estimates[action])
        )

    def _initialize_pathway_weights(self) -> torch.Tensor:
        """Initialize weights for D1 or D2 pathway with balanced, principled scaling.

        Uses fan-in scaling to ensure consistent input magnitude regardless of
        input size. Both D1 and D2 start with identical distributions - the
        competition between GO and NOGO pathways emerges purely from learning.

        Principles:
        1. Fan-in scaling: Normalizes for different input sizes (like Xavier init)
        2. Equal D1/D2: No baked-in pathway preference
        3. Minimal variance: Near-symmetric start prevents early bias lock-in
        4. Moderate values: Keeps neurons in operational firing regime

        This avoids encoding task-specific knowledge into initialization.
        """
        # Fan-in scaling for consistent input magnitude
        fan_in_scale = 1.0 / math.sqrt(self.config.n_input)

        # Base weight with MINIMAL random variance for near-symmetric start
        # Reduced variance from 0.05 to 0.01 to prevent early bias lock-in
        # With few training trials, high variance can cause one action to
        # get lucky/unlucky and create permanent bias
        base = 0.15
        variance = 0.01  # Reduced from 0.05 for more symmetric initialization
        weights = base + variance * torch.randn(self.config.n_output, self.config.n_input)

        # Scale by fan-in and w_max
        weights = weights * fan_in_scale * self.config.w_max

        return weights.clamp(self.config.w_min, self.config.w_max).to(self.device)

    def _update_stdp_eligibility(self, input_spikes: torch.Tensor, output_spikes: torch.Tensor) -> None:
        """Update STDP eligibility traces from spike timing.

        This is called from forward() so eligibility accumulates during the
        trial before reward is delivered.

        Args:
            input_spikes: Input spike tensor
            output_spikes: Output spike tensor
        """
        input_spikes = ensure_batch_dim(input_spikes)
        output_spikes = ensure_batch_dim(output_spikes)

        dt = self.config.dt_ms
        cfg = self.striatum_config

        # Decay and update spike traces
        trace_decay = 1.0 - dt / cfg.stdp_tau_ms
        self.input_trace = self.input_trace * trace_decay + input_spikes.squeeze()
        self.output_trace = self.output_trace * trace_decay + output_spikes.squeeze()

        # STDP rule:
        # - LTP: post spike with pre trace → strengthen
        # - LTD: pre spike with post trace → weaken
        ltp = torch.outer(output_spikes.squeeze(), self.input_trace)
        ltd = torch.outer(self.output_trace, input_spikes.squeeze())

        # Soft bounds: reduce learning as weights approach limits
        w_normalized = (self.weights - self.config.w_min) / (self.config.w_max - self.config.w_min)
        ltp_factor = 1.0 - w_normalized
        ltd_factor = w_normalized

        soft_ltp = ltp * ltp_factor
        soft_ltd = ltd * ltd_factor

        # Competitive anti-Hebbian: non-spiking neurons get weaker to active inputs
        non_spiking = 1.0 - output_spikes.squeeze()
        anti_hebbian = torch.outer(non_spiking, input_spikes.squeeze()) * w_normalized

        # Compute STDP weight change (direction from spike timing)
        stdp_dw = cfg.stdp_lr * (soft_ltp - cfg.heterosynaptic_ratio * soft_ltd - 0.1 * anti_hebbian)

        # Accumulate into eligibility trace (long timescale: 500-2000ms)
        eligibility_decay = 1.0 - dt / cfg.eligibility_tau_ms
        self.stdp_eligibility = self.stdp_eligibility * eligibility_decay + stdp_dw

    def _update_d1_d2_eligibility(
        self, input_spikes: torch.Tensor, d1_spikes: torch.Tensor, d2_spikes: torch.Tensor,
        chosen_action: int | None = None
    ) -> None:
        """Update separate eligibility traces for D1 and D2 pathways.

        With SEPARATE neuron populations:
        - D1 eligibility is computed from input-D1 spike coincidence
        - D2 eligibility is computed from input-D2 spike coincidence

        CRITICAL: When chosen_action is provided, eligibility is ONLY built for
        the neurons corresponding to that action. This is biologically correct:
        only synapses where the post-synaptic neuron fired should become eligible.
        This prevents both actions from building eligibility from the same input.

        Args:
            input_spikes: Input spike tensor
            d1_spikes: D1 neuron population spikes
            d2_spikes: D2 neuron population spikes
            chosen_action: If provided, only build eligibility for this action's neurons
        """
        input_spikes = ensure_batch_dim(input_spikes)
        d1_spikes = ensure_batch_dim(d1_spikes)
        d2_spikes = ensure_batch_dim(d2_spikes)

        # STDP traces are 1D - designed for batch_size=1 temporal processing
        # Skip eligibility updates for batched inputs
        if input_spikes.shape[0] != 1:
            return

        dt = self.config.dt_ms
        cfg = self.striatum_config

        # Get 1D versions for trace updates
        input_1d = input_spikes.squeeze(0)
        d1_output_1d = d1_spikes.squeeze(0)
        d2_output_1d = d2_spikes.squeeze(0)

        # CRITICAL: Mask output spikes to ONLY include the chosen action's neurons
        # This is biologically correct: only synapses where post-synaptic neurons
        # actually fired should become eligible. Without this, both actions build
        # eligibility from the same input, causing learning instability.
        if chosen_action is not None:
            action_mask = torch.zeros_like(d1_output_1d)
            if self.striatum_config.population_coding:
                pop_slice = self._get_action_population_indices(chosen_action)
                action_mask[pop_slice] = 1.0
            else:
                action_mask[chosen_action] = 1.0
            d1_output_1d = d1_output_1d * action_mask
            d2_output_1d = d2_output_1d * action_mask

        # Decay and update spike traces - SEPARATE for D1 and D2
        trace_decay = 1.0 - dt / cfg.stdp_tau_ms
        self.d1_input_trace = self.d1_input_trace * trace_decay + input_1d
        self.d1_output_trace = self.d1_output_trace * trace_decay + d1_output_1d
        self.d2_input_trace = self.d2_input_trace * trace_decay + input_1d
        self.d2_output_trace = self.d2_output_trace * trace_decay + d2_output_1d

        # STDP rule for D1 pathway (using D1 neuron spikes)
        d1_ltp = torch.outer(d1_output_1d, self.d1_input_trace)
        d1_ltd = torch.outer(self.d1_output_trace, input_1d)

        # Soft bounds for D1
        d1_w_normalized = (self.d1_weights - self.config.w_min) / (self.config.w_max - self.config.w_min)
        d1_ltp_factor = 1.0 - d1_w_normalized
        d1_ltd_factor = d1_w_normalized

        d1_soft_ltp = d1_ltp * d1_ltp_factor
        d1_soft_ltd = d1_ltd * d1_ltd_factor

        # D1 eligibility (standard STDP direction)
        d1_stdp_dw = cfg.stdp_lr * cfg.d1_lr_scale * (
            d1_soft_ltp - cfg.heterosynaptic_ratio * d1_soft_ltd
        )

        # STDP rule for D2 pathway (using D2 neuron spikes)
        d2_ltp = torch.outer(d2_output_1d, self.d2_input_trace)
        d2_ltd = torch.outer(self.d2_output_trace, input_1d)

        # Soft bounds for D2
        d2_w_normalized = (self.d2_weights - self.config.w_min) / (self.config.w_max - self.config.w_min)
        d2_ltp_factor = 1.0 - d2_w_normalized
        d2_ltd_factor = d2_w_normalized

        d2_soft_ltp = d2_ltp * d2_ltp_factor
        d2_soft_ltd = d2_ltd * d2_ltd_factor

        # D2 eligibility (standard STDP direction - inversion happens at reward delivery)
        d2_stdp_dw = cfg.stdp_lr * cfg.d2_lr_scale * (
            d2_soft_ltp - cfg.heterosynaptic_ratio * d2_soft_ltd
        )

        # Accumulate into eligibility traces
        eligibility_decay = 1.0 - dt / cfg.eligibility_tau_ms
        self.d1_eligibility = self.d1_eligibility * eligibility_decay + d1_stdp_dw
        self.d2_eligibility = self.d2_eligibility * eligibility_decay + d2_stdp_dw

    def _update_d1_d2_eligibility_all(
        self, input_spikes: torch.Tensor, d1_spikes: torch.Tensor, d2_spikes: torch.Tensor
    ) -> None:
        """Update eligibility traces for ALL active D1/D2 neurons.

        Unlike _update_d1_d2_eligibility which masks to a chosen action,
        this version accumulates eligibility for all neurons that fired.
        The action-specific masking is deferred to deliver_reward(), which
        uses last_action (set by finalize_action) to apply learning only
        to the chosen action's synapses.

        This is the correct approach when action selection happens at trial
        end (via finalize_action) rather than per-timestep.

        Args:
            input_spikes: Input spike tensor
            d1_spikes: D1 neuron population spikes
            d2_spikes: D2 neuron population spikes
        """
        # Just call the existing method with chosen_action=None
        # which will update eligibility for all active neurons
        self._update_d1_d2_eligibility(input_spikes, d1_spikes, d2_spikes, chosen_action=None)

    def _get_learning_rule(self) -> LearningRule:
        return LearningRule.THREE_FACTOR

    def _initialize_weights(self) -> torch.Tensor:
        """Initialize with small positive weights."""
        weights = torch.rand(self.config.n_output, self.config.n_input)
        weights = weights * self.config.w_max * 0.2
        return weights.clamp(self.config.w_min, self.config.w_max).to(self.device)

    def _create_neurons(self) -> ConductanceLIF:
        """Create MSN-like neurons (legacy - kept for parent class compatibility).

        NOTE: This is now only used for parent class compatibility.
        The actual D1/D2 neurons are created by _create_d1_neurons() and
        _create_d2_neurons() which are called in __init__.
        """
        neuron_config = ConductanceLIFConfig(
            v_threshold=1.0, v_reset=0.0, E_L=0.0, E_E=3.0, E_I=-0.5,
            tau_E=5.0, tau_I=5.0,
            dt=1.0,
            tau_ref=2.0,
        )
        neurons = ConductanceLIF(n_neurons=self.config.n_output, config=neuron_config)
        neurons.to(self.device)
        return neurons

    def _create_d1_neurons(self) -> ConductanceLIF:
        """Create D1-MSN population (direct pathway / GO).

        D1-MSNs express D1 dopamine receptors:
        - DA burst (positive RPE) → LTP → stronger GO signal
        - DA dip (negative RPE) → LTD → weaker GO signal

        These neurons receive excitation from d1_weights and their activity
        promotes action selection (GO signal).

        Uses spike-frequency adaptation to prevent runaway excitation and
        provide more stable, differentiated action selection signals.
        """
        neuron_config = ConductanceLIFConfig(
            v_threshold=1.0, v_reset=0.0, E_L=0.0, E_E=3.0, E_I=-0.5,
            tau_E=5.0, tau_I=5.0,
            dt=1.0,
            tau_ref=2.0,
            tau_adapt=100.0,        # Adaptation time constant
            adapt_increment=0.1,    # Enable spike-frequency adaptation
        )
        neurons = ConductanceLIF(n_neurons=self.config.n_output, config=neuron_config)
        neurons.to(self.device)
        return neurons

    def _create_d2_neurons(self) -> ConductanceLIF:
        """Create D2-MSN population (indirect pathway / NOGO).

        D2-MSNs express D2 dopamine receptors with INVERTED dopamine response:
        - DA burst (positive RPE) → LTD → weaker NOGO signal
        - DA dip (negative RPE) → LTP → stronger NOGO signal

        This inversion is the key biological insight:
        - When wrong action is punished → D2 strengthens → inhibits action next time
        - When correct action is rewarded → D2 weakens → allows action

        These neurons receive excitation from d2_weights and their activity
        opposes action selection (NOGO signal).

        Uses spike-frequency adaptation to prevent runaway excitation and
        provide more stable, differentiated action selection signals.
        """
        neuron_config = ConductanceLIFConfig(
            v_threshold=1.0, v_reset=0.0, E_L=0.0, E_E=3.0, E_I=-0.5,
            tau_E=5.0, tau_I=5.0,
            dt=1.0,
            tau_ref=2.0,
            tau_adapt=100.0,        # Adaptation time constant
            adapt_increment=0.1,    # Enable spike-frequency adaptation
        )
        neurons = ConductanceLIF(n_neurons=self.config.n_output, config=neuron_config)
        neurons.to(self.device)
        return neurons

    def forward(
        self,
        input_spikes: torch.Tensor,
        dt: float = 1.0,
        encoding_mod: float = 1.0,
        retrieval_mod: float = 1.0,
        explore: bool = True,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Process input and select action using SEPARATE D1/D2 populations.

        BIOLOGICAL ARCHITECTURE:
        - D1-MSNs: SEPARATE neuron population, receives d1_weights excitation
        - D2-MSNs: SEPARATE neuron population, receives d2_weights excitation
        - Action selection: argmax(D1_activity - D2_activity) per action

        This is fundamentally different from the previous (broken) design where
        D1/D2 weights fed the SAME neurons with different conductance types.

        Args:
            input_spikes: Input spike tensor
            dt: Time step in ms
            encoding_mod: Theta modulation for encoding phase (0-1).
            retrieval_mod: Theta modulation for retrieval phase (0-1).
            explore: If True, use uncertainty-driven exploration.

        With population coding:
        - Each action has N neurons per pathway (neurons_per_action)
        - D1_votes = sum(D1 spikes for action)
        - D2_votes = sum(D2 spikes for action)
        - NET = D1_votes - D2_votes
        - Selected action = argmax(NET)
        """
        input_spikes = ensure_batch_dim(input_spikes)

        # Reset D1 and D2 neuron states if needed
        if self.d1_neurons.membrane is None:
            self.d1_neurons.reset_state(input_spikes.shape[0])
        if self.d2_neurons.membrane is None:
            self.d2_neurons.reset_state(input_spikes.shape[0])

        # =====================================================================
        # D1/D2 SEPARATE POPULATIONS - COMPUTE ACTIVATIONS
        # =====================================================================
        # D1 and D2 weights project to SEPARATE neuron populations
        # Each population receives its weights as EXCITATORY input
        d1_activation = torch.matmul(input_spikes, self.d1_weights.T)
        d2_activation = torch.matmul(input_spikes, self.d2_weights.T)

        # =====================================================================
        # THETA MODULATION
        # =====================================================================
        theta_baseline_mod = 0.7 + 0.3 * encoding_mod  # 0.7-1.0 range
        theta_contrast_mod = 0.8 + 0.2 * retrieval_mod  # 0.8-1.0 range

        # Baseline excitation modulated by theta phase
        baseline_exc = 1.2 * theta_baseline_mod  # 0.84-1.2 range

        # =====================================================================
        # TONIC DOPAMINE MODULATION OF D1 GAIN
        # =====================================================================
        # Tonic DA increases D1 pathway responsiveness (motivation, energy)
        # This is separate from phasic DA which drives learning
        d1_gain = 1.0
        d2_gain = 1.0
        if self.striatum_config.tonic_modulates_d1_gain:
            # Higher tonic DA → stronger D1 response
            tonic_factor = self.tonic_dopamine * self.striatum_config.tonic_d1_gain_scale
            d1_gain = 1.0 + tonic_factor  # e.g., tonic=0.3, scale=0.5 → gain=1.15

        # =====================================================================
        # ACTIVITY-BASED EXCITABILITY MODULATION
        # =====================================================================
        # Intrinsic excitability based on recent activity history.
        # Low activity neurons get higher gain, high activity get lower gain.
        # This is a constraint-based approach (via UnifiedHomeostasis) rather
        # than a correction-based approach (former IntrinsicPlasticity).
        if self.unified_homeostasis is not None:
            d1_exc_gain, d2_exc_gain = self.unified_homeostasis.compute_excitability()
            d1_gain = d1_gain * d1_exc_gain
            d2_gain = d2_gain * d2_exc_gain

        # =====================================================================
        # D1 NEURON POPULATION (Direct Pathway / GO)
        # =====================================================================
        # D1 neurons receive d1_weights as excitatory input
        # They don't receive D2 as inhibition - the populations are SEPARATE
        # Apply tonic DA gain modulation to D1 pathway
        d1_g_exc = (d1_activation * theta_contrast_mod * d1_gain + baseline_exc).clamp(min=0)
        d1_g_inh = torch.zeros_like(d1_g_exc)  # No direct inhibition

        # Add lateral inhibition within D1 population if enabled
        if self.striatum_config.lateral_inhibition:
            d1_g_inh = d1_g_inh + self.recent_spikes.unsqueeze(0) * self.striatum_config.inhibition_strength * 0.5

        d1_spikes, _ = self.d1_neurons(d1_g_exc, d1_g_inh)

        # =====================================================================
        # D2 NEURON POPULATION (Indirect Pathway / NOGO)
        # =====================================================================
        # D2 neurons receive d2_weights as excitatory input
        # Their activity represents the NOGO signal for each action
        # Apply d2_gain from activity-based excitability modulation
        d2_g_exc = (d2_activation * theta_contrast_mod * d2_gain + baseline_exc).clamp(min=0)
        d2_g_inh = torch.zeros_like(d2_g_exc)  # No direct inhibition

        # Add lateral inhibition within D2 population if enabled
        if self.striatum_config.lateral_inhibition:
            d2_g_inh = d2_g_inh + self.recent_spikes.unsqueeze(0) * self.striatum_config.inhibition_strength * 0.5

        d2_spikes, _ = self.d2_neurons(d2_g_exc, d2_g_inh)

        # DEBUG: Check activation magnitudes (only on first call or after reset)
        if hasattr(self, '_debug_trial_count'):
            if self._debug_trial_count <= 1 or self._debug_trial_count % 50 == 0:
                # Get input sparsity
                n_active = input_spikes.sum().item()
                d1_act_mean = d1_activation.mean().item()
                d1_exc_mean = d1_g_exc.mean().item()
                d1_spk_sum = d1_spikes.sum().item()
                d2_spk_sum = d2_spikes.sum().item()
                # Check membrane potential
                d1_v_max = self.d1_neurons.membrane.max().item() if self.d1_neurons.membrane is not None else 0
                print(f"    [ACT t={self._debug_trial_count}] input_active={n_active:.0f}, "
                      f"d1_act_mean={d1_act_mean:.3f}, d1_g_exc={d1_exc_mean:.3f}, "
                      f"d1_v_max={d1_v_max:.3f}, d1_spks={d1_spk_sum:.0f}, d2_spks={d2_spk_sum:.0f}")

        # =====================================================================
        # UPDATE ACTIVITY HISTORY FOR EXCITABILITY MODULATION
        # =====================================================================
        # Track D1/D2 firing rates for homeostatic excitability adjustment.
        # This feeds into compute_excitability() for next timestep.
        if self.unified_homeostasis is not None:
            self.unified_homeostasis.update_activity(d1_spikes, d2_spikes)

        # =====================================================================
        # ACTION SELECTION: D1 - D2 (GO - NOGO)
        # =====================================================================
        # For each action, compute NET = D1_activity - D2_activity
        # Select action with highest NET value (or sample from softmax)
        # This is the key biological insight: D1 and D2 populations COMPETE
        d1_votes = self._count_population_votes(d1_spikes)
        d2_votes = self._count_population_votes(d2_spikes)
        net_votes = d1_votes - d2_votes

        # ACCUMULATE D1/D2 votes across timesteps for trial-level decision
        # This integrates sparse spiking evidence over time
        self._d1_votes_accumulated += d1_votes
        self._d2_votes_accumulated += d2_votes

        # =====================================================================
        # OUTPUT SPIKES: Return D1 activity (action selection happens in finalize_action)
        # =====================================================================
        # During the trial, we output D1 spikes for all neurons.
        # The final action selection (UCB + softmax + exploration) is handled by
        # finalize_action() at the end of the trial, not per-timestep.
        output_spikes = d1_spikes.clone()

        # =====================================================================
        # UPDATE ELIGIBILITY TRACES (for all active neurons)
        # =====================================================================
        # Eligibility accumulates for ALL neurons that fire during the trial.
        # When reward arrives, deliver_reward() uses last_action (set by finalize_action)
        # to apply learning only to the chosen action's synapses.
        self.eligibility.update(input_spikes, output_spikes, self.config.dt_ms)

        # Update D1/D2 eligibility traces for ALL active neurons
        # The action-specific masking happens in deliver_reward(), not here
        self._update_d1_d2_eligibility_all(input_spikes, d1_spikes, d2_spikes)

        # For REWARD_MODULATED_STDP, also update the spike-based eligibility trace
        if self.striatum_config.learning_rule == LearningRule.REWARD_MODULATED_STDP:
            self._update_stdp_eligibility(input_spikes, output_spikes)

        # NOTE: Dopamine is now managed centrally by Brain (VTA).
        # No local dopamine decay needed - the Brain sets dopamine via set_dopamine()
        # and it persists until the next reward delivery.

        # Update recent spikes (based on D1 output)
        self.recent_spikes = self.recent_spikes * 0.9 + d1_spikes.squeeze()

        # Track activity for homeostatic scaling
        self._trial_spike_count += d1_spikes.sum().item() + d2_spikes.sum().item()
        self._trial_timesteps += 1

        # Store D1 and D2 spikes for later use (debugging, eligibility)
        self._last_d1_spikes = d1_spikes
        self._last_d2_spikes = d2_spikes

        self.state.spikes = output_spikes
        # self.state.dopamine is set by Brain via set_dopamine(), no need to update here
        self.state.t += 1

        return output_spikes

    def debug_state(self, label: str = "") -> Dict[str, float]:
        """Print and return current D1/D2 weight state per action.

        Centralized debug method for understanding striatum state.
        Call this at key moments from experiment scripts.

        Args:
            label: Optional label to identify the debug point

        Returns:
            Dict with weight statistics
        """
        n_per = self.neurons_per_action

        # Per-action weights
        d1_match = self.d1_weights[0:n_per].mean().item()
        d1_nomatch = self.d1_weights[n_per:2*n_per].mean().item()
        d2_match = self.d2_weights[0:n_per].mean().item()
        d2_nomatch = self.d2_weights[n_per:2*n_per].mean().item()

        # Net (D1-D2) per action - this determines action selection
        net_match = d1_match - d2_match
        net_nomatch = d1_nomatch - d2_nomatch

        # Eligibility traces
        d1_elig_match = self.d1_eligibility[0:n_per].abs().mean().item()
        d1_elig_nomatch = self.d1_eligibility[n_per:2*n_per].abs().mean().item()

        # Value estimates (for RPE)
        val_match = 0.0
        val_nomatch = 0.0
        if self.value_estimates is not None:
            val_match = self.value_estimates[0].item()
            val_nomatch = self.value_estimates[1].item() if len(self.value_estimates) > 1 else 0.0

        prefix = f"[{label}] " if label else ""
        print(f"{prefix}D1: M={d1_match:.4f}, NM={d1_nomatch:.4f} | "
              f"D2: M={d2_match:.4f}, NM={d2_nomatch:.4f} | "
              f"NET: M={net_match:.4f}, NM={net_nomatch:.4f} | "
              f"VAL: M={val_match:.3f}, NM={val_nomatch:.3f}")

        return {
            "d1_match": d1_match, "d1_nomatch": d1_nomatch,
            "d2_match": d2_match, "d2_nomatch": d2_nomatch,
            "net_match": net_match, "net_nomatch": net_nomatch,
            "elig_match": d1_elig_match, "elig_nomatch": d1_elig_nomatch,
            "val_match": val_match, "val_nomatch": val_nomatch,
        }

    def apply_homeostatic_scaling(self) -> Dict[str, Any]:
        """Apply homeostatic regulation.

        This method is now largely obsolete because unified homeostasis
        is applied during reward delivery via normalize_d1_d2().

        Returns:
            Dict with scaling statistics for monitoring.
        """
        cfg = self.striatum_config

        if not cfg.homeostatic_enabled:
            return {"homeostatic_applied": False}

        # Calculate activity rate for this trial (still useful for diagnostics)
        max_possible = self._trial_timesteps * self.config.n_output
        if max_possible > 0:
            trial_activity = self._trial_spike_count / max_possible
        else:
            trial_activity = 0.0

        # Reset trial counters
        self._trial_spike_count = 0.0
        self._trial_timesteps = 0

        d1_mean = self.d1_weights.mean().item()
        d2_mean = self.d2_weights.mean().item()

        return {
            "homeostatic_applied": True,
            "trial_activity": trial_activity,
            "d1_mean": d1_mean,
            "d2_mean": d2_mean,
            "d1_d2_ratio": d1_mean / (d2_mean + 1e-6),
            "note": "unified homeostasis now applied during reward delivery",
        }

    def _apply_baseline_pressure(self) -> Dict[str, Any]:
        """Apply homeostatic pressure towards balanced D1/D2.

        Unlike budget normalization (which preserves D1:D2 ratios), baseline
        pressure actively drifts NET (D1-D2) towards a target value for each
        action. This prevents runaway biases where one action becomes dominant.

        Biological basis:
        - Synaptic scaling adjusts weights towards a setpoint
        - Homeostatic plasticity maintains balanced excitation/inhibition
        - Without active use, synapses drift towards baseline

        The mechanism:
        1. Calculate current NET (D1-D2) for each action
        2. Compute error from target (default: 0 = balanced)
        3. Adjust D1 down and D2 up (or vice versa) to reduce error

        Returns:
            Dict with diagnostic information about the adjustment.
        """
        cfg = self.striatum_config

        if not cfg.baseline_pressure_enabled:
            return {"baseline_pressure_applied": False}

        rate = cfg.baseline_pressure_rate
        target = cfg.baseline_target_net

        # Track adjustments for diagnostics
        net_before = []
        net_after = []

        for action in range(self.n_actions):
            start = action * self.neurons_per_action
            end = start + self.neurons_per_action

            # Current mean weights for this action
            d1_action = self.d1_weights[start:end]
            d2_action = self.d2_weights[start:end]

            d1_mean = d1_action.mean()
            d2_mean = d2_action.mean()
            current_net = d1_mean - d2_mean
            net_before.append(current_net.item())

            # Error from target
            error = current_net - target

            # Adjustment: reduce D1 and increase D2 (or vice versa)
            # to move NET towards target
            # Split the correction between both pathways
            adjustment = rate * error * 0.5

            # Apply adjustments (proportionally to current values to avoid negatives)
            self.d1_weights[start:end] = (d1_action - adjustment).clamp(
                self.config.w_min, self.config.w_max
            )
            self.d2_weights[start:end] = (d2_action + adjustment).clamp(
                self.config.w_min, self.config.w_max
            )

            # Record new NET
            new_net = self.d1_weights[start:end].mean() - self.d2_weights[start:end].mean()
            net_after.append(new_net.item())

        return {
            "baseline_pressure_applied": True,
            "net_before": net_before,
            "net_after": net_after,
            "pressure_rate": rate,
            "target_net": target,
        }

    # NOTE: The learn() method and _three_factor_learn/_reward_modulated_stdp_learn
    # have been removed. With the continuous learning paradigm:
    # - forward() builds eligibility traces during activity
    # - deliver_reward() applies D1/D2 plasticity when dopamine arrives
    # This is biologically correct: striatum uses three-factor learning where
    # plasticity is gated by dopamine, not continuous like cortical STDP.

    def deliver_reward(self, reward: float) -> Dict[str, Any]:
        """Deliver reward signal and trigger D1/D2 learning.

        The brain ALWAYS LEARNS — there is no "evaluation mode" in biology.
        Every experience shapes plasticity.

        IMPORTANT: The Brain (acting as VTA) has already computed the reward
        prediction error and set our dopamine level via set_dopamine(). This
        method just applies D1/D2 learning using that pre-computed dopamine.

        For D1/D2 opponent process, applies OPPOSITE learning rules:
        - D1 pathway: DA+ → LTP, DA- → LTD (standard)
        - D2 pathway: DA+ → LTD, DA- → LTP (inverted!)

        This naturally solves credit assignment:
        - When wrong action is punished (DA-):
          - D1 weakens (less GO signal for this action)
          - D2 strengthens (more NOGO signal = inhibit this action next time)
        - When correct action is rewarded (DA+):
          - D1 strengthens (more GO signal)
          - D2 weakens (less NOGO signal)

        Args:
            reward: Raw reward signal (for adaptive exploration tracking only)

        Returns:
            Metrics dict with dopamine level and weight changes.
        """
        cfg = self.striatum_config

        # =====================================================================
        # USE DOPAMINE FROM BRAIN (VTA)
        # =====================================================================
        # The Brain has already computed RPE and set our dopamine via set_dopamine().
        # We just use that value for learning.
        da_level = self.state.dopamine

        # Store for diagnostics (RPE now computed in Brain, but we track locally too)
        self._last_rpe = da_level  # Dopamine IS the normalized RPE
        self._last_expected = 0.0  # No longer tracked here

        # =====================================================================
        # ADAPTIVE EXPLORATION: Adjust tonic DA based on recent performance
        # =====================================================================
        # Poor performance → increase tonic DA → more exploration
        # Good performance → decrease tonic DA → more exploitation
        #
        # Biological basis: Locus coeruleus releases norepinephrine during
        # uncertainty/stress, and tonic DA levels influence exploration
        if cfg.adaptive_exploration:
            # Track this trial's outcome (reward > 0 = correct)
            was_correct = 1.0 if reward > 0 else 0.0
            self._recent_rewards.append(was_correct)

            # Keep only the most recent trials
            window = cfg.performance_window
            if len(self._recent_rewards) > window:
                self._recent_rewards = self._recent_rewards[-window:]

            # Update running accuracy estimate
            if len(self._recent_rewards) > 0:
                self._recent_accuracy = sum(self._recent_rewards) / len(self._recent_rewards)

            # Adjust tonic DA: lower accuracy → higher tonic DA → more exploration
            # Linear interpolation between min and max tonic DA
            # At 0% accuracy: use max_tonic_dopamine
            # At 100% accuracy: use min_tonic_dopamine
            old_tonic = self.tonic_dopamine
            accuracy = self._recent_accuracy
            min_tonic = cfg.min_tonic_dopamine
            max_tonic = cfg.max_tonic_dopamine

            # Smooth update: blend new estimate with old (momentum)
            target_tonic = max_tonic - accuracy * (max_tonic - min_tonic)
            momentum = 0.9  # Keep most of old value for stability
            self.tonic_dopamine = momentum * old_tonic + (1 - momentum) * target_tonic

            # Clamp to valid range
            self.tonic_dopamine = max(min_tonic, min(max_tonic, self.tonic_dopamine))

        # =====================================================================
        # D1/D2 OPPONENT PROCESS LEARNING (Always Enabled)
        # =====================================================================
        return self._deliver_reward_d1_d2(da_level)

    def _deliver_reward_d1_d2(self, da_level: float) -> Dict[str, Any]:
        """Apply D1/D2 opponent process learning.

        The brain ALWAYS LEARNS — even from exploratory actions.
        Every experience shapes plasticity.

        The key biological insight:
        - D1 (direct pathway, GO): Uses D1 dopamine receptors
          - DA burst (positive) → LTP (strengthen GO signal)
          - DA dip (negative) → LTD (weaken GO signal)

        - D2 (indirect pathway, NOGO): Uses D2 dopamine receptors
          - DA burst (positive) → LTD (weaken NOGO signal)
          - DA dip (negative) → LTP (strengthen NOGO signal)

        This OPPOSITE dopamine response in D2 is crucial:
        - When an action is punished (DA dip):
          - D1 weakens → less GO
          - D2 strengthens → more NOGO (learn to inhibit this action!)
        - When an action is rewarded (DA burst):
          - D1 strengthens → more GO
          - D2 weakens → less NOGO

        This naturally solves the credit assignment problem because even when
        we take the WRONG action and get punished, we're building up NOGO
        signal that will inhibit that action next time.

        Args:
            da_level: Dopamine level (already computed from reward)

        Returns:
            Metrics dict with dopamine level and weight changes.
        """
        cfg = self.striatum_config

        # Skip learning if plasticity is frozen (debugging escape hatch)
        if self._plasticity_frozen:
            return {
                "dopamine": da_level,
                "d1_ltp": 0.0,
                "d1_ltd": 0.0,
                "d2_ltp": 0.0,
                "d2_ltd": 0.0,
                "net_change": 0.0,
                "d1_eligibility_max": self.d1_eligibility.abs().max().item(),
                "d2_eligibility_max": self.d2_eligibility.abs().max().item(),
                "frozen": True,
            }

        if abs(da_level) < 0.01:
            return {
                "dopamine": da_level,
                "d1_ltp": 0.0,
                "d1_ltd": 0.0,
                "d2_ltp": 0.0,
                "d2_ltd": 0.0,
                "net_change": 0.0,
                "d1_eligibility_max": self.d1_eligibility.abs().max().item(),
                "d2_eligibility_max": self.d2_eligibility.abs().max().item(),
            }

        # Create action mask for selected action
        action_mask = torch.zeros(self.config.n_output, device=self.device)
        if self.last_action is not None:
            if self.striatum_config.population_coding:
                pop_slice = self._get_action_population_indices(self.last_action)
                action_mask[pop_slice] = 1.0
            else:
                action_mask[self.last_action] = 1.0
        else:
            # No action selected, apply to all
            action_mask = torch.ones(self.config.n_output, device=self.device)

        # =====================================================================
        # D1 PATHWAY LEARNING (standard dopamine response)
        # =====================================================================
        # DA+ → LTP (strengthen GO for rewarded actions)
        # DA- → LTD (weaken GO for punished actions)
        #
        # IMPORTANT: Use ABSOLUTE VALUE of eligibility trace!
        # The eligibility trace marks synapses that were active (magnitude).
        # Dopamine alone determines direction (strengthen vs weaken).
        # If we don't use abs(), negative eligibility × negative DA = positive dw,
        # which would INCREASE weights on punishment - completely backwards!
        d1_da = da_level * cfg.d1_da_sensitivity
        d1_masked_elig = self.d1_eligibility.abs() * action_mask.unsqueeze(1)
        d1_dw = d1_masked_elig * d1_da

        # =====================================================================
        # D2 PATHWAY LEARNING (INVERTED dopamine response!)
        # =====================================================================
        # DA+ → LTD (weaken NOGO for rewarded actions)
        # DA- → LTP (strengthen NOGO for punished actions)
        # The NEGATIVE sign here is the key biological insight!
        #
        # CRITICAL: D2 learning should be ASYMMETRIC!
        # Punishment (DA-) should train D2 MORE than reward weakens it.
        # This prevents D2 collapse during successful learning.
        # Biologically: D2-MSNs are more resistant to LTD than D1-MSNs.
        #
        # We implement this by scaling down LTD (positive DA, decreasing D2):
        d2_ltd_scale = 0.3  # D2 decreases 3x slower than D1 increases
        if da_level > 0:
            d2_da = -da_level * cfg.d2_da_sensitivity * d2_ltd_scale
        else:
            d2_da = -da_level * cfg.d2_da_sensitivity  # Full strength for LTP
        d2_masked_elig = self.d2_eligibility.abs() * action_mask.unsqueeze(1)
        d2_dw = d2_masked_elig * d2_da

        # =====================================================================
        # HETEROSYNAPTIC COMPETITION: Other actions get opposite update
        # =====================================================================
        # When chosen action is rewarded (+DA), other actions are slightly penalized.
        # When chosen action is punished (-DA), other actions are slightly boosted.
        #
        # This creates RELATIVE value: "A worked, so B is relatively less valuable"
        # Key: We use the OTHER actions' eligibility, not the chosen action's.
        #
        # Biological basis: Heterosynaptic LTD - competition for synaptic resources
        if cfg.heterosynaptic_competition and self.last_action is not None:
            # Create inverse mask (1 for non-chosen, 0 for chosen)
            other_mask = 1.0 - action_mask

            # Competitive update: OPPOSITE direction, scaled down
            # If +DA → chosen gets LTP → others get small LTD
            # If -DA → chosen gets LTD → others get small LTP
            competition_scale = cfg.competition_strength

            # D1: Others get opposite of chosen
            d1_other_elig = self.d1_eligibility.abs() * other_mask.unsqueeze(1)
            d1_dw_competition = d1_other_elig * (-d1_da) * competition_scale
            d1_dw = d1_dw + d1_dw_competition

            # D2: Others get opposite of chosen (remember D2 already inverted)
            d2_other_elig = self.d2_eligibility.abs() * other_mask.unsqueeze(1)
            d2_dw_competition = d2_other_elig * (-d2_da) * competition_scale
            d2_dw = d2_dw + d2_dw_competition

        old_d1 = self.d1_weights.clone()
        self.d1_weights = (self.d1_weights + d1_dw).clamp(
            self.config.w_min, self.config.w_max
        )
        d1_actual = self.d1_weights - old_d1
        d1_ltp = d1_actual[d1_actual > 0].sum().item() if (d1_actual > 0).any() else 0.0
        d1_ltd = d1_actual[d1_actual < 0].sum().item() if (d1_actual < 0).any() else 0.0

        old_d2 = self.d2_weights.clone()
        self.d2_weights = (self.d2_weights + d2_dw).clamp(
            self.config.w_min, self.config.w_max
        )
        d2_actual = self.d2_weights - old_d2
        d2_ltp = d2_actual[d2_actual > 0].sum().item() if (d2_actual > 0).any() else 0.0
        d2_ltd = d2_actual[d2_actual < 0].sum().item() if (d2_actual < 0).any() else 0.0

        # =====================================================================
        # UNIFIED HOMEOSTASIS: Apply D1/D2 constraint
        # =====================================================================
        # This is the KEY to stability. Instead of many overlapping corrections
        # (BCM, synaptic scaling, etc.), we use a single mathematical constraint:
        #
        # D1 + D2 per action MUST sum to a fixed budget.
        #
        # This GUARANTEES:
        # - Neither pathway can completely dominate
        # - If D1 grows, D2 must shrink (and vice versa)
        # - Stable equilibrium is enforced mathematically
        #
        # Unlike corrections (which might not work with wrong parameters),
        # constraints always work because they're mathematical invariants.
        d1_before_norm = self.d1_weights.mean().item()
        d2_before_norm = self.d2_weights.mean().item()

        if self.unified_homeostasis is not None:
            self.d1_weights, self.d2_weights = self.unified_homeostasis.normalize_d1_d2(
                self.d1_weights, self.d2_weights, per_action=True
            )

        d1_after_norm = self.d1_weights.mean().item()
        d2_after_norm = self.d2_weights.mean().item()

        # DEBUG: Print learning per trial (first few and every 50)
        if hasattr(self, '_debug_trial_count') and (self._debug_trial_count <= 2 or self._debug_trial_count % 50 == 0):
            action_name = "M" if self.last_action == 0 else "NM"
            print(f"    [LRN t={self._debug_trial_count}] action={action_name}, DA={da_level:.3f}, "
                  f"d1_dw={d1_ltp+d1_ltd:.4f}, d2_dw={d2_ltp+d2_ltd:.4f}, "
                  f"elig_max=(d1:{self.d1_eligibility.abs().max().item():.4f}, d2:{self.d2_eligibility.abs().max().item():.4f})")

        # =====================================================================
        # BASELINE PRESSURE: Drift D1/D2 towards balance
        # =====================================================================
        # Unlike normalization (which preserves ratios), baseline pressure
        # actively pushes NET (D1-D2) towards the target for each action.
        # This prevents runaway biases - all actions stay viable options.
        baseline_applied = self._apply_baseline_pressure()

        # NOTE: Eligibility reset is now EXPLICIT via reset_eligibility()
        # This allows counterfactual learning to use the same traces before reset.
        # The caller (e.g., BrainSystem.deliver_reward_with_counterfactual) is
        # responsible for calling reset_eligibility() when all learning is done.

        return {
            "dopamine": da_level,
            "d1_ltp": d1_ltp,
            "d1_ltd": d1_ltd,
            "d2_ltp": d2_ltp,
            "d2_ltd": d2_ltd,
            "net_change": d1_ltp + d1_ltd + d2_ltp + d2_ltd,
            "d1_eligibility_max": self.d1_eligibility.abs().max().item(),
            "d2_eligibility_max": self.d2_eligibility.abs().max().item(),
            "d1_before_norm": d1_before_norm,
            "d2_before_norm": d2_before_norm,
            "d1_after_norm": d1_after_norm,
            "d2_after_norm": d2_after_norm,
            "baseline_applied": baseline_applied,
        }

    def deliver_counterfactual_reward(
        self,
        reward: float,
        action: int,
        counterfactual_scale: float = 0.5,
    ) -> Dict[str, Any]:
        """Apply learning for a counterfactual (imagined) action outcome.

        This implements model-based RL: the brain can simulate "what if I had
        taken a different action?" and learn from the imagined outcome without
        actually taking the action.

        Key insight: We use the SAME eligibility traces (which reflect current
        input-output associations) but apply them to a DIFFERENT action's weights.
        This is like asking "if I had activated the NOMATCH population instead,
        would I have gotten reward?"

        Counterfactual learning is scaled down because:
        1. We're less certain about imagined outcomes than real ones
        2. We want real experience to dominate over simulation
        3. The brain uses this for exploration guidance, not primary learning

        NOTE: For counterfactual learning, we compute dopamine locally since
        the Brain's VTA doesn't handle counterfactuals. This is a local
        simulation within the striatum's "model" of outcomes.

        Args:
            reward: The counterfactual reward (what WOULD have happened)
            action: The action to update (the one NOT taken)
            counterfactual_scale: How much to scale the learning (default 0.5)

        Returns:
            Metrics dict with weight changes
        """
        cfg = self.striatum_config

        # =====================================================================
        # COMPUTE COUNTERFACTUAL RPE
        # =====================================================================
        # Use proper reward prediction error instead of raw reward.
        # If we imagined taking action X and it would have been rewarded,
        # but we expected X to be bad (low value), this is a POSITIVE surprise.
        # If we expected X to be good and it would have been rewarded, small surprise.
        expected_cf = self.get_expected_value(action)
        raw_rpe = reward - expected_cf

        # Scale down because this is imagined, not real experience.
        # We don't normalize like Brain does - just use scaled raw RPE.
        da_level = raw_rpe * counterfactual_scale

        # Clip to reasonable range
        da_level = max(-2.0, min(2.0, da_level))

        # =====================================================================
        # PARTIAL VALUE ESTIMATE UPDATE
        # =====================================================================
        # Update our expectation for the counterfactual action, but with reduced
        # learning rate since we didn't actually experience this outcome.
        # This prevents value estimates from becoming stale for unexplored actions.
        if self.value_estimates is not None and 0 <= action < self.n_actions:
            cf_lr = cfg.rpe_learning_rate * counterfactual_scale
            self.value_estimates[action] = (
                self.value_estimates[action]
                + cf_lr * (reward - self.value_estimates[action])
            )

        if abs(da_level) < 0.01:
            return {
                "dopamine": da_level,
                "raw_rpe": raw_rpe,
                "expected_value": expected_cf,
                "net_change": 0.0,
                "counterfactual": True,
                "action": action,
            }

        # Create action mask for the COUNTERFACTUAL action (not the one we took)
        action_mask = torch.zeros(self.config.n_output, device=self.device)
        if self.striatum_config.population_coding:
            pop_slice = self._get_action_population_indices(action)
            action_mask[pop_slice] = 1.0
        else:
            action_mask[action] = 1.0

        # D1 pathway: DA+ → strengthen GO, DA- → weaken GO
        d1_da = da_level * cfg.d1_da_sensitivity
        d1_masked_elig = self.d1_eligibility.abs() * action_mask.unsqueeze(1)
        d1_dw = d1_masked_elig * d1_da

        old_d1 = self.d1_weights.clone()
        self.d1_weights = (self.d1_weights + d1_dw).clamp(
            self.config.w_min, self.config.w_max
        )
        d1_actual = self.d1_weights - old_d1
        d1_change = d1_actual.sum().item()

        # D2 pathway: INVERTED response (DA+ → weaken NOGO, DA- → strengthen NOGO)
        d2_da = -da_level * cfg.d2_da_sensitivity
        d2_masked_elig = self.d2_eligibility.abs() * action_mask.unsqueeze(1)
        d2_dw = d2_masked_elig * d2_da

        old_d2 = self.d2_weights.clone()
        self.d2_weights = (self.d2_weights + d2_dw).clamp(
            self.config.w_min, self.config.w_max
        )
        d2_actual = self.d2_weights - old_d2
        d2_change = d2_actual.sum().item()

        return {
            "dopamine": da_level,
            "raw_rpe": raw_rpe,
            "expected_value": expected_cf,
            "d1_change": d1_change,
            "d2_change": d2_change,
            "net_change": d1_change + d2_change,
            "counterfactual": True,
            "action": action,
        }

    def reset_eligibility(self, action_only: bool = True) -> None:
        """Reset eligibility traces after learning is complete.

        With action_only=True (default, Option 4 biologically realistic):
        - Only reset the CHOSEN action's eligibility
        - Non-chosen actions keep their eligibility, which decays naturally
        - This allows exploration to eventually "use" accumulated eligibility

        With action_only=False (full reset):
        - Reset ALL eligibility traces
        - Use this for counterfactual learning where all actions learn

        Args:
            action_only: If True, only reset chosen action. If False, reset all.
        """
        if action_only and self.last_action is not None:
            # Only reset the chosen action's eligibility
            if self.striatum_config.population_coding:
                start = self.last_action * self.neurons_per_action
                end = start + self.neurons_per_action
                self.d1_eligibility[start:end].zero_()
                self.d2_eligibility[start:end].zero_()
                self.stdp_eligibility[start:end].zero_()
            else:
                self.d1_eligibility[self.last_action].zero_()
                self.d2_eligibility[self.last_action].zero_()
                self.stdp_eligibility[self.last_action].zero_()
            # Input traces are shared, so we do decay them but don't zero
            # (they'll decay naturally via the eligibility_decay factor)
        else:
            # Full reset (original behavior)
            self.d1_eligibility.zero_()
            self.d2_eligibility.zero_()
            self.stdp_eligibility.zero_()

        # Always reset the spike traces (these are per-timestep, not per-action)
        self.d1_input_trace.zero_()
        self.d2_input_trace.zero_()
        self.d1_output_trace.zero_()
        self.d2_output_trace.zero_()
        self.input_trace.zero_()
        self.output_trace.zero_()

    def reset_state(self) -> None:
        super().reset_state()
        self.eligibility.reset_state()
        # NOTE: Dopamine is now managed by Brain, no local dopamine system to reset
        self.recent_spikes.zero_()
        self.last_action = None
        # Reset STDP traces for REWARD_MODULATED_STDP
        self.input_trace.zero_()
        self.output_trace.zero_()
        self.stdp_eligibility.zero_()
        self.exploring = False
        # Reset D1/D2 traces (always enabled)
        self.d1_eligibility.zero_()
        self.d2_eligibility.zero_()
        self.d1_input_trace.zero_()
        self.d2_input_trace.zero_()
        self.d1_output_trace.zero_()
        self.d2_output_trace.zero_()
        # Reset homeostatic trial counters (but NOT the EMA - that persists)
        self._trial_spike_count = 0.0
        self._trial_timesteps = 0
        # Reset all neuron populations
        if self.neurons is not None:
            self.neurons.reset_state(1)
        if hasattr(self, 'd1_neurons') and self.d1_neurons is not None:
            self.d1_neurons.reset_state(1)
        if hasattr(self, 'd2_neurons') and self.d2_neurons is not None:
            self.d2_neurons.reset_state(1)
        # Reset RPE tracking
        self._last_rpe = 0.0
        self._last_expected = 0.0
        self._last_d1_spikes = None
        self._last_d2_spikes = None

    # =========================================================================
    # DIAGNOSTIC METHODS
    # =========================================================================

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive diagnostics using DiagnosticsMixin helpers.

        Returns consolidated diagnostic information about:
        - D1/D2 pathway weights and balance
        - Accumulated votes and NET values
        - Dopamine/RPE state
        - Exploration state
        - Value estimates (if RPE enabled)
        - Action selection history
        - Homeostatic state

        This is the primary diagnostic interface for the Striatum.
        """
        # D1/D2 per-action means and NET
        d1_per_action: list[float] = []
        d2_per_action: list[float] = []
        net_per_action: list[float] = []

        for action in range(self.n_actions):
            pop_slice = self._get_action_population_indices(action)
            d1_mean = self.d1_weights[pop_slice, :].mean().item()
            d2_mean = self.d2_weights[pop_slice, :].mean().item()
            d1_per_action.append(d1_mean)
            d2_per_action.append(d2_mean)
            net_per_action.append(d1_mean - d2_mean)

        # Accumulated votes (current trial)
        d1_votes = self._d1_votes_accumulated.tolist()
        d2_votes = self._d2_votes_accumulated.tolist()
        net_votes = self.get_accumulated_net_votes().tolist()

        # Use mixin helpers for weight statistics
        d1_weight_stats = self.weight_diagnostics(self.d1_weights, "d1")
        d2_weight_stats = self.weight_diagnostics(self.d2_weights, "d2")

        # Additional NET statistics
        net_weights = self.d1_weights - self.d2_weights
        net_stats = {
            "net_weight_mean": net_weights.mean().item(),
            "net_weight_std": net_weights.std().item(),
        }

        # Dopamine state (now managed by Brain, but we track local state)
        dopamine_state = {
            "current_level": self.state.dopamine,
            "tonic_dopamine": self.tonic_dopamine,
        }

        # Exploration state
        exploration_state = {
            "exploring": self.exploring,
            "last_uncertainty": self._last_uncertainty,
            "last_exploration_prob": self._last_exploration_prob,
            "recent_accuracy": self._recent_accuracy,
        }

        # UCB state
        ucb_state = {
            "enabled": self.striatum_config.ucb_exploration,
            "total_trials": self._total_trials,
            "action_counts": self._action_counts.tolist(),
        }

        # Value estimates (if RPE enabled)
        value_estimates = None
        if self.value_estimates is not None:
            value_estimates = self.value_estimates.tolist()

        # Use mixin helpers for eligibility trace diagnostics
        eligibility_state = {
            **self.trace_diagnostics(self.d1_eligibility, "d1_elig"),
            **self.trace_diagnostics(self.d2_eligibility, "d2_elig"),
        }

        return {
            "region": "striatum",
            "n_actions": self.n_actions,
            "neurons_per_action": self.neurons_per_action,
            "last_action": self.last_action,
            # Per-action state
            "d1_weight_means": d1_per_action,
            "d2_weight_means": d2_per_action,
            "net_weight_means": net_per_action,
            # Current trial votes
            "d1_votes": d1_votes,
            "d2_votes": d2_votes,
            "net_votes": net_votes,
            # Weight statistics (from mixin)
            **d1_weight_stats,
            **d2_weight_stats,
            **net_stats,
            # Dopamine
            "dopamine": dopamine_state,
            # Exploration
            "exploration": exploration_state,
            "ucb": ucb_state,
            # Value estimates
            "value_estimates": value_estimates,
            # Eligibility (from mixin)
            "eligibility": eligibility_state,
        }
