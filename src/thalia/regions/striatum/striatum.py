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

from contextlib import contextmanager
from dataclasses import replace
from typing import Optional, Dict, Any, List, Generator

import torch
import torch.nn as nn

from thalia.core.diagnostics_mixin import DiagnosticsMixin
from thalia.core.weight_init import WeightInitializer
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

    Mixins Provide:
    ---------------
    From DiagnosticsMixin:
        - check_health() → HealthMetrics
        - get_firing_rate(spikes) → float
        - check_weight_health(weights, name) → WeightHealth
        - detect_runaway_excitation(spikes) → bool
        - detect_silence(spikes) → bool

    From ActionSelectionMixin:
        - select_action_softmax(q_values, temperature) → int
        - select_action_greedy(q_values, epsilon) → int
        - compute_policy(q_values, temperature) → Tensor
        - add_exploration_noise(q_values, noise_std) → Tensor

    From BrainRegion (abstract base):
        - forward(input, **kwargs) → Tensor [must implement]
        - reset_state() → None
        - get_diagnostics() → Dict
        - set_dopamine(level) → None

    See Also:
        docs/patterns/mixins.md for detailed mixin patterns
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
        # TD(λ) - MULTI-STEP CREDIT ASSIGNMENT (Phase 1 Enhancement)
        # =====================================================================
        # TD(λ) extends temporal credit assignment from ~1 second to 5-10 seconds
        # by combining eligibility traces with multi-step returns.
        #
        # When enabled, replaces basic eligibility traces with TD(λ) traces that
        # accumulate with factor (γλ) instead of simple exponential decay.
        if self.striatum_config.use_td_lambda:
            from .td_lambda import TDLambdaLearner, TDLambdaConfig

            td_config = TDLambdaConfig(
                lambda_=self.striatum_config.td_lambda,
                gamma=self.striatum_config.td_gamma,
                accumulating=self.striatum_config.td_lambda_accumulating,
                device=self.config.device,
            )

            # Create TD(λ) learner for D1 and D2 pathways
            # Note: Use config.n_output (actual neuron count) not n_actions
            # because with population coding, n_output = n_actions * neurons_per_action
            self.td_lambda_d1 = TDLambdaLearner(
                n_actions=self.config.n_output,  # Total neurons, not actions
                n_input=self.config.n_input,
                config=td_config,
            )
            self.td_lambda_d2 = TDLambdaLearner(
                n_actions=self.config.n_output,  # Total neurons, not actions
                n_input=self.config.n_input,
                config=td_config,
            )
        else:
            self.td_lambda_d1 = None
            self.td_lambda_d2 = None

        # =====================================================================
        # GOAL-CONDITIONED VALUES (Phase 1 Week 2-3 Enhancement)
        # =====================================================================
        # Enable PFC goal context to modulate striatal action values via gating.
        # Biology: PFC working memory → Striatum modulation (Miller & Cohen 2001)
        # Learning: Three-factor rule extended with goal context:
        #   Δw = eligibility × dopamine × goal_context
        if self.striatum_config.use_goal_conditioning:
            # Initialize PFC → D1 modulation weights
            self.pfc_modulation_d1 = nn.Parameter(
                WeightInitializer.sparse_random(
                    n_output=self.config.n_output,  # D1 neurons
                    n_input=self.striatum_config.pfc_size,
                    sparsity=0.3,
                    device=torch.device(self.config.device),
                )
            )
            # Initialize PFC → D2 modulation weights
            self.pfc_modulation_d2 = nn.Parameter(
                WeightInitializer.sparse_random(
                    n_output=self.config.n_output,  # D2 neurons
                    n_input=self.striatum_config.pfc_size,
                    sparsity=0.3,
                    device=torch.device(self.config.device),
                )
            )
        else:
            self.pfc_modulation_d1 = None
            self.pfc_modulation_d2 = None

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

        # Initialize tracking variables (also reset in reset_state)
        self._last_rpe = 0.0
        self._last_expected = 0.0
        self._last_d1_spikes = None
        self._last_d2_spikes = None
        self._last_pfc_goal_context = None  # For goal-conditioned learning

        # =====================================================================
        # BETA OSCILLATOR TRACKING (Motor Control Modulation)
        # =====================================================================
        # Beta oscillations (13-30 Hz, typically 20 Hz) modulate action selection:
        # - High beta: Maintain current action (D1 dominant, suppress switching)
        # - Low beta: Allow action switching (D2 can suppress, facilitate change)
        # - Beta desynchronization (ERD): Action change window
        # - Beta rebound (ERS): Action stabilization
        self._beta_phase: float = 0.0
        self._theta_phase: float = 0.0
        self._beta_amplitude: float = 1.0
        self._coupled_amplitudes: Dict[str, float] = {}

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

    def evaluate_state(
        self,
        state: torch.Tensor,
        pfc_goal_context: Optional[torch.Tensor] = None
    ) -> float:
        """
        Evaluate state quality using learned action values.

        For Phase 2 model-based planning: predicts how good a simulated state is
        by computing the maximum Q-value (best action value) from that state.

        Uses existing value estimates to evaluate states during mental simulation.
        If goal-conditioning is enabled, modulates values based on goal context.

        Biology: Striatum represents action values (Q-values) learned through
        dopaminergic reinforcement. During planning, these values can evaluate
        simulated future states (Daw et al., 2011).

        Args:
            state: State to evaluate [n_input] (1D, ADR-005)
            pfc_goal_context: Optional goal context from PFC [n_pfc]

        Returns:
            state_value: Maximum action value (best Q-value) from this state

        Note:
            This is a simplified evaluator using cached Q-values. In full
            implementation, would process state through forward() to get
            D1-D2 competition values.
        """
        if self.value_estimates is None:
            # No value estimates available, return neutral value
            return 0.0

        # Get all action values
        action_values = self.value_estimates.clone()

        # If goal conditioning enabled and goal context provided, modulate values
        if (self.striatum_config.use_goal_conditioning and
            pfc_goal_context is not None and
            hasattr(self, 'pfc_modulation_d1') and
            self.pfc_modulation_d1 is not None):

            # Compute goal modulation for D1 (Go pathway)
            goal_mod_d1 = torch.sigmoid(
                self.pfc_modulation_d1 @ pfc_goal_context
            )

            # Scale action values by goal relevance
            # Higher goal modulation → boost that action's value
            if self.striatum_config.population_coding:
                # With population coding, modulation applies per action
                for action_idx in range(self.n_actions):
                    start = action_idx * self.neurons_per_action
                    end = start + self.neurons_per_action
                    action_mod = goal_mod_d1[start:end].mean()
                    # Modulate value: centered around 1.0, range [0.5, 1.5]
                    action_values[action_idx] *= (0.5 + action_mod)
            else:
                # Direct modulation per action
                action_mod = goal_mod_d1[:self.n_actions]
                action_values *= (0.5 + action_mod)

        # Return max value (best action from this state)
        return action_values.max().item()

    def add_neurons(
        self,
        n_new: int,
        initialization: str = 'xavier',
        sparsity: float = 0.1,
    ) -> None:
        """Add new action neurons to striatum without disrupting existing circuits.

        For Striatum with population coding, adds complete action populations.
        If n_new=1, adds neurons_per_action neurons total.

        Expands:
        - D1 and D2 weight matrices [n_output, n_input] → [n_output + n_new, n_input]
        - Eligibility traces (D1 and D2)
        - Spike traces (input and output for both pathways)
        - Neuron populations (D1-MSNs and D2-MSNs)
        - Action-related tracking (value estimates, Q-values, etc.)

        Args:
            n_new: Number of new actions to add (each action = neurons_per_action neurons)
            initialization: Weight init strategy ('xavier', 'sparse_random', 'uniform')
            sparsity: Connection sparsity for new neurons (0.0 = no connections, 1.0 = fully connected)

        Example:
            >>> striatum = Striatum(StriatumConfig(n_output=2, neurons_per_action=10))
            >>> # Currently: 2 actions × 10 neurons = 20 total neurons
            >>> striatum.add_neurons(n_new=1)  # Add 1 action
            >>> # Now: 3 actions × 10 neurons = 30 total neurons
        """
        # Calculate actual number of neurons to add (population coding)
        n_new_neurons = n_new * self.neurons_per_action
        old_n_output = self.config.n_output
        new_n_output = old_n_output + n_new_neurons

        # =====================================================================
        # 1. EXPAND D1 AND D2 WEIGHT MATRICES
        # =====================================================================
        # Initialize new weights using specified strategy
        if initialization == 'xavier':
            new_d1_weights = WeightInitializer.xavier(
                n_output=n_new_neurons,
                n_input=self.config.n_input,
                gain=0.2,  # Match _initialize_pathway_weights
                device=self.device,
            ) * self.config.w_max
            new_d2_weights = WeightInitializer.xavier(
                n_output=n_new_neurons,
                n_input=self.config.n_input,
                gain=0.2,
                device=self.device,
            ) * self.config.w_max
        elif initialization == 'sparse_random':
            new_d1_weights = WeightInitializer.sparse_random(
                n_output=n_new_neurons,
                n_input=self.config.n_input,
                sparsity=sparsity,
                scale=self.config.w_max * 0.2,
                device=self.device,
            )
            new_d2_weights = WeightInitializer.sparse_random(
                n_output=n_new_neurons,
                n_input=self.config.n_input,
                sparsity=sparsity,
                scale=self.config.w_max * 0.2,
                device=self.device,
            )
        else:  # uniform
            new_d1_weights = WeightInitializer.uniform(
                n_output=n_new_neurons,
                n_input=self.config.n_input,
                low=0.0,
                high=self.config.w_max * 0.2,
                device=self.device,
            )
            new_d2_weights = WeightInitializer.uniform(
                n_output=n_new_neurons,
                n_input=self.config.n_input,
                low=0.0,
                high=self.config.w_max * 0.2,
                device=self.device,
            )

        # Clamp to bounds
        new_d1_weights = new_d1_weights.clamp(self.config.w_min, self.config.w_max)
        new_d2_weights = new_d2_weights.clamp(self.config.w_min, self.config.w_max)

        # Concatenate with existing weights
        self.d1_weights = torch.cat([self.d1_weights, new_d1_weights], dim=0)
        self.d2_weights = torch.cat([self.d2_weights, new_d2_weights], dim=0)

        # Update generic weights reference (use d1_weights for compatibility)
        self.weights = self.d1_weights

        # =====================================================================
        # 2. UPDATE CONFIG (DO THIS BEFORE CREATING NEURONS!)
        # =====================================================================
        # Neurons are created based on config.n_output, so update it first
        # BOTH config.n_output and striatum_config.n_output store TOTAL NEURONS
        # We track n_actions separately in self.n_actions
        self.n_actions += n_new
        self.config = replace(self.config, n_output=new_n_output)
        self.striatum_config = replace(self.striatum_config, n_output=new_n_output)

        # =====================================================================
        # 3. EXPAND ELIGIBILITY TRACES
        # =====================================================================
        new_d1_elig = torch.zeros(n_new_neurons, self.config.n_input, device=self.device)
        new_d2_elig = torch.zeros(n_new_neurons, self.config.n_input, device=self.device)
        self.d1_eligibility = torch.cat([self.d1_eligibility, new_d1_elig], dim=0)
        self.d2_eligibility = torch.cat([self.d2_eligibility, new_d2_elig], dim=0)

        # Expand eligibility traces object
        if hasattr(self, 'eligibility'):
            self.eligibility.n_post = new_n_output
            self.eligibility.traces = torch.cat([
                self.eligibility.traces,
                torch.zeros(n_new_neurons, self.config.n_input, device=self.device)
            ], dim=0)

        # =====================================================================
        # 3. EXPAND SPIKE TRACES
        # =====================================================================
        new_d1_trace = torch.zeros(n_new_neurons, device=self.device)
        new_d2_trace = torch.zeros(n_new_neurons, device=self.device)
        self.d1_output_trace = torch.cat([self.d1_output_trace, new_d1_trace], dim=0)
        self.d2_output_trace = torch.cat([self.d2_output_trace, new_d2_trace], dim=0)

        if hasattr(self, 'recent_spikes'):
            self.recent_spikes = torch.cat([
                self.recent_spikes,
                torch.zeros(n_new_neurons, device=self.device)
            ], dim=0)
        else:
            # Initialize if it doesn't exist
            self.recent_spikes = torch.zeros(new_n_output, device=self.device)

        # =====================================================================
        # 4. EXPAND NEURON POPULATIONS
        # =====================================================================
        # Expand D1-MSN and D2-MSN neuron populations
        if hasattr(self, 'd1_neurons') and self.d1_neurons is not None:
            old_d1_membrane = self.d1_neurons.membrane.clone() if self.d1_neurons.membrane is not None else None
            old_d1_g_E = self.d1_neurons.g_E.clone() if self.d1_neurons.g_E is not None else None
            old_d1_g_I = self.d1_neurons.g_I.clone() if self.d1_neurons.g_I is not None else None
            old_d1_refractory = self.d1_neurons.refractory.clone() if self.d1_neurons.refractory is not None else None

            self.d1_neurons = self._create_d1_neurons()
            self.d1_neurons.reset_state()

            # ADR-005: Neuron state tensors are 1D [n_neurons], not 2D
            if old_d1_membrane is not None:
                self.d1_neurons.membrane[:old_n_output] = old_d1_membrane
            if old_d1_g_E is not None:
                self.d1_neurons.g_E[:old_n_output] = old_d1_g_E
            if old_d1_g_I is not None:
                self.d1_neurons.g_I[:old_n_output] = old_d1_g_I
            if old_d1_refractory is not None:
                self.d1_neurons.refractory[:old_n_output] = old_d1_refractory

        if hasattr(self, 'd2_neurons') and self.d2_neurons is not None:
            old_d2_membrane = self.d2_neurons.membrane.clone() if self.d2_neurons.membrane is not None else None
            old_d2_g_E = self.d2_neurons.g_E.clone() if self.d2_neurons.g_E is not None else None
            old_d2_g_I = self.d2_neurons.g_I.clone() if self.d2_neurons.g_I is not None else None
            old_d2_refractory = self.d2_neurons.refractory.clone() if self.d2_neurons.refractory is not None else None

            self.d2_neurons = self._create_d2_neurons()
            self.d2_neurons.reset_state()

            # ADR-005: Neuron state tensors are 1D [n_neurons], not 2D
            if old_d2_membrane is not None:
                self.d2_neurons.membrane[:old_n_output] = old_d2_membrane
            if old_d2_g_E is not None:
                self.d2_neurons.g_E[:old_n_output] = old_d2_g_E
            if old_d2_g_I is not None:
                self.d2_neurons.g_I[:old_n_output] = old_d2_g_I
            if old_d2_refractory is not None:
                self.d2_neurons.refractory[:old_n_output] = old_d2_refractory

        # =====================================================================
        # 5. UPDATE ACTION-RELATED TRACKING
        # =====================================================================
        # Expand D1/D2 vote accumulators
        self._d1_votes_accumulated = torch.cat([
            self._d1_votes_accumulated,
            torch.zeros(n_new, device=self.device)
        ], dim=0)
        self._d2_votes_accumulated = torch.cat([
            self._d2_votes_accumulated,
            torch.zeros(n_new, device=self.device)
        ], dim=0)

        # Expand action counts for UCB
        self._action_counts = torch.cat([
            self._action_counts,
            torch.zeros(n_new, device=self.device)
        ], dim=0)

        # Expand unified homeostasis if it exists
        if self.unified_homeostasis is not None:
            # Expand activity tracking buffers
            old_size = self.unified_homeostasis.n_neurons
            new_size = old_size + n_new_neurons

            # Expand D1 and D2 activity tracking
            self.unified_homeostasis.d1_activity_avg = torch.cat([
                self.unified_homeostasis.d1_activity_avg,
                torch.zeros(n_new_neurons, device=self.device)
            ], dim=0)
            self.unified_homeostasis.d2_activity_avg = torch.cat([
                self.unified_homeostasis.d2_activity_avg,
                torch.zeros(n_new_neurons, device=self.device)
            ], dim=0)

            # Expand excitability modulation factors
            self.unified_homeostasis.d1_excitability = torch.cat([
                self.unified_homeostasis.d1_excitability,
                torch.ones(n_new_neurons, device=self.device)
            ], dim=0)
            self.unified_homeostasis.d2_excitability = torch.cat([
                self.unified_homeostasis.d2_excitability,
                torch.ones(n_new_neurons, device=self.device)
            ], dim=0)

            # Update size tracking
            self.unified_homeostasis.n_actions = self.n_actions
            self.unified_homeostasis.n_neurons = new_size

            # Expand action_budgets
            self.unified_homeostasis.action_budgets = torch.cat([
                self.unified_homeostasis.action_budgets,
                torch.ones(n_new, device=self.device) * self.unified_homeostasis.config.weight_budget
            ], dim=0)

        # Value estimates for new actions (start at 0)
        if hasattr(self, 'value_estimates'):
            self.value_estimates = torch.cat([
                self.value_estimates,
                torch.zeros(n_new, device=self.device)
            ], dim=0)

        # RPE traces for new actions (only if rpe_trace was already initialized)
        if hasattr(self, 'rpe_trace') and self.rpe_trace is not None:
            self.rpe_trace = torch.cat([
                self.rpe_trace,
                torch.zeros(n_new, device=self.device)
            ], dim=0)

    def _initialize_pathway_weights(self) -> torch.Tensor:
        """Initialize weights for D1 or D2 pathway with balanced, principled scaling.

        Uses Xavier initialization to ensure consistent input magnitude regardless of
        input size. Both D1 and D2 start with identical distributions - the
        competition between GO and NOGO pathways emerges purely from learning.

        Principles:
        1. Xavier scaling: Normalizes for different input sizes
        2. Equal D1/D2: No baked-in pathway preference
        3. Minimal variance: Near-symmetric start prevents early bias lock-in (gain=0.2)
        4. Moderate values: Keeps neurons in operational firing regime

        This avoids encoding task-specific knowledge into initialization.
        """
        # Use Xavier initialization with reduced gain for minimal variance
        # Gain of 0.2 provides similar variance to old system (0.01)
        weights = WeightInitializer.xavier(
            n_output=self.config.n_output,
            n_input=self.config.n_input,
            gain=0.2,  # Reduced gain for near-symmetric start
            device=self.device
        )

        # Scale by w_max and clamp to bounds
        weights = weights * self.config.w_max

        return weights.clamp(self.config.w_min, self.config.w_max)

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

        Timestep (dt) is obtained from self.config.dt_ms for temporal dynamics.

        Args:
            input_spikes: Input spike tensor [n_input] (1D)
            d1_spikes: D1 neuron population spikes [n_d1] (1D)
            d2_spikes: D2 neuron population spikes [n_d2] (1D)
            chosen_action: If provided, only build eligibility for this action's neurons
        """
        # Get timestep from config
        dt = self.config.dt_ms

        # Ensure 1D
        if input_spikes.dim() != 1:
            input_spikes = input_spikes.squeeze()
        if d1_spikes.dim() != 1:
            d1_spikes = d1_spikes.squeeze()
        if d2_spikes.dim() != 1:
            d2_spikes = d2_spikes.squeeze()

        cfg = self.striatum_config

        # Get float versions for trace updates
        input_1d = input_spikes.float()
        d1_output_1d = d1_spikes.float()
        d2_output_1d = d2_spikes.float()

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

        Timestep (dt) is obtained from self.config.dt_ms for temporal dynamics.

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

    def set_oscillator_phases(
        self,
        phases: Dict[str, float],
        signals: Optional[Dict[str, float]] = None,
        theta_slot: int = 0,
        coupled_amplitudes: Optional[Dict[str, float]] = None,
    ) -> None:
        """Receive oscillator information from brain broadcast.

        Beta oscillations modulate action selection and maintenance:
        - High beta amplitude: Action persistence (D1 dominant)
          * Increases D1 gain → stronger GO signals
          * Decreases D2 gain → weaker NOGO signals
          * Results in action maintenance (harder to switch)

        - Low beta amplitude: Action flexibility (D2 effective)
          * Decreases D1 gain → weaker GO signals
          * Allows D2 to suppress → easier to switch actions
          * Results in action exploration (easier to change)

        Biology:
        - Motor cortex shows high beta during action maintenance
        - Beta desynchronization (ERD) precedes action changes
        - Beta rebound (ERS) stabilizes new action after selection
        - Striatum receives beta-modulated cortico-striatal input

        Args:
            phases: Oscillator phases in radians {'theta': ..., 'beta': ..., 'gamma': ...}
            signals: Oscillator signal values (sin/cos waveforms)
            theta_slot: Current theta slot [0, n_slots-1] for working memory
            coupled_amplitudes: Coupled amplitudes {'beta_by_theta': ..., etc}

        Note:
            Called automatically by Brain before each forward() call.
            Do not call this manually.
        """
        # Store oscillator phases for action selection
        self._beta_phase = phases.get('beta', 0.0)
        self._theta_phase = phases.get('theta', 0.0)

        # Store effective amplitude (pre-computed by OscillatorManager)
        # Automatic multiplicative coupling:
        # - Beta modulated by ALL slower oscillators (delta, theta, alpha)
        # OscillatorManager handles the multiplication, we just store the result.
        if coupled_amplitudes is not None:
            self._beta_amplitude = coupled_amplitudes.get('beta', 1.0)
        else:
            self._beta_amplitude = 1.0

    def _initialize_weights(self) -> torch.Tensor:
        """Initialize with small positive weights."""
        weights = WeightInitializer.uniform(
            n_output=self.config.n_output,
            n_input=self.config.n_input,
            low=0.0,
            high=self.config.w_max * 0.2,
            device=self.device
        )
        return weights.clamp(self.config.w_min, self.config.w_max)

    def _create_neurons(self) -> ConductanceLIF:
        """Create MSN-like neurons (legacy - kept for parent class compatibility).

        NOTE: This is now only used for parent class compatibility.
        The actual D1/D2 neurons are created by _create_d1_neurons() and
        _create_d2_neurons() which are called in __init__.
        """
        neuron_config = ConductanceLIFConfig(
            v_threshold=1.0, v_reset=0.0, E_L=0.0, E_E=3.0, E_I=-0.5,
            tau_E=5.0, tau_I=5.0,
            dt_ms=1.0,
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
            dt_ms=self.config.dt_ms,
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
            dt_ms=self.config.dt_ms,
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
        pfc_goal_context: Optional[torch.Tensor] = None,
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
            input_spikes: Input spike tensor [n_input] (1D)
            pfc_goal_context: Optional goal context from PFC [pfc_size] (1D)

        NOTE: Exploration is handled by finalize_action() at trial end, not per-timestep.
        NOTE: Theta modulation computed internally from self._theta_phase (set by Brain)

        With population coding:
        - Each action has N neurons per pathway (neurons_per_action)
        - D1_votes = sum(D1 spikes for action)
        - D2_votes = sum(D2 spikes for action)
        - NET = D1_votes - D2_votes
        - Selected action = argmax(NET)
        """
        # Ensure 1D input (ADR-005)
        if input_spikes.dim() != 1:
            input_spikes = input_spikes.squeeze()

        assert input_spikes.dim() == 1, (
            f"Striatum.forward: Expected 1D input (ADR-005), got shape {input_spikes.shape}. "
            "Thalia uses single-brain architecture with no batch dimension."
        )

        # Reset D1 and D2 neuron states if needed
        if self.d1_neurons.membrane is None:
            self.d1_neurons.reset_state()
        if self.d2_neurons.membrane is None:
            self.d2_neurons.reset_state()

        # =====================================================================
        # D1/D2 SEPARATE POPULATIONS - COMPUTE ACTIVATIONS
        # =====================================================================
        # D1 and D2 weights project to SEPARATE neuron populations
        # Each population receives its weights as EXCITATORY input

        # Convert bool spikes to float for matmul
        input_spikes_float = input_spikes.float() if input_spikes.dtype == torch.bool else input_spikes
        d1_activation = torch.matmul(self.d1_weights, input_spikes_float)  # [n_d1]
        d2_activation = torch.matmul(self.d2_weights, input_spikes_float)  # [n_d2]

        # =====================================================================
        # THETA MODULATION
        # =====================================================================
        # Compute theta modulation from current phase (set by Brain's OscillatorManager)
        encoding_mod = (1 + torch.cos(torch.tensor(self._theta_phase, device=self.device))) / 2
        retrieval_mod = (1 - torch.cos(torch.tensor(self._theta_phase, device=self.device))) / 2

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
        # BETA OSCILLATION MODULATION (Motor Control)
        # =====================================================================
        # Beta amplitude modulates D1/D2 balance for action maintenance vs switching:
        # - High beta (e.g., 0.8-1.0): Action maintenance
        #   * Boost D1 gain → stronger GO signals → maintain current action
        #   * Reduce D2 gain → weaker NOGO signals → harder to switch
        # - Low beta (e.g., 0.2-0.4): Action flexibility
        #   * Reduce D1 gain → weaker GO signals → easier to interrupt
        #   * Boost D2 gain → stronger NOGO signals → allow suppression
        #
        # Biology: Motor cortex shows high beta during postural maintenance,
        # beta desynchronization (ERD) before action changes, and beta rebound
        # (ERS) after new action selection.
        beta_mod = self.striatum_config.beta_modulation_strength
        d1_gain = d1_gain * (1.0 + beta_mod * (self._beta_amplitude - 0.5))
        d2_gain = d2_gain * (1.0 - beta_mod * (self._beta_amplitude - 0.5))

        # =====================================================================
        # GOAL-CONDITIONED MODULATION (PFC → Striatum Gating)
        # =====================================================================
        # PFC goal context modulates D1/D2 pathways to implement goal-conditioned
        # action selection. Different goals activate different striatal ensembles.
        # Biology: PFC working memory → Striatum gating (Miller & Cohen 2001)
        if (self.striatum_config.use_goal_conditioning and
            pfc_goal_context is not None and
            self.pfc_modulation_d1 is not None):

            # Ensure goal context is 1D
            if pfc_goal_context.dim() != 1:
                pfc_goal_context = pfc_goal_context.squeeze()

            # Convert bool to float if needed
            if pfc_goal_context.dtype == torch.bool:
                pfc_goal_context = pfc_goal_context.float()

            # Compute goal modulation via learned PFC → striatum weights
            # Uses sigmoid to get modulation in [0, 1] range
            goal_mod_d1 = torch.sigmoid(
                torch.matmul(self.pfc_modulation_d1, pfc_goal_context)
            )  # [n_output]
            goal_mod_d2 = torch.sigmoid(
                torch.matmul(self.pfc_modulation_d2, pfc_goal_context)
            )  # [n_output]

            # Modulate D1/D2 gains by goal context
            # Strength parameter controls how much goals affect action selection
            strength = self.striatum_config.goal_modulation_strength
            d1_activation = d1_activation * (1.0 + strength * (goal_mod_d1 - 0.5))
            d2_activation = d2_activation * (1.0 + strength * (goal_mod_d2 - 0.5))

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

        # =====================================================================
        # NOREPINEPHRINE GAIN MODULATION (Locus Coeruleus)
        # =====================================================================
        # High NE (arousal/uncertainty): Increase gain → more exploration
        # Low NE (baseline): Normal gain
        # Biological: NE modulates striatal excitability and action variability
        ne_level = self.state.norepinephrine
        # NE gain: 1.0 (baseline) to 1.5 (high arousal)
        ne_gain = 1.0 + 0.5 * ne_level
        d1_g_exc = d1_g_exc * ne_gain

        # Add lateral inhibition within D1 population if enabled
        if self.striatum_config.lateral_inhibition:
            d1_g_inh = d1_g_inh + self.recent_spikes * self.striatum_config.inhibition_strength * 0.5

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
            d2_g_inh = d2_g_inh + self.recent_spikes * self.striatum_config.inhibition_strength * 0.5

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
        # STORE GOAL CONTEXT AND SPIKES FOR LEARNING
        # =====================================================================
        # Store PFC goal context and D1/D2 spikes for goal-conditioned learning
        # These will be used in deliver_reward() to modulate weight updates
        if self.striatum_config.use_goal_conditioning and pfc_goal_context is not None:
            self._last_pfc_goal_context = pfc_goal_context.clone()
        else:
            self._last_pfc_goal_context = None

        # Store D1/D2 spikes for PFC modulation weight learning
        self._last_d1_spikes = d1_spikes.clone()
        self._last_d2_spikes = d2_spikes.clone()

        # =====================================================================
        # UPDATE ELIGIBILITY TRACES (for all active neurons)
        # Get timestep from config for temporal dynamics
        dt = self.config.dt_ms

        # =====================================================================
        # Eligibility accumulates for ALL neurons that fire during the trial.
        # When reward arrives, deliver_reward() uses last_action (set by finalize_action)
        # to apply learning only to the chosen action's synapses.
        self.eligibility.update(input_spikes, output_spikes, dt)

        # Update D1/D2 STDP-style eligibility (always enabled)
        self._update_d1_d2_eligibility_all(input_spikes, d1_spikes, d2_spikes)

        # UPDATE TD(λ) ELIGIBILITY (if enabled)
        # TD(λ) traces accumulate with factor (γλ) instead of simple decay,
        # enabling credit assignment over longer delays (5-10 seconds)
        if self.td_lambda_d1 is not None:
            # Update TD(λ) eligibility for D1 pathway
            # Note: We update for ALL neurons here; masking to chosen action
            # happens in deliver_reward() using last_action
            d1_gradient = torch.outer(d1_spikes.float(), input_spikes_float)
            self.td_lambda_d1.traces.update(d1_gradient)

            # Update TD(λ) eligibility for D2 pathway
            d2_gradient = torch.outer(d2_spikes.float(), input_spikes_float)
            self.td_lambda_d2.traces.update(d2_gradient)

        # Update D1/D2 eligibility traces for ALL active neurons
        # The action-specific masking happens in deliver_reward(), not here
        self._update_d1_d2_eligibility_all(input_spikes, d1_spikes, d2_spikes)

        # NOTE: All neuromodulators (DA, ACh, NE) are now managed centrally by Brain.
        # VTA updates dopamine, LC updates NE, NB updates ACh.
        # Brain broadcasts to all regions every timestep via _update_neuromodulators().
        # No local decay needed.

        # Update recent spikes (based on D1 output) - already 1D
        self.recent_spikes = self.recent_spikes.float() * 0.9 + d1_spikes.float()

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

        # Choose eligibility source: TD(λ) if enabled, otherwise basic STDP traces
        if self.td_lambda_d1 is not None:
            # TD(λ) MODE: Use multi-step returns for extended credit assignment
            # Mask TD(λ) traces to chosen action
            d1_traces = self.td_lambda_d1.traces.get().abs()
            d1_masked_elig = d1_traces * action_mask.unsqueeze(1)
        else:
            # BASIC MODE: Use STDP eligibility traces
            d1_masked_elig = self.d1_eligibility.abs() * action_mask.unsqueeze(1)

        d1_da = da_level * cfg.d1_da_sensitivity
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

        # Choose eligibility source for D2
        if self.td_lambda_d2 is not None:
            # TD(λ) MODE: Use multi-step returns for extended credit assignment
            d2_traces = self.td_lambda_d2.traces.get().abs()
            d2_masked_elig = d2_traces * action_mask.unsqueeze(1)
        else:
            # BASIC MODE: Use STDP eligibility traces
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

        # =====================================================================
        # GOAL-CONDITIONED LEARNING: Modulate weight updates by goal context
        # =====================================================================
        # Extended three-factor rule: Δw = eligibility × dopamine × goal_context
        # Only synapses active during current goal context receive full updates
        # Biology: PFC → Striatum modulation gates which synapses learn
        if (self.striatum_config.use_goal_conditioning and
            hasattr(self, '_last_pfc_goal_context') and
            self._last_pfc_goal_context is not None):

            # Get goal context from last forward pass
            goal_context = self._last_pfc_goal_context  # [pfc_size]

            # Compute goal-based modulation (which striatal neurons are active for this goal)
            # This comes from the learned PFC → striatum weights
            goal_weight_d1 = torch.sigmoid(
                torch.matmul(self.pfc_modulation_d1, goal_context)
            )  # [n_output] - which D1 neurons participate in this goal
            goal_weight_d2 = torch.sigmoid(
                torch.matmul(self.pfc_modulation_d2, goal_context)
            )  # [n_output] - which D2 neurons participate in this goal

            # Modulate weight updates: only goal-relevant neurons learn fully
            # Shape: d1_dw is [n_output, n_input], goal_weight_d1 is [n_output]
            d1_dw = d1_dw * goal_weight_d1.unsqueeze(1)
            d2_dw = d2_dw * goal_weight_d2.unsqueeze(1)

            # Also update PFC modulation weights via Hebbian learning
            # When goal active + neuron active + reward → strengthen PFC → striatum connection
            # This is local learning (no backprop!)
            if abs(da_level) > 0.01:
                # Hebbian update: outer product of post-synaptic (striatal) and pre-synaptic (PFC)
                # Modulated by dopamine to only learn during success/failure
                pfc_lr = self.striatum_config.goal_modulation_lr

                # D1 modulation learning
                if self._last_d1_spikes is not None:
                    d1_hebbian = torch.outer(
                        self._last_d1_spikes.float(),
                        goal_context
                    ) * da_level * pfc_lr
                    self.pfc_modulation_d1.data += d1_hebbian
                    self.pfc_modulation_d1.data.clamp_(self.config.w_min, self.config.w_max)

                # D2 modulation learning (inverted DA response like main D2 pathway)
                if self._last_d2_spikes is not None:
                    d2_hebbian = torch.outer(
                        self._last_d2_spikes.float(),
                        goal_context
                    ) * (-da_level) * pfc_lr  # Inverted for D2
                    self.pfc_modulation_d2.data += d2_hebbian
                    self.pfc_modulation_d2.data.clamp_(self.config.w_min, self.config.w_max)

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

                # Reset TD(λ) traces if enabled
                if self.td_lambda_d1 is not None:
                    self.td_lambda_d1.traces.traces[start:end].zero_()
                    self.td_lambda_d2.traces.traces[start:end].zero_()
            else:
                self.d1_eligibility[self.last_action].zero_()
                self.d2_eligibility[self.last_action].zero_()

                # Reset TD(λ) traces if enabled
                if self.td_lambda_d1 is not None:
                    self.td_lambda_d1.traces.traces[self.last_action].zero_()
                    self.td_lambda_d2.traces.traces[self.last_action].zero_()
            # Input traces are shared, so we do decay them but don't zero
            # (they'll decay naturally via the eligibility_decay factor)
        else:
            # Full reset (original behavior)
            self.d1_eligibility.zero_()
            self.d2_eligibility.zero_()

            # Reset TD(λ) traces if enabled
            if self.td_lambda_d1 is not None:
                self.td_lambda_d1.traces.reset_state()
                self.td_lambda_d2.traces.reset_state()

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
        self.exploring = False
        # Reset D1/D2 traces (always enabled)
        self.d1_eligibility.zero_()
        self.d2_eligibility.zero_()
        self.d1_input_trace.zero_()
        self.d2_input_trace.zero_()
        self.d1_output_trace.zero_()
        self.d2_output_trace.zero_()

        # Reset TD(λ) traces if enabled
        if self.td_lambda_d1 is not None:
            self.td_lambda_d1.reset_episode()
            self.td_lambda_d2.reset_episode()

        # Reset homeostatic trial counters (but NOT the EMA - that persists)
        self._trial_spike_count = 0.0
        self._trial_timesteps = 0
        # Reset all neuron populations
        if self.neurons is not None:
            self.neurons.reset_state()
        if hasattr(self, 'd1_neurons') and self.d1_neurons is not None:
            self.d1_neurons.reset_state()
        if hasattr(self, 'd2_neurons') and self.d2_neurons is not None:
            self.d2_neurons.reset_state()
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

        # TD(λ) diagnostics (if enabled)
        td_lambda_state = {}
        if self.td_lambda_d1 is not None:
            td_lambda_state = {
                "td_lambda_enabled": True,
                "lambda": self.td_lambda_d1.config.lambda_,
                "gamma": self.td_lambda_d1.config.gamma,
                "d1_td_lambda": self.td_lambda_d1.get_diagnostics(),
                "d2_td_lambda": self.td_lambda_d2.get_diagnostics(),
            }
        else:
            td_lambda_state = {"td_lambda_enabled": False}

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
            # TD(λ) state
            "td_lambda": td_lambda_state,
        }

    # =========================================================================
    # CHECKPOINT STATE MANAGEMENT
    # =========================================================================

    def get_full_state(self) -> Dict[str, Any]:
        """Get complete state for checkpointing.

        Returns all state needed to resume training from this exact point,
        including weights, eligibility traces, neuron state, homeostasis, etc.

        Returns:
            Dictionary with complete region state
        """
        # 1. LEARNABLE PARAMETERS (weights)
        weights = {
            "weights": self.weights.detach().clone(),
            "d1_weights": self.d1_weights.detach().clone(),
            "d2_weights": self.d2_weights.detach().clone(),
        }

        # 2. REGION STATE (dynamic state)
        # Get neuron state from all populations
        neuron_state = {}
        if self.neurons is not None:
            neuron_state["main"] = self.neurons.get_state()
        if hasattr(self, 'd1_neurons') and self.d1_neurons is not None:
            neuron_state["d1"] = self.d1_neurons.get_state()
        if hasattr(self, 'd2_neurons') and self.d2_neurons is not None:
            neuron_state["d2"] = self.d2_neurons.get_state()

        region_state = {
            "recent_spikes": self.recent_spikes.detach().clone(),
            "last_action": self.last_action,
            "neuron_state": neuron_state,
            "base_region_state": {
                "membrane": self.state.membrane.detach().clone() if self.state.membrane is not None else None,
                "spikes": self.state.spikes.detach().clone() if self.state.spikes is not None else None,
                "t": self.state.t,
            }
        }

        # 3. LEARNING STATE (eligibility traces, STDP traces, etc.)
        learning_state = {
            # Eligibility traces
            "eligibility_traces": self.eligibility.get().detach().clone(),
            "d1_eligibility": self.d1_eligibility.detach().clone(),
            "d2_eligibility": self.d2_eligibility.detach().clone(),

            # D1/D2 spike traces
            "d1_input_trace": self.d1_input_trace.detach().clone(),
            "d2_input_trace": self.d2_input_trace.detach().clone(),
            "d1_output_trace": self.d1_output_trace.detach().clone(),
            "d2_output_trace": self.d2_output_trace.detach().clone(),

            # Trial accumulators
            "d1_votes_accumulated": self._d1_votes_accumulated.detach().clone(),
            "d2_votes_accumulated": self._d2_votes_accumulated.detach().clone(),

            # Homeostatic state
            "activity_ema": self._activity_ema,
            "trial_spike_count": self._trial_spike_count,
            "trial_timesteps": self._trial_timesteps,
            "homeostatic_scaling_applied": self._homeostatic_scaling_applied,

            # Unified homeostasis state (if enabled)
            "unified_homeostasis_state": self.unified_homeostasis.get_state() if self.unified_homeostasis is not None else None,
        }

        # 4. EXPLORATION STATE
        exploration_state = {
            "exploring": self.exploring,
            "last_uncertainty": self._last_uncertainty,
            "last_exploration_prob": self._last_exploration_prob,
            "action_counts": self._action_counts.detach().clone(),
            "total_trials": self._total_trials,
            "recent_rewards": list(self._recent_rewards),
            "recent_accuracy": self._recent_accuracy,
        }

        # 5. VALUE ESTIMATION STATE (if RPE enabled)
        rpe_state = {}
        if self.value_estimates is not None:
            rpe_state = {
                "value_estimates": self.value_estimates.detach().clone(),
                "last_rpe": self._last_rpe,
                "last_expected": self._last_expected,
            }

        # 6. NEUROMODULATOR STATE (from NeuromodulatorMixin)
        neuromodulator_state = self.get_neuromodulator_state()

        # Add tonic dopamine (striatum-specific)
        neuromodulator_state["tonic_dopamine"] = self.tonic_dopamine

        # 7. OSCILLATOR STATE (striatum doesn't have oscillators, but include for consistency)
        oscillator_state = {}

        return {
            "weights": weights,
            "region_state": region_state,
            "learning_state": learning_state,
            "exploration_state": exploration_state,
            "rpe_state": rpe_state,
            "neuromodulator_state": neuromodulator_state,
            "oscillator_state": oscillator_state,
            "config": self.striatum_config,
            "n_actions": self.n_actions,  # Store explicitly for validation
        }

    def load_full_state(self, state: Dict[str, Any]) -> None:
        """Restore complete state from checkpoint.

        Args:
            state: Dictionary returned by get_full_state()

        Raises:
            ValueError: If state is incompatible with current configuration
        """
        # Validate configuration compatibility
        saved_config = state["config"]

        # Always validate input dimension
        if saved_config.n_input != self.config.n_input:
            raise ValueError(
                f"Input dimension mismatch: saved={saved_config.n_input}, "
                f"current={self.config.n_input}"
            )

        # Validate n_actions (number of actions, not total neurons)
        saved_n_actions = state.get("n_actions", saved_config.n_output)  # Fallback for old checkpoints
        if saved_n_actions != self.n_actions:
            raise ValueError(
                f"Action count mismatch: saved={saved_n_actions}, "
                f"current={self.n_actions}"
            )

        # Validate total neuron count (n_actions × neurons_per_action)
        if saved_config.n_output != self.config.n_output:
            raise ValueError(
                f"Output dimension mismatch: saved={saved_config.n_output}, "
                f"current={self.config.n_output}. "
                f"This usually means population coding settings differ "
                f"(saved: {saved_n_actions} actions, current: {self.n_actions} actions)"
            )

        # 1. RESTORE WEIGHTS
        weights = state["weights"]
        self.weights = weights["weights"].to(self.device)
        self.d1_weights = weights["d1_weights"].to(self.device)
        self.d2_weights = weights["d2_weights"].to(self.device)

        # 2. RESTORE REGION STATE
        region_state = state["region_state"]
        self.recent_spikes = region_state["recent_spikes"].to(self.device)
        self.last_action = region_state["last_action"]

        # Restore neuron state for all populations
        neuron_state = region_state["neuron_state"]
        if "main" in neuron_state and self.neurons is not None:
            self.neurons.load_state(neuron_state["main"])
        if "d1" in neuron_state and hasattr(self, 'd1_neurons') and self.d1_neurons is not None:
            self.d1_neurons.load_state(neuron_state["d1"])
        if "d2" in neuron_state and hasattr(self, 'd2_neurons') and self.d2_neurons is not None:
            self.d2_neurons.load_state(neuron_state["d2"])

        # Restore base RegionState
        base_state = region_state["base_region_state"]
        if base_state["membrane"] is not None:
            self.state.membrane = base_state["membrane"].to(self.device)
        if base_state["spikes"] is not None:
            self.state.spikes = base_state["spikes"].to(self.device)
        self.state.t = base_state["t"]

        # 3. RESTORE LEARNING STATE
        learning_state = state["learning_state"]

        # Eligibility traces
        self.eligibility.traces = learning_state["eligibility_traces"].to(self.device)
        self.d1_eligibility = learning_state["d1_eligibility"].to(self.device)
        self.d2_eligibility = learning_state["d2_eligibility"].to(self.device)

        # D1/D2 spike traces
        self.d1_input_trace = learning_state["d1_input_trace"].to(self.device)
        self.d2_input_trace = learning_state["d2_input_trace"].to(self.device)
        self.d1_output_trace = learning_state["d1_output_trace"].to(self.device)
        self.d2_output_trace = learning_state["d2_output_trace"].to(self.device)

        # Trial accumulators
        self._d1_votes_accumulated = learning_state["d1_votes_accumulated"].to(self.device)
        self._d2_votes_accumulated = learning_state["d2_votes_accumulated"].to(self.device)

        # Homeostatic state
        self._activity_ema = learning_state["activity_ema"]
        self._trial_spike_count = learning_state["trial_spike_count"]
        self._trial_timesteps = learning_state["trial_timesteps"]
        self._homeostatic_scaling_applied = learning_state["homeostatic_scaling_applied"]

        # Unified homeostasis
        if learning_state["unified_homeostasis_state"] is not None and self.unified_homeostasis is not None:
            self.unified_homeostasis.load_state(learning_state["unified_homeostasis_state"])

        # 4. RESTORE EXPLORATION STATE
        exploration_state = state["exploration_state"]
        self.exploring = exploration_state["exploring"]
        self._last_uncertainty = exploration_state["last_uncertainty"]
        self._last_exploration_prob = exploration_state["last_exploration_prob"]
        self._action_counts = exploration_state["action_counts"].to(self.device)
        self._total_trials = exploration_state["total_trials"]
        self._recent_rewards = list(exploration_state["recent_rewards"])
        self._recent_accuracy = exploration_state["recent_accuracy"]

        # 5. RESTORE RPE STATE
        rpe_state = state["rpe_state"]
        if rpe_state and self.value_estimates is not None:
            self.value_estimates = rpe_state["value_estimates"].to(self.device)
            self._last_rpe = rpe_state["last_rpe"]
            self._last_expected = rpe_state["last_expected"]

        # 6. RESTORE NEUROMODULATOR STATE
        neuromodulator_state = state["neuromodulator_state"]
        self.state.dopamine = neuromodulator_state["dopamine"]
        self.state.acetylcholine = neuromodulator_state["acetylcholine"]
        self.state.norepinephrine = neuromodulator_state["norepinephrine"]
        self.tonic_dopamine = neuromodulator_state["tonic_dopamine"]

        # 7. OSCILLATOR STATE (none for striatum, but included for consistency)
        # oscillator_state = state["oscillator_state"]  # Empty for striatum
