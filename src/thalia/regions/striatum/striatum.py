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

from thalia.core.weight_init import WeightInitializer
from thalia.core.base_manager import ManagerContext
from thalia.core.neuron_constants import (
    V_THRESHOLD_STANDARD,
    V_RESET_STANDARD,
    E_LEAK,
    E_EXCITATORY,
    E_INHIBITORY,
    THETA_BASELINE_MIN,
    THETA_BASELINE_RANGE,
    THETA_CONTRAST_MIN,
    THETA_CONTRAST_RANGE,
    BASELINE_EXCITATION_SCALE,
    TONIC_D1_GAIN_SCALE,
    NE_GAIN_RANGE,
)
from thalia.regions.base import (
    NeuralComponent,
    RegionConfig,
    LearningRule,
)
from thalia.core.neuron import ConductanceLIF, ConductanceLIFConfig

from thalia.core.component_registry import register_region

from .config import StriatumConfig
from .eligibility import EligibilityTraces
from .action_selection import ActionSelectionMixin
from .pathway_base import StriatumPathwayConfig
from .d1_pathway import D1Pathway
from .d2_pathway import D2Pathway
from .homeostasis_manager import HomeostasisManager, HomeostasisManagerConfig
from .learning_manager import LearningManager
from .checkpoint_manager import CheckpointManager
from .state_tracker import StriatumStateTracker
from .forward_coordinator import ForwardPassCoordinator


@register_region(
    "striatum",
    description="Reinforcement learning via dopamine-modulated three-factor rule with D1/D2 opponent pathways",
    version="2.0",
    author="Thalia Project"
)
class Striatum(NeuralComponent, ActionSelectionMixin):
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

        # =====================================================================
        # STATE TRACKER - Temporal State Management
        # =====================================================================
        # Consolidates all temporal state: votes, spikes, trials, actions
        self.state_tracker = StriatumStateTracker(
            n_actions=self.n_actions,
            n_output=config.n_output,
            device=self.device,
        )

        # =====================================================================
        # EXPLORATION MANAGER (UCB + Adaptive Exploration)
        # =====================================================================
        # Centralized exploration management with UCB tracking and adaptive
        # tonic dopamine adjustment based on performance.
        from thalia.regions.striatum.exploration import (
            ExplorationManager,
            ExplorationConfig,
        )

        exploration_config = ExplorationConfig(
            ucb_exploration=self.striatum_config.ucb_exploration,
            ucb_coefficient=self.striatum_config.ucb_coefficient,
            adaptive_exploration=self.striatum_config.adaptive_exploration,
            performance_window=self.striatum_config.performance_window,
            min_tonic_dopamine=self.striatum_config.min_tonic_dopamine,
            max_tonic_dopamine=self.striatum_config.max_tonic_dopamine,
            tonic_modulates_exploration=self.striatum_config.tonic_modulates_exploration,
            tonic_exploration_scale=self.striatum_config.tonic_exploration_scale,
        )
        # Create manager context for exploration
        exploration_context = ManagerContext(
            device=self.device,
            n_output=self.n_actions,
            dt_ms=config.dt_ms,
        )
        self.exploration_manager = ExplorationManager(
            config=exploration_config,
            context=exploration_context,
            initial_tonic_dopamine=self.striatum_config.tonic_dopamine,
        )

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

        # =====================================================================
        # D1/D2 PATHWAYS - Separate MSN Populations
        # =====================================================================
        # Create pathway-specific configuration
        pathway_config = StriatumPathwayConfig(
            n_input=config.n_input,
            n_output=config.n_output,
            w_min=config.w_min,
            w_max=config.w_max,
            eligibility_tau_ms=self.striatum_config.eligibility_tau_ms,
            stdp_lr=self.striatum_config.stdp_lr,
            stdp_tau_ms=self.striatum_config.stdp_tau_ms,
            device=self.device,
        )

        # Create D1 and D2 pathways
        self.d1_pathway = D1Pathway(pathway_config)
        self.d2_pathway = D2Pathway(pathway_config)

        # Create manager context for learning
        learning_context = ManagerContext(
            device=self.device,
            n_input=config.n_input,
            n_output=config.n_output,
            dt_ms=config.dt_ms,
        )

        # Create learning manager
        self.learning_manager = LearningManager(
            config=self.striatum_config,
            context=learning_context,
            d1_pathway=self.d1_pathway,
            d2_pathway=self.d2_pathway,
        )

        # Property delegation for backward compatibility
        # Old code accesses self.d1_weights, self.d2_weights, etc.
        # These now delegate to pathway objects

        # =====================================================================
        # BACKWARD COMPATIBILITY PROPERTIES (DEPRECATED - use pathways directly)
        # =====================================================================
        # These properties maintain compatibility with old checkpoints and code

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
        # Now managed by exploration_manager (set during initialization above)

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

            homeostasis_config = HomeostasisManagerConfig(
                weight_budget=dynamic_budget,
                normalization_rate=self.striatum_config.homeostatic_rate,
                baseline_pressure_enabled=self.striatum_config.baseline_pressure_enabled,
                baseline_pressure_rate=self.striatum_config.baseline_pressure_rate,
                baseline_target_net=self.striatum_config.baseline_target_net,
                w_min=self.config.w_min,
                w_max=self.config.w_max,
                device=self.device,
            )
            # Create manager context for homeostasis
            homeostasis_context = ManagerContext(
                device=self.device,
                n_output=self.n_actions,
                dt_ms=config.dt_ms,
                metadata={"neurons_per_action": self.neurons_per_action},
            )
            self.homeostasis_manager = HomeostasisManager(
                config=homeostasis_config,
                context=homeostasis_context,
            )
        else:
            self.homeostasis_manager = None

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
        # Previous implementation: D1/D2 were weight matrices on the same neurons
        # Current implementation: D1/D2 are separate pathway objects with their own neurons
        #
        # Neurons are now managed by pathway objects (d1_pathway.neurons, d2_pathway.neurons)
        # For backward compatibility, we expose them as properties

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
        # CHECKPOINT MANAGER
        # =====================================================================
        # Handles state serialization/deserialization
        self.checkpoint_manager = CheckpointManager(self)

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

    # =========================================================================
    # EXPLORATION STATE PROPERTIES (Backward Compatibility)
    # =========================================================================
    # These properties delegate to ExplorationManager for backward compatibility.
    # External code may access _action_counts, tonic_dopamine, etc. directly.

    @property
    def _action_counts(self) -> torch.Tensor:
        """UCB action counts (delegates to exploration_manager)."""
        return self.exploration_manager._action_counts

    @property
    def _total_trials(self) -> int:
        """Total trial count (delegates to exploration_manager)."""
        return self.exploration_manager._total_trials

    @property
    def _recent_rewards(self) -> List[float]:
        """Recent reward history (delegates to exploration_manager)."""
        return self.exploration_manager._recent_rewards

    @property
    def _recent_accuracy(self) -> float:
        """Running accuracy estimate (delegates to exploration_manager)."""
        return self.exploration_manager._recent_accuracy

    @property
    def tonic_dopamine(self) -> float:
        """Current tonic dopamine level (delegates to exploration_manager)."""
        return self.exploration_manager.tonic_dopamine

    @tonic_dopamine.setter
    def tonic_dopamine(self, value: float) -> None:
        """Set tonic dopamine level (delegates to exploration_manager)."""
        self.exploration_manager.tonic_dopamine = value

    # =========================================================================
    # PATHWAY DELEGATION PROPERTIES
    # =========================================================================
    # These properties delegate to D1/D2 pathway objects for backward compatibility.
    # Old code accesses self.d1_weights, self.d2_weights, etc.

    @property
    def d1_weights(self) -> nn.Parameter:
        """D1 pathway weights (delegates to d1_pathway)."""
        return self.d1_pathway.weights

    @property
    def d2_weights(self) -> nn.Parameter:
        """D2 pathway weights (delegates to d2_pathway)."""
        return self.d2_pathway.weights

    @property
    def d1_eligibility(self) -> torch.Tensor:
        """D1 eligibility traces (delegates to d1_pathway)."""
        return self.d1_pathway.eligibility

    @d1_eligibility.setter
    def d1_eligibility(self, value: torch.Tensor) -> None:
        """Set D1 eligibility traces."""
        self.d1_pathway.eligibility = value

    @property
    def d2_eligibility(self) -> torch.Tensor:
        """D2 eligibility traces (delegates to d2_pathway)."""
        return self.d2_pathway.eligibility

    @d2_eligibility.setter
    def d2_eligibility(self, value: torch.Tensor) -> None:
        """Set D2 eligibility traces."""
        self.d2_pathway.eligibility = value

    @property
    def d1_neurons(self) -> ConductanceLIF:
        """D1 neuron population (delegates to d1_pathway)."""
        return self.d1_pathway.neurons

    @property
    def d2_neurons(self) -> ConductanceLIF:
        """D2 neuron population (delegates to d2_pathway)."""
        return self.d2_pathway.neurons

    @property
    def d1_input_trace(self) -> torch.Tensor:
        """D1 input STDP trace (delegates to d1_pathway)."""
        return self.d1_pathway.input_trace

    @d1_input_trace.setter
    def d1_input_trace(self, value: torch.Tensor) -> None:
        """Set D1 input STDP trace."""
        self.d1_pathway.input_trace = value

    @property
    def d2_input_trace(self) -> torch.Tensor:
        """D2 input STDP trace (delegates to d2_pathway)."""
        return self.d2_pathway.input_trace

    @d2_input_trace.setter
    def d2_input_trace(self, value: torch.Tensor) -> None:
        """Set D2 input STDP trace."""
        self.d2_pathway.input_trace = value

    @property
    def d1_output_trace(self) -> torch.Tensor:
        """D1 output STDP trace (delegates to d1_pathway)."""
        return self.d1_pathway.output_trace

    @d1_output_trace.setter
    def d1_output_trace(self, value: torch.Tensor) -> None:
        """Set D1 output STDP trace."""
        self.d1_pathway.output_trace = value

    @property
    def d2_output_trace(self) -> torch.Tensor:
        """D2 output STDP trace (delegates to d2_pathway)."""
        return self.d2_pathway.output_trace

    @d2_output_trace.setter
    def d2_output_trace(self, value: torch.Tensor) -> None:
        """Set D2 output STDP trace."""
        self.d2_pathway.output_trace = value

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
    ) -> float:
        """
        Evaluate state quality using learned action values.

        For Phase 2 model-based planning: predicts how good a simulated state is
        by computing the maximum Q-value (best action value) from that state.

        Uses existing value estimates to evaluate states during mental simulation.
        If goal-conditioning is enabled, modulates values based on PFC component of state.

        Biology: Striatum represents action values (Q-values) learned through
        dopaminergic reinforcement. During planning, these values can evaluate
        simulated future states (Daw et al., 2011).

        Args:
            state: State to evaluate [n_input] (1D, ADR-005)
                   Format: [cortex_l5 | hippocampus | pfc] (concatenated)

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

        # If goal conditioning enabled, extract PFC component from state and modulate values
        if (self.striatum_config.use_goal_conditioning and
            hasattr(self, 'pfc_modulation_d1') and
            self.pfc_modulation_d1 is not None):

            # Extract PFC component from concatenated state tensor
            # Format: [cortex_l5 | hippocampus | pfc]
            pfc_size = self.striatum_config.pfc_size
            pfc_goal_context = state[-pfc_size:]

            # Shape assertion: PFC goal context must match modulation matrix columns
            expected_pfc_size = self.pfc_modulation_d1.shape[1]
            actual_pfc_size = pfc_goal_context.shape[0] if pfc_goal_context.dim() == 1 else pfc_goal_context.shape[-1]
            assert actual_pfc_size == expected_pfc_size, \
                f"PFC goal context size mismatch: got {actual_pfc_size}, expected {expected_pfc_size}. " \
                f"pfc_modulation_d1 shape: {self.pfc_modulation_d1.shape}, pfc_goal_context shape: {pfc_goal_context.shape}. " \
                f"Check that striatum_config.pfc_size matches actual PFC output size in brain config."

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
        # 1. EXPAND D1 AND D2 WEIGHT MATRICES using base helper
        # =====================================================================
        # Use base class helper with striatum-specific scale (w_max * 0.2)
        self.d1_pathway.weights = self._expand_weights(
            current_weights=self.d1_pathway.weights,
            n_new=n_new_neurons,
            initialization=initialization,
            sparsity=sparsity,
            scale=self.config.w_max * 0.2,
        )
        self.d2_pathway.weights = self._expand_weights(
            current_weights=self.d2_pathway.weights,
            n_new=n_new_neurons,
            initialization=initialization,
            sparsity=sparsity,
            scale=self.config.w_max * 0.2,
        )

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
        # 3. EXPAND STATE TENSORS using base helper
        # =====================================================================
        # Build state dict for all 2D tensors [n_neurons, dim]
        state_2d = {
            'd1_eligibility': self.d1_eligibility,
            'd2_eligibility': self.d2_eligibility,
        }
        # Add eligibility object traces if present
        if hasattr(self, 'eligibility'):
            state_2d['eligibility_traces'] = self.eligibility.traces

        # Add TD-lambda traces if present
        if hasattr(self, 'td_lambda_d1') and self.td_lambda_d1 is not None:
            state_2d['td_lambda_d1_traces'] = self.td_lambda_d1.traces.traces
        if hasattr(self, 'td_lambda_d2') and self.td_lambda_d2 is not None:
            state_2d['td_lambda_d2_traces'] = self.td_lambda_d2.traces.traces

        # Expand all 2D state tensors at once
        expanded_2d = self._expand_state_tensors(state_2d, n_new_neurons)
        self.d1_eligibility = expanded_2d['d1_eligibility']
        self.d2_eligibility = expanded_2d['d2_eligibility']
        if hasattr(self, 'eligibility'):
            self.eligibility.traces = expanded_2d['eligibility_traces']
            self.eligibility.n_post = new_n_output
        if hasattr(self, 'td_lambda_d1') and self.td_lambda_d1 is not None:
            self.td_lambda_d1.traces.traces = expanded_2d['td_lambda_d1_traces']
            self.td_lambda_d1.traces.n_output = new_n_output
            self.td_lambda_d1.n_actions = self.n_actions
        if hasattr(self, 'td_lambda_d2') and self.td_lambda_d2 is not None:
            self.td_lambda_d2.traces.traces = expanded_2d['td_lambda_d2_traces']
            self.td_lambda_d2.traces.n_output = new_n_output
            self.td_lambda_d2.n_actions = self.n_actions

        # Build state dict for all 1D tensors [n_neurons]
        state_1d = {
            'd1_output_trace': self.d1_output_trace,
            'd2_output_trace': self.d2_output_trace,
        }
        if hasattr(self, 'recent_spikes'):
            state_1d['recent_spikes'] = self.recent_spikes

        # Expand all 1D state tensors at once
        expanded_1d = self._expand_state_tensors(state_1d, n_new_neurons)
        self.d1_output_trace = expanded_1d['d1_output_trace']
        self.d2_output_trace = expanded_1d['d2_output_trace']
        if hasattr(self, 'recent_spikes'):
            self.recent_spikes = expanded_1d['recent_spikes']
        else:
            # Initialize if it doesn't exist
            self.recent_spikes = torch.zeros(new_n_output, device=self.device)

        # =====================================================================
        # 4. EXPAND NEURON POPULATIONS using base helper
        # =====================================================================
        # Expand D1-MSN and D2-MSN neuron populations
        if hasattr(self, 'd1_neurons') and self.d1_neurons is not None:
            self.d1_neurons = self._recreate_neurons_with_state(
                neuron_factory=self._create_d1_neurons,
                old_n_output=old_n_output,
            )
        if hasattr(self, 'd2_neurons') and self.d2_neurons is not None:
            self.d2_neurons = self._recreate_neurons_with_state(
                neuron_factory=self._create_d2_neurons,
                old_n_output=old_n_output,
            )

        # =====================================================================
        # 5. UPDATE ACTION-RELATED TRACKING (1D per action, not per neuron)
        # =====================================================================
        # Expand D1/D2 vote accumulators [n_actions]
        self._d1_votes_accumulated = torch.cat([
            self._d1_votes_accumulated,
            torch.zeros(n_new, device=self.device)
        ], dim=0)
        self._d2_votes_accumulated = torch.cat([
            self._d2_votes_accumulated,
            torch.zeros(n_new, device=self.device)
        ], dim=0)

        # Expand exploration manager (handles action_counts and other exploration state)
        self.exploration_manager.grow(self.n_actions)

        # Expand homeostasis manager if it exists
        if self.homeostasis_manager is not None:
            self.homeostasis_manager.grow(n_new_neurons)

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

        # =====================================================================
        # 6. EXPAND PFC MODULATION WEIGHTS (using base helper)
        # =====================================================================
        # PFC modulation weights [n_output, pfc_size] need to expand for new neurons
        if hasattr(self, 'pfc_modulation_d1') and self.pfc_modulation_d1 is not None:
            self.pfc_modulation_d1 = self._expand_weights(
                current_weights=self.pfc_modulation_d1,
                n_new=n_new_neurons,
                initialization='sparse_random',
                sparsity=0.3,
                scale=1.0,  # Default scale for PFC modulation
            )

        if hasattr(self, 'pfc_modulation_d2') and self.pfc_modulation_d2 is not None:
            self.pfc_modulation_d2 = self._expand_weights(
                current_weights=self.pfc_modulation_d2,
                n_new=n_new_neurons,
                initialization='sparse_random',
                sparsity=0.3,
                scale=1.0,  # Default scale for PFC modulation
            )

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

        # Update traces and eligibility using pathway managers
        # Note: D1 and D2 pathways have separate trace managers
        self.d1_pathway._trace_manager.update_traces(input_1d, d1_output_1d, dt)
        self.d2_pathway._trace_manager.update_traces(input_1d, d2_output_1d, dt)

        # Compute STDP eligibility with pathway-specific learning rate scaling
        d1_eligibility_update = self.d1_pathway._trace_manager.compute_stdp_eligibility_separate_ltd(
            input_spikes=input_1d,
            output_spikes=d1_output_1d,
            weights=self.d1_weights,
            lr_scale=cfg.d1_lr_scale,
        )

        d2_eligibility_update = self.d2_pathway._trace_manager.compute_stdp_eligibility_separate_ltd(
            input_spikes=input_1d,
            output_spikes=d2_output_1d,
            weights=self.d2_weights,
            lr_scale=cfg.d2_lr_scale,
        )

        # Accumulate into eligibility traces with decay
        self.d1_pathway._trace_manager.accumulate_eligibility(d1_eligibility_update, dt)
        self.d2_pathway._trace_manager.accumulate_eligibility(d2_eligibility_update, dt)

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
            v_threshold=V_THRESHOLD_STANDARD,
            v_reset=V_RESET_STANDARD,
            E_L=E_LEAK,
            E_E=E_EXCITATORY,
            E_I=E_INHIBITORY,
            tau_E=5.0,
            tau_I=5.0,
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
            v_threshold=V_THRESHOLD_STANDARD,
            v_reset=V_RESET_STANDARD,
            E_L=E_LEAK,
            E_E=E_EXCITATORY,
            E_I=E_INHIBITORY,
            tau_E=5.0,
            tau_I=5.0,
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
            v_threshold=V_THRESHOLD_STANDARD,
            v_reset=V_RESET_STANDARD,
            E_L=E_LEAK,
            E_E=E_EXCITATORY,
            E_I=E_INHIBITORY,
            tau_E=5.0,
            tau_I=5.0,
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
                         Format: [cortex_l5 | hippocampus | pfc] (concatenated)
                         PFC component is extracted automatically for goal modulation.

        NOTE: Exploration is handled by finalize_action() at trial end, not per-timestep.
        NOTE: Theta modulation computed internally from self._theta_phase (set by Brain)
        NOTE: In event-driven mode, pfc_goal_context is extracted from the input_spikes
              (the PFC component of the concatenated pathway input)

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

        theta_baseline_mod = THETA_BASELINE_MIN + THETA_BASELINE_RANGE * encoding_mod  # 0.7-1.0 range
        theta_contrast_mod = THETA_CONTRAST_MIN + THETA_CONTRAST_RANGE * retrieval_mod  # 0.8-1.0 range

        # Baseline excitation modulated by theta phase
        baseline_exc = BASELINE_EXCITATION_SCALE * theta_baseline_mod  # 0.84-1.2 range

        # =====================================================================
        # TONIC DOPAMINE MODULATION OF D1 GAIN
        # =====================================================================
        # Tonic DA increases D1 pathway responsiveness (motivation, energy)
        # This is separate from phasic DA which drives learning
        d1_gain = 1.0
        d2_gain = 1.0
        if self.striatum_config.tonic_modulates_d1_gain:
            # Higher tonic DA → stronger D1 response
            tonic_factor = self.tonic_dopamine * TONIC_D1_GAIN_SCALE
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
        #
        # Extract PFC component from concatenated input if goal conditioning is enabled.
        # Input format: [cortex_l5 | hippocampus | pfc]
        # PFC is at the end, size determined by pfc_size config.
        pfc_goal_context = None
        if (self.striatum_config.use_goal_conditioning and
            self.pfc_modulation_d1 is not None):

            # Extract PFC component from end of input
            pfc_size = self.striatum_config.pfc_size
            if input_spikes.shape[0] >= pfc_size:
                pfc_goal_context = input_spikes[-pfc_size:]

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
        # This is a constraint-based approach (via HomeostasisManager) rather
        # than a correction-based approach (former IntrinsicPlasticity).
        if self.homeostasis_manager is not None:
            d1_exc_gain, d2_exc_gain = self.homeostasis_manager.compute_excitability()
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
        ne_gain = 1.0 + NE_GAIN_RANGE * ne_level
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
        if self.homeostasis_manager is not None:
            self.homeostasis_manager.update_activity(d1_spikes, d2_spikes)

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
        self.state_tracker.accumulate_votes(d1_votes, d2_votes)

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
        self.state_tracker.store_spikes_for_learning(d1_spikes, d2_spikes, pfc_goal_context)

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

        # Update recent spikes and trial activity via state_tracker
        self.state_tracker.update_recent_spikes(d1_spikes, decay=0.9)
        self.state_tracker.update_trial_activity(d1_spikes, d2_spikes)

        # Store D1 and D2 spikes for learning manager
        self.learning_manager.store_spikes(d1_spikes, d2_spikes)

        self.state.spikes = output_spikes
        # self.state.dopamine is set by Brain via set_dopamine(), no need to update here
        self.state.t += 1

        # Apply axonal delay (biological reality: ALL neural connections have delays)
        delayed_spikes = self._apply_axonal_delay(output_spikes, dt)

        return delayed_spikes

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

    # NOTE: The _apply_baseline_pressure() method has been moved to HomeostasisManager
    # to consolidate homeostatic regulation logic. The method is now accessed via
    # self.homeostasis_manager.apply_baseline_pressure()

    # NOTE: The learn() method and _three_factor_learn/_reward_modulated_stdp_learn
    # have been removed. With the continuous learning paradigm:
    # - forward() builds eligibility traces during activity
    # - deliver_reward() applies D1/D2 plasticity when dopamine arrives
    # This is biologically correct: striatum uses three-factor learning where
    # plasticity is gated by dopamine, not continuous like cortical STDP.

    def deliver_reward(self, reward: float) -> Dict[str, Any]:
        """Deliver reward signal and trigger D1/D2 learning.

        Delegates to LearningManager for dopamine-gated three-factor learning.

        Args:
            reward: Raw reward signal (for adaptive exploration tracking only)

        Returns:
            Metrics dict with dopamine level and weight changes.
        """
        # Use dopamine from Brain (VTA) - already computed and set via set_dopamine()
        da_level = self.state.dopamine

        # Store for diagnostics
        self._last_rpe = da_level
        self._last_expected = 0.0

        # Adjust exploration based on performance
        self.exploration_manager.adjust_tonic_dopamine(reward)

        # Delegate to learning manager
        goal_context = self._last_pfc_goal_context if hasattr(self, '_last_pfc_goal_context') else None

        # Skip learning if plasticity frozen
        if self._plasticity_frozen:
            return {
                "dopamine": da_level,
                "d1_ltp": 0.0,
                "d1_ltd": 0.0,
                "d2_ltp": 0.0,
                "d2_ltd": 0.0,
                "net_change": 0.0,
                "frozen": True,
            }

        return self.learning_manager.apply_dopamine_learning(da_level, goal_context)



    def deliver_counterfactual_reward(
        self,
        reward: float,
        action: int,
        counterfactual_scale: float = 0.5,
    ) -> Dict[str, Any]:
        """Apply learning for a counterfactual (imagined) action outcome.

        Delegates to LearningManager for model-based "what if" learning.

        Args:
            reward: The counterfactual reward (what WOULD have happened)
            action: The action to update (the one NOT taken)
            counterfactual_scale: How much to scale the learning (default 0.5)

        Returns:
            Metrics dict with weight changes
        """
        # Compute expected value for counterfactual action
        expected_cf = self.get_expected_value(action)

        # Update value estimate for counterfactual action
        if self.value_estimates is not None and 0 <= action < self.n_actions:
            cf_lr = self.striatum_config.rpe_learning_rate * counterfactual_scale
            self.value_estimates[action] = (
                self.value_estimates[action]
                + cf_lr * (reward - self.value_estimates[action])
            )

        # Delegate to learning manager
        return self.learning_manager.apply_counterfactual_learning(
            reward, action, expected_cf, counterfactual_scale
        )

    def reset_eligibility(self, action_only: bool = True) -> None:
        """Reset eligibility traces after learning is complete.

        Delegates to LearningManager for eligibility trace management.

        Args:
            action_only: If True, only reset chosen action. If False, reset all.
        """
        self.learning_manager.reset_eligibility(self.last_action, action_only)

        # Always reset the spike traces (these are per-timestep, not per-action)
        self.d1_input_trace.zero_()
        self.d2_input_trace.zero_()
        self.d1_output_trace.zero_()
        self.d2_output_trace.zero_()
        self.input_trace.zero_()
        self.output_trace.zero_()

    def reset_state(self) -> None:
        super().reset_state()

        # Reset state tracker (votes, recent spikes, trial stats, last action)
        self.state_tracker.reset_state()

        # Reset managers and subsystems
        self._reset_subsystems('eligibility', 'd1_neurons', 'd2_neurons')

        # Reset trace tensors (eligibility traces delegated to pathways)
        self._reset_tensors(
            'd1_input_trace', 'd2_input_trace',
            'd1_output_trace', 'd2_output_trace'
        )

        # Reset TD(λ) traces if enabled
        if self.td_lambda_d1 is not None:
            self.td_lambda_d1.reset_episode()
            self.td_lambda_d2.reset_episode()

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

        # Accumulated votes (current trial) - from state_tracker
        d1_votes, d2_votes = self.state_tracker.get_accumulated_votes()
        net_votes = self.state_tracker.get_net_votes()

        d1_votes_list = d1_votes.tolist()
        d2_votes_list = d2_votes.tolist()
        net_votes_list = net_votes.tolist()

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

        # Exploration state - from state_tracker
        exploration_state = {
            "exploring": self.state_tracker.exploring,
            "last_uncertainty": self.state_tracker._last_uncertainty,
            "last_exploration_prob": self.state_tracker._last_exploration_prob,
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

        # Custom metrics for striatum (per-action analysis, votes, etc.)
        custom = {
            "n_actions": self.n_actions,
            "neurons_per_action": self.neurons_per_action,
            "last_action": self.state_tracker.last_action,
            # Per-action state
            "d1_weight_means": d1_per_action,
            "d2_weight_means": d2_per_action,
            "net_weight_means": net_per_action,
            # Current trial votes
            "d1_votes": d1_votes_list,
            "d2_votes": d2_votes_list,
            "net_votes": net_votes_list,
            # Weight statistics (manual due to NET computation)
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
            # TD(λ) state
            "td_lambda": td_lambda_state,
        }

        # Use collect_standard_diagnostics for trace statistics
        return self.collect_standard_diagnostics(
            region_name="striatum",
            trace_tensors={
                "d1_elig": self.d1_eligibility,
                "d2_elig": self.d2_eligibility,
            },
            custom_metrics=custom,
        )

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
        # Delegate to checkpoint manager
        checkpoint_state = self.checkpoint_manager.get_full_state()

        # Add additional metadata for backward compatibility
        checkpoint_state.update({
            "config": self.striatum_config,
            "n_actions": self.n_actions,
            "neuromodulator_state": self.get_neuromodulator_state(),
            "oscillator_state": {},  # Empty for striatum
        })

        # Add tonic dopamine to neuromodulator state
        checkpoint_state["neuromodulator_state"]["tonic_dopamine"] = self.tonic_dopamine

        return checkpoint_state

    def load_full_state(self, state: Dict[str, Any]) -> None:
        """Restore complete state from checkpoint.

        Args:
            state: Dictionary returned by get_full_state()

        Raises:
            ValueError: If state is incompatible with current configuration
        """
        # Delegate to checkpoint manager
        self.checkpoint_manager.load_full_state(state)

        # Restore tonic dopamine if present in neuromodulator state
        if "neuromodulator_state" in state:
            if "tonic_dopamine" in state["neuromodulator_state"]:
                self.tonic_dopamine = state["neuromodulator_state"]["tonic_dopamine"]
