"""
Striatum configuration module.

Contains StriatumConfig with all configurable parameters for the striatal
reinforcement learning system, and StriatumState for state management.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch

from thalia.core.base.component_config import NeuralComponentConfig
from thalia.core.region_state import BaseRegionState
from thalia.regulation.learning_constants import EMA_DECAY_FAST
from thalia.regulation.region_architecture_constants import (
    STRIATUM_NEURONS_PER_ACTION,
)


@dataclass
class StriatumConfig(NeuralComponentConfig):
    """Configuration specific to striatal regions.

    Key Features:
    =============
    1. THREE-FACTOR LEARNING: Δw = eligibility × dopamine
    2. D1/D2 OPPONENT PATHWAYS: Go/No-Go balance
    3. POPULATION CODING: Multiple neurons per action
    4. ADAPTIVE EXPLORATION: UCB + uncertainty-driven

    Note: Dopamine/RPE computation has been centralized at the Brain level
    (Brain acts as VTA). Striatum receives dopamine via set_dopamine().
    """

    # Learning rate for homeostatic normalization
    learning_rate: float = 0.005  # Region-specific override (5x base for faster RL updates)
    # Note: stdp_lr and tau_plus_ms/tau_minus_ms inherited from NeuralComponentConfig

    # Action selection
    lateral_inhibition: bool = True
    inhibition_strength: float = 2.0

    # =========================================================================
    # POPULATION CODING
    # =========================================================================
    population_coding: bool = True
    neurons_per_action: int = STRIATUM_NEURONS_PER_ACTION

    # =========================================================================
    # D1/D2 OPPONENT PATHWAYS
    # =========================================================================
    d1_lr_scale: float = 1.0
    d2_lr_scale: float = 1.0
    d1_da_sensitivity: float = 1.0
    d2_da_sensitivity: float = 1.0

    # =========================================================================
    # HOMEOSTATIC PLASTICITY
    # =========================================================================
    # NOTE: weight_budget is computed dynamically from initialized weights
    # to automatically adapt to any architecture (population_coding, n_input, etc.)
    homeostatic_soft: bool = True
    homeostatic_rate: float = 0.1
    activity_decay: float = EMA_DECAY_FAST  # EMA decay for activity tracking (~100 timestep window)

    # Note: heterosynaptic_competition and heterosynaptic_ratio inherited from base
    # Striatum-specific competition handled via baseline_pressure mechanism below

    # =========================================================================
    # BASELINE PRESSURE (drift towards balanced D1/D2)
    # =========================================================================
    baseline_pressure_enabled: bool = True
    baseline_pressure_rate: float = 0.015
    baseline_target_net: float = 0.0

    # =========================================================================
    # SOFTMAX ACTION SELECTION
    # =========================================================================
    softmax_action_selection: bool = True
    softmax_temperature: float = 2.0

    # =========================================================================
    # ADAPTIVE EXPLORATION (performance-based)
    # =========================================================================
    adaptive_exploration: bool = True
    performance_window: int = 10
    performance_exploration_scale: float = 0.3
    min_tonic_dopamine: float = 0.1
    max_tonic_dopamine: float = 0.5

    # =========================================================================
    # TD(λ) - MULTI-STEP CREDIT ASSIGNMENT (Phase 1 Enhancement)
    # =========================================================================
    use_td_lambda: bool = True  # Enable TD(λ) instead of basic TD(0) [DEFAULT: Enabled]
    td_lambda: float = 0.9  # Trace decay rate (0=TD(0), 0.9=~10 steps, 1.0=Monte Carlo)
    td_gamma: float = 0.99  # Discount factor for future rewards
    td_lambda_accumulating: bool = True  # Accumulating vs replacing traces

    # =========================================================================
    # UCB EXPLORATION BONUS
    # =========================================================================
    ucb_exploration: bool = True
    ucb_coefficient: float = 2.0

    # =========================================================================
    # UNCERTAINTY-DRIVEN EXPLORATION
    # =========================================================================
    uncertainty_temperature: float = 0.05
    min_exploration_boost: float = 0.05

    # =========================================================================
    # REWARD PREDICTION ERROR (RPE)
    # =========================================================================
    rpe_enabled: bool = True
    rpe_learning_rate: float = 0.1
    rpe_initial_value: float = 0.0

    # =========================================================================
    # TONIC vs PHASIC DOPAMINE
    # =========================================================================
    tonic_dopamine: float = 0.3
    tonic_modulates_d1_gain: bool = True
    tonic_d1_gain_scale: float = 0.5
    tonic_modulates_exploration: bool = True
    tonic_exploration_scale: float = 0.1

    # =========================================================================
    # BETA OSCILLATION MODULATION (Motor Control)
    # =========================================================================
    # Beta amplitude modulates D1/D2 balance for action maintenance vs switching
    # High beta → action persistence (D1 dominant, D2 suppressed)
    # Low beta → action flexibility (D2 effective, D1 reduced)
    beta_modulation_strength: float = 0.3  # [0, 1] - strength of beta influence

    # =========================================================================
    # GOAL-CONDITIONED VALUES (Phase 1 Week 2-3 Enhancement)
    # =========================================================================
    # Enable PFC goal context to modulate striatal action values
    # Biology: PFC → Striatum projections gate action selection by goal context
    use_goal_conditioning: bool = True  # Enable goal-conditioned value learning
    pfc_size: int = 128  # Size of PFC goal context input (must match PFC n_output)
    goal_modulation_strength: float = 0.5  # How strongly goals modulate values
    goal_modulation_lr: float = 0.001  # Learning rate for PFC → striatum weights

    # =========================================================================
    # D1/D2 PATHWAY DELAYS (Temporal Competition)
    # =========================================================================
    # Biological timing for opponent pathways creates temporal competition:
    # - D1 "Go" pathway: Striatum → GPi/SNr → Thalamus (~15-20ms total)
    #   Direct inhibition of GPi/SNr → disinhibits thalamus → facilitates action
    # - D2 "No-Go" pathway: Striatum → GPe → STN → GPi/SNr (~23-28ms total)
    #   Indirect route via GPe and STN → inhibits thalamus → suppresses action
    # - Key insight: D1 pathway is ~8ms FASTER than D2 pathway
    #   Creates temporal competition window where D1 "vote" arrives first,
    #   D2 "veto" arrives later. Explains action selection timing and impulsivity.
    d1_to_output_delay_ms: float = 15.0  # D1 direct pathway delay
    d2_to_output_delay_ms: float = 25.0  # D2 indirect pathway delay (slower!)

    # =========================================================================
    # SHORT-TERM PLASTICITY (STP)
    # =========================================================================
    # Biologically, different striatal input pathways have distinct STP properties:
    # - Cortex→MSNs: DEPRESSING (U=0.4) - prevents sustained cortical input from
    #   saturating striatum, enables novelty detection (fresh inputs get stronger)
    # - Thalamus→MSNs: WEAK FACILITATION (U=0.25) - phasic input amplification,
    #   balances phasic (thalamus) and tonic (cortex) command signals
    #
    # References:
    # - Charpier et al. (1999): Corticostriatal EPSPs
    # - Partridge et al. (2000): Synaptic plasticity in striatum
    # - Ding et al. (2008): Thalamostriatal facilitation
    stp_enabled: bool = True  # Enable STP by default
    # Note: STP types use presets from stp_presets.py ("corticostriatal", "thalamostriatal")

    # =========================================================================
    # ELASTIC TENSOR CHECKPOINT FORMAT (Phase 1 - Growth Support)
    # =========================================================================
    # Enable elastic tensor format for checkpoint-growth compatibility.
    # Pre-allocates tensors with reserved capacity to enable fast growth.
    # Biology: Analogous to neural reserve capacity in brain development.
    growth_enabled: bool = True  # Enable elastic tensor format
    reserve_capacity: float = 0.5  # Fraction of extra capacity (0.5 = 50% headroom)
    # Example: 10 neurons with reserve_capacity=0.5 → allocate 15 neurons worth of memory
    # Growth within reserved space requires no reallocation (fast)
    # Growth beyond capacity triggers reallocation with new headroom (slower)


# =====================================================================
# STRIATUM STATE (State Management Refactoring - Phase 3.2)
# =====================================================================

@dataclass
class StriatumState(BaseRegionState):
    """Complete state for Striatum region.

    Stores all striatal state including:
    - D1/D2 opponent pathway states
    - Vote accumulation for action selection
    - Action selection history
    - Exploration state
    - Value estimates and RPE tracking
    - Goal modulation weights
    - Delay buffers for temporal competition
    - Homeostatic tracking
    - Neuromodulator levels

    Design Notes:
    =============
    - D1/D2 pathway states are stored as opaque dicts (from pathway.get_state())
    - Delay buffers include circular buffer pointers
    - Optional features (RPE, goal conditioning) may be None if disabled
    - Neuromodulators are explicit fields (not from mixin)
    """

    STATE_VERSION: int = 1

    # ========================================================================
    # D1/D2 OPPONENT PATHWAY STATES
    # ========================================================================
    d1_pathway_state: Optional[Dict[str, Any]] = None
    """D1 'Go' pathway state from d1_pathway.get_state()."""

    d2_pathway_state: Optional[Dict[str, Any]] = None
    """D2 'No-Go' pathway state from d2_pathway.get_state()."""

    # ========================================================================
    # VOTE ACCUMULATION (Trial-based Action Selection)
    # ========================================================================
    d1_votes_accumulated: Optional[torch.Tensor] = None
    """Accumulated D1 votes [n_neurons] for current trial."""

    d2_votes_accumulated: Optional[torch.Tensor] = None
    """Accumulated D2 votes [n_neurons] for current trial."""

    # ========================================================================
    # ACTION SELECTION STATE
    # ========================================================================
    last_action: Optional[int] = None
    """Last selected action index (for credit assignment)."""

    recent_spikes: Optional[torch.Tensor] = None
    """Recent spike history [n_neurons] for lateral inhibition."""

    # ========================================================================
    # EXPLORATION STATE
    # ========================================================================
    exploring: bool = False
    """Flag indicating if currently exploring (not exploiting)."""

    last_uncertainty: Optional[float] = None
    """Last computed action uncertainty (for adaptive exploration)."""

    last_exploration_prob: Optional[float] = None
    """Last exploration probability (for diagnostics)."""

    exploration_manager_state: Optional[Dict[str, Any]] = None
    """Exploration manager state (action counts, uncertainties, etc.)."""

    # ========================================================================
    # VALUE ESTIMATES AND RPE (Reward Prediction Error)
    # ========================================================================
    value_estimates: Optional[torch.Tensor] = None
    """Learned value estimates [n_actions] (if RPE enabled)."""

    last_rpe: Optional[float] = None
    """Last computed reward prediction error."""

    last_expected: Optional[float] = None
    """Last expected value (for RPE computation)."""

    # ========================================================================
    # GOAL MODULATION (PFC → Striatum)
    # ========================================================================
    pfc_modulation_d1: Optional[torch.Tensor] = None
    """PFC goal modulation weights for D1 pathway [n_neurons, pfc_size]."""

    pfc_modulation_d2: Optional[torch.Tensor] = None
    """PFC goal modulation weights for D2 pathway [n_neurons, pfc_size]."""

    # ========================================================================
    # DELAY BUFFERS (Temporal Competition)
    # ========================================================================
    d1_delay_buffer: Optional[torch.Tensor] = None
    """D1 pathway delay buffer [delay_steps, n_neurons]."""

    d2_delay_buffer: Optional[torch.Tensor] = None
    """D2 pathway delay buffer [delay_steps, n_neurons]."""

    d1_delay_ptr: int = 0
    """Circular buffer pointer for D1 delay buffer."""

    d2_delay_ptr: int = 0
    """Circular buffer pointer for D2 delay buffer."""

    # ========================================================================
    # HOMEOSTATIC TRACKING
    # ========================================================================
    activity_ema: float = 0.0
    """Exponential moving average of activity (for homeostasis)."""

    trial_spike_count: int = 0
    """Total spikes in current trial."""

    trial_timesteps: int = 0
    """Number of timesteps in current trial."""

    homeostatic_scaling_applied: bool = False
    """Flag indicating if homeostatic scaling was applied."""

    homeostasis_manager_state: Optional[Dict[str, Any]] = None
    """Unified homeostasis manager state."""

    # ========================================================================
    # SHORT-TERM PLASTICITY (STP)
    # ========================================================================
    stp_corticostriatal_u: Optional[torch.Tensor] = None
    """STP release probability for corticostriatal pathway [n_pre, n_post]."""

    stp_corticostriatal_x: Optional[torch.Tensor] = None
    """STP available resources for corticostriatal pathway [n_pre, n_post]."""

    stp_thalamostriatal_u: Optional[torch.Tensor] = None
    """STP release probability for thalamostriatal pathway [n_pre, n_post]."""

    stp_thalamostriatal_x: Optional[torch.Tensor] = None
    """STP available resources for thalamostriatal pathway [n_pre, n_post]."""

    # ========================================================================
    # NEUROMODULATORS (explicit, not from mixin)
    # ========================================================================
    dopamine: float = 0.0
    acetylcholine: float = 0.0
    norepinephrine: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary.

        Returns:
            Dictionary containing all state fields.
        """
        return {
            "state_version": self.STATE_VERSION,

            # Base state (spikes, membrane from BaseRegionState)
            "spikes": self.spikes,
            "membrane": self.membrane,

            # D1/D2 pathways
            "d1_pathway_state": self.d1_pathway_state,
            "d2_pathway_state": self.d2_pathway_state,

            # Vote accumulation
            "d1_votes_accumulated": self.d1_votes_accumulated,
            "d2_votes_accumulated": self.d2_votes_accumulated,

            # Action selection
            "last_action": self.last_action,
            "recent_spikes": self.recent_spikes,

            # Exploration
            "exploring": self.exploring,
            "last_uncertainty": self.last_uncertainty,
            "last_exploration_prob": self.last_exploration_prob,
            "exploration_manager_state": self.exploration_manager_state,

            # Value/RPE
            "value_estimates": self.value_estimates,
            "last_rpe": self.last_rpe,
            "last_expected": self.last_expected,

            # Goal modulation
            "pfc_modulation_d1": self.pfc_modulation_d1,
            "pfc_modulation_d2": self.pfc_modulation_d2,

            # Delay buffers
            "d1_delay_buffer": self.d1_delay_buffer,
            "d2_delay_buffer": self.d2_delay_buffer,
            "d1_delay_ptr": self.d1_delay_ptr,
            "d2_delay_ptr": self.d2_delay_ptr,

            # Homeostasis
            "activity_ema": self.activity_ema,
            "trial_spike_count": self.trial_spike_count,
            "trial_timesteps": self.trial_timesteps,
            "homeostatic_scaling_applied": self.homeostatic_scaling_applied,
            "homeostasis_manager_state": self.homeostasis_manager_state,

            # STP
            "stp_corticostriatal_u": self.stp_corticostriatal_u,
            "stp_corticostriatal_x": self.stp_corticostriatal_x,
            "stp_thalamostriatal_u": self.stp_thalamostriatal_u,
            "stp_thalamostriatal_x": self.stp_thalamostriatal_x,

            # Neuromodulators
            "dopamine": self.dopamine,
            "acetylcholine": self.acetylcholine,
            "norepinephrine": self.norepinephrine,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], device: str = "cpu") -> "StriatumState":
        """Deserialize state from dictionary.

        Args:
            data: Dictionary from to_dict()
            device: Target device for tensors

        Returns:
            New StriatumState instance with data on specified device
        """
        # Helper function to transfer tensors
        def to_device(x):
            if x is not None and isinstance(x, torch.Tensor):
                return x.to(device)
            return x

        return cls(
            # Base state
            spikes=to_device(data.get("spikes")),
            membrane=to_device(data.get("membrane")),

            # D1/D2 pathways
            d1_pathway_state=data.get("d1_pathway_state"),
            d2_pathway_state=data.get("d2_pathway_state"),

            # Vote accumulation
            d1_votes_accumulated=to_device(data.get("d1_votes_accumulated")),
            d2_votes_accumulated=to_device(data.get("d2_votes_accumulated")),

            # Action selection
            last_action=data.get("last_action"),
            recent_spikes=to_device(data.get("recent_spikes")),

            # Exploration
            exploring=data.get("exploring", False),
            last_uncertainty=data.get("last_uncertainty"),
            last_exploration_prob=data.get("last_exploration_prob"),
            exploration_manager_state=data.get("exploration_manager_state"),

            # Value/RPE
            value_estimates=to_device(data.get("value_estimates")),
            last_rpe=data.get("last_rpe"),
            last_expected=data.get("last_expected"),

            # Goal modulation
            pfc_modulation_d1=to_device(data.get("pfc_modulation_d1")),
            pfc_modulation_d2=to_device(data.get("pfc_modulation_d2")),

            # Delay buffers
            d1_delay_buffer=to_device(data.get("d1_delay_buffer")),
            d2_delay_buffer=to_device(data.get("d2_delay_buffer")),
            d1_delay_ptr=data.get("d1_delay_ptr", 0),
            d2_delay_ptr=data.get("d2_delay_ptr", 0),

            # Homeostasis
            activity_ema=data.get("activity_ema", 0.0),
            trial_spike_count=data.get("trial_spike_count", 0),
            trial_timesteps=data.get("trial_timesteps", 0),
            homeostatic_scaling_applied=data.get("homeostatic_scaling_applied", False),
            homeostasis_manager_state=data.get("homeostasis_manager_state"),

            # STP
            stp_corticostriatal_u=to_device(data.get("stp_corticostriatal_u")),
            stp_corticostriatal_x=to_device(data.get("stp_corticostriatal_x")),
            stp_thalamostriatal_u=to_device(data.get("stp_thalamostriatal_u")),
            stp_thalamostriatal_x=to_device(data.get("stp_thalamostriatal_x")),

            # Neuromodulators
            dopamine=data.get("dopamine", 0.0),
            acetylcholine=data.get("acetylcholine", 0.0),
            norepinephrine=data.get("norepinephrine", 0.0),
        )

    def reset(self) -> None:
        """Reset state to initial conditions.

        Clears:
        - Spike history and membrane potentials
        - Vote accumulation
        - Action selection history
        - Exploration flags
        - Value/RPE tracking
        - Trial statistics
        - Neuromodulator levels

        Preserves:
        - Synaptic weights (not part of state)
        - Goal modulation weights (learned parameters)
        - Pathway configurations
        """
        # Reset base state
        self.spikes = None
        self.membrane = None

        # Reset pathways (delegate to pathway reset)
        # Note: Pathway states will be reset via pathway.reset() calls

        # Reset vote accumulation
        if self.d1_votes_accumulated is not None:
            self.d1_votes_accumulated.zero_()
        if self.d2_votes_accumulated is not None:
            self.d2_votes_accumulated.zero_()

        # Reset action selection
        self.last_action = None
        if self.recent_spikes is not None:
            self.recent_spikes.zero_()

        # Reset exploration
        self.exploring = False
        self.last_uncertainty = None
        self.last_exploration_prob = None

        # Reset value/RPE
        if self.value_estimates is not None:
            self.value_estimates.zero_()
        self.last_rpe = None
        self.last_expected = None

        # Reset delay buffers
        if self.d1_delay_buffer is not None:
            self.d1_delay_buffer.zero_()
        if self.d2_delay_buffer is not None:
            self.d2_delay_buffer.zero_()
        self.d1_delay_ptr = 0
        self.d2_delay_ptr = 0

        # Reset homeostasis tracking
        self.activity_ema = 0.0
        self.trial_spike_count = 0
        self.trial_timesteps = 0
        self.homeostatic_scaling_applied = False

        # Reset neuromodulators
        self.dopamine = 0.0
        self.acetylcholine = 0.0
        self.norepinephrine = 0.0
