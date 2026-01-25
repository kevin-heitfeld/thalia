"""
Striatum State Tracker - Temporal State Management

This component manages all temporal state variables for the striatum,
extracted from the main Striatum class to consolidate state management
and improve code organization.

**Responsibilities:**
- Track D1/D2 vote accumulation across timesteps within a trial
- Maintain recent spike history for lateral inhibition
- Record trial activity statistics (spike counts, timestep counts)
- Store last action for temporal credit assignment
- Cache last spikes and goal context for learning updates
- Track exploration state and uncertainty
- Monitor reward prediction errors (RPE)
- Maintain homeostatic activity tracking (EMA)

**Used By:**
- `Striatum` (main region class)
- `ForwardPassCoordinator`: Updates recent_spikes and last spikes
- `LearningComponent`: Accesses last_action, RPE for credit assignment
- `ExplorationComponent`: Updates exploration state

**Coordinates With:**
- `ForwardPassCoordinator`: Provides recent spikes for lateral inhibition
- `LearningComponent`: Provides last_action and last spikes for learning
- `CheckpointManager`: Serializes/deserializes state variables
- `D1Pathway` and `D2Pathway`: Stores pathway-specific spikes

**Why Extracted:**
- Orthogonal concern: State tracking separate from forward/learning logic
- Clarity: Consolidates ~15 state variables previously scattered
- Single Responsibility: All temporal state management in one place
- Testability: Can test state tracking independently
- Growth support: Simplifies expanding to multi-timestep state when needed

**Key State Variables:**
- `_d1_votes_accumulated`, `_d2_votes_accumulated`: [n_actions] tensors
- `recent_spikes`: [n_output] tensor for lateral inhibition
- `last_action`: int or None, most recent action selection
- `_trial_spike_count`, `_trial_timesteps`: Trial activity metrics
- `exploring`: bool, whether last action was exploratory
- `_activity_ema`: float, exponential moving average of activity

Author: Thalia Project
Date: December 9, 2025 (extracted during striatum refactoring)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

from thalia.constants.neuromodulation import DA_BASELINE_STRIATUM
from thalia.core.region_state import BaseRegionState


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
    # FSI (FAST-SPIKING INTERNEURON) STATE
    # ========================================================================
    fsi_membrane: Optional[torch.Tensor] = None
    """FSI membrane potentials [n_fsi] for gap junction coupling."""

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
    # VALUE ESTIMATES REMOVED (Emergent from D1-D2 competition)
    # ========================================================================
    # value_estimates field removed - action values now emerge from D1-D2 weights
    # RPE tracking variables kept for compatibility with learning diagnostics

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
    # Per-source STP modules
    stp_modules_state: Dict[str, Dict[str, Optional[torch.Tensor]]] = field(default_factory=dict)
    """Per-source STP states. Keys are source-pathway names (e.g., 'cortex:l5_d1'),
    values are dicts with 'u' and 'x' tensors."""

    # Note: Neuromodulators (dopamine, acetylcholine, norepinephrine) are
    # inherited from BaseRegionState

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
            # RPE tracking (no value_estimates, just diagnostic fields)
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
            # Neuromodulators inherited from BaseRegionState
            "dopamine": self.dopamine,
            "acetylcholine": self.acetylcholine,
            "norepinephrine": self.norepinephrine,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], device: str = "cpu") -> StriatumState:
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
            # FSI state (backward compatible)
            fsi_membrane=to_device(data.get("fsi_membrane")),
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
            # RPE tracking (no value_estimates)
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
            # STP (unified format)
            stp_modules_state=data.get("stp_modules_state", {}),
            # Neuromodulators inherited from BaseRegionState
            # Striatum uses higher baseline for RL
            dopamine=data.get("dopamine", DA_BASELINE_STRIATUM),
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
        # Reset base state (spikes, membrane, neuromodulators)
        super().reset()

        # Striatum uses higher tonic dopamine than base
        # This matches config.tonic_dopamine for exploration/RL
        self.dopamine = DA_BASELINE_STRIATUM

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


class StriatumStateTracker:
    """Tracks temporal state variables for striatum.

    Consolidates:
    - Vote accumulation (D1/D2 across timesteps)
    - Recent spike tracking (for lateral inhibition)
    - Trial activity statistics
    - Last action for credit assignment
    - Last spikes for learning
    - Last goal context for learning
    """

    def __init__(
        self,
        n_actions: int,
        n_output: int,  # Total neurons (may be > n_actions with population coding)
        device: torch.device,
    ):
        """Initialize state tracker.

        Args:
            n_actions: Number of discrete actions
            n_output: Total number of output neurons
            device: Torch device for tensors
        """
        self.n_actions = n_actions
        self.n_output = n_output
        self.device = device

        # Vote accumulation for trial-level decision
        self._d1_votes_accumulated = torch.zeros(n_actions, device=device)
        self._d2_votes_accumulated = torch.zeros(n_actions, device=device)

        # Recent spikes for lateral inhibition
        self.recent_spikes = torch.zeros(n_output, device=device)

        # Trial activity statistics
        self._trial_spike_count = 0.0
        self._trial_timesteps = 0

        # Last action for credit assignment
        self.last_action: Optional[int] = None

        # Last spikes for learning (stored by forward pass)
        self._last_d1_spikes: Optional[torch.Tensor] = None
        self._last_d2_spikes: Optional[torch.Tensor] = None

        # Last goal context for learning
        self._last_pfc_goal_context: Optional[torch.Tensor] = None

        # Exploration tracking
        self.exploring = False
        self._last_uncertainty = 0.0
        self._last_exploration_prob = 0.0

        # RPE tracking
        self._last_rpe = 0.0
        self._last_expected = 0.0

        # Homeostatic tracking
        self._activity_ema = 0.5  # Exponential moving average of activity
        self._homeostatic_scaling_applied = False

    def reset_trial_votes(self) -> None:
        """Reset vote accumulators at start of trial."""
        self._d1_votes_accumulated.zero_()
        self._d2_votes_accumulated.zero_()

    def accumulate_votes(self, d1_votes: torch.Tensor, d2_votes: torch.Tensor) -> None:
        """Accumulate D1/D2 votes for this timestep.

        Args:
            d1_votes: D1 votes per action [n_actions]
            d2_votes: D2 votes per action [n_actions]
        """
        self._d1_votes_accumulated += d1_votes
        self._d2_votes_accumulated += d2_votes

    def get_net_votes(self) -> torch.Tensor:
        """Get net votes (D1 - D2) for all actions.

        Returns:
            Net votes per action [n_actions]
        """
        return self._d1_votes_accumulated - self._d2_votes_accumulated

    def get_accumulated_votes(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get accumulated D1 and D2 votes.

        Returns:
            (d1_votes, d2_votes) each [n_actions]
        """
        return self._d1_votes_accumulated.clone(), self._d2_votes_accumulated.clone()

    def update_recent_spikes(
        self, d1_spikes: torch.Tensor, d2_spikes: torch.Tensor, decay: float = 0.9
    ) -> None:
        """Update recent spike history with decay for both D1 and D2 pathways.

        Args:
            d1_spikes: Current D1 spikes [d1_size]
            d2_spikes: Current D2 spikes [d2_size]
            decay: Decay factor for exponential averaging
        """
        # Concatenate D1 and D2 spikes to form full MSN spike vector
        combined_spikes = torch.cat([d1_spikes, d2_spikes], dim=0)
        self.recent_spikes = self.recent_spikes.float() * decay + combined_spikes.float()

    def update_trial_activity(self, d1_spikes: torch.Tensor, d2_spikes: torch.Tensor) -> None:
        """Update trial activity statistics.

        Args:
            d1_spikes: D1 spikes this timestep
            d2_spikes: D2 spikes this timestep
        """
        self._trial_spike_count += d1_spikes.sum().item() + d2_spikes.sum().item()
        self._trial_timesteps += 1

    def get_trial_activity_rate(self) -> float:
        """Get average activity rate for current trial.

        Returns:
            Spikes per timestep per neuron
        """
        if self._trial_timesteps == 0:
            return 0.0
        return self._trial_spike_count / (self._trial_timesteps * self.n_output * 2)  # D1 + D2

    def store_spikes_for_learning(
        self,
        d1_spikes: torch.Tensor,
        d2_spikes: torch.Tensor,
        pfc_goal_context: Optional[torch.Tensor] = None,
    ) -> None:
        """Store spikes and goal context for learning.

        Args:
            d1_spikes: D1 spikes to store
            d2_spikes: D2 spikes to store
            pfc_goal_context: Optional PFC goal context
        """
        self._last_d1_spikes = d1_spikes.clone()
        self._last_d2_spikes = d2_spikes.clone()
        if pfc_goal_context is not None:
            self._last_pfc_goal_context = pfc_goal_context.clone()
        else:
            self._last_pfc_goal_context = None

    def get_last_spikes(self) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get last stored D1/D2 spikes.

        Returns:
            (last_d1_spikes, last_d2_spikes)
        """
        return self._last_d1_spikes, self._last_d2_spikes

    def get_last_goal_context(self) -> Optional[torch.Tensor]:
        """Get last stored PFC goal context.

        Returns:
            Last PFC goal context or None
        """
        return self._last_pfc_goal_context

    def set_last_action(self, action: int, exploring: bool = False) -> None:
        """Set last selected action.

        Args:
            action: Selected action index
            exploring: Whether this was an exploratory action
        """
        self.last_action = action
        self.exploring = exploring

    def update_exploration_stats(self, uncertainty: float, exploration_prob: float) -> None:
        """Update exploration statistics.

        Args:
            uncertainty: Current uncertainty value
            exploration_prob: Probability of exploration
        """
        self._last_uncertainty = uncertainty
        self._last_exploration_prob = exploration_prob

    def update_rpe_stats(self, rpe: float, expected_value: float) -> None:
        """Update RPE tracking statistics.

        Args:
            rpe: Reward prediction error
            expected_value: Expected value for action
        """
        self._last_rpe = rpe
        self._last_expected = expected_value

    def reset_state(self) -> None:
        """Reset all temporal state (for new episode)."""
        # Reset vote accumulators
        self._d1_votes_accumulated.zero_()
        self._d2_votes_accumulated.zero_()

        # Reset recent spikes
        self.recent_spikes.zero_()

        # Reset trial statistics
        self._trial_spike_count = 0.0
        self._trial_timesteps = 0

        # Reset action tracking
        self.last_action = None
        self.exploring = False

        # Reset learning storage
        self._last_d1_spikes = None
        self._last_d2_spikes = None
        self._last_pfc_goal_context = None

        # Reset statistics
        self._last_uncertainty = 0.0
        self._last_exploration_prob = 0.0
        self._last_rpe = 0.0
        self._last_expected = 0.0
        self._homeostatic_scaling_applied = False

    def grow(self, n_new_actions: int, n_new_neurons: int) -> None:
        """Grow state tracker to accommodate more actions and neurons.

        Args:
            n_new_actions: Number of new actions being added
            n_new_neurons: Number of new neurons being added to n_output
        """
        # Expand vote accumulators [n_actions]
        self._d1_votes_accumulated = torch.cat(
            [self._d1_votes_accumulated, torch.zeros(n_new_actions, device=self.device)], dim=0
        )
        self._d2_votes_accumulated = torch.cat(
            [self._d2_votes_accumulated, torch.zeros(n_new_actions, device=self.device)], dim=0
        )

        # Expand recent_spikes [n_output]
        self.recent_spikes = torch.cat(
            [self.recent_spikes, torch.zeros(n_new_neurons, device=self.device)], dim=0
        )

        # Update counts
        self.n_actions += n_new_actions
        self.n_output += n_new_neurons

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostics for state tracker.

        Returns:
            Dictionary of diagnostic information
        """
        d1_votes, d2_votes = self.get_accumulated_votes()
        net_votes = self.get_net_votes()

        return {
            "last_action": self.last_action,
            "exploring": self.exploring,
            "d1_votes_mean": d1_votes.mean().item(),
            "d2_votes_mean": d2_votes.mean().item(),
            "net_votes_mean": net_votes.mean().item(),
            "net_votes_std": net_votes.std().item(),
            "recent_spikes_mean": self.recent_spikes.mean().item(),
            "trial_activity_rate": self.get_trial_activity_rate(),
            "trial_timesteps": self._trial_timesteps,
            "last_uncertainty": self._last_uncertainty,
            "last_exploration_prob": self._last_exploration_prob,
            "last_rpe": self._last_rpe,
            "last_expected": self._last_expected,
        }
