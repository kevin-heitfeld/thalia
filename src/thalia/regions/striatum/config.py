"""
Striatum configuration module.

Contains StriatumConfig with all configurable parameters for the striatal
reinforcement learning system.
"""

from __future__ import annotations

from dataclasses import dataclass

from thalia.core.base.component_config import NeuralComponentConfig
from thalia.regulation.learning_constants import EMA_DECAY_FAST


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
    neurons_per_action: int = 10

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
