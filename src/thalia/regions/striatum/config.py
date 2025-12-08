"""
Striatum configuration module.

Contains StriatumConfig with all configurable parameters for the striatal
reinforcement learning system.
"""

from __future__ import annotations

from dataclasses import dataclass

from thalia.regions.base import RegionConfig, LearningRule


@dataclass
class StriatumConfig(RegionConfig):
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

    # Eligibility trace parameters (biological: 500-2000ms)
    eligibility_tau_ms: float = 1000.0

    # Learning rate for homeostatic normalization
    learning_rate: float = 0.005

    # STDP learning rate for weight updates
    stdp_lr: float = 0.005

    # Action selection
    lateral_inhibition: bool = True
    inhibition_strength: float = 2.0

    # REWARD_MODULATED_STDP parameters
    # Uses D1/D2 eligibility traces: spike-timing correlations modulated by dopamine
    # Δw_d1 = d1_eligibility × dopamine (standard)
    # Δw_d2 = d2_eligibility × (-dopamine) (inverted)
    learning_rule: LearningRule = LearningRule.REWARD_MODULATED_STDP
    stdp_tau_ms: float = 20.0
    heterosynaptic_ratio: float = 0.3

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
    homeostatic_enabled: bool = True
    # NOTE: weight_budget is computed dynamically from initialized weights
    # to automatically adapt to any architecture (population_coding, n_input, etc.)
    homeostatic_soft: bool = True
    homeostatic_rate: float = 0.1

    # =========================================================================
    # HETEROSYNAPTIC COMPETITION
    # =========================================================================
    heterosynaptic_competition: bool = False
    competition_strength: float = 0.2

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
