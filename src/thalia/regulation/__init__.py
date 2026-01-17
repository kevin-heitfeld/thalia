"""
Regulation Constants.

Constants and utilities for homeostasis, learning, exploration, and normalization.

NOTE: Constants have been fully consolidated to thalia.constants module.
This file maintains exports for backward compatibility within the regulation module.
"""

from __future__ import annotations

from thalia.constants.exploration import (  # Exploration strategies
    DEFAULT_EPSILON_EXPLORATION,
    EPSILON_DECAY,
    EPSILON_MIN,
    SOFTMAX_TEMPERATURE_DEFAULT,
    SOFTMAX_TEMPERATURE_MAX,
    SOFTMAX_TEMPERATURE_MIN,
    UCB_CONFIDENCE_MULTIPLIER,
    UCB_MIN_VISITS,
)
from thalia.constants.homeostasis import (
    FIRING_RATE_WINDOW_MS,
    HOMEOSTATIC_TAU_FAST,
    HOMEOSTATIC_TAU_SLOW,
    HOMEOSTATIC_TAU_STANDARD,
    INTRINSIC_PLASTICITY_RATE,
    MAX_FIRING_RATE_HZ,
    MIN_FIRING_RATE_HZ,
    SYNAPTIC_SCALING_MAX,
    SYNAPTIC_SCALING_MIN,
    SYNAPTIC_SCALING_RATE,
    TARGET_FIRING_RATE_HIGH,
    TARGET_FIRING_RATE_LOW,
    TARGET_FIRING_RATE_MEDIUM,
    TARGET_FIRING_RATE_STANDARD,
)
from thalia.constants.learning import (  # Learning rates; STDP; BCM; Eligibility; Weight initialization; EMA; Working memory; Region-specific learning rates
    EMA_DECAY_FAST,
    EMA_DECAY_SLOW,
    LEARNING_RATE_BCM,
    LEARNING_RATE_DEFAULT,
    LEARNING_RATE_HEBBIAN,
    LEARNING_RATE_STDP,
    LR_CEREBELLUM_DEFAULT,
    LR_CORTEX_DEFAULT,
    LR_FAST,
    LR_HIPPOCAMPUS_DEFAULT,
    LR_MODERATE,
    LR_PFC_DEFAULT,
    LR_SLOW,
    LR_STRIATUM_DEFAULT,
    LR_VERY_SLOW,
    STDP_A_MINUS_CORTEX,
    STDP_A_MINUS_HIPPOCAMPUS,
    STDP_A_PLUS_CORTEX,
    STDP_A_PLUS_HIPPOCAMPUS,
    TAU_BCM_THRESHOLD,
    TAU_ELIGIBILITY_STANDARD,
    TAU_STDP_MINUS,
    TAU_STDP_PLUS,
    WEIGHT_INIT_SCALE_PREDICTIVE,
    WEIGHT_INIT_SCALE_RECURRENT,
    WM_NOISE_STD_DEFAULT,
)
from thalia.constants.oscillator import (  # Theta modulation; Hippocampal gating; Gamma attention; ACh suppression; Cortical gating; Prefrontal gating; Cerebellum gating; Neuromodulator interactions; Theta-gamma coupling; Striatal learning
    ACH_ENCODING_BOOST_BASE,
    ACH_ENCODING_BOOST_RANGE,
    ACH_RECURRENT_SUPPRESSION,
    ACH_RECURRENT_SUPPRESSION_MAX,
    ACH_RECURRENT_SUPPRESSION_MIN,
    ACH_THRESHOLD_FOR_SUPPRESSION,
    CA1_SPARSITY_RETRIEVAL_BOOST,
    CA3_CA1_ENCODING_SCALE,
    CA3_RECURRENT_GATE_MIN,
    CA3_RECURRENT_GATE_RANGE,
    CEREBELLUM_INPUT_BASE_GAIN,
    CEREBELLUM_INPUT_MODULATION_RANGE,
    DG_CA3_GATE_MIN,
    DG_CA3_GATE_RANGE,
    EC_CA3_GATE_MIN,
    EC_CA3_GATE_RANGE,
    GAMMA_ATTENTION_THRESHOLD,
    GAMMA_LEARNING_MODULATION_SCALE,
    L4_INPUT_ENCODING_SCALE,
    L23_RECURRENT_RETRIEVAL_SCALE,
    PFC_FEEDFORWARD_BASE_GAIN,
    PFC_FEEDFORWARD_MODULATION_RANGE,
    PFC_RECURRENT_BASE_GAIN,
    PFC_RECURRENT_MODULATION_RANGE,
    STRIATUM_PFC_MODULATION_LR,
    THETA_ENCODING_PHASE_SCALE,
    THETA_GAMMA_PHASE_DIFF_SIGMA,
    THETA_RETRIEVAL_PHASE_SCALE,
)
from thalia.regulation.normalization import (
    ContrastNormalization,
    DivisiveNormalization,
    DivisiveNormConfig,
    SpatialDivisiveNorm,
)

__all__ = [
    # Exploration Constants - Strategies
    "DEFAULT_EPSILON_EXPLORATION",
    "EPSILON_MIN",
    "EPSILON_DECAY",
    "UCB_CONFIDENCE_MULTIPLIER",
    "UCB_MIN_VISITS",
    "SOFTMAX_TEMPERATURE_DEFAULT",
    "SOFTMAX_TEMPERATURE_MIN",
    "SOFTMAX_TEMPERATURE_MAX",
    # Exploration Constants - Learning Rates
    "LR_VERY_SLOW",
    "LR_SLOW",
    "LR_MODERATE",
    "LR_FAST",
    "LR_CORTEX_DEFAULT",
    "LR_HIPPOCAMPUS_DEFAULT",
    "LR_STRIATUM_DEFAULT",
    "LR_CEREBELLUM_DEFAULT",
    "LR_PFC_DEFAULT",
    # Homeostasis Constants - Firing Rate Targets
    "TARGET_FIRING_RATE_STANDARD",
    "TARGET_FIRING_RATE_LOW",
    "TARGET_FIRING_RATE_MEDIUM",
    "TARGET_FIRING_RATE_HIGH",
    # Homeostasis Constants - Time Constants
    "HOMEOSTATIC_TAU_STANDARD",
    "HOMEOSTATIC_TAU_FAST",
    "HOMEOSTATIC_TAU_SLOW",
    # Homeostasis Constants - Scaling
    "SYNAPTIC_SCALING_RATE",
    "SYNAPTIC_SCALING_MIN",
    "SYNAPTIC_SCALING_MAX",
    "INTRINSIC_PLASTICITY_RATE",
    "FIRING_RATE_WINDOW_MS",
    # Homeostasis Constants - Rate Bounds
    "MIN_FIRING_RATE_HZ",
    "MAX_FIRING_RATE_HZ",
    # Learning Constants - Rates
    "LEARNING_RATE_DEFAULT",
    "LEARNING_RATE_STDP",
    "LEARNING_RATE_BCM",
    "LEARNING_RATE_HEBBIAN",
    # Learning Constants - Time Constants
    "TAU_ELIGIBILITY_STANDARD",
    "TAU_BCM_THRESHOLD",
    "TAU_STDP_PLUS",
    "TAU_STDP_MINUS",
    # Learning Constants - STDP Amplitudes
    "STDP_A_PLUS_CORTEX",
    "STDP_A_MINUS_CORTEX",
    "STDP_A_PLUS_HIPPOCAMPUS",
    "STDP_A_MINUS_HIPPOCAMPUS",
    # Learning Constants - Weight Initialization
    "WEIGHT_INIT_SCALE_PREDICTIVE",
    "WEIGHT_INIT_SCALE_RECURRENT",
    # Learning Constants - Activity Tracking
    "EMA_DECAY_FAST",
    "EMA_DECAY_SLOW",
    # Learning Constants - Noise Parameters
    "WM_NOISE_STD_DEFAULT",
    # Normalization Classes
    "DivisiveNormConfig",
    "DivisiveNormalization",
    "ContrastNormalization",
    "SpatialDivisiveNorm",
    # ACh suppression
    "ACH_RECURRENT_SUPPRESSION_MIN",
    "ACH_RECURRENT_SUPPRESSION_MAX",
    # Oscillator Constants - Theta modulation
    "THETA_ENCODING_PHASE_SCALE",
    "THETA_RETRIEVAL_PHASE_SCALE",
    # Oscillator Constants - Hippocampal gating
    "DG_CA3_GATE_MIN",
    "DG_CA3_GATE_RANGE",
    "EC_CA3_GATE_MIN",
    "EC_CA3_GATE_RANGE",
    "CA3_RECURRENT_GATE_MIN",
    "CA3_RECURRENT_GATE_RANGE",
    "CA3_CA1_ENCODING_SCALE",
    "CA1_SPARSITY_RETRIEVAL_BOOST",
    # Oscillator Constants - Cortical gating
    "L4_INPUT_ENCODING_SCALE",
    "L23_RECURRENT_RETRIEVAL_SCALE",
    # Oscillator Constants - Prefrontal gating
    "PFC_FEEDFORWARD_BASE_GAIN",
    "PFC_FEEDFORWARD_MODULATION_RANGE",
    "PFC_RECURRENT_BASE_GAIN",
    "PFC_RECURRENT_MODULATION_RANGE",
    # Oscillator Constants - Cerebellum gating
    "CEREBELLUM_INPUT_BASE_GAIN",
    "CEREBELLUM_INPUT_MODULATION_RANGE",
    # Oscillator Constants - Gamma attention
    "GAMMA_ATTENTION_THRESHOLD",
    "GAMMA_LEARNING_MODULATION_SCALE",
    # Oscillator Constants - Neuromodulator interactions
    "ACH_RECURRENT_SUPPRESSION",
    "ACH_THRESHOLD_FOR_SUPPRESSION",
    "ACH_ENCODING_BOOST_BASE",
    "ACH_ENCODING_BOOST_RANGE",
    # Oscillator Constants - Theta-gamma coupling
    "THETA_GAMMA_PHASE_DIFF_SIGMA",
    # Oscillator Constants - Striatal learning
    "STRIATUM_PFC_MODULATION_LR",
]
