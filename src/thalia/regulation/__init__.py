"""
Regulation Constants.

Constants and utilities for homeostasis, learning, and normalization.
"""

from thalia.regulation.homeostasis_constants import (
    TARGET_FIRING_RATE_STANDARD,
    TARGET_FIRING_RATE_LOW,
    TARGET_FIRING_RATE_MEDIUM,
    TARGET_FIRING_RATE_HIGH,
    HOMEOSTATIC_TAU_STANDARD,
    HOMEOSTATIC_TAU_FAST,
    HOMEOSTATIC_TAU_SLOW,
    SYNAPTIC_SCALING_RATE,
    SYNAPTIC_SCALING_MIN,
    SYNAPTIC_SCALING_MAX,
    INTRINSIC_PLASTICITY_RATE,
    FIRING_RATE_WINDOW_MS,
    MIN_FIRING_RATE_HZ,
    MAX_FIRING_RATE_HZ,
)
from thalia.regulation.learning_constants import (
    LEARNING_RATE_DEFAULT,
    LEARNING_RATE_STDP,
    LEARNING_RATE_BCM,
    LEARNING_RATE_HEBBIAN,
    TAU_ELIGIBILITY_STANDARD,
    TAU_BCM_THRESHOLD,
    TAU_STDP_PLUS,
    TAU_STDP_MINUS,
    STDP_A_PLUS_CORTEX,
    STDP_A_MINUS_CORTEX,
    STDP_A_PLUS_HIPPOCAMPUS,
    STDP_A_MINUS_HIPPOCAMPUS,
    WEIGHT_INIT_SCALE_PREDICTIVE,
    WEIGHT_INIT_SCALE_RECURRENT,
    EMA_DECAY_FAST,
    EMA_DECAY_SLOW,
    WM_NOISE_STD_DEFAULT,
)
from thalia.regulation.region_constants import (
    # Thalamus - Mode Switching
    THALAMUS_BURST_THRESHOLD,
    THALAMUS_TONIC_THRESHOLD,
    THALAMUS_BURST_SPIKE_COUNT,
    THALAMUS_BURST_GAIN,
    # Thalamus - Attention Gating
    THALAMUS_ALPHA_SUPPRESSION,
    THALAMUS_ALPHA_GATE_THRESHOLD,
    # Thalamus - TRN
    THALAMUS_TRN_RATIO,
    THALAMUS_TRN_INHIBITION,
    THALAMUS_TRN_RECURRENT,
    # Thalamus - Spatial Filtering
    THALAMUS_SPATIAL_FILTER_WIDTH,
    THALAMUS_CENTER_EXCITATION,
    THALAMUS_SURROUND_INHIBITION,
    # Thalamus - Relay
    THALAMUS_RELAY_STRENGTH,
    # Striatum - TD(λ) Learning
    STRIATUM_TD_LAMBDA,
    STRIATUM_GAMMA,
    STRIATUM_TD_MIN_TRACE,
    STRIATUM_TD_ACCUMULATING,
)
from thalia.regulation.region_architecture_constants import (
    # Hippocampus
    HIPPOCAMPUS_DG_EXPANSION_FACTOR,
    HIPPOCAMPUS_CA3_SIZE_RATIO,
    HIPPOCAMPUS_CA1_SIZE_RATIO,
    HIPPOCAMPUS_SPARSITY_TARGET,
    # Cortex
    CORTEX_L4_RATIO,
    CORTEX_L23_RATIO,
    CORTEX_L5_RATIO,
    CORTEX_L6_RATIO,
    # Striatum
    STRIATUM_NEURONS_PER_ACTION,
    STRIATUM_D1_D2_RATIO,
    # Prefrontal
    PFC_WM_CAPACITY_RATIO,
    # Cerebellum
    CEREBELLUM_GRANULE_EXPANSION,
    CEREBELLUM_PURKINJE_PER_DCN,
    # Metacognition
    METACOG_ABSTENTION_STAGE1,
    METACOG_ABSTENTION_STAGE2,
    METACOG_ABSTENTION_STAGE3,
    METACOG_ABSTENTION_STAGE4,
    METACOG_CALIBRATION_LR,
)
from thalia.regulation.normalization import (
    DivisiveNormConfig,
    DivisiveNormalization,
    ContrastNormalization,
    SpatialDivisiveNorm,
)
from thalia.regulation.oscillator_constants import (
    # Theta modulation
    THETA_ENCODING_PHASE_SCALE,
    THETA_RETRIEVAL_PHASE_SCALE,
    # Hippocampal gating
    DG_CA3_GATE_MIN,
    DG_CA3_GATE_RANGE,
    EC_CA3_GATE_MIN,
    EC_CA3_GATE_RANGE,
    CA3_RECURRENT_GATE_MIN,
    CA3_RECURRENT_GATE_RANGE,
    CA3_CA1_ENCODING_SCALE,
    CA1_SPARSITY_RETRIEVAL_BOOST,
    # Cortical gating
    L4_INPUT_ENCODING_SCALE,
    L23_RECURRENT_RETRIEVAL_SCALE,
    # Prefrontal gating
    PFC_FEEDFORWARD_BASE_GAIN,
    PFC_FEEDFORWARD_MODULATION_RANGE,
    PFC_RECURRENT_BASE_GAIN,
    PFC_RECURRENT_MODULATION_RANGE,
    # Cerebellum gating
    CEREBELLUM_INPUT_BASE_GAIN,
    CEREBELLUM_INPUT_MODULATION_RANGE,
    # Neuromodulator interactions
    ACH_RECURRENT_SUPPRESSION,
    ACH_THRESHOLD_FOR_SUPPRESSION,
    ACH_ENCODING_BOOST_BASE,
    ACH_ENCODING_BOOST_RANGE,
    # Gamma attention
    GAMMA_ATTENTION_THRESHOLD,
    GAMMA_LEARNING_MODULATION_SCALE,
    # Theta-gamma coupling
    THETA_GAMMA_PHASE_DIFF_SIGMA,
    # Striatal learning
    STRIATUM_PFC_MODULATION_LR,
)

__all__ = [
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
    # Region Constants - Thalamus Mode Switching
    "THALAMUS_BURST_THRESHOLD",
    "THALAMUS_TONIC_THRESHOLD",
    "THALAMUS_BURST_SPIKE_COUNT",
    "THALAMUS_BURST_GAIN",
    # Region Constants - Thalamus Attention
    "THALAMUS_ALPHA_SUPPRESSION",
    "THALAMUS_ALPHA_GATE_THRESHOLD",
    # Region Constants - Thalamus TRN
    "THALAMUS_TRN_RATIO",
    "THALAMUS_TRN_INHIBITION",
    "THALAMUS_TRN_RECURRENT",
    # Region Constants - Thalamus Spatial Filtering
    "THALAMUS_SPATIAL_FILTER_WIDTH",
    "THALAMUS_CENTER_EXCITATION",
    "THALAMUS_SURROUND_INHIBITION",
    # Region Constants - Thalamus Relay
    "THALAMUS_RELAY_STRENGTH",
    # Region Constants - Striatum TD(λ)
    "STRIATUM_TD_LAMBDA",
    "STRIATUM_GAMMA",
    "STRIATUM_TD_MIN_TRACE",
    "STRIATUM_TD_ACCUMULATING",
    # Region Architecture - Hippocampus
    "HIPPOCAMPUS_DG_EXPANSION_FACTOR",
    "HIPPOCAMPUS_CA3_SIZE_RATIO",
    "HIPPOCAMPUS_CA1_SIZE_RATIO",
    "HIPPOCAMPUS_SPARSITY_TARGET",
    # Region Architecture - Cortex
    "CORTEX_L4_RATIO",
    "CORTEX_L23_RATIO",
    "CORTEX_L5_RATIO",
    "CORTEX_L6_RATIO",
    # Region Architecture - Striatum
    "STRIATUM_NEURONS_PER_ACTION",
    "STRIATUM_D1_D2_RATIO",
    # Region Architecture - Prefrontal
    "PFC_WM_CAPACITY_RATIO",
    # Region Architecture - Cerebellum
    "CEREBELLUM_GRANULE_EXPANSION",
    "CEREBELLUM_PURKINJE_PER_DCN",
    # Region Architecture - Metacognition
    "METACOG_ABSTENTION_STAGE1",
    "METACOG_ABSTENTION_STAGE2",
    "METACOG_ABSTENTION_STAGE3",
    "METACOG_ABSTENTION_STAGE4",
    "METACOG_CALIBRATION_LR",
    # Normalization Classes
    "DivisiveNormConfig",
    "DivisiveNormalization",
    "ContrastNormalization",
    "SpatialDivisiveNorm",
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
    # Oscillator Constants - Neuromodulator interactions
    "ACH_RECURRENT_SUPPRESSION",
    "ACH_THRESHOLD_FOR_SUPPRESSION",
    "ACH_ENCODING_BOOST_BASE",
    "ACH_ENCODING_BOOST_RANGE",
    # Oscillator Constants - Gamma attention
    "GAMMA_ATTENTION_THRESHOLD",
    "GAMMA_LEARNING_MODULATION_SCALE",
    # Oscillator Constants - Theta-gamma coupling
    "THETA_GAMMA_PHASE_DIFF_SIGMA",
    # Oscillator Constants - Striatal learning
    "STRIATUM_PFC_MODULATION_LR",
]
