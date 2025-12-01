"""
Learning rules: STDP, homeostatic mechanisms, and reward-modulated learning.
"""

from thalia.learning.stdp import STDP, STDPConfig, TripletSTDP, TripletSTDPConfig
from thalia.learning.homeostatic import (
    IntrinsicPlasticity,
    IntrinsicPlasticityConfig,
    SynapticScaling,
    SynapticScalingConfig,
)
from thalia.learning.reward import RewardModulatedSTDP, RSTDPConfig
from thalia.learning.modulation import (
    EligibilityTraces,
    DopamineSystem,
    TargetMapping,
    create_default_mapping,
    create_shuffled_mapping,
    create_reversed_mapping,
    apply_dopamine_modulated_update,
)
from thalia.learning.hebbian import (
    hebbian_update,
    synaptic_scaling,
    PredictiveCoding,
)
from thalia.learning.phase_homeostasis import (
    PhaseHomeostasis,
    update_bcm_threshold,
    update_homeostatic_excitability,
    update_homeostatic_excitability_step,
    update_homeostatic_conductance,
    update_homeostatic_conductance_step,
    update_homeostatic_conductance_bidirectional,
)
from thalia.learning.meta_homeostasis import (
    MetaHomeostasis,
    MetaHomeostasisConfig,
    compute_weight_diversity,
    compute_temporal_bias,
    compute_weight_saturation,
    compute_weight_change,
)

__all__ = [
    # STDP
    "STDP",
    "STDPConfig",
    "TripletSTDP",
    "TripletSTDPConfig",
    # Homeostatic
    "IntrinsicPlasticity",
    "IntrinsicPlasticityConfig",
    "SynapticScaling",
    "SynapticScalingConfig",
    # Reward-modulated
    "RewardModulatedSTDP",
    "RSTDPConfig",
    # Neuromodulation (dopamine, eligibility)
    "EligibilityTraces",
    "DopamineSystem",
    "TargetMapping",
    "create_default_mapping",
    "create_shuffled_mapping",
    "create_reversed_mapping",
    "apply_dopamine_modulated_update",
    # Hebbian learning
    "hebbian_update",
    "synaptic_scaling",
    "PredictiveCoding",
    # Phase homeostasis
    "PhaseHomeostasis",
    "update_bcm_threshold",
    "update_homeostatic_excitability",
    "update_homeostatic_excitability_step",
    "update_homeostatic_conductance",
    "update_homeostatic_conductance_step",
    "update_homeostatic_conductance_bidirectional",
    # Meta-learning
    "MetaHomeostasis",
    "MetaHomeostasisConfig",
    "compute_weight_diversity",
    "compute_temporal_bias",
    "compute_weight_saturation",
    "compute_weight_change",
]
