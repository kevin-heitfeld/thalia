"""
Learning rules: STDP, homeostatic mechanisms, and reward-modulated learning.
"""

from thalia.learning.stdp import STDP, STDPConfig
from thalia.learning.homeostatic import (
    IntrinsicPlasticity, 
    IntrinsicPlasticityConfig,
    SynapticScaling,
    SynapticScalingConfig,
)
from thalia.learning.reward import RewardModulatedSTDP, RSTDPConfig

__all__ = [
    "STDP",
    "STDPConfig",
    "IntrinsicPlasticity",
    "IntrinsicPlasticityConfig", 
    "SynapticScaling",
    "SynapticScalingConfig",
    "RewardModulatedSTDP",
    "RSTDPConfig",
]
