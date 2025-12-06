"""
RobustnessConfig - Configuration for hyperparameter robustness mechanisms.

This module provides a unified configuration for all robustness mechanisms:
- E/I Balance Regulation
- Divisive Normalization
- Intrinsic Plasticity
- Criticality Monitoring
- Metabolic Constraints

These mechanisms work together to make the network robust to hyperparameter
variation, similar to how biological brains maintain stable function despite
neural variability.

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from ..learning.ei_balance import EIBalanceConfig
from ..learning.intrinsic_plasticity import IntrinsicPlasticityConfig
from ..learning.metabolic import MetabolicConfig
from ..core.normalization import DivisiveNormConfig
from ..diagnostics.criticality import CriticalityConfig


@dataclass
class RobustnessConfig:
    """Unified configuration for robustness mechanisms.
    
    This config controls which robustness mechanisms are enabled and their
    parameters. Mechanisms can be enabled/disabled individually.
    
    **Ablation Study Results (Dec 2025):**
    - Divisive Norm: CRITICAL (+1080% variance without it)
    - E/I Balance: VALUABLE (+26% variance without it)
    - Intrinsic Plasticity: MINOR (15% adaptation loss)
    - Combined: SEVERE (+184% variance when all removed)
    
    **Recommended presets:**
    - minimal(): Just divisive norm (CRITICAL mechanism only)
      → Use for: Quick prototyping, minimal overhead
      
    - stable(): + E/I balance (recommended default)
      → Use for: Most experiments, good stability/performance balance
      
    - full(): All mechanisms enabled
      → Use for: Production, maximum robustness, research
    
    **When to customize:**
    - Debugging: Use minimal() and enable mechanisms one-by-one
    - Sparse coding: Enable metabolic constraints
    - Long training: Consider enabling intrinsic plasticity
    - Research: Enable criticality monitoring for branching ratio tracking
    
    Attributes:
        enable_ei_balance: Enable E/I balance regulation
            Maintains healthy ratio between excitation and inhibition.
            VALUABLE: Reduces variance by 26% (ablation study).
            
        enable_divisive_norm: Enable divisive normalization
            Provides automatic gain control, contrast invariance.
            CRITICAL: Prevents +1080% variance increase (ablation study).
            
        enable_intrinsic_plasticity: Enable threshold adaptation
            Neurons adjust their thresholds based on activity history.
            MINOR: 15% adaptation improvement (ablation study).
            
        enable_criticality: Enable criticality monitoring
            Tracks branching ratio, can correct toward critical state.
            More expensive, research/diagnostics use.
            
        enable_metabolic: Enable metabolic constraints
            Penalizes excessive activity, encourages efficiency.
            Useful for sparse coding goals.
            
        ei_balance: E/I balance configuration
        divisive_norm: Divisive normalization configuration
        intrinsic_plasticity: Intrinsic plasticity configuration
        criticality: Criticality monitoring configuration
        metabolic: Metabolic constraint configuration
    """
    # Enable/disable flags
    enable_ei_balance: bool = True
    enable_divisive_norm: bool = True
    enable_intrinsic_plasticity: bool = True
    enable_criticality: bool = False  # More expensive
    enable_metabolic: bool = False    # Optional
    
    # Sub-configurations
    ei_balance: EIBalanceConfig = field(default_factory=EIBalanceConfig)
    divisive_norm: DivisiveNormConfig = field(default_factory=DivisiveNormConfig)
    intrinsic_plasticity: IntrinsicPlasticityConfig = field(
        default_factory=IntrinsicPlasticityConfig
    )
    criticality: CriticalityConfig = field(default_factory=CriticalityConfig)
    metabolic: MetabolicConfig = field(default_factory=MetabolicConfig)
    
    @classmethod
    def minimal(cls) -> "RobustnessConfig":
        """Create minimal config with only CRITICAL mechanisms.
        
        Based on ablation study evidence:
        - Enables divisive norm (CRITICAL: prevents +1080% variance)
        - Disables all other mechanisms
        
        Use cases:
        - Quick prototyping and debugging
        - Understanding core dynamics without extra complexity
        - Fastest execution (minimal overhead)
        - May be unstable with strong inputs
        
        Performance impact: ~5-10% overhead vs no robustness
        """
        return cls(
            enable_ei_balance=False,
            enable_divisive_norm=True,  # CRITICAL mechanism
            enable_intrinsic_plasticity=False,
            enable_criticality=False,
            enable_metabolic=False,
        )
    
    @classmethod
    def stable(cls) -> "RobustnessConfig":
        """Create stable config with CRITICAL + VALUABLE mechanisms.
        
        **RECOMMENDED DEFAULT for most work.**
        
        Based on ablation study evidence:
        - Enables divisive norm (CRITICAL: prevents +1080% variance)
        - Enables E/I balance (VALUABLE: prevents +26% variance)
        - Combined effect: -55% CV reduction vs minimal
        
        Use cases:
        - Most experiments and research
        - Production systems requiring reliability
        - Good balance of stability and performance
        - Handles varied input strengths well
        
        Performance impact: ~15-20% overhead vs no robustness
        """
        return cls(
            enable_ei_balance=True,      # VALUABLE mechanism
            enable_divisive_norm=True,   # CRITICAL mechanism
            enable_intrinsic_plasticity=False,
            enable_criticality=False,
            enable_metabolic=False,
        )
    
    @classmethod
    def full(cls) -> "RobustnessConfig":
        """Create full config with ALL robustness mechanisms.
        
        Maximum robustness with all mechanisms enabled.
        
        Based on ablation study evidence:
        - All core mechanisms provide measurable benefit
        - Intrinsic plasticity: MINOR but may help in long contexts
        - Criticality monitoring: Useful for research/diagnostics
        - Metabolic: Optional, enables sparse coding
        
        Use cases:
        - Production systems requiring maximum stability
        - Research exploring all mechanisms
        - Long training runs (benefits from IP adaptation)
        - Sparse coding objectives (metabolic constraints)
        
        Performance impact: ~30-40% overhead vs no robustness
        """
        return cls(
            enable_ei_balance=True,
            enable_divisive_norm=True,
            enable_intrinsic_plasticity=True,
            enable_criticality=True,
            enable_metabolic=True,
        )
    
    def get_enabled_mechanisms(self) -> List[str]:
        """Get list of enabled mechanism names."""
        enabled: List[str] = []
        if self.enable_ei_balance:
            enabled.append("ei_balance")
        if self.enable_divisive_norm:
            enabled.append("divisive_norm")
        if self.enable_intrinsic_plasticity:
            enabled.append("intrinsic_plasticity")
        if self.enable_criticality:
            enabled.append("criticality")
        if self.enable_metabolic:
            enabled.append("metabolic")
        return enabled
    
    def summary(self) -> str:
        """Get a summary of the robustness configuration."""
        lines = [
            "Robustness Configuration:",
            f"  E/I Balance: {'ON' if self.enable_ei_balance else 'OFF'}",
            f"  Divisive Norm: {'ON' if self.enable_divisive_norm else 'OFF'}",
            f"  Intrinsic Plasticity: {'ON' if self.enable_intrinsic_plasticity else 'OFF'}",
            f"  Criticality: {'ON' if self.enable_criticality else 'OFF'}",
            f"  Metabolic: {'ON' if self.enable_metabolic else 'OFF'}",
        ]
        
        if self.enable_ei_balance:
            lines.append(f"    E/I target ratio: {self.ei_balance.target_ratio}")
        if self.enable_divisive_norm:
            lines.append(f"    Divisive sigma: {self.divisive_norm.sigma}")
        if self.enable_intrinsic_plasticity:
            lines.append(f"    IP target rate: {self.intrinsic_plasticity.target_rate}")
        if self.enable_criticality:
            lines.append(f"    Target branching: {self.criticality.target_branching}")
        if self.enable_metabolic:
            lines.append(f"    Energy budget: {self.metabolic.energy_budget}")
        
        return "\n".join(lines)
