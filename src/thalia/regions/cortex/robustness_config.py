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

from thalia.diagnostics.criticality import CriticalityConfig
from thalia.learning.ei_balance import EIBalanceConfig
from thalia.learning.homeostasis.metabolic import MetabolicConfig


@dataclass
class RobustnessConfig:
    """Cortex-specific robustness mechanisms.

    This config contains mechanisms NOT already handled by UnifiedHomeostasis:
    - E/I Balance: Critical for recurrent cortical stability
    - Criticality: Optional research/diagnostics tool
    - Metabolic: Optional sparse coding objective

    Note: The following are handled by UnifiedHomeostasis (base class):
    - Weight normalization (budget constraints)
    - Activity regulation (threshold adaptation)
    - Competitive dynamics (winner-take-all)

    Divisive normalization removed: ConductanceLIF neurons provide natural
    gain control via shunting inhibition, making explicit divisive norm redundant.

    **Recommended presets:**
    - minimal(): Just E/I balance (essential for recurrence)
      → Use for: Most cortical regions, minimal overhead

    - full(): All mechanisms enabled
      → Use for: Research, diagnostics, sparse coding goals

    **When to customize:**
    - Debugging: Disable all, enable E/I balance only
    - Sparse coding: Enable metabolic constraints
    - Research: Enable criticality monitoring for branching ratio

    Attributes:
        enable_ei_balance: Enable E/I balance regulation
            Maintains healthy ratio between excitation and inhibition.
            Critical for recurrent cortical circuits (prevents oscillations).

        enable_criticality: Enable criticality monitoring
            Tracks branching ratio, can correct toward critical state.
            More expensive, research/diagnostics use only.

        enable_metabolic: Enable metabolic constraints
            Penalizes excessive activity, encourages sparse coding.
            Useful when energy efficiency is an explicit goal.

        ei_balance: E/I balance configuration
        criticality: Criticality monitoring configuration
        metabolic: Metabolic constraint configuration
    """

    # Enable/disable flags
    enable_ei_balance: bool = True
    enable_criticality: bool = False  # Research/diagnostics only
    enable_metabolic: bool = False  # Sparse coding objective

    # Sub-configurations
    ei_balance: EIBalanceConfig = field(default_factory=EIBalanceConfig)
    criticality: CriticalityConfig = field(default_factory=CriticalityConfig)
    metabolic: MetabolicConfig = field(default_factory=MetabolicConfig)

    @classmethod
    def minimal(cls) -> RobustnessConfig:
        """Create minimal config with only essential mechanisms.

        Enables E/I balance only (critical for recurrent stability).

        Use cases:
        - Most cortical regions (default choice)
        - Quick prototyping and debugging
        - Minimal computational overhead
        - Essential recurrence stability without extras

        Performance impact: ~10-15% overhead vs no robustness
        """
        return cls(
            enable_ei_balance=True,  # Essential for recurrence
            enable_criticality=False,
            enable_metabolic=False,
        )

    @classmethod
    def full(cls) -> RobustnessConfig:
        """Create full config with ALL robustness mechanisms.

        Maximum robustness with all mechanisms enabled.

        Use cases:
        - Research exploring criticality dynamics
        - Sparse coding objectives (metabolic constraints)
        - Maximum diagnostics and monitoring

        Performance impact: ~20-30% overhead vs minimal
        """
        return cls(
            enable_ei_balance=True,
            enable_criticality=True,
            enable_metabolic=True,
        )

    def get_enabled_mechanisms(self) -> List[str]:
        """Get list of enabled mechanism names."""
        enabled: List[str] = []
        if self.enable_ei_balance:
            enabled.append("ei_balance")
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
            f"  Divisive Norm: {'ON' if hasattr(self, 'enable_divisive_norm') and self.enable_divisive_norm else 'OFF'}",
            f"  Intrinsic Plasticity: {'ON' if hasattr(self, 'enable_intrinsic_plasticity') and self.enable_intrinsic_plasticity else 'OFF'}",
            f"  Criticality: {'ON' if self.enable_criticality else 'OFF'}",
            f"  Metabolic: {'ON' if self.enable_metabolic else 'OFF'}",
        ]

        if self.enable_ei_balance:
            lines.append(f"    E/I target ratio: {self.ei_balance.target_ratio}")
        if (
            hasattr(self, "enable_divisive_norm")
            and self.enable_divisive_norm
            and hasattr(self, "divisive_norm")
        ):
            lines.append(f"    Divisive sigma: {self.divisive_norm.sigma}")  # type: ignore[attr-defined]
        if (
            hasattr(self, "enable_intrinsic_plasticity")
            and self.enable_intrinsic_plasticity
            and hasattr(self, "intrinsic_plasticity")
        ):
            lines.append(f"    IP target rate: {self.intrinsic_plasticity.target_rate}")  # type: ignore[attr-defined]
        if self.enable_criticality:
            lines.append(f"    Target branching: {self.criticality.target_branching}")
        if self.enable_metabolic:
            lines.append(f"    Energy budget: {self.metabolic.energy_budget}")

        return "\n".join(lines)
