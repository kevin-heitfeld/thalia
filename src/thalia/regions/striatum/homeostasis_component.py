"""
Striatum Homeostasis Component  

Manages homeostatic regulation for D1/D2 opponent pathways.
Standardized component following the region_components pattern.
"""

from __future__ import annotations

from typing import Dict, Any, TYPE_CHECKING

import torch.nn as nn

from thalia.core.region_components import HomeostasisComponent
from thalia.core.base_manager import ManagerContext
from thalia.learning.unified_homeostasis import (
    StriatumHomeostasis,
    UnifiedHomeostasisConfig,
)

if TYPE_CHECKING:
    from thalia.regions.striatum.config import StriatumConfig


class StriatumHomeostasisComponent(HomeostasisComponent):
    """Manages homeostatic regulation for striatal D1/D2 pathways.

    Provides budget-constrained weight normalization and baseline pressure
    to maintain stable D1/D2 competition.
    """

    def __init__(
        self,
        config: StriatumConfig,
        context: ManagerContext,
    ):
        """Initialize striatum homeostasis component.

        Args:
            config: Striatum configuration
            context: Manager context (device, dimensions, etc.)
        """
        super().__init__(config, context)

        # Extract dimensions from context
        self.n_actions = context.n_output if context.n_output else 1
        self.neurons_per_action = context.metadata.get("neurons_per_action", 1) if context.metadata else 1
        self.n_neurons = self.n_actions * self.neurons_per_action

        # Create unified homeostasis controller
        if config.unified_homeostasis_enabled:
            homeostasis_config = UnifiedHomeostasisConfig(
                weight_budget=config.weight_budget,
                w_min=config.w_min,
                w_max=config.w_max,
                normalization_rate=0.1,
                device=self.context.device,
            )

            self.unified_homeostasis = StriatumHomeostasis(
                n_actions=self.n_actions,
                neurons_per_action=self.neurons_per_action,
                config=homeostasis_config,
                target_rate=config.target_firing_rate_hz,
                excitability_tau=100.0,
            )
        else:
            self.unified_homeostasis = None

    def apply_homeostasis(
        self,
        d1_weights: nn.Parameter,
        d2_weights: nn.Parameter,
        **kwargs
    ) -> Dict[str, Any]:
        """Apply homeostatic regulation to D1/D2 weights.

        Args:
            d1_weights: D1 pathway weights
            d2_weights: D2 pathway weights
            **kwargs: Additional parameters

        Returns:
            Dict with homeostasis metrics
        """
        if self.unified_homeostasis is None:
            return {"homeostasis_applied": False}

        # Apply unified homeostasis normalization
        d1_norm, d2_norm = self.unified_homeostasis.normalize_d1_d2(
            d1_weights.data,
            d2_weights.data,
            per_action=True,
        )

        # Update weights in-place
        d1_weights.data.copy_(d1_norm)
        d2_weights.data.copy_(d2_norm)

        return {
            "homeostasis_applied": True,
            "d1_weight_mean": d1_weights.mean().item(),
            "d2_weight_mean": d2_weights.mean().item(),
        }

    def get_homeostasis_diagnostics(self) -> Dict[str, Any]:
        """Get homeostasis-specific diagnostics."""
        diag = super().get_homeostasis_diagnostics()
        diag.update({
            "unified_homeostasis_enabled": self.unified_homeostasis is not None,
        })
        return diag


# Backwards compatibility alias
HomeostasisManager = StriatumHomeostasisComponent
