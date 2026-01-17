"""
Striatum Homeostasis Component

**Scope**: Region-specific integration of homeostatic mechanisms for striatum
**Focus**: Coordinates D1/D2 pathway stability using synaptic homeostasis

Manages homeostatic regulation for D1/D2 opponent pathways.
Standardized component following the region_components pattern.

Related Homeostasis Modules:
=============================
This component INTEGRATES multiple global homeostasis mechanisms:

1. **This component** (striatum/homeostasis_component.py):
   - Striatum-specific coordination of homeostasis
   - D1/D2 opponent pathway balance
   - Uses synaptic_homeostasis.py for weight normalization

2. **Global mechanisms available** (not all used here):
   - **synaptic_homeostasis.py**: Weight normalization (USED by striatum)
   - **intrinsic_plasticity.py**: Threshold adaptation (available)
   - **metabolic.py**: Energy budgets (available)
   - **neuromodulation/homeostasis.py**: Receptor sensitivity (managed globally)

**Architecture Pattern**:
Region-specific homeostasis components integrate global mechanisms:
- Hippocampus might use intrinsic_plasticity + synaptic_homeostasis
- Striatum uses synaptic_homeostasis (D1/D2 balance)
- Cortex might use all mechanisms

**When to Modify This Component**:
- Need striatum-specific homeostatic tuning
- Want to add metabolic constraints to action selection
- Require different homeostasis for D1 vs D2 pathways
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
import torch.nn as nn

from thalia.config.base import BaseConfig
from thalia.constants.learning import EMA_DECAY_FAST
from thalia.core.region_components import HomeostasisComponent
from thalia.learning.homeostasis.synaptic_homeostasis import (
    StriatumHomeostasis,
    UnifiedHomeostasisConfig,
)
from thalia.managers.base_manager import ManagerContext


@dataclass
class HomeostasisManagerConfig(BaseConfig):
    """Configuration for striatum homeostasis management.

    Extends UnifiedHomeostasisConfig with striatum-specific parameters.
    """

    unified_homeostasis_enabled: bool = True
    target_firing_rate_hz: float = 5.0
    weight_budget: float = 1.0
    normalization_rate: float = 0.1
    baseline_pressure_enabled: bool = True
    baseline_pressure_rate: float = 0.001
    baseline_target_net: float = 0.0
    w_min: float = 0.0
    w_max: float = 1.0
    activity_decay: float = EMA_DECAY_FAST  # EMA decay for activity tracking (~100 timestep window)


class StriatumHomeostasisComponent(HomeostasisComponent):
    """Manages homeostatic regulation for striatal D1/D2 pathways.

    This component implements constraint-based homeostasis to maintain stable
    D1/D2 competition and prevent pathological states (runaway D2 inhibition,
    weight saturation).

    Responsibilities:
    =================
    1. **Budget Enforcement**: Constrains total D1/D2 weight sums
    2. **Weight Normalization**: Keeps weights within biological bounds
    3. **Baseline Pressure**: Prevents D1-D2 imbalance from growing unbounded
    4. **Diagnostics**: Reports homeostatic state and violations

    Key Mechanisms:
    ===============
    - **Unified Homeostasis**: Uses constraint-based weight normalization
      (see thalia.learning.unified_homeostasis)
    - **Separate D1/D2 Budgets**: Independent weight budgets per pathway
    - **Baseline Pressure**: Soft constraint toward net_balance ≈ 0

    Biological Motivation:
    =====================
    In biological striatum, homeostatic mechanisms prevent runaway competition:
    - Synaptic scaling maintains stable firing rates
    - Intrinsic plasticity adjusts neuronal excitability
    - Inhibitory plasticity balances excitation/inhibition

    This component consolidates these mechanisms into constraint-based regulation.

    Usage:
    ======
        homeostasis = StriatumHomeostasisComponent(config, context)

        # Apply normalization during learning
        d1_weights = homeostasis.apply_homeostasis(
            d1_weights, pathway='d1', metrics=d1_metrics
        )
        d2_weights = homeostasis.apply_homeostasis(
            d2_weights, pathway='d2', metrics=d2_metrics
        )

        # Check for violations
        diagnostics = homeostasis.get_diagnostics()

    See Also:
    =========
    - `thalia.learning.unified_homeostasis.StriatumHomeostasis`
    - `docs/patterns/mixins.md` for component patterns
    - `docs/patterns/state-management.md` for state handling
    """

    def __init__(
        self,
        config: HomeostasisManagerConfig,
        context: ManagerContext,
    ):
        """Initialize striatum homeostasis component.

        Args:
            config: Homeostasis manager configuration
            context: Manager context (device, dimensions, etc.)
        """
        super().__init__(config, context)  # type: ignore[arg-type]

        # Type narrowing for mypy: config is HomeostasisManagerConfig
        self.config: HomeostasisManagerConfig = config  # type: ignore[assignment]

        # Extract dimensions from context
        self.n_actions = context.n_output if context.n_output else 1
        self.neurons_per_action = (
            context.metadata.get("neurons_per_action", 1) if context.metadata else 1
        )

        # Get D1/D2 sizes from metadata (computed in striatum.__init__)
        # These are the actual pathway sizes, NOT n_actions * neurons_per_action
        self.d1_size = context.metadata.get("d1_size", self.n_actions)
        self.d2_size = context.metadata.get("d2_size", self.n_actions)
        self.n_neurons = self.d1_size + self.d2_size  # Total neurons across both pathways
        self.activity_decay = config.activity_decay

        # Create unified homeostasis controller
        if config.unified_homeostasis_enabled:
            homeostasis_config = UnifiedHomeostasisConfig(
                weight_budget=1.0,  # Will be updated dynamically from actual weights
                w_min=config.w_min,
                w_max=config.w_max,
                normalization_rate=0.1,
                device=str(self.context.device),  # Convert device to str
            )

            # StriatumHomeostasis needs neurons per pathway (d1 or d2), not total
            # When neurons_per_action=4 with d1_d2_ratio=0.5, each pathway gets 2 neurons/action
            d1_neurons_per_action = self.d1_size // self.n_actions if self.n_actions > 0 else 1
            d2_neurons_per_action = self.d2_size // self.n_actions if self.n_actions > 0 else 1

            self.unified_homeostasis = StriatumHomeostasis(
                n_actions=self.n_actions,
                neurons_per_action=d1_neurons_per_action,  # D1 pathway size per action
                d2_neurons_per_action=d2_neurons_per_action,  # D2 pathway size per action
                config=homeostasis_config,
                target_rate=config.target_firing_rate_hz,
                excitability_tau=100.0,
            )
        else:
            self.unified_homeostasis = None  # type: ignore[assignment]

    def apply_homeostasis(
        self, d1_weights: nn.Parameter, d2_weights: nn.Parameter, **kwargs
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

    def compute_excitability(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute excitability gain factors for D1 and D2 pathways.

        Uses activity history to modulate neuronal excitability:
        - Neurons firing above target → lower excitability (harder to fire)
        - Neurons firing below target → higher excitability (easier to fire)

        This implements intrinsic plasticity / homeostatic threshold adaptation.

        Returns:
            Tuple of (d1_gain, d2_gain) tensors of shape [n_neurons].
            Values > 1.0 mean more excitable, < 1.0 mean less excitable.
            Falls back to neutral (1.0, 1.0) if unified homeostasis is disabled.
        """
        if self.unified_homeostasis is None:
            # Return scalar tensors with neutral gain
            device = self.context.device
            return (torch.tensor(1.0, device=device), torch.tensor(1.0, device=device))

        # Delegate to unified homeostasis controller
        return self.unified_homeostasis.compute_excitability()

    def update_activity(
        self, d1_spikes: torch.Tensor, d2_spikes: torch.Tensor, decay: float | None = None
    ) -> None:
        """Update activity tracking for homeostatic regulation.

        Tracks running average of D1/D2 firing rates to enable dynamic
        excitability modulation. Should be called every timestep during forward pass.

        Args:
            d1_spikes: D1 pathway spike output [batch, n_neurons] or [n_neurons]
            d2_spikes: D2 pathway spike output [batch, n_neurons] or [n_neurons]
            decay: Exponential decay for running average (default: use config.activity_decay)

        Note:
            Activity tracking enables compute_excitability() to return
            meaningful modulation factors based on recent firing history.
        """
        if self.unified_homeostasis is None:
            return

        # Use config decay if not explicitly provided
        decay_value = decay if decay is not None else self.activity_decay

        # Delegate to unified homeostasis controller
        self.unified_homeostasis.update_activity(d1_spikes, d2_spikes, decay=decay_value)

    def grow(self, n_new_d1: int, n_new_d2: int) -> None:
        """Grow the homeostasis component to support more D1 and D2 neurons.

        Args:
            n_new_d1: Number of new D1 neurons to add
            n_new_d2: Number of new D2 neurons to add
        """
        # Update internal neuron count (total)
        self.n_neurons += n_new_d1 + n_new_d2

        # Delegate to unified homeostasis to expand activity tracking for D1 and D2
        if self.unified_homeostasis is not None:
            self.unified_homeostasis.grow(n_new_d1, n_new_d2)

    def get_homeostasis_diagnostics(self) -> Dict[str, Any]:
        """Get homeostasis-specific diagnostics."""
        diag = super().get_homeostasis_diagnostics()
        diag.update(
            {
                "unified_homeostasis_enabled": self.unified_homeostasis is not None,
            }
        )
        return diag


__all__ = ["StriatumHomeostasisComponent", "HomeostasisManagerConfig"]
