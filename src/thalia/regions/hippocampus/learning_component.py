"""Hippocampus Learning Component

Manages homeostatic plasticity (synaptic scaling and intrinsic plasticity) for CA3.
CA3 recurrent learning uses one-shot Hebbian (theta-phase modulated) in forward().
Standardized component following the region_components pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.nn as nn

from thalia.constants.architecture import (
    ACTIVITY_HISTORY_DECAY,
    ACTIVITY_HISTORY_INCREMENT,
)
from thalia.core.diagnostics_keys import DiagnosticKeys as DK
from thalia.core.region_components import LearningComponent
from thalia.learning.homeostasis.synaptic_homeostasis import (
    UnifiedHomeostasis,
    UnifiedHomeostasisConfig,
)
from thalia.managers.base_manager import ManagerContext

if TYPE_CHECKING:
    from thalia.regions.hippocampus.config import HippocampusConfig, HippocampusState


class HippocampusLearningComponent(LearningComponent):
    """Manages homeostatic plasticity for hippocampus.

    CA3 recurrent learning uses one-shot Hebbian learning (theta-phase modulated)
    which happens directly in the forward() pass, not here.

    This component handles:
    1. Synaptic scaling (homeostatic weight normalization)
    2. Intrinsic plasticity (threshold adaptation)
    """

    def __init__(
        self,
        config: HippocampusConfig,
        context: ManagerContext,
    ):
        """Initialize hippocampus learning component.

        Args:
            config: Hippocampus configuration
            context: Manager context (device, dimensions, etc.)
        """
        super().__init__(config, context)

        # Extract CA3 size from context
        self.ca3_size = (
            context.metadata.get("ca3_size", context.n_output)
            if context.metadata
            else context.n_output
        )

        # Intrinsic plasticity state
        self._ca3_activity_history: Optional[torch.Tensor] = None
        self._ca3_threshold_offset: Optional[torch.Tensor] = None

        # Homeostasis for synaptic scaling
        homeostasis_config = UnifiedHomeostasisConfig(
            weight_budget=config.weight_budget * self.ca3_size,  # Total budget
            w_min=config.w_min,
            w_max=config.w_max,
            soft_normalization=config.soft_normalization,
            normalization_rate=config.normalization_rate,
            device=self.context.device,
        )
        self.homeostasis = UnifiedHomeostasis(homeostasis_config)

    def apply_learning(
        self,
        state: HippocampusState,
        w_ca3_ca3: nn.Parameter,
        effective_learning_rate: float,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        """Apply homeostatic normalization to CA3 recurrent weights.

        Note: CA3 recurrent learning (one-shot Hebbian) happens in forward().
        This component only handles homeostatic regulation (synaptic scaling).

        Args:
            state: Current hippocampus state
            w_ca3_ca3: CA3 recurrent weight matrix
            *args, **kwargs: Additional parameters

        Returns:
            Dict with learning metrics
        """
        cfg = self.config

        # Apply synaptic scaling (homeostatic normalization)
        if cfg.homeostasis_enabled:
            w_ca3_ca3.data = self.homeostasis.normalize_weights(w_ca3_ca3.data, dim=1)
            w_ca3_ca3.data.fill_diagonal_(0.0)  # Maintain no self-connections

            return {
                "learning_applied": True,
                DK.WEIGHT_MEAN: w_ca3_ca3.data.mean().item(),
            }

        return {"learning_applied": False}

    def apply_intrinsic_plasticity(
        self,
        ca3_spikes: torch.Tensor,
    ) -> torch.Tensor:
        """Update per-neuron threshold offsets based on firing history.

        Args:
            ca3_spikes: Current CA3 spike pattern

        Returns:
            Threshold offset tensor
        """
        # Early return if homeostasis disabled
        self._ca3_threshold_offset = self._init_tensor_if_needed(
            "_ca3_threshold_offset", (self.ca3_size,)
        )
        if not self.config.homeostasis_enabled:
            return self._ca3_threshold_offset

        cfg = self.config
        ca3_spikes_1d = (
            ca3_spikes.float().mean(dim=0) if ca3_spikes.dim() > 1 else ca3_spikes.float()
        )

        # Initialize activity history if needed
        self._ca3_activity_history = self._init_tensor_if_needed(
            "_ca3_activity_history", (self.ca3_size,)
        )

        # Update activity history (exponential moving average)
        self._update_activity_history(
            self._ca3_activity_history,
            ca3_spikes_1d,
            decay=ACTIVITY_HISTORY_DECAY,
            increment=ACTIVITY_HISTORY_INCREMENT,
        )

        # Adjust threshold based on rate error
        rate_error = self._ca3_activity_history - cfg.activity_target
        self._ca3_threshold_offset.add_(rate_error, alpha=cfg.normalization_rate)
        self._safe_clamp(self._ca3_threshold_offset, -0.5, 0.5)

        return self._ca3_threshold_offset

    def reset_state(self) -> None:
        """Reset learning component state."""
        self._trace_manager.reset_traces()
        self._ca3_activity_history = None
        self._ca3_threshold_offset = None

    def get_learning_diagnostics(self) -> Dict[str, Any]:
        """Get learning-specific diagnostics."""
        diag = super().get_learning_diagnostics()
        diag.update(
            {
                "ca3_size": self.ca3_size,
                "intrinsic_plasticity_enabled": self.config.intrinsic_plasticity_enabled,
            }
        )
        return diag
