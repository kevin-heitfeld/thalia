"""
Hippocampus Learning Component

Manages STDP plasticity, synaptic scaling, and intrinsic plasticity for CA3 recurrent connections.
Standardized component following the region_components pattern.
"""

from __future__ import annotations

from typing import Optional, Dict, Any, TYPE_CHECKING

import torch
import torch.nn as nn

from thalia.core.region_components import LearningComponent
from thalia.core.base_manager import ManagerContext
from thalia.core.utils import clamp_weights
from thalia.core.eligibility_utils import EligibilityTraceManager, STDPConfig
from thalia.learning.unified_homeostasis import UnifiedHomeostasis, UnifiedHomeostasisConfig

if TYPE_CHECKING:
    from thalia.regions.hippocampus.config import HippocampusConfig, HippocampusState


class HippocampusLearningComponent(LearningComponent):
    """Manages STDP learning and homeostatic plasticity for hippocampus.

    Handles:
    1. CA3 recurrent STDP (spike-timing dependent plasticity)
    2. Synaptic scaling (homeostatic weight normalization)
    3. Intrinsic plasticity (threshold adaptation)
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
        self.ca3_size = context.metadata.get("ca3_size", context.n_output) if context.metadata else context.n_output

        # STDP trace manager for CA3 recurrent plasticity
        self._trace_manager = EligibilityTraceManager(
            n_input=self.ca3_size,
            n_output=self.ca3_size,
            config=STDPConfig(
                stdp_tau_ms=config.stdp_tau_plus,
                eligibility_tau_ms=1000.0,
                a_plus=1.0,
                a_minus=0.5,
                stdp_lr=1.0,
            ),
            device=self.context.device,
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
        **kwargs
    ) -> Dict[str, Any]:
        """Apply STDP learning to CA3 recurrent weights.

        Args:
            state: Current hippocampus state
            w_ca3_ca3: CA3 recurrent weight matrix
            effective_learning_rate: ACh-modulated learning rate
            *args, **kwargs: Additional parameters

        Returns:
            Dict with learning metrics
        """
        if effective_learning_rate < 1e-8:
            return {"learning_applied": False, "reason": "low_learning_rate"}

        cfg = self.config

        # CA3 recurrent STDP
        if state.ca3_spikes is not None:
            ca3_spikes = state.ca3_spikes.squeeze()

            # Update traces and compute LTP/LTD
            self._trace_manager.update_traces(ca3_spikes, ca3_spikes, cfg.dt_ms)
            ltp, ltd = self._trace_manager.compute_ltp_ltd_separate(ca3_spikes, ca3_spikes)

            # Compute weight change
            dW = effective_learning_rate * (ltp - ltd) if isinstance(ltp, torch.Tensor) or isinstance(ltd, torch.Tensor) else 0

            if isinstance(dW, torch.Tensor):
                with torch.no_grad():
                    w_ca3_ca3.data += dW
                    w_ca3_ca3.data.fill_diagonal_(0.0)  # No self-connections
                    clamp_weights(w_ca3_ca3.data, cfg.w_min, cfg.w_max)

                    # Synaptic scaling (homeostatic) using UnifiedHomeostasis
                    if cfg.homeostasis_enabled:
                        w_ca3_ca3.data = self.homeostasis.normalize_weights(w_ca3_ca3.data, dim=1)
                        w_ca3_ca3.data.fill_diagonal_(0.0)  # Maintain no self-connections

                return {
                    "learning_applied": True,
                    "mean_weight": w_ca3_ca3.data.mean().item(),
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
        if not self.config.homeostasis_enabled:
            if self._ca3_threshold_offset is None:
                self._ca3_threshold_offset = torch.zeros(self.ca3_size, device=self.context.device)
            return self._ca3_threshold_offset

        cfg = self.config
        ca3_spikes_1d = ca3_spikes.float().mean(dim=0) if ca3_spikes.dim() > 1 else ca3_spikes.float()

        # Initialize if needed
        if self._ca3_activity_history is None:
            self._ca3_activity_history = torch.zeros(self.ca3_size, device=self.context.device)
        if self._ca3_threshold_offset is None:
            self._ca3_threshold_offset = torch.zeros(self.ca3_size, device=self.context.device)

        # Update activity history (exponential moving average)
        self._ca3_activity_history.mul_(0.99).add_(ca3_spikes_1d, alpha=0.01)

        # Adjust threshold
        rate_error = self._ca3_activity_history - cfg.activity_target
        self._ca3_threshold_offset.add_(rate_error, alpha=cfg.normalization_rate)
        self._ca3_threshold_offset.clamp_(-0.5, 0.5)

        return self._ca3_threshold_offset

    def reset_state(self) -> None:
        """Reset learning component state."""
        self._trace_manager.reset_traces()
        self._ca3_activity_history = None
        self._ca3_threshold_offset = None

    def get_learning_diagnostics(self) -> Dict[str, Any]:
        """Get learning-specific diagnostics."""
        diag = super().get_learning_diagnostics()
        diag.update({
            "ca3_size": self.ca3_size,
            "intrinsic_plasticity_enabled": self.config.intrinsic_plasticity_enabled,
        })
        return diag


# Backwards compatibility alias
PlasticityManager = HippocampusLearningComponent
