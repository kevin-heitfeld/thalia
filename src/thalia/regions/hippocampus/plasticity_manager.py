"""
Plasticity Manager for Trisynaptic Hippocampus.

Handles STDP learning, synaptic scaling, and intrinsic plasticity for CA3 recurrent connections.
"""

from typing import Optional, Dict, Any

import torch

from thalia.core.base_manager import BaseManager, ManagerContext
from thalia.core.utils import clamp_weights
from thalia.core.eligibility_utils import EligibilityTraceManager, STDPConfig
from thalia.regions.hippocampus.config import TrisynapticConfig, TrisynapticState


class PlasticityManager(BaseManager[TrisynapticConfig]):
    """Manages learning dynamics for hippocampal circuit weights.
    
    Responsibilities:
    - CA3 recurrent STDP learning
    - Synaptic scaling (homeostatic)
    - Intrinsic plasticity (threshold adaptation)
    """
    
    def __init__(
        self,
        config: TrisynapticConfig,
        context: ManagerContext,
    ):
        """Initialize plasticity manager.
        
        Args:
            config: Hippocampus configuration
            context: Manager context (device, dimensions, etc.)
        """
        super().__init__(config, context)
        
        # Extract CA3 size from context (stored in metadata)
        self.ca3_size = context.metadata.get("ca3_size", context.n_output) if context.metadata else context.n_output
        
        # STDP trace manager for CA3 recurrent plasticity
        self._trace_manager = EligibilityTraceManager(
            n_input=self.ca3_size,
            n_output=self.ca3_size,
            config=STDPConfig(
                stdp_tau_ms=config.stdp_tau_plus,  # Use tau_plus for both (symmetric)
                eligibility_tau_ms=1000.0,  # Not used in this context
                a_plus=1.0,  # Will scale by learning rate in apply_plasticity
                a_minus=0.5,  # Weaker LTD
                stdp_lr=1.0,  # Will apply learning rate in apply_plasticity
            ),
            device=self.context.device,
        )
        
        # Intrinsic plasticity state
        self._ca3_activity_history: Optional[torch.Tensor] = None
        self._ca3_threshold_offset: Optional[torch.Tensor] = None
        
    def apply_plasticity(
        self,
        state: TrisynapticState,
        w_ca3_ca3: torch.nn.Parameter,
        effective_learning_rate: float,
    ) -> None:
        """Apply continuous STDP learning to CA3 recurrent weights.
        
        Args:
            state: Current hippocampus state (contains spikes and traces)
            w_ca3_ca3: CA3 recurrent weight matrix to update
            effective_learning_rate: Dopamine-modulated learning rate
        """
        if effective_learning_rate < 1e-8:
            return
            
        cfg = self.config
        
        # CA3 recurrent STDP: strengthen connections between co-active neurons
        if state.ca3_spikes is not None:
            ca3_spikes = state.ca3_spikes.squeeze()
            
            # Update traces and compute LTP/LTD using trace manager
            self._trace_manager.update_traces(ca3_spikes, ca3_spikes, cfg.dt_ms)
            ltp, ltd = self._trace_manager.compute_ltp_ltd_separate(ca3_spikes, ca3_spikes)
            
            # Compute weight change
            dW = effective_learning_rate * (ltp - ltd) if isinstance(ltp, torch.Tensor) or isinstance(ltd, torch.Tensor) else 0

            if isinstance(dW, torch.Tensor):
                with torch.no_grad():
                    w_ca3_ca3.data += dW
                    w_ca3_ca3.data.fill_diagonal_(0.0)  # No self-connections
                    clamp_weights(w_ca3_ca3.data, cfg.w_min, cfg.w_max)

                # =============================================================
                # SYNAPTIC SCALING (Homeostatic)
                # =============================================================
                # Multiplicatively adjust all weights towards target mean.
                # This prevents runaway LTP from causing weight explosion.
                if cfg.synaptic_scaling_enabled:
                    mean_weight = w_ca3_ca3.data.mean()
                    scaling = 1.0 + cfg.synaptic_scaling_rate * (cfg.synaptic_scaling_target - mean_weight)
                    w_ca3_ca3.data *= scaling
                    w_ca3_ca3.data.fill_diagonal_(0.0)  # Maintain no self-connections
                    clamp_weights(w_ca3_ca3.data, cfg.w_min, cfg.w_max)
                    
    def apply_intrinsic_plasticity(
        self,
        ca3_spikes: torch.Tensor,
    ) -> torch.Tensor:
        """Update per-neuron threshold offsets based on firing history.
        
        This operates on LONGER timescales than spike-frequency adaptation.
        
        Args:
            ca3_spikes: Current CA3 spike pattern [ca3_size] or [batch, ca3_size]
            
        Returns:
            ca3_threshold_offset: Per-neuron threshold adjustments [ca3_size]
        """
        if not self.config.intrinsic_plasticity_enabled:
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

        # Update activity history (exponential moving average) - in-place to avoid memory leak
        self._ca3_activity_history.mul_(0.99).add_(ca3_spikes_1d, alpha=0.01)

        # Adjust threshold: high activity → higher threshold (less excitable) - in-place
        rate_error = self._ca3_activity_history - cfg.intrinsic_target_rate
        self._ca3_threshold_offset.add_(rate_error, alpha=cfg.intrinsic_adaptation_rate)
        self._ca3_threshold_offset.clamp_(-0.5, 0.5)  # Limit threshold adjustment range
        
        return self._ca3_threshold_offset
        
    def reset_state(self) -> None:
        """Reset plasticity state (for new trial)."""
        self._trace_manager.reset_traces()
        self._ca3_activity_history = None
        self._ca3_threshold_offset = None
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get diagnostic metrics for monitoring.
        
        Returns:
            Dict with plasticity-related metrics
        """
        diagnostics = {}
        
        if self._ca3_activity_history is not None:
            diagnostics["ca3_activity_mean"] = self._ca3_activity_history.mean().item()
            diagnostics["ca3_activity_std"] = self._ca3_activity_history.std().item()
        
        if self._ca3_threshold_offset is not None:
            diagnostics["ca3_threshold_offset_mean"] = self._ca3_threshold_offset.mean().item()
            diagnostics["ca3_threshold_offset_std"] = self._ca3_threshold_offset.std().item()
        
        return diagnostics
    
    def to(self, device: torch.device) -> "PlasticityManager":
        """Move all tensors to specified device.
        
        Args:
            device: Target device
            
        Returns:
            Self for chaining
        """
        self.context.device = device
        
        if self._ca3_activity_history is not None:
            self._ca3_activity_history = self._ca3_activity_history.to(device)
        if self._ca3_threshold_offset is not None:
            self._ca3_threshold_offset = self._ca3_threshold_offset.to(device)
        
        # Trace manager will handle its own tensors
        self._trace_manager.to(device)
        
        return self
