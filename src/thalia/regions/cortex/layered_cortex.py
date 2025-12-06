"""
Layered Cortex - Multi-layer cortical microcircuit.

This implements a biologically realistic cortical column with distinct layers:

Architecture (based on canonical cortical microcircuit):
=========================================================

         Feedback from higher areas
                    │
                    ▼
    ┌───────────────────────────────────┐
    │          LAYER 2/3                │ ← Superficial pyramidal cells
    │   (Cortico-cortical output)       │ → To other cortical areas
    │   - Receives from L4              │ → Attention pathway target
    │   - Lateral recurrent connections │
    │   - Top-down feedback target      │
    └───────────────┬───────────────────┘
                    │
    ┌───────────────┴───────────────────┐
    │          LAYER 4                  │ ← Spiny stellate cells
    │   (Feedforward input layer)       │ ← From thalamus/lower areas
    │   - Main sensory input recipient  │
    │   - No recurrent connections      │
    │   - Fast, feedforward processing  │
    └───────────────┬───────────────────┘
                    │
    ┌───────────────┴───────────────────┐
    │          LAYER 5                  │ ← Deep pyramidal cells
    │   (Subcortical output layer)      │ → To striatum, brainstem, etc.
    │   - Receives from L2/3            │ → Motor/action-related output
    │   - Different output pathway      │
    │   - Burst-capable neurons         │
    └───────────────────────────────────┘

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from thalia.regions.base import BrainRegion, RegionConfig, LearningRule
from thalia.core.neuron import LIFNeuron, LIFConfig
from thalia.regions.theta_dynamics import FeedforwardInhibition
from thalia.learning.bcm import BCMRule, BCMConfig
from thalia.core.utils import ensure_batch_dim, ensure_1d, clamp_weights
from thalia.core.traces import update_trace
from thalia.core.diagnostics_mixin import DiagnosticsMixin

from .config import LayeredCortexConfig, LayeredCortexState


class LayeredCortex(DiagnosticsMixin, BrainRegion):
    """
    Multi-layer cortical microcircuit with proper layer separation.

    This implements a canonical cortical column with:
    - L4: Input layer (receives external input)
    - L2/3: Processing layer (recurrent, outputs to other cortex)
    - L5: Output layer (outputs to subcortical structures)

    The key insight is that OUTPUT to next region comes from a different
    layer than the one receiving RECURRENT feedback, solving the
    contamination problem in single-layer models.

    Usage:
        config = LayeredCortexConfig(n_input=256, n_output=64)
        cortex = LayeredCortex(config)

        # Process input
        output = cortex.forward(input_spikes)

        # Output contains both L2/3 (cortico-cortical) and L5 (subcortical)
        l23_out = output[:, :cortex.l23_size]
        l5_out = output[:, cortex.l23_size:]
    """

    def __init__(self, config: LayeredCortexConfig):
        """Initialize layered cortex."""
        self.layer_config = config

        # Compute layer sizes
        self.l4_size = int(config.n_output * config.l4_ratio)
        self.l23_size = int(config.n_output * config.l23_ratio)
        self.l5_size = int(config.n_output * config.l5_ratio)

        # Actual output size depends on dual_output setting
        if config.dual_output:
            actual_output = self.l23_size + self.l5_size
        elif config.output_layer == "L5":
            actual_output = self.l5_size
        else:
            actual_output = self.l23_size

        # Create modified config for parent
        parent_config = RegionConfig(
            n_input=config.n_input,
            n_output=actual_output,
            dt_ms=config.dt_ms,
            device=config.device,
        )

        # Store output size before parent init
        self._actual_output = actual_output

        # Call parent init
        super().__init__(parent_config)

        # Initialize layers
        self._init_layers()

        # Initialize inter-layer weights
        self._init_weights()

        # Initialize feedforward inhibition (FFI)
        if config.ffi_enabled:
            self.feedforward_inhibition = FeedforwardInhibition(
                threshold=config.ffi_threshold,
                max_inhibition=config.ffi_strength * 10.0,
                decay_rate=1.0 - (1.0 / config.ffi_tau),
            )
        else:
            self.feedforward_inhibition = None

        # BCM for each layer
        if config.bcm_enabled:
            device = torch.device(config.device)
            bcm_cfg = config.bcm_config or BCMConfig(
                tau_theta=config.bcm_tau_theta,
                theta_init=config.bcm_theta_init,
            )
            self.bcm_l4 = BCMRule(n_post=self.l4_size, config=bcm_cfg)
            self.bcm_l4.to(device)
            self.bcm_l23 = BCMRule(n_post=self.l23_size, config=bcm_cfg)
            self.bcm_l23.to(device)
            self.bcm_l5 = BCMRule(n_post=self.l5_size, config=bcm_cfg)
            self.bcm_l5.to(device)
        else:
            self.bcm_l4 = None
            self.bcm_l23 = None
            self.bcm_l5 = None

        # State
        self.state = LayeredCortexState()

    def _get_learning_rule(self) -> LearningRule:
        """Cortex uses unsupervised Hebbian learning."""
        return LearningRule.HEBBIAN

    def _initialize_weights(self) -> torch.Tensor:
        """Placeholder - real weights in _init_weights."""
        return nn.Parameter(
            torch.zeros(self._actual_output, self.layer_config.n_input)
        )

    def _create_neurons(self):
        """Placeholder - neurons created in _init_layers."""
        return None

    def _init_layers(self) -> None:
        """Initialize LIF neurons for each layer."""
        l4_config = LIFConfig(tau_mem=15.0, v_threshold=1.0)
        l23_config = LIFConfig(tau_mem=20.0, v_threshold=1.0)
        l5_config = LIFConfig(tau_mem=20.0, v_threshold=0.9)

        self.l4_neurons = LIFNeuron(self.l4_size, l4_config)
        self.l23_neurons = LIFNeuron(self.l23_size, l23_config)
        self.l5_neurons = LIFNeuron(self.l5_size, l5_config)

    def _init_weights(self) -> None:
        """Initialize inter-layer weight matrices.
        
        Feedforward weights use positive initialization to ensure sparse
        presynaptic activity can drive postsynaptic neurons above threshold.
        With ~10-15% sparsity, we need weights scaled so that:
            sum(w_ij * spike_j) * strength ~ threshold
        
        Using uniform [0, max] with max scaled by fan-in and expected sparsity.
        """
        device = torch.device(self.layer_config.device)
        cfg = self.layer_config
        
        # Expected number of active inputs given sparsity
        expected_active_l4 = max(1, int(self.l4_size * cfg.l4_sparsity))
        expected_active_l23 = max(1, int(self.l23_size * cfg.l23_sparsity))
        
        # Feedforward weights: positive, scaled so sparse input reaches threshold
        # With n_active inputs, threshold ~1.0, strength factor applied later:
        # target = threshold / (n_active * strength) ≈ 1.0 / (n_active * strength)
        # We initialize to mean ≈ target, with some variance for diversity
        
        # Input → L4: positive excitatory weights
        w_scale_input = 1.0 / max(1, int(cfg.n_input * 0.15))  # Assume 15% input sparsity
        self.w_input_l4 = nn.Parameter(
            torch.abs(torch.randn(self.l4_size, cfg.n_input, device=device)) * w_scale_input
        )
        
        # L4 → L2/3: positive excitatory weights
        w_scale_l4_l23 = 1.0 / expected_active_l4
        self.w_l4_l23 = nn.Parameter(
            torch.abs(torch.randn(self.l23_size, self.l4_size, device=device)) * w_scale_l4_l23
        )
        
        # L2/3 recurrent: can be positive or negative (E/I balance)
        self.w_l23_recurrent = nn.Parameter(
            torch.randn(self.l23_size, self.l23_size, device=device) * 0.2
        )
        with torch.no_grad():
            self.w_l23_recurrent.data.fill_diagonal_(0.0)
        
        # L2/3 → L5: positive excitatory weights
        w_scale_l23_l5 = 1.0 / expected_active_l23
        self.w_l23_l5 = nn.Parameter(
            torch.abs(torch.randn(self.l5_size, self.l23_size, device=device)) * w_scale_l23_l5
        )
        
        # L2/3 inhibition: positive (inhibitory connections suppress)
        self.w_l23_inhib = nn.Parameter(
            torch.ones(self.l23_size, self.l23_size, device=device) * 0.3
        )
        with torch.no_grad():
            self.w_l23_inhib.data.fill_diagonal_(0.0)

        self.weights = self.w_input_l4

    def reset(self) -> None:
        """Reset state for new episode."""
        super().reset()
        self.reset_state(1)

    def reset_state(self, batch_size: int = 1) -> None:
        """Reset all layer states."""
        dev = self.device

        self.l4_neurons.reset_state(batch_size)
        self.l23_neurons.reset_state(batch_size)
        self.l5_neurons.reset_state(batch_size)

        self.state = LayeredCortexState(
            l4_spikes=torch.zeros(batch_size, self.l4_size, device=dev),
            l23_spikes=torch.zeros(batch_size, self.l23_size, device=dev),
            l5_spikes=torch.zeros(batch_size, self.l5_size, device=dev),
            l23_recurrent_activity=torch.zeros(batch_size, self.l23_size, device=dev),
            l4_trace=torch.zeros(batch_size, self.l4_size, device=dev),
            l23_trace=torch.zeros(batch_size, self.l23_size, device=dev),
            l5_trace=torch.zeros(batch_size, self.l5_size, device=dev),
            top_down_modulation=None,
            ffi_strength=0.0,
        )

        # Note: FFI state decays naturally, no hard reset needed

    def forward(
        self,
        input_spikes: torch.Tensor,
        dt: float = 1.0,
        encoding_mod: float = 1.0,
        retrieval_mod: float = 1.0,
        top_down: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Process input through layered cortical circuit with continuous plasticity.
        
        This method both processes spikes AND applies synaptic plasticity. Learning
        happens continuously at each timestep, modulated by neuromodulators (dopamine).
        This is how biological cortex works - plasticity is part of the dynamics,
        not a separate training phase.
        """
        input_spikes = ensure_batch_dim(input_spikes)

        batch_size = input_spikes.shape[0]

        if self.state.l4_spikes is None:
            self.reset_state(batch_size)

        cfg = self.layer_config

        # L4: Input processing
        l4_input = (
            torch.matmul(input_spikes.float(), self.w_input_l4.t())
            * cfg.input_to_l4_strength
        )
        l4_input = l4_input * (0.5 + 0.5 * encoding_mod)
        l4_spikes, _ = self.l4_neurons(l4_input)
        l4_spikes = self._apply_sparsity(l4_spikes, cfg.l4_sparsity)
        self.state.l4_spikes = l4_spikes

        # L2/3: Processing with recurrence
        l23_ff = (
            torch.matmul(l4_spikes.float(), self.w_l4_l23.t())
            * cfg.l4_to_l23_strength
        )

        # Feedforward inhibition
        ffi_suppression = 1.0
        if self.feedforward_inhibition is not None:
            ffi = self.feedforward_inhibition.compute(input_spikes, return_tensor=False)
            raw_ffi = ffi.item() if hasattr(ffi, "item") else float(ffi)
            self.state.ffi_strength = min(
                1.0, raw_ffi / self.feedforward_inhibition.max_inhibition
            )
            ffi_suppression = 1.0 - self.state.ffi_strength * cfg.ffi_strength

        # Recurrent
        if self.state.l23_recurrent_activity is not None:
            recurrent_scale = 0.5 + 0.5 * retrieval_mod
            l23_rec = (
                torch.matmul(
                    self.state.l23_recurrent_activity,
                    self.w_l23_recurrent.t(),
                )
                * cfg.l23_recurrent_strength
                * recurrent_scale
                * ffi_suppression
            )
        else:
            l23_rec = torch.zeros_like(l23_ff)

        # Top-down modulation
        l23_td = top_down * cfg.l23_top_down_strength if top_down is not None else 0.0

        l23_input = l23_ff + l23_rec + l23_td

        # Lateral inhibition
        if self.state.l23_spikes is not None:
            l23_inhib = torch.matmul(
                self.state.l23_spikes.float(),
                self.w_l23_inhib.t(),
            )
            l23_input = l23_input - l23_inhib

        l23_spikes, _ = self.l23_neurons(F.relu(l23_input))
        l23_spikes = self._apply_sparsity(l23_spikes, cfg.l23_sparsity)
        self.state.l23_spikes = l23_spikes

        # Update recurrent activity trace
        if self.state.l23_recurrent_activity is not None:
            self.state.l23_recurrent_activity = (
                self.state.l23_recurrent_activity * cfg.l23_recurrent_decay
                + l23_spikes.float()
            )
        else:
            self.state.l23_recurrent_activity = l23_spikes.float()

        # L5: Subcortical output
        l5_input = (
            torch.matmul(l23_spikes.float(), self.w_l23_l5.t())
            * cfg.l23_to_l5_strength
        )
        l5_spikes, _ = self.l5_neurons(l5_input)
        l5_spikes = self._apply_sparsity(l5_spikes, cfg.l5_sparsity)
        self.state.l5_spikes = l5_spikes

        # Update STDP traces using utility function
        if self.state.l4_trace is not None:
            update_trace(self.state.l4_trace, l4_spikes, tau=cfg.stdp_tau_plus, dt=dt)
        if self.state.l23_trace is not None:
            update_trace(self.state.l23_trace, l23_spikes, tau=cfg.stdp_tau_plus, dt=dt)
        if self.state.l5_trace is not None:
            update_trace(self.state.l5_trace, l5_spikes, tau=cfg.stdp_tau_plus, dt=dt)

        self.state.spikes = l5_spikes
        
        # Store input for plasticity
        self.state.input_spikes = input_spikes
        
        # Apply continuous plasticity (learning happens as part of forward dynamics)
        self._apply_plasticity()

        # Construct output
        if cfg.dual_output:
            output = torch.cat([l23_spikes, l5_spikes], dim=-1)
        elif cfg.output_layer == "L5":
            output = l5_spikes
        else:
            output = l23_spikes

        return output

    def _apply_sparsity(
        self,
        spikes: torch.Tensor,
        target_sparsity: float,
    ) -> torch.Tensor:
        """Apply winner-take-all sparsity."""
        batch_size, n_neurons = spikes.shape
        k = max(1, int(n_neurons * target_sparsity))

        sparse_spikes = torch.zeros_like(spikes)

        for b in range(batch_size):
            active = spikes[b].nonzero(as_tuple=True)[0]
            if len(active) > k:
                keep_indices = active[torch.randperm(len(active))[:k]]
                sparse_spikes[b, keep_indices] = spikes[b, keep_indices]
            else:
                sparse_spikes[b] = spikes[b]

        return sparse_spikes

    def _apply_plasticity(self) -> None:
        """Apply continuous STDP learning with BCM modulation.
        
        This is called automatically at each forward() timestep.
        Learning rate is modulated by dopamine (via get_effective_learning_rate).
        
        In biological cortex, synaptic plasticity happens continuously based on
        pre/post spike timing. Dopamine doesn't trigger learning - it modulates
        how much weight change occurs from the spike-timing-based plasticity.
        """
        if self.state.l4_spikes is None or self.state.input_spikes is None:
            return

        cfg = self.layer_config
        
        # Get dopamine-modulated learning rate
        base_lr = cfg.stdp_lr
        effective_lr = self.get_effective_learning_rate(base_lr)
        
        # Early exit if learning rate is too small
        if effective_lr < 1e-8:
            self.state.last_plasticity_delta = 0.0
            return

        # Get 1D versions of spike tensors for torch.outer
        l4_spikes = ensure_1d(self.state.l4_spikes)
        l23_spikes = ensure_1d(self.state.l23_spikes) if self.state.l23_spikes is not None else None
        l5_spikes = ensure_1d(self.state.l5_spikes) if self.state.l5_spikes is not None else None
        input_spikes = ensure_1d(self.state.input_spikes)

        total_change = 0.0

        # BCM modulation factors
        l4_bcm_mod = 1.0
        l23_bcm_mod = 1.0
        l5_bcm_mod = 1.0

        if self.bcm_l4 is not None:
            l4_activity = l4_spikes.mean()
            l4_bcm_mod = self.bcm_l4.compute_phi(l4_activity)
            self.bcm_l4.update_threshold(l4_activity)

        if self.bcm_l23 is not None and l23_spikes is not None:
            l23_activity = l23_spikes.mean()
            l23_bcm_mod = self.bcm_l23.compute_phi(l23_activity)
            self.bcm_l23.update_threshold(l23_activity)

        if self.bcm_l5 is not None and l5_spikes is not None:
            l5_activity = l5_spikes.mean()
            l5_bcm_mod = self.bcm_l5.compute_phi(l5_activity)
            self.bcm_l5.update_threshold(l5_activity)

        def bcm_to_scale(mod: Any) -> float:
            if isinstance(mod, torch.Tensor):
                return float(1.0 + 0.5 * torch.tanh(mod).item())
            return 1.0 + 0.5 * torch.tanh(torch.tensor(mod)).item()

        # Input → L4
        dw = effective_lr * bcm_to_scale(l4_bcm_mod) * torch.outer(
            l4_spikes.float(),
            input_spikes.float(),
        )
        with torch.no_grad():
            self.w_input_l4.data += dw
            clamp_weights(self.w_input_l4.data, cfg.w_min, cfg.w_max)
        total_change += dw.abs().mean().item()

        # L4 → L2/3
        if l23_spikes is not None:
            dw = effective_lr * bcm_to_scale(l23_bcm_mod) * torch.outer(
                l23_spikes.float(),
                l4_spikes.float(),
            )
            with torch.no_grad():
                self.w_l4_l23.data += dw
                clamp_weights(self.w_l4_l23.data, cfg.w_min, cfg.w_max)
            total_change += dw.abs().mean().item()

            # L2/3 recurrent
            dw = effective_lr * 0.5 * bcm_to_scale(l23_bcm_mod) * torch.outer(
                l23_spikes.float(), l23_spikes.float()
            )
            with torch.no_grad():
                self.w_l23_recurrent.data += dw
                self.w_l23_recurrent.data.fill_diagonal_(0.0)
                clamp_weights(self.w_l23_recurrent.data, cfg.w_min, cfg.w_max)
            total_change += dw.abs().mean().item()

            # L2/3 → L5
            if l5_spikes is not None:
                dw = effective_lr * bcm_to_scale(l5_bcm_mod) * torch.outer(
                    l5_spikes.float(),
                    l23_spikes.float(),
                )
                with torch.no_grad():
                    self.w_l23_l5.data += dw
                    clamp_weights(self.w_l23_l5.data, cfg.w_min, cfg.w_max)
                total_change += dw.abs().mean().item()

        # Store for monitoring
        self.state.last_plasticity_delta = total_change

    def get_layer_outputs(self) -> Dict[str, Optional[torch.Tensor]]:
        """Get outputs from all layers."""
        return {
            "L4": self.state.l4_spikes,
            "L2/3": self.state.l23_spikes,
            "L5": self.state.l5_spikes,
        }

    def get_cortical_output(self) -> Optional[torch.Tensor]:
        """Get L2/3 output (for cortico-cortical pathways)."""
        return self.state.l23_spikes

    def get_subcortical_output(self) -> Optional[torch.Tensor]:
        """Get L5 output (for subcortical pathways)."""
        return self.state.l5_spikes

    def get_diagnostics(self) -> Dict[str, Any]:
        """Get layer-specific diagnostics using DiagnosticsMixin helpers."""
        diag: Dict[str, Any] = {
            "l4_size": self.l4_size,
            "l23_size": self.l23_size,
            "l5_size": self.l5_size,
        }
        
        # Spike diagnostics for each layer
        if self.state.l4_spikes is not None:
            diag.update(self.spike_diagnostics(self.state.l4_spikes, "l4"))
        if self.state.l23_spikes is not None:
            diag.update(self.spike_diagnostics(self.state.l23_spikes, "l23"))
        if self.state.l5_spikes is not None:
            diag.update(self.spike_diagnostics(self.state.l5_spikes, "l5"))
        
        # Recurrent activity
        if self.state.l23_recurrent_activity is not None:
            diag["l23_recurrent_mean"] = self.state.l23_recurrent_activity.mean().item()
        
        # Weight diagnostics for inter-layer connections
        diag.update(self.weight_diagnostics(self.w_input_l4.data, "input_l4"))
        diag.update(self.weight_diagnostics(self.w_l4_l23.data, "l4_l23"))
        diag.update(self.weight_diagnostics(self.w_l23_recurrent.data, "l23_rec"))
        diag.update(self.weight_diagnostics(self.w_l23_l5.data, "l23_l5"))
        
        return diag
