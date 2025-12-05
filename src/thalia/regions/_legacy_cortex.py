"""
Layered Cortex - Multi-layer cortical microcircuit with proper layer separation.

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

Key Design Principles:
======================
1. SEPARATE INPUT FROM OUTPUT: L4 receives input, L5 outputs to subcortical
2. RECURRENCE ONLY IN PROCESSING LAYER: L2/3 has lateral connections
3. DIFFERENT TARGETS PER LAYER: L2/3→cortex, L5→subcortex
4. PROPER TIMING: Input → L4 → L2/3 → L5 (sequential processing)

Benefits over single-layer:
===========================
- Input processing doesn't contaminate output
- Recurrent dynamics are contained in L2/3
- Different output pathways for different targets
- Enables predictive coding (predictions from L2/3, errors to L4)

Author: Thalia Project
Date: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from thalia.regions.base import (
    BrainRegion,
    RegionConfig,
    RegionState,
    LearningRule,
)
from thalia.core.neuron import LIFNeuron, LIFConfig
from thalia.regions.theta_dynamics import FeedforwardInhibition
from thalia.learning.bcm import BCMRule, BCMConfig


class CorticalLayer(Enum):
    """Cortical layer identifiers."""
    L4 = "L4"      # Input layer
    L23 = "L2/3"   # Processing/cortico-cortical output
    L5 = "L5"      # Subcortical output


@dataclass
class LayeredCortexConfig(RegionConfig):
    """Configuration for layered cortical microcircuit.
    
    Layer Sizes:
        By default, layers are sized relative to the output size:
        - L4: Same as output (input processing)
        - L2/3: 1.5x output (processing, more neurons for recurrence)
        - L5: Same as output (subcortical output)
    """
    
    # Layer size ratios (relative to n_output)
    l4_ratio: float = 1.0       # Input layer
    l23_ratio: float = 1.5      # Processing layer (larger for recurrence)
    l5_ratio: float = 1.0       # Output layer
    
    # Layer sparsity (fraction of neurons active)
    l4_sparsity: float = 0.15   # Moderate sparsity
    l23_sparsity: float = 0.10  # Sparser (more selective)
    l5_sparsity: float = 0.20   # Less sparse (motor commands)
    
    # Recurrence in L2/3
    l23_recurrent_strength: float = 0.3  # Lateral connection strength
    l23_recurrent_decay: float = 0.9     # Recurrent activity decay
    
    # Feedforward connection strengths
    input_to_l4_strength: float = 0.5    # External input → L4
    l4_to_l23_strength: float = 0.4      # L4 → L2/3
    l23_to_l5_strength: float = 0.4      # L2/3 → L5
    
    # Top-down modulation (for attention pathway)
    l23_top_down_strength: float = 0.2   # Feedback to L2/3
    
    # STDP learning parameters
    stdp_lr: float = 0.01
    stdp_tau_plus: float = 20.0
    stdp_tau_minus: float = 20.0
    
    # Weight bounds
    w_max: float = 1.0
    w_min: float = 0.0
    
    # Which layer to use as output to next region
    output_layer: str = "L5"  # "L2/3" for cortical, "L5" for subcortical
    
    # Whether to output both layers (for different pathways)
    dual_output: bool = True  # Output both L2/3 and L5
    
    # Feedforward Inhibition (FFI) parameters
    # FFI detects stimulus changes and transiently suppresses recurrent activity
    # This is how the cortex naturally "clears" old representations when new input arrives
    ffi_enabled: bool = True         # Enable FFI mechanism
    ffi_threshold: float = 0.3       # Input change threshold to trigger FFI
    ffi_strength: float = 0.8        # How much FFI suppresses L2/3 recurrent activity
    ffi_tau: float = 5.0             # FFI decay time constant (ms)
    
    # =========================================================================
    # BCM SLIDING THRESHOLD (Metaplasticity)
    # =========================================================================
    # The BCM rule provides a sliding threshold for synaptic modification that
    # automatically adjusts based on postsynaptic activity history. This is
    # particularly important for cortical learning because:
    # 1. Prevents runaway potentiation in highly active neurons
    # 2. Maintains selectivity during feature learning
    # 3. Enables competitive dynamics between feature detectors
    #
    # In visual cortex, BCM explains orientation selectivity development:
    # neurons that respond strongly to one orientation have high thresholds,
    # making them less likely to respond to other orientations.
    bcm_enabled: bool = False
    bcm_tau_theta: float = 5000.0    # Threshold adaptation time constant (ms)
    bcm_theta_init: float = 0.01     # Initial sliding threshold
    bcm_config: Optional[BCMConfig] = None  # Custom BCM configuration


@dataclass 
class LayeredCortexState(RegionState):
    """State for layered cortex."""
    
    # Per-layer spike states
    l4_spikes: Optional[torch.Tensor] = None
    l23_spikes: Optional[torch.Tensor] = None
    l5_spikes: Optional[torch.Tensor] = None
    
    # L2/3 recurrent activity (accumulated over time)
    l23_recurrent_activity: Optional[torch.Tensor] = None
    
    # STDP traces per layer
    l4_trace: Optional[torch.Tensor] = None
    l23_trace: Optional[torch.Tensor] = None
    l5_trace: Optional[torch.Tensor] = None
    
    # Top-down modulation state
    top_down_modulation: Optional[torch.Tensor] = None
    
    # Feedforward inhibition strength (0-1, 1 = max suppression)
    ffi_strength: float = 0.0


class LayeredCortex(BrainRegion):
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
        # FFI detects stimulus changes and transiently suppresses recurrent activity
        # This provides a biologically realistic way to "clear" old representations
        if config.ffi_enabled:
            self.feedforward_inhibition = FeedforwardInhibition(
                threshold=config.ffi_threshold,
                max_inhibition=config.ffi_strength * 10.0,  # Scale to appropriate range
                decay_rate=1.0 - (1.0 / config.ffi_tau),  # Convert tau to rate
            )
        else:
            self.feedforward_inhibition = None
        
        # =====================================================================
        # BCM SLIDING THRESHOLD (Metaplasticity for stable learning)
        # =====================================================================
        # BCM rule for each layer - prevents runaway potentiation and enables
        # competitive feature learning
        if config.bcm_enabled:
            device = torch.device(config.device)
            bcm_cfg = config.bcm_config or BCMConfig(
                tau_theta=config.bcm_tau_theta,
                theta_init=config.bcm_theta_init,
            )
            # BCM for L4 (input feature detectors)
            self.bcm_l4 = BCMRule(n_post=self.l4_size, config=bcm_cfg)
            self.bcm_l4.to(device)
            
            # BCM for L2/3 (higher-level representations)
            self.bcm_l23 = BCMRule(n_post=self.l23_size, config=bcm_cfg)
            self.bcm_l23.to(device)
            
            # BCM for L5 (output layer)
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
        return nn.Parameter(torch.zeros(self._actual_output, self.layer_config.n_input))
    
    def _create_neurons(self):
        """Placeholder - neurons created in _init_layers."""
        return None
    
    def _init_layers(self) -> None:
        """Initialize LIF neurons for each layer."""
        device = torch.device(self.layer_config.device)
        
        # LIF config - slightly different per layer
        l4_config = LIFConfig(tau_mem=15.0, v_threshold=1.0)  # Fast, feedforward
        l23_config = LIFConfig(tau_mem=20.0, v_threshold=1.0)  # Slower, recurrent
        l5_config = LIFConfig(tau_mem=20.0, v_threshold=0.9)  # Lower threshold, burst-capable
        
        self.l4_neurons = LIFNeuron(self.l4_size, l4_config)
        self.l23_neurons = LIFNeuron(self.l23_size, l23_config)
        self.l5_neurons = LIFNeuron(self.l5_size, l5_config)
    
    def _init_weights(self) -> None:
        """Initialize inter-layer weight matrices."""
        device = torch.device(self.layer_config.device)
        cfg = self.layer_config
        
        # Input → L4: Feedforward from external input
        self.w_input_l4 = nn.Parameter(
            torch.randn(self.l4_size, cfg.n_input, device=device) * 0.3
        )
        
        # L4 → L2/3: Feedforward within column
        self.w_l4_l23 = nn.Parameter(
            torch.randn(self.l23_size, self.l4_size, device=device) * 0.3
        )
        
        # L2/3 → L2/3: Recurrent lateral connections
        self.w_l23_recurrent = nn.Parameter(
            torch.randn(self.l23_size, self.l23_size, device=device) * 0.2
        )
        # No self-connections
        with torch.no_grad():
            self.w_l23_recurrent.data.fill_diagonal_(0.0)
        
        # L2/3 → L5: Feedforward to output layer
        self.w_l23_l5 = nn.Parameter(
            torch.randn(self.l5_size, self.l23_size, device=device) * 0.3
        )
        
        # L2/3 lateral inhibition
        self.w_l23_inhib = nn.Parameter(
            torch.ones(self.l23_size, self.l23_size, device=device) * 0.3
        )
        with torch.no_grad():
            self.w_l23_inhib.data.fill_diagonal_(0.0)
        
        # Store main weights reference for compatibility
        self.weights = self.w_input_l4
    
    def reset(self) -> None:
        """Reset state for new episode."""
        super().reset()
        self.reset_state(1)
    
    def reset_state(self, batch_size: int = 1) -> None:
        """Reset all layer states."""
        dev = self.device
        
        # Reset neuron states
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
        
        # Clear FFI history
        if self.feedforward_inhibition is not None:
            self.feedforward_inhibition.clear()
    
    def new_trial(self) -> None:
        """Prepare cortex for a new trial.
        
        This clears FFI input history so that stimulus onset will be
        detected properly and trigger transient suppression of recurrent
        activity. This is more biologically realistic than hard-resetting
        all recurrent state.
        
        Call this at the start of process_sample() - not before test phase!
        The FFI mechanism will naturally detect the stimulus change from
        delay period to test phase.
        """
        if self.feedforward_inhibition is not None:
            self.feedforward_inhibition.clear()
        self.state.ffi_strength = 0.0
    
    def forward(
        self,
        input_spikes: torch.Tensor,
        dt: float = 1.0,
        encoding_mod: float = 1.0,
        retrieval_mod: float = 1.0,
        top_down: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        Process input through layered cortical circuit.
        
        Signal flow:
            Input → L4 → L2/3 (+ recurrent) → L5 → Output
        
        Args:
            input_spikes: External input [batch, n_input]
            dt: Time step in ms
            encoding_mod: Theta modulation for encoding (boosts input gain)
            retrieval_mod: Theta modulation for retrieval (boosts recurrence)
            top_down: Optional top-down modulation for L2/3 [batch, l23_size]
            
        Returns:
            Output spikes [batch, l23_size + l5_size] if dual_output
            or [batch, output_layer_size] otherwise
        """
        if input_spikes.dim() == 1:
            input_spikes = input_spikes.unsqueeze(0)
        
        batch_size = input_spikes.shape[0]
        
        # Ensure state is initialized
        if self.state.l4_spikes is None:
            self.reset_state(batch_size)
        
        cfg = self.layer_config
        
        # =====================================================================
        # 1. LAYER 4: Input processing (no recurrence)
        # =====================================================================
        # L4 receives external input only - pure feedforward
        # Theta modulates input gain: encoding phase boosts external input
        l4_input = torch.matmul(input_spikes.float(), self.w_input_l4.t()) * cfg.input_to_l4_strength
        l4_input = l4_input * (0.5 + 0.5 * encoding_mod)  # Scale by encoding strength
        
        # Run through L4 neurons
        l4_spikes, _ = self.l4_neurons(l4_input)
        
        # Apply sparsity
        l4_spikes = self._apply_sparsity(l4_spikes, cfg.l4_sparsity)
        self.state.l4_spikes = l4_spikes
        
        # =====================================================================
        # 2. LAYER 2/3: Processing with recurrence
        # =====================================================================
        # Feedforward from L4
        l23_ff = torch.matmul(l4_spikes.float(), self.w_l4_l23.t()) * cfg.l4_to_l23_strength
        
        # =====================================================================
        # FEEDFORWARD INHIBITION (FFI)
        # =====================================================================
        # FFI detects stimulus changes and transiently suppresses recurrent
        # activity. This is how the cortex naturally "clears" old representations
        # when new input arrives, without needing explicit resets.
        #
        # Biological basis: Fast-spiking PV+ interneurons respond rapidly to
        # input changes and briefly inhibit pyramidal cells, allowing a "fresh
        # start" for processing the new stimulus.
        ffi_suppression = 1.0  # No suppression by default
        if self.feedforward_inhibition is not None:
            ffi = self.feedforward_inhibition.compute(input_spikes, return_tensor=False)
            raw_ffi = ffi.item() if hasattr(ffi, 'item') else float(ffi)
            # Store normalized FFI strength (0-1)
            self.state.ffi_strength = min(1.0, raw_ffi / self.feedforward_inhibition.max_inhibition)
            # Compute suppression factor (1.0 = no suppression, 0.0 = full suppression)
            ffi_suppression = 1.0 - self.state.ffi_strength * cfg.ffi_strength
        
        # Recurrent from previous L2/3 activity
        # Theta modulates recurrence: retrieval phase boosts internal processing
        # FFI suppresses recurrence when stimulus changes
        if self.state.l23_recurrent_activity is not None:
            recurrent_scale = 0.5 + 0.5 * retrieval_mod  # Scale by retrieval strength
            l23_rec = torch.matmul(
                self.state.l23_recurrent_activity,
                self.w_l23_recurrent.t()
            ) * cfg.l23_recurrent_strength * recurrent_scale * ffi_suppression
        else:
            l23_rec = torch.zeros_like(l23_ff)
        
        # Top-down modulation (if provided)
        if top_down is not None:
            l23_td = top_down * cfg.l23_top_down_strength
        else:
            l23_td = 0.0
        
        # Total L2/3 input
        l23_input = l23_ff + l23_rec + l23_td
        
        # Lateral inhibition from previous L2/3 spikes
        if self.state.l23_spikes is not None:
            l23_inhib = torch.matmul(
                self.state.l23_spikes.float(),
                self.w_l23_inhib.t()
            )
            l23_input = l23_input - l23_inhib
        
        # Run through L2/3 neurons
        l23_spikes, _ = self.l23_neurons(F.relu(l23_input))
        
        # Apply sparsity
        l23_spikes = self._apply_sparsity(l23_spikes, cfg.l23_sparsity)
        self.state.l23_spikes = l23_spikes
        
        # Update recurrent activity trace (decays over time)
        if self.state.l23_recurrent_activity is not None:
            self.state.l23_recurrent_activity = (
                self.state.l23_recurrent_activity * cfg.l23_recurrent_decay
                + l23_spikes.float()
            )
        else:
            self.state.l23_recurrent_activity = l23_spikes.float()
        
        # =====================================================================
        # 3. LAYER 5: Subcortical output (no recurrence)
        # =====================================================================
        # L5 receives from L2/3 only - clean output layer
        l5_input = torch.matmul(l23_spikes.float(), self.w_l23_l5.t()) * cfg.l23_to_l5_strength
        
        # Run through L5 neurons
        l5_spikes, _ = self.l5_neurons(l5_input)
        
        # Apply sparsity (less sparse for motor output)
        l5_spikes = self._apply_sparsity(l5_spikes, cfg.l5_sparsity)
        self.state.l5_spikes = l5_spikes
        
        # =====================================================================
        # Update STDP traces
        # =====================================================================
        decay = torch.exp(torch.tensor(-dt / cfg.stdp_tau_plus))
        if self.state.l4_trace is not None:
            self.state.l4_trace = self.state.l4_trace * decay + l4_spikes.float()
        if self.state.l23_trace is not None:
            self.state.l23_trace = self.state.l23_trace * decay + l23_spikes.float()
        if self.state.l5_trace is not None:
            self.state.l5_trace = self.state.l5_trace * decay + l5_spikes.float()
        
        # Store in base state for compatibility
        self.state.spikes = l5_spikes  # Primary output
        
        # =====================================================================
        # Construct output based on configuration
        # =====================================================================
        if cfg.dual_output:
            # Concatenate L2/3 (cortical) and L5 (subcortical) outputs
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
                # Keep only top k (by membrane potential if available, else random)
                keep_indices = active[torch.randperm(len(active))[:k]]
                sparse_spikes[b, keep_indices] = spikes[b, keep_indices]
            else:
                sparse_spikes[b] = spikes[b]
        
        return sparse_spikes
    
    def learn(
        self,
        input_spikes: torch.Tensor,
        output_spikes: torch.Tensor,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Apply STDP learning to inter-layer connections with optional BCM modulation.
        
        Learning occurs on:
        - Input → L4 weights (feature extraction)
        - L4 → L2/3 weights (representation learning)
        - L2/3 recurrent weights (pattern completion)
        - L2/3 → L5 weights (output mapping)
        
        If BCM is enabled, learning rates are modulated by a sliding threshold
        that prevents runaway potentiation and enables competitive learning.
        """
        if self.state.l4_spikes is None:
            return {"weight_change": 0.0}
        
        cfg = self.layer_config
        total_change = 0.0
        
        # Get spike tensors with type narrowing
        l4_spikes = self.state.l4_spikes
        l23_spikes = self.state.l23_spikes
        l5_spikes = self.state.l5_spikes
        l4_trace = self.state.l4_trace
        l23_trace = self.state.l23_trace
        l5_trace = self.state.l5_trace
        
        if l4_spikes is None or l23_spikes is None or l5_spikes is None:
            return {"weight_change": 0.0}
        
        # =====================================================================
        # COMPUTE BCM MODULATION FACTORS (if enabled)
        # =====================================================================
        # BCM modulates learning based on postsynaptic activity relative to
        # a sliding threshold: φ(c,θ) = c(c-θ)/θ
        # - Above threshold → positive (LTP dominant)
        # - Below threshold → negative (LTD dominant)
        l4_bcm_mod = 1.0
        l23_bcm_mod = 1.0
        l5_bcm_mod = 1.0
        
        if self.bcm_l4 is not None:
            l4_activity = l4_spikes.mean()
            l4_bcm_mod = self.bcm_l4.compute_phi(l4_activity)
            self.bcm_l4.update_threshold(l4_activity)
            
        if self.bcm_l23 is not None:
            l23_activity = l23_spikes.mean()
            l23_bcm_mod = self.bcm_l23.compute_phi(l23_activity)
            self.bcm_l23.update_threshold(l23_activity)
            
        if self.bcm_l5 is not None:
            l5_activity = l5_spikes.mean()
            l5_bcm_mod = self.bcm_l5.compute_phi(l5_activity)
            self.bcm_l5.update_threshold(l5_activity)
        
        # Convert BCM modulation to scaling factor (avoid negative learning rates)
        def bcm_to_scale(mod):
            # Map BCM φ to a positive scale: φ>0 → stronger LTP, φ<0 → LTD
            if isinstance(mod, torch.Tensor):
                return 1.0 + 0.5 * torch.tanh(mod)
            return 1.0 + 0.5 * torch.tanh(torch.tensor(mod)).item()
        
        # Input → L4: STDP based on input and L4 activity
        if l4_trace is not None:
            effective_lr = cfg.stdp_lr * bcm_to_scale(l4_bcm_mod)
            dw = effective_lr * torch.outer(
                l4_spikes.squeeze().float(),
                input_spikes.squeeze().float()
            )
            with torch.no_grad():
                self.w_input_l4.data += dw
                self.w_input_l4.data.clamp_(cfg.w_min, cfg.w_max)
            total_change += dw.abs().mean().item()
        
        # L4 → L2/3: STDP
        if l4_trace is not None and l23_trace is not None:
            effective_lr = cfg.stdp_lr * bcm_to_scale(l23_bcm_mod)
            dw = effective_lr * torch.outer(
                l23_spikes.squeeze().float(),
                l4_spikes.squeeze().float()
            )
            with torch.no_grad():
                self.w_l4_l23.data += dw
                self.w_l4_l23.data.clamp_(cfg.w_min, cfg.w_max)
            total_change += dw.abs().mean().item()
        
        # L2/3 recurrent: Hebbian with BCM
        if l23_trace is not None:
            l23_activity = l23_spikes.squeeze().float()
            effective_lr = cfg.stdp_lr * 0.5 * bcm_to_scale(l23_bcm_mod)
            dw = effective_lr * torch.outer(l23_activity, l23_activity)
            with torch.no_grad():
                self.w_l23_recurrent.data += dw
                self.w_l23_recurrent.data.fill_diagonal_(0.0)
                self.w_l23_recurrent.data.clamp_(cfg.w_min, cfg.w_max)
            total_change += dw.abs().mean().item()
        
        # L2/3 → L5: STDP
        if l23_trace is not None and l5_trace is not None:
            effective_lr = cfg.stdp_lr * bcm_to_scale(l5_bcm_mod)
            dw = effective_lr * torch.outer(
                l5_spikes.squeeze().float(),
                l23_spikes.squeeze().float()
            )
            with torch.no_grad():
                self.w_l23_l5.data += dw
                self.w_l23_l5.data.clamp_(cfg.w_min, cfg.w_max)
            total_change += dw.abs().mean().item()
        
        return {"weight_change": total_change}
    
    def get_layer_outputs(self) -> Dict[str, Optional[torch.Tensor]]:
        """Get outputs from all layers (for debugging/analysis)."""
        return {
            "L4": self.state.l4_spikes,
            "L2/3": self.state.l23_spikes,
            "L5": self.state.l5_spikes,
        }
    
    def get_cortical_output(self) -> Optional[torch.Tensor]:
        """Get L2/3 output (for cortico-cortical pathways)."""
        return self.state.l23_spikes
    
    def get_subcortical_output(self) -> Optional[torch.Tensor]:
        """Get L5 output (for subcortical pathways like striatum)."""
        return self.state.l5_spikes
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get layer-specific diagnostics."""
        return {
            "l4_size": self.l4_size,
            "l23_size": self.l23_size,
            "l5_size": self.l5_size,
            "l4_firing_rate": self.state.l4_spikes.mean().item() if self.state.l4_spikes is not None else 0,
            "l23_firing_rate": self.state.l23_spikes.mean().item() if self.state.l23_spikes is not None else 0,
            "l5_firing_rate": self.state.l5_spikes.mean().item() if self.state.l5_spikes is not None else 0,
            "l23_recurrent_activity": (
                self.state.l23_recurrent_activity.mean().item() 
                if self.state.l23_recurrent_activity is not None else 0
            ),
            "w_input_l4_mean": self.w_input_l4.data.mean().item(),
            "w_l4_l23_mean": self.w_l4_l23.data.mean().item(),
            "w_l23_recurrent_mean": self.w_l23_recurrent.data.mean().item(),
            "w_l23_l5_mean": self.w_l23_l5.data.mean().item(),
        }
